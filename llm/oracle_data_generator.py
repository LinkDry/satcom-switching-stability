"""LLM-as-Oracle Data Generator.

Generates (KPI scenario → reward weights) training pairs using the LLM,
covering both known regimes and novel regimes. The LLM runs offline in batch
mode — no real-time latency constraint.

Output: JSON file with [{kpi: {...}, weights: {...}, regime: str, quality_score: float}, ...]
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

from llm.mlp_architect import KPI_KEYS, WEIGHT_KEYS, SCALES, KPI_PROFILES, EXPERT_WEIGHTS
from llm.rag_knowledge import retrieve_context
from llm.prompts import SYSTEM_PROMPT

# Novel regime KPI profiles (not in MLP training data)
NOVEL_KPI_PROFILES = {
    "iot_burst": {
        "avg_demand": (15, 8), "demand_variance": (3000, 1200),
        "spatial_gini": (0.55, 0.15), "peak_beam_demand": (120, 40),
        "active_beam_fraction": (0.35, 0.15),
    },
    "polar_handover": {
        "avg_demand": (40, 12), "demand_variance": (1200, 400),
        "spatial_gini": (0.35, 0.1), "peak_beam_demand": (80, 25),
        "active_beam_fraction": (0.75, 0.1),
    },
    "hot_cold": {
        "avg_demand": (35, 15), "demand_variance": (3500, 1000),
        "spatial_gini": (0.60, 0.12), "peak_beam_demand": (130, 35),
        "active_beam_fraction": (0.50, 0.15),
    },
}

# Heuristic expert weights for novel regimes (initial oracle guidance)
NOVEL_EXPERT_HINTS = {
    "iot_burst": {"sum_rate": 0.5, "fairness": 0.6, "outage": 1.5, "switching": 0.1, "queue": 0.8},
    "polar_handover": {"sum_rate": 0.8, "fairness": 0.3, "outage": 1.0, "switching": 0.8, "queue": 0.2},
    "hot_cold": {"sum_rate": 1.0, "fairness": 0.5, "outage": 1.2, "switching": 0.05, "queue": 0.3},
}

ORACLE_PROMPT_TEMPLATE = """You are the MDP Architect for a 19-beam LEO satellite beam scheduling system.

Current traffic regime: {regime_description}

Network KPIs:
{kpi_text}

{rag_context}

Based on these KPIs, output the optimal reward weights as JSON.
The reward formula is: R = sum_rate*throughput - outage*outage_count - switching*switch_count - queue*avg_queue + fairness*fairness_index

Rules:
- ALL weights must be POSITIVE (the environment handles signs internally)
- sum_rate: typically 0.3-2.0 (higher = prioritize throughput)
- fairness: 0.0-1.0 (higher = prioritize equal beam service)
- outage: 0.5-3.0 (higher = penalize outages more)
- switching: 0.01-1.0 (higher = penalize beam switching)
- queue: 0.0-1.0 (higher = penalize queue buildup)

{hint_text}

Output ONLY valid JSON with keys: sum_rate, fairness, outage, switching, queue."""


REGIME_DESCRIPTIONS = {
    "urban": "Urban dense traffic — high aggregate demand concentrated in center beams",
    "maritime": "Maritime sparse traffic — low uniform demand across all beams",
    "disaster": "Disaster spike — sudden extreme demand in 2-3 beams, others baseline",
    "mixed": "Mixed traffic — moderate demand with some spatial variation",
    "iot_burst": "IoT burst — very low baseline with sudden synchronized micro-bursts in random beams, high variance",
    "polar_handover": "Polar handover — demand wave migrating across beams during satellite handover, stability critical",
    "hot_cold": "Hot-cold split — extreme spatial divide, half beams saturated, half nearly idle",
}


def _sample_kpi(regime: str, rng: np.random.Generator) -> dict:
    """Sample a KPI vector from a regime's profile."""
    if regime in KPI_PROFILES:
        profile = KPI_PROFILES[regime]
    elif regime in NOVEL_KPI_PROFILES:
        profile = NOVEL_KPI_PROFILES[regime]
    else:
        raise ValueError(f"Unknown regime: {regime}")

    kpi = {}
    for key in KPI_KEYS:
        mean, std = profile[key]
        kpi[key] = round(float(max(0, rng.normal(mean, std))), 2)
    return kpi


def _build_oracle_prompt(kpi: dict, regime: str, include_hint: bool = True) -> str:
    """Build the LLM prompt for a single oracle query."""
    kpi_text = "\n".join(f"- {k}: {v}" for k, v in kpi.items())
    desc = REGIME_DESCRIPTIONS.get(regime, f"Unknown regime: {regime}")

    # RAG context
    rag_ctx = retrieve_context(desc, top_k=2)
    rag_section = f"Domain context:\n{rag_ctx}" if rag_ctx else ""

    # Hint for novel regimes
    hint_text = ""
    if include_hint and regime in NOVEL_EXPERT_HINTS:
        hint = NOVEL_EXPERT_HINTS[regime]
        hint_text = f"Suggested weight range for this regime type: {json.dumps(hint)}\nYou may adjust these based on the specific KPIs."

    return ORACLE_PROMPT_TEMPLATE.format(
        regime_description=desc,
        kpi_text=kpi_text,
        rag_context=rag_section,
        hint_text=hint_text,
    )


def _parse_weights(response: str) -> Optional[dict]:
    """Parse weight JSON from LLM response."""
    import re
    # Try JSON code block
    m = re.search(r'```(?:json)?\s*(\{[^}]+\})\s*```', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    m = re.search(r'\{[^}]+\}', response)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def generate_oracle_data_llm(
    n_per_regime: int = 200,
    regimes: Optional[list] = None,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: int = 42,
    output_path: Optional[str] = None,
    verbose: int = 1,
) -> list[dict]:
    """Generate oracle data using actual LLM API calls.

    For each regime, samples n_per_regime KPI vectors and queries the LLM
    for optimal weights. Slow but produces genuine LLM reasoning.
    """
    from llm.architect import LLMMDPArchitect

    rng = np.random.default_rng(seed)
    if regimes is None:
        regimes = list(REGIME_DESCRIPTIONS.keys())

    # Use the existing LLM architect's API client
    architect = LLMMDPArchitect(
        model=model, temperature=0.15, api_key=api_key, base_url=base_url
    )

    samples = []
    for regime in regimes:
        if verbose:
            print(f"  Generating {n_per_regime} samples for regime: {regime}")
        for i in range(n_per_regime):
            kpi = _sample_kpi(regime, rng)
            prompt = _build_oracle_prompt(kpi, regime)
            response = architect._call_llm(prompt)

            if response is None:
                continue

            weights = _parse_weights(response)
            if weights is None:
                continue

            # Validate all keys present and positive
            if not all(k in weights for k in WEIGHT_KEYS):
                continue
            if any(weights[k] < 0 for k in WEIGHT_KEYS):
                continue

            samples.append({
                "kpi": kpi,
                "weights": {k: round(weights[k], 4) for k in WEIGHT_KEYS},
                "regime": regime,
                "source": "llm_oracle",
            })

            if verbose and (i + 1) % 50 == 0:
                print(f"    {regime}: {i+1}/{n_per_regime} done")

    if verbose:
        print(f"  Total oracle samples: {len(samples)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

    return samples


def generate_oracle_data_synthetic(
    n_per_regime: int = 500,
    regimes: Optional[list] = None,
    seed: int = 42,
    noise_std: float = 0.05,
    output_path: Optional[str] = None,
    verbose: int = 1,
) -> list[dict]:
    """Generate oracle data synthetically using expert hints + perturbation.

    Fast fallback when LLM API is unavailable. Uses heuristic expert weights
    for novel regimes and known expert weights for known regimes, with
    Gaussian noise for diversity.
    """
    rng = np.random.default_rng(seed)
    if regimes is None:
        regimes = list(REGIME_DESCRIPTIONS.keys())

    # Merge known + novel expert weights
    all_experts = {}
    for regime, w_list in EXPERT_WEIGHTS.items():
        all_experts[regime] = dict(zip(WEIGHT_KEYS, w_list))
    all_experts.update(NOVEL_EXPERT_HINTS)

    samples = []
    for regime in regimes:
        if regime not in all_experts:
            if verbose:
                print(f"  Skipping {regime}: no expert weights available")
            continue

        expert_w = all_experts[regime]
        if verbose:
            print(f"  Generating {n_per_regime} synthetic samples for: {regime}")

        for _ in range(n_per_regime):
            kpi = _sample_kpi(regime, rng)
            # Add noise to expert weights
            weights = {}
            for k in WEIGHT_KEYS:
                base = expert_w[k]
                noisy = base + rng.normal(0, noise_std * max(base, 0.1))
                weights[k] = round(max(0.001, noisy), 4)

            samples.append({
                "kpi": kpi,
                "weights": weights,
                "regime": regime,
                "source": "synthetic_oracle",
            })

    # Add transition samples between regimes
    n_transitions = len(regimes) * n_per_regime // 5
    regime_pairs = [(r1, r2) for r1 in regimes for r2 in regimes if r1 != r2]
    for _ in range(n_transitions):
        r1, r2 = regime_pairs[rng.integers(len(regime_pairs))]
        if r1 not in all_experts or r2 not in all_experts:
            continue
        alpha = rng.uniform(0.2, 0.8)

        # Interpolate KPIs
        kpi1 = _sample_kpi(r1, rng)
        kpi2 = _sample_kpi(r2, rng)
        kpi = {k: round((1 - alpha) * kpi1[k] + alpha * kpi2[k], 2) for k in KPI_KEYS}

        # Interpolate weights
        w1, w2 = all_experts[r1], all_experts[r2]
        weights = {k: round(max(0.001, (1 - alpha) * w1[k] + alpha * w2[k]), 4) for k in WEIGHT_KEYS}

        samples.append({
            "kpi": kpi,
            "weights": weights,
            "regime": f"transition_{r1}_{r2}",
            "source": "synthetic_oracle",
        })

    rng.shuffle(samples)

    if verbose:
        print(f"  Total synthetic oracle samples: {len(samples)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

    return samples


def generate_oracle_data_evolved(
    evolved_weights: list[dict],
    n_per_regime: int = 200,
    noise_std: float = 0.03,
    previous_best: list[dict] = None,
    seed: int = 42,
    output_path: Optional[str] = None,
    verbose: int = 1,
) -> list[dict]:
    """Generate training data from evolved weight centers.

    Args:
        evolved_weights: list of {regime, weights, reasoning} from evolution round
        n_per_regime: samples per regime around evolved center
        noise_std: tighter noise than original (0.03 vs 0.05)
        previous_best: best samples from prior rounds for experience replay
        seed: random seed
        output_path: optional save path

    Returns:
        list of {kpi, weights, regime, source, quality_score} dicts
    """
    rng = np.random.default_rng(seed)
    samples = []

    for entry in evolved_weights:
        regime = entry["regime"]
        center_w = entry["weights"]

        if verbose:
            print(f"  Generating {n_per_regime} evolved samples for: {regime}")

        for _ in range(n_per_regime):
            kpi = _sample_kpi(regime, rng)
            weights = {}
            for k in WEIGHT_KEYS:
                base = center_w[k]
                noisy = base + rng.normal(0, noise_std * max(base, 0.05))
                weights[k] = round(max(0.001, noisy), 4)
            samples.append({
                "kpi": kpi,
                "weights": weights,
                "regime": regime,
                "source": f"evolved_{entry.get('reasoning', '')[:30]}",
                "quality_score": 1.2,
            })

    # Mix in previous best samples (experience replay across rounds)
    if previous_best:
        samples.extend(previous_best)
        if verbose:
            print(f"  Added {len(previous_best)} experience replay samples")

    # Transition samples
    regime_names = [e["regime"] for e in evolved_weights]
    all_w = {e["regime"]: e["weights"] for e in evolved_weights}
    n_transitions = len(regime_names) * n_per_regime // 5
    for _ in range(n_transitions):
        if len(regime_names) < 2:
            break
        r1, r2 = rng.choice(regime_names, 2, replace=False)
        alpha = rng.uniform(0.2, 0.8)
        kpi1 = _sample_kpi(r1, rng)
        kpi2 = _sample_kpi(r2, rng)
        kpi = {k: round((1 - alpha) * kpi1[k] + alpha * kpi2[k], 2) for k in KPI_KEYS}
        w1, w2 = all_w[r1], all_w[r2]
        weights = {k: round(max(0.001, (1 - alpha) * w1[k] + alpha * w2[k]), 4) for k in WEIGHT_KEYS}
        samples.append({
            "kpi": kpi, "weights": weights,
            "regime": f"transition_{r1}_{r2}",
            "source": "evolved_transition",
            "quality_score": 1.0,
        })

    rng.shuffle(samples)
    if verbose:
        print(f"  Total evolved samples: {len(samples)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)

    return samples
