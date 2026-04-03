"""LLM Causal Reasoning Prompt + Experience Accumulation.

Builds structured prompts that show the LLM a causal sensitivity table
(from perturbation probes) and accumulated experience, enabling it to
reason about cause-effect relationships and generate improved weights.
"""

import json
import time
import re
from typing import Optional

import numpy as np

from llm.mlp_architect import WEIGHT_KEYS
from llm.quality_filter import WEIGHT_BOUNDS
from llm.perturbation_probe import format_sensitivity_table


# --- Causal Prompt Template ---

CAUSAL_PROMPT = """You are optimizing reward weights for regime "{regime}" in a 19-beam LEO satellite scheduler.
Round {round_id}. Use the CAUSAL TABLE below to decide which weights to adjust.

Reward: R = sum_rate*throughput/100 - outage*outage_count - switching*switch_count - queue*avg_queue/100 + fairness*fairness_index

{sensitivity_table}

Target: rate > {target_rate:.0f} Mbps (baseline MLP achieves this).

Bounds: sum_rate[{lb_sr},{ub_sr}] fairness[{lb_f},{ub_f}] outage[{lb_o},{ub_o}] switching[{lb_sw},{ub_sw}] queue[{lb_q},{ub_q}]

RULES:
- Change at most ±30% per weight from current values
- Prioritize probed directions that INCREASE rate (positive Δ Rate)
- Avoid directions that increase outage above 0.5
- Keep sum_rate in [0.8, 1.2] for stability
{experience_section}

Output ONLY JSON (no explanation outside):
{{"sum_rate":...,"fairness":...,"outage":...,"switching":...,"queue":...,"reasoning":"brief"}}"""


def build_causal_prompt(
    regime: str,
    probe_result: dict,
    round_id: int = 1,
    target_rate: float = 342.1,
    experience_context: str = "",
) -> str:
    """Build a causal reasoning prompt from probe results.

    Args:
        regime: regime name
        probe_result: output of probe_single_regime()
        round_id: current evolution round
        target_rate: baseline MLP rate target
        experience_context: accumulated experience from prior rounds

    Returns:
        Formatted prompt string
    """
    sensitivity_table = format_sensitivity_table(probe_result)

    experience_section = ""
    if experience_context.strip():
        experience_section = f"\nEXPERIENCE FROM PREVIOUS ROUNDS:\n{experience_context}\n"

    bounds = {k: WEIGHT_BOUNDS[k] for k in WEIGHT_KEYS}

    return CAUSAL_PROMPT.format(
        regime=regime,
        round_id=round_id,
        sensitivity_table=sensitivity_table,
        target_rate=target_rate,
        experience_section=experience_section,
        lb_sr=bounds["sum_rate"][0], ub_sr=bounds["sum_rate"][1],
        lb_f=bounds["fairness"][0], ub_f=bounds["fairness"][1],
        lb_o=bounds["outage"][0], ub_o=bounds["outage"][1],
        lb_sw=bounds["switching"][0], ub_sw=bounds["switching"][1],
        lb_q=bounds["queue"][0], ub_q=bounds["queue"][1],
    )


def llm_causal_evolve(
    prompt: str,
    regime: str,
    base_weights: dict,
    model: str = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
    api_key: str = None,
    base_url: str = "http://localhost:1234/v1",
    max_change: float = 0.3,
    verbose: int = 1,
    max_retries: int = 3,
) -> Optional[dict]:
    """Call LLM with causal prompt and return evolved weights.

    Args:
        prompt: the causal reasoning prompt
        regime: regime name (for logging)
        base_weights: current weights (for gradual clamping)
        model: LLM model name
        api_key: API key (None for local)
        base_url: API base URL
        max_change: maximum ±change fraction from base (0.3 = 30%)
        verbose: print progress
        max_retries: number of retry attempts

    Returns:
        {"sum_rate": ..., "fairness": ..., ...} or None on failure
    """
    import requests as req

    if verbose:
        print(f"  [LLM] Sending causal prompt for {regime} ({len(prompt)} chars)...")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "lm-studio":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert satellite system optimizer. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.15,
        "max_tokens": 4000,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            r = req.post(url, json=payload, headers=headers, timeout=300)
            latency = time.time() - t0

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

            data = r.json()
            content = data["choices"][0]["message"]["content"]

            if verbose:
                print(f"  [LLM] Response for {regime} in {latency:.1f}s: {content[:200]}...")

            weights = _parse_causal_response(content)
            if weights:
                # Apply gradual clamp: ±max_change from base
                clamped = _gradual_clamp(weights, base_weights, max_change)
                if verbose:
                    reasoning = weights.get("reasoning", "")
                    print(f"  [LLM] {regime}: {clamped} — {reasoning[:80]}")
                return clamped

            print(f"  [LLM] Attempt {attempt+1}: parse failed for {regime}, retrying...")
            last_error = "parse_failure"

        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"  [LLM] Attempt {attempt+1}/{max_retries} for {regime} failed: {e}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                if verbose:
                    print(f"  [LLM] Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  [LLM] All {max_retries} attempts failed for {regime} (last: {last_error})")
    return None


def _parse_causal_response(content: str) -> Optional[dict]:
    """Parse LLM response into weights dict."""
    # Try code block
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
    if m:
        text = m.group(1)
    else:
        # Try raw JSON object
        m = re.search(r'\{[^{}]*\}', content)
        text = m.group(0) if m else content

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and all(k in parsed for k in WEIGHT_KEYS):
            result = {}
            for k in WEIGHT_KEYS:
                val = float(parsed[k])
                lo, hi = WEIGHT_BOUNDS[k]
                result[k] = round(max(lo, min(hi, val)), 4)
            result["reasoning"] = parsed.get("reasoning", "")
            return result
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        pass

    return None


def _gradual_clamp(weights: dict, base_weights: dict, max_change: float = 0.3) -> dict:
    """Enforce ±max_change gradual constraint relative to base weights.

    For each weight, clamp to [base * (1 - max_change), base * (1 + max_change)].
    For very small base values, use an absolute floor of 0.005.
    """
    clamped = {}
    for k in WEIGHT_KEYS:
        base_val = base_weights.get(k, weights.get(k, 0.5))
        floor = max(base_val, 0.005)
        lo = floor * (1.0 - max_change)
        hi = floor * (1.0 + max_change)
        # Also respect global bounds
        glo, ghi = WEIGHT_BOUNDS[k]
        lo = max(lo, glo)
        hi = min(hi, ghi)
        clamped[k] = round(max(lo, min(hi, weights.get(k, base_val))), 4)
    return clamped


def synthetic_causal_evolve(
    probe_result: dict,
    base_weights: dict,
    max_change: float = 0.3,
    verbose: int = 1,
) -> dict:
    """Heuristic fallback when LLM is unavailable.

    Strategy: follow the gradient — increase weights where positive delta_rate
    was observed, decrease where negative delta_rate was observed.
    """
    regime = probe_result["regime"]
    weights = dict(base_weights)

    # Collect gradient information from probes
    gradients = {}  # weight_key -> estimated gradient (positive = increase is good)
    for p in probe_result["probes"]:
        wk = p["weight"]
        delta_rate = p["delta_rate"]
        direction_sign = 1 if "+" in p["direction"] else -1

        # If increasing the weight increased rate, gradient is positive
        # If decreasing the weight increased rate, gradient is negative
        grad = delta_rate * direction_sign
        if wk not in gradients:
            gradients[wk] = []
        gradients[wk].append(grad)

    # Average gradients and apply
    step_size = 0.15  # conservative step
    for wk in WEIGHT_KEYS:
        if wk in gradients and gradients[wk]:
            avg_grad = sum(gradients[wk]) / len(gradients[wk])
            if abs(avg_grad) > 1.0:  # Only move if signal is strong enough
                # Move in direction of positive gradient
                change = step_size if avg_grad > 0 else -step_size
                weights[wk] = weights[wk] * (1.0 + change)

    # Clamp
    weights = _gradual_clamp(weights, base_weights, max_change)

    if verbose:
        print(f"  [Synthetic] {regime}: {weights}")

    return weights


# --- Experience Accumulation ---

def update_experience_context(
    context: str,
    round_id: int,
    regime: str,
    probe_result: dict,
    applied_weights: dict,
    final_performance: dict,
    max_rounds_to_keep: int = 5,
) -> str:
    """Append this round's observations to the experience context.

    Args:
        context: existing experience context string
        round_id: current round
        regime: regime name
        probe_result: probe results (sensitivity table)
        applied_weights: the weights actually applied
        final_performance: performance after applying weights
        max_rounds_to_keep: max rounds of history to retain

    Returns:
        Updated experience context string
    """
    bp = probe_result["base_performance"]
    bw = probe_result["base_weights"]

    # Summarize key probe findings
    positive_probes = [p for p in probe_result["probes"] if p["delta_rate"] > 0]
    negative_probes = [p for p in probe_result["probes"] if p["delta_rate"] < 0]

    pos_summary = ", ".join(
        f"{p['weight']} {p['direction']}→rate{p['delta_rate']:+.0f}"
        for p in sorted(positive_probes, key=lambda x: -x["delta_rate"])[:3]
    ) if positive_probes else "none"

    neg_summary = ", ".join(
        f"{p['weight']} {p['direction']}→rate{p['delta_rate']:+.0f}"
        for p in sorted(negative_probes, key=lambda x: x["delta_rate"])[:3]
    ) if negative_probes else "none"

    # Compute improvement
    old_rate = bp["rate_mbps"]
    new_rate = final_performance.get("rate_mbps", 0)
    improvement = new_rate - old_rate
    pct = improvement / max(old_rate, 1) * 100

    # Format entry
    entry = (
        f"  Round {round_id} ({regime}):\n"
        f"    Helpful: {pos_summary}\n"
        f"    Harmful: {neg_summary}\n"
        f"    Applied: sr={applied_weights.get('sum_rate',0):.3f} f={applied_weights.get('fairness',0):.3f} "
        f"o={applied_weights.get('outage',0):.3f} sw={applied_weights.get('switching',0):.3f} "
        f"q={applied_weights.get('queue',0):.3f}\n"
        f"    Result: rate {old_rate:.1f}→{new_rate:.1f} ({improvement:+.1f}, {pct:+.1f}%)\n"
        f"    {'POSITIVE' if improvement > 0 else 'NEGATIVE'}\n"
    )

    # Parse existing entries and add new one
    if context.strip():
        lines = context.strip() + "\n" + entry
    else:
        lines = entry

    # Trim to max_rounds_to_keep
    entries = lines.strip().split("  Round ")
    entries = [e for e in entries if e.strip()]
    if len(entries) > max_rounds_to_keep:
        # Keep first entry summary + last (max_rounds_to_keep - 1) entries
        first = entries[0]
        recent = entries[-(max_rounds_to_keep - 1):]
        entries = [first] + recent

    result = "  Round ".join([""] + entries).strip()
    return result


RAG_CAUSAL_PROMPT = """You are optimizing reward weights for regime "{regime}" in a 19-beam LEO satellite scheduler.
Round {round_id}. Use the ANCHORS and CAUSAL TABLE below to decide which weights to propose.

Reward: R = sum_rate*throughput/100 - outage*outage_count - switching*switch_count - queue*avg_queue/100 + fairness*fairness_index

{anchor_text}

{sensitivity_table}

Target: rate > {target_rate:.0f} Mbps (baseline MLP achieves this).

Bounds: sum_rate[{lb_sr},{ub_sr}] fairness[{lb_f},{ub_f}] outage[{lb_o},{ub_o}] switching[{lb_sw},{ub_sw}] queue[{lb_q},{ub_q}]

RULES:
- Use the best anchor as your starting point (it is a verified high-performance example)
- Stay within ±30% of the best anchor's weights — do NOT generate from scratch
- Prioritize the causal table directions that INCREASE rate (positive Δ Rate)
- Avoid directions that increase outage above 0.5
- Keep sum_rate in [0.8, 1.2] for stability
{experience_section}

Output ONLY JSON (no explanation outside):
{{"sum_rate":...,"fairness":...,"outage":...,"switching":...,"queue":...,"reasoning":"brief"}}"""


def build_rag_causal_prompt(
    regime: str,
    probe_result: dict,
    anchor_db,
    query_kpi: np.ndarray,
    round_id: int = 1,
    target_rate: float = 342.1,
    experience_context: str = "",
    top_k: int = 5,
    verbose: int = 0,
) -> str:
    """Build a RAG-augmented causal prompt.

    Retrieves top-K similar high-performing weight anchors from anchor_db
    and injects them before the causal sensitivity table, grounding the LLM
    in verified historical examples rather than free-form generation.

    Args:
        regime: regime name
        probe_result: output of probe_single_regime()
        anchor_db: AnchorDB instance (from rag_anchor_db.py)
        query_kpi: (5,) KPI vector for current state
        round_id: current evolution round
        target_rate: baseline MLP rate target
        experience_context: accumulated experience from prior rounds
        top_k: number of anchors to retrieve
        verbose: pass to anchor_db.retrieve()

    Returns:
        Formatted RAG-augmented prompt string
    """
    from llm.rag_anchor_db import format_anchors_for_prompt

    anchors = anchor_db.retrieve(
        query_kpi=np.asarray(query_kpi, dtype=float),
        regime=regime,
        top_k=top_k,
        verbose=verbose,
    )
    anchor_text = format_anchors_for_prompt(anchors)
    sensitivity_table = format_sensitivity_table(probe_result)

    experience_section = ""
    if experience_context.strip():
        experience_section = f"\nEXPERIENCE FROM PREVIOUS ROUNDS:\n{experience_context}\n"

    bounds = {k: WEIGHT_BOUNDS[k] for k in WEIGHT_KEYS}

    return RAG_CAUSAL_PROMPT.format(
        regime=regime,
        round_id=round_id,
        anchor_text=anchor_text,
        sensitivity_table=sensitivity_table,
        target_rate=target_rate,
        experience_section=experience_section,
        lb_sr=bounds["sum_rate"][0], ub_sr=bounds["sum_rate"][1],
        lb_f=bounds["fairness"][0], ub_f=bounds["fairness"][1],
        lb_o=bounds["outage"][0], ub_o=bounds["outage"][1],
        lb_sw=bounds["switching"][0], ub_sw=bounds["switching"][1],
        lb_q=bounds["queue"][0], ub_q=bounds["queue"][1],
    )


def get_anchor_base_weights(
    anchor_db,
    query_kpi: np.ndarray,
    regime: str,
) -> dict:
    """Return the best anchor's weights as the gradual-clamp base.

    When using RAG, the ±30% clamp should be applied relative to the best
    anchor, not the current MLP weights. This ensures the LLM output stays
    close to verified high-performing examples.

    Returns None if no anchors available (fall back to current weights).
    """
    anchors = anchor_db.retrieve(
        query_kpi=np.asarray(query_kpi, dtype=float),
        regime=regime,
        top_k=1,
    )
    if anchors:
        return dict(anchors[0].weights)
    return None


def build_known_regime_weights() -> dict:
    """Return expert weights for known regimes (no evolution needed)."""
    from llm.mlp_architect import EXPERT_WEIGHTS
    return {
        regime: dict(zip(WEIGHT_KEYS, weights))
        for regime, weights in EXPERT_WEIGHTS.items()
    }
