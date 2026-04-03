"""LLM Reflection Prompt Builder + Synthetic Evolution Fallback.

Builds structured prompts that show the LLM experimental results,
enabling it to reason about failures and generate improved weights.
Also provides a synthetic fallback that mimics this evolution without LLM API.
"""

import json
import time
import numpy as np
from typing import Optional

from llm.evolution_db import EvolutionDB, WEIGHT_KEYS
from llm.quality_filter import WEIGHT_BOUNDS

# --- Prompt Template ---

REFLECTION_PROMPT = """You are optimizing reward weights for a 19-beam LEO satellite scheduler.
Round {round_id}. Analyze results below, output IMPROVED weights for each regime.

Reward: R = sum_rate*throughput - outage*outages - switching*switches - queue*queues + fairness*fair_idx

Bounds: sum_rate[{lb_sr},{ub_sr}] fairness[{lb_f},{ub_f}] outage[{lb_o},{ub_o}] switching[{lb_sw},{ub_sw}] queue[{lb_q},{ub_q}]

Baseline MLP (~342 Mbps): sum_rate≈1.0, outage≈1.0-1.9, fairness/switching/queue very small (<0.1).

CONSTRAINT: Each weight ±50% of previous round. Keep sum_rate near 1.0.
{prev_weights_section}

Performance:
{round_summary}

Gap vs Baseline:
{gap_section}

Trajectory:
{trajectory_section}

Regimes: {regime_list}

Output ONLY JSON array (no explanation):
[{{"regime":"name","weights":{{"sum_rate":...,"fairness":...,"outage":...,"switching":...,"queue":...}},"reasoning":"brief"}}]"""


def _format_record(rec) -> str:
    w = rec.weights_used
    p = rec.performance
    return (f"  Round {rec.round_id}: rate={p.get('rate_mbps',0):.1f} outage={p.get('outage',0):.2f} "
            f"fairness={p.get('fairness',0):.3f}\n"
            f"    weights: sr={w.get('sum_rate',0):.3f} fair={w.get('fairness',0):.3f} "
            f"out={w.get('outage',0):.3f} sw={w.get('switching',0):.3f} q={w.get('queue',0):.3f}")


def build_round_summary(db: EvolutionDB, round_id: int) -> str:
    summary = db.get_round_summary(round_id)
    if not summary:
        return "(no data for this round)"
    lines = [f"{'Regime':<20} {'Rate(Mbps)':>10} {'Outage':>8} {'Fairness':>10}"]
    lines.append("-" * 52)
    for regime, perf in sorted(summary.items()):
        lines.append(f"{regime:<20} {perf.get('rate_mbps',0):>10.1f} "
                     f"{perf.get('outage',0):>8.2f} {perf.get('fairness',0):>10.3f}")
    return "\n".join(lines)


def build_best_section(db: EvolutionDB, regimes: list[str], top_k: int = 2) -> str:
    parts = []
    for regime in regimes:
        best = db.get_best_per_regime(regime, top_k)
        if not best:
            continue
        parts.append(f"\n### {regime}")
        for rec in best:
            parts.append(_format_record(rec))
    return "\n".join(parts) if parts else "(no data)"


def build_worst_section(db: EvolutionDB, regimes: list[str], top_k: int = 2) -> str:
    parts = []
    for regime in regimes:
        worst = db.get_worst_per_regime(regime, top_k)
        if not worst:
            continue
        parts.append(f"\n### {regime}")
        for rec in worst:
            parts.append(_format_record(rec))
    return "\n".join(parts) if parts else "(no data)"


def build_gap_section(db: EvolutionDB, regimes: list[str]) -> str:
    lines = [f"{'Regime':<20} {'Your Best':>10} {'Baseline':>10} {'Gap':>8}"]
    lines.append("-" * 52)
    for regime in regimes:
        gap = db.get_baseline_gap(regime)
        if gap:
            lines.append(f"{regime:<20} {gap['current_best_rate']:>10.1f} "
                         f"{gap['baseline_rate']:>10.1f} {gap['rate_gap_pct']:>+7.1f}%")
    return "\n".join(lines)


def build_trajectory_section(db: EvolutionDB, regimes: list[str]) -> str:
    parts = []
    for regime in regimes:
        traj = db.get_regime_trajectory(regime)
        if traj:
            steps = " → ".join(f"R{t['round_id']}:{t.get('rate_mbps',0):.0f}" for t in traj)
            parts.append(f"  {regime}: {steps}")
    return "\n".join(parts) if parts else "(no trajectory data)"


def build_prev_weights_section(db: EvolutionDB, regimes: list[str]) -> str:
    """Show previous round's best weights per regime as anchor for gradual evolution."""
    parts = ["Previous round best weights (your new weights must stay within ±50% of these):"]
    for regime in regimes:
        best = db.get_best_per_regime(regime, top_k=1)
        if best:
            w = best[0].weights_used
            parts.append(f"  {regime}: sr={w.get('sum_rate',1.0):.3f} fair={w.get('fairness',0.0):.3f} "
                         f"out={w.get('outage',1.0):.3f} sw={w.get('switching',0.01):.3f} q={w.get('queue',0.0):.3f}")
    return "\n".join(parts)


def build_reflection_prompt(db: EvolutionDB, round_id: int, regimes: list[str]) -> str:
    """Assemble the reflection prompt — compact version for local models."""
    prev_round = round_id - 1
    bounds = {k: WEIGHT_BOUNDS[k] for k in WEIGHT_KEYS}
    return REFLECTION_PROMPT.format(
        round_id=round_id,
        round_summary=build_round_summary(db, prev_round),
        gap_section=build_gap_section(db, regimes),
        trajectory_section=build_trajectory_section(db, regimes),
        prev_weights_section=build_prev_weights_section(db, regimes),
        regime_list=", ".join(regimes),
        lb_sr=bounds["sum_rate"][0], ub_sr=bounds["sum_rate"][1],
        lb_f=bounds["fairness"][0], ub_f=bounds["fairness"][1],
        lb_o=bounds["outage"][0], ub_o=bounds["outage"][1],
        lb_sw=bounds["switching"][0], ub_sw=bounds["switching"][1],
        lb_q=bounds["queue"][0], ub_q=bounds["queue"][1],
    )


# --- LLM Evolution (real API) ---

def llm_evolve(
    db: EvolutionDB,
    round_id: int,
    regimes: list[str],
    model: str = "qwen/qwen3.6-plus-preview:free",
    api_key: str = None,
    base_url: str = "https://openrouter.ai/api/v1",
    verbose: int = 1,
    max_retries: int = 3,
) -> list[dict]:
    """Call LLM to analyze results and generate improved weights.

    Uses requests directly instead of openai SDK to avoid compatibility issues
    with local LM Studio servers.
    """
    import requests as req

    prompt = build_reflection_prompt(db, round_id, regimes)

    if verbose:
        print(f"  [LLM] Sending reflection prompt ({len(prompt)} chars)...")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "lm-studio":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert satellite communication system optimizer. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 8000,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            r = req.post(url, json=payload, headers=headers, timeout=600)
            latency = time.time() - t0

            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

            data = r.json()
            content = data["choices"][0]["message"]["content"]

            if verbose:
                print(f"  [LLM] Response in {latency:.1f}s: {content[:300]}...")

            evolved = _parse_evolved_response(content, regimes)
            if evolved:
                evolved = _apply_gradual_clamp(db, evolved)
                if verbose:
                    for e in evolved:
                        print(f"  [LLM] {e['regime']}: {e['weights']} — {e.get('reasoning','')[:60]}")
                return evolved
            else:
                print(f"  [LLM] Attempt {attempt+1}: parse failed, retrying...")
                last_error = "parse_failure"

        except Exception as e:
            last_error = str(e)
            if verbose:
                print(f"  [LLM] Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                if verbose:
                    print(f"  [LLM] Retrying in {wait}s...")
                time.sleep(wait)

    print(f"  [LLM] All {max_retries} attempts failed (last error: {last_error})")
    return []


def _parse_evolved_response(content: str, expected_regimes: list[str]) -> list[dict]:
    """Parse LLM response into list of {regime, weights, reasoning}."""
    import re

    # Try to extract JSON array
    # First try code block
    m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', content)
    if m:
        text = m.group(1)
    else:
        # Try raw array
        m = re.search(r'\[[\s\S]*\]', content)
        text = m.group(0) if m else content

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if "regime" in item and "weights" in item:
                    w = item["weights"]
                    # Validate and clamp
                    clamped = {}
                    for k in WEIGHT_KEYS:
                        val = float(w.get(k, 0.5))
                        lo, hi = WEIGHT_BOUNDS[k]
                        clamped[k] = round(max(lo, min(hi, val)), 4)
                    results.append({
                        "regime": item["regime"],
                        "weights": clamped,
                        "reasoning": item.get("reasoning", ""),
                    })
            if results:
                return results
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: return empty (caller should use synthetic)
    print("  [LLM] Failed to parse response, falling back to synthetic evolution")
    return []


def _apply_gradual_clamp(db: EvolutionDB, evolved: list[dict], max_change: float = 0.5) -> list[dict]:
    """Enforce ±max_change (50%) gradual constraint relative to previous best weights.

    For each regime, clamp every weight to [prev * (1 - max_change), prev * (1 + max_change)].
    If no previous data exists, skip clamping for that regime.
    """
    clamped = []
    for entry in evolved:
        regime = entry["regime"]
        weights = dict(entry["weights"])

        best = db.get_best_per_regime(regime, top_k=1)
        if best:
            prev_w = best[0].weights_used
            for k in WEIGHT_KEYS:
                prev_val = prev_w.get(k, weights[k])
                # For very small values, use absolute floor to avoid stuck-at-zero
                floor = max(prev_val, 0.005)
                lo = floor * (1.0 - max_change)
                hi = floor * (1.0 + max_change)
                # Also respect global bounds
                glo, ghi = WEIGHT_BOUNDS[k]
                lo = max(lo, glo)
                hi = min(hi, ghi)
                weights[k] = round(max(lo, min(hi, weights[k])), 4)

        clamped.append({
            "regime": entry["regime"],
            "weights": weights,
            "reasoning": entry.get("reasoning", ""),
        })
    return clamped


# --- Synthetic Evolution Fallback ---

def synthetic_evolve(
    db: EvolutionDB,
    round_id: int,
    regimes: list[str],
    verbose: int = 1,
) -> list[dict]:
    """Programmatic evolution — mimics LLM reasoning with heuristic rules.

    Strategy:
    1. Start from best-performing weights of previous rounds
    2. Compute direction: best - worst (what to move toward)
    3. Apply domain corrections based on baseline gap analysis
    4. Enforce gradual ±50% constraint relative to previous best
    5. Clamp to valid bounds
    """
    evolved = []

    for regime in regimes:
        best = db.get_best_per_regime(regime, top_k=1)
        worst = db.get_worst_per_regime(regime, top_k=1)
        gap = db.get_baseline_gap(regime)

        if not best:
            center = {"sum_rate": 1.0, "fairness": 0.1, "outage": 1.0, "switching": 0.02, "queue": 0.05}
        else:
            center = dict(best[0].weights_used)

            # Evolution direction: move away from worst
            if worst and worst[0].weights_used != best[0].weights_used:
                step = 0.15 / max(1, round_id ** 0.5)
                for k in WEIGHT_KEYS:
                    direction = center[k] - worst[0].weights_used.get(k, center[k])
                    center[k] += direction * step

            # Domain corrections based on gap analysis
            if gap:
                rate_gap = gap.get("rate_gap_pct", 0)
                best_perf = best[0].performance

                if rate_gap > 40:
                    # Cap sum_rate at 1.2 — baseline MLP uses 1.0, going higher causes instability
                    center["sum_rate"] = min(center["sum_rate"] * 1.15, 1.2)
                    center["fairness"] = max(center["fairness"] * 0.7, WEIGHT_BOUNDS["fairness"][0])
                    center["queue"] = max(center["queue"] * 0.7, WEIGHT_BOUNDS["queue"][0])
                elif rate_gap > 20:
                    center["sum_rate"] = min(center["sum_rate"] * 1.1, 1.2)

                if best_perf.get("outage", 0) > 0.1:
                    center["outage"] = min(center["outage"] + 0.2, WEIGHT_BOUNDS["outage"][1])

                if best_perf.get("outage", 0) == 0 and rate_gap > 30:
                    center["outage"] = max(center["outage"] * 0.9, WEIGHT_BOUNDS["outage"][0])

        # Clamp all to bounds
        for k in WEIGHT_KEYS:
            lo, hi = WEIGHT_BOUNDS[k]
            center[k] = round(max(lo, min(hi, center[k])), 4)

        evolved.append({
            "regime": regime,
            "weights": center,
            "reasoning": f"synthetic_round_{round_id}",
        })

        if verbose:
            print(f"  [Synthetic] {regime}: {center}")

    # Apply gradual clamp (±50% of previous best)
    evolved = _apply_gradual_clamp(db, evolved)
    return evolved
