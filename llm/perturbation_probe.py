"""Single-Variable Perturbation Probing for Causal Feedback.

For each regime, perturbs ONE reward weight at a time (±delta),
runs a short PPO experiment, and measures the effect on performance.
Produces a sensitivity table that shows causal: "change X → effect on Y".
"""

import time
import numpy as np
from typing import Optional

from llm.mlp_architect import MLPArchitect, WEIGHT_KEYS
from llm.quality_filter import WEIGHT_BOUNDS
from simulator.env import BeamAllocationEnv, FlatActionWrapper
from agents.ppo_agent import PPOAgent, evaluate_agent


def _clamp_weight(value: float, key: str) -> float:
    """Clamp a weight to its valid bounds."""
    lo, hi = WEIGHT_BOUNDS[key]
    return max(lo, min(hi, value))


def evaluate_with_fixed_weights(
    regime: str,
    fixed_weights: dict,
    steps: int = 50_000,
    seed: int = 42,
    verbose: int = 0,
) -> dict:
    """Run PPO on a single regime with FIXED reward weights (no MLP adaptation).

    This is different from evaluate_single_regime() which uses MLP predictions.
    Here we set the weights once and keep them fixed throughout training,
    giving a clean measurement of how a specific weight config performs.

    Returns:
        {"rate_mbps": float, "outage": float, "fairness": float}
    """
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=[regime], epochs_per_regime=500, seed=seed))

    # Set fixed weights
    env.unwrapped.update_reward_weights(fixed_weights)

    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    # Train with fixed weights
    segment = min(12500, steps)
    total = 0
    while total < steps:
        s = min(segment, steps - total)
        agent.train(total_timesteps=s)
        total += s
        if verbose and total % 20480 == 0:
            print(f"      step={total} training...")

    # Evaluate
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=[regime], epochs_per_regime=100, seed=seed + 7777))
    eval_env.unwrapped.update_reward_weights(fixed_weights)
    final = evaluate_agent(agent, eval_env, n_episodes=3)

    return {
        "rate_mbps": final["mean_sum_rate_mbps"],
        "outage": final["mean_outage_count"],
        "fairness": final.get("mean_fairness_index", 0.0),
    }


def probe_single_regime(
    regime: str,
    base_weights: dict,
    delta: float = 0.2,
    probe_steps: int = 50_000,
    weight_keys_to_probe: Optional[list] = None,
    bidirectional: bool = True,
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    """Probe a single regime by perturbing one weight at a time.

    Args:
        regime: regime name (e.g., "iot_burst")
        base_weights: current weight config to perturb from
        delta: perturbation fraction (0.2 = ±20%)
        probe_steps: training steps per probe experiment
        weight_keys_to_probe: which weights to perturb (None = all 5)
        bidirectional: if True, probe both +delta and -delta; if False, only +delta
        seed: random seed
        verbose: print progress

    Returns:
        {
            "regime": str,
            "base_weights": dict,
            "base_performance": {"rate_mbps", "outage", "fairness"},
            "probes": [
                {"weight": "outage", "direction": "+20%",
                 "new_value": 1.44,
                 "performance": {"rate_mbps", "outage", "fairness"},
                 "delta_rate": -15.2,
                 "delta_outage": +0.05,
                 "delta_fairness": -0.01},
                ...
            ],
            "probe_time_s": float,
        }
    """
    if weight_keys_to_probe is None:
        weight_keys_to_probe = list(WEIGHT_KEYS)

    t0 = time.time()

    if verbose:
        print(f"\n  Probing regime: {regime}")
        print(f"  Base weights: {base_weights}")

    # 1. Baseline evaluation
    if verbose:
        print(f"  [Base] Running {probe_steps} steps...")
    base_perf = evaluate_with_fixed_weights(
        regime, base_weights, steps=probe_steps, seed=seed, verbose=0)
    if verbose:
        print(f"  [Base] rate={base_perf['rate_mbps']:.1f} "
              f"outage={base_perf['outage']:.2f} "
              f"fairness={base_perf['fairness']:.3f}")

    # 2. Perturbation probes
    probes = []
    directions = [+1, -1] if bidirectional else [+1]

    for wk in weight_keys_to_probe:
        for direction in directions:
            perturbed = dict(base_weights)
            old_val = perturbed[wk]
            # For very small values, use an absolute floor to avoid stuck-at-zero
            effective_base = max(old_val, 0.01)
            new_val = _clamp_weight(effective_base * (1.0 + direction * delta), wk)

            # Skip if clamping made it identical to base
            if abs(new_val - old_val) < 1e-4:
                if verbose:
                    print(f"  [{wk} {'+' if direction > 0 else '-'}{int(delta*100)}%] "
                          f"Skipped (clamped to same value)")
                continue

            perturbed[wk] = new_val
            dir_str = f"{'+' if direction > 0 else '-'}{int(delta*100)}%"

            if verbose:
                print(f"  [{wk} {dir_str}] {old_val:.4f} → {new_val:.4f} ...", end="", flush=True)

            perf = evaluate_with_fixed_weights(
                regime, perturbed, steps=probe_steps,
                seed=seed + hash(f"{wk}_{direction}") % 10000,
                verbose=0)

            delta_rate = perf["rate_mbps"] - base_perf["rate_mbps"]
            delta_outage = perf["outage"] - base_perf["outage"]
            delta_fairness = perf["fairness"] - base_perf["fairness"]

            if verbose:
                print(f" Δrate={delta_rate:+.1f} Δoutage={delta_outage:+.2f} "
                      f"Δfair={delta_fairness:+.3f}")

            probes.append({
                "weight": wk,
                "direction": dir_str,
                "old_value": round(old_val, 4),
                "new_value": round(new_val, 4),
                "performance": perf,
                "delta_rate": round(delta_rate, 2),
                "delta_outage": round(delta_outage, 3),
                "delta_fairness": round(delta_fairness, 4),
            })

    probe_time = time.time() - t0

    result = {
        "regime": regime,
        "base_weights": base_weights,
        "base_performance": base_perf,
        "probes": probes,
        "probe_time_s": round(probe_time, 1),
    }

    if verbose:
        print(f"  Probe complete for {regime}: {len(probes)} probes in {probe_time:.0f}s")
        # Print summary table
        print(f"\n  {'Weight':<15} {'Direction':<10} {'ΔRate':>8} {'ΔOutage':>9} {'ΔFair':>8}")
        print(f"  {'-'*50}")
        for p in probes:
            print(f"  {p['weight']:<15} {p['direction']:<10} "
                  f"{p['delta_rate']:>+8.1f} {p['delta_outage']:>+9.3f} "
                  f"{p['delta_fairness']:>+8.4f}")

    return result


def probe_all_novel_regimes(
    novel_regimes: list,
    regime_weights: dict,
    delta: float = 0.2,
    probe_steps: int = 50_000,
    weight_keys_to_probe: Optional[list] = None,
    bidirectional: bool = True,
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    """Probe all novel regimes and return combined results.

    Args:
        novel_regimes: list of regime names to probe
        regime_weights: {regime_name: weights_dict} for each regime
        Other args: same as probe_single_regime

    Returns:
        {regime_name: probe_result_dict, ...}
    """
    all_results = {}
    total_t0 = time.time()

    for i, regime in enumerate(novel_regimes):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Probing regime {i+1}/{len(novel_regimes)}: {regime}")
            print(f"{'='*50}")

        base_w = regime_weights.get(regime)
        if base_w is None:
            print(f"  WARNING: No weights for {regime}, skipping")
            continue

        result = probe_single_regime(
            regime=regime,
            base_weights=base_w,
            delta=delta,
            probe_steps=probe_steps,
            weight_keys_to_probe=weight_keys_to_probe,
            bidirectional=bidirectional,
            seed=seed + i * 1000,
            verbose=verbose,
        )
        all_results[regime] = result

    total_time = time.time() - total_t0
    if verbose:
        print(f"\n  All probes complete: {len(all_results)} regimes in {total_time:.0f}s")

    return all_results


def format_sensitivity_table(probe_result: dict) -> str:
    """Format probe result as a markdown table for LLM consumption."""
    lines = []
    lines.append(f"Regime: {probe_result['regime']}")
    bp = probe_result["base_performance"]
    bw = probe_result["base_weights"]
    lines.append(f"Base weights: sr={bw.get('sum_rate',0):.3f} fair={bw.get('fairness',0):.3f} "
                 f"out={bw.get('outage',0):.3f} sw={bw.get('switching',0):.3f} q={bw.get('queue',0):.3f}")
    lines.append(f"Base performance: rate={bp['rate_mbps']:.1f} outage={bp['outage']:.2f} "
                 f"fairness={bp['fairness']:.3f}")
    lines.append("")
    lines.append(f"| {'Weight':<12} | {'Direction':<10} | {'New Value':>10} | "
                 f"{'Δ Rate':>8} | {'Δ Outage':>9} | {'Δ Fairness':>10} |")
    lines.append(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*11}|{'-'*12}|")

    for p in probe_result["probes"]:
        lines.append(f"| {p['weight']:<12} | {p['direction']:<10} | {p['new_value']:>10.4f} | "
                     f"{p['delta_rate']:>+8.1f} | {p['delta_outage']:>+9.3f} | "
                     f"{p['delta_fairness']:>+10.4f} |")

    return "\n".join(lines)
