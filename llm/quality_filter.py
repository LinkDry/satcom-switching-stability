"""Quality filter for LLM-generated oracle data.

Filters (KPI → weights) pairs by:
1. Bounds check: all weights within valid ranges
2. Consistency check: weights make physical sense for the KPI regime
3. Optional short-rollout verification: run a quick sim episode to check performance
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

from llm.mlp_architect import WEIGHT_KEYS

# Valid weight bounds (physical constraints)
WEIGHT_BOUNDS = {
    "sum_rate": (0.01, 3.0),
    "fairness": (0.0, 2.0),
    "outage": (0.1, 5.0),
    "switching": (0.001, 2.0),
    "queue": (0.0, 2.0),
}

# Regime-specific consistency rules
REGIME_CONSISTENCY = {
    "urban": {"sum_rate": (0.5, None), "fairness": (None, 0.5)},
    "maritime": {"fairness": (0.1, None)},
    "disaster": {"outage": (1.0, None)},
    "iot_burst": {"queue": (0.3, None)},
    "polar_handover": {"switching": (0.3, None)},
    "hot_cold": {"outage": (0.5, None)},
}


def bounds_check(weights: dict) -> bool:
    """Check all weights are within valid physical bounds."""
    for k in WEIGHT_KEYS:
        if k not in weights:
            return False
        lo, hi = WEIGHT_BOUNDS[k]
        if weights[k] < lo or weights[k] > hi:
            return False
    return True


def consistency_check(weights: dict, regime: str) -> bool:
    """Check weights are consistent with regime characteristics."""
    rules = REGIME_CONSISTENCY.get(regime)
    if rules is None:
        return True  # No rules for unknown regimes, pass by default

    for key, (lo, hi) in rules.items():
        if key not in weights:
            continue
        if lo is not None and weights[key] < lo:
            return False
        if hi is not None and weights[key] > hi:
            return False
    return True


def stability_check(weights: dict, reference_weights: Optional[dict] = None, max_deviation: float = 2.0) -> bool:
    """Check weights don't deviate too far from a reference (anti-oscillation)."""
    if reference_weights is None:
        return True
    for k in WEIGHT_KEYS:
        if k in reference_weights and k in weights:
            ref = max(reference_weights[k], 0.01)
            ratio = weights[k] / ref
            if ratio > max_deviation or ratio < 1.0 / max_deviation:
                return False
    return True


def rollout_verify(
    weights: dict,
    regime: str,
    n_steps: int = 2000,
    seed: int = 42,
    min_rate_threshold: float = 50.0,
) -> dict:
    """Run a short rollout with given weights to verify they produce reasonable performance.

    Returns dict with {passed: bool, rate: float, outage: float, reason: str}.
    """
    from simulator.env import BeamAllocationEnv, FlatActionWrapper
    from agents.ppo_agent import PPOAgent, evaluate_agent

    regime_list = [regime] if regime in [
        "urban", "maritime", "disaster", "mixed",
        "iot_burst", "polar_handover", "hot_cold"
    ] else ["mixed"]

    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=regime_list, epochs_per_regime=500, seed=seed))
    env.unwrapped.update_reward_weights(weights)

    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)
    agent.train(total_timesteps=n_steps)

    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=regime_list, epochs_per_regime=100, seed=seed + 1))
    eval_env.unwrapped.update_reward_weights(weights)
    metrics = evaluate_agent(agent, eval_env, n_episodes=2)

    rate = metrics["mean_sum_rate_mbps"]
    outage = metrics["mean_outage_count"]

    passed = rate >= min_rate_threshold
    reason = "ok" if passed else f"rate={rate:.1f} < {min_rate_threshold}"

    return {"passed": passed, "rate": rate, "outage": outage, "reason": reason}


def filter_oracle_data(
    samples: list[dict],
    use_bounds: bool = True,
    use_consistency: bool = True,
    use_rollout: bool = False,
    rollout_fraction: float = 0.1,
    seed: int = 42,
    verbose: int = 1,
    skip_consistency: bool = False,
) -> list[dict]:
    """Apply quality filters to oracle-generated samples.

    Args:
        samples: list of {kpi, weights, regime, source} dicts
        use_bounds: apply weight bounds check
        use_consistency: apply regime consistency check
        use_rollout: apply short-rollout verification (expensive)
        rollout_fraction: fraction of samples to rollout-verify
        seed: random seed for rollout sampling
        verbose: print progress

    Returns:
        Filtered list of samples with added quality_score field.
    """
    rng = np.random.default_rng(seed)
    filtered = []
    stats = {"total": len(samples), "bounds_fail": 0, "consistency_fail": 0,
             "rollout_fail": 0, "passed": 0}

    for i, sample in enumerate(samples):
        weights = sample["weights"]
        regime = sample.get("regime", "unknown").split("transition_")[-1].split("_")[0] if "transition" in sample.get("regime", "") else sample.get("regime", "unknown")

        # Bounds check
        if use_bounds and not bounds_check(weights):
            stats["bounds_fail"] += 1
            continue

        # Consistency check
        if use_consistency and not skip_consistency and not consistency_check(weights, regime):
            stats["consistency_fail"] += 1
            continue

        # Quality score based on how well weights match expected patterns
        score = 1.0
        if regime in REGIME_CONSISTENCY:
            rules = REGIME_CONSISTENCY[regime]
            for key, (lo, hi) in rules.items():
                if key in weights:
                    if lo is not None:
                        score *= min(1.0, weights[key] / lo)
                    if hi is not None and weights[key] > 0:
                        score *= min(1.0, hi / weights[key])

        sample_out = {**sample, "quality_score": round(score, 3)}
        filtered.append(sample_out)

    stats["passed"] = len(filtered)

    # Optional rollout verification on a subset
    if use_rollout and filtered:
        n_verify = max(1, int(len(filtered) * rollout_fraction))
        verify_indices = rng.choice(len(filtered), size=n_verify, replace=False)

        if verbose:
            print(f"  Rollout-verifying {n_verify} samples...")

        rollout_failures = 0
        for idx in verify_indices:
            s = filtered[idx]
            result = rollout_verify(s["weights"], s.get("regime", "mixed"), seed=seed + idx)
            if not result["passed"]:
                filtered[idx]["quality_score"] *= 0.3  # Penalize but don't remove
                rollout_failures += 1

        stats["rollout_fail"] = rollout_failures
        if verbose:
            print(f"  Rollout: {rollout_failures}/{n_verify} below threshold")

    if verbose:
        print(f"  Filter stats: {stats}")

    return filtered
