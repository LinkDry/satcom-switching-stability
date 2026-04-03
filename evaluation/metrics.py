"""Evaluation metrics: sum rate, outage, fairness, recovery time."""

import numpy as np
from typing import Optional


def compute_sum_rate(rates: list[float]) -> dict:
    """Aggregate sum rate statistics."""
    arr = np.array(rates)
    return {
        "mean_mbps": float(np.mean(arr)),
        "std_mbps": float(np.std(arr)),
        "median_mbps": float(np.median(arr)),
        "min_mbps": float(np.min(arr)),
        "max_mbps": float(np.max(arr)),
    }


def compute_outage_probability(
    per_step_outage_counts: list[int],
    per_step_active_counts: list[int],
) -> float:
    """Fraction of active beams that are in outage."""
    total_outage = sum(per_step_outage_counts)
    total_active = sum(per_step_active_counts)
    if total_active == 0:
        return 0.0
    return total_outage / total_active


def compute_jains_fairness(per_beam_rates: list[list[float]]) -> float:
    """Jain's fairness index across beams, averaged over time.

    J(x) = (sum(x))^2 / (n * sum(x^2))
    """
    fairness_values = []
    for rates in per_beam_rates:
        arr = np.array(rates)
        arr = arr[arr > 0]  # Only active beams
        if len(arr) < 2:
            continue
        jain = (np.sum(arr) ** 2) / (len(arr) * np.sum(arr ** 2))
        fairness_values.append(jain)
    return float(np.mean(fairness_values)) if fairness_values else 0.0


def compute_recovery_time(
    per_step_rates: list[float],
    regime_change_steps: list[int],
    steady_state_fraction: float = 0.9,
    window_size: int = 20,
) -> list[dict]:
    """Compute recovery time after each regime change.

    Recovery = epochs until rolling average reaches 90% of pre-change level.
    """
    rates = np.array(per_step_rates)
    results = []

    for change_step in regime_change_steps:
        if change_step < window_size or change_step >= len(rates) - window_size:
            continue

        # Pre-change steady state
        pre_mean = np.mean(rates[change_step - window_size : change_step])
        target = pre_mean * steady_state_fraction

        # Find recovery time
        recovery = None
        for t in range(change_step, min(change_step + 500, len(rates) - window_size)):
            rolling_mean = np.mean(rates[t : t + window_size])
            if rolling_mean >= target:
                recovery = t - change_step
                break

        results.append({
            "change_step": change_step,
            "pre_change_rate": float(pre_mean),
            "target_rate": float(target),
            "recovery_epochs": recovery,
        })

    return results


def compute_per_regime_metrics(
    per_step_rates: list[float],
    per_step_regimes: list[str],
) -> dict:
    """Compute metrics broken down by regime type."""
    regime_rates = {}
    for rate, regime in zip(per_step_rates, per_step_regimes):
        if regime not in regime_rates:
            regime_rates[regime] = []
        regime_rates[regime].append(rate)

    return {
        regime: compute_sum_rate(rates)
        for regime, rates in regime_rates.items()
    }


def aggregate_experiment_metrics(
    per_step_rates: list[float],
    per_step_outages: list[int],
    per_step_active: list[int],
    per_beam_rates: list[list[float]],
    per_step_regimes: list[str],
    regime_change_steps: list[int],
) -> dict:
    """Compute all metrics for a complete experiment run."""
    return {
        "sum_rate": compute_sum_rate(per_step_rates),
        "outage_probability": compute_outage_probability(per_step_outages, per_step_active),
        "jains_fairness": compute_jains_fairness(per_beam_rates),
        "recovery_times": compute_recovery_time(per_step_rates, regime_change_steps),
        "per_regime": compute_per_regime_metrics(per_step_rates, per_step_regimes),
    }
