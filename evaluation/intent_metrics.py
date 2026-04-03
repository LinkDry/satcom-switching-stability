"""Intent Satisfaction Metrics for Hybrid Architecture Evaluation.

Key insight: Raw throughput alone cannot measure hybrid value.
Each operator intent has DIFFERENT success criteria.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class IntentMetrics:
    """Per-step metrics collected during evaluation."""
    sum_rate: float = 0.0
    outage_count: float = 0.0
    fairness_index: float = 0.0      # Jain's fairness index
    active_beam_ratio: float = 0.0    # For energy efficiency
    queue_overflow: float = 0.0       # For queue-sensitive intents
    beam_coverage: float = 0.0        # Fraction of beams meeting min QoS


# Intent-specific satisfaction functions
# Each returns 0-1 score based on how well metrics match the intent

def satisfaction_max_throughput(m: IntentMetrics) -> float:
    """Operator wants maximum throughput."""
    rate_score = min(m.sum_rate / 400.0, 1.0)  # Normalize to ~400 Mbps max
    outage_penalty = max(0, 1.0 - m.outage_count * 0.1)
    return 0.8 * rate_score + 0.2 * outage_penalty


def satisfaction_emergency(m: IntentMetrics) -> float:
    """Operator wants zero outage + maximum coverage during emergency."""
    outage_score = max(0, 1.0 - m.outage_count * 0.5)  # Heavy penalty
    coverage_score = m.beam_coverage
    rate_score = min(m.sum_rate / 400.0, 1.0)
    return 0.15 * rate_score + 0.50 * outage_score + 0.35 * coverage_score


def satisfaction_fairness(m: IntentMetrics) -> float:
    """Operator wants fair resource distribution across beams."""
    fairness_score = m.fairness_index  # Jain's index, 0-1
    rate_score = min(m.sum_rate / 400.0, 1.0)
    outage_penalty = max(0, 1.0 - m.outage_count * 0.2)
    return 0.20 * rate_score + 0.60 * fairness_score + 0.20 * outage_penalty


def satisfaction_energy_saving(m: IntentMetrics) -> float:
    """Operator wants to minimize active beams while maintaining service."""
    efficiency = 1.0 - m.active_beam_ratio  # Fewer active = better
    outage_penalty = max(0, 1.0 - m.outage_count * 0.3)
    rate_score = min(m.sum_rate / 400.0, 1.0)
    return 0.25 * rate_score + 0.45 * efficiency + 0.30 * outage_penalty


def satisfaction_iot_collection(m: IntentMetrics) -> float:
    """Operator wants broad coverage for IoT data collection."""
    coverage_score = m.beam_coverage
    queue_score = max(0, 1.0 - m.queue_overflow * 0.1)
    rate_score = min(m.sum_rate / 400.0, 1.0)
    return 0.15 * rate_score + 0.45 * coverage_score + 0.40 * queue_score


# Map intent descriptions to satisfaction functions
SATISFACTION_MAP = {
    "max_throughput": satisfaction_max_throughput,
    "emergency": satisfaction_emergency,
    "fairness": satisfaction_fairness,
    "energy_saving": satisfaction_energy_saving,
    "iot_collection": satisfaction_iot_collection,
    # Fallback
    "default": satisfaction_max_throughput,
}


def compute_intent_satisfaction(metrics: IntentMetrics, intent_desc: str) -> float:
    """Compute satisfaction score for given metrics and intent."""
    # Match intent description to satisfaction function
    for key, func in SATISFACTION_MAP.items():
        if key in intent_desc.lower():
            return func(metrics)
    return SATISFACTION_MAP["default"](metrics)


def compute_jain_fairness(rates: np.ndarray) -> float:
    """Jain's fairness index: (sum(x))^2 / (n * sum(x^2))."""
    if len(rates) == 0 or np.sum(rates) == 0:
        return 0.0
    n = len(rates)
    return float(np.sum(rates) ** 2 / (n * np.sum(rates ** 2)))


@dataclass
class IntentSatisfactionTracker:
    """Track satisfaction scores across intent phases."""
    phase_scores: Dict[str, List[float]] = field(default_factory=dict)
    phase_metrics: Dict[str, List[IntentMetrics]] = field(default_factory=dict)

    def record(self, intent_desc: str, metrics: IntentMetrics):
        score = compute_intent_satisfaction(metrics, intent_desc)
        if intent_desc not in self.phase_scores:
            self.phase_scores[intent_desc] = []
            self.phase_metrics[intent_desc] = []
        self.phase_scores[intent_desc].append(score)
        self.phase_metrics[intent_desc].append(metrics)

    def summary(self) -> Dict:
        result = {}
        for intent, scores in self.phase_scores.items():
            metrics_list = self.phase_metrics[intent]
            result[intent] = {
                "mean_satisfaction": float(np.mean(scores)),
                "mean_rate": float(np.mean([m.sum_rate for m in metrics_list])),
                "mean_outage": float(np.mean([m.outage_count for m in metrics_list])),
                "mean_fairness": float(np.mean([m.fairness_index for m in metrics_list])),
                "n_evals": len(scores),
            }
        # Overall weighted satisfaction
        all_scores = [s for scores in self.phase_scores.values() for s in scores]
        result["overall_satisfaction"] = float(np.mean(all_scores)) if all_scores else 0.0
        return result
