"""Operator Intent System for Hybrid Architecture.

Three-timescale design:
  Strategic (LLM):   NL operator command → objective profile (minutes-hours)
  Tactical (MLP):    objective profile + KPIs → reward weights (seconds)
  Operational (DRL): reward weights → beam scheduling (ms)
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ObjectiveProfile:
    """Formal representation of operator intent — output of LLM layer."""
    throughput_priority: float = 0.5    # 0-1: how much to prioritize total throughput
    fairness_priority: float = 0.0      # 0-1: how much to prioritize beam fairness
    outage_tolerance: float = 0.5       # 0-1: 0=zero tolerance, 1=outage acceptable
    switching_tolerance: float = 0.5    # 0-1: 0=minimize switching, 1=switch freely
    queue_priority: float = 0.0         # 0-1: how much to prioritize queue reduction
    description: str = ""               # Human-readable description of intent

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.throughput_priority, self.fairness_priority,
            self.outage_tolerance, self.switching_tolerance, self.queue_priority
        ], dtype=np.float32)

    @staticmethod
    def from_vector(v, desc=""):
        return ObjectiveProfile(
            throughput_priority=float(v[0]), fairness_priority=float(v[1]),
            outage_tolerance=float(v[2]), switching_tolerance=float(v[3]),
            queue_priority=float(v[4]), description=desc
        )


# Pre-defined operator intent scenarios
INTENT_SCENARIOS = {
    "maximize_throughput": {
        "nl_command": "Maximize total network throughput. Service quality is secondary.",
        "profile": ObjectiveProfile(1.0, 0.0, 0.8, 0.8, 0.0, "max throughput"),
    },
    "emergency_priority": {
        "nl_command": "Emergency situation in the coverage area. Prioritize zero outage above all else. Accept lower throughput if needed.",
        "profile": ObjectiveProfile(0.3, 0.0, 0.0, 0.2, 0.3, "emergency zero-outage"),
    },
    "fair_coverage": {
        "nl_command": "Ensure fair bandwidth distribution across all beams. No single beam should be starved.",
        "profile": ObjectiveProfile(0.5, 1.0, 0.3, 0.3, 0.2, "fair coverage"),
    },
    "maritime_service": {
        "nl_command": "Serving maritime users with sparse but critical connectivity needs. Minimize outage for active beams, fairness matters.",
        "profile": ObjectiveProfile(0.6, 0.7, 0.1, 0.3, 0.1, "maritime service"),
    },
    "power_saving": {
        "nl_command": "Satellite entering eclipse period. Minimize beam switching to conserve power. Maintain basic service.",
        "profile": ObjectiveProfile(0.4, 0.0, 0.5, 0.0, 0.0, "power saving"),
    },
    "peak_hour": {
        "nl_command": "Peak usage hour in urban coverage. Push throughput as high as possible while keeping outage under control.",
        "profile": ObjectiveProfile(0.9, 0.1, 0.2, 0.6, 0.1, "peak hour urban"),
    },
    "disaster_relief": {
        "nl_command": "Natural disaster in beam 5-8 coverage area. Route maximum capacity to affected beams. Zero outage tolerance for those beams. Other beams can be degraded.",
        "profile": ObjectiveProfile(0.4, 0.0, 0.0, 0.1, 0.8, "disaster relief"),
    },
    "iot_collection": {
        "nl_command": "IoT data collection window. Many small packets across all beams. Fairness and queue management are critical, throughput per beam is low.",
        "profile": ObjectiveProfile(0.2, 0.8, 0.3, 0.2, 0.9, "iot collection"),
    },
}
