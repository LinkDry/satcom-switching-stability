"""Regime change detection from KPI time series."""

import numpy as np
from typing import Optional


class CUSUMDetector:
    """CUSUM-based change-point detector for traffic regime shifts.

    Monitors multiple KPIs and triggers when cumulative sum of deviations
    from the running mean exceeds a threshold.
    """

    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 0.5,
        min_interval: int = 30,
        kpi_keys: Optional[list[str]] = None,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_interval = min_interval
        self.kpi_keys = kpi_keys or ["avg_demand", "demand_variance", "spatial_gini"]

        self.history: list[dict] = []
        self.steps_since_last_detection = 0
        self.cusum_pos = {k: 0.0 for k in self.kpi_keys}
        self.cusum_neg = {k: 0.0 for k in self.kpi_keys}

    def update(self, kpi: dict) -> bool:
        """Update with new KPI snapshot, return True if change-point detected."""
        self.history.append(kpi)
        self.steps_since_last_detection += 1

        if len(self.history) < self.window_size:
            return False

        if self.steps_since_last_detection < self.min_interval:
            return False

        # Compute running statistics from recent window
        window = self.history[-self.window_size :]
        detected = False

        for key in self.kpi_keys:
            values = [w[key] for w in window if key in w]
            if len(values) < self.window_size // 2:
                continue

            mean = np.mean(values[: len(values) // 2])  # first half mean
            std = max(np.std(values[: len(values) // 2]), 1e-6)

            # Current value normalized deviation
            current = values[-1]
            z = (current - mean) / std

            # Update CUSUM
            self.cusum_pos[key] = max(0, self.cusum_pos[key] + z - 0.5)
            self.cusum_neg[key] = max(0, self.cusum_neg[key] - z - 0.5)

            if self.cusum_pos[key] > self.threshold or self.cusum_neg[key] > self.threshold:
                detected = True

        if detected:
            self._reset_cusum()
            self.steps_since_last_detection = 0

        return detected

    def _reset_cusum(self):
        for k in self.kpi_keys:
            self.cusum_pos[k] = 0.0
            self.cusum_neg[k] = 0.0

    def get_regime_summary(self) -> dict:
        """Get summary of current KPI statistics for LLM context."""
        if len(self.history) < 10:
            return {"status": "insufficient_data", "samples": len(self.history)}

        recent = self.history[-min(50, len(self.history)) :]
        summary = {}
        for key in self.kpi_keys:
            values = [w[key] for w in recent if key in w]
            if values:
                summary[key] = {
                    "current": float(values[-1]),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "trend": float(np.polyfit(range(len(values)), values, 1)[0])
                    if len(values) > 3
                    else 0.0,
                }

        # Add derived regime hints
        if "avg_demand" in summary and "spatial_gini" in summary:
            avg = summary["avg_demand"]["current"]
            gini = summary["spatial_gini"]["current"]
            peak = self.history[-1].get("peak_beam_demand", 0)

            if peak > 120:
                summary["regime_hint"] = "disaster-like (high peak demand)"
            elif avg > 40 and gini > 0.3:
                summary["regime_hint"] = "urban-like (high concentrated demand)"
            elif avg < 20:
                summary["regime_hint"] = "maritime-like (low uniform demand)"
            else:
                summary["regime_hint"] = "mixed/transition"

        summary["status"] = "ready"
        summary["samples"] = len(self.history)
        return summary


def detect_regime_change(kpi_window: list[dict], threshold: float = 0.5) -> bool:
    """Simple stateless regime detection for use as a callback.

    Compares second half of window to first half using normalized deviation.
    """
    if len(kpi_window) < 20:
        return False

    mid = len(kpi_window) // 2
    first_half = kpi_window[:mid]
    second_half = kpi_window[mid:]

    keys = ["avg_demand", "demand_variance", "spatial_gini"]
    for key in keys:
        v1 = [w[key] for w in first_half if key in w]
        v2 = [w[key] for w in second_half if key in w]
        if not v1 or not v2:
            continue
        m1, s1 = np.mean(v1), max(np.std(v1), 1e-6)
        m2 = np.mean(v2)
        if abs(m2 - m1) / s1 > threshold:
            return True

    return False
