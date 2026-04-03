"""Performance Feedback Database for Self-Evolving LLM-MLP.

Stores (round, regime, weights, performance) records from each evolution round.
Supports querying best/worst per regime, baseline gap analysis, and evolution trajectory.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

WEIGHT_KEYS = ["sum_rate", "fairness", "outage", "switching", "queue"]


@dataclass
class ExperimentRecord:
    round_id: int
    regime: str
    kpi_snapshot: dict
    weights_used: dict
    performance: dict  # {rate_mbps, outage, fairness}
    source: str  # "expert_hint" | "llm_evolved" | "synthetic_evolved"
    timestamp: str
    reasoning: str = ""


class EvolutionDB:
    """JSON-backed performance feedback database."""

    def __init__(self, db_path: str, baseline_metrics: Optional[dict] = None):
        self.db_path = Path(db_path)
        self.records: list[ExperimentRecord] = []
        self.baseline_metrics = baseline_metrics or {}

    def add_record(self, record: ExperimentRecord):
        self.records.append(record)

    def save(self):
        data = {
            "metadata": {"baseline_metrics": self.baseline_metrics},
            "records": [asdict(r) for r in self.records],
        }
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if self.db_path.exists():
            with open(self.db_path) as f:
                data = json.load(f)
            self.baseline_metrics = data.get("metadata", {}).get("baseline_metrics", {})
            self.records = [ExperimentRecord(**r) for r in data["records"]]

    def _composite_score(self, perf: dict) -> float:
        """Ranking: 0.6*normalized_rate + 0.2*(1-outage) + 0.2*fairness."""
        rate = perf.get("rate_mbps", 0) / 400.0
        outage = perf.get("outage", 0)
        fairness = perf.get("fairness", 0)
        return 0.6 * rate + 0.2 * max(0, 1.0 - outage) + 0.2 * fairness

    def get_best_per_regime(self, regime: str, top_k: int = 3) -> list[ExperimentRecord]:
        recs = [r for r in self.records if r.regime == regime]
        recs.sort(key=lambda r: self._composite_score(r.performance), reverse=True)
        return recs[:top_k]

    def get_worst_per_regime(self, regime: str, top_k: int = 3) -> list[ExperimentRecord]:
        recs = [r for r in self.records if r.regime == regime]
        recs.sort(key=lambda r: self._composite_score(r.performance))
        return recs[:top_k]

    def get_round_summary(self, round_id: int) -> dict:
        """Returns {regime: {rate_mbps, outage, fairness}} for a round."""
        summary = {}
        for r in self.records:
            if r.round_id == round_id:
                summary[r.regime] = r.performance
        return summary

    def get_regime_trajectory(self, regime: str) -> list[dict]:
        """Returns [{round_id, rate_mbps, outage, fairness}, ...] sorted by round."""
        entries = []
        for r in self.records:
            if r.regime == regime:
                entries.append({"round_id": r.round_id, **r.performance})
        entries.sort(key=lambda x: x["round_id"])
        return entries

    def get_baseline_gap(self, regime: str) -> dict:
        """Returns gap vs baseline MLP for a regime."""
        best = self.get_best_per_regime(regime, top_k=1)
        if not best or regime not in self.baseline_metrics:
            return {}
        bp = best[0].performance
        bl = self.baseline_metrics[regime]
        bl_rate = bl.get("rate_mbps", 1.0)
        return {
            "current_best_rate": bp.get("rate_mbps", 0),
            "baseline_rate": bl_rate,
            "rate_gap_pct": round((bl_rate - bp.get("rate_mbps", 0)) / max(bl_rate, 1) * 100, 1),
            "outage_gap": round(bp.get("outage", 0) - bl.get("outage", 0), 3),
            "fairness_gap": round(bp.get("fairness", 0) - bl.get("fairness", 0), 3),
        }

    def get_latest_round(self) -> int:
        if not self.records:
            return -1
        return max(r.round_id for r in self.records)

    def get_best_weights_for_regime(self, regime: str) -> Optional[dict]:
        """Convenience: return the single best weight dict for a regime."""
        best = self.get_best_per_regime(regime, top_k=1)
        return best[0].weights_used if best else None
