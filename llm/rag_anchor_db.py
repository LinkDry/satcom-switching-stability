"""Performance-Grounded RAG Anchor Database.

Stores historical (KPI_snapshot, reward_weights, RL_performance) tuples
and retrieves the most relevant high-performing examples to ground LLM output.

Design:
- No vector DB needed at our scale (<5000 entries, 5D vectors)
- sklearn KD-tree for exact KNN on KPI vectors
- RBF kernel similarity × normalized performance = outcome-weighted scoring
  (follows 2025 ICLR/NeurIPS literature on reward-driven retrieval)
- Optional: regime-conditional pre-filtering before KNN

Usage:
    db = AnchorDB()
    db.load_from_probe_files(["results/probe_round1.json", ...])
    anchors = db.retrieve(query_kpi, regime="polar_handover", top_k=5)
    # anchors: list of AnchorEntry sorted by outcome-weighted score
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import KDTree

# KPI vector key order (must match extract_kpi_vector ordering)
KPI_KEYS = ["avg_queue", "sum_rate", "outage_count", "switch_count", "fairness_index"]

# Weight vector key order
WEIGHT_KEYS = ["sum_rate", "fairness", "outage", "switching", "queue"]

# RBF bandwidth (tuned for typical KPI scale in our satellite env)
DEFAULT_RBF_SIGMA = 0.5


@dataclass
class AnchorEntry:
    """One historical (KPI, weights, performance) data point."""
    regime: str
    kpi_vector: np.ndarray        # shape (5,) — raw KPI values
    weights: dict                 # {"sum_rate": ..., "fairness": ..., ...}
    perf_mbps: float              # actual RL novel rate Mbps
    source: str = ""              # e.g., "probe_round1", "evolution_v2_r2"
    notes: str = ""               # optional annotation

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "kpi_vector": self.kpi_vector.tolist(),
            "weights": self.weights,
            "perf_mbps": self.perf_mbps,
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnchorEntry":
        return cls(
            regime=d["regime"],
            kpi_vector=np.array(d["kpi_vector"], dtype=float),
            weights=d["weights"],
            perf_mbps=float(d["perf_mbps"]),
            source=d.get("source", ""),
            notes=d.get("notes", ""),
        )


class AnchorDB:
    """Outcome-weighted KNN retrieval database for LLM weight anchoring."""

    def __init__(self, rbf_sigma: float = DEFAULT_RBF_SIGMA):
        self.entries: list[AnchorEntry] = []
        self.rbf_sigma = rbf_sigma
        self._kpi_matrix: Optional[np.ndarray] = None  # (N, 5) cache
        self._perf_array: Optional[np.ndarray] = None   # (N,) cache
        self._kdtree: Optional[KDTree] = None           # full-set KD-tree cache

    # ------------------------------------------------------------------
    # Loading / Saving
    # ------------------------------------------------------------------

    def add(self, entry: AnchorEntry):
        self.entries.append(entry)
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._kpi_matrix = None
        self._perf_array = None
        self._kdtree = None

    def _build_cache(self):
        if self._kpi_matrix is None and self.entries:
            self._kpi_matrix = np.stack([e.kpi_vector for e in self.entries])
            self._perf_array = np.array([e.perf_mbps for e in self.entries])
            self._kdtree = KDTree(self._kpi_matrix)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.entries = [AnchorEntry.from_dict(d) for d in data]
        self._invalidate_cache()
        return self

    # ------------------------------------------------------------------
    # Ingestion from experimental results
    # ------------------------------------------------------------------

    def load_from_probe_files(
        self,
        probe_paths: list[str],
        min_perf_mbps: float = 5.0,
        verbose: int = 1,
    ) -> int:
        """Extract anchor entries from probe_round*.json files.

        Each probe entry gives us: regime, weights_used, actual_perf_mbps.
        Both the base experiment AND each perturbation experiment are added
        (perturbations that worked well are valuable anchors too).
        """
        added = 0
        for path_str in probe_paths:
            p = Path(path_str)
            if not p.exists():
                if verbose:
                    print(f"  [AnchorDB] Skipping {path_str} (not found)")
                continue
            with open(p) as f:
                data = json.load(f)
            source = p.stem  # e.g., "probe_round1"

            for regime, pr in data.items():
                # Base experiment
                bw = pr["base_weights"]
                bp = pr["base_performance"]
                if bp["rate_mbps"] >= min_perf_mbps:
                    kpi = _kpi_from_performance(bp, regime)
                    self.add(AnchorEntry(
                        regime=regime,
                        kpi_vector=kpi,
                        weights=dict(bw),
                        perf_mbps=bp["rate_mbps"],
                        source=source,
                        notes="base",
                    ))
                    added += 1

                # Each perturbation experiment
                for probe in pr.get("probes", []):
                    perf = probe.get("performance", {})
                    if perf.get("rate_mbps", 0) < min_perf_mbps:
                        continue
                    # Reconstruct the perturbed weights
                    perturbed_w = dict(bw)
                    perturbed_w[probe["weight"]] = probe["new_value"]
                    kpi = _kpi_from_performance(perf, regime)
                    self.add(AnchorEntry(
                        regime=regime,
                        kpi_vector=kpi,
                        weights=perturbed_w,
                        perf_mbps=perf["rate_mbps"],
                        source=source,
                        notes=f"{probe['weight']} {probe['direction']}",
                    ))
                    added += 1

        self._invalidate_cache()
        if verbose:
            print(f"  [AnchorDB] Loaded {added} entries from {len(probe_paths)} probe files")
            self._print_stats()
        return added

    def load_from_evolution_results(
        self,
        results_dir: str,
        pattern: str = "ORACLE_MLP_seed*/metrics.json",
        min_perf_mbps: float = 50.0,
        verbose: int = 1,
    ) -> int:
        """Add entries from evolution/oracle_mlp result metrics.json files.

        These are aggregate-level entries (one per run, not per regime),
        so they're less precise than probe entries but still useful anchors.
        """
        import glob
        paths = glob.glob(str(Path(results_dir) / pattern))
        added = 0
        for path_str in sorted(paths):
            try:
                with open(path_str) as f:
                    m = json.load(f)
                source = Path(path_str).parent.name
                novel_rate = m.get("novel_rate", 0)
                if novel_rate < min_perf_mbps:
                    continue
                # For aggregate results, use best-guess weights from method info
                # We only have aggregate KPI here — approximate with zeros
                kpi = np.zeros(5, dtype=float)
                # Use known expert weights as proxy
                for regime in ["iot_burst", "polar_handover", "hot_cold"]:
                    self.add(AnchorEntry(
                        regime=regime,
                        kpi_vector=kpi,
                        weights=_get_probe_optimal_weights(regime),
                        perf_mbps=novel_rate,
                        source=source,
                        notes="aggregate_novel",
                    ))
                    added += 1
            except Exception:
                pass

        self._invalidate_cache()
        if verbose:
            print(f"  [AnchorDB] Added {added} aggregate entries from evolution results")
        return added

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query_kpi: np.ndarray,
        regime: Optional[str] = None,
        top_k: int = 5,
        min_perf_percentile: float = 40.0,
        verbose: int = 0,
    ) -> list[AnchorEntry]:
        """Retrieve top-K anchors using outcome-weighted KNN.

        Scoring: score = RBF_similarity(kpi) × normalized_performance
        This retrieves entries that are both KPI-similar AND high-performing.

        Args:
            query_kpi: (5,) KPI vector for current state
            regime: if given, filter to same-regime entries first
            top_k: number of anchors to return
            min_perf_percentile: pre-filter to entries above this performance percentile
            verbose: print retrieval info

        Returns:
            List of AnchorEntry sorted by score (highest first)
        """
        if not self.entries:
            return []

        self._build_cache()

        # Step 1: regime filter
        if regime is not None:
            regime_mask = np.array([e.regime == regime for e in self.entries])
            if regime_mask.sum() < top_k:
                # Fall back to all entries if not enough regime-specific ones
                regime_mask = np.ones(len(self.entries), dtype=bool)
        else:
            regime_mask = np.ones(len(self.entries), dtype=bool)

        filtered_entries = [e for e, m in zip(self.entries, regime_mask) if m]
        if not filtered_entries:
            return []

        filtered_kpi = self._kpi_matrix[regime_mask]
        filtered_perf = self._perf_array[regime_mask]

        # Step 2: performance threshold pre-filter
        perf_threshold = np.percentile(filtered_perf, min_perf_percentile)
        perf_mask = filtered_perf >= perf_threshold
        if perf_mask.sum() < 1:
            perf_mask = np.ones(len(filtered_entries), dtype=bool)

        pool_entries = [e for e, m in zip(filtered_entries, perf_mask) if m]
        pool_kpi = filtered_kpi[perf_mask]
        pool_perf = filtered_perf[perf_mask]

        # Step 3: KD-tree candidate retrieval → RBF similarity × normalized performance
        # Build a KD-tree on the filtered pool (typically <200 entries — fast).
        # Fetch min(top_k*3, pool_size) nearest candidates by Euclidean distance,
        # then rerank by outcome-weighted score to get top_k final results.
        query = np.asarray(query_kpi, dtype=float)
        pool_size = len(pool_entries)
        k_cands = min(top_k * 3, pool_size)

        if pool_size >= 2:
            pool_kdtree = KDTree(pool_kpi)
            dists, raw_idx = pool_kdtree.query(query.reshape(1, -1), k=k_cands)
            cand_idx = raw_idx[0]
            sq_dists = dists[0] ** 2
        else:
            # Degenerate pool — score all directly
            cand_idx = np.arange(pool_size)
            sq_dists = np.sum((pool_kpi - query) ** 2, axis=1)

        rbf_sim = np.exp(-sq_dists / (2.0 * self.rbf_sigma ** 2))
        cand_perf = pool_perf[cand_idx]
        perf_max = cand_perf.max()
        norm_perf = cand_perf / perf_max if perf_max > 0 else np.ones_like(cand_perf)
        scores = rbf_sim * norm_perf

        # Step 4: top-K from candidates
        k = min(top_k, len(cand_idx))
        top_local = np.argsort(scores)[::-1][:k]
        top_idx = cand_idx[top_local]

        results = [pool_entries[i] for i in top_idx]

        if verbose:
            print(f"  [AnchorDB] Retrieved {len(results)} anchors for {regime or 'any'}:")
            for i, e in enumerate(results):
                print(f"    [{i+1}] {e.regime} {e.source}/{e.notes}: "
                      f"perf={e.perf_mbps:.1f}Mbps "
                      f"sw={e.weights.get('switching',0):.3f}")

        return results

    def _print_stats(self):
        if not self.entries:
            print("  [AnchorDB] Empty database")
            return
        perfs = [e.perf_mbps for e in self.entries]
        regimes = {}
        for e in self.entries:
            regimes[e.regime] = regimes.get(e.regime, 0) + 1
        print(f"  [AnchorDB] {len(self.entries)} entries | "
              f"perf: {min(perfs):.1f}-{max(perfs):.1f} Mbps (mean {sum(perfs)/len(perfs):.1f}) | "
              f"regimes: {regimes}")

    def __len__(self):
        return len(self.entries)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _kpi_from_performance(perf: dict, regime: str) -> np.ndarray:
    """Build an approximate 5D KPI vector from a performance dict.

    probe_performance only has rate_mbps, outage, fairness — we approximate
    the rest. This is good enough for retrieval (similarity is relative).
    """
    # Rough KPI mapping: [avg_queue, sum_rate, outage_count, switch_count, fairness_index]
    return np.array([
        0.0,                           # avg_queue: not in probe perf
        perf.get("rate_mbps", 0) / 100.0,  # normalize sum_rate
        perf.get("outage", 0),
        0.0,                           # switch_count: not in probe perf
        perf.get("fairness", 0),
    ], dtype=float)


def _get_probe_optimal_weights(regime: str) -> dict:
    """Probe-derived optimal weights (from probe_round1 analysis)."""
    defaults = {
        "iot_burst":       {"sum_rate": 0.6, "fairness": 0.6, "outage": 1.5, "switching": 0.1,  "queue": 0.8},
        "polar_handover":  {"sum_rate": 0.8, "fairness": 0.3, "outage": 1.2, "switching": 0.96, "queue": 0.24},
        "hot_cold":        {"sum_rate": 1.2, "fairness": 0.6, "outage": 1.2, "switching": 0.06, "queue": 0.36},
    }
    return defaults.get(regime, {"sum_rate": 1.0, "fairness": 0.5, "outage": 1.0, "switching": 0.05, "queue": 0.3})


def format_anchors_for_prompt(anchors: list[AnchorEntry]) -> str:
    """Format retrieved anchors into an LLM-readable string."""
    if not anchors:
        return "(no historical anchors available)"

    lines = ["RETRIEVED HISTORICAL ANCHORS (verified high-performance examples):"]
    for i, a in enumerate(anchors):
        w = a.weights
        lines.append(
            f"  Anchor {i+1} [{a.regime}, {a.perf_mbps:.1f} Mbps, source={a.source}]:"
            f"\n    weights: sr={w.get('sum_rate',0):.3f} fair={w.get('fairness',0):.3f} "
            f"out={w.get('outage',0):.3f} sw={w.get('switching',0):.3f} q={w.get('queue',0):.3f}"
        )
    lines.append("")
    lines.append("Use these anchors as starting points — stay within ±30% of the best anchor's weights.")
    return "\n".join(lines)


def build_global_anchor_db(
    results_dir: str = "results",
    save_path: str = "results/anchor_db.json",
    verbose: int = 1,
) -> AnchorDB:
    """Build and save the anchor DB from all available experiment data."""
    db = AnchorDB()

    # Primary source: probe files (most precise — per-regime, per-weight data)
    probe_files = sorted(Path(results_dir).glob("probe_round*.json"))
    if probe_files:
        db.load_from_probe_files(
            [str(p) for p in probe_files],
            min_perf_mbps=5.0,
            verbose=verbose,
        )

    # Secondary: high-performing evolution runs
    db.load_from_evolution_results(
        results_dir,
        pattern="EVOLVED_v2_round*/metrics.json",
        min_perf_mbps=100.0,
        verbose=verbose,
    )

    db.save(save_path)
    if verbose:
        print(f"  [AnchorDB] Saved to {save_path}")
    return db
