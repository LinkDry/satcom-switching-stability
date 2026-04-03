#!/usr/bin/env python3
"""Analyze and compare RAG vs baseline evolution results.

Reads metrics.json files from:
  - RAG_EVOLVED_v2_round*_seed*/   (RAG-augmented)
  - EVOLVED_v2_round*_seed*/       (baseline, no RAG)
  - ORACLE_MLP_seed*/              (oracle MLP reference)
  - GEN_mlp_seed*/                 (static weights, upper bound)

Outputs:
  - Per-method mean±std novel rate, known rate, outage
  - Paired t-test: RAG vs oracle_mlp, RAG vs no-RAG
  - Improvement table suitable for paper
  - Per-round trajectory for RAG (shows convergence)

Usage:
    cd workspace/code
    python scripts/analyze_rag_results.py
    python scripts/analyze_rag_results.py --latex    # LaTeX table output
    python scripts/analyze_rag_results.py --rounds   # per-round breakdown
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
WEIGHT_KEYS = ["sum_rate", "fairness", "outage", "switching", "queue"]


def load_metrics(pattern: str) -> list[dict]:
    """Load all metrics.json files matching glob pattern."""
    paths = sorted(RESULTS_DIR.glob(pattern))
    results = []
    for p in paths:
        try:
            with open(p) as f:
                m = json.load(f)
            m["_source"] = p.parent.name
            results.append(m)
        except Exception as e:
            print(f"  WARNING: Could not load {p}: {e}")
    return results


def summarize(records: list[dict], key: str = "novel_rate") -> dict:
    """Compute mean, std, median, min, max for a metric key."""
    vals = [r[key] for r in records if key in r]
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.array(vals)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": arr.tolist(),
    }


def print_method_table(methods: dict, latex: bool = False):
    """Print comparison table for all methods."""
    if latex:
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("Method & Novel Rate (Mbps) & Novel Std & Known Rate & Outage \\\\")
        print("\\midrule")
        for name, records in methods.items():
            ns = summarize(records, "novel_rate")
            ks = summarize(records, "known_rate")
            os_ = summarize(records, "novel_outage")
            if ns["n"] == 0:
                continue
            print(f"{name} & {ns['mean']:.1f}$\\pm${ns['std']:.1f} & "
                  f"{ns['std']:.1f} & {ks['mean']:.1f} & {os_['mean']:.2f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    else:
        print(f"\n{'Method':<28} {'n':>4} {'Novel Mean':>12} {'Novel Std':>11} "
              f"{'Novel Median':>14} {'Known Mean':>12} {'Outage':>8}")
        print("-" * 95)
        for name, records in methods.items():
            ns = summarize(records, "novel_rate")
            ks = summarize(records, "known_rate")
            os_ = summarize(records, "novel_outage")
            if ns["n"] == 0:
                print(f"  {name:<26} {'N/A':>4}")
                continue
            print(f"  {name:<26} {ns['n']:>4} {ns['mean']:>12.1f} {ns['std']:>11.1f} "
                  f"{ns['median']:>14.1f} {ks['mean']:>12.1f} {os_['mean']:>8.3f}")


def run_ttest(a_records: list[dict], b_records: list[dict],
              name_a: str, name_b: str, key: str = "novel_rate"):
    """Paired or independent t-test between two method groups."""
    a_vals = np.array([r[key] for r in a_records if key in r])
    b_vals = np.array([r[key] for r in b_records if key in r])
    if len(a_vals) < 2 or len(b_vals) < 2:
        print(f"  {name_a} vs {name_b}: insufficient data for t-test")
        return

    if len(a_vals) == len(b_vals):
        t, p = stats.ttest_rel(a_vals, b_vals)
        test_type = "paired"
    else:
        t, p = stats.ttest_ind(a_vals, b_vals)
        test_type = "independent"

    diff = a_vals.mean() - b_vals.mean()
    print(f"  {name_a} vs {name_b}: "
          f"Δ={diff:+.1f} Mbps  t={t:.3f}  p={p:.3f}  ({test_type}) "
          f"{'*sig*' if p < 0.05 else 'n.s.'}")


def print_round_trajectory(rag_records: list[dict], norag_records: list[dict]):
    """Show per-round novel rate for RAG vs no-RAG."""
    def by_round(records):
        rounds = {}
        for r in records:
            src = r.get("_source", "")
            # Extract round number from directory name e.g. RAG_EVOLVED_v2_round2_seed42
            import re
            m = re.search(r"round(\d+)", src)
            if m:
                rd = int(m.group(1))
                rounds.setdefault(rd, []).append(r.get("novel_rate", 0))
        return rounds

    rag_r = by_round(rag_records)
    norag_r = by_round(norag_records)

    all_rounds = sorted(set(list(rag_r.keys()) + list(norag_r.keys())))
    if not all_rounds:
        return

    print(f"\n{'Round':>6} {'RAG (mean)':>12} {'RAG (std)':>11} {'No-RAG (mean)':>15} {'No-RAG (std)':>13}")
    print("-" * 62)
    for rd in all_rounds:
        rag_v = rag_r.get(rd, [])
        norag_v = norag_r.get(rd, [])
        rag_mean = np.mean(rag_v) if rag_v else float("nan")
        rag_std = np.std(rag_v) if rag_v else float("nan")
        norag_mean = np.mean(norag_v) if norag_v else float("nan")
        norag_std = np.std(norag_v) if norag_v else float("nan")
        print(f"  {rd:>4} {rag_mean:>12.1f} {rag_std:>11.1f} {norag_mean:>15.1f} {norag_std:>13.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG evolution results")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    parser.add_argument("--rounds", action="store_true", help="Show per-round trajectory")
    args = parser.parse_args()

    print("=== RAG Evolution Results Analysis ===\n")

    # Load all method groups
    rag_all = load_metrics("RAG_EVOLVED_v2_round*_seed*/metrics.json")
    # Use final round only for fair comparison (if multi-round)
    rag_final = [r for r in rag_all if "_round3_" in r["_source"] or
                 (not any(f"_round{i}_" in r["_source"] for i in range(1, 3)))]
    if not rag_final:
        rag_final = rag_all  # fallback: use all rounds

    norag_all = load_metrics("EVOLVED_v2_round*_seed*/metrics.json")
    norag_final = [r for r in norag_all if "_round3_" in r["_source"] or
                   (not any(f"_round{i}_" in r["_source"] for i in range(1, 3)))]
    if not norag_final:
        norag_final = norag_all

    oracle = load_metrics("ORACLE_MLP_seed*/metrics.json")
    gen_mlp = load_metrics("GEN_mlp_seed*/metrics.json")

    # Also load best single-round if round3 not available
    rag_best = rag_all  # all rounds for trajectory

    print(f"Data loaded:")
    print(f"  RAG (all rounds):    {len(rag_all)} records")
    print(f"  RAG (final round):   {len(rag_final)} records")
    print(f"  No-RAG (final round):{len(norag_final)} records")
    print(f"  Oracle MLP:          {len(oracle)} records")
    print(f"  GEN MLP (static):    {len(gen_mlp)} records")

    methods = {
        "GEN MLP (static weights)": gen_mlp,
        "Oracle MLP": oracle,
        "Evolution v2 no-RAG": norag_final,
        "Evolution v2 RAG (ours)": rag_final,
    }

    print_method_table(methods, latex=args.latex)

    # Statistical tests
    print("\n--- Statistical Tests (novel_rate) ---")
    if rag_final and oracle:
        run_ttest(rag_final, oracle, "RAG", "Oracle MLP")
    if rag_final and norag_final:
        run_ttest(rag_final, norag_final, "RAG", "No-RAG")
    if norag_final and oracle:
        run_ttest(norag_final, oracle, "No-RAG", "Oracle MLP")

    # Per-round trajectory
    if args.rounds and rag_all:
        print("\n--- Per-Round Trajectory ---")
        print_round_trajectory(rag_all, norag_all)

    # Anchor DB stats (if available)
    anchor_path = RESULTS_DIR / "anchor_db.json"
    if anchor_path.exists():
        with open(anchor_path) as f:
            entries = json.load(f)
        regime_counts = {}
        perf_vals = []
        for e in entries:
            regime_counts[e["regime"]] = regime_counts.get(e["regime"], 0) + 1
            perf_vals.append(e["perf_mbps"])
        print(f"\n--- Anchor DB Stats ({len(entries)} entries) ---")
        for regime, cnt in sorted(regime_counts.items()):
            print(f"  {regime:<20} {cnt} entries")
        print(f"  Performance range: {min(perf_vals):.1f} – {max(perf_vals):.1f} Mbps "
              f"(mean {np.mean(perf_vals):.1f})")


if __name__ == "__main__":
    main()
