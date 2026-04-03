#!/usr/bin/env python3
"""Idea 2: Performance-Grounded RAG Quick Validation.

Tests whether RAG-anchored LLM output has lower weight variance than
the baseline causal prompt (no anchors). Uses a single round of probing
+ LLM weight generation to compare outputs statistically.

Experiment design:
  1. Build anchor DB from available probe_round*.json files
  2. Run single-variable probe on novel regimes (50k steps, same as v2)
  3. Call LLM N=10 times with RAG-augmented prompt vs baseline causal prompt
  4. Compare: weight CV, deviation from best anchor, resulting weights

Quick mode (default): LLM comparison only — no 500k training run.
Full mode (--full): runs 500k evolution round + compares to oracle_mlp.

Usage:
    cd workspace/code
    python scripts/run_idea2_rag.py                # quick comparison
    python scripts/run_idea2_rag.py --full         # full 500k validation
    python scripts/run_idea2_rag.py --build-db-only  # just build anchor DB
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.rag_anchor_db import AnchorDB, build_global_anchor_db, _get_probe_optimal_weights
from llm.causal_prompt import (
    build_causal_prompt, build_rag_causal_prompt, get_anchor_base_weights,
    llm_causal_evolve, update_experience_context, build_known_regime_weights,
)
from llm.perturbation_probe import probe_all_novel_regimes
from llm.oracle_data_generator import NOVEL_KPI_PROFILES, NOVEL_EXPERT_HINTS, _sample_kpi
from llm.mlp_architect import KPI_KEYS, WEIGHT_KEYS
from llm.evolution_db import EvolutionDB, WEIGHT_KEYS

KNOWN_REGIMES = ["urban", "maritime", "disaster", "mixed"]
NOVEL_REGIMES = ["iot_burst", "polar_handover", "hot_cold"]
RESULTS_DIR = Path(__file__).parent.parent / "results"

LLM_MODEL = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2"
LLM_BASE_URL = "http://localhost:1234/v1"


# ---------------------------------------------------------------------------
# Anchor DB
# ---------------------------------------------------------------------------

def build_anchor_db(results_dir: Path, verbose: int = 1) -> AnchorDB:
    """Build anchor DB from all available probe files."""
    db_path = results_dir / "anchor_db.json"

    if db_path.exists():
        db = AnchorDB()
        db.load(str(db_path))
        if verbose:
            print(f"  [AnchorDB] Loaded {len(db)} entries from {db_path}")
        return db

    if verbose:
        print("  [AnchorDB] Building from probe files...")

    db = build_global_anchor_db(
        results_dir=str(results_dir),
        save_path=str(db_path),
        verbose=verbose,
    )
    return db


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------

def get_regime_query_kpi(regime: str, probe_result: dict = None) -> np.ndarray:
    """Get KPI vector for retrieval query.

    Uses probe base_performance if available; otherwise falls back to
    regime profile mean values.
    """
    if probe_result is not None:
        bp = probe_result.get("base_performance", {})
        if bp:
            from llm.rag_anchor_db import _kpi_from_performance
            return _kpi_from_performance(bp, regime)

    # Fallback: use novel KPI profile means
    profile = NOVEL_KPI_PROFILES.get(regime, {})
    kpi = np.array([
        np.mean(profile.get("avg_queue", [0.5])),
        np.mean(profile.get("sum_rate", [50.0])) / 100.0,
        np.mean(profile.get("outage_count", [0.0])),
        np.mean(profile.get("switch_count", [5.0])),
        np.mean(profile.get("fairness_index", [0.5])),
    ], dtype=float)
    return kpi


# ---------------------------------------------------------------------------
# LLM Output Variance Comparison
# ---------------------------------------------------------------------------

def compare_rag_vs_baseline(
    regime: str,
    probe_result: dict,
    anchor_db: AnchorDB,
    n_calls: int = 10,
    api_key: str = None,
    verbose: int = 1,
) -> dict:
    """Call LLM n_calls times with both prompts, measure output variance.

    Returns dict with per-weight CV and deviation stats for both modes.
    """
    query_kpi = get_regime_query_kpi(regime, probe_result)
    base_weights = probe_result["base_weights"]

    # Get anchor base for RAG clamping
    anchor_base = get_anchor_base_weights(anchor_db, query_kpi, regime)
    if anchor_base is None:
        anchor_base = base_weights
        if verbose:
            print(f"  [RAG] No anchor for {regime}, using probe base weights")

    baseline_prompt = build_causal_prompt(
        regime=regime, probe_result=probe_result, round_id=1,
    )
    rag_prompt = build_rag_causal_prompt(
        regime=regime, probe_result=probe_result,
        anchor_db=anchor_db, query_kpi=query_kpi, round_id=1,
    )

    if verbose:
        print(f"\n  === {regime}: baseline prompt ({len(baseline_prompt)} chars) ===")
        print(f"  === {regime}: RAG prompt ({len(rag_prompt)} chars) ===")
        # Show anchor section
        lines = rag_prompt.split("\n")
        for line in lines[:30]:
            print(f"    {line}")
        print("    ...")

    baseline_outputs = []
    rag_outputs = []

    for i in range(n_calls):
        if verbose:
            print(f"\n  --- Call {i+1}/{n_calls} ---")

        w_base = llm_causal_evolve(
            prompt=baseline_prompt, regime=regime,
            base_weights=base_weights,
            model=LLM_MODEL, api_key=api_key, base_url=LLM_BASE_URL,
            max_change=0.3, verbose=verbose, max_retries=2,
        )
        if w_base:
            baseline_outputs.append(w_base)

        w_rag = llm_causal_evolve(
            prompt=rag_prompt, regime=regime,
            base_weights=anchor_base,
            model=LLM_MODEL, api_key=api_key, base_url=LLM_BASE_URL,
            max_change=0.3, verbose=verbose, max_retries=2,
        )
        if w_rag:
            rag_outputs.append(w_rag)

        time.sleep(1)  # throttle local LM Studio

    return _compute_variance_stats(regime, baseline_outputs, rag_outputs, anchor_base, verbose)


def _compute_variance_stats(
    regime: str,
    baseline_outputs: list,
    rag_outputs: list,
    anchor_base: dict,
    verbose: int = 1,
) -> dict:
    """Compute CV and deviation stats for both prompt modes."""

    def _stats(outputs: list) -> dict:
        if not outputs:
            return {"n": 0}
        arr = np.array([[o[k] for k in WEIGHT_KEYS] for o in outputs])
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        cvs = stds / (means + 1e-9)
        return {
            "n": len(outputs),
            "mean": dict(zip(WEIGHT_KEYS, means.tolist())),
            "std": dict(zip(WEIGHT_KEYS, stds.tolist())),
            "cv": dict(zip(WEIGHT_KEYS, cvs.tolist())),
            "mean_cv": float(cvs.mean()),
            "switching_cv": float(cvs[WEIGHT_KEYS.index("switching")]),
        }

    def _anchor_deviation(outputs: list, anchor: dict) -> float:
        if not outputs:
            return float("nan")
        devs = []
        for o in outputs:
            dev = np.mean([abs(o[k] - anchor.get(k, o[k])) / max(anchor.get(k, 1e-3), 1e-3)
                           for k in WEIGHT_KEYS])
            devs.append(dev)
        return float(np.mean(devs))

    baseline_stats = _stats(baseline_outputs)
    rag_stats = _stats(rag_outputs)

    result = {
        "regime": regime,
        "baseline": baseline_stats,
        "rag": rag_stats,
        "baseline_anchor_deviation": _anchor_deviation(baseline_outputs, anchor_base),
        "rag_anchor_deviation": _anchor_deviation(rag_outputs, anchor_base),
    }

    if verbose:
        print(f"\n  === VARIANCE COMPARISON: {regime} ===")
        print(f"  Baseline  n={baseline_stats.get('n', 0)} "
              f"mean_CV={baseline_stats.get('mean_cv', float('nan')):.3f} "
              f"sw_CV={baseline_stats.get('switching_cv', float('nan')):.3f}")
        print(f"  RAG       n={rag_stats.get('n', 0)} "
              f"mean_CV={rag_stats.get('mean_cv', float('nan')):.3f} "
              f"sw_CV={rag_stats.get('switching_cv', float('nan')):.3f}")
        print(f"  Anchor deviation: baseline={result['baseline_anchor_deviation']:.3f} "
              f"rag={result['rag_anchor_deviation']:.3f}")

        if baseline_stats.get("n", 0) > 0 and rag_stats.get("n", 0) > 0:
            cv_reduction = (baseline_stats["mean_cv"] - rag_stats["mean_cv"]) / max(baseline_stats["mean_cv"], 1e-9)
            print(f"  CV reduction: {cv_reduction:+.1%}  ({'PASS' if cv_reduction > 0.3 else 'FAIL — <30% improvement'})")

    return result


# ---------------------------------------------------------------------------
# Full evolution round with RAG
# ---------------------------------------------------------------------------

def run_rag_evolution_round(
    seed: int = 42,
    probe_steps: int = 50_000,
    novel_steps: int = 250_000,
    known_steps: int = 250_000,
    api_key: str = None,
    verbose: int = 1,
) -> dict:
    """Run one complete RAG-augmented evolution round (PROBE+LEARN+APPLY)."""
    from llm.oracle_data_generator import generate_oracle_data_evolved
    from llm.quality_filter import filter_oracle_data
    from llm.mlp_architect import incremental_train_mlp
    from scripts.run_oracle_mlp import run_oracle_mlp_experiment

    anchor_db = build_anchor_db(RESULTS_DIR, verbose=verbose)

    print("\n--- Phase 1: PROBE ---")
    oracle_mlp_path = str(RESULTS_DIR / f"ORACLE_MLP_seed{seed}" / "mlp_model.pt")
    probe_results = probe_all_novel_regimes(
        base_mlp_dir=str(RESULTS_DIR / f"ORACLE_MLP_seed{seed}"),
        probe_steps=probe_steps,
        bidirectional=True,
        verbose=verbose,
    )

    # Save probe results
    probe_path = RESULTS_DIR / f"probe_rag_seed{seed}.json"
    with open(probe_path, "w") as f:
        # Probe results may have non-serializable objects — convert
        safe = {}
        for regime, pr in probe_results.items():
            safe[regime] = {
                "base_weights": pr["base_weights"],
                "base_performance": pr["base_performance"],
                "probes": pr["probes"],
            }
        json.dump(safe, f, indent=2)
    if verbose:
        print(f"  Probe results saved to {probe_path}")

    # Rebuild anchor DB with new probe round
    anchor_db.load_from_probe_files([str(probe_path)], verbose=verbose)
    anchor_db.save(str(RESULTS_DIR / "anchor_db.json"))

    print("\n--- Phase 2: LEARN (RAG-augmented) ---")
    evolved_weights = {}
    for regime in NOVEL_REGIMES:
        pr = probe_results.get(regime)
        if pr is None:
            evolved_weights[regime] = NOVEL_EXPERT_HINTS[regime]
            continue

        query_kpi = get_regime_query_kpi(regime, pr)
        anchor_base = get_anchor_base_weights(anchor_db, query_kpi, regime)
        base_weights = anchor_base or pr["base_weights"]

        prompt = build_rag_causal_prompt(
            regime=regime, probe_result=pr,
            anchor_db=anchor_db, query_kpi=query_kpi,
            round_id=1, verbose=1,
        )

        weights = llm_causal_evolve(
            prompt=prompt, regime=regime,
            base_weights=base_weights,
            model=LLM_MODEL, api_key=api_key, base_url=LLM_BASE_URL,
            max_change=0.3, verbose=verbose,
        )
        if weights is None:
            from llm.causal_prompt import synthetic_causal_evolve
            weights = synthetic_causal_evolve(pr, base_weights, verbose=verbose)
        evolved_weights[regime] = weights

    print("\n--- Phase 3: APPLY ---")
    known_weights = build_known_regime_weights()
    all_regime_weights = {}
    all_regime_weights.update(known_weights)
    all_regime_weights.update(evolved_weights)

    evolved_spec = [
        {"regime": r, "weights": all_regime_weights[r]}
        for r in KNOWN_REGIMES + NOVEL_REGIMES
    ]

    evolved_data = generate_oracle_data_evolved(
        evolved_regime_weights=evolved_spec,
        known_steps=known_steps // 500,
        novel_steps=novel_steps // 500,
        seed=seed,
        verbose=verbose,
    )
    filtered = filter_oracle_data(evolved_data, verbose=verbose)
    save_dir = str(RESULTS_DIR / f"RAG_EVOLVED_seed{seed}")
    model = incremental_train_mlp(
        new_data=filtered,
        base_model_path=str(RESULTS_DIR / f"ORACLE_MLP_seed{seed}" / "mlp_model.pt"),
        save_dir=save_dir,
        verbose=verbose,
    )

    metrics = run_oracle_mlp_experiment(
        oracle_mlp_dir=save_dir,
        known_steps=known_steps,
        novel_steps=novel_steps,
        seed=seed,
        verbose=verbose,
    )

    print(f"\n=== RAG EVOLUTION RESULT ===")
    print(f"  Known rate:  {metrics.get('known_rate', '?'):.1f} Mbps")
    print(f"  Novel rate:  {metrics.get('novel_rate', '?'):.1f} Mbps")
    print(f"  Novel outage:{metrics.get('novel_outage', '?'):.2f}")

    return {
        "evolved_weights": evolved_weights,
        "metrics": metrics,
        "anchor_db_size": len(anchor_db),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Idea 2 RAG Quick Validation")
    parser.add_argument("--full", action="store_true", help="Run full 500k evolution round")
    parser.add_argument("--build-db-only", action="store_true", help="Build anchor DB and exit")
    parser.add_argument("--n-calls", type=int, default=5, help="LLM calls per regime for variance test")
    parser.add_argument("--regime", default=None, help="Single regime to test (default: all novel)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--probe-steps", type=int, default=50_000)
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip probe, use existing probe_round*.json for LLM comparison only")
    args = parser.parse_args()

    print("=== Idea 2: Performance-Grounded RAG Validation ===\n")

    # Build anchor DB
    anchor_db = build_anchor_db(RESULTS_DIR, verbose=1)
    print(f"  Anchor DB: {len(anchor_db)} entries\n")

    if args.build_db_only:
        print("Done (--build-db-only). Anchor DB saved to results/anchor_db.json")
        return

    if args.full:
        run_rag_evolution_round(
            seed=args.seed,
            probe_steps=args.probe_steps,
            api_key=args.api_key,
        )
        return

    # Quick mode: LLM variance comparison only
    # Need probe results — either run probes or load from existing files
    if args.skip_probe:
        # Load latest probe round from disk
        probe_files = sorted(RESULTS_DIR.glob("probe_round*.json"))
        if not probe_files:
            print("ERROR: No probe_round*.json found. Run without --skip-probe first.")
            sys.exit(1)

        latest = probe_files[-1]
        print(f"  Loading probe results from {latest}")
        with open(latest) as f:
            probe_results = json.load(f)
        # Normalize structure
        for regime in probe_results:
            pr = probe_results[regime]
            pr["regime"] = regime
    else:
        # Run probes for the target regime(s)
        print("--- Running probes for novel regimes ---")
        regimes_to_probe = [args.regime] if args.regime else NOVEL_REGIMES

        # Probe using oracle MLP
        from llm.perturbation_probe import probe_single_regime
        probe_results = {}
        oracle_dir = str(RESULTS_DIR / f"ORACLE_MLP_seed{args.seed}")
        for regime in regimes_to_probe:
            base_weights = _get_probe_optimal_weights(regime)
            probe_results[regime] = probe_single_regime(
                base_mlp_dir=oracle_dir,
                regime=regime,
                base_weights=base_weights,
                delta=0.2,
                probe_steps=args.probe_steps,
                verbose=1,
            )

    # LLM variance comparison
    regimes_to_test = [args.regime] if args.regime else NOVEL_REGIMES
    all_results = []
    for regime in regimes_to_test:
        pr = probe_results.get(regime)
        if pr is None:
            print(f"  WARNING: No probe result for {regime}, skipping")
            continue
        result = compare_rag_vs_baseline(
            regime=regime,
            probe_result=pr,
            anchor_db=anchor_db,
            n_calls=args.n_calls,
            api_key=args.api_key,
            verbose=1,
        )
        all_results.append(result)

    # Summary
    print("\n\n=== SUMMARY ===")
    print(f"{'Regime':<20} {'Baseline CV':>12} {'RAG CV':>10} {'Reduction':>12} {'Pass?':>8}")
    print("-" * 65)
    for r in all_results:
        b_cv = r["baseline"].get("mean_cv", float("nan"))
        rag_cv = r["rag"].get("mean_cv", float("nan"))
        reduction = (b_cv - rag_cv) / max(b_cv, 1e-9) if not (np.isnan(b_cv) or np.isnan(rag_cv)) else float("nan")
        passed = "PASS" if reduction > 0.3 else ("FAIL" if not np.isnan(reduction) else "N/A")
        print(f"  {r['regime']:<18} {b_cv:>12.3f} {rag_cv:>10.3f} {reduction:>11.1%} {passed:>8}")

    # Save results
    out_path = RESULTS_DIR / f"rag_validation_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
