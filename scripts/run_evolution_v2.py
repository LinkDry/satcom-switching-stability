#!/usr/bin/env python3
"""Self-Evolving LLM-MLP Closed-Loop v2: Perturbation-Based Causal Feedback.

Evolution loop:
  Round 0: Seed from existing oracle_mlp results
  Round 1..N:
    1. PROBE  — Single-variable perturbation on each novel regime (50k/probe)
    2. LEARN  — LLM sees causal sensitivity table, proposes improved weights
    3. APPLY  — Generate training data → MLP incremental train → 500k experiment
    4. ACCUMULATE — Record experience for next round's LLM context

Key improvements over v1:
  - Per-regime independent probing and feedback
  - Single-variable perturbation → causal attribution
  - Experience accumulation across rounds
  - Synthetic fallback uses probe gradients (not blind heuristics)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.evolution_db import EvolutionDB, ExperimentRecord, WEIGHT_KEYS
from llm.perturbation_probe import probe_all_novel_regimes
from llm.causal_prompt import (
    build_causal_prompt, build_rag_causal_prompt, get_anchor_base_weights,
    llm_causal_evolve, synthetic_causal_evolve,
    update_experience_context, build_known_regime_weights,
)
from llm.rag_anchor_db import AnchorDB, build_global_anchor_db, _kpi_from_performance
from llm.oracle_data_generator import (
    generate_oracle_data_evolved, NOVEL_KPI_PROFILES, NOVEL_EXPERT_HINTS,
    _sample_kpi,
)
from llm.mlp_architect import (
    MLPArchitect, incremental_train_mlp, KPI_KEYS, WEIGHT_KEYS,
    EXPERT_WEIGHTS, KPI_PROFILES,
)
from llm.quality_filter import filter_oracle_data
from scripts.run_oracle_mlp import run_oracle_mlp_experiment

KNOWN_REGIMES = ["urban", "maritime", "disaster", "mixed"]
NOVEL_REGIMES = ["iot_burst", "polar_handover", "hot_cold"]
ALL_REGIMES = KNOWN_REGIMES + NOVEL_REGIMES
RESULTS_DIR = Path(__file__).parent.parent / "results"

# LLM config — local LM Studio
LLM_MODEL = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2"
LLM_BASE_URL = "http://localhost:1234/v1"


def load_baseline_metrics(seed: int = 42, verbose: int = 1) -> dict:
    """Get baseline metrics from GEN_mlp results."""
    metrics_path = RESULTS_DIR / f"GEN_mlp_seed{seed}" / "metrics.json"
    if not metrics_path.exists():
        print(f"  WARNING: No baseline at {metrics_path}")
        return {}

    with open(metrics_path) as f:
        m = json.load(f)

    baseline = {}
    for r in KNOWN_REGIMES:
        baseline[r] = {
            "rate_mbps": m["known_rate"],
            "outage": m["known_outage"],
            "fairness": m.get("known_fairness", 0),
        }
    for r in NOVEL_REGIMES:
        baseline[r] = {
            "rate_mbps": m["novel_rate"],
            "outage": m["novel_outage"],
            "fairness": m.get("novel_fairness", 0),
        }

    if verbose:
        print(f"  Baseline MLP: known={m['known_rate']:.1f} novel={m['novel_rate']:.1f}")
    return baseline


def populate_round0(db: EvolutionDB, seed: int = 42, verbose: int = 1) -> bool:
    """Populate Round 0 from existing oracle_mlp experiment results."""
    metrics_path = RESULTS_DIR / f"ORACLE_MLP_seed{seed}" / "metrics.json"
    if not metrics_path.exists():
        print(f"  ERROR: No oracle_mlp results at {metrics_path}")
        return False

    with open(metrics_path) as f:
        m = json.load(f)

    if verbose:
        print(f"=== Populating Round 0 from oracle_mlp results ===")
        print(f"  known={m['known_rate']:.1f} novel={m['novel_rate']:.1f}")

    all_experts = {r: dict(zip(WEIGHT_KEYS, w)) for r, w in EXPERT_WEIGHTS.items()}
    all_experts.update(NOVEL_EXPERT_HINTS)

    ts = time.strftime("%Y-%m-%d %H:%M")

    for regime in KNOWN_REGIMES:
        kpi_snapshot = {k: KPI_PROFILES[regime][k][0] for k in KPI_KEYS}
        db.add_record(ExperimentRecord(
            round_id=0, regime=regime,
            kpi_snapshot=kpi_snapshot,
            weights_used=all_experts.get(regime, {}),
            performance={
                "rate_mbps": m["known_rate"],
                "outage": m["known_outage"],
                "fairness": m.get("known_fairness", 0),
            },
            source="oracle_mlp_aggregated", timestamp=ts,
        ))

    for regime in NOVEL_REGIMES:
        kpi_snapshot = {k: NOVEL_KPI_PROFILES[regime][k][0] for k in KPI_KEYS}
        db.add_record(ExperimentRecord(
            round_id=0, regime=regime,
            kpi_snapshot=kpi_snapshot,
            weights_used=all_experts.get(regime, {}),
            performance={
                "rate_mbps": m["novel_rate"],
                "outage": m["novel_outage"],
                "fairness": m.get("novel_fairness", 0),
            },
            source="oracle_mlp_aggregated", timestamp=ts,
        ))

    db.save()
    return True


def run_evolution_v2(
    max_rounds: int = 3,
    seed: int = 42,
    use_llm: bool = True,
    use_rag_anchors: bool = True,
    api_key: str = None,
    probe_steps: int = 50_000,
    probe_delta: float = 0.2,
    probe_bidirectional: bool = True,
    known_steps: int = 250_000,
    novel_steps: int = 250_000,
    convergence_threshold: float = 5.0,
    save_prefix: str = "EVOLVED_v2",
    verbose: int = 1,
):
    """Main v2 evolution loop: PROBE → LEARN → APPLY → ACCUMULATE."""

    db_path = RESULTS_DIR / "evolution_v2_db.json"

    # Step 0: Load baseline
    if verbose:
        print("\n--- Loading baseline metrics ---")
    baseline = load_baseline_metrics(seed=seed, verbose=verbose)
    db = EvolutionDB(str(db_path), baseline_metrics=baseline)

    if db_path.exists():
        db.load()
        if verbose:
            print(f"  Loaded existing DB with {len(db.records)} records")
    else:
        if not populate_round0(db, seed=seed, verbose=verbose):
            print("ERROR: Cannot populate Round 0. Run run_oracle_mlp.py first.")
            return
        db.save()

    start_round = db.get_latest_round() + 1
    prev_novel_rate = None
    experience_context = ""  # Accumulates across rounds

    # Load experience from DB if resuming
    experience_path = RESULTS_DIR / "evolution_v2_experience.json"
    if experience_path.exists():
        with open(experience_path) as f:
            saved = json.load(f)
            experience_context = saved.get("experience_context", "")
        if verbose:
            print(f"  Loaded experience context ({len(experience_context)} chars)")

    # Load or build anchor DB for RAG
    anchor_db_path = str(RESULTS_DIR / "anchor_db.json")
    anchor_db = AnchorDB()
    if use_rag_anchors:
        if Path(anchor_db_path).exists():
            anchor_db.load(anchor_db_path)
            if verbose:
                print(f"  AnchorDB loaded: {len(anchor_db)} entries")
        else:
            anchor_db = build_global_anchor_db(str(RESULTS_DIR), anchor_db_path, verbose=verbose)

    # Known regime weights (fixed, from expert)
    known_weights = build_known_regime_weights()

    for round_id in range(start_round, start_round + max_rounds):
        print(f"\n{'='*60}")
        print(f"EVOLUTION v2 ROUND {round_id}")
        print(f"{'='*60}")

        # ==============================
        # Phase 1: PROBE
        # ==============================
        print(f"\n--- Phase 1: PROBE (single-variable perturbation) ---")

        # Get current best weights for each novel regime
        novel_weights = {}
        for regime in NOVEL_REGIMES:
            best = db.get_best_weights_for_regime(regime)
            novel_weights[regime] = best or NOVEL_EXPERT_HINTS[regime]

        probe_results = probe_all_novel_regimes(
            novel_regimes=NOVEL_REGIMES,
            regime_weights=novel_weights,
            delta=probe_delta,
            probe_steps=probe_steps,
            bidirectional=probe_bidirectional,
            seed=seed + round_id * 1000,
            verbose=verbose,
        )

        # Save probe results
        probe_save_path = RESULTS_DIR / f"probe_round{round_id}.json"
        with open(probe_save_path, "w") as f:
            json.dump(probe_results, f, indent=2, default=str)
        if verbose:
            print(f"  Probe results saved to {probe_save_path}")

        # ==============================
        # Phase 2: LEARN (LLM causal reasoning)
        # ==============================
        print(f"\n--- Phase 2: LEARN (LLM causal reasoning) ---")

        evolved = []

        # Known regimes: use expert weights (no evolution needed)
        for regime in KNOWN_REGIMES:
            evolved.append({
                "regime": regime,
                "weights": known_weights[regime],
                "reasoning": "expert_weights_fixed",
            })

        # Novel regimes: LLM or synthetic evolution based on probes
        for regime in NOVEL_REGIMES:
            base_w = novel_weights[regime]

            if regime not in probe_results:
                print(f"  WARNING: No probe data for {regime}, using base weights")
                evolved.append({
                    "regime": regime,
                    "weights": base_w,
                    "reasoning": "no_probe_data",
                })
                continue

            pr = probe_results[regime]

            if use_llm:
                # Build query KPI from current probe baseline performance
                query_kpi = _kpi_from_performance(pr["base_performance"], regime)

                if use_rag_anchors and len(anchor_db) > 0:
                    prompt = build_rag_causal_prompt(
                        regime=regime,
                        probe_result=pr,
                        anchor_db=anchor_db,
                        query_kpi=query_kpi,
                        round_id=round_id,
                        target_rate=baseline.get(regime, {}).get("rate_mbps", 342.1),
                        experience_context=experience_context,
                    )
                    clamp_base = get_anchor_base_weights(anchor_db, query_kpi, regime) or base_w
                    if verbose:
                        print(f"  [RAG] {regime}: using anchor-grounded prompt, clamp_base=anchor")
                else:
                    prompt = build_causal_prompt(
                        regime=regime,
                        probe_result=pr,
                        round_id=round_id,
                        target_rate=baseline.get(regime, {}).get("rate_mbps", 342.1),
                        experience_context=experience_context,
                    )
                    clamp_base = base_w

                new_weights = llm_causal_evolve(
                    prompt=prompt,
                    regime=regime,
                    base_weights=clamp_base,
                    model=LLM_MODEL,
                    api_key=api_key,
                    base_url=LLM_BASE_URL,
                    max_change=0.3,
                    verbose=verbose,
                )

                if new_weights:
                    evolved.append({
                        "regime": regime,
                        "weights": new_weights,
                        "reasoning": f"llm_causal_round_{round_id}",
                    })
                    continue

            # Fallback: synthetic causal evolution
            if verbose:
                print(f"  Using synthetic causal evolution for {regime}")
            synth_w = synthetic_causal_evolve(pr, base_w, verbose=verbose)
            evolved.append({
                "regime": regime,
                "weights": synth_w,
                "reasoning": f"synthetic_causal_round_{round_id}",
            })

        if verbose:
            print(f"\n  Evolved weights summary:")
            for e in evolved:
                w = e["weights"]
                print(f"    {e['regime']:<20} sr={w.get('sum_rate',0):.3f} f={w.get('fairness',0):.3f} "
                      f"o={w.get('outage',0):.3f} sw={w.get('switching',0):.3f} q={w.get('queue',0):.3f}")

        # ==============================
        # Phase 3: APPLY (generate data + train MLP + experiment)
        # ==============================
        print(f"\n--- Phase 3: APPLY (MLP training + experiment) ---")

        # Step 3a: Generate training data
        prev_best = []
        for regime in ALL_REGIMES:
            best = db.get_best_per_regime(regime, top_k=1)
            if best:
                rec = best[0]
                prev_best.append({
                    "kpi": rec.kpi_snapshot,
                    "weights": rec.weights_used,
                    "regime": regime,
                    "source": "prev_best",
                    "quality_score": 1.5,
                })

        evolved_data = generate_oracle_data_evolved(
            evolved_weights=evolved,
            n_per_regime=200,
            noise_std=0.03,
            previous_best=prev_best,
            seed=seed + round_id * 100,
            output_path=str(RESULTS_DIR / f"evolution_v2_round{round_id}_data.json"),
            verbose=verbose,
        )

        # Step 3b: Quality filter
        filtered = filter_oracle_data(
            evolved_data, use_bounds=True,
            use_consistency=True, skip_consistency=True,
            verbose=verbose,
        )

        # Step 3c: MLP incremental training (always from oracle_mlp base)
        base_dir = "models/oracle_mlp"
        save_dir = f"models/evolved_v2_mlp_round{round_id}"
        model = incremental_train_mlp(
            evolved_samples=filtered,
            previous_best_samples=prev_best,
            base_model_dir=base_dir,
            save_dir=save_dir,
            epochs=150, lr=3e-4,
            verbose=verbose,
        )

        # Step 3d: Full feedback experiment
        print(f"\n  Running feedback experiment ({known_steps + novel_steps} steps)...")
        fb_save_dir = str(RESULTS_DIR / f"{save_prefix}_round{round_id}_seed{seed}")
        results = run_oracle_mlp_experiment(
            seed=seed,
            known_steps=known_steps,
            novel_steps=novel_steps,
            oracle_mlp_dir=save_dir,
            save_dir=fb_save_dir,
            verbose=verbose,
        )

        # ==============================
        # Phase 4: ACCUMULATE (record + experience)
        # ==============================
        print(f"\n--- Phase 4: ACCUMULATE (record results + update experience) ---")

        ts = time.strftime("%Y-%m-%d %H:%M")
        evolved_map = {e["regime"]: e for e in evolved}

        # Record per-regime results to DB
        for regime in KNOWN_REGIMES:
            e = evolved_map.get(regime, {})
            kpi_snapshot = {k: KPI_PROFILES[regime][k][0] for k in KPI_KEYS}
            db.add_record(ExperimentRecord(
                round_id=round_id, regime=regime,
                kpi_snapshot=kpi_snapshot,
                weights_used=e.get("weights", {}),
                performance={
                    "rate_mbps": results["known_rate"],
                    "outage": results["known_outage"],
                    "fairness": results.get("known_fairness", 0),
                },
                source=e.get("reasoning", "unknown"),
                timestamp=ts,
            ))

        for regime in NOVEL_REGIMES:
            e = evolved_map.get(regime, {})
            kpi_snapshot = {k: NOVEL_KPI_PROFILES[regime][k][0] for k in KPI_KEYS}
            db.add_record(ExperimentRecord(
                round_id=round_id, regime=regime,
                kpi_snapshot=kpi_snapshot,
                weights_used=e.get("weights", {}),
                performance={
                    "rate_mbps": results["novel_rate"],
                    "outage": results["novel_outage"],
                    "fairness": results.get("novel_fairness", 0),
                },
                source=e.get("reasoning", "unknown"),
                timestamp=ts,
            ))
        db.save()

        # Update experience context for novel regimes
        for regime in NOVEL_REGIMES:
            if regime in probe_results:
                applied_w = evolved_map.get(regime, {}).get("weights", {})
                experience_context = update_experience_context(
                    context=experience_context,
                    round_id=round_id,
                    regime=regime,
                    probe_result=probe_results[regime],
                    applied_weights=applied_w,
                    final_performance={
                        "rate_mbps": results["novel_rate"],
                        "outage": results["novel_outage"],
                        "fairness": results.get("novel_fairness", 0),
                    },
                )

        # Save experience context for resume
        with open(experience_path, "w") as f:
            json.dump({"experience_context": experience_context}, f, indent=2)

        # Update anchor DB with new probe data (grows the DB each round)
        if use_rag_anchors:
            probe_save_path_str = str(RESULTS_DIR / f"probe_round{round_id}.json")
            if Path(probe_save_path_str).exists():
                added = anchor_db.load_from_probe_files(
                    [probe_save_path_str], min_perf_mbps=5.0, verbose=0
                )
                if added > 0:
                    anchor_db.save(anchor_db_path)
                    if verbose:
                        print(f"  AnchorDB updated: {len(anchor_db)} entries (+{added} from round {round_id})")

        # ==============================
        # Check convergence
        # ==============================
        novel_rate = results["novel_rate"]
        known_rate = results["known_rate"]
        print(f"\n  Round {round_id} results:")
        print(f"    Known rate: {known_rate:.1f} Mbps")
        print(f"    Novel rate: {novel_rate:.1f} Mbps")

        if prev_novel_rate is not None:
            improvement = novel_rate - prev_novel_rate
            pct = improvement / max(prev_novel_rate, 1) * 100
            print(f"    Novel improvement: {improvement:+.1f} Mbps ({pct:+.1f}%)")
            if abs(pct) < convergence_threshold and round_id >= start_round + 1:
                print(f"    Converged (improvement < {convergence_threshold}%)")
                break
        prev_novel_rate = novel_rate

        for regime in NOVEL_REGIMES:
            gap = db.get_baseline_gap(regime)
            if gap:
                print(f"    {regime}: gap={gap['rate_gap_pct']:+.1f}% vs baseline")

    # Final summary
    best_round = db.get_latest_round()
    best_model_dir = f"models/evolved_v2_mlp_round{best_round}"
    print(f"\n{'='*60}")
    print(f"Evolution v2 complete. Best model: {best_model_dir}")
    print(f"Run full comparison with:")
    print(f"  python -m scripts.run_oracle_mlp --timesteps 500000 --seeds 42,123,456 "
          f"--skip-phase0 --oracle-model-dir {best_model_dir}")
    print(f"{'='*60}")

    # Print experience log
    if experience_context:
        print(f"\n--- Accumulated Experience ---")
        print(experience_context)

    return best_model_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-Evolving LLM-MLP v2 (Causal Feedback + RAG)")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--known-steps", type=int, default=250_000)
    parser.add_argument("--novel-steps", type=int, default=250_000)
    parser.add_argument("--probe-steps", type=int, default=50_000)
    parser.add_argument("--probe-delta", type=float, default=0.2, help="Perturbation fraction (0.2=±20%%)")
    parser.add_argument("--no-bidirectional", action="store_true", help="Only probe +delta (not -delta)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-llm", action="store_true", help="Use LLM API for causal reasoning")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG anchoring (use baseline causal prompt)")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--convergence", type=float, default=5.0)
    parser.add_argument("--save-prefix", type=str, default="EVOLVED_v2",
                        help="Prefix for result directory names (e.g. RAG_EVOLVED_v2)")
    args = parser.parse_args()

    run_evolution_v2(
        max_rounds=args.max_rounds,
        seed=args.seed,
        use_llm=args.use_llm,
        use_rag_anchors=not args.no_rag,
        api_key=args.api_key,
        probe_steps=args.probe_steps,
        probe_delta=args.probe_delta,
        probe_bidirectional=not args.no_bidirectional,
        known_steps=args.known_steps,
        novel_steps=args.novel_steps,
        convergence_threshold=args.convergence,
        save_prefix=args.save_prefix,
        verbose=1,
    )
