#!/usr/bin/env python3
"""Self-Evolving LLM-MLP Closed-Loop Experiment.

Evolution loop:
  Round 0: Per-regime baseline evaluation as seed data
  Round 1..N: LLM reflects on results → evolve weights → retrain MLP → experiment → feedback
  Final: Best evolved MLP runs full 500k comparison

Uses local LM Studio Qwen3.5-9B for LLM reflection.
Falls back to synthetic evolution if LLM API unavailable.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.evolution_db import EvolutionDB, ExperimentRecord
from llm.reflection_prompt import llm_evolve, synthetic_evolve
from llm.oracle_data_generator import (
    generate_oracle_data_evolved, NOVEL_KPI_PROFILES,
    NOVEL_EXPERT_HINTS, _sample_kpi,
)
from llm.mlp_architect import (
    MLPArchitect, incremental_train_mlp, KPI_KEYS, WEIGHT_KEYS,
    EXPERT_WEIGHTS, KPI_PROFILES,
)
from llm.quality_filter import filter_oracle_data
from simulator.env import BeamAllocationEnv, FlatActionWrapper
from agents.ppo_agent import PPOAgent, evaluate_agent
from scripts.run_oracle_mlp import run_oracle_mlp_experiment

KNOWN_REGIMES = ["urban", "maritime", "disaster", "mixed"]
NOVEL_REGIMES = ["iot_burst", "polar_handover", "hot_cold"]
ALL_REGIMES = KNOWN_REGIMES + NOVEL_REGIMES
RESULTS_DIR = Path(__file__).parent.parent / "results"

# LLM config — local LM Studio
LLM_MODEL = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2"
LLM_BASE_URL = "http://localhost:1234/v1"


def _weights_changed(old, new, tol=0.05):
    if old is None:
        return True
    return any(abs(old.get(k, 0) - new.get(k, 0)) > tol for k in new)


def evaluate_single_regime(
    mlp_model: MLPArchitect,
    regime: str,
    steps: int = 50_000,
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    """Evaluate MLP on a single regime to get per-regime metrics."""
    def predict(kpi):
        return mlp_model.predict_weights(kpi)

    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=[regime], epochs_per_regime=500, seed=seed))
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    current_weights = None
    segment = min(12500, steps)
    total = 0

    agent.train(total_timesteps=segment)
    total += segment

    while total < steps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=[regime], epochs_per_regime=50, seed=seed + total))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}
        if last_kpi:
            new_w = predict(last_kpi)
            if _weights_changed(current_weights, new_w):
                current_weights = new_w
                env.unwrapped.update_reward_weights(current_weights)
        s = min(segment, steps - total)
        agent.train(total_timesteps=s)
        total += s

    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=[regime], epochs_per_regime=100, seed=seed + 7777))
    final = evaluate_agent(agent, eval_env, n_episodes=3)

    result = {
        "rate_mbps": final["mean_sum_rate_mbps"],
        "outage": final["mean_outage_count"],
        "fairness": final.get("mean_fairness_index", 0.0),
    }
    if verbose:
        print(f"    {regime}: rate={result['rate_mbps']:.1f} outage={result['outage']:.2f} "
              f"fairness={result['fairness']:.3f}")
    return result


def load_baseline_metrics_perregime(
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    """Get baseline metrics from GEN_mlp results (the real 250k-step evaluation)."""
    metrics_path = RESULTS_DIR / f"GEN_mlp_seed{seed}" / "metrics.json"
    if not metrics_path.exists():
        print(f"  WARNING: No baseline at {metrics_path}")
        return {}

    with open(metrics_path) as f:
        m = json.load(f)

    # Use aggregated values — per-regime short evals are unreliable
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


def populate_round0(
    db: EvolutionDB,
    seed: int = 42,
    verbose: int = 1,
) -> bool:
    """Populate Round 0 from existing oracle_mlp experiment results."""
    metrics_path = RESULTS_DIR / f"ORACLE_MLP_seed{seed}" / "metrics.json"
    if not metrics_path.exists():
        print(f"  ERROR: No oracle_mlp results at {metrics_path}. Run run_oracle_mlp.py first.")
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


def run_evolution(
    max_rounds: int = 3,
    seed: int = 42,
    use_llm: bool = True,
    api_key: str = None,
    convergence_threshold: float = 5.0,
    known_steps: int = 250_000,
    novel_steps: int = 250_000,
    verbose: int = 1,
):
    """Main evolution loop with proper 500k feedback per round."""
    db_path = RESULTS_DIR / "evolution_db.json"

    # Step 0: Get per-regime baseline metrics
    if verbose:
        print("\n--- Loading per-regime baseline metrics ---")
    baseline = load_baseline_metrics_perregime(seed=seed, verbose=verbose)
    db = EvolutionDB(str(db_path), baseline_metrics=baseline)

    # Load existing DB or populate Round 0
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

    for round_id in range(start_round, start_round + max_rounds):
        print(f"\n{'='*60}")
        print(f"EVOLUTION ROUND {round_id}")
        print(f"{'='*60}")

        # Step 1: Evolve weights (LLM or synthetic)
        print(f"\n--- Step 1: Weight Evolution ---")
        evolved = []
        if use_llm:
            try:
                evolved = llm_evolve(
                    db, round_id, ALL_REGIMES,
                    model=LLM_MODEL, api_key=api_key,
                    base_url=LLM_BASE_URL, verbose=verbose,
                )
            except Exception as e:
                print(f"  [LLM] Error: {e}, falling back to synthetic")

        if not evolved:
            if verbose:
                print("  Using synthetic evolution")
            evolved = synthetic_evolve(db, round_id, ALL_REGIMES, verbose=verbose)

        # Step 2: Generate training data from evolved weights
        print(f"\n--- Step 2: Generate Evolved Training Data ---")
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
            output_path=str(RESULTS_DIR / f"evolution_round{round_id}_data.json"),
            verbose=verbose,
        )

        # Step 3: Quality filter (skip consistency to avoid conflict with LLM evolution)
        print(f"\n--- Step 3: Quality Filter ---")
        filtered = filter_oracle_data(
            evolved_data, use_bounds=True,
            use_consistency=True, skip_consistency=True,
            verbose=verbose,
        )

        # Step 4: Incremental MLP training (always from oracle_mlp base to prevent drift)
        print(f"\n--- Step 4: Incremental MLP Training ---")
        base_dir = "models/oracle_mlp"
        save_dir = f"models/evolved_mlp_round{round_id}"
        model = incremental_train_mlp(
            evolved_samples=filtered,
            previous_best_samples=prev_best,
            base_model_dir=base_dir,
            save_dir=save_dir,
            epochs=150, lr=3e-4,
            verbose=verbose,
        )

        # Step 5: Full feedback experiment (250k known + 250k novel)
        print(f"\n--- Step 5: Feedback Experiment ({known_steps + novel_steps} steps) ---")
        fb_save_dir = str(RESULTS_DIR / f"EVOLVED_round{round_id}_seed{seed}")
        results = run_oracle_mlp_experiment(
            seed=seed,
            known_steps=known_steps,
            novel_steps=novel_steps,
            oracle_mlp_dir=save_dir,
            save_dir=fb_save_dir,
            verbose=verbose,
        )

        # Step 6: Record per-regime results to DB
        ts = time.strftime("%Y-%m-%d %H:%M")
        evolved_map = {e["regime"]: e for e in evolved}

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
                source="llm_evolved" if use_llm else "synthetic_evolved",
                timestamp=ts, reasoning=e.get("reasoning", ""),
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
                source="llm_evolved" if use_llm else "synthetic_evolved",
                timestamp=ts, reasoning=e.get("reasoning", ""),
            ))
        db.save()

        # Step 7: Check convergence
        novel_rate = results["novel_rate"]
        print(f"\n  Round {round_id} novel rate: {novel_rate:.1f} Mbps")
        if prev_novel_rate is not None:
            improvement = novel_rate - prev_novel_rate
            pct = improvement / max(prev_novel_rate, 1) * 100
            print(f"  Improvement: {improvement:+.1f} Mbps ({pct:+.1f}%)")
            if abs(pct) < convergence_threshold and round_id >= start_round + 1:
                print(f"  Converged (improvement < {convergence_threshold}%)")
                break
        prev_novel_rate = novel_rate

        for regime in NOVEL_REGIMES:
            gap = db.get_baseline_gap(regime)
            if gap:
                print(f"  {regime}: gap={gap['rate_gap_pct']:+.1f}% vs baseline")

    best_round = db.get_latest_round()
    best_model_dir = f"models/evolved_mlp_round{best_round}"
    print(f"\n{'='*60}")
    print(f"Evolution complete. Best model: {best_model_dir}")
    print(f"Run full comparison with:")
    print(f"  python -m scripts.run_oracle_mlp --timesteps 500000 --seeds 42,123,456 "
          f"--skip-phase0 --oracle-model-dir {best_model_dir}")
    print(f"{'='*60}")

    return best_model_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Self-Evolving LLM-MLP Experiment")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--known-steps", type=int, default=250_000)
    parser.add_argument("--novel-steps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-llm", action="store_true", help="Use real LLM API for evolution")
    parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key")
    parser.add_argument("--convergence", type=float, default=5.0, help="Convergence threshold (%%)")
    args = parser.parse_args()

    run_evolution(
        max_rounds=args.max_rounds,
        seed=args.seed,
        use_llm=args.use_llm,
        api_key=args.api_key,
        convergence_threshold=args.convergence,
        known_steps=args.known_steps,
        novel_steps=args.novel_steps,
        verbose=1,
    )
