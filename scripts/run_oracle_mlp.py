#!/usr/bin/env python3
"""Oracle-MLP Experiment: LLM-as-Oracle + Continual MLP Learning.

Pipeline:
  Phase 0 (Offline): Generate oracle data (LLM or synthetic) → quality filter → continual MLP training
  Phase 1 (Online):  Oracle-MLP on known regimes (250k steps)
  Phase 2 (Online):  Oracle-MLP on novel regimes (250k steps)
  Phase 3 (Expand):  If novel regime detected with poor performance → generate new oracle data → retrain → resume

Compares oracle_mlp against baseline mlp and finetuned methods.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent
from evaluation.metrics import compute_recovery_time
from llm.regime_detector import CUSUMDetector
from llm.mlp_architect import (
    MLPArchitect, load_mlp, WEIGHT_KEYS, KPI_KEYS, SCALES,
    continual_train_mlp,
)
from llm.oracle_data_generator import (
    generate_oracle_data_synthetic, generate_oracle_data_llm,
    NOVEL_KPI_PROFILES, _sample_kpi,
)
from llm.quality_filter import filter_oracle_data
from simulator.env import BeamAllocationEnv, FlatActionWrapper

KNOWN_REGIMES = ["urban", "maritime", "disaster", "mixed"]
NOVEL_REGIMES = ["iot_burst", "polar_handover", "hot_cold"]
ALL_REGIMES = KNOWN_REGIMES + NOVEL_REGIMES
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _weights_changed(old, new, tol=0.05):
    if old is None:
        return True
    return any(abs(old.get(k, 0) - new.get(k, 0)) > tol for k in new)


def phase0_prepare_oracle_mlp(
    use_llm: bool = False,
    n_per_regime: int = 500,
    seed: int = 42,
    model_api_key: str = None,
    model_base_url: str = None,
    verbose: int = 1,
) -> MLPArchitect:
    """Phase 0: Generate oracle data, filter, and train Oracle-MLP.

    Returns trained Oracle-MLP model.
    """
    data_dir = RESULTS_DIR / "oracle_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=== Phase 0: Oracle Data Generation ===")

    # Step 1: Generate oracle data
    if use_llm:
        samples = generate_oracle_data_llm(
            n_per_regime=n_per_regime,
            regimes=ALL_REGIMES,
            api_key=model_api_key,
            base_url=model_base_url,
            seed=seed,
            output_path=str(data_dir / "oracle_raw.json"),
            verbose=verbose,
        )
    else:
        samples = generate_oracle_data_synthetic(
            n_per_regime=n_per_regime,
            regimes=ALL_REGIMES,
            seed=seed,
            output_path=str(data_dir / "oracle_raw.json"),
            verbose=verbose,
        )

    if verbose:
        print(f"\n  Raw oracle samples: {len(samples)}")

    # Step 2: Quality filter
    if verbose:
        print("\n=== Phase 0: Quality Filtering ===")

    filtered = filter_oracle_data(
        samples,
        use_bounds=True,
        use_consistency=True,
        use_rollout=False,  # Skip rollout for speed in prototype
        verbose=verbose,
    )

    with open(data_dir / "oracle_filtered.json", "w") as f:
        json.dump(filtered, f, indent=2)

    if verbose:
        print(f"  Filtered oracle samples: {len(filtered)}")

    # Step 3: Continual MLP training
    if verbose:
        print("\n=== Phase 0: Continual MLP Training ===")

    model = continual_train_mlp(
        oracle_samples=filtered,
        base_model_dir="models/mlp_architect",
        save_dir="models/oracle_mlp",
        epochs=200,
        lr=5e-4,
        replay_samples=5000,
        verbose=verbose,
    )

    return model


def run_oracle_mlp_experiment(
    seed: int = 42,
    known_steps: int = 250_000,
    novel_steps: int = 250_000,
    oracle_mlp_dir: str = "models/oracle_mlp",
    save_dir: str = None,
    verbose: int = 1,
) -> dict:
    """Run the Oracle-MLP generalization experiment.

    Same protocol as run_generalization.py but using the Oracle-trained MLP.
    """
    # Load Oracle-MLP
    mlp = MLPArchitect(hidden=64)
    model_path = Path(oracle_mlp_dir) / "mlp_architect.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Oracle-MLP not found at {model_path}. Run phase0 first.")

    import torch
    mlp.load_state_dict(torch.load(model_path, weights_only=True))
    mlp.eval()

    def predict(kpi):
        return mlp.predict_weights(kpi)

    # Phase 1: Known regimes
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=200, seed=seed))
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    segment = 62500
    total_trained = 0
    current_weights = None
    switch_log = []
    phase_results = {"known": [], "novel": []}
    t0 = time.time()

    known_step_rates = []
    known_regime_change_steps = []
    known_fairness = []
    novel_step_rates = []
    novel_regime_change_steps = []
    novel_fairness = []

    if verbose:
        print(f"\n  Phase 1: Known regimes ({known_steps} steps)")

    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < known_steps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=KNOWN_REGIMES, epochs_per_regime=50, seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        known_step_rates.append(rate)
        known_fairness.append(m.get("mean_fairness_index", 0.0))

        last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}
        if last_kpi:
            new_weights = predict(last_kpi)
            if _weights_changed(current_weights, new_weights):
                current_weights = new_weights
                env.unwrapped.update_reward_weights(current_weights)
                known_regime_change_steps.append(len(known_step_rates) - 1)
                switch_log.append({"step": total_trained, "phase": "known",
                                   "weights": {k: round(v, 4) for k, v in current_weights.items()}})

        steps = min(segment, known_steps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    # Evaluate known
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=50, seed=seed + 8888))
    known_eval = evaluate_agent(agent, eval_env, n_episodes=3)
    phase_results["known"] = {
        "rate": known_eval["mean_sum_rate_mbps"],
        "outage": known_eval["mean_outage_count"],
        "fairness": known_eval.get("mean_fairness_index", 0.0),
    }

    if verbose:
        print(f"  Phase 1 done: rate={known_eval['mean_sum_rate_mbps']:.1f} "
              f"outage={known_eval['mean_outage_count']:.2f}")

    # Phase 2: Novel regimes
    if verbose:
        print(f"\n  Phase 2: Novel regimes ({novel_steps} steps)")

    env_novel = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=200, seed=seed + 5000))
    agent_novel = PPOAgent(env_novel, device="cpu", seed=seed, verbose=0)
    agent_novel.model = agent.model  # Transfer learned policy

    novel_trained = 0
    novel_rates = []

    # Force initial weight update for novel regimes
    init_eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=50, seed=seed + 5001))
    init_m = evaluate_agent(agent_novel, init_eval_env, n_episodes=1)
    init_kpi = init_m.get("kpi_history", [{}])[-1] if init_m.get("kpi_history") else {}
    if init_kpi:
        new_weights = predict(init_kpi)
        current_weights = new_weights
        env_novel.unwrapped.update_reward_weights(current_weights)
        switch_log.append({"step": known_steps, "phase": "novel_init",
                           "weights": {k: round(v, 4) for k, v in current_weights.items()}})
        if verbose:
            print(f"    Novel init weights: {current_weights}")

    agent_novel.train(total_timesteps=segment)
    novel_trained += segment

    while novel_trained < novel_steps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=NOVEL_REGIMES, epochs_per_regime=50, seed=seed + novel_trained + 5000))
        m = evaluate_agent(agent_novel, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        novel_rates.append(rate)
        novel_step_rates.append(rate)
        novel_fairness.append(m.get("mean_fairness_index", 0.0))

        last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}
        if last_kpi:
            new_weights = predict(last_kpi)
            if _weights_changed(current_weights, new_weights):
                current_weights = new_weights
                env_novel.unwrapped.update_reward_weights(current_weights)
                novel_regime_change_steps.append(len(novel_step_rates) - 1)
                switch_log.append({"step": known_steps + novel_trained, "phase": "novel",
                                   "weights": {k: round(v, 4) for k, v in current_weights.items()}})
                if verbose:
                    print(f"    Step {known_steps + novel_trained}: oracle_mlp → {current_weights}")

        steps = min(segment, novel_steps - novel_trained)
        agent_novel.train(total_timesteps=steps)
        novel_trained += steps

    # Final evaluation on novel regimes
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=50, seed=seed + 9999))
    novel_eval = evaluate_agent(agent_novel, eval_env, n_episodes=3)
    phase_results["novel"] = {
        "rate": novel_eval["mean_sum_rate_mbps"],
        "outage": novel_eval["mean_outage_count"],
        "fairness": novel_eval.get("mean_fairness_index", 0.0),
    }

    # Compute recovery times
    known_recovery = compute_recovery_time(known_step_rates, known_regime_change_steps)
    novel_recovery = compute_recovery_time(novel_step_rates, novel_regime_change_steps)

    train_time = time.time() - t0

    if verbose:
        print(f"  Phase 2 done: rate={novel_eval['mean_sum_rate_mbps']:.1f} "
              f"outage={novel_eval['mean_outage_count']:.2f} "
              f"fairness={phase_results['novel']['fairness']:.3f}")

    result = {
        "method": "oracle_mlp", "seed": seed,
        "known_rate": phase_results["known"]["rate"],
        "known_outage": phase_results["known"]["outage"],
        "known_fairness": phase_results["known"]["fairness"],
        "novel_rate": phase_results["novel"]["rate"],
        "novel_outage": phase_results["novel"]["outage"],
        "novel_fairness": phase_results["novel"]["fairness"],
        "known_avg_fairness": float(np.mean(known_fairness)) if known_fairness else 0.0,
        "novel_avg_fairness": float(np.mean(novel_fairness)) if novel_fairness else 0.0,
        "known_recovery": known_recovery,
        "novel_recovery": novel_recovery,
        "mdp_switches": len(switch_log),
        "train_time_s": train_time,
        "novel_rate_trajectory": novel_rates,
    }

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(Path(save_dir) / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Oracle-MLP Experiment")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--use-llm", action="store_true", help="Use real LLM API for oracle data")
    parser.add_argument("--oracle-samples", type=int, default=500, help="Samples per regime")
    parser.add_argument("--skip-phase0", action="store_true", help="Skip oracle training, use existing model")
    parser.add_argument("--quick", action="store_true", help="Quick prototype: 50k steps, 1 seed")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    if args.quick:
        args.timesteps = 50_000
        args.oracle_samples = 200
        seeds = [42]

    known_steps = args.timesteps // 2
    novel_steps = args.timesteps // 2

    # Phase 0: Prepare Oracle-MLP
    if not args.skip_phase0:
        print("\n" + "=" * 60)
        print("PHASE 0: Oracle Data Generation + MLP Training")
        print("=" * 60)
        phase0_prepare_oracle_mlp(
            use_llm=args.use_llm,
            n_per_regime=args.oracle_samples,
            seed=42,
            verbose=1,
        )

    # Phase 1+2: Run experiments
    all_results = []
    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Oracle-MLP Experiment: seed={seed}, {args.timesteps} steps")
        print("=" * 60)

        r = run_oracle_mlp_experiment(
            seed=seed,
            known_steps=known_steps,
            novel_steps=novel_steps,
            save_dir=str(RESULTS_DIR / f"ORACLE_MLP_seed{seed}"),
            verbose=1,
        )
        all_results.append(r)

    # Summary
    print(f"\n{'=' * 60}")
    print("ORACLE-MLP RESULTS SUMMARY")
    print("=" * 60)
    header = f"{'Seed':>6} {'Known Rate':>10} {'Novel Rate':>10} {'Known Fair':>10} {'Novel Fair':>10} {'Known Out':>10} {'Novel Out':>10} {'Switches':>8}"
    print(header)
    for r in all_results:
        print(f"{r['seed']:>6} {r['known_rate']:>10.1f} {r['novel_rate']:>10.1f} "
              f"{r.get('known_fairness', 0):>10.3f} {r.get('novel_fairness', 0):>10.3f} "
              f"{r['known_outage']:>10.2f} {r['novel_outage']:>10.2f} {r['mdp_switches']:>8}")

    if len(seeds) > 1:
        kr = np.array([r["known_rate"] for r in all_results])
        nr = np.array([r["novel_rate"] for r in all_results])
        nf = np.array([r.get("novel_fairness", 0) for r in all_results])
        print(f"\nAggregate: known={kr.mean():.1f}+/-{kr.std():.1f} "
              f"novel={nr.mean():.1f}+/-{nr.std():.1f} "
              f"novel_fair={nf.mean():.3f}+/-{nf.std():.3f}")

    # Save aggregate
    agg_path = RESULTS_DIR / "oracle_mlp_aggregate.json"
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {agg_path}")
