#!/usr/bin/env python3
"""Generalization experiment: MLP vs Fine-tuned LLM on novel regimes.

Phase 1: Train on known regimes (urban, maritime, disaster, mixed) — 250k steps
Phase 2: Encounter novel regimes (iot_burst, polar_handover, hot_cold) — 250k steps

Hypothesis: LLM generalizes better to unseen regimes via reasoning.
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
from simulator.env import BeamAllocationEnv, FlatActionWrapper

KNOWN_REGIMES = ["urban", "maritime", "disaster", "mixed"]
NOVEL_REGIMES = ["iot_burst", "polar_handover", "hot_cold"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _weights_changed(old, new, tol=0.05):
    if old is None:
        return True
    return any(abs(old.get(k, 0) - new.get(k, 0)) > tol for k in new)


def run_generalization(method="mlp", seed=42, known_steps=250_000, novel_steps=250_000,
                       save_dir=None, verbose=1):
    """Run generalization experiment for a given method."""

    # Phase 1: Train on known regimes
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=200, seed=seed))
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    # Load the architect
    if method == "mlp":
        from llm.mlp_architect import MLPArchitect, load_mlp, WEIGHT_KEYS
        mlp = load_mlp()
        def predict(kpi):
            return mlp.predict_weights(kpi)
    elif method == "mlp_clipped":
        from llm.mlp_architect import MLPArchitect, load_mlp, WEIGHT_KEYS
        mlp = load_mlp()
        def predict(kpi):
            raw = mlp.predict_weights(kpi)
            # Clamp to expert weight range [0.01, 2.0] to prevent extrapolation failure
            return {k: max(0.01, min(2.0, v)) for k, v in raw.items()}
    elif method == "finetuned":
        from llm.finetuned_architect import FinetunedLLMArchitect
        arch = FinetunedLLMArchitect()
        def predict(kpi):
            return arch.predict_weights(kpi)
    elif method == "rule":
        from agents.rule_selector import RuleBasedMDPSelector
        selector = RuleBasedMDPSelector()
        def predict(kpi):
            spec, _ = selector.select_spec(kpi)
            w = {"sum_rate": 1.0, "outage": 1.0, "switching": 0.01, "queue": 0.0, "fairness": 0.0}
            if spec.spec_id == "maritime":
                w["fairness"] = 0.3
            elif spec.spec_id == "disaster":
                w["outage"] = 2.0
                w["queue"] = 0.1
            return w
    else:
        raise ValueError(f"Unknown method: {method}")

    segment = 62500
    total_trained = 0
    current_weights = None
    switch_log = []
    phase_results = {"known": [], "novel": []}
    t0 = time.time()

    # Tracking for recovery time and fairness
    known_step_rates = []
    known_regime_change_steps = []
    known_fairness = []
    novel_step_rates = []
    novel_regime_change_steps = []
    novel_fairness = []

    if verbose:
        print(f"  Phase 1: Training on known regimes ({known_steps} steps)")

    # Phase 1: Known regimes
    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < known_steps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=KNOWN_REGIMES, epochs_per_regime=50, seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        known_step_rates.append(rate)
        known_fairness.append(m.get("mean_fairness_index", 0.0))

        # Always update weights at each evaluation point to directly compare architects
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

    # Evaluate on known regimes
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=50, seed=seed + 8888))
    known_eval = evaluate_agent(agent, eval_env, n_episodes=3)
    phase_results["known"] = {
        "rate": known_eval["mean_sum_rate_mbps"],
        "outage": known_eval["mean_outage_count"],
        "fairness": known_eval.get("mean_fairness_index", 0.0),
    }

    if verbose:
        print(f"  Phase 1 done: rate={known_eval['mean_sum_rate_mbps']:.1f} outage={known_eval['mean_outage_count']:.2f}")

    # Phase 2: Novel regimes
    if verbose:
        print(f"  Phase 2: Novel regimes ({novel_steps} steps)")

    env_novel = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=200, seed=seed + 5000))
    # Transfer agent to novel env
    agent_novel = PPOAgent(env_novel, device="cpu", seed=seed, verbose=0)
    agent_novel.model = agent.model  # Transfer learned policy

    detector_novel = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    novel_trained = 0
    novel_rates = []

    # Force initial weight update for novel regimes before any training
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

        # Always update weights at each evaluation point
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
                    print(f"    Step {known_steps + novel_trained}: {method} → {current_weights}")
            last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}
            new_weights = predict(last_kpi)
            if _weights_changed(current_weights, new_weights):
                current_weights = new_weights
                env_novel.unwrapped.update_reward_weights(current_weights)
                switch_log.append({"step": known_steps + novel_trained, "phase": "novel",
                                   "weights": {k: round(v, 4) for k, v in current_weights.items()}})
                if verbose:
                    print(f"    Step {known_steps + novel_trained}: {method} → {current_weights}")

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
        if novel_recovery:
            avg_rec = np.mean([r["recovery_epochs"] for r in novel_recovery if r["recovery_epochs"] is not None])
            print(f"  Novel recovery time: {avg_rec:.1f} eval steps (avg)")

    result = {
        "method": method, "seed": seed,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=None, help="Single seed (legacy)")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.seed is not None:
        seeds = [args.seed]

    known_steps = args.timesteps // 2
    novel_steps = args.timesteps // 2

    all_results = []
    for seed in seeds:
        for method in ["rule", "mlp", "mlp_clipped", "finetuned"]:
            print(f"\n=== Generalization: {method} seed={seed} ===")
            r = run_generalization(
                method=method, seed=seed,
                known_steps=known_steps, novel_steps=novel_steps,
                save_dir=str(RESULTS_DIR / f"GEN_{method}_seed{seed}"),
                verbose=1)
            all_results.append(r)

    # Aggregate by method
    print("\n=== GENERALIZATION SUMMARY (per seed) ===")
    header = f"{'Method':<12} {'Seed':>6} {'Known Rate':>10} {'Novel Rate':>10} {'Known Fair':>10} {'Novel Fair':>10} {'Known Out':>10} {'Novel Out':>10} {'Switches':>8}"
    print(header)
    for r in all_results:
        novel_rec = r.get("novel_recovery", [])
        print(f"{r['method']:<12} {r['seed']:>6} {r['known_rate']:>10.1f} {r['novel_rate']:>10.1f} "
              f"{r.get('known_fairness', 0):>10.3f} {r.get('novel_fairness', 0):>10.3f} "
              f"{r['known_outage']:>10.2f} {r['novel_outage']:>10.2f} {r['mdp_switches']:>8}")

    # Aggregate means across seeds
    print("\n=== AGGREGATE (mean +/- std across seeds) ===")
    methods = ["rule", "mlp", "mlp_clipped", "finetuned"]
    print(f"{'Method':<12} {'Known Rate':>14} {'Novel Rate':>14} {'Novel Fair':>14} {'Novel Out':>14}")
    for method in methods:
        mrs = [r for r in all_results if r["method"] == method]
        if not mrs:
            continue
        kr = np.array([r["known_rate"] for r in mrs])
        nr = np.array([r["novel_rate"] for r in mrs])
        nf = np.array([r.get("novel_fairness", 0) for r in mrs])
        no = np.array([r["novel_outage"] for r in mrs])
        print(f"{method:<12} {kr.mean():>6.1f}+/-{kr.std():>4.1f} {nr.mean():>6.1f}+/-{nr.std():>4.1f} "
              f"{nf.mean():>6.3f}+/-{nf.std():>.3f} {no.mean():>6.2f}+/-{no.std():>4.2f}")

    # Save aggregate
    agg_path = RESULTS_DIR / "generalization_aggregate.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {agg_path}")
