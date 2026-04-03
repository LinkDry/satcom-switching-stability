#!/usr/bin/env python3
"""Run Oracle-MLP experiment with Causal Correction Layer.

Uses the standard oracle_mlp model but applies probe-based per-regime
weight corrections for novel regimes. This should close the gap between
LLM's suggested weights and what MLP actually predicts.

Compares:
  1. oracle_mlp (no correction) — baseline
  2. oracle_mlp + best_round correction (override mode)
  3. oracle_mlp + best_round correction (blend mode)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent, RegimeWeightSwitcher
from evaluation.metrics import compute_recovery_time
from llm.regime_detector import CUSUMDetector
from llm.causal_correction import load_corrected_mlp, CorrectedMLPArchitect
from llm.mlp_architect import MLPArchitect, load_mlp, WEIGHT_KEYS, KPI_KEYS
from simulator.env import BeamAllocationEnv, FlatActionWrapper

KNOWN_REGIMES = ["urban", "maritime", "disaster", "mixed"]
NOVEL_REGIMES = ["iot_burst", "polar_handover", "hot_cold"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


def _weights_changed(old, new, tol=0.05):
    if old is None:
        return True
    return any(abs(old.get(k, 0) - new.get(k, 0)) > tol for k in new)


def run_corrected_experiment(
    seed: int = 42,
    known_steps: int = 250_000,
    novel_steps: int = 250_000,
    mlp_dir: str = "models/oracle_mlp",
    correction_source: str = "best_round",
    correction_mode: str = "override",
    save_dir: str = None,
    verbose: int = 1,
) -> dict:
    """Run experiment with corrected MLP architect."""

    # Load corrected MLP
    corrected_mlp = load_corrected_mlp(
        mlp_dir=mlp_dir,
        correction_source=correction_source,
        correction_mode=correction_mode,
        verbose=verbose,
    )

    def predict(kpi, regime=None):
        return corrected_mlp.predict_weights(kpi, regime=regime)

    def predict_from_history(kpi_history):
        return corrected_mlp.predict_weights_from_history(kpi_history)

    # Phase 1: Known regimes
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=200, seed=seed))
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    segment = 62500
    total_trained = 0
    current_weights = None
    switch_log = []
    known_step_rates = []
    known_fairness = []
    known_regime_change_steps = []
    t0 = time.time()

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
        kpi_history = m.get("kpi_history", [])
        if kpi_history:
            new_weights = predict_from_history(kpi_history)
            regime_label = last_kpi.get("regime_type") or last_kpi.get("_regime")
            if _weights_changed(current_weights, new_weights):
                current_weights = new_weights
                env.unwrapped.update_reward_weights(current_weights)
                known_regime_change_steps.append(len(known_step_rates) - 1)
                switch_log.append({"step": total_trained, "phase": "known",
                                   "regime": regime_label,
                                   "weights": {k: round(v, 4) for k, v in current_weights.items()}})

        steps = min(segment, known_steps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    # Evaluate known
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=50, seed=seed + 8888))
    known_eval = evaluate_agent(agent, eval_env, n_episodes=3)

    if verbose:
        print(f"  Phase 1 done: rate={known_eval['mean_sum_rate_mbps']:.1f} "
              f"outage={known_eval['mean_outage_count']:.2f}")

    # Phase 2: Novel regimes with per-regime dynamic weight switching
    if verbose:
        print(f"\n  Phase 2: Novel regimes ({novel_steps} steps)")

    env_novel = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=200, seed=seed + 5000))
    agent_novel = PPOAgent(env_novel, device="cpu", seed=seed, verbose=0)
    agent_novel.model = agent.model  # Transfer learned policy

    novel_trained = 0
    novel_rates = []
    novel_step_rates = []
    novel_fairness = []
    novel_regime_change_steps = []

    # Build per-regime weight function for the switcher callback
    def regime_weight_fn(regime_label):
        """Return corrected weights for a given regime label."""
        # Use a representative KPI for MLP base prediction (doesn't matter much
        # since corrections override the key weights)
        dummy_kpi = {k: 0.0 for k in KPI_KEYS}
        return corrected_mlp.predict_weights(dummy_kpi, regime=regime_label)

    # Create callback that switches weights on every regime transition
    from stable_baselines3.common.callbacks import CallbackList
    switcher = RegimeWeightSwitcher(weight_fn=regime_weight_fn, verbose=1 if verbose else 0)
    logger = None  # Will be created in the train call

    if verbose:
        # Preview what weights will be used for each regime
        for r in NOVEL_REGIMES:
            w = regime_weight_fn(r)
            print(f"    {r}: sw={w['switching']:.4f} sr={w['sum_rate']:.4f} "
                  f"f={w['fairness']:.4f} o={w['outage']:.4f} q={w['queue']:.4f}")

    # Train with regime-aware switching
    from agents.ppo_agent import TrainingLogger
    train_logger = TrainingLogger(verbose=1 if verbose else 0)
    callback_list = CallbackList([switcher, train_logger])

    agent_novel.model.learn(
        total_timesteps=novel_steps,
        callback=callback_list,
    )
    agent_novel.total_trained_steps += novel_steps

    if verbose:
        print(f"    Regime switches during training: {switcher.switch_count}")
        if switcher.regime_switch_log:
            # Show a sample of switches
            for entry in switcher.regime_switch_log[:5]:
                print(f"      step={entry['step']}: {entry['from']}->{entry['to']} sw={entry['switching']:.4f}")
            if len(switcher.regime_switch_log) > 5:
                print(f"      ... ({len(switcher.regime_switch_log)} total switches)")

    # Periodic evaluation during training (for rate trajectory)
    # Run short evals to capture novel_step_rates
    for eval_i in range(4):
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=NOVEL_REGIMES, epochs_per_regime=50,
            seed=seed + eval_i + 7000))
        m = evaluate_agent(agent_novel, eval_env, n_episodes=1)
        novel_step_rates.append(m["mean_sum_rate_mbps"])
        novel_fairness.append(m.get("mean_fairness_index", 0.0))

    # Final evaluation
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=50, seed=seed + 9999))
    novel_eval = evaluate_agent(agent_novel, eval_env, n_episodes=3)

    known_recovery = compute_recovery_time(known_step_rates, known_regime_change_steps)
    novel_recovery = compute_recovery_time(novel_step_rates, novel_regime_change_steps)
    train_time = time.time() - t0

    if verbose:
        print(f"  Phase 2 done: rate={novel_eval['mean_sum_rate_mbps']:.1f} "
              f"outage={novel_eval['mean_outage_count']:.2f} "
              f"fairness={novel_eval.get('mean_fairness_index', 0):.3f}")
        print(f"  Regime switches: {switcher.switch_count}")

    result = {
        "method": f"corrected_mlp_{correction_source}_{correction_mode}",
        "seed": seed,
        "known_rate": known_eval["mean_sum_rate_mbps"],
        "known_outage": known_eval["mean_outage_count"],
        "known_fairness": known_eval.get("mean_fairness_index", 0.0),
        "novel_rate": novel_eval["mean_sum_rate_mbps"],
        "novel_outage": novel_eval["mean_outage_count"],
        "novel_fairness": novel_eval.get("mean_fairness_index", 0.0),
        "known_avg_fairness": float(np.mean(known_fairness)) if known_fairness else 0.0,
        "novel_avg_fairness": float(np.mean(novel_fairness)) if novel_fairness else 0.0,
        "known_recovery": known_recovery,
        "novel_recovery": novel_recovery,
        "mdp_switches": len(switch_log) + switcher.switch_count,
        "regime_switches_during_training": switcher.switch_count,
        "train_time_s": train_time,
        "novel_rate_trajectory": novel_step_rates,
        "correction_source": correction_source,
        "correction_mode": correction_mode,
        "regime_hits": corrected_mlp.get_diagnostics()["regime_hits"],
    }

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(Path(save_dir) / "switch_log.json", "w") as f:
            # Combine switch_log (from Phase 1) with regime switcher log
            full_log = switch_log + [
                {"step": e["step"], "phase": "novel_train",
                 "from_regime": e["from"], "to_regime": e["to"],
                 "switching": e["switching"]}
                for e in switcher.regime_switch_log
            ]
            json.dump(full_log, f, indent=2)

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Corrected Oracle-MLP Experiment")
    parser.add_argument("--known-steps", type=int, default=250_000)
    parser.add_argument("--novel-steps", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds for multi-seed run")
    parser.add_argument("--correction-source", type=str, default="best_round",
                        choices=["best_round", "probe_average"])
    parser.add_argument("--correction-mode", type=str, default="override",
                        choices=["override", "blend"])
    parser.add_argument("--mlp-dir", type=str, default="models/oracle_mlp")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Corrected MLP Experiment: seed={seed}, "
              f"source={args.correction_source}, mode={args.correction_mode}")
        print(f"{'='*60}")

        save_dir = str(RESULTS_DIR / f"CORRECTED_{args.correction_source}_{args.correction_mode}_seed{seed}")
        r = run_corrected_experiment(
            seed=seed,
            known_steps=args.known_steps,
            novel_steps=args.novel_steps,
            mlp_dir=args.mlp_dir,
            correction_source=args.correction_source,
            correction_mode=args.correction_mode,
            save_dir=save_dir,
            verbose=1,
        )
        all_results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print(f"CORRECTED MLP RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  seed={r['seed']}: known={r['known_rate']:.1f} novel={r['novel_rate']:.1f} "
              f"n_outage={r['novel_outage']:.2f} n_fair={r['novel_fairness']:.3f} "
              f"hits={r['regime_hits']}")

    if len(seeds) > 1:
        kr = np.array([r["known_rate"] for r in all_results])
        nr = np.array([r["novel_rate"] for r in all_results])
        print(f"\n  Aggregate: known={kr.mean():.1f}±{kr.std():.1f} "
              f"novel={nr.mean():.1f}±{nr.std():.1f}")
