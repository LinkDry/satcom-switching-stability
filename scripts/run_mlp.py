#!/usr/bin/env python3
"""Run MLP architect experiment — compare with rule-based and fixed baseline.

Uses the SAME code structure as run_ablation.py:run_rule_based (which achieves 342.8 Mbps)
but replaces RuleBasedMDPSelector with MLPArchitect.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent
from llm.mlp_architect import MLPArchitect, load_mlp, WEIGHT_KEYS
from llm.regime_detector import CUSUMDetector
from simulator.env import BeamAllocationEnv, FlatActionWrapper

REGIME_SEQ = ["urban", "maritime", "disaster", "mixed"]


def run_mlp_architect(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """MLP architect: CUSUM detects change → MLP predicts weights from KPIs."""
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    # Load trained MLP
    mlp = load_mlp()

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    switch_log = []
    current_weights = None
    t0 = time.time()

    if verbose:
        print(f"  MLP architect: seed={seed}")

    # Initial training
    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < timesteps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=REGIME_SEQ, epochs_per_regime=50, seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        all_rates.append(rate)

        # Feed KPI history to CUSUM detector
        regime_changed = False
        for kpi in m.get("kpi_history", []):
            if detector.update(kpi):
                regime_changed = True
                break

        if regime_changed:
            # Get last KPI snapshot
            last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}

            # MLP predicts weights (~1ms)
            new_weights = mlp.predict_weights(last_kpi)

            # Only apply if weights actually changed significantly
            if current_weights is None or _weights_changed(current_weights, new_weights):
                env.unwrapped.update_reward_weights(new_weights)
                current_weights = new_weights.copy()
                switch_log.append({
                    "step": total_trained,
                    "weights": {k: round(v, 4) for k, v in new_weights.items()},
                    "kpi": {k: round(float(last_kpi.get(k, 0)), 2) for k in ["avg_demand", "spatial_gini", "peak_beam_demand"]}
                })
                if verbose:
                    print(f"    Step {total_trained}: MLP → {new_weights}")

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)
    result = {**final, "train_time_s": train_time, "seed": seed,
              "mdp_switches": len(switch_log), "method": "mlp_architect"}

    if verbose:
        print(f"  Result: rate={final['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={final['mean_outage_count']:.1f} switches={len(switch_log)}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(Path(save_dir) / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))
    return result


def _weights_changed(old_w, new_w, threshold=0.05):
    """Check if weights changed significantly (avoid redundant switches)."""
    for k in new_w:
        if abs(new_w.get(k, 0) - old_w.get(k, 0)) > threshold:
            return True
    return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    results_base = Path(__file__).parent.parent / "results"

    # Run MLP architect
    print("=== MLP Architect ===")
    r = run_mlp_architect(seed=args.seed, timesteps=args.timesteps,
                          save_dir=str(results_base / f"R7_mlp_seed{args.seed}"))
    print(f"\nMLP: rate={r['mean_sum_rate_mbps']:.1f} outage={r['mean_outage_count']:.2f} sw={r['mdp_switches']}")
