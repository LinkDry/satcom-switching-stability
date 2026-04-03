#!/usr/bin/env python3
"""M1: Baseline experiments (R005-R008).

Train fixed-MDP PPO, SAC, heuristic, and reward-only LLM baselines.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulator.env import BeamAllocationEnv, FlatActionWrapper
from agents.ppo_agent import PPOAgent, evaluate_agent
from agents.baselines import MaxWeightHeuristic
from training.trainer import train_fixed_mdp


RESULTS_DIR = Path(__file__).parent.parent / "results"
REGIME_SEQUENCE = ["urban", "maritime", "disaster", "mixed"]
EPOCHS_PER_REGIME = 200


def run_ppo_baseline(seed: int = 42, total_timesteps: int = 500_000):
    """R005: Fixed-MDP PPO trained on urban, evaluated across all regimes."""
    print("\n=== R005: Fixed-MDP PPO ===")
    save_dir = str(RESULTS_DIR / f"R005_fixed_ppo_seed{seed}")

    result = train_fixed_mdp(
        regime_sequence=REGIME_SEQUENCE,
        epochs_per_regime=EPOCHS_PER_REGIME,
        total_timesteps=total_timesteps,
        seed=seed,
        save_dir=save_dir,
        device="cpu",
        verbose=1,
    )
    print(f"  Result: {json.dumps(result['metrics'], indent=2)}")
    return result


def run_heuristic_baseline(seed: int = 42):
    """R007: Max-weight matching heuristic."""
    print("\n=== R007: Max-Weight Heuristic ===")

    env = FlatActionWrapper(
        BeamAllocationEnv(
            regime_sequence=REGIME_SEQUENCE,
            epochs_per_regime=EPOCHS_PER_REGIME,
            seed=seed,
        )
    )
    heuristic = MaxWeightHeuristic(num_beams=env.env.num_beams)

    all_rates = []
    all_outages = []
    obs, info = env.reset()

    total_steps = len(REGIME_SEQUENCE) * EPOCHS_PER_REGIME
    t0 = time.time()

    for step in range(total_steps):
        action = heuristic.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        all_rates.append(info["sum_rate_mbps"])
        all_outages.append(info["outage_count"])
        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - t0

    metrics = {
        "mean_sum_rate_mbps": float(np.mean(all_rates)),
        "std_sum_rate_mbps": float(np.std(all_rates)),
        "mean_outage_count": float(np.mean(all_outages)),
        "eval_time_s": elapsed,
    }

    save_dir = RESULTS_DIR / f"R007_heuristic_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Rate: {metrics['mean_sum_rate_mbps']:.1f} Mbps")
    print(f"  Outage: {metrics['mean_outage_count']:.1f}")
    print(f"  Time: {elapsed:.1f}s")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="M1 Baseline Experiments")
    parser.add_argument("--run", choices=["ppo", "heuristic", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer steps")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 10_000
        global EPOCHS_PER_REGIME
        EPOCHS_PER_REGIME = 100

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.run in ("ppo", "all"):
        run_ppo_baseline(seed=args.seed, total_timesteps=args.timesteps)
    if args.run in ("heuristic", "all"):
        run_heuristic_baseline(seed=args.seed)

    print("\n=== M1 Baselines Complete ===")


if __name__ == "__main__":
    main()
