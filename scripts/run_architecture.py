#!/usr/bin/env python3
"""M4: Architecture stress tests — rapid regime oscillation and novel regimes.

Tests robustness of the two-timescale method under adversarial conditions.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_main import run_two_timescale


def run_rapid_oscillation(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Stress test: regimes change every 100 epochs instead of 1000."""
    if verbose:
        print(f"  Rapid oscillation: seed={seed}, 100 epochs/regime")
    return run_two_timescale(
        seed=seed, timesteps=timesteps, epochs_per_regime=100,
        save_dir=save_dir, verbose=verbose,
    )


def run_novel_regime(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Stress test: include a 'novel' regime not in the standard set."""
    if verbose:
        print(f"  Novel regime: seed={seed}")

    # Train on standard regimes first, then encounter novel
    from agents.ppo_agent import PPOAgent, evaluate_agent
    from llm.architect import LLMMDPArchitect
    from llm.regime_detector import CUSUMDetector
    from simulator.env import BeamAllocationEnv, FlatActionWrapper

    LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
    LLM_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
    LLM_MODEL = os.environ.get("LLM_MODEL", "glm-5")

    # Phase 1: train on known regimes
    known_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=["urban", "maritime"], epochs_per_regime=200, seed=seed))
    agent = PPOAgent(known_env, device="cpu", seed=seed, verbose=0)
    agent.train(total_timesteps=timesteps // 2)

    if verbose:
        print(f"    Phase 1 done: trained on urban+maritime")

    # Phase 2: encounter novel regime (disaster — never seen)
    novel_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=["disaster", "mixed"], epochs_per_regime=200, seed=seed + 500))

    architect = LLMMDPArchitect(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)

    # Evaluate on novel regime WITHOUT adaptation
    no_adapt_metrics = evaluate_agent(agent, novel_env, n_episodes=2)
    if verbose:
        print(f"    No-adapt on novel: rate={no_adapt_metrics['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={no_adapt_metrics['mean_outage_count']:.1f}")

    # Now adapt with LLM
    agent_adapt = PPOAgent(novel_env, device="cpu", seed=seed, verbose=0)
    # Copy weights from trained agent
    agent_adapt.model.set_parameters(agent.model.get_parameters())

    segment = timesteps // 8
    total_trained = 0
    switch_log = []
    t0 = time.time()

    while total_trained < timesteps // 2:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=["disaster", "mixed"], epochs_per_regime=50, seed=seed + total_trained + 1000))
        m = evaluate_agent(agent_adapt, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]

        regime_changed = False
        for kpi in m.get("kpi_history", []):
            if detector.update(kpi):
                regime_changed = True
                break

        if regime_changed:
            last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}
            rich_kpi = {
                "avg_demand": last_kpi.get("avg_demand", rate),
                "demand_variance": last_kpi.get("demand_variance", 0),
                "spatial_gini": last_kpi.get("spatial_gini", 0.3),
                "peak_beam_demand": last_kpi.get("peak_beam_demand", rate),
                "mean_sum_rate_mbps": rate,
                "outage_rate": m["mean_outage_count"],
                "num_beams": 19, "max_active_beams": 10,
                "context": "Novel regime encountered — disaster/emergency scenario never seen during training",
            }
            spec = architect.generate_full_spec(rich_kpi)
            if spec and spec.reward_components:
                new_weights = {}
                for rc in spec.reward_components:
                    name_map = {"sum_rate": "sum_rate", "throughput": "sum_rate",
                                "outage_penalty": "outage", "outage": "outage",
                                "switching_cost": "switching", "switching": "switching",
                                "queue_penalty": "queue", "queue": "queue",
                                "fairness": "fairness"}
                    mapped = name_map.get(rc.name)
                    if mapped:
                        new_weights[mapped] = abs(rc.weight)  # ALL positive — env subtracts penalties
                if new_weights:
                    novel_env.unwrapped.update_reward_weights(new_weights)
                    switch_log.append({"step": total_trained, "weights": new_weights})
                    if verbose:
                        print(f"    Step {total_trained}: adapted MDP for novel regime")

        steps = min(segment, timesteps // 2 - total_trained)
        agent_adapt.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0

    # Final eval with adaptation
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=["disaster", "mixed"], epochs_per_regime=200, seed=seed + 9999))
    adapt_metrics = evaluate_agent(agent_adapt, eval_env, n_episodes=3)

    result = {
        "no_adapt": {k: v for k, v in no_adapt_metrics.items()},
        "with_adapt": {k: v for k, v in adapt_metrics.items()},
        "improvement_rate_mbps": adapt_metrics["mean_sum_rate_mbps"] - no_adapt_metrics["mean_sum_rate_mbps"],
        "improvement_outage": no_adapt_metrics["mean_outage_count"] - adapt_metrics["mean_outage_count"],
        "train_time_s": train_time,
        "seed": seed,
        "mdp_switches": len(switch_log),
        "llm_stats": architect.get_stats(),
        "method": "novel_regime",
    }

    if verbose:
        print(f"  Adapted: rate={adapt_metrics['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={adapt_metrics['mean_outage_count']:.1f}")
        print(f"  Improvement: +{result['improvement_rate_mbps']:.1f}Mbps rate, "
              f"-{result['improvement_outage']:.1f} outage")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["rapid", "novel", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    results_base = Path(__file__).parent.parent / "results"

    if args.run in ("rapid", "all"):
        print(f"\n=== M4: Rapid Oscillation seed={args.seed} ===")
        run_rapid_oscillation(seed=args.seed, timesteps=args.timesteps,
                              save_dir=str(results_base / f"R015_rapid_oscillation_seed{args.seed}"))

    if args.run in ("novel", "all"):
        print(f"\n=== M4: Novel Regime seed={args.seed} ===")
        run_novel_regime(seed=args.seed, timesteps=args.timesteps,
                         save_dir=str(results_base / f"R016_novel_regime_seed{args.seed}"))

    print("\n=== M4 Complete ===")


if __name__ == "__main__":
    main()
