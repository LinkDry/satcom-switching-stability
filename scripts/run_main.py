#!/usr/bin/env python3
"""M2: Main method — Two-timescale LLM MDP Architect + DRL.

The full proposed system: regime detection triggers LLM to redesign
reward weights, then DRL warm-starts under the new MDP.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent
from llm.architect import LLMMDPArchitect
from llm.regime_detector import CUSUMDetector
from simulator.env import BeamAllocationEnv, FlatActionWrapper


LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
LLM_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-5")


def run_two_timescale(seed=42, timesteps=500_000, epochs_per_regime=200, save_dir=None, verbose=1):
    """Run the full two-timescale method."""
    regime_sequence = ["urban", "maritime", "disaster", "mixed"]

    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=regime_sequence,
        epochs_per_regime=epochs_per_regime,
        seed=seed,
    ))

    architect = LLMMDPArchitect(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.2,
    )
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)

    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    # Train in segments, checking for regime changes between segments
    segment_steps = timesteps // (len(regime_sequence) * 2)  # ~8 segments
    total_trained = 0
    all_rates = []
    all_outages = []
    switch_log = []
    current_regime = None

    t0 = time.time()

    if verbose:
        print(f"M2 Two-timescale: seed={seed}, {timesteps} steps, {len(regime_sequence)} regimes")

    # Initial training segment
    agent.train(total_timesteps=segment_steps)
    total_trained += segment_steps

    while total_trained < timesteps:
        # Evaluate current performance
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=regime_sequence,
            epochs_per_regime=epochs_per_regime // 4,
            seed=seed + total_trained,
        ))
        metrics = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = metrics["mean_sum_rate_mbps"]
        outage = metrics["mean_outage_count"]
        all_rates.append(rate)
        all_outages.append(outage)

        # Feed per-step KPI dicts to detector
        regime_changed = False
        for kpi in metrics.get("kpi_history", []):
            if detector.update(kpi):
                regime_changed = True
                break

        if regime_changed:
            if verbose:
                print(f"  Step {total_trained}: Regime change detected (rate={rate:.1f})")

            # Build rich KPI summary from last kpi_history entry
            last_kpi = metrics.get("kpi_history", [{}])[-1] if metrics.get("kpi_history") else {}
            kpi_summary = {
                "avg_demand": last_kpi.get("avg_demand", rate),
                "demand_variance": last_kpi.get("demand_variance", 0),
                "spatial_gini": last_kpi.get("spatial_gini", 0.3),
                "peak_beam_demand": last_kpi.get("peak_beam_demand", rate),
                "active_beam_fraction": last_kpi.get("active_beam_fraction", 1.0),
                "mean_sum_rate_mbps": rate,
                "outage_rate": outage,
                "num_beams": 19,
                "max_active_beams": 10,
                "recent_rate_trend": "declining" if len(all_rates) > 1 and rate < all_rates[-2] else "stable",
            }
            # Add regime hint
            if last_kpi.get("peak_beam_demand", 0) > 120:
                kpi_summary["regime_hint"] = "disaster-like (high peak demand)"
            elif last_kpi.get("avg_demand", 0) > 40 and last_kpi.get("spatial_gini", 0) > 0.3:
                kpi_summary["regime_hint"] = "urban-like (high concentrated demand)"
            elif last_kpi.get("avg_demand", 999) < 20:
                kpi_summary["regime_hint"] = "maritime-like (low uniform demand)"
            else:
                kpi_summary["regime_hint"] = "mixed/transition"

            spec = architect.generate_full_spec(kpi_summary)
            if spec and spec.reward_components:
                # Extract reward weights from LLM spec
                new_weights = {}
                for rc in spec.reward_components:
                    name_map = {
                        "sum_rate": "sum_rate", "throughput": "sum_rate",
                        "outage_penalty": "outage", "outage": "outage",
                        "switching_cost": "switching", "switching": "switching",
                        "queue_penalty": "queue", "queue": "queue",
                        "fairness": "fairness", "proportional_fairness": "fairness",
                    }
                    mapped = name_map.get(rc.name, None)
                    if mapped:
                        new_weights[mapped] = abs(rc.weight)  # ALL positive — env subtracts penalties

                if new_weights:
                    env.unwrapped.update_reward_weights(new_weights)
                    switch_log.append({
                        "step": total_trained,
                        "spec_id": spec.spec_id,
                        "new_weights": new_weights,
                        "trigger_rate": rate,
                    })
                    if verbose:
                        print(f"    MDP updated: {new_weights}")

        # Continue training
        steps_this_segment = min(segment_steps, timesteps - total_trained)
        agent.train(total_timesteps=steps_this_segment)
        total_trained += steps_this_segment

        if verbose and total_trained % (segment_steps * 2) == 0:
            print(f"  Progress: {total_trained}/{timesteps} steps, rate={rate:.1f}Mbps")

    train_time = time.time() - t0

    # Final evaluation
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=regime_sequence,
        epochs_per_regime=epochs_per_regime,
        seed=seed + 9999,
    ))
    final_metrics = evaluate_agent(agent, eval_env, n_episodes=3)

    result = {
        **final_metrics,
        "train_time_s": train_time,
        "seed": seed,
        "mdp_switches": len(switch_log),
        "llm_stats": architect.get_stats(),
        "method": "two_timescale_full",
    }

    if verbose:
        print(f"  Final: rate={final_metrics['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={final_metrics['mean_outage_count']:.1f} "
              f"switches={len(switch_log)} time={train_time:.0f}s")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        agent.save(str(save_path / "model"))
        with open(save_path / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(save_path / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    args = parser.parse_args()

    results_base = Path(__file__).parent.parent / "results"

    if args.all_seeds:
        for seed in [42, 123, 456]:
            print(f"\n=== M2: Two-timescale seed={seed} ===")
            save_dir = str(results_base / f"R009_two_timescale_seed{seed}")
            run_two_timescale(seed=seed, timesteps=args.timesteps, save_dir=save_dir)
    else:
        save_dir = args.save_dir or str(results_base / f"R009_two_timescale_seed{args.seed}")
        run_two_timescale(seed=args.seed, timesteps=args.timesteps, save_dir=save_dir)

    print("\n=== M2 Complete ===")


if __name__ == "__main__":
    main()
