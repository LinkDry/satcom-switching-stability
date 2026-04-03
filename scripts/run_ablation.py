#!/usr/bin/env python3
"""M3: Ablation experiments — isolate contribution of each LLM component.

Variants:
- reward-only: LLM only changes reward weights (no state/action changes)
- rule-based: Rule-based MDP selector (no LLM at all)
- no-warmstart: Full method but cold-start DRL after each MDP change
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent
from agents.rule_selector import RuleBasedMDPSelector
from llm.architect import LLMMDPArchitect
from llm.regime_detector import CUSUMDetector
from simulator.env import BeamAllocationEnv, FlatActionWrapper

LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
LLM_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-5")

REGIME_SEQ = ["urban", "maritime", "disaster", "mixed"]


def run_reward_only(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Ablation: LLM only redesigns reward weights."""
    env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    architect = LLMMDPArchitect(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    switch_log = []
    t0 = time.time()

    if verbose:
        print(f"  Reward-only ablation: seed={seed}")

    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < timesteps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=50, seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        all_rates.append(rate)

        regime_changed = False
        for kpi in m.get("kpi_history", []):
            if detector.update(kpi):
                regime_changed = True
                break

        if regime_changed:
            # Build rich KPI from last kpi_history entry
            last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}
            rich_kpi = {
                "avg_demand": last_kpi.get("avg_demand", rate),
                "demand_variance": last_kpi.get("demand_variance", 0),
                "spatial_gini": last_kpi.get("spatial_gini", 0.3),
                "peak_beam_demand": last_kpi.get("peak_beam_demand", rate),
                "mean_sum_rate_mbps": rate,
                "outage_rate": m["mean_outage_count"],
                "num_beams": 19, "max_active_beams": 10,
            }
            reward_components = architect.generate_reward_only(
                rich_kpi, current_features=["queue_lengths", "channel_snr", "demand_current"],
                current_action="per_beam"
            )
            if reward_components:
                new_weights = {}
                for rc in reward_components:
                    if isinstance(rc, dict):
                        name = rc.get("name", "")
                        weight = rc.get("weight", 0)
                        name_map = {"sum_rate": "sum_rate", "throughput": "sum_rate", "outage_penalty": "outage", "outage": "outage",
                                    "switching_cost": "switching", "switching": "switching", "queue_penalty": "queue", "queue": "queue",
                                    "fairness": "fairness", "proportional_fairness": "fairness"}
                        mapped = name_map.get(name)
                        if mapped:
                            new_weights[mapped] = abs(weight) if mapped in ("sum_rate", "fairness") else -abs(weight)
                if new_weights:
                    env.unwrapped.update_reward_weights(new_weights)
                    switch_log.append({"step": total_trained, "weights": new_weights})
                    if verbose:
                        print(f"    Step {total_trained}: reward updated: {new_weights}")

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)
    result = {**final, "train_time_s": train_time, "seed": seed, "mdp_switches": len(switch_log),
              "llm_stats": architect.get_stats(), "method": "reward_only"}

    if verbose:
        print(f"  Result: rate={final['mean_sum_rate_mbps']:.1f}Mbps outage={final['mean_outage_count']:.1f}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))
    return result


def run_rule_based(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Ablation: Rule-based MDP selector (no LLM)."""
    env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    selector = RuleBasedMDPSelector()
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    switch_log = []
    t0 = time.time()

    if verbose:
        print(f"  Rule-based ablation: seed={seed}")

    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < timesteps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=50, seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        all_rates.append(rate)

        regime_changed = False
        for kpi in m.get("kpi_history", []):
            if detector.update(kpi):
                regime_changed = True
                break

        if regime_changed:
            # Use last KPI snapshot from eval (has avg_demand, peak_beam_demand, spatial_gini)
            last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {
                "avg_demand": rate, "peak_beam_demand": rate, "spatial_gini": 0.3
            }
            spec, changed = selector.select_spec(last_kpi)
            if changed:
                # Apply rule-based reward weights
                weights = {"sum_rate": 1.0, "outage": 1.0, "switching": 0.01, "queue": 0.0, "fairness": 0.0}
                if spec.spec_id == "maritime":
                    weights["fairness"] = 0.3
                elif spec.spec_id == "disaster":
                    weights["outage"] = 2.0
                    weights["queue"] = 0.1
                env.unwrapped.update_reward_weights(weights)
                switch_log.append({"step": total_trained, "spec": spec.spec_id})
                if verbose:
                    print(f"    Step {total_trained}: rule switch to {spec.spec_id}")

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)
    result = {**final, "train_time_s": train_time, "seed": seed, "mdp_switches": len(switch_log), "method": "rule_based"}

    if verbose:
        print(f"  Result: rate={final['mean_sum_rate_mbps']:.1f}Mbps outage={final['mean_outage_count']:.1f}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))
    return result


def run_no_warmstart(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Ablation: Full LLM method but cold-start DRL after each MDP change."""
    env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    architect = LLMMDPArchitect(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    switch_log = []
    t0 = time.time()

    # Fresh agent each time
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    if verbose:
        print(f"  No-warmstart ablation: seed={seed}")

    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < timesteps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=50, seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        all_rates.append(rate)

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
            }
            spec = architect.generate_full_spec(rich_kpi)
            if spec and spec.reward_components:
                new_weights = {}
                for rc in spec.reward_components:
                    name_map = {"sum_rate": "sum_rate", "throughput": "sum_rate", "outage_penalty": "outage", "outage": "outage",
                                "switching_cost": "switching", "switching": "switching", "queue_penalty": "queue", "queue": "queue",
                                "fairness": "fairness", "proportional_fairness": "fairness"}
                    mapped = name_map.get(rc.name)
                    if mapped:
                        new_weights[mapped] = abs(rc.weight)  # ALL positive — env subtracts penalties
                if new_weights:
                    env.unwrapped.update_reward_weights(new_weights)
                    # COLD START: create brand new agent
                    agent = PPOAgent(env, device="cpu", seed=seed + total_trained, verbose=0)
                    switch_log.append({"step": total_trained, "cold_restart": True})
                    if verbose:
                        print(f"    Step {total_trained}: cold restart with new MDP")

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)
    result = {**final, "train_time_s": train_time, "seed": seed, "mdp_switches": len(switch_log),
              "llm_stats": architect.get_stats(), "method": "no_warmstart"}

    if verbose:
        print(f"  Result: rate={final['mean_sum_rate_mbps']:.1f}Mbps outage={final['mean_outage_count']:.1f}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["reward_only", "rule_based", "no_warmstart", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=500_000)
    args = parser.parse_args()

    results_base = Path(__file__).parent.parent / "results"
    runners = {
        "reward_only": (run_reward_only, "R012_reward_only"),
        "rule_based": (run_rule_based, "R013_rule_based"),
        "no_warmstart": (run_no_warmstart, "R014_no_warmstart"),
    }

    targets = runners.keys() if args.run == "all" else [args.run]

    for name in targets:
        fn, prefix = runners[name]
        print(f"\n=== M3: {name} seed={args.seed} ===")
        save_dir = str(results_base / f"{prefix}_seed{args.seed}")
        fn(seed=args.seed, timesteps=args.timesteps, save_dir=save_dir)

    print("\n=== M3 Complete ===")


if __name__ == "__main__":
    main()
