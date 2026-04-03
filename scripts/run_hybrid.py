#!/usr/bin/env python3
"""Hybrid Architecture Experiment: LLM Intent + MLP Adaptation + DRL Scheduling.

Demonstrates the three-timescale design:
  1. Strategic (LLM): Parse operator NL commands → objective profiles
  2. Tactical (MLP): KPIs + objective → reward weights (1ms)
  3. Operational (DRL): PPO beam scheduling (real-time)

Key experiment: Same network conditions, different operator intents → different behavior.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent
from llm.intent_mlp import IntentAwareMLP, load_intent_mlp
from llm.intent_parser import LLMIntentParser, RuleIntentParser
from llm.operator_intent import ObjectiveProfile, INTENT_SCENARIOS
from llm.regime_detector import CUSUMDetector
from simulator.env import BeamAllocationEnv, FlatActionWrapper

REGIME_SEQ = ["urban", "maritime", "disaster", "mixed"]
WEIGHT_KEYS = ["sum_rate", "fairness", "outage", "switching", "queue"]


def _weights_changed(old, new, threshold=0.05):
    if old is None:
        return True
    return any(abs(old.get(k, 0) - new.get(k, 0)) > threshold for k in new)


def run_hybrid(seed=42, timesteps=500_000, intent_commands=None,
               use_llm=True, save_dir=None, verbose=1):
    """Run hybrid architecture experiment.

    intent_commands: list of (step, nl_command) tuples.
        Operator changes intent at specified steps.
    """
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))

    # Load intent-aware MLP
    mlp, kpi_mean, kpi_std = load_intent_mlp()

    # Intent parser (LLM or rule-based fallback)
    if use_llm:
        try:
            parser = LLMIntentParser()
        except Exception as e:
            if verbose:
                print(f"  LLM parser failed ({e}), using rule-based fallback")
            parser = RuleIntentParser()
    else:
        parser = RuleIntentParser()

    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    # Default intent commands if not provided
    if intent_commands is None:
        intent_commands = [
            (0, "Maximize network throughput for all users."),
            (timesteps // 4, "Emergency alert: prioritize disaster relief communications, minimize service outages at all costs."),
            (timesteps // 2, "Switch to fair resource sharing mode, ensure all maritime users get minimum guaranteed bandwidth."),
            (3 * timesteps // 4, "Resume normal operations, balance throughput and fairness."),
        ]

    # Parse initial intent
    current_intent_idx = 0
    current_command = intent_commands[0][1]
    current_profile = parser.parse(current_command)
    if verbose:
        print(f"  Intent[0]: '{current_command[:60]}...' → {current_profile.description}")

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    current_weights = None
    switch_log = []
    intent_log = [{"step": 0, "command": current_command, "profile": current_profile.__dict__}]

    t0 = time.time()
    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < timesteps:
        # Check for intent changes
        while (current_intent_idx + 1 < len(intent_commands) and
               intent_commands[current_intent_idx + 1][0] <= total_trained):
            current_intent_idx += 1
            current_command = intent_commands[current_intent_idx][1]
            current_profile = parser.parse(current_command)
            intent_log.append({
                "step": total_trained, "command": current_command,
                "profile": current_profile.__dict__
            })
            if verbose:
                print(f"  Intent[{current_intent_idx}] at step {total_trained}: '{current_command[:50]}...' → {current_profile.description}")

        # Evaluate and detect regime changes
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=REGIME_SEQ, epochs_per_regime=50,
            seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)
        rate = m["mean_sum_rate_mbps"]
        all_rates.append(rate)

        regime_changed = False
        for kpi in m.get("kpi_history", []):
            if detector.update(kpi):
                regime_changed = True

        if regime_changed:
            last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}

            # Intent-aware MLP: KPIs + objective profile → weights
            new_weights = mlp.predict_weights(last_kpi, current_profile)

            if _weights_changed(current_weights, new_weights):
                env.unwrapped.update_reward_weights(new_weights)
                current_weights = new_weights
                switch_log.append({
                    "step": total_trained,
                    "weights": {k: round(v, 4) for k, v in new_weights.items()},
                    "intent": current_profile.description,
                })
                if verbose:
                    print(f"    Step {total_trained}: [{current_profile.description}] → {new_weights}")

        if verbose and total_trained % (segment * 2) == 0:
            print(f"  Progress: {total_trained}/{timesteps} steps, rate={rate:.1f}Mbps")

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)

    result = {
        **final, "train_time_s": train_time, "seed": seed,
        "mdp_switches": len(switch_log), "method": "hybrid",
        "intent_changes": len(intent_log),
        "use_llm": use_llm,
    }

    if verbose:
        print(f"  Final: rate={final['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={final['mean_outage_count']:.2f} sw={len(switch_log)} "
              f"intents={len(intent_log)} time={train_time:.0f}s")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(Path(save_dir) / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)
        with open(Path(save_dir) / "intent_log.json", "w") as f:
            json.dump(intent_log, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))

    return result


def run_no_intent_baseline(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Baseline: MLP without intent awareness (original MLP architect)."""
    from llm.mlp_architect import MLPArchitect, load_mlp, WEIGHT_KEYS as MLP_WKEYS

    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    mlp = load_mlp()
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    segment = timesteps // 8
    total_trained = 0
    current_weights = None
    switch_log = []

    t0 = time.time()
    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < timesteps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=REGIME_SEQ, epochs_per_regime=50,
            seed=seed + total_trained))
        m = evaluate_agent(agent, eval_env, n_episodes=1)

        regime_changed = False
        for kpi in m.get("kpi_history", []):
            if detector.update(kpi):
                regime_changed = True

        if regime_changed:
            last_kpi = m.get("kpi_history", [{}])[-1] if m.get("kpi_history") else {}
            pred = mlp.predict_weights(last_kpi)
            new_weights = {k: float(pred[k]) for k in MLP_WKEYS}

            if _weights_changed(current_weights, new_weights):
                env.unwrapped.update_reward_weights(new_weights)
                current_weights = new_weights
                switch_log.append({"step": total_trained, "weights": new_weights})

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)

    result = {**final, "train_time_s": train_time, "seed": seed,
              "mdp_switches": len(switch_log), "method": "mlp_no_intent"}

    if verbose:
        print(f"  MLP (no intent): rate={final['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={final['mean_outage_count']:.2f} sw={len(switch_log)}")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_base = Path(__file__).parent.parent / "results"

    print("=== Hybrid Architecture Experiment ===\n")

    # Experiment 1: Hybrid with LLM intent parsing
    print("--- Hybrid (LLM intent + MLP + DRL) ---")
    r1 = run_hybrid(seed=args.seed, timesteps=args.timesteps, use_llm=True,
                    save_dir=str(results_base / f"hybrid_llm_seed{args.seed}"))

    # Experiment 2: Hybrid with rule-based intent (ablation)
    print("\n--- Hybrid (rule-based intent + MLP + DRL) ---")
    r2 = run_hybrid(seed=args.seed, timesteps=args.timesteps, use_llm=False,
                    save_dir=str(results_base / f"hybrid_rule_seed{args.seed}"))

    # Experiment 3: MLP without intent awareness (baseline)
    print("\n--- MLP only (no intent) ---")
    r3 = run_no_intent_baseline(seed=args.seed, timesteps=args.timesteps,
                                save_dir=str(results_base / f"mlp_no_intent_seed{args.seed}"))

    print("\n=== SUMMARY ===")
    print(f"Hybrid (LLM):  rate={r1['mean_sum_rate_mbps']:.1f} outage={r1['mean_outage_count']:.2f} sw={r1['mdp_switches']}")
    print(f"Hybrid (rule):  rate={r2['mean_sum_rate_mbps']:.1f} outage={r2['mean_outage_count']:.2f} sw={r2['mdp_switches']}")
    print(f"MLP (no intent): rate={r3['mean_sum_rate_mbps']:.1f} outage={r3['mean_outage_count']:.2f} sw={r3['mdp_switches']}")
