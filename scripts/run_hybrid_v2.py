#!/usr/bin/env python3
"""Hybrid Architecture Experiment v2: With Intent Satisfaction Metrics.

Measures not just throughput, but whether each method fulfills operator intent.
This is the key experiment proving hybrid's value.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent
from llm.intent_parser import LLMIntentParser, RuleIntentParser
from llm.intent_mlp import load_intent_mlp
from llm.mlp_architect import load_mlp
from llm.regime_detector import CUSUMDetector
from llm.operator_intent import ObjectiveProfile
from evaluation.intent_metrics import IntentSatisfactionTracker, IntentMetrics, compute_intent_satisfaction
from simulator.env import BeamAllocationEnv, FlatActionWrapper

REGIME_SEQ = ["urban", "maritime", "disaster", "mixed"]

# Dynamic operator intent scenario: 4 phases, each 125k steps
INTENT_PHASES = [
    {
        "command": "Maximize network throughput and capacity for all users during peak hours.",
        "profile": ObjectiveProfile(0.9, 0.2, 0.3, 0.5, 0.3, "max_throughput"),
        "steps": 125_000,
    },
    {
        "command": "Emergency alert: natural disaster detected. Prioritize coverage and minimize service outages at all costs.",
        "profile": ObjectiveProfile(0.3, 0.5, 0.9, 0.2, 0.8, "emergency"),
        "steps": 125_000,
    },
    {
        "command": "Ensure fair resource distribution across all beams. No single region should be underserved.",
        "profile": ObjectiveProfile(0.4, 0.9, 0.4, 0.3, 0.6, "fairness"),
        "steps": 125_000,
    },
    {
        "command": "Optimize for energy efficiency. Minimize unnecessary beam switching while maintaining acceptable service.",
        "profile": ObjectiveProfile(0.5, 0.3, 0.3, 0.9, 0.4, "energy_saving"),
        "steps": 125_000,
    },
]


def _weights_changed(old, new, threshold=0.05):
    if old is None:
        return True
    return any(abs(old.get(k, 0) - new.get(k, 0)) > threshold for k in new)


def run_method(method_name, parser, mlp, kpi_mean, kpi_std, seed=42,
               use_intent=True, verbose=1):
    """Run one method through all intent phases, collecting per-phase metrics."""
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    evaluator = IntentSatisfactionTracker()

    segment = 6250
    total_trained = 0
    current_weights = None
    switch_log = []
    phase_results = []

    # Default profile for non-intent methods
    default_profile = ObjectiveProfile(0.5, 0.3, 0.3, 0.5, 0.5, "default")

    if verbose:
        print(f"\n--- {method_name} ---")

    for phase_idx, phase in enumerate(INTENT_PHASES):
        phase_steps = phase["steps"]
        phase_trained = 0

        # Parse intent (only hybrid uses this)
        if use_intent and parser:
            current_profile = parser.parse(phase["command"])
        else:
            current_profile = default_profile

        if verbose:
            print(f"  Phase {phase_idx}: '{phase['command'][:50]}...' → {current_profile.description}")

        # Collect per-phase episode data
        phase_rates = []
        phase_outages = []
        phase_fairness = []
        phase_switches = 0

        while phase_trained < phase_steps:
            # Evaluate
            eval_env = FlatActionWrapper(BeamAllocationEnv(
                regime_sequence=REGIME_SEQ, epochs_per_regime=50,
                seed=seed + total_trained))
            m = evaluate_agent(agent, eval_env, n_episodes=1)
            rate = m["mean_sum_rate_mbps"]
            phase_rates.append(rate)
            phase_outages.append(m.get("mean_outage_count", 0))

            # Use fairness from evaluate_agent (computed from per-beam rates)
            jain = m.get("mean_fairness_index", 0.5)
            phase_fairness.append(jain)

            # Detect regime change and update weights
            regime_changed = detector.update(m.get("kpi_history", [{}])[-1])
            if regime_changed:
                last_kpi = m.get("kpi_history", [{}])[-1]
                if use_intent:
                    new_weights = mlp.predict_weights(last_kpi, current_profile, kpi_mean, kpi_std)
                else:
                    new_weights = mlp.predict_weights(last_kpi, default_profile, kpi_mean, kpi_std)

                if _weights_changed(current_weights, new_weights):
                    env.unwrapped.update_reward_weights(new_weights)
                    current_weights = new_weights
                    phase_switches += 1
                    switch_log.append({
                        "step": total_trained, "phase": phase_idx,
                        "intent": current_profile.description,
                        "weights": {k: round(v, 4) for k, v in new_weights.items()},
                    })

            steps = min(segment, phase_steps - phase_trained)
            agent.train(total_timesteps=steps)
            total_trained += steps
            phase_trained += steps

        # Evaluate intent satisfaction for this phase
        import numpy as np
        phase_info = {
            "mean_rate": float(np.mean(phase_rates)) if phase_rates else 0,
            "mean_outage": float(np.mean(phase_outages)) if phase_outages else 0,
            "mean_fairness": float(np.mean(phase_fairness)) if phase_fairness else 0,
            "switches": phase_switches,
        }

        metrics = IntentMetrics(
            sum_rate=phase_info["mean_rate"],
            outage_count=phase_info["mean_outage"],
            fairness_index=phase_info["mean_fairness"],
        )
        evaluator.record(phase["profile"].description, metrics)
        satisfaction = compute_intent_satisfaction(metrics, phase["profile"].description)

        phase_results.append({
            "phase": phase_idx,
            "intent": phase["profile"].description,
            "parsed_as": current_profile.description,
            **phase_info,
            "satisfaction": satisfaction,
        })

        if verbose:
            print(f"    rate={phase_info['mean_rate']:.1f} outage={phase_info['mean_outage']:.2f} "
                  f"fairness={phase_info['mean_fairness']:.2f} satisfaction={satisfaction:.2f}")

    # Final evaluation
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)

    import numpy as np
    avg_satisfaction = float(np.mean([p["satisfaction"] for p in phase_results]))

    return {
        **final,
        "method": method_name,
        "seed": seed,
        "phase_results": phase_results,
        "avg_satisfaction": avg_satisfaction,
        "mdp_switches": len(switch_log),
        "switch_log": switch_log,
    }


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Comma-separated seeds")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    intent_mlp, kpi_mean, kpi_std = load_intent_mlp()

    print("=== Hybrid Architecture v2: Intent Satisfaction (Multi-Seed) ===\n")

    all_results = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        llm_parser = LLMIntentParser()
        r1 = run_method("Hybrid-LLM", llm_parser, intent_mlp, kpi_mean, kpi_std,
                         seed=seed, use_intent=True)
        all_results.append(r1)

        rule_parser = RuleIntentParser()
        r2 = run_method("Hybrid-Rule", rule_parser, intent_mlp, kpi_mean, kpi_std,
                         seed=seed, use_intent=True)
        all_results.append(r2)

        r3 = run_method("MLP-NoIntent", None, intent_mlp, kpi_mean, kpi_std,
                         seed=seed, use_intent=False)
        all_results.append(r3)

    # Per-seed summary
    print("\n=== PER-SEED SUMMARY ===")
    print(f"{'Method':<15} {'Seed':>6} {'Rate':>8} {'Outage':>8} {'Satisfaction':>12} {'Switches':>8}")
    for r in all_results:
        print(f"{r['method']:<15} {r.get('seed', '?'):>6} {r['mean_sum_rate_mbps']:>8.1f} "
              f"{r['mean_outage_count']:>8.2f} {r['avg_satisfaction']:>12.2f} "
              f"{r['mdp_switches']:>8}")

    # Aggregate means
    print("\n=== AGGREGATE (mean +/- std) ===")
    methods = ["Hybrid-LLM", "Hybrid-Rule", "MLP-NoIntent"]
    print(f"{'Method':<15} {'Rate':>14} {'Satisfaction':>14} {'Outage':>14}")
    for method in methods:
        mrs = [r for r in all_results if r["method"] == method]
        if not mrs:
            continue
        rates = np.array([r["mean_sum_rate_mbps"] for r in mrs])
        sats = np.array([r["avg_satisfaction"] for r in mrs])
        outs = np.array([r["mean_outage_count"] for r in mrs])
        print(f"{method:<15} {rates.mean():>6.1f}+/-{rates.std():>4.1f} "
              f"{sats.mean():>6.2f}+/-{sats.std():>4.2f} "
              f"{outs.mean():>6.2f}+/-{outs.std():>4.2f}")

    # Save results
    results_dir = Path(__file__).parent.parent / "results" / "hybrid_v2"
    results_dir.mkdir(exist_ok=True)
    for r in all_results:
        seed_val = r.get("seed", "unknown")
        with open(results_dir / f"{r['method']}_seed{seed_val}.json", "w") as f:
            json.dump(r, f, indent=2, default=str)
    with open(results_dir / "aggregate.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {results_dir}")
