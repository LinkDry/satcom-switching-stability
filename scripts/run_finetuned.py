#!/usr/bin/env python3
"""Run fine-tuned Qwen3-4B experiment — SAME structure as run_mlp.py."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent
from llm.finetuned_architect import FinetunedLLMArchitect
from llm.regime_detector import CUSUMDetector
from simulator.env import BeamAllocationEnv, FlatActionWrapper

REGIME_SEQ = ["urban", "maritime", "disaster", "mixed"]


def _weights_changed(old, new, threshold=0.05):
    """Check if weights changed significantly."""
    for k in new:
        if k not in old or abs(old[k] - new[k]) > threshold:
            return True
    return False


def run_finetuned(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Fine-tuned LLM architect: CUSUM detects change → LLM predicts weights."""
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    arch = FinetunedLLMArchitect()

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    switch_log = []
    current_weights = None
    t0 = time.time()

    if verbose:
        print(f"  Fine-tuned LLM: seed={seed}")

    agent.train(total_timesteps=segment)
    total_trained += segment

    while total_trained < timesteps:
        eval_env = FlatActionWrapper(BeamAllocationEnv(
            regime_sequence=REGIME_SEQ, epochs_per_regime=50, seed=seed + total_trained))
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
            new_weights = arch.predict_weights(last_kpi)

            if current_weights is None or _weights_changed(current_weights, new_weights):
                env.unwrapped.update_reward_weights(new_weights)
                current_weights = new_weights.copy()
                switch_log.append({
                    "step": total_trained,
                    "weights": {k: round(v, 4) for k, v in new_weights.items()},
                })
                if verbose:
                    print(f"    Step {total_trained}: LLM → {new_weights}")

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
        "mdp_switches": len(switch_log), "llm_calls": arch.call_count,
        "llm_latency_s": arch.total_latency,
        "method": "finetuned_qwen3_4b",
    }

    if verbose:
        print(f"  Final: rate={final['mean_sum_rate_mbps']:.1f}Mbps outage={final['mean_outage_count']:.2f} "
              f"sw={len(switch_log)} llm_calls={arch.call_count} time={train_time:.0f}s")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(Path(save_dir) / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_base = Path(__file__).parent.parent / "results"
    for seed in [42, 123, 456]:
        r = run_finetuned(seed=seed, timesteps=args.timesteps,
                          save_dir=str(results_base / f"FT_v2_seed{seed}"))
        print(f"FT seed={seed}: rate={r['mean_sum_rate_mbps']:.1f} outage={r['mean_outage_count']:.2f} sw={r['mdp_switches']}")
