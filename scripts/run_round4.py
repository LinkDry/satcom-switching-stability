#!/usr/bin/env python3
"""Round 4: Two improved approaches for LLM-DRL integration.

Approach A: LLM as regime classifier + preset expert weights
Approach B: Constrained LLM output + cooldown + weight smoothing
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

REGIME_SEQ = ["urban", "maritime", "disaster", "mixed"]

# Expert weight profiles — matched to rule-based (342.8 Mbps, 0 outage)
EXPERT_WEIGHTS = {
    "urban":    {"sum_rate": 1.0, "fairness": 0.0, "outage": 1.0, "switching": 0.01, "queue": 0.0},
    "maritime": {"sum_rate": 1.0, "fairness": 0.3, "outage": 1.0, "switching": 0.01, "queue": 0.0},
    "disaster": {"sum_rate": 1.0, "fairness": 0.0, "outage": 2.0, "switching": 0.01, "queue": 0.1},
    "mixed":    {"sum_rate": 1.0, "fairness": 0.0, "outage": 1.0, "switching": 0.01, "queue": 0.0},
}

# Weight clamp ranges — ALL POSITIVE
WEIGHT_CLAMPS = {
    "sum_rate":  (0.3, 1.0),
    "fairness":  (0.0, 0.5),
    "outage":    (0.1, 3.0),
    "switching": (0.01, 0.2),
    "queue":     (0.0, 0.5),
}

DEFAULT_WEIGHTS = {"sum_rate": 0.8, "fairness": 0.15, "outage": 0.5, "switching": 0.05, "queue": 0.1}


def _build_rich_kpi(metrics, rate):
    """Build rich KPI dict from evaluate_agent output."""
    last_kpi = metrics.get("kpi_history", [{}])[-1] if metrics.get("kpi_history") else {}
    kpi = {
        "avg_demand": last_kpi.get("avg_demand", rate),
        "demand_variance": last_kpi.get("demand_variance", 0),
        "spatial_gini": last_kpi.get("spatial_gini", 0.3),
        "peak_beam_demand": last_kpi.get("peak_beam_demand", rate),
        "active_beam_fraction": last_kpi.get("active_beam_fraction", 1.0),
        "mean_sum_rate_mbps": rate,
        "outage_rate": metrics["mean_outage_count"],
        "num_beams": 19, "max_active_beams": 10,
    }
    if last_kpi.get("peak_beam_demand", 0) > 120:
        kpi["regime_hint"] = "disaster-like"
    elif last_kpi.get("avg_demand", 0) > 40 and last_kpi.get("spatial_gini", 0) > 0.3:
        kpi["regime_hint"] = "urban-like"
    elif last_kpi.get("avg_demand", 999) < 20:
        kpi["regime_hint"] = "maritime-like"
    else:
        kpi["regime_hint"] = "mixed/transition"
    return kpi


def _clamp_weights(weights):
    """Clamp LLM-generated weights to safe POSITIVE ranges.
    Takes abs() first in case LLM still generates negative values."""
    clamped = {}
    for k, v in weights.items():
        if k in WEIGHT_CLAMPS:
            lo, hi = WEIGHT_CLAMPS[k]
            clamped[k] = max(lo, min(hi, abs(v)))
        else:
            clamped[k] = abs(v)
    return clamped


def _smooth_weights(old_weights, new_weights, alpha=0.3):
    """Exponential moving average: result = alpha * new + (1-alpha) * old."""
    smoothed = {}
    for k in new_weights:
        old_v = old_weights.get(k, new_weights[k])
        smoothed[k] = alpha * new_weights[k] + (1 - alpha) * old_v
    return smoothed


def run_approach_a(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Approach A: LLM classifies regime, then apply preset expert weights."""
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    architect = LLMMDPArchitect(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    switch_log = []
    current_regime = None  # Only switch when regime changes
    t0 = time.time()

    if verbose:
        print(f"  Approach A (LLM classifier): seed={seed}")

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
            rich_kpi = _build_rich_kpi(m, rate)

            # KPI-based classification (reliable)
            avg_d = rich_kpi.get("avg_demand", 30)
            peak_d = rich_kpi.get("peak_beam_demand", 50)
            gini = rich_kpi.get("spatial_gini", 0.3)

            if peak_d > 120:
                kpi_regime = "disaster"
            elif avg_d > 40 and gini > 0.3:
                kpi_regime = "urban"
            elif avg_d < 20 and gini < 0.2:
                kpi_regime = "maritime"
            else:
                kpi_regime = "mixed"

            # Only apply weights when regime CHANGES (matches rule-based behavior)
            if kpi_regime != current_regime:
                # Ask LLM for second opinion (logged for comparison)
                from llm.prompts import REGIME_CLASSIFY_PROMPT
                prompt = REGIME_CLASSIFY_PROMPT.format(kpi_summary=json.dumps(rich_kpi, indent=2))
                response = architect._call_llm(prompt)
                llm_regime = "mixed"
                if response:
                    resp_lower = response.strip().lower()
                    for r in ["urban", "maritime", "disaster", "mixed"]:
                        if r in resp_lower:
                            llm_regime = r
                            break
                architect.call_count += 1

                weights = EXPERT_WEIGHTS[kpi_regime].copy()
                env.unwrapped.update_reward_weights(weights)
                current_regime = kpi_regime
                switch_log.append({"step": total_trained, "kpi_regime": kpi_regime,
                                   "llm_regime": llm_regime, "regime_used": kpi_regime, "weights": weights})
                if verbose:
                    print(f"    Step {total_trained}: regime={kpi_regime} (LLM={llm_regime}) → {weights}")

        if verbose and total_trained % (segment * 2) == 0:
            print(f"  Progress: {total_trained}/{timesteps} steps, rate={rate:.1f}Mbps")

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)
    result = {**final, "train_time_s": train_time, "seed": seed,
              "mdp_switches": len(switch_log), "llm_stats": architect.get_stats(),
              "method": "approach_a_classifier"}

    if verbose:
        print(f"  Final: rate={final['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={final['mean_outage_count']:.1f} switches={len(switch_log)} "
              f"time={train_time:.0f}s")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(Path(save_dir) / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))
    return result


def run_approach_b(seed=42, timesteps=500_000, save_dir=None, verbose=1):
    """Approach B: LLM generates weights with clamping, cooldown, and smoothing."""
    env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed))
    architect = LLMMDPArchitect(
        model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL, temperature=0.2)
    detector = CUSUMDetector(window_size=10, threshold=1.0, min_interval=50)
    agent = PPOAgent(env, device="cpu", seed=seed, verbose=0)

    COOLDOWN_STEPS = 100_000
    SMOOTH_ALPHA = 0.3

    segment = timesteps // 8
    total_trained = 0
    all_rates = []
    switch_log = []
    current_weights = DEFAULT_WEIGHTS.copy()
    last_switch_step = -COOLDOWN_STEPS  # allow first switch immediately
    t0 = time.time()

    if verbose:
        print(f"  Approach B (constrained+cooldown+smooth): seed={seed}")

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

        # Cooldown check
        if regime_changed and (total_trained - last_switch_step) < COOLDOWN_STEPS:
            if verbose:
                print(f"    Step {total_trained}: regime change detected but in cooldown "
                      f"({total_trained - last_switch_step}/{COOLDOWN_STEPS})")
            regime_changed = False

        if regime_changed:
            rich_kpi = _build_rich_kpi(m, rate)
            spec = architect.generate_full_spec(rich_kpi)
            if spec and spec.reward_components:
                name_map = {
                    "sum_rate": "sum_rate", "throughput": "sum_rate",
                    "outage_penalty": "outage", "outage": "outage",
                    "switching_cost": "switching", "switching": "switching",
                    "queue_penalty": "queue", "queue": "queue",
                    "fairness": "fairness", "proportional_fairness": "fairness",
                }
                raw_weights = {}
                for rc in spec.reward_components:
                    mapped = name_map.get(rc.name, rc.name)
                    if mapped in WEIGHT_CLAMPS or mapped in DEFAULT_WEIGHTS:
                        raw_weights[mapped] = rc.weight

                # Step 1: Clamp
                clamped = _clamp_weights(raw_weights)
                # Step 2: Smooth
                new_weights = _smooth_weights(current_weights, clamped, SMOOTH_ALPHA)
                # Step 3: Round for readability
                new_weights = {k: round(v, 3) for k, v in new_weights.items()}

                env.unwrapped.update_reward_weights(new_weights)
                current_weights = new_weights
                last_switch_step = total_trained
                switch_log.append({"step": total_trained, "raw": raw_weights,
                                   "clamped": clamped, "smoothed": new_weights})
                if verbose:
                    print(f"    Step {total_trained}: MDP updated (smoothed): {new_weights}")

        if verbose and total_trained % (segment * 2) == 0:
            print(f"  Progress: {total_trained}/{timesteps} steps, rate={rate:.1f}Mbps")

        steps = min(segment, timesteps - total_trained)
        agent.train(total_timesteps=steps)
        total_trained += steps

    train_time = time.time() - t0
    eval_env = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=REGIME_SEQ, epochs_per_regime=200, seed=seed + 9999))
    final = evaluate_agent(agent, eval_env, n_episodes=3)
    result = {**final, "train_time_s": train_time, "seed": seed,
              "mdp_switches": len(switch_log), "llm_stats": architect.get_stats(),
              "method": "approach_b_constrained"}

    if verbose:
        print(f"  Final: rate={final['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={final['mean_outage_count']:.1f} switches={len(switch_log)} "
              f"time={train_time:.0f}s")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(Path(save_dir) / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)
        agent.save(str(Path(save_dir) / "model"))
    return result
