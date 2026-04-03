#!/usr/bin/env python3
"""Path C: PPO Stability via Smooth Weight Transitions.

Hypothesis: frequent instant reward weight changes (MDP switching) destabilize
PPO's value function, causing high cross-seed variance and low mean performance.

Fix: interpolate weights smoothly over `smooth_steps` when regime changes,
giving PPO time to adapt its value estimate before the reward fully changes.

Also uses probe-derived optimal weights (from Path B probe results) to
initialize regime-specific weights, combining B+C approaches.

Usage:
    python scripts/run_path_c.py --seed 42 --smooth-steps 2000
    python scripts/run_path_c.py --seed 42 --smooth-steps 0   # instant (baseline)
    python scripts/run_path_c.py --seeds 42,123,456,789 --smooth-steps 2000
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, evaluate_agent, SmoothRegimeWeightSwitcher, RegimeWeightSwitcher
from llm.mlp_architect import MLPArchitect, load_mlp, WEIGHT_KEYS, KPI_KEYS
from simulator.env import BeamAllocationEnv, FlatActionWrapper

KNOWN_REGIMES = ["urban", "maritime", "disaster", "mixed"]
NOVEL_REGIMES = ["iot_burst", "polar_handover", "hot_cold"]
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Probe-derived optimal weights for novel regimes (from probe_round1.json analysis)
# Derived by taking expert hint baseline and applying positive gradient signals > +10 Mbps
PROBE_OPTIMAL_WEIGHTS = {
    "iot_burst": {
        "sum_rate": 0.6,   # +20% from 0.5: delta_rate=+35.2
        "fairness": 0.6,
        "outage": 1.5,
        "switching": 0.1,
        "queue": 0.8,
    },
    "polar_handover": {
        "sum_rate": 0.8,
        "fairness": 0.3,
        "outage": 1.2,     # +20% from 1.0: delta_rate=+20.8
        "switching": 0.96, # +20% from 0.8: delta_rate=+66.5 (strongest signal)
        "queue": 0.24,     # +20% from 0.2: delta_rate=+20.2
    },
    "hot_cold": {
        "sum_rate": 1.2,   # +20% from 1.0: delta_rate=+30.8
        "fairness": 0.6,   # +20% from 0.5: delta_rate=+15.6
        "outage": 1.2,
        "switching": 0.06, # +20% from 0.05: delta_rate=+128.4 (STRONGEST signal)
        "queue": 0.36,     # +20% from 0.3: delta_rate=+33.6
    },
}

# Fallback: use probe-optimal for novel, standard expert for known
KNOWN_EXPERT_WEIGHTS = {
    "urban":     {"sum_rate": 1.0, "fairness": 0.0, "outage": 1.2, "switching": 0.05, "queue": 0.0},
    "maritime":  {"sum_rate": 1.0, "fairness": 0.0, "outage": 1.5, "switching": 0.1,  "queue": 0.0},
    "disaster":  {"sum_rate": 0.5, "fairness": 2.0, "outage": 3.0, "switching": 0.05, "queue": 0.5},
    "mixed":     {"sum_rate": 1.0, "fairness": 0.5, "outage": 1.0, "switching": 0.05, "queue": 0.2},
}
ALL_WEIGHTS = {**KNOWN_EXPERT_WEIGHTS, **PROBE_OPTIMAL_WEIGHTS}


def run_path_c_experiment(
    seed: int = 42,
    known_steps: int = 250_000,
    novel_steps: int = 250_000,
    smooth_steps: int = 100,
    min_switch_interval: int = 1000,
    use_mlp: bool = False,
    oracle_mlp_dir: str = "models/oracle_mlp",
    save_dir: str = None,
    verbose: int = 1,
) -> dict:
    """Run Path C smooth-transition experiment.

    Args:
        seed: random seed
        known_steps: Phase 1 training steps
        novel_steps: Phase 2 training steps
        smooth_steps: weight interpolation window per switch (0 = instant).
                      Note: regime switches every ~198 steps; smooth_steps should
                      be less than this (e.g., 100) to complete before next switch.
        min_switch_interval: minimum steps between accepted switches (throttle).
                             1000 = only switch at most once per 1000 steps,
                             giving PPO stable reward for ~5× longer per regime.
                             0 = no throttling (accept every regime change).
        use_mlp: if True, use oracle_mlp predictions instead of probe-optimal weights
        oracle_mlp_dir: path to oracle MLP model (only used if use_mlp=True)
        save_dir: directory to save results
        verbose: verbosity

    Returns:
        metrics dict
    """
    t0 = time.time()

    if save_dir is None:
        mode_parts = []
        if min_switch_interval > 0:
            mode_parts.append(f"throttle{min_switch_interval}")
        if smooth_steps > 0:
            mode_parts.append(f"smooth{smooth_steps}")
        if not mode_parts:
            mode_parts.append("instant")
        mode = "_".join(mode_parts)
        tag = "mlp" if use_mlp else "probe"
        save_dir = str(RESULTS_DIR / f"PATHC_{mode}_{tag}_seed{seed}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        mode_str = []
        if min_switch_interval > 0:
            mode_str.append(f"throttle({min_switch_interval})")
        if smooth_steps > 0:
            mode_str.append(f"smooth({smooth_steps})")
        if not mode_str:
            mode_str.append("instant")
        weight_src = "oracle_mlp" if use_mlp else "probe_optimal"
        print(f"\n{'='*60}")
        print(f"PATH C: {'+'.join(mode_str)}, weights={weight_src}, seed={seed}")
        print(f"{'='*60}")

    # Load oracle MLP if needed
    mlp = None
    if use_mlp:
        mlp = load_mlp(oracle_mlp_dir)
        if verbose:
            print(f"  Loaded oracle MLP from {oracle_mlp_dir}")

    def get_weights(regime: str) -> dict:
        if mlp is not None:
            # Use MLP prediction with a dummy KPI for the regime
            dummy_kpi = {k: 0.5 for k in KPI_KEYS}
            return mlp.predict_weights(dummy_kpi)
        return ALL_WEIGHTS.get(regime, KNOWN_EXPERT_WEIGHTS.get(regime))

    # --- Phase 1: Known regimes ---
    if verbose:
        print(f"\n  Phase 1: Known regimes ({known_steps} steps)")

    env_known = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=200, seed=seed))
    agent = PPOAgent(env_known, device="cpu", seed=seed, verbose=0)

    # Set initial weights
    init_weights = get_weights(KNOWN_REGIMES[0])
    if init_weights:
        env_known.unwrapped.update_reward_weights(init_weights)

    # Create switcher for Phase 1
    if smooth_steps > 0 or min_switch_interval > 0:
        switcher1 = SmoothRegimeWeightSwitcher(
            weight_fn=get_weights,
            smooth_steps=smooth_steps,
            min_switch_interval=min_switch_interval,
            verbose=verbose)
    else:
        switcher1 = RegimeWeightSwitcher(weight_fn=get_weights, verbose=verbose)

    agent.train(total_timesteps=known_steps, callback=switcher1)

    eval_env_k = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=KNOWN_REGIMES, epochs_per_regime=50, seed=seed + 8888))
    if init_weights:
        eval_env_k.unwrapped.update_reward_weights(init_weights)
    known_eval = evaluate_agent(agent, eval_env_k, n_episodes=3)

    if verbose:
        print(f"  Phase 1 done: rate={known_eval['mean_sum_rate_mbps']:.1f} "
              f"outage={known_eval['mean_outage_count']:.2f} "
              f"switches={switcher1.switch_count}")

    # --- Phase 2: Novel regimes ---
    if verbose:
        print(f"\n  Phase 2: Novel regimes ({novel_steps} steps)")

    env_novel = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=200, seed=seed + 5000))
    agent_novel = PPOAgent(env_novel, device="cpu", seed=seed, verbose=0)

    # Warm-start from Phase 1 policy
    agent_novel.set_policy_params(agent.get_policy_params())

    # Set initial novel weights
    init_novel_weights = get_weights(NOVEL_REGIMES[0])
    if init_novel_weights:
        env_novel.unwrapped.update_reward_weights(init_novel_weights)

    # Create switcher for Phase 2
    if smooth_steps > 0 or min_switch_interval > 0:
        switcher2 = SmoothRegimeWeightSwitcher(
            weight_fn=get_weights,
            smooth_steps=smooth_steps,
            min_switch_interval=min_switch_interval,
            verbose=verbose)
    else:
        switcher2 = RegimeWeightSwitcher(weight_fn=get_weights, verbose=verbose)

    agent_novel.train(total_timesteps=novel_steps, callback=switcher2)

    eval_env_n = FlatActionWrapper(BeamAllocationEnv(
        regime_sequence=NOVEL_REGIMES, epochs_per_regime=50, seed=seed + 9999))
    if init_novel_weights:
        eval_env_n.unwrapped.update_reward_weights(init_novel_weights)
    novel_eval = evaluate_agent(agent_novel, eval_env_n, n_episodes=3)

    if verbose:
        print(f"  Phase 2 done: rate={novel_eval['mean_sum_rate_mbps']:.1f} "
              f"outage={novel_eval['mean_outage_count']:.2f} "
              f"switches={switcher2.switch_count}")

    total_time = time.time() - t0

    metrics = {
        "method": "path_c",
        "smooth_steps": smooth_steps,
        "min_switch_interval": min_switch_interval,
        "weight_source": "oracle_mlp" if use_mlp else "probe_optimal",
        "seed": seed,
        "known_rate": known_eval["mean_sum_rate_mbps"],
        "known_outage": known_eval["mean_outage_count"],
        "known_fairness": known_eval.get("mean_fairness_index", 0.0),
        "novel_rate": novel_eval["mean_sum_rate_mbps"],
        "novel_outage": novel_eval["mean_outage_count"],
        "novel_fairness": novel_eval.get("mean_fairness_index", 0.0),
        "phase1_switches": switcher1.switch_count,
        "phase2_switches": switcher2.switch_count,
        "train_time_s": round(total_time, 1),
    }

    # Save metrics
    with open(Path(save_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"\n  Metrics saved to {save_dir}/metrics.json")
        print(f"  known_rate={metrics['known_rate']:.1f} novel_rate={metrics['novel_rate']:.1f} "
              f"time={total_time:.0f}s")

    return metrics


def run_multiseed(
    seeds: list,
    smooth_steps: int = 100,
    min_switch_interval: int = 1000,
    known_steps: int = 250_000,
    novel_steps: int = 250_000,
    use_mlp: bool = False,
    oracle_mlp_dir: str = "models/oracle_mlp",
    verbose: int = 1,
) -> dict:
    """Run Path C across multiple seeds and compute statistics."""
    all_known = []
    all_novel = []
    results_per_seed = []

    for seed in seeds:
        print(f"\n{'='*40}")
        print(f"SEED {seed}")
        print(f"{'='*40}")
        m = run_path_c_experiment(
            seed=seed,
            known_steps=known_steps,
            novel_steps=novel_steps,
            smooth_steps=smooth_steps,
            min_switch_interval=min_switch_interval,
            use_mlp=use_mlp,
            oracle_mlp_dir=oracle_mlp_dir,
            verbose=verbose,
        )
        all_known.append(m["known_rate"])
        all_novel.append(m["novel_rate"])
        results_per_seed.append(m)
        print(f"  seed={seed}: known={m['known_rate']:.1f} novel={m['novel_rate']:.1f}")

    summary = {
        "smooth_steps": smooth_steps,
        "min_switch_interval": min_switch_interval,
        "weight_source": "oracle_mlp" if use_mlp else "probe_optimal",
        "seeds": seeds,
        "known_rates": all_known,
        "novel_rates": all_novel,
        "known_mean": float(np.mean(all_known)),
        "known_std": float(np.std(all_known)),
        "novel_mean": float(np.mean(all_novel)),
        "novel_std": float(np.std(all_novel)),
        "novel_median": float(np.median(all_novel)),
        "per_seed": results_per_seed,
    }

    # Save multi-seed summary
    tag = "mlp" if use_mlp else "probe"
    out_path = RESULTS_DIR / f"pathc_throttle{min_switch_interval}_smooth{smooth_steps}_{tag}_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PATH C MULTI-SEED SUMMARY")
    print(f"  smooth_steps={smooth_steps} min_switch_interval={min_switch_interval} weights={summary['weight_source']}")
    print(f"  known: mean={summary['known_mean']:.1f} ± {summary['known_std']:.1f}")
    print(f"  novel: mean={summary['novel_mean']:.1f} ± {summary['novel_std']:.1f} (median={summary['novel_median']:.1f})")
    print(f"  Saved to {out_path}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Path C: Smooth Weight Transition Experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed run (e.g. 42,123,456)")
    parser.add_argument("--smooth-steps", type=int, default=100,
                        help="Weight interpolation window per switch. 0=instant. "
                             "Should be < switch interval (~198 steps for epochs_per_regime=200).")
    parser.add_argument("--min-switch-interval", type=int, default=1000,
                        help="Minimum steps between accepted regime switches (throttle). "
                             "0=no throttling. 1000=5x fewer switches than baseline.")
    parser.add_argument("--known-steps", type=int, default=250_000)
    parser.add_argument("--novel-steps", type=int, default=250_000)
    parser.add_argument("--use-mlp", action="store_true",
                        help="Use oracle_mlp predictions instead of probe-optimal weights")
    parser.add_argument("--oracle-mlp-dir", type=str, default="models/oracle_mlp")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        run_multiseed(
            seeds=seeds,
            smooth_steps=args.smooth_steps,
            min_switch_interval=args.min_switch_interval,
            known_steps=args.known_steps,
            novel_steps=args.novel_steps,
            use_mlp=args.use_mlp,
            oracle_mlp_dir=args.oracle_mlp_dir,
            verbose=args.verbose,
        )
    else:
        run_path_c_experiment(
            seed=args.seed,
            known_steps=args.known_steps,
            novel_steps=args.novel_steps,
            smooth_steps=args.smooth_steps,
            min_switch_interval=args.min_switch_interval,
            use_mlp=args.use_mlp,
            oracle_mlp_dir=args.oracle_mlp_dir,
            verbose=args.verbose,
        )
