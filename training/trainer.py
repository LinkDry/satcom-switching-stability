"""Two-timescale training loop and standard fixed-MDP trainer."""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from agents.ppo_agent import PPOAgent, evaluate_agent
from simulator.env import BeamAllocationEnv, FlatActionWrapper


def train_fixed_mdp(
    regime_sequence: list[str],
    epochs_per_regime: int = 1000,
    total_timesteps: int = 500_000,
    seed: int = 42,
    save_dir: Optional[str] = None,
    device: str = "cpu",
    verbose: int = 1,
) -> dict:
    """Train a standard PPO agent on a fixed MDP (no adaptation).

    This is the baseline training pipeline for R005-R008.
    """
    env = FlatActionWrapper(
        BeamAllocationEnv(
            regime_sequence=regime_sequence,
            epochs_per_regime=epochs_per_regime,
            seed=seed,
        )
    )

    agent = PPOAgent(env, device=device, seed=seed, verbose=verbose)

    if verbose:
        print(f"Training fixed-MDP PPO: {total_timesteps} steps, seed={seed}")
        print(f"  Regimes: {regime_sequence}, epochs_per_regime={epochs_per_regime}")

    t0 = time.time()
    agent.train(total_timesteps=total_timesteps)
    train_time = time.time() - t0

    if verbose:
        print(f"  Training completed in {train_time:.1f}s")

    # Evaluate
    eval_env = FlatActionWrapper(
        BeamAllocationEnv(
            regime_sequence=regime_sequence,
            epochs_per_regime=epochs_per_regime,
            seed=seed + 1000,
        )
    )
    metrics = evaluate_agent(agent, eval_env, n_episodes=3)

    if verbose:
        print(f"  Eval: rate={metrics['mean_sum_rate_mbps']:.1f}Mbps "
              f"outage={metrics['mean_outage_count']:.1f}")

    # Save
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        agent.save(str(save_path / "model"))
        with open(save_path / "metrics.json", "w") as f:
            json.dump({**metrics, "train_time_s": train_time, "seed": seed}, f, indent=2)

    return {
        "agent": agent,
        "metrics": metrics,
        "train_time_s": train_time,
    }


def train_two_timescale(
    regime_sequence: list[str],
    epochs_per_regime: int = 1000,
    drl_steps_per_mdp: int = 50_000,
    regime_detect_fn=None,
    mdp_architect_fn=None,
    seed: int = 42,
    save_dir: Optional[str] = None,
    device: str = "cpu",
    verbose: int = 1,
) -> dict:
    """Two-timescale training: slow LLM MDP architect + fast DRL scheduler.

    This is the main method training pipeline for R009-R011.

    Args:
        regime_detect_fn: callable(kpi_history) -> bool  (regime change detected?)
        mdp_architect_fn: callable(kpi_snapshot, context) -> MDPSpec
        drl_steps_per_mdp: training steps between potential MDP reformulations
    """
    env = FlatActionWrapper(
        BeamAllocationEnv(
            regime_sequence=regime_sequence,
            epochs_per_regime=epochs_per_regime,
            seed=seed,
        )
    )

    agent = PPOAgent(env, device=device, seed=seed, verbose=0)

    total_steps = len(regime_sequence) * epochs_per_regime
    mdp_switches = 0
    all_rates = []
    all_kpis = []
    switch_log = []

    if verbose:
        print(f"Two-timescale training: {total_steps} total env steps")
        print(f"  DRL steps per MDP: {drl_steps_per_mdp}")

    t0 = time.time()
    obs, info = env.reset()
    steps_since_mdp_change = 0

    for step in range(total_steps):
        # Collect KPI
        if "kpi" in info:
            all_kpis.append(info["kpi"])

        # Check for regime change (slow timescale)
        if (
            regime_detect_fn is not None
            and len(all_kpis) > 10
            and steps_since_mdp_change > 50
        ):
            if regime_detect_fn(all_kpis[-50:]):
                if mdp_architect_fn is not None and verbose:
                    print(f"  Step {step}: Regime change detected, reformulating MDP...")

                # LLM generates new MDP spec (slow timescale action)
                if mdp_architect_fn is not None:
                    new_spec = mdp_architect_fn(
                        all_kpis[-1],
                        {"step": step, "prev_switches": mdp_switches},
                    )
                    switch_log.append({
                        "step": step,
                        "spec_id": new_spec.spec_id if new_spec else "none",
                        "kpi": all_kpis[-1],
                    })

                mdp_switches += 1
                steps_since_mdp_change = 0

                # Train DRL on new MDP (fast timescale burst)
                agent.train(total_timesteps=min(drl_steps_per_mdp, 10000))

        # Take action (fast timescale)
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_since_mdp_change += 1

        if "sum_rate_mbps" in info:
            all_rates.append(info["sum_rate_mbps"])

        if terminated or truncated:
            obs, info = env.reset()

        # Periodic DRL training
        if step > 0 and step % drl_steps_per_mdp == 0:
            agent.train(total_timesteps=drl_steps_per_mdp)

    train_time = time.time() - t0

    metrics = {
        "mean_sum_rate_mbps": float(np.mean(all_rates)) if all_rates else 0.0,
        "std_sum_rate_mbps": float(np.std(all_rates)) if all_rates else 0.0,
        "mdp_switches": mdp_switches,
        "total_steps": total_steps,
        "train_time_s": train_time,
    }

    if verbose:
        print(f"  Done: rate={metrics['mean_sum_rate_mbps']:.1f}Mbps "
              f"switches={mdp_switches} time={train_time:.1f}s")

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        agent.save(str(save_path / "model"))
        with open(save_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(save_path / "switch_log.json", "w") as f:
            json.dump(switch_log, f, indent=2)

    return {
        "agent": agent,
        "metrics": metrics,
        "switch_log": switch_log,
    }
