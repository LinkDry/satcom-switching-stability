"""PPO agent wrapper with warm-start support for MDP switching."""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLogger(BaseCallback):
    """Logs training metrics per rollout."""

    def __init__(self, log_interval: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_rates = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "sum_rate_mbps" in info:
                self.episode_rates.append(info["sum_rate_mbps"])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_rates) > 0 and self.num_timesteps % (self.log_interval * 2048) < 2048:
            avg_rate = np.mean(self.episode_rates[-100:])
            if self.verbose:
                print(f"  step={self.num_timesteps} avg_rate={avg_rate:.1f}Mbps")


class RegimeWeightSwitcher(BaseCallback):
    """Callback that dynamically switches reward weights when env regime changes.

    Monitors info["regime"] and info["regime_changed"] on each step.
    When a regime transition is detected, applies the appropriate corrected
    weights for the new regime via env.update_reward_weights().
    """

    def __init__(self, weight_fn, verbose: int = 0):
        """
        Args:
            weight_fn: callable(regime_str) -> dict of weights, or None
                       Returns the weights to use for a given regime.
                       If None is returned, no weight change is made.
            verbose: verbosity level
        """
        super().__init__(verbose)
        self.weight_fn = weight_fn
        self.current_regime = None
        self.switch_count = 0
        self.regime_switch_log = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            regime = info.get("regime")
            if regime and regime != self.current_regime:
                new_weights = self.weight_fn(regime)
                if new_weights is not None:
                    # Apply weights to the unwrapped env
                    env = self.training_env.envs[0].unwrapped
                    env.update_reward_weights(new_weights)
                    self.switch_count += 1
                    self.regime_switch_log.append({
                        "step": self.num_timesteps,
                        "from": self.current_regime,
                        "to": regime,
                        "switching": new_weights.get("switching", 0),
                    })
                    if self.verbose >= 2:
                        print(f"    [RegimeSwitcher] {self.current_regime}->{regime} "
                              f"sw={new_weights.get('switching', 0):.4f} "
                              f"at step {self.num_timesteps}")
                self.current_regime = regime
        return True


class SmoothRegimeWeightSwitcher(BaseCallback):
    """Callback that GRADUALLY transitions reward weights when env regime changes.

    Instead of instant weight jumps (which destabilize PPO's value function),
    supports two stabilization strategies:

    1. Smooth interpolation: linearly interpolate from old to new weights
       over `smooth_steps` steps (useful when switch interval >> smooth_steps).
       Set smooth_steps < switch_interval for this to work correctly.

    2. Throttled switching: only respond to regime changes at most once every
       `min_switch_interval` steps (useful to reduce update frequency).
       When regime switches faster than this, the new regime is queued but
       weights only actually change after the cooldown expires.

    Both can be combined: throttle first, then smooth on each accepted switch.
    This is Path C: fix PPO non-stationarity from MDP weight switching.

    Note: with epochs_per_regime=200, regime switches occur every ~198 steps.
    Recommended: smooth_steps=100 (completes before next switch at ~198 steps)
    OR min_switch_interval=1000 (enforce 5× longer stability per regime).
    """

    def __init__(
        self,
        weight_fn,
        smooth_steps: int = 100,
        min_switch_interval: int = 0,
        verbose: int = 0,
    ):
        """
        Args:
            weight_fn: callable(regime_str) -> dict of weights, or None
            smooth_steps: steps to interpolate on each accepted switch.
                          Set 0 for instant jump. Should be < min_switch_interval
                          or < switch frequency (~198 steps for epochs_per_regime=200).
            min_switch_interval: minimum steps between weight changes (throttle).
                                 0 = no throttling (accept every regime change).
            verbose: verbosity level
        """
        super().__init__(verbose)
        self.weight_fn = weight_fn
        self.smooth_steps = smooth_steps
        self.min_switch_interval = min_switch_interval
        self.current_regime = None
        self.switch_count = 0
        self.regime_switch_log = []
        self._last_accepted_switch_step = -min_switch_interval  # allow first switch

        # Smooth transition state
        self._transitioning = False
        self._transition_start_step = 0
        self._from_weights = None
        self._to_weights = None

    def _on_step(self) -> bool:
        # Check for regime transition
        for info in self.locals.get("infos", []):
            regime = info.get("regime")
            if regime and regime != self.current_regime:
                # Throttle: only accept switch if cooldown has passed
                steps_since_last = self.num_timesteps - self._last_accepted_switch_step
                accept = (self.min_switch_interval <= 0 or
                          steps_since_last >= self.min_switch_interval)

                if accept:
                    new_weights = self.weight_fn(regime)
                    if new_weights is not None:
                        env = self.training_env.envs[0].unwrapped
                        if self.smooth_steps > 0:
                            # Start smooth transition from current weights
                            current = getattr(env, "reward_weights", None)
                            if current is None:
                                env.update_reward_weights(new_weights)
                            else:
                                self._from_weights = dict(current)
                                self._to_weights = dict(new_weights)
                                self._transition_start_step = self.num_timesteps
                                self._transitioning = True
                        else:
                            # Instant jump
                            env.update_reward_weights(new_weights)

                        self._last_accepted_switch_step = self.num_timesteps
                        self.switch_count += 1
                        self.regime_switch_log.append({
                            "step": self.num_timesteps,
                            "from": self.current_regime,
                            "to": regime,
                            "smooth_steps": self.smooth_steps,
                            "throttled": False,
                        })
                        if self.verbose >= 2:
                            print(f"    [SmoothSwitcher] {self.current_regime}->{regime} "
                                  f"(smooth={self.smooth_steps}) at step {self.num_timesteps}")
                else:
                    if self.verbose >= 2:
                        print(f"    [SmoothSwitcher] Throttled: {self.current_regime}->{regime} "
                              f"(cooldown {self.min_switch_interval - steps_since_last} left)")

                self.current_regime = regime

        # Apply smooth interpolation if in progress
        if self._transitioning and self._from_weights and self._to_weights:
            elapsed = self.num_timesteps - self._transition_start_step
            alpha = min(1.0, elapsed / max(self.smooth_steps, 1))
            env = self.training_env.envs[0].unwrapped
            interp = {
                k: (1.0 - alpha) * self._from_weights[k] + alpha * self._to_weights[k]
                for k in self._to_weights
            }
            env.update_reward_weights(interp)
            if alpha >= 1.0:
                self._transitioning = False

        return True


class PPOAgent:
    """PPO agent with warm-start and save/load for MDP switching experiments."""

    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        net_arch: Optional[list] = None,
        device: str = "cpu",
        seed: int = 42,
        verbose: int = 0,
    ):
        if net_arch is None:
            net_arch = [256, 256, 128]

        policy_kwargs = dict(net_arch=net_arch)

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            device=device,
            seed=seed,
            verbose=verbose,
        )
        self.env = env
        self.total_trained_steps = 0

    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None) -> dict:
        """Train the agent. Returns training info."""
        if callback is None:
            callback = TrainingLogger(verbose=1)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.total_trained_steps += total_timesteps
        return {"total_steps": self.total_trained_steps}

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Predict action for a given observation."""
        action, _states = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        """Load model from disk into current env."""
        self.model = PPO.load(path, env=self.env)

    def get_policy_params(self) -> dict:
        """Extract policy network parameters for warm-start transfer."""
        params = {}
        for name, param in self.model.policy.named_parameters():
            params[name] = param.data.clone()
        return params

    def set_policy_params(self, params: dict, strict: bool = False):
        """Load policy network parameters, skipping mismatched layers.

        Args:
            params: dict of name -> tensor from a previous model
            strict: if True, raise on mismatch; if False, skip mismatched layers
        """
        current_params = dict(self.model.policy.named_parameters())
        transferred = 0
        skipped = 0
        for name, new_val in params.items():
            if name in current_params:
                if current_params[name].shape == new_val.shape:
                    current_params[name].data.copy_(new_val)
                    transferred += 1
                else:
                    if strict:
                        raise ValueError(
                            f"Shape mismatch for {name}: "
                            f"{current_params[name].shape} vs {new_val.shape}"
                        )
                    skipped += 1
            else:
                skipped += 1
        return {"transferred": transferred, "skipped": skipped}


def evaluate_agent(agent: PPOAgent, env, n_episodes: int = 5) -> dict:
    """Evaluate agent over multiple episodes, return aggregate metrics.

    Returns dict with aggregate metrics and a 'kpi_history' list of
    per-step KPI dicts that can be fed to the regime detector.
    """
    all_rates = []
    all_outages = []
    all_rewards = []
    all_fairness = []
    kpi_history = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep + 100)
        ep_reward = 0
        ep_rates = []
        ep_outages = []
        ep_fairness = []

        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_rates.append(info["sum_rate_mbps"])
            ep_outages.append(info["outage_count"])
            # Compute Jain's fairness from per-beam rates
            pbr = info.get("per_beam_rates")
            if pbr is not None and len(pbr) > 0 and np.sum(pbr) > 0:
                pbr = np.array(pbr, dtype=float)
                jain = float(np.sum(pbr)**2 / (len(pbr) * np.sum(pbr**2) + 1e-10))
                ep_fairness.append(jain)
            if "kpi" in info:
                kpi_entry = dict(info["kpi"])
                if "regime" in info:
                    kpi_entry["_regime"] = info["regime"]
                kpi_history.append(kpi_entry)

        all_rewards.append(ep_reward)
        all_rates.extend(ep_rates)
        all_outages.extend(ep_outages)
        all_fairness.extend(ep_fairness)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "mean_sum_rate_mbps": float(np.mean(all_rates)),
        "mean_outage_count": float(np.mean(all_outages)),
        "mean_fairness_index": float(np.mean(all_fairness)) if all_fairness else 0.0,
        "std_sum_rate_mbps": float(np.std(all_rates)),
        "kpi_history": kpi_history,
    }
