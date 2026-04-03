"""Gymnasium environment for multi-beam LEO satellite beam resource allocation."""

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulator.channel import NTNChannel
from simulator.satellite import LEOSatellite
from simulator.traffic import RegimeSequence, RegimeType


class BeamAllocationEnv(gym.Env):
    """Multi-beam LEO satellite resource allocation environment.

    Observation (default MDP spec):
        - per-beam queue lengths (normalized)
        - per-beam channel states (SNR in dB, normalized)
        - per-beam demand (normalized)
        - global KPIs (avg demand, variance, gini)

    Action (hybrid):
        - beam_activation: MultiBinary(num_beams) — which beams to illuminate
        - power_allocation: Box(num_beams) — power fraction per beam [0, 1]

    Reward:
        weighted_sum_rate - outage_penalty - switching_cost
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_rings: int = 2,
        regime_sequence: Optional[list[str]] = None,
        epochs_per_regime: int = 200,
        max_active_beams: int = 10,
        min_rate_threshold_mbps: float = 10.0,
        switching_cost_weight: float = 0.01,
        outage_penalty_weight: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()

        self.seed_val = seed
        self.rng = np.random.default_rng(seed)

        # Build satellite
        self.satellite = LEOSatellite(num_rings=num_rings, seed=seed)
        self.num_beams = self.satellite.num_beams

        # Build channel
        self.channel = NTNChannel(seed=seed)

        # Build traffic
        if regime_sequence is None:
            regime_sequence = ["urban", "maritime", "disaster", "mixed"]
        regime_types = [RegimeType(r) for r in regime_sequence]
        self.traffic = RegimeSequence(
            self.num_beams, regime_types, epochs_per_regime, seed=seed
        )

        # Parameters
        self.max_active_beams = max_active_beams
        self.min_rate_threshold_mbps = min_rate_threshold_mbps
        self.switching_cost_weight = switching_cost_weight
        self.outage_penalty_weight = outage_penalty_weight
        self.reward_weights = {"sum_rate": 1.0, "outage": outage_penalty_weight, "switching": switching_cost_weight, "queue": 0.0, "fairness": 0.0}

        # State tracking
        self.queue_lengths = np.zeros(self.num_beams)
        self.prev_active_beams = np.zeros(self.num_beams, dtype=bool)
        self.current_demand = np.zeros(self.num_beams)
        self.current_channel_gains = np.zeros(self.num_beams)
        self.step_count = 0

        # Define spaces
        # Observation: queue(N) + channel(N) + demand(N) + kpi(3) = 3N + 3
        obs_dim = 3 * self.num_beams + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: dict space with beam selection + power allocation
        self.action_space = spaces.Dict(
            {
                "beam_activation": spaces.MultiBinary(self.num_beams),
                "power_allocation": spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_beams,), dtype=np.float32
                ),
            }
        )

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        # Normalize queue lengths
        queue_norm = self.queue_lengths / max(self.queue_lengths.max(), 1.0)
        # Normalize channel gains (log scale)
        channel_db = 10 * np.log10(np.clip(self.current_channel_gains, 1e-20, None))
        channel_norm = (channel_db + 200) / 200  # rough normalization
        # Normalize demand
        demand_norm = self.current_demand / max(self.current_demand.max(), 1.0)
        # Global KPIs
        kpi = np.array(
            [
                np.mean(self.current_demand) / 100,  # normalized avg demand
                np.var(self.current_demand) / 1e4,  # normalized variance
                self.traffic._gini(self.current_demand),  # gini
            ],
            dtype=np.float32,
        )
        return np.concatenate([queue_norm, channel_norm, demand_norm, kpi]).astype(
            np.float32
        )

    def _compute_rates(
        self, active_beams: np.ndarray, power_alloc: np.ndarray
    ) -> np.ndarray:
        """Compute achievable rate (Mbps) per beam given allocation."""
        rates = np.zeros(self.num_beams)
        interference = self.satellite.inter_beam_interference(active_beams, power_alloc)

        ant_gain_linear = 10 ** (self.satellite.antenna_gain_db / 10)
        for i in range(self.num_beams):
            if not active_beams[i] or power_alloc[i] < 1e-6:
                continue
            signal_power = power_alloc[i] * ant_gain_linear * self.current_channel_gains[i]
            noise_plus_interference = self.channel.noise_power_w + interference[i]
            sinr = signal_power / max(noise_plus_interference, 1e-20)
            # Shannon capacity (Mbps)
            bw = self.satellite.bandwidth_per_subband
            rates[i] = (bw * np.log2(1 + sinr)) / 1e6
        return rates

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.traffic.reset()
        self.queue_lengths = np.zeros(self.num_beams)
        self.prev_active_beams = np.zeros(self.num_beams, dtype=bool)
        self.step_count = 0

        # Sample initial demand and channel
        self.current_demand = self.traffic.sample()
        self._update_channels()

        obs = self._get_obs()
        info = {
            "regime": self.traffic.current_regime_type.value,
            "epoch": self.traffic.current_epoch,
        }
        return obs, info

    def _update_channels(self):
        """Re-sample channel gains for all beams."""
        for i in range(self.num_beams):
            self.current_channel_gains[i] = self.channel.compute_channel_gain(
                self.satellite.beam_elevations[i]
            )

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        beam_activation = np.asarray(action["beam_activation"], dtype=bool)
        power_frac = np.asarray(action["power_allocation"], dtype=np.float32)

        # Enforce max active beams
        if beam_activation.sum() > self.max_active_beams:
            # Keep top-K beams by power fraction
            indices = np.argsort(-power_frac)
            beam_activation = np.zeros(self.num_beams, dtype=bool)
            beam_activation[indices[: self.max_active_beams]] = True

        # Convert power fractions to absolute power
        power_alloc = power_frac * beam_activation * self.satellite.max_tx_power_w

        # Compute rates
        rates = self._compute_rates(beam_activation, power_alloc)

        # Update queues: arrivals (demand) minus served (rates)
        self.queue_lengths = np.maximum(
            self.queue_lengths + self.current_demand - rates, 0.0
        )

        # Compute reward components
        sum_rate = rates.sum()
        outage_count = np.sum(
            (rates < self.min_rate_threshold_mbps) & beam_activation
        )
        switching_count = np.sum(beam_activation != self.prev_active_beams)

        reward = (
            self.reward_weights.get("sum_rate", 1.0) * sum_rate / 100.0
            - self.reward_weights.get("outage", self.outage_penalty_weight) * outage_count
            - self.reward_weights.get("switching", self.switching_cost_weight) * switching_count
            - self.reward_weights.get("queue", 0.0) * np.mean(self.queue_lengths) / 100.0
            + self.reward_weights.get("fairness", 0.0) * (np.min(rates[beam_activation]) / max(np.max(rates[beam_activation]), 1e-6) if beam_activation.any() else 0.0)
        )

        # Advance traffic
        self.prev_active_beams = beam_activation.copy()
        self.current_demand, regime_changed = self.traffic.step()
        self._update_channels()
        self.step_count += 1

        # Episode termination
        terminated = self.step_count >= self.traffic.total_epochs
        truncated = False

        info = {
            "sum_rate_mbps": float(sum_rate),
            "per_beam_rates": rates.tolist(),
            "outage_count": int(outage_count),
            "switching_count": int(switching_count),
            "queue_lengths": self.queue_lengths.tolist(),
            "regime": self.traffic.current_regime_type.value,
            "regime_changed": regime_changed,
            "epoch": self.traffic.current_epoch,
            "kpi": self.traffic.get_kpi_snapshot(self.current_demand),
        }

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def update_reward_weights(self, weights: dict):
        """Dynamically update reward weights (called by LLM MDP Architect)."""
        for k, v in weights.items():
            if k in self.reward_weights:
                self.reward_weights[k] = float(v)


class FlatActionWrapper(gym.ActionWrapper):
    """Wraps the Dict action space into a flat Box for compatibility with SB3.

    Flat action: first num_beams values = beam activation logits (sigmoid → binary),
    next num_beams values = power fractions [0, 1].
    """

    def __init__(self, env: BeamAllocationEnv):
        super().__init__(env)
        n = env.num_beams
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * n,), dtype=np.float32
        )

    def action(self, flat_action: np.ndarray) -> dict:
        n = self.env.num_beams
        # First half: beam activation via threshold at 0
        beam_logits = flat_action[:n]
        beam_activation = (beam_logits > 0).astype(np.int8)
        # Second half: power fractions clipped to [0, 1]
        power_frac = np.clip((flat_action[n:] + 1) / 2, 0, 1).astype(np.float32)
        return {"beam_activation": beam_activation, "power_allocation": power_frac}
