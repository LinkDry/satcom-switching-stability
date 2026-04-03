"""Classical baseline: max-weight matching heuristic for beam allocation."""

import numpy as np


class MaxWeightHeuristic:
    """Greedy max-weight beam allocation baseline.

    At each epoch:
    1. Score each beam by queue_length * channel_gain
    2. Activate top-K beams by score
    3. Allocate power proportional to demand
    """

    def __init__(self, num_beams: int, max_active: int = 10, max_power: float = 20.0):
        self.num_beams = num_beams
        self.max_active = max_active
        self.max_power = max_power

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Take flat observation, return flat action compatible with FlatActionWrapper.

        Obs layout: queue(N) + channel(N) + demand(N) + kpi(3)
        Action layout: beam_logits(N) + power_frac(N)
        """
        n = self.num_beams
        queues = obs[:n]
        channels = obs[n : 2 * n]
        demand = obs[2 * n : 3 * n]

        # Score = queue * channel (both normalized, higher = more urgent + better channel)
        scores = queues * channels + demand * 0.5

        # Activate top-K beams
        top_k = min(self.max_active, n)
        top_indices = np.argsort(-scores)[:top_k]

        beam_logits = np.full(n, -1.0)
        beam_logits[top_indices] = 1.0

        # Power proportional to demand, normalized to [0, 1]
        power_frac = np.zeros(n)
        active_demand = demand[top_indices]
        if active_demand.sum() > 0:
            power_frac[top_indices] = active_demand / active_demand.sum()
        else:
            power_frac[top_indices] = 1.0 / top_k

        # Flat action: concat beam_logits and power_frac scaled to [-1, 1]
        action = np.concatenate([beam_logits, power_frac * 2 - 1])
        return action.astype(np.float32)


class RandomBaseline:
    """Random beam allocation baseline for sanity comparison."""

    def __init__(self, num_beams: int, rng_seed: int = 42):
        self.num_beams = num_beams
        self.rng = np.random.default_rng(rng_seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        n = self.num_beams
        beam_logits = self.rng.uniform(-1, 1, n)
        power_frac = self.rng.uniform(-1, 1, n)
        return np.concatenate([beam_logits, power_frac]).astype(np.float32)
