"""Build Gym observation/action/reward configuration from an MDP spec."""

import numpy as np

from mdp.spec import MDPSpec


# All available feature extractors
FEATURE_EXTRACTORS = {
    "queue_lengths": lambda env: env.queue_lengths / max(env.queue_lengths.max(), 1.0),
    "channel_snr": lambda env: (
        10 * np.log10(np.clip(env.current_channel_gains, 1e-20, None)) + 200
    ) / 200,
    "demand_current": lambda env: env.current_demand / max(env.current_demand.max(), 1.0),
    "demand_history_mean": lambda env: np.full(
        env.num_beams, np.mean(env.current_demand) / 100
    ),
    "demand_history_var": lambda env: np.full(
        env.num_beams, np.var(env.current_demand) / 1e4
    ),
    "spatial_gini": lambda env: np.array([env.traffic._gini(env.current_demand)]),
    "peak_beam_demand": lambda env: np.array([env.current_demand.max() / 200]),
    "active_beam_fraction": lambda env: np.array(
        [np.mean(env.current_demand > 5.0)]
    ),
    "prev_beam_activation": lambda env: env.prev_active_beams.astype(np.float32),
    "avg_queue_length": lambda env: np.array(
        [np.mean(env.queue_lengths) / 100]
    ),
    "queue_growth_rate": lambda env: np.array(
        [np.clip(np.mean(env.current_demand - 50) / 100, -1, 1)]
    ),
    "channel_variance": lambda env: np.array(
        [np.var(env.current_channel_gains) / max(np.var(env.current_channel_gains), 1e-20)]
    ),
    "interference_estimate": lambda env: np.zeros(env.num_beams),
}

# Feature output dimensions (for building obs space)
FEATURE_DIMS = {
    "queue_lengths": lambda n: n,
    "channel_snr": lambda n: n,
    "demand_current": lambda n: n,
    "demand_history_mean": lambda n: n,
    "demand_history_var": lambda n: n,
    "spatial_gini": lambda _: 1,
    "peak_beam_demand": lambda _: 1,
    "active_beam_fraction": lambda _: 1,
    "prev_beam_activation": lambda n: n,
    "avg_queue_length": lambda _: 1,
    "queue_growth_rate": lambda _: 1,
    "channel_variance": lambda _: 1,
    "interference_estimate": lambda n: n,
}

# Reward component calculators
REWARD_CALCULATORS = {
    "sum_rate": lambda rates, **kw: np.sum(rates) / 100.0,
    "min_rate": lambda rates, **kw: np.min(rates[rates > 0]) / 100.0
    if np.any(rates > 0)
    else 0.0,
    "proportional_fairness": lambda rates, **kw: np.sum(np.log(np.clip(rates, 0.1, None)))
    / max(len(rates), 1),
    "outage_penalty": lambda rates, active, threshold=10.0, **kw: -np.sum(
        (rates < threshold) & active
    ),
    "switching_cost": lambda prev_active, active, **kw: -np.sum(
        prev_active != active
    ),
    "queue_penalty": lambda queues, **kw: -np.mean(queues) / 100.0,
    "power_efficiency": lambda rates, power, **kw: np.sum(rates)
    / max(np.sum(power), 1e-6)
    / 100.0,
}


def compute_obs_dim(spec: MDPSpec, num_beams: int) -> int:
    """Compute total observation dimension for a given MDP spec."""
    total = 0
    for feat in spec.state_features:
        total += FEATURE_DIMS[feat](num_beams)
    return total


def extract_observation(spec: MDPSpec, env) -> np.ndarray:
    """Extract observation vector from environment according to MDP spec."""
    parts = []
    for feat in spec.state_features:
        parts.append(FEATURE_EXTRACTORS[feat](env))
    return np.concatenate(parts).astype(np.float32)


def compute_reward(
    spec: MDPSpec,
    rates: np.ndarray,
    active_beams: np.ndarray,
    prev_active: np.ndarray,
    power_alloc: np.ndarray,
    queue_lengths: np.ndarray,
) -> float:
    """Compute reward according to MDP spec's reward components."""
    total = 0.0
    for rc in spec.reward_components:
        calc = REWARD_CALCULATORS.get(rc.name)
        if calc is None:
            continue
        val = calc(
            rates=rates,
            active=active_beams,
            prev_active=prev_active,
            power=power_alloc,
            queues=queue_lengths,
        )
        total += rc.weight * val
    return total
