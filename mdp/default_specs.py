"""Expert-designed MDP specifications for each traffic regime."""

from mdp.spec import Constraint, MDPSpec, RewardComponent


def urban_spec() -> MDPSpec:
    """MDP spec optimized for urban dense traffic (R1).

    Focus on sum rate maximization in center beams, with fairness penalty
    to prevent total starvation of edge beams.
    """
    return MDPSpec(
        spec_id="expert-urban",
        state_features=[
            "queue_lengths",
            "channel_snr",
            "demand_current",
            "spatial_gini",
            "prev_beam_activation",
        ],
        action_type="per_beam",
        action_params={"max_active_beams": 10, "power_levels": "continuous"},
        reward_components=[
            RewardComponent("sum_rate", 1.0),
            RewardComponent("proportional_fairness", 0.3),
            RewardComponent("outage_penalty", 0.5),
            RewardComponent("switching_cost", 0.01),
        ],
        constraints=[
            Constraint("max_total_power", 20.0),
            Constraint("max_active_beams", 10),
        ],
        description="Expert MDP for urban dense: prioritize throughput with fairness guard",
    )


def maritime_spec() -> MDPSpec:
    """MDP spec optimized for maritime sparse traffic (R2).

    Focus on energy efficiency — fewer active beams, lower power.
    """
    return MDPSpec(
        spec_id="expert-maritime",
        state_features=[
            "queue_lengths",
            "channel_snr",
            "demand_current",
            "active_beam_fraction",
        ],
        action_type="per_beam",
        action_params={"max_active_beams": 6, "power_levels": "discrete_3"},
        reward_components=[
            RewardComponent("sum_rate", 0.5),
            RewardComponent("power_efficiency", 1.0),
            RewardComponent("outage_penalty", 0.3),
            RewardComponent("switching_cost", 0.02),
        ],
        constraints=[
            Constraint("max_total_power", 12.0),
            Constraint("max_active_beams", 6),
        ],
        description="Expert MDP for maritime: prioritize energy efficiency, low beam count",
    )


def disaster_spec() -> MDPSpec:
    """MDP spec optimized for disaster spike traffic (R3).

    Focus on guaranteeing min rate to spike beams — outage penalty dominates.
    """
    return MDPSpec(
        spec_id="expert-disaster",
        state_features=[
            "queue_lengths",
            "channel_snr",
            "demand_current",
            "peak_beam_demand",
            "queue_growth_rate",
        ],
        action_type="per_beam",
        action_params={"max_active_beams": 12, "power_levels": "continuous"},
        reward_components=[
            RewardComponent("min_rate", 1.5),
            RewardComponent("outage_penalty", 2.0),
            RewardComponent("queue_penalty", 0.5),
            RewardComponent("switching_cost", 0.005),
        ],
        constraints=[
            Constraint("max_total_power", 20.0),
            Constraint("max_active_beams", 12),
            Constraint("min_beam_rate", 5.0),
        ],
        description="Expert MDP for disaster: guarantee min rate, aggressive beam activation",
    )


def mixed_spec() -> MDPSpec:
    """MDP spec for mixed/transition traffic (R4).

    Balanced between urban and maritime — moderate everything.
    """
    return MDPSpec(
        spec_id="expert-mixed",
        state_features=[
            "queue_lengths",
            "channel_snr",
            "demand_current",
            "demand_history_mean",
            "spatial_gini",
        ],
        action_type="per_beam",
        action_params={"max_active_beams": 8, "power_levels": "continuous"},
        reward_components=[
            RewardComponent("sum_rate", 0.8),
            RewardComponent("proportional_fairness", 0.4),
            RewardComponent("outage_penalty", 0.5),
            RewardComponent("switching_cost", 0.015),
        ],
        constraints=[
            Constraint("max_total_power", 16.0),
            Constraint("max_active_beams", 8),
        ],
        description="Expert MDP for mixed transition: balanced throughput/efficiency",
    )


DEFAULT_SPECS = {
    "urban": urban_spec,
    "maritime": maritime_spec,
    "disaster": disaster_spec,
    "mixed": mixed_spec,
}


def get_default_spec(regime_name: str) -> MDPSpec:
    if regime_name not in DEFAULT_SPECS:
        raise ValueError(f"Unknown regime: {regime_name}. Available: {list(DEFAULT_SPECS.keys())}")
    return DEFAULT_SPECS[regime_name]()
