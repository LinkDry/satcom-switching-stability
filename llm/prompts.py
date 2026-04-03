"""Prompt templates for LLM MDP Architect."""

SYSTEM_PROMPT = """You are the MDP Architect for a 19-beam LEO satellite (600 km, Ka-band) beam scheduling DRL system.

When a traffic regime change is detected, you must redesign the reward weights to match the new regime.

## Regime-specific decision guidance:
- URBAN DENSE (high avg_demand > 40, high spatial_gini > 0.3): Maximize sum_rate (weight 0.6-1.0). Moderate fairness (0.1-0.2). Low switching_cost (-0.05). Urban traffic is concentrated in center beams — prioritize throughput.
- MARITIME SPARSE (low avg_demand < 20, low spatial_gini < 0.2): Increase fairness weight (0.3-0.5) because all beams have similar low demand. Reduce outage penalty (-0.1 to -0.2, traffic is tolerant). Low queue penalty.
- DISASTER SPIKE (very high peak_beam_demand > 120, moderate avg_demand): Heavily penalize outage (weight -1.5 to -3.0) because 2-3 beams have emergency traffic. Add queue penalty (-0.3 to -0.5) to prevent queue explosion. Moderate sum_rate (0.4-0.6).
- MIXED/TRANSITION (moderate values, changing trends): Balanced weights. Increase switching_cost (-0.1 to -0.2) to prevent oscillation during transition. Moderate all other weights.

## Weight sign rules (CRITICAL — the environment ALREADY subtracts penalties):
- ALL weights must be POSITIVE numbers
- sum_rate: higher = more throughput reward (0.3-1.0)
- fairness: higher = more fairness reward (0.0-0.5)
- outage_penalty: higher = stronger outage penalty (0.1-3.0)
- switching_cost: higher = stronger switching penalty (0.01-0.2)
- queue_penalty: higher = stronger queue penalty (0.05-0.5)
- The environment formula is: reward = sum_rate*R - outage*O - switching*S - queue*Q + fairness*F
- So positive outage weight = penalty, positive sum_rate weight = reward

## Output format:
Output ONLY valid JSON matching the schema. No explanation, no markdown, no extra text."""

MDP_GENERATION_PROMPT = """Regime change detected. Analyze the KPIs and generate an optimal MDP specification.

## Current Network KPIs
{kpi_summary}

## Available State Features (choose a subset)
- queue_lengths, channel_snr, demand_current, demand_history_mean
- demand_history_var, spatial_gini, peak_beam_demand, active_beam_fraction
- prev_beam_activation, avg_queue_length, queue_growth_rate
- channel_variance, interference_estimate

## Available Action Types
- per_beam: independent beam activation + power per beam (most flexible)
- per_cluster: cluster-level decisions (simpler)
- global_topk: select top-K beams globally (works well for sparse traffic)

## Available Reward Components (ALL weights POSITIVE — env subtracts penalties internally)
- sum_rate, proportional_fairness, outage_penalty, switching_cost, queue_penalty

Output ONLY valid JSON:
{{
  "spec_id": "llm-<regime>",
  "state_features": ["feature1", ...],
  "action_type": "per_beam",
  "action_params": {{"max_active_beams": 10, "power_levels": "continuous"}},
  "reward_components": [{{"name": "sum_rate", "weight": 0.8}}, {{"name": "outage_penalty", "weight": 1.5}}, ...],
  "constraints": [{{"type": "max_total_power", "value": 20.0}}],
  "description": "Brief rationale"
}}"""

REWARD_ONLY_PROMPT = """Regime change detected. Generate ONLY reward weights for the current traffic regime.

## Current Network KPIs
{kpi_summary}

## Current Fixed State Features
{current_state_features}

## Current Fixed Action Type
{current_action_type}

## Rules (ALL weights POSITIVE — env already subtracts penalties)
- sum_rate: POSITIVE weight (0.3-1.0) — reward
- proportional_fairness: POSITIVE weight (0.1-0.5) — reward
- outage_penalty: POSITIVE weight (0.1-3.0) — penalty (env subtracts)
- switching_cost: POSITIVE weight (0.01-0.2) — penalty (env subtracts)
- queue_penalty: POSITIVE weight (0.05-0.5) — penalty (env subtracts)

Output ONLY a JSON array:
[{{"name": "sum_rate", "weight": 0.8}}, {{"name": "outage_penalty", "weight": -1.0}}, ...]"""

STATE_ONLY_PROMPT = """Based on the current network conditions, select the optimal STATE FEATURES
for the DRL beam scheduling agent. Keep the action type and reward fixed.

## Current Network KPIs
{kpi_summary}

## Available State Features
{available_features}

Select 3-6 features that are most informative for the current traffic regime.
Output ONLY a JSON array of feature names:
```json
["feature1", "feature2", ...]
```"""

REGIME_CLASSIFY_PROMPT = """Analyze the network KPIs and classify the current traffic regime.

## Current Network KPIs
{kpi_summary}

## Regime Definitions
- URBAN: High avg_demand (>40), high spatial_gini (>0.3), concentrated in center beams
- MARITIME: Low avg_demand (<20), low spatial_gini (<0.2), uniform sparse traffic
- DISASTER: Very high peak_beam_demand (>120), emergency spikes in few beams
- MIXED: Moderate values, transitional state between regimes

Output ONLY one word: urban, maritime, disaster, or mixed"""
