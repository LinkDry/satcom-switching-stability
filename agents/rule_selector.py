"""Rule-based MDP selector baseline — switches among pre-designed MDPs by KPI thresholds."""

import numpy as np
from mdp.default_specs import get_default_spec
from mdp.spec import MDPSpec


class RuleBasedMDPSelector:
    """Baseline that detects regime from KPIs and selects a pre-designed MDP spec.

    Uses simple threshold-based classification:
    - High avg demand + high gini -> urban
    - Low avg demand + low gini -> maritime
    - High peak demand -> disaster
    - Otherwise -> mixed
    """

    def __init__(
        self,
        urban_demand_threshold: float = 40.0,
        maritime_demand_threshold: float = 20.0,
        disaster_peak_threshold: float = 120.0,
        gini_threshold: float = 0.3,
    ):
        self.urban_demand_threshold = urban_demand_threshold
        self.maritime_demand_threshold = maritime_demand_threshold
        self.disaster_peak_threshold = disaster_peak_threshold
        self.gini_threshold = gini_threshold
        self.current_spec: MDPSpec | None = None

    def classify_regime(self, kpi: dict) -> str:
        """Classify current traffic regime from KPI snapshot."""
        avg_demand = kpi["avg_demand"]
        peak = kpi["peak_beam_demand"]
        gini = kpi["spatial_gini"]

        if peak > self.disaster_peak_threshold:
            return "disaster"
        if avg_demand > self.urban_demand_threshold and gini > self.gini_threshold:
            return "urban"
        if avg_demand < self.maritime_demand_threshold:
            return "maritime"
        return "mixed"

    def select_spec(self, kpi: dict) -> tuple[MDPSpec, bool]:
        """Select MDP spec based on KPI snapshot.

        Returns (spec, changed) where changed=True if a new spec was selected.
        """
        regime = self.classify_regime(kpi)
        new_spec = get_default_spec(regime)

        if self.current_spec is None or self.current_spec.spec_id != new_spec.spec_id:
            self.current_spec = new_spec
            return new_spec, True
        return self.current_spec, False


class MetaRLSelector:
    """Placeholder for meta-RL MDP selector baseline.

    Uses a small MLP that maps KPI vector -> MDP index, trained on
    regime-performance pairs from initial data collection.
    """

    def __init__(self, num_specs: int = 4, seed: int = 42):
        self.num_specs = num_specs
        self.spec_names = ["urban", "maritime", "disaster", "mixed"]
        self.rng = np.random.default_rng(seed)
        # Simple Q-table: kpi_bins x specs
        self.q_table = np.zeros((100, num_specs))
        self.learning_rate = 0.1
        self.epsilon = 0.2
        self.current_spec_idx = 0

    def _kpi_to_bin(self, kpi: dict) -> int:
        """Discretize KPI into a bin index."""
        avg = min(int(kpi["avg_demand"] / 5), 9)
        gini = min(int(kpi["spatial_gini"] * 10), 9)
        return avg * 10 + gini

    def select_spec(self, kpi: dict, reward: float = 0.0) -> tuple[MDPSpec, bool]:
        """Select MDP spec using epsilon-greedy on Q-table."""
        kpi_bin = self._kpi_to_bin(kpi)

        # Update Q-table with last reward
        self.q_table[kpi_bin, self.current_spec_idx] += self.learning_rate * (
            reward - self.q_table[kpi_bin, self.current_spec_idx]
        )

        # Epsilon-greedy selection
        if self.rng.random() < self.epsilon:
            new_idx = self.rng.integers(self.num_specs)
        else:
            new_idx = int(np.argmax(self.q_table[kpi_bin]))

        changed = new_idx != self.current_spec_idx
        self.current_spec_idx = new_idx
        spec = get_default_spec(self.spec_names[new_idx])
        return spec, changed


class RandomMDPSelector:
    """Random MDP selector — switches randomly at regime boundaries."""

    def __init__(self, seed: int = 42):
        self.spec_names = ["urban", "maritime", "disaster", "mixed"]
        self.rng = np.random.default_rng(seed)
        self.current_spec = None

    def select_spec(self, regime_changed: bool) -> tuple[MDPSpec, bool]:
        if regime_changed or self.current_spec is None:
            name = self.rng.choice(self.spec_names)
            self.current_spec = get_default_spec(name)
            return self.current_spec, True
        return self.current_spec, False
