"""Traffic regime generators for multi-beam LEO satellite."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class RegimeType(Enum):
    URBAN = "urban"
    MARITIME = "maritime"
    DISASTER = "disaster"
    MIXED = "mixed"
    FLASH_CROWD = "flash_crowd"
    # Novel regimes (not in MLP training data)
    IOT_BURST = "iot_burst"
    POLAR_HANDOVER = "polar_handover"
    HOT_COLD = "hot_cold"


@dataclass
class TrafficConfig:
    """Configuration for a single traffic regime."""

    regime_type: RegimeType
    num_beams: int
    # Demand parameters (Mbps per beam per epoch)
    mean_demand: np.ndarray  # shape (num_beams,)
    std_demand: np.ndarray   # shape (num_beams,)


def make_urban_config(num_beams: int, rng: np.random.Generator) -> TrafficConfig:
    """R1: Urban dense — high aggregate, concentrated in center beams."""
    mean = np.zeros(num_beams)
    std = np.zeros(num_beams)
    # Center beams (indices 0-6 for 19-beam) get high demand
    n_center = min(7, num_beams)
    mean[:n_center] = rng.uniform(80, 120, n_center)  # Mbps
    mean[n_center:] = rng.uniform(10, 30, num_beams - n_center)
    std[:] = mean * 0.2  # 20% variance
    return TrafficConfig(RegimeType.URBAN, num_beams, mean, std)


def make_maritime_config(num_beams: int, rng: np.random.Generator) -> TrafficConfig:
    """R2: Maritime sparse — low aggregate, uniform across beams."""
    mean = rng.uniform(5, 15, num_beams)
    std = mean * 0.3
    return TrafficConfig(RegimeType.MARITIME, num_beams, mean, std)


def make_disaster_config(
    num_beams: int, rng: np.random.Generator, spike_beams: Optional[list] = None
) -> TrafficConfig:
    """R3: Disaster spike — sudden high demand in 2-3 beams, others baseline."""
    mean = rng.uniform(10, 25, num_beams)
    std = mean * 0.15
    # Spike 2-3 random beams
    if spike_beams is None:
        spike_beams = rng.choice(num_beams, size=min(3, num_beams), replace=False)
    for b in spike_beams:
        mean[b] = rng.uniform(150, 250)  # very high demand
        std[b] = mean[b] * 0.1
    return TrafficConfig(RegimeType.DISASTER, num_beams, mean, std)


def make_flash_crowd_config(num_beams: int, rng: np.random.Generator) -> TrafficConfig:
    """R_novel: Flash crowd — simultaneous spike in ALL beams."""
    mean = rng.uniform(100, 180, num_beams)
    std = mean * 0.15
    return TrafficConfig(RegimeType.FLASH_CROWD, num_beams, mean, std)


def make_iot_burst_config(num_beams: int, rng: np.random.Generator) -> TrafficConfig:
    """Novel R1: IoT burst — very low baseline + micro-bursts in 2-3 random beams."""
    mean = rng.uniform(2, 8, num_beams)  # Very low IoT baseline
    burst_beams = rng.choice(num_beams, size=min(3, num_beams), replace=False)
    mean[burst_beams] = rng.uniform(80, 150, len(burst_beams))  # Sudden sync bursts
    std = mean * 0.4  # High variance (bursty)
    return TrafficConfig(RegimeType.IOT_BURST, num_beams, mean, std)


def make_polar_handover_config(num_beams: int, rng: np.random.Generator) -> TrafficConfig:
    """Novel R2: Polar handover — demand wave migrating across beams."""
    phase = rng.uniform(0, 2 * np.pi)
    beam_idx = np.arange(num_beams)
    # Sinusoidal demand wave across beams
    mean = 20 + 60 * np.abs(np.sin(phase + beam_idx * np.pi / num_beams))
    std = mean * 0.25
    return TrafficConfig(RegimeType.POLAR_HANDOVER, num_beams, mean, std)


def make_hot_cold_config(num_beams: int, rng: np.random.Generator) -> TrafficConfig:
    """Novel R3: Hot-cold — extreme spatial split, half beams saturated, half idle."""
    mean = np.zeros(num_beams)
    hot_beams = rng.choice(num_beams, size=num_beams // 2, replace=False)
    mean[hot_beams] = rng.uniform(80, 140, len(hot_beams))  # Hot beams
    cold_mask = np.ones(num_beams, dtype=bool)
    cold_mask[hot_beams] = False
    mean[cold_mask] = rng.uniform(1, 5, cold_mask.sum())  # Cold beams near zero
    std = mean * 0.15
    return TrafficConfig(RegimeType.HOT_COLD, num_beams, mean, std)


class TrafficGenerator:
    """Generates per-beam demand samples for a given regime."""

    def __init__(self, config: TrafficConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        """Sample one epoch of demand (Mbps per beam). Clipped to >= 0."""
        demand = self.rng.normal(self.config.mean_demand, self.config.std_demand)
        return np.clip(demand, 0, None)


class RegimeSequence:
    """Manages a sequence of traffic regimes with transitions.

    Each regime runs for `epochs_per_regime` steps. Supports:
    - Step transitions (instant switch)
    - Mixed regime (linear interpolation between two configs)
    """

    def __init__(
        self,
        num_beams: int,
        regime_sequence: list[RegimeType],
        epochs_per_regime: int = 200,
        seed: int = 42,
    ):
        self.num_beams = num_beams
        self.regime_sequence = regime_sequence
        self.epochs_per_regime = epochs_per_regime
        self.total_epochs = len(regime_sequence) * epochs_per_regime
        self.rng = np.random.default_rng(seed)

        # Pre-build configs
        self.configs: list[TrafficConfig] = []
        for rt in regime_sequence:
            cfg = self._build_config(rt)
            self.configs.append(cfg)

        self.current_epoch = 0
        self._current_gen: Optional[TrafficGenerator] = None
        self._update_generator()

    def _build_config(self, regime_type: RegimeType) -> TrafficConfig:
        builders = {
            RegimeType.URBAN: make_urban_config,
            RegimeType.MARITIME: make_maritime_config,
            RegimeType.DISASTER: make_disaster_config,
            RegimeType.FLASH_CROWD: make_flash_crowd_config,
            RegimeType.IOT_BURST: make_iot_burst_config,
            RegimeType.POLAR_HANDOVER: make_polar_handover_config,
            RegimeType.HOT_COLD: make_hot_cold_config,
        }
        if regime_type == RegimeType.MIXED:
            # Mixed: interpolate between urban and maritime
            return make_urban_config(self.num_beams, self.rng)
        return builders[regime_type](self.num_beams, self.rng)

    def _update_generator(self):
        idx = min(
            self.current_epoch // self.epochs_per_regime,
            len(self.configs) - 1,
        )
        self._current_gen = TrafficGenerator(self.configs[idx], seed=self.rng.integers(0, 2**31))

    @property
    def current_regime_idx(self) -> int:
        return min(
            self.current_epoch // self.epochs_per_regime,
            len(self.configs) - 1,
        )

    @property
    def current_regime_type(self) -> RegimeType:
        return self.regime_sequence[self.current_regime_idx]

    def sample(self) -> np.ndarray:
        """Sample demand for current epoch, handling mixed regime interpolation."""
        idx = self.current_regime_idx
        rt = self.regime_sequence[idx]

        # All regime transitions are abrupt (no interpolation)
        return self._current_gen.sample()

    def step(self) -> tuple[np.ndarray, bool]:
        """Advance one epoch. Returns (demand, regime_changed)."""
        old_idx = self.current_regime_idx
        demand = self.sample()
        self.current_epoch += 1
        new_idx = self.current_regime_idx
        regime_changed = new_idx != old_idx
        if regime_changed:
            self._update_generator()
        return demand, regime_changed

    def reset(self):
        self.current_epoch = 0
        self._update_generator()

    def get_kpi_snapshot(self, demand: np.ndarray) -> dict:
        """Compute KPI snapshot from current demand — used for regime detection."""
        return {
            "avg_demand": float(np.mean(demand)),
            "demand_variance": float(np.var(demand)),
            "spatial_gini": float(self._gini(demand)),
            "peak_beam_demand": float(np.max(demand)),
            "active_beam_fraction": float(np.mean(demand > 5.0)),
            "regime_type": self.current_regime_type.value,
            "epoch": self.current_epoch,
        }

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Gini coefficient as spatial concentration measure. 0=equal, 1=concentrated."""
        sorted_v = np.sort(values)
        n = len(sorted_v)
        if n == 0 or sorted_v.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_v) / (n * np.sum(sorted_v))) - (n + 1) / n)
