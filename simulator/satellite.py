"""LEO satellite with 19-beam hexagonal cell layout."""

import numpy as np
from simulator.channel import EARTH_RADIUS


def hex_beam_centers(num_rings: int = 2) -> np.ndarray:
    """Generate hexagonal beam center offsets (in normalized coordinates).

    Returns array of shape (N, 2) with (x, y) offsets.
    num_rings=0 -> 1 beam (center only)
    num_rings=1 -> 7 beams
    num_rings=2 -> 19 beams

    Uses axial coordinates with cube constraint s = -q - r.
    Total beams = 3*n*(n+1) + 1 for n rings.
    """
    centers = set()
    centers.add((0, 0))
    for q in range(-num_rings, num_rings + 1):
        for r in range(max(-num_rings, -q - num_rings), min(num_rings, -q + num_rings) + 1):
            centers.add((q, r))
    # Convert axial (q, r) to cartesian
    result = []
    for q, r in sorted(centers):
        x = q + r * 0.5
        y = r * (np.sqrt(3) / 2)
        result.append((x, y))
    return np.array(result)


class LEOSatellite:
    """Single LEO satellite with multi-beam antenna."""

    def __init__(
        self,
        altitude_m: float = 600e3,
        num_rings: int = 2,
        beam_spacing_deg: float = 1.5,
        max_tx_power_w: float = 20.0,
        antenna_gain_db: float = 40.0,
        total_bandwidth_hz: float = 500e6,
        num_subbands: int = 4,
        seed: int = 42,
    ):
        self.altitude_m = altitude_m
        self.beam_spacing_deg = beam_spacing_deg
        self.max_tx_power_w = max_tx_power_w
        self.antenna_gain_db = antenna_gain_db
        self.total_bandwidth_hz = total_bandwidth_hz
        self.num_subbands = num_subbands
        self.bandwidth_per_subband = total_bandwidth_hz / num_subbands
        self.rng = np.random.default_rng(seed)

        # Generate beam centers
        raw_centers = hex_beam_centers(num_rings)
        self.num_beams = len(raw_centers)
        # Scale to angular offsets from nadir
        self.beam_offsets_deg = raw_centers * beam_spacing_deg

        # Compute beam footprint radius on ground
        nadir_angle_rad = np.radians(beam_spacing_deg / 2)
        self.beam_radius_km = (altitude_m * np.tan(nadir_angle_rad)) / 1000

        # Compute elevation angles for each beam center
        self.beam_elevations = self._compute_elevations()

        # Build adjacency matrix for inter-beam interference
        self.adjacency = self._build_adjacency()

    def _compute_elevations(self) -> np.ndarray:
        """Compute elevation angle (deg) for each beam center as seen from ground."""
        elevations = np.zeros(self.num_beams)
        for i, (ox, oy) in enumerate(self.beam_offsets_deg):
            off_nadir = np.sqrt(ox ** 2 + oy ** 2)
            # Elevation from ground = 90 - off-nadir angle (simplified for LEO)
            # More accurate: use earth geometry
            if off_nadir < 0.01:
                elevations[i] = 90.0
            else:
                off_rad = np.radians(off_nadir)
                # Ground distance from sub-satellite point
                ground_dist = self.altitude_m * np.tan(off_rad)
                # Elevation at ground point
                el = np.degrees(
                    np.arctan2(
                        self.altitude_m - EARTH_RADIUS * (1 - np.cos(ground_dist / EARTH_RADIUS)),
                        ground_dist,
                    )
                )
                elevations[i] = max(el, 10.0)
        return elevations

    def _build_adjacency(self) -> np.ndarray:
        """Build beam adjacency matrix. Beams within 1.5x spacing are adjacent."""
        adj = np.zeros((self.num_beams, self.num_beams), dtype=bool)
        threshold = 1.5 * self.beam_spacing_deg
        for i in range(self.num_beams):
            for j in range(i + 1, self.num_beams):
                dist = np.linalg.norm(
                    self.beam_offsets_deg[i] - self.beam_offsets_deg[j]
                )
                if dist < threshold:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    def beam_gain_db(self, beam_idx: int, off_axis_deg: float = 0.0) -> float:
        """Beam antenna gain with off-axis roll-off (simplified parabolic).

        G(theta) = G_max - 12*(theta/theta_3dB)^2 for theta < theta_3dB
        """
        theta_3db = self.beam_spacing_deg / 2  # 3dB beamwidth
        if off_axis_deg == 0:
            return self.antenna_gain_db
        rolloff = 12 * (off_axis_deg / theta_3db) ** 2
        return max(self.antenna_gain_db - rolloff, 0.0)

    def inter_beam_interference(
        self, active_beams: np.ndarray, power_alloc: np.ndarray
    ) -> np.ndarray:
        """Compute inter-beam interference power for each beam.

        Args:
            active_beams: boolean array (num_beams,) — which beams are illuminated
            power_alloc: float array (num_beams,) — transmit power per beam (W)

        Returns:
            interference: float array (num_beams,) — interference power at each beam
        """
        interference = np.zeros(self.num_beams)
        for i in range(self.num_beams):
            if not active_beams[i]:
                continue
            for j in range(self.num_beams):
                if i == j or not active_beams[j]:
                    continue
                if self.adjacency[i, j]:
                    # Adjacent beam interference: -20 dB sidelobe isolation
                    interference[i] += power_alloc[j] * 0.01  # -20 dB
        return interference

    def get_state_summary(self) -> dict:
        """Return a summary dict for logging/debugging."""
        return {
            "num_beams": self.num_beams,
            "altitude_km": self.altitude_m / 1000,
            "beam_spacing_deg": self.beam_spacing_deg,
            "beam_radius_km": self.beam_radius_km,
            "min_elevation_deg": float(self.beam_elevations.min()),
            "max_elevation_deg": float(self.beam_elevations.max()),
            "total_bandwidth_mhz": self.total_bandwidth_hz / 1e6,
            "num_subbands": self.num_subbands,
            "max_tx_power_w": self.max_tx_power_w,
        }
