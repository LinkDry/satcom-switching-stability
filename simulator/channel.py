"""3GPP TR 38.821 NTN Ka-band channel model for LEO satellite."""

import numpy as np


# Constants
SPEED_OF_LIGHT = 3e8  # m/s
BOLTZMANN_K = 1.38e-23  # J/K
EARTH_RADIUS = 6371e3  # m


def free_space_path_loss_db(distance_m: float, frequency_hz: float) -> float:
    """Free-space path loss in dB. FSPL = 20*log10(4*pi*d*f/c)."""
    return 20 * np.log10(4 * np.pi * distance_m * frequency_hz / SPEED_OF_LIGHT)


def slant_range_m(altitude_m: float, elevation_deg: float) -> float:
    """Slant range from ground user to satellite given elevation angle.

    Based on geometry: d = -R_e*sin(el) + sqrt((R_e*sin(el))^2 + 2*R_e*h + h^2)
    where R_e = earth radius, h = satellite altitude, el = elevation angle.
    """
    el_rad = np.radians(elevation_deg)
    R = EARTH_RADIUS
    h = altitude_m
    sin_el = np.sin(el_rad)
    return -R * sin_el + np.sqrt((R * sin_el) ** 2 + 2 * R * h + h ** 2)


def atmospheric_attenuation_db(elevation_deg: float, frequency_ghz: float) -> float:
    """Simplified atmospheric attenuation (ITU-R P.618).

    For Ka-band (20 GHz downlink), typical clear-sky attenuation.
    Uses a simplified model: A = A_zenith / sin(elevation).
    """
    # Zenith attenuation at Ka-band (20 GHz): ~0.5 dB clear sky
    a_zenith = 0.05 * frequency_ghz  # rough linear approx for clear sky
    el_rad = np.radians(max(elevation_deg, 5.0))  # clip to avoid division issues
    return a_zenith / np.sin(el_rad)


def rain_attenuation_db(
    elevation_deg: float, frequency_ghz: float, rain_rate_mmh: float = 0.0
) -> float:
    """Simplified rain attenuation model (ITU-R P.618 approximation).

    For Ka-band, rain causes significant attenuation.
    A_rain ~ k * R^alpha * L_s, where L_s is slant path through rain.
    """
    if rain_rate_mmh <= 0:
        return 0.0
    # ITU-R P.838 coefficients for Ka-band (20 GHz), approximate
    k = 0.0751
    alpha = 1.099
    # Effective rain height ~ 3 km (mid-latitude)
    rain_height_m = 3000
    el_rad = np.radians(max(elevation_deg, 5.0))
    slant_path_m = rain_height_m / np.sin(el_rad)
    specific_atten = k * (rain_rate_mmh ** alpha)  # dB/km
    return specific_atten * (slant_path_m / 1000)


def rician_k_factor_db(elevation_deg: float) -> float:
    """Rician K-factor as a function of elevation angle.

    Higher elevation -> stronger LOS -> higher K-factor.
    Based on 3GPP TR 38.821 Table 6.6.1-1 (simplified).
    """
    # Linear interpolation: K=5 dB at 10°, K=15 dB at 90°
    return 5.0 + (elevation_deg - 10.0) * (10.0 / 80.0)


def generate_rician_fading(
    k_factor_db: float, num_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate Rician fading envelope samples (linear scale).

    Returns power gains (|h|^2) with mean = 1.
    """
    k_linear = 10 ** (k_factor_db / 10)
    # LOS component
    los_amp = np.sqrt(k_linear / (1 + k_linear))
    # Scatter component std
    scatter_std = np.sqrt(1 / (2 * (1 + k_linear)))
    real = los_amp + scatter_std * rng.standard_normal(num_samples)
    imag = scatter_std * rng.standard_normal(num_samples)
    return real ** 2 + imag ** 2


def doppler_shift_hz(
    satellite_velocity_ms: float, frequency_hz: float, elevation_deg: float
) -> float:
    """Doppler shift for LEO satellite.

    f_d = (v * f / c) * cos(elevation).
    Max Doppler at low elevation, zero at zenith.
    """
    el_rad = np.radians(elevation_deg)
    return (satellite_velocity_ms * frequency_hz / SPEED_OF_LIGHT) * np.cos(el_rad)


class NTNChannel:
    """Complete NTN Ka-band channel model for a single beam-to-user link."""

    def __init__(
        self,
        altitude_m: float = 600e3,
        frequency_ghz: float = 20.0,
        bandwidth_hz: float = 400e6,
        satellite_velocity_ms: float = 7560.0,
        noise_temp_k: float = 290.0,
        seed: int = 42,
    ):
        self.altitude_m = altitude_m
        self.frequency_hz = frequency_ghz * 1e9
        self.frequency_ghz = frequency_ghz
        self.bandwidth_hz = bandwidth_hz
        self.satellite_velocity_ms = satellite_velocity_ms
        self.noise_temp_k = noise_temp_k
        self.noise_power_w = BOLTZMANN_K * noise_temp_k * bandwidth_hz
        self.rng = np.random.default_rng(seed)

    def compute_path_loss_db(
        self, elevation_deg: float, rain_rate_mmh: float = 0.0
    ) -> float:
        """Total path loss in dB."""
        dist = slant_range_m(self.altitude_m, elevation_deg)
        fspl = free_space_path_loss_db(dist, self.frequency_hz)
        atm = atmospheric_attenuation_db(elevation_deg, self.frequency_ghz)
        rain = rain_attenuation_db(elevation_deg, self.frequency_ghz, rain_rate_mmh)
        return fspl + atm + rain

    def compute_channel_gain(
        self, elevation_deg: float, rain_rate_mmh: float = 0.0
    ) -> float:
        """Channel power gain (linear) including path loss + Rician fading."""
        pl_db = self.compute_path_loss_db(elevation_deg, rain_rate_mmh)
        pl_linear = 10 ** (-pl_db / 10)
        k_db = rician_k_factor_db(elevation_deg)
        fading = generate_rician_fading(k_db, 1, self.rng)[0]
        return pl_linear * fading

    def compute_snr_db(
        self,
        elevation_deg: float,
        tx_power_w: float,
        antenna_gain_db: float = 40.0,
        rain_rate_mmh: float = 0.0,
    ) -> float:
        """SNR in dB for a single link."""
        channel_gain = self.compute_channel_gain(elevation_deg, rain_rate_mmh)
        ant_gain_linear = 10 ** (antenna_gain_db / 10)
        rx_power = tx_power_w * ant_gain_linear * channel_gain
        snr_linear = rx_power / self.noise_power_w
        return 10 * np.log10(max(snr_linear, 1e-10))

    def compute_capacity_bps(
        self,
        elevation_deg: float,
        tx_power_w: float,
        antenna_gain_db: float = 40.0,
        rain_rate_mmh: float = 0.0,
    ) -> float:
        """Shannon capacity in bps."""
        snr_db = self.compute_snr_db(
            elevation_deg, tx_power_w, antenna_gain_db, rain_rate_mmh
        )
        snr_linear = 10 ** (snr_db / 10)
        return self.bandwidth_hz * np.log2(1 + snr_linear)
