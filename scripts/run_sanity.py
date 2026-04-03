#!/usr/bin/env python3
"""M0 Sanity checks: verify simulator, MDP spec, and environment."""

import sys
import json
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


def check_channel():
    """R001: Verify channel model produces realistic values."""
    from simulator.channel import NTNChannel, slant_range_m, free_space_path_loss_db

    print("=== R001: Channel Model ===")
    ch = NTNChannel()

    # Check slant range
    d_90 = slant_range_m(600e3, 90.0)
    d_30 = slant_range_m(600e3, 30.0)
    d_10 = slant_range_m(600e3, 10.0)
    print(f"  Slant range (90°): {d_90/1e3:.1f} km (expect ~600)")
    print(f"  Slant range (30°): {d_30/1e3:.1f} km (expect ~1040)")
    print(f"  Slant range (10°): {d_10/1e3:.1f} km (expect ~1930)")
    assert 590 < d_90 / 1e3 < 610, f"Bad 90° range: {d_90/1e3}"
    assert 900 < d_30 / 1e3 < 1200, f"Bad 30° range: {d_30/1e3}"

    # Check path loss
    pl_90 = ch.compute_path_loss_db(90.0)
    pl_30 = ch.compute_path_loss_db(30.0)
    print(f"  Path loss (90°): {pl_90:.1f} dB")
    print(f"  Path loss (30°): {pl_30:.1f} dB")
    assert 170 < pl_90 < 210, f"Path loss out of range: {pl_90}"
    assert pl_30 > pl_90, "Path loss should increase at lower elevation"

    # Check SNR
    snr_90 = ch.compute_snr_db(90.0, tx_power_w=10.0)
    print(f"  SNR (90°, 10W, 40dBi): {snr_90:.1f} dB")

    # Check capacity
    cap = ch.compute_capacity_bps(90.0, tx_power_w=10.0)
    print(f"  Shannon capacity (90°): {cap/1e6:.1f} Mbps")
    assert cap > 0, "Capacity should be positive"

    print("  [PASS] Channel model OK\n")


def check_satellite():
    """R001: Verify satellite beam geometry."""
    from simulator.satellite import LEOSatellite

    print("=== R001: Satellite Geometry ===")
    sat = LEOSatellite(num_rings=2)
    summary = sat.get_state_summary()

    print(f"  Beams: {summary['num_beams']} (expect 19)")
    print(f"  Altitude: {summary['altitude_km']} km")
    print(f"  Beam radius: {summary['beam_radius_km']:.1f} km")
    print(f"  Elevation range: {summary['min_elevation_deg']:.1f}° - {summary['max_elevation_deg']:.1f}°")

    assert summary["num_beams"] == 19, f"Expected 19 beams, got {summary['num_beams']}"
    assert summary["altitude_km"] == 600.0

    # Check adjacency
    adj_count = sat.adjacency.sum()
    print(f"  Adjacent beam pairs: {adj_count // 2}")
    assert adj_count > 0, "Should have some adjacent beams"

    # Check interference
    active = np.ones(19, dtype=bool)
    power = np.ones(19) * 5.0
    interference = sat.inter_beam_interference(active, power)
    print(f"  Max interference (all beams active): {interference.max():.4f} W")
    assert interference.max() > 0, "Should have some interference"

    print("  [PASS] Satellite geometry OK\n")


def check_traffic():
    """R001: Verify traffic generators."""
    from simulator.traffic import RegimeSequence, RegimeType

    print("=== R001: Traffic Generators ===")
    seq = RegimeSequence(
        num_beams=19,
        regime_sequence=[RegimeType.URBAN, RegimeType.MARITIME, RegimeType.DISASTER],
        epochs_per_regime=100,
    )

    # Sample from urban
    demand = seq.sample()
    kpi = seq.get_kpi_snapshot(demand)
    print(f"  Urban: avg={kpi['avg_demand']:.1f} Mbps, gini={kpi['spatial_gini']:.3f}")
    assert kpi["avg_demand"] > 20, "Urban demand too low"
    assert kpi["spatial_gini"] > 0.1, "Urban should be spatially concentrated"

    # Advance to maritime
    for _ in range(100):
        seq.step()
    demand2 = seq.sample()
    kpi2 = seq.get_kpi_snapshot(demand2)
    print(f"  Maritime: avg={kpi2['avg_demand']:.1f} Mbps, gini={kpi2['spatial_gini']:.3f}")
    assert kpi2["avg_demand"] < kpi["avg_demand"], "Maritime should have lower demand"

    # Advance to disaster
    for _ in range(100):
        seq.step()
    demand3 = seq.sample()
    kpi3 = seq.get_kpi_snapshot(demand3)
    print(f"  Disaster: peak={kpi3['peak_beam_demand']:.1f} Mbps")
    assert kpi3["peak_beam_demand"] > 100, "Disaster should have high peak demand"

    print("  [PASS] Traffic generators OK\n")


def check_env():
    """R002: Verify Gymnasium environment."""
    from simulator.env import BeamAllocationEnv, FlatActionWrapper

    print("=== R002: Gymnasium Environment ===")
    env = BeamAllocationEnv(
        regime_sequence=["urban"],
        epochs_per_regime=50,
    )

    obs, info = env.reset()
    n = env.num_beams
    expected_dim = 3 * n + 3
    print(f"  Obs shape: {obs.shape} (expect ({expected_dim},))")
    assert obs.shape == (expected_dim,), f"Bad obs shape: {obs.shape}, expected ({expected_dim},)"
    assert info["regime"] == "urban"

    # Take a random action
    n = env.num_beams
    action = {
        "beam_activation": np.random.randint(0, 2, n).astype(np.int8),
        "power_allocation": np.random.uniform(0, 1, n).astype(np.float32),
    }
    obs2, reward, term, trunc, info2 = env.step(action)
    print(f"  Step reward: {reward:.3f}")
    print(f"  Sum rate: {info2['sum_rate_mbps']:.1f} Mbps")
    print(f"  Outage count: {info2['outage_count']}")
    assert isinstance(reward, float)
    assert obs2.shape == obs.shape

    # Test flat wrapper
    print("  Testing FlatActionWrapper...")
    flat_env = FlatActionWrapper(env)
    obs3, _ = flat_env.reset()
    flat_action = flat_env.action_space.sample()
    obs4, r, _, _, _ = flat_env.step(flat_action)
    print(f"  Flat action shape: {flat_action.shape} (expect ({2*19},))")
    assert flat_action.shape == (2 * 19,)

    print("  [PASS] Environment OK\n")


def check_mdp_spec():
    """R003: Verify MDP spec validation."""
    from mdp.spec import MDPSpec, validate_spec
    from mdp.default_specs import get_default_spec

    print("=== R003: MDP Spec Validation ===")

    for regime in ["urban", "maritime", "disaster", "mixed"]:
        spec = get_default_spec(regime)
        valid, err = validate_spec(spec)
        print(f"  {regime} spec: valid={valid}")
        assert valid, f"Spec {regime} failed validation: {err}"

        # Round-trip test
        json_str = spec.to_json()
        spec2 = MDPSpec.from_json(json_str)
        assert spec2.spec_id == spec.spec_id
        assert spec2.state_features == spec.state_features

    # Test invalid spec
    bad_spec = MDPSpec(
        spec_id="bad",
        state_features=["nonexistent_feature"],
        action_type="per_beam",
        reward_components=[],
    )
    valid, err = validate_spec(bad_spec)
    print(f"  Invalid spec correctly rejected: {not valid}")
    assert not valid, "Should reject invalid spec"

    print("  [PASS] MDP spec validation OK\n")


def main():
    print("\n" + "=" * 60)
    print("  M0 SANITY CHECKS — satcom-llm-drl")
    print("=" * 60 + "\n")

    checks = {
        "channel": check_channel,
        "satellite": check_satellite,
        "traffic": check_traffic,
        "env": check_env,
        "mdp": check_mdp_spec,
    }

    # Parse optional --check argument
    if len(sys.argv) > 2 and sys.argv[1] == "--check":
        selected = sys.argv[2]
        if selected in checks:
            checks[selected]()
        else:
            print(f"Unknown check: {selected}. Available: {list(checks.keys())}")
            sys.exit(1)
    else:
        # Run all
        passed = 0
        failed = 0
        for name, fn in checks.items():
            try:
                fn()
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}\n")
                failed += 1

        print("=" * 60)
        print(f"  Results: {passed} passed, {failed} failed")
        print("=" * 60)
        sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
