"""Tests for the SP3-c precise-reference parser, range-only filters, and the
externally defined DBAR validation slice.

These are cheap and offline: SP3 parsing/interpolation and the self-contained
range-only filters are exercised on tiny synthetic inputs, and the committed
artifact (if present) is checked for schema and confusion consistency. No
network access and no model training here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnn_state_estimation.sp3 import (
    RangeObs,
    Sp3Interpolator,
    circular_orbit_speed_mps,
    held_out_position_rmse,
    mean_r_eff_scale,
    median_nis_r,
    parse_sp3,
    propagate_compact,
    run_range_aukf,
    run_range_ekf,
    run_range_ukf_fixed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_sp3() -> str:
    """A minimal valid SP3-c snippet: a LAGEOS-like circular arc for L51.

    Twelve 120-second nodes on a circle of radius ~12 270 km in the XY plane,
    so a downstream Lagrange interpolation has a smooth ground truth.
    """
    r = 1.2270e7  # m
    speed = circular_orbit_speed_mps(r)
    omega = speed / r
    lines = [
        "#cP2026  5  3  0  0  0.00000000      12 SLR   ECF FIT TEST",
        "## 2417      0.00000000   120.00000000 61163 0.0000000000000",
        "+    1   L51  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "%c L  cc UTC ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "/* synthetic test product",
    ]
    base_y, base_mo, base_d = 2026, 5, 3
    for k in range(12):
        sec = 120.0 * k
        ang = omega * sec
        x_km = (r * np.cos(ang)) / 1e3
        y_km = (r * np.sin(ang)) / 1e3
        z_km = 0.0
        vx_dm = (-r * omega * np.sin(ang)) * 10.0  # m/s -> dm/s
        vy_dm = (r * omega * np.cos(ang)) * 10.0
        mm = int(sec // 60)
        ss = sec - 60 * mm
        lines.append(
            f"*  {base_y}  {base_mo}  {base_d}  0 {mm:2d} {ss:11.8f}"
        )
        lines.append(f"PL51 {x_km:14.6f} {y_km:14.6f} {z_km:14.6f}")
        lines.append(f"VL51 {vx_dm:14.6f} {vy_dm:14.6f} {0.0:14.6f}")
    lines.append("EOF")
    return "\n".join(lines)


def test_parse_sp3_units_and_metadata() -> None:
    eph = parse_sp3(_synthetic_sp3(), "L51")
    assert eph.sat_id == "L51"
    assert eph.epochs_unix.size == 12
    assert np.all(np.diff(eph.epochs_unix) > 0)
    # 120 s nodal spacing recovered from the epoch records.
    assert abs(float(np.median(np.diff(eph.epochs_unix))) - 120.0) < 1e-6
    # Positions are km -> m: |r| ~ 12 270 km.
    radii = np.linalg.norm(eph.positions_m, axis=1)
    assert np.allclose(radii, 1.2270e7, rtol=1e-6)
    # Velocities are dm/s -> m/s: circular speed ~ 5.7 km/s.
    speeds = np.linalg.norm(eph.velocities_mps, axis=1)
    assert np.all((speeds > 5.0e3) & (speeds < 6.5e3))
    assert eph.time_system == "UTC"


def test_sp3_interpolator_node_exactness_and_smoothness() -> None:
    eph = parse_sp3(_synthetic_sp3(), "L51")
    interp = Sp3Interpolator(eph, order=9)
    # Lagrange interpolation is exact at the sample nodes.
    for idx in (3, 6, 9):
        t = float(eph.epochs_unix[idx])
        got = interp.position_ecef_m(t)
        assert np.allclose(got, eph.positions_m[idx], atol=1e-3)
    # Off-node points stay on the circular ground truth.
    t_mid = float(0.5 * (eph.epochs_unix[5] + eph.epochs_unix[6]))
    r_mid = np.linalg.norm(interp.position_ecef_m(t_mid))
    assert abs(r_mid - 1.2270e7) < 1.0e3


def test_propagate_compact_is_reversible() -> None:
    r = 1.2270e7
    speed = circular_orbit_speed_mps(r)
    state = np.array([r, 0.0, 0.0, 0.0, speed, 0.0], dtype=np.float64)
    fwd = propagate_compact(state, 1800.0, max_step_s=30.0)
    back = propagate_compact(fwd, -1800.0, max_step_s=30.0)
    assert np.linalg.norm(back[:3] - state[:3]) < 50.0
    # Orbit radius stays physical (no blow-up) over half an hour.
    assert 1.0e7 < np.linalg.norm(fwd[:3]) < 1.5e7


def test_range_filters_track_consistent_observations() -> None:
    """With self-consistent range observations the filters stay bounded and the
    fixed-noise UKF reports no measurement-noise inflation."""
    r = 1.2270e7
    speed = circular_orbit_speed_mps(r)
    x_true = np.array([r, 0.0, 0.0, 0.0, speed, 0.0], dtype=np.float64)
    station = np.array([6.371e6, 0.0, 0.0], dtype=np.float64)
    obs = []
    t0 = 1.0e9
    state = x_true.copy()
    for k in range(20):
        t = t0 + 60.0 * k
        if k > 0:
            state = propagate_compact(state, 60.0, 30.0)
        rng = float(np.linalg.norm(state[:3] - station))
        obs.append(RangeObs(epoch_unix=t, station_pi_m=station, range_m=rng))
    x0 = x_true + np.array([200.0, -150.0, 0.0, 0.1, 0.1, 0.0])
    p0 = np.diag([1e4, 1e4, 1e4, 1.0, 1.0, 1.0]).astype(np.float64)
    ekf = run_range_ekf(obs, x0, p0, 20.0, 1e-12)
    ukf = run_range_ukf_fixed(obs, x0, p0, 20.0, 1e-12)
    aukf = run_range_aukf(obs, x0, p0, 20.0, 1e-12)
    for res in (ekf, ukf, aukf):
        assert np.all(np.isfinite(res["state"]))
        assert len(res["records"]) == len(obs)
    # The fixed-noise UKF never inflates R; the adaptive scale is >= 1.
    assert all(rec["r_eff_scale"] == 1.0 for rec in ukf["records"])
    assert all(rec["r_eff_scale"] >= 1.0 for rec in aukf["records"])
    assert median_nis_r(ukf["records"]) >= 0.0
    assert mean_r_eff_scale(ukf["records"]) == 1.0


def test_held_out_rmse_finite() -> None:
    r = 1.2270e7
    speed = circular_orbit_speed_mps(r)
    state = np.array([r, 0.0, 0.0, 0.0, speed, 0.0], dtype=np.float64)
    eph = parse_sp3(_synthetic_sp3(), "L51")
    interp = Sp3Interpolator(eph, order=9)
    held = held_out_position_rmse(
        state,
        float(eph.epochs_unix[0]),
        eph.epochs_unix[2:6],
        interp,
        max_step_s=30.0,
    )
    assert held["count"] == 4
    assert np.isfinite(held["rms_m"])


def test_external_dbar_label_is_not_self_referential() -> None:
    """The committed artifact's outcome label must be external precise-SP3
    state error, never the DBAR statistic or a simulator self-twin."""
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_od"
        / "real_slr_sp3_od_validation.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_od_v1"
    assert d["status"] == "completed"
    pooled = d["pooled_held_out_position_rmse_m"]
    for name in (
        "EKF",
        "UKF (fixed-noise)",
        "AUKF (adaptive)",
        "SP3-IC propagation",
    ):
        assert name in pooled
    ext = d["dbar_external_validation"]
    c = ext["confusion"]
    total = (
        c["true_fire"]
        + c["true_no_fire"]
        + c["false_fire"]
        + c["false_no_fire"]
    )
    assert total == ext["n_arcs_scored"]
    assert ext["n_correct"] == c["true_fire"] + c["true_no_fire"]
    label = ext["external_label_definition"].lower()
    assert "external precise reference" in label
    assert "not the dbar statistic" in label
    # Every completed arc carries an external (SP3) outcome, not a self label.
    for arc in d["arcs"]:
        if arc.get("status") != "completed":
            continue
        db = arc["dbar"]
        assert "adaptation_counterproductive_external" in db
        assert "dbar_correct_external" in db
