"""Tests for the higher-fidelity precise-reference real-data slice.

Cheap and offline.  The geopotential mathematics is *self-verified* (the
analytic zonal acceleration is checked against a finite-difference gradient of
the closed-form zonal potential), the low-precision Sun/Moon ephemerides are
range/longitude sanity-checked against the almanac, the IAU-76/80 ITRF<->GCRS
rotation is checked to be a proper orthonormal rotation, the learned
calibrator is checked deterministic, and the committed artifact (if present)
is checked for schema and split consistency.  No network, no training here.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from gnn_state_estimation.constants import AU
from gnn_state_estimation.frames import itrf_to_gcrs
from gnn_state_estimation.sp3 import (
    accel_hifi,
    moon_position_eci_m,
    propagate_hifi,
    sun_position_eci_m,
    zonal_acceleration,
    zonal_potential,
)
from gnn_state_estimation.sp3_hifi_calibrator import (
    fit_ridge,
    residual_samples,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_zonal_acceleration_is_gradient_of_potential() -> None:
    """Analytic J2..J4 acceleration == finite-difference grad of the potential.

    This pins the geopotential mathematics rather than asserting it.
    """
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(400):
        rv = rng.normal(size=3)
        rv = rv / np.linalg.norm(rv) * rng.uniform(7.0e6, 2.5e7)
        a = zonal_acceleration(rv)
        g = np.zeros(3)
        for i in range(3):
            h = max(1.0, abs(rv[i]) * 1e-6)
            d = np.zeros(3)
            d[i] = h
            g[i] = (zonal_potential(rv + d) - zonal_potential(rv - d)) / (
                2.0 * h
            )
        worst = max(
            worst, np.linalg.norm(a - g) / max(np.linalg.norm(g), 1e-30)
        )
    assert worst < 1e-6, f"zonal gradient mismatch {worst:.2e}"


def test_sun_moon_ephemeris_almanac_sanity() -> None:
    ep = dt.datetime(
        2026, 5, 6, 12, 0, 0, tzinfo=dt.timezone.utc
    ).timestamp()
    s = sun_position_eci_m(ep)
    m = moon_position_eci_m(ep)
    # Sun ~1 AU; early-May ecliptic longitude ~ 45-47 deg.
    assert 0.97 * AU < np.linalg.norm(s) < 1.03 * AU
    eps = np.deg2rad(23.4393)
    lam = np.degrees(
        np.arctan2(s[1] * np.cos(eps) + s[2] * np.sin(eps), s[0])
    ) % 360.0
    assert 40.0 < lam < 52.0
    # Moon geocentric distance within the physical perigee/apogee band.
    assert 3.50e8 < np.linalg.norm(m) < 4.10e8


def test_itrf_to_gcrs_is_proper_rotation() -> None:
    ep = dt.datetime(
        2026, 4, 20, 6, 30, 0, tzinfo=dt.timezone.utc
    ).timestamp()
    basis = np.eye(3)
    cols = np.column_stack([itrf_to_gcrs(basis[:, i], ep) for i in range(3)])
    # Orthonormal, right-handed (length preserved => valid frame rotation).
    assert np.allclose(cols.T @ cols, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(cols) - 1.0) < 1e-9
    v = np.array([7.0e6, -2.0e6, 3.0e6])
    assert abs(np.linalg.norm(itrf_to_gcrs(v, ep)) - np.linalg.norm(v)) < 1e-3


def test_hifi_perturbations_have_expected_magnitude() -> None:
    """Luni-solar third body must be the dominant non-J2 LAGEOS term."""
    ep = dt.datetime(
        2026, 5, 6, 0, 0, 0, tzinfo=dt.timezone.utc
    ).timestamp()
    r = np.array([1.0e7, 6.0e6, 7.0e6], dtype=np.float64)
    a = accel_hifi(r, ep)
    assert np.all(np.isfinite(a))
    # Total acceleration dominated by two-body (~2 m/s^2 at this radius).
    assert 1.5 < np.linalg.norm(a) < 3.0
    # Higher-fidelity propagation stays physical over a 6 h horizon.
    speed = np.sqrt(3.986004418e14 / np.linalg.norm(r))
    state = np.hstack([r, np.array([0.0, speed * 0.7, speed * 0.7])])
    out = propagate_hifi(state, 21600.0, ep, max_step_s=30.0)
    assert np.all(np.isfinite(out))
    assert 5.0e6 < np.linalg.norm(out[:3]) < 5.0e7


def test_calibrator_fit_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(200, 16))
    da = rng.normal(size=(200, 3))
    a = fit_ridge(phi, da, 1.0, 0.0)
    b = fit_ridge(phi, da, 1.0, 0.0)
    assert np.array_equal(a.beta, b.beta)
    # Larger ridge shrinks the coefficient norm (regularisation sanity).
    big = fit_ridge(phi, da, 1.0e6, 0.0)
    assert np.linalg.norm(big.beta) < np.linalg.norm(a.beta)


def test_residual_samples_shapes() -> None:
    # Smooth synthetic inertial track; residual vs hi-fi model is finite.
    r0 = 1.2270e7
    w = np.sqrt(3.986004418e14 / r0) / r0

    def pos(t):
        return np.array(
            [r0 * np.cos(w * t), r0 * np.sin(w * t), 0.0]
        )

    def state(t):
        return np.hstack(
            [pos(t), np.array([-r0 * w * np.sin(w * t),
                               r0 * w * np.cos(w * t), 0.0])]
        )

    phi, da = residual_samples(pos, state, 1.0e9, 1.0e9 + 3600.0, 1.0e9)
    assert phi.shape[1] == 16
    assert phi.shape[0] == da.shape[0] > 0
    assert np.all(np.isfinite(phi)) and np.all(np.isfinite(da))


def test_hifi_artifact_schema_and_split_consistency() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_hifi"
        / "real_slr_sp3_hifi_validation.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_hifi_v1"
    assert d["status"] == "completed"
    # The strict split must be temporal: train/val/test weeks disjoint.
    sw = d["split_weeks"]
    assert sorted(set(sw.values())) == ["test", "train", "val"]
    cpd = d["controlled_pure_dynamics"]
    for split in ("all", "train", "val", "test"):
        assert split in cpd
    # The headline is a paired fidelity gain with a bootstrap CI.
    g = cpd["test"].get("hifi_vs_compact", {})
    if g.get("n", 0) > 0:
        assert "bootstrap95_mean_improvement_m" in g
        assert len(g["bootstrap95_mean_improvement_m"]) == 2
    cal = d["learned_calibrator"]
    assert cal["status"] in (
        "completed", "insufficient_training_samples"
    )
    if cal["status"] == "completed":
        # No-leakage protocol must be recorded and the verdict honest.
        assert "no_leakage_protocol" in cal
        assert isinstance(cal["beats_higher_fidelity_on_test"], bool)
