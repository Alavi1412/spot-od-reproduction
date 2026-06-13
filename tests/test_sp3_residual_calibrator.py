"""Tests for the held-out SP3-supervised dynamics-residual calibrator.

Cheap and offline: the RSW frame, Fourier basis, ridge fit, and corrected
propagation are exercised on tiny synthetic inputs, and the committed
artifact (if present) is checked for schema and honest-verdict consistency.
No network access and no expensive sweep here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnn_state_estimation.sp3 import circular_orbit_speed_mps, propagate_compact
from gnn_state_estimation.sp3_calibrator import (
    FOURIER_ORDER,
    LAGEOS_BALLISTIC_COEFF,
    ResidualCalibrator,
    argument_of_latitude,
    compact_acceleration,
    fit_calibrator,
    fourier_features,
    propagate_corrected,
    rsw_basis,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _state() -> np.ndarray:
    r = 1.2270e7
    speed = circular_orbit_speed_mps(r)
    return np.array([r, 0.0, 0.0, 0.0, speed * 0.9, speed * 0.3],
                    dtype=np.float64)


def test_ballistic_coeff_matches_sp3_module() -> None:
    import gnn_state_estimation.sp3 as sp3_mod

    assert LAGEOS_BALLISTIC_COEFF == sp3_mod._LAGEOS_BALLISTIC_COEFF


def test_rsw_basis_is_orthonormal_right_handed() -> None:
    r_hat, s_hat, w_hat = rsw_basis(_state())
    for u in (r_hat, s_hat, w_hat):
        assert abs(np.linalg.norm(u) - 1.0) < 1e-9
    assert abs(np.dot(r_hat, s_hat)) < 1e-9
    assert abs(np.dot(r_hat, w_hat)) < 1e-9
    assert abs(np.dot(s_hat, w_hat)) < 1e-9
    # S = W x R closes the right-handed triad.
    assert np.allclose(np.cross(w_hat, r_hat), s_hat, atol=1e-9)


def test_argument_of_latitude_and_fourier_shapes() -> None:
    u = argument_of_latitude(_state())
    assert -np.pi <= u <= np.pi
    phi = fourier_features(u)
    assert phi.shape == (2 * FOURIER_ORDER + 1,)
    assert abs(phi[0] - 1.0) < 1e-12


def test_fit_recovers_constant_radial_field() -> None:
    """A pure constant radial residual must be recovered by the LSQ fit."""
    rng = np.random.default_rng(0)
    phis = []
    das = []
    for _ in range(200):
        u = rng.uniform(-np.pi, np.pi)
        phis.append(fourier_features(u))
        das.append([0.0123, 0.0, 0.0])  # constant radial accel
    cal = fit_calibrator(np.vstack(phis), np.vstack(das))
    assert cal.n_train_samples == 200
    # Constant term of the radial row dominates; reconstructs ~0.0123.
    out = cal.beta[0] @ fourier_features(0.7)
    assert abs(out - 0.0123) < 1e-6
    # Along-track / cross-track rows stay ~0.
    assert abs(cal.beta[1] @ fourier_features(0.7)) < 1e-6


def test_zero_calibrator_matches_compact_propagation() -> None:
    n_feat = 2 * FOURIER_ORDER + 1
    zero = ResidualCalibrator(
        beta=np.zeros((3, n_feat)),
        order=FOURIER_ORDER,
        ridge_lambda=0.0,
        n_train_samples=0,
    )
    s = _state()
    base = propagate_compact(s, 600.0, max_step_s=30.0)
    corr = propagate_corrected(s, 600.0, zero, max_step_s=30.0)
    assert np.allclose(base, corr, atol=1e-6)


def test_compact_acceleration_is_two_body_dominated() -> None:
    s = _state()
    a = compact_acceleration(s)
    r = np.linalg.norm(s[:3])
    a_two_body = 398600441800000.0 / r**2
    # J2 is a sub-1% perturbation at LAGEOS altitude.
    assert 0.9 * a_two_body < np.linalg.norm(a) < 1.1 * a_two_body


def test_committed_artifact_is_honest_negative_if_present() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_od"
        / "sp3_residual_calibrator.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "sp3_residual_calibrator_v1"
    pooled = d["pooled_held_out_position_rmse_m"]
    for proto in ("uncalibrated", "loao_calibrated", "looo_calibrated"):
        assert proto in pooled
    v = d["verdict"]
    # The artifact must carry an explicit, consistent honest verdict: the
    # calibrator is claimed positive only if it beats the best uncalibrated
    # classical reference on BOTH no-leakage protocols.
    claimed = v["claimed_as_positive_contribution"]
    both = (
        v["loao"]["beats_best_uncalibrated_reference"]
        and v["looo"]["beats_best_uncalibrated_reference"]
    )
    assert claimed == both
    assert "no_leakage_protocol" in d
