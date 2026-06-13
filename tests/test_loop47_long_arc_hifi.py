"""Tests for the Loop 47 long-arc higher-fidelity force/density slice.

Cheap and offline. Pins the analytic tesseral C(2,2)/S(2,2) acceleration as
the gradient of its potential against a central finite-difference, confirms
that the long-arc acceleration is non-degenerate (sectoral term moves the
acceleration relative to the J2..J6+luni-solar baseline), and validates the
Loop-47 predeclaration and floor artefacts have the expected schemas. No
network, no training here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnn_state_estimation.sp3 import (
    accel_hifi_extended,
    accel_hifi_long_arc,
    tesseral_22_acceleration_ecef,
    tesseral_22_acceleration_inertial,
    tesseral_22_potential_ecef,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tesseral_acceleration_is_gradient_of_potential() -> None:
    """C(2,2)/S(2,2) acceleration in ECEF equals the finite-difference
    gradient of the corresponding potential. Pins the sectoral
    spherical-harmonic mathematics rather than asserting it.
    """
    rng = np.random.default_rng(47)
    worst = 0.0
    for _ in range(300):
        rv = rng.normal(size=3)
        rv = rv / np.linalg.norm(rv) * rng.uniform(7.0e6, 2.5e7)
        a = tesseral_22_acceleration_ecef(rv)
        g = np.zeros(3)
        for i in range(3):
            h = max(1.0, abs(rv[i]) * 1e-6)
            d = np.zeros(3)
            d[i] = h
            g[i] = (
                tesseral_22_potential_ecef(rv + d)
                - tesseral_22_potential_ecef(rv - d)
            ) / (2.0 * h)
        denom = max(float(np.linalg.norm(g)), 1e-30)
        worst = max(worst, float(np.linalg.norm(a - g) / denom))
    assert worst < 1e-5, f"tesseral C22/S22 gradient mismatch {worst:.2e}"


def test_long_arc_acceleration_includes_sectoral_term() -> None:
    """The long-arc propagator is materially different from the extended
    propagator: the sectoral C(2,2)/S(2,2) channel produces a non-negligible
    acceleration shift that the J2..J6+luni-solar propagator does not.
    """
    rng = np.random.default_rng(471)
    deltas = []
    for _ in range(40):
        # LEO range so the sectoral term is at meaningful scale.
        rv = rng.normal(size=3)
        rv = rv / np.linalg.norm(rv) * rng.uniform(6.8e6, 7.5e6)
        ep = 1.736_640_000e9 / 1e3  # any epoch in the GMST domain
        ep = 1_736_640_000.0
        a_ext = accel_hifi_extended(rv, ep)
        a_long = accel_hifi_long_arc(rv, ep)
        deltas.append(float(np.linalg.norm(a_long - a_ext)))
    # The sectoral term is on the order of 1e-5 m/s^2 at LEO altitudes; we
    # require the median to be at least 1e-7 (well above numerical noise).
    median_delta = float(np.median(deltas))
    assert median_delta > 1e-7, f"sectoral term too small: median {median_delta:.3e}"


def test_long_arc_predeclaration_schema() -> None:
    """The Loop 47 predeclaration JSON has the expected schema, including
    the disjoint validation/test seeds, the predeclared arc length, and
    the link to the astrodynamics-grounded floor artefact.
    """
    pred = json.loads(
        (REPO_ROOT / "release" / "predeclarations" / "long_arc_hifi_rule_loop47.json")
        .read_text()
    )
    assert pred["predeclared_on_utc"] == "2026-05-20"
    assert pred["arc"]["steps"] == 540
    assert pred["arc"]["dt_s"] == 20.0
    assert pred["arc"]["arc_length_s"] == 10800.0
    vt = pred["validation_tuning"]
    ep = pred["evaluation_protocol"]
    assert vt["validation_seed"] != ep["test_seed"]
    assert ep["test_seed"] not in {20260645, 20260545, 770000, 42}
    assert ep["astrodynamics_floor_artifact"].endswith(
        "astrodynamics_floor_loop47.json"
    )
    assert ep["n_trajectories_planned"] == 36
    # Grid must contain at least 3 points (predeclared 4) and span >=
    # the prior 600 s decorrelation time.
    grid = vt["tuning_grid"]
    assert len(grid) >= 3
    taus = [float(g["drag_scale_tau_s"]) for g in grid]
    assert max(taus) >= 1800.0
    th = pred["thresholds"]
    assert "tesseral_C22_normalized" in th
    assert "tesseral_S22_normalized" in th
    assert "alpha_longitudinal" in th


def test_astrodynamics_floor_schema() -> None:
    """The Loop 47 astrodynamics-grounded floor JSON has the expected
    derivation steps and reports an absolute metres floor.
    """
    f = json.loads(
        (REPO_ROOT / "release" / "predeclarations" / "astrodynamics_floor_loop47.json")
        .read_text()
    )
    assert f["schema_version"] == "astrodynamics_floor_v2"
    d = f["derivation"]
    assert d["step_1_per_update_position_sigma_m"] > 0.0
    assert 0.0 < d["step_2_single_station_visibility_fraction"] < 1.0
    assert d["step_3_network_visibility_fraction_upper_bound"] > 0.0
    assert d["step_4_expected_independent_observations_per_arc"] > 1.0
    assert d["step_5_per_arc_position_sigma_m"] > 0.0
    floor_m = float(f["practical_significance_floor_m_absolute"])
    # Sanity bounds: the per-arc floor must be lower than the per-update
    # floor and strictly positive.
    assert 1.0 < floor_m < d["step_1_per_update_position_sigma_m"]


def test_tesseral_acceleration_inertial_uses_gmst() -> None:
    """The inertial-frame sectoral acceleration changes with epoch (because
    the C(2,2)/S(2,2) channel is Earth-rotation-tracked), so two different
    epochs at the same inertial position must give different accelerations.
    """
    r = np.array([7.0e6, 1.5e6, 1.0e6], dtype=np.float64)
    a1 = tesseral_22_acceleration_inertial(r, epoch_unix=1_736_640_000.0)
    a2 = tesseral_22_acceleration_inertial(r, epoch_unix=1_736_643_600.0)  # +1 h
    # The sectoral acceleration should rotate over one hour by a non-trivial
    # angle (~15 deg of Earth rotation).
    assert np.linalg.norm(a1 - a2) > 1e-9
