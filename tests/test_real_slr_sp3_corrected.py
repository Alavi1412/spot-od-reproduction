"""Tests for the operationally-corrected precise-reference sanity probe.

Cheap and offline: the IERS EOP parser/interpolator, the corrected
ITRF->GCRS transform, the Marini--Murray troposphere, the Shapiro delay, and
the CRD meteorology/wavelength parsing are exercised on tiny synthetic inputs,
and the committed artifact (if present) is checked for schema, the
correction-sensitivity-audit structure, and honest-bounded framing.  No
network access and no model training here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnn_state_estimation.eop import EopSeries, parse_finals2000a_csv
from gnn_state_estimation.frames import itrf_to_gcrs, itrf_to_gcrs_eop
from gnn_state_estimation.slr import (
    LAGEOS_CENTRE_OF_MASS_OFFSET_M,
    marini_murray_range_correction_m,
    parse_crd_v2_meteorology,
    parse_crd_v2_transmit_wavelength_nm,
    shapiro_delay_m,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_SYNTH_EOP_CSV = (
    "MJD;Year;Month;Day;Type;x_pole;sigma_x_pole;y_pole;sigma_y_pole;"
    "x_rate;sigma_x_rate;y_rate;sigma_y_rate;Type;UT1-UTC\n"
    "61165;2026;05;05;final;0.100000;0;0.200000;0;0;0;0;0;final;0.030000\n"
    "61166;2026;05;06;final;0.200000;0;0.400000;0;0;0;0;0;final;0.050000\n"
    "61167;2026;05;07;final;0.300000;0;0.600000;0;0;0;0;0;final;0.070000\n"
)


def test_eop_parser_and_linear_interpolation() -> None:
    eop = parse_finals2000a_csv(_SYNTH_EOP_CSV)
    assert isinstance(eop, EopSeries)
    assert eop.mjd.size == 3
    # MJD 61165.5 is exactly halfway between rows one and two.
    mid_unix = (61165.5 - 40587.0) * 86400.0
    xp, yp = eop.polar_motion_rad(mid_unix)
    arcsec = 180.0 * 3600.0 / np.pi
    assert abs(xp * arcsec - 0.15) < 1e-6
    assert abs(yp * arcsec - 0.30) < 1e-6
    assert abs(eop.ut1_minus_utc_s(mid_unix) - 0.040) < 1e-9
    assert eop.covers(mid_unix)


def test_corrected_frame_reduces_to_analytic_when_eop_zero() -> None:
    r_itrf = np.array([1.1e7, 3.0e6, 4.0e6], dtype=np.float64)
    epoch = (61166.0 - 40587.0) * 86400.0
    analytic = itrf_to_gcrs(r_itrf, epoch)
    zero_eop = itrf_to_gcrs_eop(r_itrf, epoch, 0.0, 0.0, 0.0)
    assert np.allclose(analytic, zero_eop, atol=1e-7)
    # Real-magnitude EOP is a rotation: norm preserved, shift bounded (tens of
    # metres at the LAGEOS radius for ~0.3 arcsec / 0.05 s).
    with_eop = itrf_to_gcrs_eop(
        r_itrf, epoch, 1.5e-6, 2.0e-6, 0.05
    )
    assert abs(np.linalg.norm(with_eop) - np.linalg.norm(r_itrf)) < 1e-6
    assert 0.0 < np.linalg.norm(with_eop - analytic) < 200.0


def test_marini_murray_is_physical() -> None:
    phi = np.deg2rad(-29.0)
    z = marini_murray_range_correction_m(
        np.pi / 2.0, 1013.25, 288.15, 50.0, phi, 0.0, 0.532
    )
    # Standard-atmosphere zenith delay is ~2.4--2.6 m for a green SLR laser.
    assert 2.0 < z < 3.0
    low = marini_murray_range_correction_m(
        np.deg2rad(15.0), 1013.25, 288.15, 50.0, phi, 0.0, 0.532
    )
    # Monotone increase toward the horizon (~1/sin E behaviour).
    assert low > z * 2.0
    # Higher surface pressure increases the delay.
    hi_p = marini_murray_range_correction_m(
        np.pi / 2.0, 1030.0, 288.15, 50.0, phi, 0.0, 0.532
    )
    assert hi_p > z


def test_shapiro_delay_scale() -> None:
    station = np.array([4.7e6, 2.6e6, -3.07e6], dtype=np.float64)
    sat = np.array([8.0e6, 7.0e6, 4.0e6], dtype=np.float64)
    dr = shapiro_delay_m(station, sat)
    # A few millimetres to ~centimetre for a LAGEOS-scale geometry.
    assert 1e-3 < dr < 5e-2
    assert LAGEOS_CENTRE_OF_MASS_OFFSET_M == 0.251


def test_crd_meteorology_and_wavelength_parser() -> None:
    crd = (
        "h4  1 2026  5  9  7 26 58 2026  5  9  7 38  3  0 0 0 1 1 0 2 0\n"
        "c0 0 532.000 std1 ml1 mcp mt1 met\n"
        "11 26841.30 0.0464219147257 std1 2 120.0 29 26.7 0.662 1.278 na\n"
        "20 26841.30  960.12 289.40  65. 0\n"
        "20 26937.90  960.20 289.70  63. 0\n"
        "h8\n"
    )
    met = parse_crd_v2_meteorology(crd)
    assert len(met) == 2
    assert abs(met[0].pressure_hpa - 960.12) < 1e-6
    assert abs(met[0].temperature_k - 289.40) < 1e-6
    assert abs(met[0].humidity_pct - 65.0) < 1e-6
    assert parse_crd_v2_transmit_wavelength_nm(crd) == 532.0


def test_committed_corrected_artifact_is_bounded_and_audited() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "real_slr_sp3_corrected"
        / "real_slr_sp3_corrected_validation.json"
    )
    if not path.exists():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["schema_version"] == "real_slr_sp3_corrected_v1"
    assert d["status"] == "completed"
    pooled = d["pooled_held_out_position_rmse_m"]
    for cfg in (
        "full",
        "no_eop",
        "no_troposphere",
        "no_centre_of_mass",
        "no_relativity",
        "gmst_only",
    ):
        assert cfg in pooled
        for est in (
            "EKF",
            "UKF (fixed-noise)",
            "AUKF (adaptive)",
            "SP3-IC propagation",
        ):
            assert est in pooled[cfg]
    # The sensitivity audit reports a per-correction delta vs the full stack.
    audit = d["correction_sensitivity_audit"]
    assert "no_eop" in audit and "delta_vs_full_m" in audit["no_eop"]
    h2h = d["head_to_head_vs_committed_gmst_only"]
    assert "committed_real_slr_sp3_od_mean_m" in h2h
    assert "corrected_full_mean_m" in h2h
    # Honest framing: explicitly a bounded sanity probe, not operational OD.
    cav = d["caveats"].lower()
    assert "not a centimetre operational" in cav
    assert "bounded sanity" in cav
    # The EOP series is the real public IERS product (many tabulated rows).
    assert d["eop_series"]["n_rows"] > 1000
