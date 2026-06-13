"""Tests for the real ILRS SLR CRD parser, range conversion, and schema."""

from __future__ import annotations

import datetime as dt

import numpy as np

from gnn_state_estimation.slr import (
    SPEED_OF_LIGHT_MPS,
    SUPPORTED_STATIONS,
    gmst_rad,
    one_way_range_m,
    parse_crd_v2_normal_points,
    station_pseudo_inertial_m,
    summarize_residuals,
)

# Minimal CRD v2 snippet: one supported station (YARL 7090) with two normal
# points, one unsupported station (CDP 1234) that must be filtered out, and a
# second supported pass (WETL 8834) whose first NP crosses UTC midnight.
CRD_SNIPPET = """
h1 crd  2 2026  5 18  1
h2       YARL 7090 77  1  4       ILRS
h3 lageos2     9207002 5986 22195    0 1  1
h4  1 2026  5 18  1  3 38 2026  5 18  1 10 12  0 0 0 1 1 0 2 0
c0 0 532.000 std1 ml1 mcp mt1 met
11  3830.1040000000000     0.0534773910846 std1 2  120.0     24      32.2  -0.404   0.151 na      2.0 0 na
11  3890.0000000000000     0.0540000000000 std1 2  120.0     30      28.0  -0.100   0.050 na      2.5 0 na
h8
h2       NONE 1234 77  1  4       ILRS
h4  1 2026  5 18  2  0  0 2026  5 18  2 10  0  0 0 0 1 1 0 2 0
11  7300.0000000000000     0.0500000000000 std1 2  120.0     10      40.0  0.000    0.000 na      1.0 0 na
h8
h2       WETL 8834 77  1  4       ILRS
h4  1 2026  5 18 23 53 20 2026  5 19  0 10  0  0 0 0 1 1 0 2 0
11    200.0000000000000     0.0480000000000 std1 2  120.0     12      35.0  0.010    0.020 na      1.5 0 na
h8
h9
"""


def test_one_way_range_m_matches_two_way_tof():
    tof = 0.0534773910846
    assert one_way_range_m(tof) == tof * SPEED_OF_LIGHT_MPS / 2.0
    # LAGEOS-2 (~5.9 Mm altitude) one-way range is megametre-scale.
    assert 5.0e6 < one_way_range_m(tof) < 2.0e7


def test_parser_filters_unsupported_and_keeps_supported():
    points = parse_crd_v2_normal_points(CRD_SNIPPET)
    # Two YARL + one WETL kept; the CDP 1234 record must be dropped.
    assert [p.station_code for p in points] == ["YARL", "YARL", "WETL"]
    assert all(p.cdp_id in SUPPORTED_STATIONS for p in points)
    assert 1234 not in {p.cdp_id for p in points}


def test_parser_timestamp_and_range_conversion():
    points = parse_crd_v2_normal_points(CRD_SNIPPET)
    first = points[0]
    midnight = dt.datetime(2026, 5, 18, tzinfo=dt.timezone.utc).timestamp()
    assert first.epoch_unix == midnight + 3830.104
    assert first.epoch_iso.startswith("2026-05-18T01:03:50")
    assert first.range_m == one_way_range_m(0.0534773910846)
    assert first.raw_count == 24
    assert first.np_window_s == 120.0
    assert first.bin_rms_ps == 32.2


def test_parser_handles_utc_midnight_rollover():
    points = parse_crd_v2_normal_points(CRD_SNIPPET)
    wetl = next(p for p in points if p.station_code == "WETL")
    # H4 start is 23:53:20 on 2026-05-18; sod=200 rolls to the next UTC day.
    assert wetl.epoch_iso.startswith("2026-05-19T00:03:20")


def test_gmst_in_range_and_time_varying():
    base = dt.datetime(2026, 5, 18, tzinfo=dt.timezone.utc).timestamp()
    g0 = gmst_rad(base)
    g1 = gmst_rad(base + 3600.0)
    assert 0.0 <= g0 < 2.0 * np.pi
    assert 0.0 <= g1 < 2.0 * np.pi
    assert abs(g0 - g1) > 1e-3  # one hour of Earth rotation is resolvable


def test_station_pseudo_inertial_norm_is_geocentric_radius():
    station = SUPPORTED_STATIONS[7090]
    epoch = dt.datetime(2026, 5, 18, 1, 0, tzinfo=dt.timezone.utc).timestamp()
    r_eci = station_pseudo_inertial_m(station, epoch)
    # A z-rotation preserves vector norm: |r_eci| == |r_ecef|.
    assert np.isclose(np.linalg.norm(r_eci), np.linalg.norm(station.ecef_m()))
    assert 6.3e6 < np.linalg.norm(r_eci) < 6.4e6


def test_orbital_period_seconds_from_tle_line2():
    from gnn_state_estimation.slr_learned import orbital_period_seconds

    # Real LAGEOS-2 line 2: mean motion 6.47293620 rev/day -> ~13.35 ks period.
    line2 = "2 22195  52.6609 287.0389 0137740 173.1968   5.5334  6.47293620793337"
    period = orbital_period_seconds(line2)
    assert 13000.0 < period < 13800.0
    # Unparseable input falls back to the documented default.
    assert orbital_period_seconds("garbage") == 13360.0


def test_build_feature_matrix_is_leak_free_and_shaped():
    from gnn_state_estimation.slr_learned import build_feature_matrix

    codes = ["YARL", "MATM", "WETL", "HERL"]
    epochs = np.array([100.0, 200.0, 300.0, 400.0])
    pred = np.array([7.0e6, 7.1e6, 7.2e6, 7.3e6])
    feats = build_feature_matrix(codes, epochs, pred, 100.0, 300.0, 13360.0)
    # 1 time + 4 station one-hot + sin + cos + predicted-range = 8 columns.
    assert feats.shape == (4, 8)
    # Normalized time is monotone in [0, 1] over the span.
    assert feats[0, 0] == 0.0 and feats[-1, 0] == 1.0
    # Exactly one station indicator is active per row.
    assert np.allclose(feats[:, 1:5].sum(axis=1), 1.0)
    # The predicted-range column is the supplied prediction (no truth used).
    assert np.allclose(feats[:, -1], pred)


def test_learned_residual_correction_deterministic_and_bounded():
    from gnn_state_estimation.slr_learned import learned_residual_correction

    rng = np.random.default_rng(7)
    feats = rng.normal(size=(40, 6))
    residuals = rng.normal(scale=30.0, size=40)
    train_idx = np.arange(0, 28)

    corr_a, backend_a = learned_residual_correction(
        feats, residuals, train_idx, seed=0, clip_k=4.0
    )
    corr_b, backend_b = learned_residual_correction(
        feats, residuals, train_idx, seed=0, clip_k=4.0
    )
    # Fully deterministic for a fixed seed (regression guard).
    assert backend_a == backend_b
    assert np.array_equal(corr_a, corr_b)
    # The correction is hard-bounded by clip_k * fit-arc residual RMS.
    train_rms = float(np.sqrt(np.mean(residuals[train_idx] ** 2)))
    assert np.all(np.abs(corr_a) <= 4.0 * train_rms + 1e-6)
    assert corr_a.shape == (40,)


def test_summarize_residuals_schema():
    summary = summarize_residuals(np.array([-3.0, 4.0, 0.0, 5.0]))
    assert set(summary) == {"count", "rms_m", "mae_m", "median_abs_m", "p95_abs_m"}
    assert summary["count"] == 4
    assert summary["rms_m"] > 0.0
    empty = summarize_residuals(np.array([]))
    assert empty["count"] == 0
    assert np.isnan(empty["rms_m"])
