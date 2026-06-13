"""Analytic ITRF<->GCRS reduction (IAU-76 precession, IAU-80 nutation).

This module is *additive*: the approximate GMST-only pseudo-inertial transform
in :mod:`gnn_state_estimation.slr` / :mod:`gnn_state_estimation.sp3` is left
byte-for-byte unchanged so the committed bounded slice carries zero regression
risk.  It provides a substantially higher-fidelity, purely analytic
Earth-orientation transformation (no external Earth-orientation-parameter
download is required): IAU-1976 precession, the IAU-1980 nutation series
truncated to its dominant terms, the equation of the equinoxes (mean -> apparent
sidereal time), and Greenwich apparent sidereal time.

Polar motion and the sub-second UT1-UTC offset are deliberately *not* applied
(they need IERS Earth-orientation parameters); at the LAGEOS radius they are a
bounded tens-of-metres residual, an order of magnitude below the
hundreds-of-metres GMST-only Earth-orientation error this transform removes.
This is documented wherever the slice is reported.
"""

from __future__ import annotations

import numpy as np

from .slr import gmst_rad, julian_date_from_unix

_ARCSEC = np.pi / (180.0 * 3600.0)
_DEG = np.pi / 180.0

# IAU-1980 nutation series, dominant terms.  Columns:
#   l l' F D Omega  (Delaunay multipliers),
#   A0  (0.1 mas, in-phase longitude),  B0 (0.1 mas, in-phase obliquity).
# Truncated to the largest ~20 terms; the neglected tail is < ~0.003" and far
# below the bounded fidelity of this slice.
_NUT_TERMS = (
    (0, 0, 0, 0, 1, -171996.0, 92025.0),
    (0, 0, 2, -2, 2, -13187.0, 5736.0),
    (0, 0, 2, 0, 2, -2274.0, 977.0),
    (0, 0, 0, 0, 2, 2062.0, -895.0),
    (0, 1, 0, 0, 0, 1426.0, 54.0),
    (1, 0, 0, 0, 0, 712.0, -7.0),
    (0, 1, 2, -2, 2, -517.0, 224.0),
    (0, 0, 2, 0, 1, -386.0, 200.0),
    (1, 0, 2, 0, 2, -301.0, 129.0),
    (0, -1, 2, -2, 2, 217.0, -95.0),
    (1, 0, 0, -2, 0, -158.0, -1.0),
    (0, 0, 2, -2, 1, 129.0, -70.0),
    (-1, 0, 2, 0, 2, 123.0, -53.0),
    (1, 0, 0, 0, 1, 63.0, -33.0),
    (0, 0, 0, 2, 0, 63.0, -2.0),
    (-1, 0, 2, 2, 2, -59.0, 26.0),
    (-1, 0, 0, 0, 1, -58.0, 32.0),
    (1, 0, 2, 0, 1, -51.0, 27.0),
    (2, 0, 0, -2, 0, 48.0, 1.0),
    (-2, 0, 2, 0, 1, 46.0, -24.0),
    (0, 0, 2, 2, 2, -38.0, 16.0),
    (2, 0, 2, 0, 2, -31.0, 13.0),
    (1, 0, 2, -2, 2, 29.0, -12.0),
    (0, 0, 2, 0, 0, 29.0, -1.0),
)


def _rot1(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]], dtype=np.float64)


def _rot3(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float64)


def _t_tt(epoch_unix: float) -> float:
    """Julian centuries since J2000.0 (UTC used for TT; offset negligible)."""
    return (julian_date_from_unix(epoch_unix) - 2451545.0) / 36525.0


def precession_matrix(t: float) -> np.ndarray:
    """IAU-1976 precession matrix (mean-of-J2000 -> mean-of-date)."""
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t**3) * _ARCSEC
    z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t**3) * _ARCSEC
    theta = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t**3) * _ARCSEC
    return _rot3(-z) @ _rot1(theta) @ _rot3(-zeta)


def _nutation_angles(t: float) -> tuple[float, float, float, float]:
    """Return (dpsi, deps, mean_obliquity, true_obliquity) in radians."""
    # Fundamental (Delaunay) arguments, degrees -> radians.
    l = (134.96298139 + 477198.867398 * t) * _DEG
    lp = (357.52772333 + 35999.050340 * t) * _DEG
    f = (93.27191028 + 483202.017538 * t) * _DEG
    d = (297.85036306 + 445267.111480 * t) * _DEG
    om = (125.04452222 - 1934.136261 * t) * _DEG
    dpsi = 0.0
    deps = 0.0
    for li, lpi, fi, di, omi, a0, b0 in _NUT_TERMS:
        arg = li * l + lpi * lp + fi * f + di * d + omi * om
        dpsi += a0 * np.sin(arg)
        deps += b0 * np.cos(arg)
    # Series amplitudes are in units of 0.0001".
    dpsi *= 1.0e-4 * _ARCSEC
    deps *= 1.0e-4 * _ARCSEC
    eps0 = (
        84381.448 - 46.8150 * t - 0.00059 * t * t + 0.001813 * t**3
    ) * _ARCSEC
    return dpsi, deps, eps0, eps0 + deps


def nutation_matrix(t: float) -> tuple[np.ndarray, float, float]:
    """IAU-1980 nutation matrix (mean-of-date -> true-of-date)."""
    dpsi, deps, eps0, eps = _nutation_angles(t)
    n = _rot1(-eps) @ _rot3(-dpsi) @ _rot1(eps0)
    return n, dpsi, eps0


def gast_rad(epoch_unix: float, t: float) -> float:
    """Greenwich apparent sidereal time = GMST + equation of equinoxes."""
    dpsi, _, eps0, _ = _nutation_angles(t)
    return gmst_rad(epoch_unix) + dpsi * np.cos(eps0)


def itrf_to_gcrs(r_itrf: np.ndarray, epoch_unix: float) -> np.ndarray:
    """Rotate an ITRF/ECEF position into a GCRS-class true inertial frame.

    Polar motion is neglected (no IERS parameters); precession, nutation and
    apparent sidereal time are applied analytically.
    """
    t = _t_tt(epoch_unix)
    p = precession_matrix(t)
    n, _, _ = nutation_matrix(t)
    theta = gast_rad(epoch_unix, t)
    # r_gcrs = P^T N^T R3(-GAST) r_itrf  (no polar motion).
    return (p.T @ n.T @ _rot3(-theta)) @ np.asarray(
        r_itrf, dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Operationally-corrected ITRF<->GCRS (additive): the full IAU-76/80 equinox
# reduction with real IERS Earth-orientation parameters (polar motion and the
# UT1-UTC offset).  The analytic transform above is left byte-for-byte
# unchanged so the committed bounded / higher-fidelity slices carry zero
# regression risk.  This closes the two Earth-orientation items the loop-30
# review listed as missing precise-SLR-reduction corrections; the
# correction-sensitivity audit quantifies how large each term actually is.
# ---------------------------------------------------------------------------


def _rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float64)


def polar_motion_matrix(xp_rad: float, yp_rad: float) -> np.ndarray:
    """ITRF -> PEF polar-motion rotation (IAU-76/80 equinox convention).

    Vallado (2013) Eq. 3-78 uses ``[W] = ROT1(y_p) ROT2(x_p)`` for the
    PEF -> ITRF map, so ITRF -> PEF is its inverse ``ROT2(-x_p) ROT1(-y_p)``.
    At the LAGEOS radius the IERS pole offsets (~0.2--0.4 arcsec) move the
    station by only tens of metres; the term is included for completeness and
    its magnitude is reported in the correction-sensitivity audit.
    """
    return _rot2(-xp_rad) @ _rot1(-yp_rad)


def gast_rad_ut1(epoch_unix: float, t: float, dut1_s: float) -> float:
    """Greenwich apparent sidereal time using UT1 = UTC + (UT1-UTC).

    The sidereal-angle dependence on UT1 dominates the UT1-UTC effect; the
    sub-second change in the slowly varying GMST polynomial term is far below
    this slice's fidelity and is intentionally not separately modelled.
    """
    dpsi, _, eps0, _ = _nutation_angles(t)
    return gmst_rad(epoch_unix + float(dut1_s)) + dpsi * np.cos(eps0)


def itrf_to_gcrs_eop(
    r_itrf: np.ndarray,
    epoch_unix: float,
    xp_rad: float,
    yp_rad: float,
    dut1_s: float,
) -> np.ndarray:
    """Full IAU-76/80 ITRF -> GCRS with IERS polar motion and UT1-UTC.

    ``r_gcrs = P^T N^T R3(-GAST_UT1) W_{ITRF->PEF} r_itrf`` where ``W`` is the
    polar-motion rotation and the apparent sidereal time uses UT1.
    """
    t = _t_tt(epoch_unix)
    p = precession_matrix(t)
    n, _, _ = nutation_matrix(t)
    theta = gast_rad_ut1(epoch_unix, t, dut1_s)
    w = polar_motion_matrix(xp_rad, yp_rad)
    return (p.T @ n.T @ _rot3(-theta) @ w) @ np.asarray(
        r_itrf, dtype=np.float64
    )
