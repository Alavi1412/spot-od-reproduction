"""SP3-c precise-orbit parsing and a self-contained range-only filter bank.

This module supports a *bounded precise-reference* real-measurement orbit
determination (OD) slice.  It parses public ILRS analysis-centre SP3-c precise
orbit products (the independent reference) and provides:

* a robust SP3-c parser (UTC epochs, ITRF/ECEF positions in km, velocities in
  dm/s) keyed by the SP3 satellite identifier (``L51`` LAGEOS-1, ``L52``
  LAGEOS-2);
* high-order Lagrange interpolation of the reference ephemeris to arbitrary
  epochs;
* an approximate GMST rotation into the same pseudo-inertial frame used for the
  laser-ranging station geometry, so the real range observations and the
  precise reference state live in one internally consistent frame (the GMST
  approximation is common-mode between observation geometry and reference
  scoring and therefore largely cancels);
* a deliberately self-contained range-only EKF, a fixed-noise UKF, and an
  innovation-adaptive UKF (the simulator filters are intentionally left
  untouched), used to score held-out state error against the *external* precise
  SP3 reference and to drive the DBAR adaptive-filter risk indicator from an
  externally defined counterproductivity outcome.

It deliberately does not implement precise SLR reduction (no relativistic,
tropospheric, centre-of-mass, polar-motion, or solid-Earth-tide corrections)
and propagates a compact two-body+J2 model, so absolute magnitudes are a
bounded-fidelity model-mismatch stress, not centimetre operational OD.  This is
documented everywhere it is used.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH
from .coordinates import rot_z
from .dynamics import rk4_step
from .slr import SlrStation, gmst_rad


@dataclass(frozen=True)
class Sp3Ephemeris:
    """Parsed SP3-c reference ephemeris for one satellite.

    ``epochs_unix`` is strictly increasing UTC POSIX seconds; ``positions_m``
    and ``velocities_mps`` are ITRF/ECEF metres and metres-per-second.
    """

    sat_id: str
    epochs_unix: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    coordinate_frame: str
    time_system: str
    analysis_center: str

    @property
    def start_unix(self) -> float:
        return float(self.epochs_unix[0])

    @property
    def end_unix(self) -> float:
        return float(self.epochs_unix[-1])

    def covers(self, epoch_unix: float, *, margin_s: float = 0.0) -> bool:
        return (
            self.start_unix - margin_s
            <= float(epoch_unix)
            <= self.end_unix + margin_s
        )


def parse_sp3(text: str, sat_id: str) -> Sp3Ephemeris:
    """Parse an SP3-c product, returning the ephemeris for ``sat_id``.

    Only the position/velocity records for the requested satellite token (e.g.
    ``L51``) are retained.  Positions are converted km -> m and velocities
    dm/s -> m/s per the SP3 standard.
    """
    lines = text.splitlines()
    if not lines or not lines[0].startswith("#"):
        raise ValueError("not an SP3 file (missing '#' header)")

    time_system = "UTC"
    coordinate_frame = "ECEF"
    analysis_center = ""
    header0 = lines[0]
    # Columns 47-51 carry the data-used / orbit-type / agency tail; agency is
    # the trailing token of the first header line.
    tail = header0.split()
    if tail:
        analysis_center = tail[-1]
    if "ECF" in header0 or "ECEF" in header0:
        coordinate_frame = "ECEF/ITRF"

    epochs: list[float] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    have_velocity = False
    cur_epoch: float | None = None
    pos_target = "P" + sat_id
    vel_target = "V" + sat_id

    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith("%c"):
            toks = line.split()
            # %c <file-type> cc <time-system> ...
            if len(toks) >= 4 and toks[3] in (
                "GPS",
                "UTC",
                "TAI",
                "UT1",
                "GLO",
                "GAL",
                "QZS",
                "BDT",
            ):
                time_system = toks[3]
            continue
        if line.startswith("*"):
            parts = line.split()
            # * YYYY MM DD HH MM SS.SSSSSSSS
            year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
            hour, minute = int(parts[4]), int(parts[5])
            sec = float(parts[6])
            whole = int(sec)
            micro = int(round((sec - whole) * 1_000_000))
            dt = _dt.datetime(
                year, month, day, hour, minute, whole,
                micro, tzinfo=_dt.timezone.utc,
            )
            cur_epoch = dt.timestamp()
            continue
        if line.startswith(pos_target) and cur_epoch is not None:
            parts = line.split()
            x_km, y_km, z_km = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
            epochs.append(cur_epoch)
            positions.append([x_km * 1e3, y_km * 1e3, z_km * 1e3])
            velocities.append([np.nan, np.nan, np.nan])
            continue
        if line.startswith(vel_target) and cur_epoch is not None and positions:
            parts = line.split()
            # SP3 velocity records are in dm/s.
            vx, vy, vz = (
                float(parts[1]) * 1e-1,
                float(parts[2]) * 1e-1,
                float(parts[3]) * 1e-1,
            )
            velocities[-1] = [vx, vy, vz]
            have_velocity = True
            continue

    if len(epochs) < 4:
        raise ValueError(
            f"SP3 has too few epochs for satellite {sat_id!r} "
            f"({len(epochs)} found)"
        )

    ep = np.asarray(epochs, dtype=np.float64)
    order = np.argsort(ep, kind="stable")
    ep = ep[order]
    pos = np.asarray(positions, dtype=np.float64)[order]
    vel = np.asarray(velocities, dtype=np.float64)[order]
    if not have_velocity:
        vel = np.full_like(pos, np.nan)

    return Sp3Ephemeris(
        sat_id=sat_id,
        epochs_unix=ep,
        positions_m=pos,
        velocities_mps=vel,
        coordinate_frame=coordinate_frame,
        time_system=time_system,
        analysis_center=analysis_center,
    )


def _lagrange_vector(
    nodes_t: np.ndarray, nodes_y: np.ndarray, t: float
) -> np.ndarray:
    """Barycentric-free Lagrange interpolation of vector samples at ``t``."""
    n = nodes_t.size
    # Work relative to the first node to keep the products well-scaled.
    tt = nodes_t - nodes_t[0]
    tq = t - nodes_t[0]
    out = np.zeros(nodes_y.shape[1], dtype=np.float64)
    for j in range(n):
        num = 1.0
        den = 1.0
        for m in range(n):
            if m == j:
                continue
            num *= tq - tt[m]
            den *= tt[j] - tt[m]
        out += nodes_y[j] * (num / den)
    return out


class Sp3Interpolator:
    """High-order Lagrange interpolation of an SP3 reference ephemeris.

    LAGEOS orbits are extremely smooth and the SP3 product is densely sampled
    (2-minute nodes here), so a 9th--10th order Lagrange interpolation over a
    short local window reproduces the precise reference to well below the
    bounded-fidelity model-error budget of this slice.
    """

    def __init__(self, eph: Sp3Ephemeris, order: int = 9) -> None:
        self.eph = eph
        self.order = int(order)
        self._t = eph.epochs_unix

    def position_ecef_m(self, epoch_unix: float) -> np.ndarray:
        t = float(epoch_unix)
        n = self._t.size
        k = int(np.searchsorted(self._t, t))
        half = (self.order + 1) // 2
        lo = min(max(k - half, 0), max(n - (self.order + 1), 0))
        hi = min(lo + self.order + 1, n)
        lo = max(hi - (self.order + 1), 0)
        return _lagrange_vector(
            self._t[lo:hi], self.eph.positions_m[lo:hi], t
        )

    def velocity_ecef_mps(
        self, epoch_unix: float, h_s: float = 1.0
    ) -> np.ndarray:
        """Central-difference ECEF velocity of the interpolated reference."""
        p_plus = self.position_ecef_m(epoch_unix + h_s)
        p_minus = self.position_ecef_m(epoch_unix - h_s)
        return (p_plus - p_minus) / (2.0 * h_s)

    # --- Pseudo-inertial mapping (same GMST rotation as station geometry) ----
    def position_pseudo_inertial_m(self, epoch_unix: float) -> np.ndarray:
        theta = gmst_rad(epoch_unix)
        return rot_z(theta).T @ self.position_ecef_m(epoch_unix)

    def velocity_pseudo_inertial_mps(
        self, epoch_unix: float, h_s: float = 1.0
    ) -> np.ndarray:
        """Velocity of the *pseudo-inertial* reference track.

        Differencing the rotated positions keeps the reference state, the
        propagation, and the held-out scoring in one consistent frame, so the
        common-mode GMST approximation cancels rather than entering as an
        Earth-rotation bias.
        """
        p_plus = self.position_pseudo_inertial_m(epoch_unix + h_s)
        p_minus = self.position_pseudo_inertial_m(epoch_unix - h_s)
        return (p_plus - p_minus) / (2.0 * h_s)

    def state_pseudo_inertial_m(self, epoch_unix: float) -> np.ndarray:
        return np.hstack(
            [
                self.position_pseudo_inertial_m(epoch_unix),
                self.velocity_pseudo_inertial_mps(epoch_unix),
            ]
        ).astype(np.float64)


# --- Compact-model propagation in the pseudo-inertial frame -----------------
# LAGEOS-1/-2 are passive geodetic spheres at ~5.9 Mm altitude where drag is
# negligible; a near-zero ballistic coefficient with the compact two-body+J2
# model is used (identical treatment to the existing real-SLR audit).
_LAGEOS_BALLISTIC_COEFF = 1.0e-9


def propagate_compact(
    state: np.ndarray, dt_s: float, max_step_s: float = 30.0
) -> np.ndarray:
    """Propagate a 6-vector by ``dt_s`` with the compact two-body+J2 model."""
    state = np.asarray(state, dtype=np.float64).copy()
    if dt_s == 0.0:
        return state
    sign = 1.0 if dt_s > 0 else -1.0
    remaining = abs(dt_s)
    while remaining > 1e-9:
        step = sign * min(max_step_s, remaining)
        state = rk4_step(
            state,
            dt=step,
            ballistic_coeff_m2_per_kg=_LAGEOS_BALLISTIC_COEFF,
        )
        remaining -= abs(step)
    return state


def _state_transition_jac(
    state: np.ndarray, dt_s: float, max_step_s: float = 30.0
) -> np.ndarray:
    """Numerical state-transition Jacobian via central finite differences."""
    n = 6
    phi = np.zeros((n, n), dtype=np.float64)
    perturb = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3], dtype=np.float64)
    for j in range(n):
        dp = np.zeros(n)
        dp[j] = perturb[j]
        sp = propagate_compact(state + dp, dt_s, max_step_s)
        sm = propagate_compact(state - dp, dt_s, max_step_s)
        phi[:, j] = (sp - sm) / (2.0 * perturb[j])
    return phi


def _process_cov(dt_s: float, accel_psd: float) -> np.ndarray:
    """Discrete white-noise-acceleration process covariance."""
    dt = abs(float(dt_s))
    q = accel_psd
    q11 = (dt**4) / 4.0 * q
    q12 = (dt**3) / 2.0 * q
    q22 = (dt**2) * q
    blk = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        blk[i, i] = q11
        blk[i, i + 3] = q12
        blk[i + 3, i] = q12
        blk[i + 3, i + 3] = q22
    return blk


@dataclass
class RangeObs:
    """One range observation: epoch, station pseudo-inertial position, range."""

    epoch_unix: float
    station_pi_m: np.ndarray
    range_m: float


def _range_and_jac(
    state: np.ndarray, station_pi: np.ndarray
) -> tuple[float, np.ndarray]:
    los = state[:3] - station_pi
    rng = float(np.linalg.norm(los))
    h = np.zeros(6, dtype=np.float64)
    if rng > 1.0:
        h[:3] = los / rng
    return rng, h


def run_range_ekf(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    max_step_s: float = 30.0,
) -> dict:
    """Range-only EKF.  Returns the post-fit state and per-update records."""
    x = np.asarray(x0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    r_nom = float(range_std_m) ** 2
    records: list[dict] = []
    t_prev = obs[0].epoch_unix
    for o in obs:
        dt = o.epoch_unix - t_prev
        if dt != 0.0:
            phi = _state_transition_jac(x, dt, max_step_s)
            x = propagate_compact(x, dt, max_step_s)
            p = phi @ p @ phi.T + _process_cov(dt, accel_psd)
            t_prev = o.epoch_unix
        rng, hvec = _range_and_jac(x, o.station_pi_m)
        innov = o.range_m - rng
        s = float(hvec @ p @ hvec) + r_nom
        k = (p @ hvec) / s
        x = x + k * innov
        p = p - np.outer(k, hvec) @ p
        p = 0.5 * (p + p.T)
        records.append(
            {"epoch_unix": o.epoch_unix, "innovation_m": innov,
             "s": s, "nis_r": innov * innov / r_nom}
        )
    return {"state": x, "cov": p, "records": records}


# --- Unscented machinery (self-contained, range-only) -----------------------
def _sigma_points(x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    n = x.size
    p_sym = 0.5 * (p + p.T) + 1e-6 * np.eye(n)
    try:
        s = np.linalg.cholesky((n + lam) * p_sym)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh((n + lam) * p_sym)
        w = np.clip(w, 1e-9, None)
        s = v @ np.diag(np.sqrt(w))
    pts = np.zeros((2 * n + 1, n), dtype=np.float64)
    pts[0] = x
    for i in range(n):
        pts[1 + i] = x + s[:, i]
        pts[1 + n + i] = x - s[:, i]
    return pts


def _ukf_weights(n: int, alpha: float, beta: float, kappa: float):
    lam = alpha * alpha * (n + kappa) - n
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1.0 - alpha * alpha + beta)
    return wm, wc, lam


def _run_range_ukf(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    adaptive: bool,
    max_step_s: float = 30.0,
    r_window: int = 8,
    r_scale_cap: float = 50.0,
) -> dict:
    """Range-only UKF.

    With ``adaptive=False`` this is a fixed-measurement-noise UKF.  With
    ``adaptive=True`` the measurement-noise scale is inflated by an
    innovation-consistency (residual-matching) factor over a sliding window --
    the classical adaptive mechanism whose risk under dynamics bias DBAR flags.
    ``nis_r`` is always the *R-only* normalized innovation (innovation^2 over
    the nominal R), so the DBAR ratio is comparable across the filter pair.
    """
    n = 6
    x = np.asarray(x0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    alpha, beta, kappa = 1e-3, 2.0, 0.0
    wm, wc, lam = _ukf_weights(n, alpha, beta, kappa)
    r_nom = float(range_std_m) ** 2
    recent_sq: list[float] = []
    records: list[dict] = []
    t_prev = obs[0].epoch_unix
    for o in obs:
        dt = o.epoch_unix - t_prev
        if dt != 0.0:
            pts = _sigma_points(x, p, lam)
            prop = np.vstack(
                [propagate_compact(pt, dt, max_step_s) for pt in pts]
            )
            x = wm @ prop
            dx = prop - x
            p = (wc[:, None, None] * dx[:, :, None] * dx[:, None, :]).sum(0)
            p = p + _process_cov(dt, accel_psd)
            p = 0.5 * (p + p.T)
            t_prev = o.epoch_unix

        pts = _sigma_points(x, p, lam)
        z_pred = np.array(
            [np.linalg.norm(pt[:3] - o.station_pi_m) for pt in pts]
        )
        z_mean = float(wm @ z_pred)
        dz = z_pred - z_mean
        innov = o.range_m - z_mean
        nis_r = innov * innov / r_nom

        r_eff_scale = 1.0
        if adaptive:
            recent_sq.append(innov * innov)
            if len(recent_sq) > r_window:
                recent_sq.pop(0)
            pzz_nom = float((wc * dz) @ dz) + r_nom
            mean_sq = float(np.mean(recent_sq))
            # Residual-matching inflation: only ever inflate (scale >= 1).
            r_eff_scale = float(
                np.clip(mean_sq / max(pzz_nom, 1e-9), 1.0, r_scale_cap)
            )
        r_used = r_nom * r_eff_scale
        pzz = float((wc * dz) @ dz) + r_used
        pxz = (wc[:, None] * (pts - x) * dz[:, None]).sum(0)
        k = pxz / pzz
        x = x + k * innov
        p = p - np.outer(k, k) * pzz
        p = 0.5 * (p + p.T)
        records.append(
            {
                "epoch_unix": o.epoch_unix,
                "innovation_m": innov,
                "nis_r": nis_r,
                "r_eff_scale": r_eff_scale,
            }
        )
    return {"state": x, "cov": p, "records": records}


def run_range_ukf_fixed(
    obs, x0, p0, range_std_m, accel_psd, max_step_s: float = 30.0
) -> dict:
    return _run_range_ukf(
        obs, x0, p0, range_std_m, accel_psd, False, max_step_s
    )


def run_range_aukf(
    obs, x0, p0, range_std_m, accel_psd, max_step_s: float = 30.0
) -> dict:
    return _run_range_ukf(
        obs, x0, p0, range_std_m, accel_psd, True, max_step_s
    )


def median_nis_r(records: list[dict]) -> float:
    vals = [r["nis_r"] for r in records if np.isfinite(r.get("nis_r", np.nan))]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def mean_r_eff_scale(records: list[dict]) -> float:
    vals = [
        r["r_eff_scale"]
        for r in records
        if np.isfinite(r.get("r_eff_scale", np.nan))
    ]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def held_out_position_rmse(
    final_state: np.ndarray,
    fit_last_epoch: float,
    held_epochs: np.ndarray,
    interp: Sp3Interpolator,
    max_step_s: float = 30.0,
) -> dict:
    """Predict-only propagation of ``final_state`` scored vs precise SP3.

    The post-fit state is propagated (no further measurement updates) through
    the held-out epochs and compared to the *external* SP3 reference state in
    the shared pseudo-inertial frame -- a precise-reference held-out OD score.
    """
    errs: list[float] = []
    state = np.asarray(final_state, dtype=np.float64).copy()
    t_prev = float(fit_last_epoch)
    for te in np.asarray(held_epochs, dtype=np.float64):
        state = propagate_compact(state, te - t_prev, max_step_s)
        t_prev = te
        ref = interp.position_pseudo_inertial_m(te)
        errs.append(float(np.linalg.norm(state[:3] - ref)))
    arr = np.asarray(errs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "rms_m": float("nan"), "p95_abs_m": float("nan")}
    return {
        "count": int(arr.size),
        "rms_m": float(np.sqrt(np.mean(arr**2))),
        "mean_m": float(np.mean(arr)),
        "p95_abs_m": float(np.percentile(arr, 95.0)),
    }


def circular_orbit_speed_mps(radius_m: float) -> float:
    return float(np.sqrt(MU_EARTH / max(radius_m, 1.0)))


# ---------------------------------------------------------------------------
# Higher-fidelity precise-reference propagation (additive; the compact
# two-body+J2 path above is intentionally left byte-for-byte unchanged so the
# committed bounded slice carries zero regression risk).
#
# This adds the two perturbations that dominate the LAGEOS held-out
# propagation budget at this altitude and are *not* learnable as a mean
# empirical correction across disjoint arcs:
#
#   * luni-solar third-body point-mass gravity, with a real-epoch
#     low-precision analytic Sun and Moon ephemeris (Montenbruck & Gill,
#     "Satellite Orbits", Sec. 3.3.2); and
#   * the J3 and J4 zonal geopotential terms.
#
# The Earth-rotation treatment is unchanged (the common-mode GMST-only
# transform), so this is a *physically principled fidelity increase* of the
# real-data slice rather than a centimetre operational SLR reduction; it is
# documented as such everywhere it is used.
# ---------------------------------------------------------------------------

from .constants import AU, J2, J3, J4, J5, J6, MU_MOON, MU_SUN, R_EARTH  # noqa: E402
from .dynamics import third_body_acceleration  # noqa: E402
from .slr import julian_date_from_unix  # noqa: E402

_ARCSEC = np.pi / (180.0 * 3600.0)
_DEG = np.pi / 180.0


def _julian_centuries_tt(epoch_unix: float) -> float:
    """Julian centuries since J2000.0.

    UTC is used in place of TT (offset <~70 s); for a third-body direction at
    LAGEOS this is far below the bounded-fidelity budget of the slice.
    """
    jd = julian_date_from_unix(epoch_unix)
    return (jd - 2451545.0) / 36525.0


def sun_position_eci_m(epoch_unix: float) -> np.ndarray:
    """Geocentric equatorial Sun position (m), low-precision analytic series.

    Montenbruck & Gill, "Satellite Orbits" Sec. 3.3.2: ~0.01 deg direction
    accuracy, more than sufficient for a third-body perturbation on LAGEOS.
    """
    t = _julian_centuries_tt(epoch_unix)
    # Mean anomaly and ecliptic longitude (degrees).
    m = (357.5256 + 35999.049 * t) * _DEG
    lam = (
        282.9400
        + np.degrees(m)
        + (6892.0 / 3600.0) * np.sin(m)
        + (72.0 / 3600.0) * np.sin(2.0 * m)
    ) * _DEG
    # Geocentric distance (m).
    r = (
        149.619e9
        - 2.499e9 * np.cos(m)
        - 0.021e9 * np.cos(2.0 * m)
    )
    eps = (23.43929111 - 0.0130042 * t) * _DEG
    cl, sl = np.cos(lam), np.sin(lam)
    ce, se = np.cos(eps), np.sin(eps)
    return np.array(
        [r * cl, r * ce * sl, r * se * sl], dtype=np.float64
    )


def moon_position_eci_m(epoch_unix: float) -> np.ndarray:
    """Geocentric equatorial Moon position (m), low-precision analytic series.

    Montenbruck & Gill, "Satellite Orbits" Sec. 3.3.2 truncated series:
    direction accuracy ~ a few arc-minutes and distance ~ 0.1 %, which keeps
    the lunar third-body acceleration error well below the slice budget.
    """
    t = _julian_centuries_tt(epoch_unix)
    l0 = (218.31617 + 481267.88088 * t - 1.3972 * t) * _DEG
    lp = (134.96292 + 477198.86753 * t) * _DEG       # Moon mean anomaly
    ls = (357.52543 + 35999.04944 * t) * _DEG        # Sun mean anomaly
    f = (93.27283 + 483202.01873 * t) * _DEG         # arg. of latitude
    d = (297.85027 + 445267.11135 * t) * _DEG        # mean elongation

    lam = l0 + _ARCSEC * (
        22640.0 * np.sin(lp)
        + 769.0 * np.sin(2.0 * lp)
        - 4586.0 * np.sin(lp - 2.0 * d)
        + 2370.0 * np.sin(2.0 * d)
        - 668.0 * np.sin(ls)
        - 412.0 * np.sin(2.0 * f)
        - 212.0 * np.sin(2.0 * lp - 2.0 * d)
        - 206.0 * np.sin(lp + ls - 2.0 * d)
        + 192.0 * np.sin(lp + 2.0 * d)
        - 165.0 * np.sin(ls - 2.0 * d)
        + 148.0 * np.sin(lp - ls)
        - 125.0 * np.sin(d)
        - 110.0 * np.sin(lp + ls)
        - 55.0 * np.sin(2.0 * f - 2.0 * d)
    )
    beta = _ARCSEC * (
        18520.0 * np.sin(
            f
            + lam
            - l0
            + _ARCSEC * (412.0 * np.sin(2.0 * f) + 541.0 * np.sin(ls))
        )
        - 526.0 * np.sin(f - 2.0 * d)
        + 44.0 * np.sin(lp + f - 2.0 * d)
        - 31.0 * np.sin(-lp + f - 2.0 * d)
        - 25.0 * np.sin(-2.0 * lp + f)
        - 23.0 * np.sin(ls + f - 2.0 * d)
        + 21.0 * np.sin(-lp + f)
        + 11.0 * np.sin(-ls + f - 2.0 * d)
    )
    r = (
        385000.0e3
        - 20905.0e3 * np.cos(lp)
        - 3699.0e3 * np.cos(2.0 * d - lp)
        - 2956.0e3 * np.cos(2.0 * d)
        - 570.0e3 * np.cos(2.0 * lp)
        + 246.0e3 * np.cos(2.0 * lp - 2.0 * d)
        - 205.0e3 * np.cos(ls - 2.0 * d)
        - 171.0e3 * np.cos(lp + 2.0 * d)
        - 152.0e3 * np.cos(lp + ls - 2.0 * d)
    )
    eps = (23.43929111 - 0.0130042 * t) * _DEG
    cl, sl = np.cos(lam), np.sin(lam)
    cb, sb = np.cos(beta), np.sin(beta)
    # Ecliptic -> equatorial rotation about the x-axis by the obliquity.
    x = r * cb * cl
    y_e = r * cb * sl
    z_e = r * sb
    ce, se = np.cos(eps), np.sin(eps)
    return np.array(
        [x, ce * y_e - se * z_e, se * y_e + ce * z_e], dtype=np.float64
    )


def _legendre(n: int, u: float) -> tuple[float, float]:
    """Legendre polynomial ``P_n(u)`` and its derivative ``P_n'(u)``."""
    if n == 2:
        return 0.5 * (3.0 * u * u - 1.0), 3.0 * u
    if n == 3:
        return 0.5 * (5.0 * u**3 - 3.0 * u), 0.5 * (15.0 * u * u - 3.0)
    if n == 4:
        return (
            (35.0 * u**4 - 30.0 * u * u + 3.0) / 8.0,
            0.5 * (35.0 * u**3 - 15.0 * u),
        )
    if n == 5:
        return (
            (63.0 * u**5 - 70.0 * u**3 + 15.0 * u) / 8.0,
            (315.0 * u**4 - 210.0 * u**2 + 15.0) / 8.0,
        )
    if n == 6:
        return (
            (231.0 * u**6 - 315.0 * u**4 + 105.0 * u * u - 5.0) / 16.0,
            (1386.0 * u**5 - 1260.0 * u**3 + 210.0 * u) / 16.0,
        )
    raise ValueError(f"unsupported zonal degree {n}")


_ZONAL_J = {2: J2, 3: J3, 4: J4}
_ZONAL_J_EXTENDED = {2: J2, 3: J3, 4: J4, 5: J5, 6: J6}


def zonal_potential(r_vec: np.ndarray) -> float:
    """Perturbing zonal potential ``R`` (J2..J4); two-body part excluded.

    ``a_zonal = grad(R)`` exactly; a unit test pins the analytic acceleration
    below against a finite difference of this closed form, so the geopotential
    mathematics is self-verified rather than asserted.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return 0.0
    u = r_vec[2] / r
    total = 0.0
    for n, jn in _ZONAL_J.items():
        pn, _ = _legendre(n, u)
        total += -(MU_EARTH / r) * jn * (R_EARTH / r) ** n * pn
    return float(total)


def zonal_acceleration(r_vec: np.ndarray) -> np.ndarray:
    """Analytic gradient of :func:`zonal_potential` (J2..J4)."""
    r_vec = np.asarray(r_vec, dtype=np.float64)
    x, y, z = r_vec
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    u = z / r
    ax = ay = az = 0.0
    for n, jn in _ZONAL_J.items():
        cn = MU_EARTH * jn * (R_EARTH**n)
        pn, dpn = _legendre(n, u)
        radial = cn / r ** (n + 3)
        common = (n + 1) * pn + u * dpn
        ax += radial * x * common
        ay += radial * y * common
        az += (cn / r ** (n + 2)) * ((n + 1) * u * pn - (1.0 - u * u) * dpn)
    return np.array([ax, ay, az], dtype=np.float64)


def accel_hifi(
    r_vec: np.ndarray, epoch_unix: float
) -> np.ndarray:
    """Higher-fidelity acceleration: two-body + J2..J4 + luni-solar third body.

    Sun/Moon positions are real-epoch low-precision analytic ephemerides in
    the geocentric equatorial frame, which the GMST-only pseudo-inertial frame
    approximates to precession/nutation level -- far below the third-body
    direction sensitivity, so this is internally consistent.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    a = -MU_EARTH * r_vec / (r**3)
    a = a + zonal_acceleration(r_vec)
    r_sun = sun_position_eci_m(epoch_unix)
    r_moon = moon_position_eci_m(epoch_unix)
    a = a + third_body_acceleration(r_vec, r_sun, MU_SUN)
    a = a + third_body_acceleration(r_vec, r_moon, MU_MOON)
    return a.astype(np.float64)


def zonal_acceleration_extended(r_vec: np.ndarray) -> np.ndarray:
    """Analytic gradient of the extended zonal potential (J2..J6).

    Identical structural formula to :func:`zonal_acceleration` but iterates
    the extended ``{2,3,4,5,6}`` set of zonal coefficients so the extended
    higher-fidelity propagator can be exercised without disturbing the
    existing J2..J4 path.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    x, y, z = r_vec
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    u = z / r
    ax = ay = az = 0.0
    for n, jn in _ZONAL_J_EXTENDED.items():
        cn = MU_EARTH * jn * (R_EARTH ** n)
        pn, dpn = _legendre(n, u)
        radial = cn / r ** (n + 3)
        common = (n + 1) * pn + u * dpn
        ax += radial * x * common
        ay += radial * y * common
        az += (cn / r ** (n + 2)) * ((n + 1) * u * pn - (1.0 - u * u) * dpn)
    return np.array([ax, ay, az], dtype=np.float64)


def accel_hifi_extended(
    r_vec: np.ndarray, epoch_unix: float
) -> np.ndarray:
    """Extended higher-fidelity acceleration: two-body + J2..J6 + luni-solar third body.

    This is an additive extension of :func:`accel_hifi` that includes the J5
    and J6 zonal geopotential terms (well-established EGM-class nominal
    coefficients, see :mod:`gnn_state_estimation.constants`). All other
    structure is identical to :func:`accel_hifi`; the existing function is
    left byte-identical so the previously committed slice and its tests
    remain unchanged.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    a = -MU_EARTH * r_vec / (r**3)
    a = a + zonal_acceleration_extended(r_vec)
    r_sun = sun_position_eci_m(epoch_unix)
    r_moon = moon_position_eci_m(epoch_unix)
    a = a + third_body_acceleration(r_vec, r_sun, MU_SUN)
    a = a + third_body_acceleration(r_vec, r_moon, MU_MOON)
    return a.astype(np.float64)


# ---------------------------------------------------------------------------
# Long-arc higher-fidelity additive extension (Loop 47): adds the dominant
# non-zonal spherical-harmonic terms C(2,2) and S(2,2), rotated from the
# Earth-fixed frame to the same GMST-only pseudo-inertial frame used
# throughout this study. All other structure is identical to
# :func:`accel_hifi_extended`; the existing functions and their tests remain
# byte-identical.
# ---------------------------------------------------------------------------

# EGM-class normalised nominal sectoral (n=m=2) coefficients. Equivalent
# unnormalised values are recovered by the standard Kaula factor
# N(2,2) = sqrt(5 * (n-m)! / ((2-delta_{0,m}) * (n+m)!)) = sqrt(5 / 24).
EGM_C22_NORMALIZED = 2.4393836e-6  # unitless
EGM_S22_NORMALIZED = -1.4002737e-6  # unitless


def tesseral_22_potential_ecef(r_ecef: np.ndarray) -> float:
    """Perturbing sectoral spherical-harmonic potential R_{22} in the
    Earth-fixed frame, using the EGM-class normalised C(2,2)/S(2,2).

    For n=m=2, the unnormalised sectoral Legendre function is
    P_{22}(sin phi) = 3 * cos^2(phi). The normalised sectoral coefficients
    relate to unnormalised by the Kaula factor sqrt(5/24). The resulting
    expression is

        R_{22}(r,phi,lambda) = -mu/r * (R/r)^2 * 3 * cos^2(phi)
                                  * (Cbar22 * cos(2 lambda) + Sbar22 * sin(2 lambda))
                                  * sqrt(5/24).

    """
    r_vec = np.asarray(r_ecef, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return 0.0
    x, y, z = r_vec
    sin_phi2 = (z * z) / (r * r)
    cos_phi2 = 1.0 - sin_phi2
    rho2 = x * x + y * y
    if rho2 < 1.0:
        return 0.0
    cos_2l = (x * x - y * y) / rho2
    sin_2l = 2.0 * x * y / rho2
    norm22 = float(np.sqrt(5.0 / 24.0))
    return float(
        -(MU_EARTH / r)
        * (R_EARTH / r) ** 2
        * 3.0
        * cos_phi2
        * (EGM_C22_NORMALIZED * cos_2l + EGM_S22_NORMALIZED * sin_2l)
        * norm22
    )


def tesseral_22_acceleration_ecef(r_ecef: np.ndarray) -> np.ndarray:
    """Analytic gradient of :func:`tesseral_22_potential_ecef` in the
    Earth-fixed frame.

    Computed analytically and pinned in the test suite against a central
    finite-difference of the potential.
    """
    r_vec = np.asarray(r_ecef, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    x, y, z = r_vec
    rho2 = x * x + y * y
    if rho2 < 1.0:
        return np.zeros(3, dtype=np.float64)
    norm22 = float(np.sqrt(5.0 / 24.0))
    c22 = EGM_C22_NORMALIZED * norm22
    s22 = EGM_S22_NORMALIZED * norm22
    # R_{22} = K * (R_e^2 / r^5) * (x^2 - y^2) * c22 + K * (R_e^2 / r^5) * 2xy * s22
    # with K = -3 * mu (because cos^2(phi) = rho^2/r^2 and (rho^2 / r^2) * (cos2l + i sin2l)
    # = ((x^2 - y^2) + 2i*xy)/r^2; combined with the -mu/r * (R/r)^2 = -mu * R_e^2 / r^3
    # prefactor gives -3 mu R_e^2 / r^5 * [(x^2 - y^2) c22 + 2xy s22]).
    k_c = -3.0 * MU_EARTH * (R_EARTH ** 2) * c22
    k_s = -3.0 * MU_EARTH * (R_EARTH ** 2) * s22
    inv_r5 = 1.0 / (r ** 5)
    inv_r7 = 1.0 / (r ** 7)
    # Gradient of (x^2 - y^2) / r^5
    fac_c_x = (2.0 * x * inv_r5) - 5.0 * x * (x * x - y * y) * inv_r7
    fac_c_y = (-2.0 * y * inv_r5) - 5.0 * y * (x * x - y * y) * inv_r7
    fac_c_z = -5.0 * z * (x * x - y * y) * inv_r7
    # Gradient of 2xy / r^5
    fac_s_x = (2.0 * y * inv_r5) - 5.0 * x * 2.0 * x * y * inv_r7
    fac_s_y = (2.0 * x * inv_r5) - 5.0 * y * 2.0 * x * y * inv_r7
    fac_s_z = -5.0 * z * 2.0 * x * y * inv_r7
    ax = k_c * fac_c_x + k_s * fac_s_x
    ay = k_c * fac_c_y + k_s * fac_s_y
    az = k_c * fac_c_z + k_s * fac_s_z
    return np.array([ax, ay, az], dtype=np.float64)


def tesseral_22_acceleration_inertial(
    r_inertial: np.ndarray, epoch_unix: float
) -> np.ndarray:
    """C(2,2)/S(2,2) sectoral acceleration in the GMST-only pseudo-inertial
    frame used throughout this study. Rotates the position into the Earth-
    fixed frame by the GMST angle, evaluates the gradient there, and rotates
    back so the acceleration is in the same frame as :func:`accel_hifi`.
    """
    theta = gmst_rad(epoch_unix)
    rz = rot_z(theta)
    r_ecef = rz @ np.asarray(r_inertial, dtype=np.float64)
    a_ecef = tesseral_22_acceleration_ecef(r_ecef)
    return (rz.T @ a_ecef).astype(np.float64)


def accel_hifi_long_arc(
    r_vec: np.ndarray, epoch_unix: float
) -> np.ndarray:
    """Long-arc higher-fidelity acceleration (Loop 47):
    two-body + J2..J6 + luni-solar third body + dominant non-zonal sectoral
    spherical-harmonic terms C(2,2) and S(2,2). The sectoral terms are
    rotated from the Earth-fixed frame to the GMST-only pseudo-inertial
    frame so the entire acceleration is internally consistent with the
    existing higher-fidelity propagator.
    """
    r_vec = np.asarray(r_vec, dtype=np.float64)
    r = float(np.linalg.norm(r_vec))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    a = accel_hifi_extended(r_vec, epoch_unix)
    a = a + tesseral_22_acceleration_inertial(r_vec, epoch_unix)
    return a.astype(np.float64)


def _deriv_hifi(state: np.ndarray, epoch_unix: float) -> np.ndarray:
    a = accel_hifi(state[:3], epoch_unix)
    return np.hstack([state[3:6], a]).astype(np.float64)


def propagate_hifi(
    state: np.ndarray,
    dt_s: float,
    epoch0_unix: float,
    max_step_s: float = 30.0,
) -> np.ndarray:
    """RK4 propagation of a 6-vector with the higher-fidelity model.

    ``epoch0_unix`` is the absolute UTC epoch of ``state``; the integrator
    advances the Sun/Moon ephemeris with the true epoch within each step.
    """
    state = np.asarray(state, dtype=np.float64).copy()
    if dt_s == 0.0:
        return state
    sign = 1.0 if dt_s > 0 else -1.0
    remaining = abs(dt_s)
    t = float(epoch0_unix)
    while remaining > 1e-9:
        step = sign * min(max_step_s, remaining)
        k1 = _deriv_hifi(state, t)
        k2 = _deriv_hifi(state + 0.5 * step * k1, t + 0.5 * step)
        k3 = _deriv_hifi(state + 0.5 * step * k2, t + 0.5 * step)
        k4 = _deriv_hifi(state + step * k3, t + step)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += step
        remaining -= abs(step)
    return state.astype(np.float64)


def _stm_hifi(
    state: np.ndarray,
    dt_s: float,
    epoch0_unix: float,
    max_step_s: float = 30.0,
) -> np.ndarray:
    """Numerical state-transition Jacobian of the higher-fidelity flow."""
    phi = np.zeros((6, 6), dtype=np.float64)
    perturb = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3], dtype=np.float64)
    for j in range(6):
        dp = np.zeros(6)
        dp[j] = perturb[j]
        sp = propagate_hifi(state + dp, dt_s, epoch0_unix, max_step_s)
        sm = propagate_hifi(state - dp, dt_s, epoch0_unix, max_step_s)
        phi[:, j] = (sp - sm) / (2.0 * perturb[j])
    return phi


def run_range_ekf_hifi(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    max_step_s: float = 30.0,
) -> dict:
    """Range-only EKF on the higher-fidelity dynamics (epoch-aware)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    r_nom = float(range_std_m) ** 2
    records: list[dict] = []
    t_prev = obs[0].epoch_unix
    for o in obs:
        dt = o.epoch_unix - t_prev
        if dt != 0.0:
            phi = _stm_hifi(x, dt, t_prev, max_step_s)
            x = propagate_hifi(x, dt, t_prev, max_step_s)
            p = phi @ p @ phi.T + _process_cov(dt, accel_psd)
            t_prev = o.epoch_unix
        rng, hvec = _range_and_jac(x, o.station_pi_m)
        innov = o.range_m - rng
        s = float(hvec @ p @ hvec) + r_nom
        k = (p @ hvec) / s
        x = x + k * innov
        p = p - np.outer(k, hvec) @ p
        p = 0.5 * (p + p.T)
        records.append(
            {"epoch_unix": o.epoch_unix, "innovation_m": innov,
             "s": s, "nis_r": innov * innov / r_nom}
        )
    return {"state": x, "cov": p, "records": records}


def _run_range_ukf_hifi(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    adaptive: bool,
    max_step_s: float = 30.0,
    r_window: int = 8,
    r_scale_cap: float = 50.0,
) -> dict:
    """Range-only UKF on the higher-fidelity dynamics (epoch-aware).

    Mirrors :func:`_run_range_ukf` exactly except for the epoch-aware
    higher-fidelity propagation, so the fixed/adaptive contrast and the DBAR
    inputs remain directly comparable to the compact slice.
    """
    n = 6
    x = np.asarray(x0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    alpha, beta, kappa = 1e-3, 2.0, 0.0
    wm, wc, lam = _ukf_weights(n, alpha, beta, kappa)
    r_nom = float(range_std_m) ** 2
    recent_sq: list[float] = []
    records: list[dict] = []
    t_prev = obs[0].epoch_unix
    for o in obs:
        dt = o.epoch_unix - t_prev
        if dt != 0.0:
            pts = _sigma_points(x, p, lam)
            prop = np.vstack(
                [propagate_hifi(pt, dt, t_prev, max_step_s) for pt in pts]
            )
            x = wm @ prop
            dx = prop - x
            p = (wc[:, None, None] * dx[:, :, None] * dx[:, None, :]).sum(0)
            p = p + _process_cov(dt, accel_psd)
            p = 0.5 * (p + p.T)
            t_prev = o.epoch_unix

        pts = _sigma_points(x, p, lam)
        z_pred = np.array(
            [np.linalg.norm(pt[:3] - o.station_pi_m) for pt in pts]
        )
        z_mean = float(wm @ z_pred)
        dz = z_pred - z_mean
        innov = o.range_m - z_mean
        nis_r = innov * innov / r_nom

        r_eff_scale = 1.0
        if adaptive:
            recent_sq.append(innov * innov)
            if len(recent_sq) > r_window:
                recent_sq.pop(0)
            pzz_nom = float((wc * dz) @ dz) + r_nom
            mean_sq = float(np.mean(recent_sq))
            r_eff_scale = float(
                np.clip(mean_sq / max(pzz_nom, 1e-9), 1.0, r_scale_cap)
            )
        r_used = r_nom * r_eff_scale
        pzz = float((wc * dz) @ dz) + r_used
        pxz = (wc[:, None] * (pts - x) * dz[:, None]).sum(0)
        k = pxz / pzz
        x = x + k * innov
        p = p - np.outer(k, k) * pzz
        p = 0.5 * (p + p.T)
        records.append(
            {
                "epoch_unix": o.epoch_unix,
                "innovation_m": innov,
                "nis_r": nis_r,
                "r_eff_scale": r_eff_scale,
            }
        )
    return {"state": x, "cov": p, "records": records}


def run_range_ukf_fixed_hifi(
    obs, x0, p0, range_std_m, accel_psd, max_step_s: float = 30.0
) -> dict:
    return _run_range_ukf_hifi(
        obs, x0, p0, range_std_m, accel_psd, False, max_step_s
    )


def run_range_aukf_hifi(
    obs, x0, p0, range_std_m, accel_psd, max_step_s: float = 30.0
) -> dict:
    return _run_range_ukf_hifi(
        obs, x0, p0, range_std_m, accel_psd, True, max_step_s
    )


def held_out_position_rmse_hifi(
    final_state: np.ndarray,
    fit_last_epoch: float,
    held_epochs: np.ndarray,
    interp: Sp3Interpolator,
    max_step_s: float = 30.0,
) -> dict:
    """Predict-only higher-fidelity propagation scored vs precise SP3."""
    errs: list[float] = []
    state = np.asarray(final_state, dtype=np.float64).copy()
    t_prev = float(fit_last_epoch)
    for te in np.asarray(held_epochs, dtype=np.float64):
        state = propagate_hifi(state, te - t_prev, t_prev, max_step_s)
        t_prev = float(te)
        ref = interp.position_pseudo_inertial_m(te)
        errs.append(float(np.linalg.norm(state[:3] - ref)))
    arr = np.asarray(errs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "rms_m": float("nan"), "p95_abs_m": float("nan")}
    return {
        "count": int(arr.size),
        "rms_m": float(np.sqrt(np.mean(arr**2))),
        "mean_m": float(np.mean(arr)),
        "p95_abs_m": float(np.percentile(arr, 95.0)),
    }
