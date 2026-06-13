"""Strictly held-out SP3-supervised empirical dynamics-residual calibrator.

This module implements a lightweight *learned* (least-squares-fit) calibrator
for the compact two-body+J2 dynamics used in the bounded precise-reference
real-data OD slice (:mod:`gnn_state_estimation.sp3`).  It is the loop-27
attempt at a genuine positive, externally validated real-data contribution
requested by the Acta reviewer.

Design (predeclared, no leakage):

* For each *training* arc, the independent ILRS analysis-centre SP3-c precise
  orbit is differentiated twice in the shared pseudo-inertial frame to obtain
  the realised total acceleration ``a_true``.  The compact two-body+J2 model
  acceleration ``a_model`` is removed, giving the unmodelled residual
  acceleration ``da = a_true - a_model`` (luni-solar third body, geopotential
  beyond J2, SRP, relativistic, and the bounded GMST-frame approximation).
* ``da`` is expressed in the orbit-local radial / along-track / cross-track
  (RSW) frame and regressed on a low-order Fourier series in the argument of
  latitude ``u`` --- the classical empirical-acceleration parameterisation
  (constant + 1/rev + 2/rev per axis, 15 ridge-regularised coefficients).  The
  fitted correction is a function of the *runtime state alone*, so it is usable
  predict-only with no future information.
* The calibrator is **never fitted on the held-out arc**.  Two strict
  protocols are evaluated: leave-one-arc-out (LOAO) and the stronger
  leave-one-object-out (LOOO, train on LAGEOS-1 -> score LAGEOS-2 and vice
  versa).  Held-out error remains a predict-only state error versus the
  external SP3 precise reference, identical to the uncalibrated slice.

The simulator filters and :mod:`gnn_state_estimation.sp3` are intentionally
left untouched; the corrected propagation and the corrected range-only
EKF / fixed-noise UKF are reimplemented here in isolation so the existing
slice carries zero regression risk.  Absolute magnitudes inherit the same
bounded fidelity as the uncalibrated slice and are reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import acceleration_eci
from .sp3 import RangeObs, Sp3Interpolator

# Identical to ``sp3._LAGEOS_BALLISTIC_COEFF``; LAGEOS drag is negligible so the
# compact model is effectively two-body+J2.  A unit test pins this equality.
LAGEOS_BALLISTIC_COEFF = 1.0e-9

# Predeclared calibrator hyper-parameters (fixed a priori, not tuned to the
# held-out scores).  The Fourier order and ridge strength are chosen from
# standard empirical-acceleration practice, not from a held-out grid search.
FOURIER_ORDER = 2          # constant + 1/rev + 2/rev per RSW axis
RIDGE_LAMBDA = 1.0e-18     # tiny Tikhonov term for numerical conditioning
ACCEL_SAMPLE_STEP_S = 60.0  # SP3 residual-acceleration training grid spacing
SECOND_DIFF_STEP_S = 20.0  # central-difference step for the SP3 acceleration


def compact_acceleration(state: np.ndarray) -> np.ndarray:
    """Two-body+J2 acceleration of the compact model (drag negligible)."""
    r = np.asarray(state[:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)
    return acceleration_eci(
        r, v, ballistic_coeff_m2_per_kg=LAGEOS_BALLISTIC_COEFF
    )


def rsw_basis(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial / along-track / cross-track unit vectors for ``state``."""
    r = np.asarray(state[:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    r_hat = r / max(r_norm, 1.0)
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    w_hat = h / max(h_norm, 1.0)            # cross-track (orbit normal)
    s_hat = np.cross(w_hat, r_hat)          # along-track (in-plane)
    return r_hat, s_hat, w_hat


def argument_of_latitude(state: np.ndarray) -> float:
    """Argument of latitude ``u`` (angle from ascending node to the sat)."""
    r_hat, _, w_hat = rsw_basis(state)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    node = np.cross(z, w_hat)
    node_norm = float(np.linalg.norm(node))
    if node_norm < 1e-9:                     # near-equatorial degeneracy
        node = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        node_norm = 1.0
    node_hat = node / node_norm
    in_plane = np.cross(w_hat, node_hat)
    return float(np.arctan2(np.dot(r_hat, in_plane), np.dot(r_hat, node_hat)))


def fourier_features(u: float, order: int = FOURIER_ORDER) -> np.ndarray:
    """``[1, cos u, sin u, cos 2u, sin 2u, ...]`` Fourier basis."""
    feats = [1.0]
    for k in range(1, order + 1):
        feats.append(np.cos(k * u))
        feats.append(np.sin(k * u))
    return np.asarray(feats, dtype=np.float64)


@dataclass(frozen=True)
class ResidualCalibrator:
    """Fitted empirical RSW residual-acceleration correction.

    ``beta`` is ``(3, n_features)``: rows are the radial / along-track /
    cross-track empirical-acceleration coefficients.
    """

    beta: np.ndarray
    order: int
    ridge_lambda: float
    n_train_samples: int

    def acceleration(self, state: np.ndarray) -> np.ndarray:
        """Calibrator correction acceleration in the inertial frame."""
        r_hat, s_hat, w_hat = rsw_basis(state)
        u = argument_of_latitude(state)
        phi = fourier_features(u, self.order)
        a_rsw = self.beta @ phi
        return a_rsw[0] * r_hat + a_rsw[1] * s_hat + a_rsw[2] * w_hat


def sp3_residual_samples(
    interp: Sp3Interpolator,
    t_start: float,
    t_end: float,
    *,
    step_s: float = ACCEL_SAMPLE_STEP_S,
    h_s: float = SECOND_DIFF_STEP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """SP3-derived (Fourier features, RSW residual acceleration) samples.

    The SP3 pseudo-inertial position is differentiated twice (central) for the
    realised acceleration; the compact-model acceleration at the SP3 state is
    removed.  Returns ``(Phi, da_rsw)`` stacked over the sample grid.
    """
    eph = interp.eph
    lo = max(t_start, eph.start_unix + 3.0 * h_s)
    hi = min(t_end, eph.end_unix - 3.0 * h_s)
    if hi <= lo:
        return (
            np.zeros((0, 2 * FOURIER_ORDER + 1)),
            np.zeros((0, 3)),
        )
    n = max(2, int(np.floor((hi - lo) / step_s)) + 1)
    grid = np.linspace(lo, hi, n)
    phi_rows: list[np.ndarray] = []
    da_rows: list[np.ndarray] = []
    for t in grid:
        p0 = interp.position_pseudo_inertial_m(t)
        p_plus = interp.position_pseudo_inertial_m(t + h_s)
        p_minus = interp.position_pseudo_inertial_m(t - h_s)
        a_true = (p_plus - 2.0 * p0 + p_minus) / (h_s * h_s)
        state = interp.state_pseudo_inertial_m(t)
        a_model = compact_acceleration(state)
        da = a_true - a_model
        r_hat, s_hat, w_hat = rsw_basis(state)
        da_rsw = np.array(
            [np.dot(da, r_hat), np.dot(da, s_hat), np.dot(da, w_hat)],
            dtype=np.float64,
        )
        phi_rows.append(fourier_features(argument_of_latitude(state)))
        da_rows.append(da_rsw)
    return np.vstack(phi_rows), np.vstack(da_rows)


def fit_calibrator(
    phi: np.ndarray,
    da_rsw: np.ndarray,
    *,
    order: int = FOURIER_ORDER,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> ResidualCalibrator:
    """Ridge least-squares fit of the empirical RSW correction."""
    phi = np.asarray(phi, dtype=np.float64)
    da_rsw = np.asarray(da_rsw, dtype=np.float64)
    n_feat = phi.shape[1]
    if phi.shape[0] < n_feat:
        # Under-determined: fall back to a zero correction (honest no-op).
        return ResidualCalibrator(
            beta=np.zeros((3, n_feat)),
            order=order,
            ridge_lambda=ridge_lambda,
            n_train_samples=int(phi.shape[0]),
        )
    gram = phi.T @ phi + ridge_lambda * np.eye(n_feat)
    rhs = phi.T @ da_rsw                     # (n_feat, 3)
    coeff = np.linalg.solve(gram, rhs)       # (n_feat, 3)
    return ResidualCalibrator(
        beta=coeff.T,                        # (3, n_feat)
        order=order,
        ridge_lambda=ridge_lambda,
        n_train_samples=int(phi.shape[0]),
    )


# --- Corrected propagation and corrected range-only filters -----------------
# Reimplemented in isolation (sp3.py untouched). The corrected derivative adds
# the calibrator acceleration, which depends on the runtime state alone.
def _deriv(state: np.ndarray, calib: ResidualCalibrator) -> np.ndarray:
    a = compact_acceleration(state) + calib.acceleration(state)
    return np.hstack([state[3:6], a]).astype(np.float64)


def _rk4_corrected(
    state: np.ndarray, dt: float, calib: ResidualCalibrator
) -> np.ndarray:
    k1 = _deriv(state, calib)
    k2 = _deriv(state + 0.5 * dt * k1, calib)
    k3 = _deriv(state + 0.5 * dt * k2, calib)
    k4 = _deriv(state + dt * k3, calib)
    return (state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).astype(
        np.float64
    )


def propagate_corrected(
    state: np.ndarray,
    dt_s: float,
    calib: ResidualCalibrator,
    max_step_s: float = 30.0,
) -> np.ndarray:
    """Compact two-body+J2 propagation plus the learned RSW correction."""
    state = np.asarray(state, dtype=np.float64).copy()
    if dt_s == 0.0:
        return state
    sign = 1.0 if dt_s > 0 else -1.0
    remaining = abs(dt_s)
    while remaining > 1e-9:
        step = sign * min(max_step_s, remaining)
        state = _rk4_corrected(state, step, calib)
        remaining -= abs(step)
    return state


def _proc_cov(dt_s: float, accel_psd: float) -> np.ndarray:
    dt = abs(float(dt_s))
    q = accel_psd
    blk = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        blk[i, i] = (dt**4) / 4.0 * q
        blk[i, i + 3] = (dt**3) / 2.0 * q
        blk[i + 3, i] = (dt**3) / 2.0 * q
        blk[i + 3, i + 3] = (dt**2) * q
    return blk


def _range_and_jac(
    state: np.ndarray, station_pi: np.ndarray
) -> tuple[float, np.ndarray]:
    los = state[:3] - station_pi
    rng = float(np.linalg.norm(los))
    h = np.zeros(6, dtype=np.float64)
    if rng > 1.0:
        h[:3] = los / rng
    return rng, h


def _stm_corrected(
    state: np.ndarray, dt_s: float, calib: ResidualCalibrator, max_step_s: float
) -> np.ndarray:
    perturb = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3], dtype=np.float64)
    phi = np.zeros((6, 6), dtype=np.float64)
    for j in range(6):
        dp = np.zeros(6)
        dp[j] = perturb[j]
        sp = propagate_corrected(state + dp, dt_s, calib, max_step_s)
        sm = propagate_corrected(state - dp, dt_s, calib, max_step_s)
        phi[:, j] = (sp - sm) / (2.0 * perturb[j])
    return phi


def run_corrected_ekf(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    calib: ResidualCalibrator,
    max_step_s: float = 30.0,
) -> dict:
    """Range-only EKF on the calibrator-corrected dynamics."""
    x = np.asarray(x0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    r_nom = float(range_std_m) ** 2
    t_prev = obs[0].epoch_unix
    for o in obs:
        dt = o.epoch_unix - t_prev
        if dt != 0.0:
            stm = _stm_corrected(x, dt, calib, max_step_s)
            x = propagate_corrected(x, dt, calib, max_step_s)
            p = stm @ p @ stm.T + _proc_cov(dt, accel_psd)
            t_prev = o.epoch_unix
        rng, hvec = _range_and_jac(x, o.station_pi_m)
        innov = o.range_m - rng
        s = float(hvec @ p @ hvec) + r_nom
        k = (p @ hvec) / s
        x = x + k * innov
        p = p - np.outer(k, hvec) @ p
        p = 0.5 * (p + p.T)
    return {"state": x, "cov": p}


def _sigma_points(x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    n = x.size
    p_sym = 0.5 * (p + p.T) + 1e-6 * np.eye(n)
    try:
        sm = np.linalg.cholesky((n + lam) * p_sym)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh((n + lam) * p_sym)
        w = np.clip(w, 1e-9, None)
        sm = v @ np.diag(np.sqrt(w))
    pts = np.zeros((2 * n + 1, n), dtype=np.float64)
    pts[0] = x
    for i in range(n):
        pts[1 + i] = x + sm[:, i]
        pts[1 + n + i] = x - sm[:, i]
    return pts


def run_corrected_ukf_fixed(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    calib: ResidualCalibrator,
    max_step_s: float = 30.0,
) -> dict:
    """Fixed-noise range-only UKF on the calibrator-corrected dynamics."""
    n = 6
    x = np.asarray(x0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    alpha, beta_w, kappa = 1e-3, 2.0, 0.0
    lam = alpha * alpha * (n + kappa) - n
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1.0 - alpha * alpha + beta_w)
    r_nom = float(range_std_m) ** 2
    t_prev = obs[0].epoch_unix
    for o in obs:
        dt = o.epoch_unix - t_prev
        if dt != 0.0:
            pts = _sigma_points(x, p, lam)
            prop = np.vstack(
                [propagate_corrected(pt, dt, calib, max_step_s) for pt in pts]
            )
            x = wm @ prop
            dx = prop - x
            p = (wc[:, None, None] * dx[:, :, None] * dx[:, None, :]).sum(0)
            p = p + _proc_cov(dt, accel_psd)
            p = 0.5 * (p + p.T)
            t_prev = o.epoch_unix
        pts = _sigma_points(x, p, lam)
        z_pred = np.array(
            [np.linalg.norm(pt[:3] - o.station_pi_m) for pt in pts]
        )
        z_mean = float(wm @ z_pred)
        dz = z_pred - z_mean
        innov = o.range_m - z_mean
        pzz = float((wc * dz) @ dz) + r_nom
        pxz = (wc[:, None] * (pts - x) * dz[:, None]).sum(0)
        k = pxz / pzz
        x = x + k * innov
        p = p - np.outer(k, k) * pzz
        p = 0.5 * (p + p.T)
    return {"state": x, "cov": p}


def held_out_position_rmse_corrected(
    final_state: np.ndarray,
    fit_last_epoch: float,
    held_epochs: np.ndarray,
    interp: Sp3Interpolator,
    calib: ResidualCalibrator,
    max_step_s: float = 30.0,
) -> dict:
    """Predict-only corrected propagation scored vs the external SP3 state."""
    errs: list[float] = []
    state = np.asarray(final_state, dtype=np.float64).copy()
    t_prev = float(fit_last_epoch)
    for te in np.asarray(held_epochs, dtype=np.float64):
        state = propagate_corrected(state, te - t_prev, calib, max_step_s)
        t_prev = float(te)
        ref = interp.position_pseudo_inertial_m(te)
        errs.append(float(np.linalg.norm(state[:3] - ref)))
    arr = np.asarray(errs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "rms_m": float("nan")}
    return {
        "count": int(arr.size),
        "rms_m": float(np.sqrt(np.mean(arr**2))),
        "mean_m": float(np.mean(arr)),
        "p95_abs_m": float(np.percentile(arr, 95.0)),
    }
