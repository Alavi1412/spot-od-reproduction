"""Strictly held-out learned residual calibrator on the higher-fidelity model.

Loop-28 second attempt at a genuine, externally validated positive real-data
contribution.  Unlike the loop-27 calibrator -- which tried to learn a *mean*
empirical acceleration on top of a compact two-body+J2 model whose dominant
unmodelled signal was the day/geometry-varying luni-solar third body -- this
calibrator sits on top of the higher-fidelity model (luni-solar third body and
J3/J4 already modelled physically, evaluated in a proper IAU-76/80 inertial
frame).  The *remaining* SP3 residual is therefore much smaller and dominated
by smooth, slowly varying, recurring effects (the neglected polar-motion /
UT1-UTC Earth-orientation terms, solid-Earth tides, relativity, SP3
interpolation), which have a genuine, physically motivated chance of
transferring across held-out days and the held-out object.

Design (predeclared, strict no-leakage):

* For each *training* arc the independent ILRS SP3 precise orbit is
  differentiated twice in the proper inertial frame to obtain the realised
  acceleration; the higher-fidelity model acceleration is removed, leaving the
  residual ``da`` expressed in the radial / along-track / cross-track frame.
* ``da`` is regressed (ridge least squares) on a predeclared basis: a 2/rev
  Fourier series in the argument of latitude crossed with an Earth-rotation
  phase (``1, cos g, sin g`` with ``g`` the Greenwich sidereal angle) plus a
  slow secular term -- i.e. exactly the structure of the dominant remaining
  Earth-orientation/geopotential residual, not an arbitrary high-capacity fit.
* The ridge strength is selected only on a disjoint *validation* week and the
  frozen rule is then evaluated on a strictly later *test* week and, more
  stringently, on the held-out object (train on one LAGEOS, score the other).
  The held-out arc's SP3 and ranges never enter any fit.

This module is self-contained; :mod:`gnn_state_estimation.sp3` and the
committed slices are untouched (zero regression risk).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sp3 import accel_hifi

# Predeclared calibrator hyper-parameters (fixed a priori from standard
# empirical-acceleration / Earth-rotation-phase practice, not tuned to the
# held-out test scores).  Only the ridge strength is selected, and only on a
# disjoint validation week from a fixed small predeclared grid.
FOURIER_ORDER = 2
SECOND_DIFF_STEP_S = 20.0
ACCEL_SAMPLE_STEP_S = 60.0
RIDGE_GRID = (1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6)


def rsw_basis(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(state[:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)
    r_hat = r / max(float(np.linalg.norm(r)), 1.0)
    h = np.cross(r, v)
    w_hat = h / max(float(np.linalg.norm(h)), 1.0)
    s_hat = np.cross(w_hat, r_hat)
    return r_hat, s_hat, w_hat


def argument_of_latitude(state: np.ndarray) -> float:
    r_hat, _, w_hat = rsw_basis(state)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    node = np.cross(z, w_hat)
    nn = float(np.linalg.norm(node))
    if nn < 1e-9:
        node = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        nn = 1.0
    node_hat = node / nn
    in_plane = np.cross(w_hat, node_hat)
    return float(
        np.arctan2(np.dot(r_hat, in_plane), np.dot(r_hat, node_hat))
    )


def _earth_phase(epoch_unix: float) -> float:
    """Sidereal Earth-rotation phase (rad); transferable epoch feature."""
    # Greenwich mean sidereal angle modulo 2*pi; smooth and recurring.
    return (epoch_unix * 7.2921159e-5) % (2.0 * np.pi)


def feature_vector(
    state: np.ndarray, epoch_unix: float, t0_unix: float
) -> np.ndarray:
    """Predeclared transfer basis: (u-Fourier) x (Earth-phase) + secular."""
    u = argument_of_latitude(state)
    g = _earth_phase(epoch_unix)
    u_feats = [1.0]
    for k in range(1, FOURIER_ORDER + 1):
        u_feats += [np.cos(k * u), np.sin(k * u)]
    g_feats = [1.0, np.cos(g), np.sin(g)]
    cross = [a * b for a in u_feats for b in g_feats]
    secular = (epoch_unix - t0_unix) / 86400.0  # days, slow drift term
    return np.asarray(cross + [secular], dtype=np.float64)


N_FEATURES = (2 * FOURIER_ORDER + 1) * 3 + 1


@dataclass(frozen=True)
class HifiCalibrator:
    """Fitted ridge RSW residual-acceleration correction (frozen, predict-only)."""

    beta: np.ndarray            # (3, N_FEATURES)
    ridge_lambda: float
    t0_unix: float
    n_train_samples: int
    feature_scale: np.ndarray   # (N_FEATURES,) std used to standardise

    def acceleration(self, state: np.ndarray, epoch_unix: float) -> np.ndarray:
        phi = feature_vector(state, epoch_unix, self.t0_unix) / self.feature_scale
        a_rsw = self.beta @ phi
        r_hat, s_hat, w_hat = rsw_basis(state)
        return a_rsw[0] * r_hat + a_rsw[1] * s_hat + a_rsw[2] * w_hat


def residual_samples(
    interp_pos, state_fn, t_start: float, t_end: float, t0_unix: float,
    *, step_s: float = ACCEL_SAMPLE_STEP_S, h_s: float = SECOND_DIFF_STEP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """(features, RSW residual accel) samples from one arc's SP3 span.

    ``interp_pos(t)`` -> precise inertial position; ``state_fn(t)`` -> precise
    inertial 6-state.  The higher-fidelity model acceleration is removed from
    the twice-differenced SP3 acceleration.
    """
    lo = t_start + 3.0 * h_s
    hi = t_end - 3.0 * h_s
    if hi <= lo:
        return np.zeros((0, N_FEATURES)), np.zeros((0, 3))
    n = max(2, int(np.floor((hi - lo) / step_s)) + 1)
    grid = np.linspace(lo, hi, n)
    phi_rows, da_rows = [], []
    for t in grid:
        p0 = interp_pos(t)
        a_true = (
            interp_pos(t + h_s) - 2.0 * p0 + interp_pos(t - h_s)
        ) / (h_s * h_s)
        state = state_fn(t)
        da = a_true - accel_hifi(state[:3], t)
        r_hat, s_hat, w_hat = rsw_basis(state)
        da_rows.append(
            np.array(
                [da @ r_hat, da @ s_hat, da @ w_hat], dtype=np.float64
            )
        )
        phi_rows.append(feature_vector(state, t, t0_unix))
    return np.vstack(phi_rows), np.vstack(da_rows)


def fit_ridge(
    phi: np.ndarray, da_rsw: np.ndarray, ridge_lambda: float, t0_unix: float
) -> HifiCalibrator:
    phi = np.asarray(phi, dtype=np.float64)
    da_rsw = np.asarray(da_rsw, dtype=np.float64)
    if phi.shape[0] < phi.shape[1]:
        return HifiCalibrator(
            beta=np.zeros((3, N_FEATURES)),
            ridge_lambda=ridge_lambda,
            t0_unix=t0_unix,
            n_train_samples=int(phi.shape[0]),
            feature_scale=np.ones(N_FEATURES),
        )
    scale = phi.std(axis=0)
    scale[scale < 1e-12] = 1.0
    phi_s = phi / scale
    gram = phi_s.T @ phi_s + ridge_lambda * np.eye(phi_s.shape[1])
    coeff = np.linalg.solve(gram, phi_s.T @ da_rsw)  # (n_feat, 3)
    return HifiCalibrator(
        beta=coeff.T,
        ridge_lambda=ridge_lambda,
        t0_unix=t0_unix,
        n_train_samples=int(phi.shape[0]),
        feature_scale=scale,
    )


def propagate_hifi_corrected(
    state: np.ndarray,
    dt_s: float,
    epoch0_unix: float,
    calib: HifiCalibrator,
    max_step_s: float = 30.0,
) -> np.ndarray:
    """Higher-fidelity propagation plus the frozen calibrator correction."""
    state = np.asarray(state, dtype=np.float64).copy()
    if dt_s == 0.0:
        return state
    sign = 1.0 if dt_s > 0 else -1.0
    remaining = abs(dt_s)
    t = float(epoch0_unix)

    def deriv(s: np.ndarray, te: float) -> np.ndarray:
        a = accel_hifi(s[:3], te) + calib.acceleration(s, te)
        return np.hstack([s[3:6], a]).astype(np.float64)

    while remaining > 1e-9:
        step = sign * min(max_step_s, remaining)
        k1 = deriv(state, t)
        k2 = deriv(state + 0.5 * step * k1, t + 0.5 * step)
        k3 = deriv(state + 0.5 * step * k2, t + 0.5 * step)
        k4 = deriv(state + step * k3, t + step)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += step
        remaining -= abs(step)
    return state.astype(np.float64)


def held_out_rmse_corrected(
    final_state: np.ndarray,
    fit_last_epoch: float,
    held_epochs: np.ndarray,
    interp_pos,
    calib: HifiCalibrator,
    max_step_s: float = 30.0,
) -> float:
    errs = []
    state = np.asarray(final_state, dtype=np.float64).copy()
    t_prev = float(fit_last_epoch)
    for te in np.asarray(held_epochs, dtype=np.float64):
        state = propagate_hifi_corrected(
            state, te - t_prev, t_prev, calib, max_step_s
        )
        t_prev = float(te)
        errs.append(float(np.linalg.norm(state[:3] - interp_pos(te))))
    arr = np.asarray(errs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr**2)))
