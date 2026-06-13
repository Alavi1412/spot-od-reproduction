"""Stochastic (perturbed-observation) Ensemble Kalman Filter for OD.

This estimator is the Monte-Carlo ensemble counterpart of the EKF/UKF family
used throughout this study. Instead of propagating an analytic mean and
covariance (EKF) or a deterministic sigma-point set (UKF), it carries a finite
ensemble of state realizations through the *same* compact two-body + J2 + drag
deterministic flow, samples the process noise into the ensemble, and performs
the measurement update with the stochastic perturbed-observation analysis of
Evensen (1994, 2003) and Burgers, van Leeuwen, and Evensen (1998). The
posterior mean and covariance reported at each step are the ensemble sample
mean and sample covariance, so the EnKF can be scored against the same
observed-step position-RMSE metric as every other estimator.

The motivation for including it is purely to close the Ensemble-Kalman-family
exclusion risk: the manuscript's claim audit compares EKF/UKF/AUKF/PUKF and
several structural-channel augmentations, but the ensemble Gaussian filter is a
major nonlinear-filtering family in its own right. This implementation keeps
the *exact same* deterministic flow, process-noise model, measurement model,
and topocentric-azimuth de-weighting as the EKF/UKF baselines, so the only
structural difference is the Monte-Carlo ensemble representation of the
forecast/analysis distribution. No part of the recursion is tuned against any
held-out result.

Stochastic analysis step (one visible station, sequential assimilation):

.. math::

    K = P^{xy} (P^{yy} + R)^{-1}, \\qquad
    x^{(i)} \\leftarrow x^{(i)} + K\\,(z + v^{(i)} - h(x^{(i)})),

with :math:`v^{(i)} \\sim \\mathcal N(0, R)` independent perturbations,
:math:`P^{xy}` and :math:`P^{yy}` the ensemble sample cross- and
innovation-covariances, and :math:`R` the nominal measurement-noise covariance
(optionally azimuth-de-weighted exactly as in the EKF/UKF). The perturbed
observations make the updated ensemble covariance statistically consistent with
the Kalman posterior in the large-ensemble linear-Gaussian limit.

The recursion is seeded (``EnKFConfig.seed``) so a given trajectory produces a
bit-for-bit reproducible filtered path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..coordinates import StationGeometry, line_of_sight_measurement
from .ekf import (
    azimuth_deweight_factor,
    build_init_covariance,
    build_process_covariance,
    wrap_angle_pi,
)
from .ukf import _safe_sigma_propagation, _stabilize_cov


@dataclass(frozen=True)
class EnKFConfig:
    """Stochastic Ensemble Kalman Filter configuration.

    Attributes
    ----------
    q_pos_m, q_vel_mps:
        Continuous-time process-noise standard deviations applied per second
        to the position and velocity blocks; match the UKF baseline so the
        only structural difference is the ensemble representation. The
        discrete-time process covariance is sampled into the ensemble at every
        predict step (the defining feature of the *stochastic* EnKF).
    init_pos_std_m, init_vel_std_mps:
        Initial-state standard deviations (identical to the UKF baseline); the
        initial ensemble is drawn from ``N(x0_est, P0)``.
    ensemble_size:
        Number of Monte-Carlo members. Predeclared, not tuned.
    inflation:
        Multiplicative covariance inflation applied to the *forecast* ensemble
        anomalies before each measurement update (``>= 1.0``). The default of
        ``1.0`` is the pure canonical EnKF with no inflation; a value slightly
        above one is the standard remedy for ensemble under-dispersion from
        finite-ensemble sampling error.
    seed:
        RNG seed for the initial ensemble draw, the process-noise sampling, and
        the observation perturbations, so a trajectory's filtered path is
        reproducible.
    angle_deweight_elev_cap_deg:
        Optional topocentric-azimuth de-weighting cap, applied identically to
        every estimator in the study (see
        :func:`gnn_state_estimation.filters.ekf.azimuth_deweight_factor`).
    """

    q_pos_m: float = 4.0
    q_vel_mps: float = 0.04
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    ensemble_size: int = 64
    inflation: float = 1.0
    seed: int = 20260163
    angle_deweight_elev_cap_deg: float | None = None


def _ensemble_mean_cov(ensemble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and (N-1)-normalized sample covariance of an ensemble."""
    n = ensemble.shape[0]
    mean = np.mean(ensemble, axis=0)
    anom = ensemble - mean
    cov = (anom.T @ anom) / max(n - 1, 1)
    return mean, _stabilize_cov(cov)


def _predicted_meas_mean(z_ens: np.ndarray) -> np.ndarray:
    """Ensemble mean of predicted measurements, angle-aware on azimuth.

    The azimuth channel (index 1) is cyclic, so its mean is computed as the
    circular mean (``arctan2`` of the mean sine/cosine); range, elevation, and
    range-rate use the arithmetic mean. Mirrors the UKF's ``_mean_with_angle``.
    """
    mean = np.mean(z_ens, axis=0)
    s = float(np.mean(np.sin(z_ens[:, 1])))
    c = float(np.mean(np.cos(z_ens[:, 1])))
    mean[1] = float(np.arctan2(s, c))
    return mean


def run_enkf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: EnKFConfig,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run the stochastic perturbed-observation EnKF on one trajectory.

    Returns ``(x_hat, p_hat, diagnostics)`` where ``x_hat`` is the per-step
    ensemble-mean state ``[T, 6]``, ``p_hat`` the per-step ensemble sample
    covariance ``[T, 6, 6]`` (so the EnKF is scored on the same observed-step
    position-RMSE metric as every other estimator), and ``diagnostics`` carries
    the per-step ensemble spread for auditing ensemble health.
    """
    t_len = measurements.shape[0]
    n = 6
    n_ens = max(int(cfg.ensemble_size), 2)
    rng = np.random.default_rng(int(cfg.seed))

    x_hat = np.zeros((t_len, n), dtype=np.float64)
    p_hat = np.zeros((t_len, n, n), dtype=np.float64)
    pos_spread = np.zeros(t_len, dtype=np.float64)
    vel_spread = np.zeros(t_len, dtype=np.float64)

    ekf_cfg_like = type(
        "cfg",
        (),
        {"init_pos_std_m": cfg.init_pos_std_m, "init_vel_std_mps": cfg.init_vel_std_mps},
    )
    p0 = _stabilize_cov(build_init_covariance(ekf_cfg_like))  # type: ignore[arg-type]

    # Draw the initial ensemble from N(x0_est, P0).
    x0 = np.asarray(x0_est, dtype=np.float64)
    chol0 = np.linalg.cholesky(p0)
    ensemble = x0[None, :] + (rng.standard_normal((n_ens, n)) @ chol0.T)

    inflation = max(float(cfg.inflation), 1.0)
    base_r_diag = np.asarray(meas_std_vector, dtype=np.float64) ** 2
    std_vec = np.asarray(meas_std_vector, dtype=np.float64)

    q_cfg_like = type("cfg", (), {"q_pos_m": cfg.q_pos_m, "q_vel_mps": cfg.q_vel_mps})

    mean0, cov0 = _ensemble_mean_cov(ensemble)
    x_hat[0] = mean0
    p_hat[0] = cov0
    pos_spread[0] = float(np.sqrt(max(np.trace(cov0[:3, :3]), 0.0)))
    vel_spread[0] = float(np.sqrt(max(np.trace(cov0[3:, 3:]), 0.0)))

    prop_kwargs = dict(
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
        enable_third_body=enable_third_body,
        enable_srp=enable_srp,
        srp_area_to_mass_m2_per_kg=srp_area_to_mass_m2_per_kg,
        srp_cr=srp_cr,
        sun_initial_phase_rad=sun_initial_phase_rad,
        moon_initial_phase_rad=moon_initial_phase_rad,
    )

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])
        q = build_process_covariance(q_cfg_like, dt)  # type: ignore[arg-type]
        q_diag = np.clip(np.diag(q), 0.0, None)
        q_sqrt = np.sqrt(q_diag)

        # --- Forecast: propagate each member through the compact flow and
        #     sample the discrete-time process noise into the ensemble. ---
        forecast = np.zeros_like(ensemble)
        for i in range(n_ens):
            step = _safe_sigma_propagation(
                ensemble[i],
                dt=dt,
                t_s=float(times_s[k - 1]),
                **prop_kwargs,
            )
            forecast[i] = step
        forecast = forecast + rng.standard_normal((n_ens, n)) * q_sqrt[None, :]

        # Multiplicative inflation of the forecast anomalies (no-op at 1.0).
        if inflation > 1.0:
            f_mean = np.mean(forecast, axis=0)
            forecast = f_mean[None, :] + inflation * (forecast - f_mean[None, :])

        ensemble = forecast

        # --- Analysis: sequential per-station stochastic update. ---
        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue

            z = np.asarray(measurements[k, s_idx], dtype=np.float64)

            z_ens = np.zeros((n_ens, 4), dtype=np.float64)
            for i in range(n_ens):
                z_i, _ = line_of_sight_measurement(ensemble[i], station, times_s[k])
                z_ens[i] = z_i

            z_mean = _predicted_meas_mean(z_ens)
            x_mean = np.mean(ensemble, axis=0)

            # Innovation/state anomalies (azimuth wrapped to (-pi, pi]).
            y_anom = z_ens - z_mean
            y_anom[:, 1] = (y_anom[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
            x_anom = ensemble - x_mean

            denom = max(n_ens - 1, 1)
            p_xy = (x_anom.T @ y_anom) / denom
            p_yy = (y_anom.T @ y_anom) / denom

            r_eff_diag = base_r_diag.copy()
            if cfg.angle_deweight_elev_cap_deg is not None:
                r_eff_diag[1] *= azimuth_deweight_factor(
                    float(z_mean[2]), cfg.angle_deweight_elev_cap_deg
                )
            s_mat = p_yy + np.diag(r_eff_diag)
            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)
            k_gain = p_xy @ s_inv

            # Perturbed observations: independent N(0, R_eff) draws per member.
            r_std = np.sqrt(np.clip(r_eff_diag, 0.0, None))
            perturb = rng.standard_normal((n_ens, 4)) * r_std[None, :]
            innovations = (z[None, :] + perturb) - z_ens
            innovations[:, 1] = (innovations[:, 1] + np.pi) % (2.0 * np.pi) - np.pi

            ensemble = ensemble + innovations @ k_gain.T

            if not np.all(np.isfinite(ensemble)):
                finite_rows = np.all(np.isfinite(ensemble), axis=1)
                if np.any(finite_rows):
                    repl = np.mean(ensemble[finite_rows], axis=0)
                else:
                    repl = x_mean
                ensemble[~finite_rows] = repl

        mean_k, cov_k = _ensemble_mean_cov(ensemble)
        x_hat[k] = mean_k
        p_hat[k] = cov_k
        pos_spread[k] = float(np.sqrt(max(np.trace(cov_k[:3, :3]), 0.0)))
        vel_spread[k] = float(np.sqrt(max(np.trace(cov_k[3:, 3:]), 0.0)))

    diagnostics: dict[str, Any] = {
        "ensemble_size": int(n_ens),
        "inflation": float(inflation),
        "pos_spread_m": pos_spread,
        "vel_spread_mps": vel_spread,
        "mean_pos_spread_m": float(np.mean(pos_spread[1:])) if t_len > 1 else float("nan"),
        "mean_vel_spread_mps": float(np.mean(vel_spread[1:])) if t_len > 1 else float("nan"),
        "meas_std_vector": std_vec,
    }
    return x_hat, p_hat, diagnostics
