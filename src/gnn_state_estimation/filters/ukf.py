"""Unscented Kalman Filter for nonlinear orbit determination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..coordinates import StationGeometry, line_of_sight_measurement
from ..dynamics import rk4_step
from .ekf import (
    azimuth_deweight_factor,
    build_init_covariance,
    build_process_covariance,
    wrap_angle_pi,
)

# Measurement channel order for the line-of-sight model, used by the adaptive
# UKF diagnostics so per-channel scale/proposal/effective columns are labelled.
AUKF_MEAS_CHANNELS: tuple[str, ...] = ("range", "azimuth", "elevation", "range_rate")


def _stabilize_cov(p: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    p_sym = 0.5 * (p + p.T)
    eigvals, eigvecs = np.linalg.eigh(p_sym)
    eigvals = np.clip(eigvals, min_eig, None)
    return (eigvecs * eigvals) @ eigvecs.T


@dataclass(frozen=True)
class UKFConfig:
    q_pos_m: float = 4.0
    q_vel_mps: float = 0.04
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    alpha: float = 0.2
    beta: float = 2.0
    kappa: float = 0.0
    # See ``EKFConfig.angle_deweight_elev_cap_deg``. None => bit-for-bit
    # identical to the legacy UKF (default everywhere).
    angle_deweight_elev_cap_deg: float | None = None


@dataclass(frozen=True)
class ProcessNoiseAdaptiveUKFConfig:
    """Process-noise-adaptive UKF (PUKF) configuration.

    Symmetric counterpart of :class:`AdaptiveUKFConfig`: holds the measurement
    noise ``R`` fixed at nominal and instead inflates the process noise ``Q``
    whenever a sliding-window estimate of the R-only normalized innovation
    squared (NIS) exceeds the chi-square-with-4-channels expectation, on the
    classical statistical-OD intuition that under a true dynamics/force-model
    bias the *prior* (not the measurement) is the wrong distribution.

    The decision rule and thresholds are predeclared (loop 41) before
    evaluation and are not chosen by inspecting any held-out result; see
    the predeclared rule artifact under ``release/predeclarations/``.
    """

    q_pos_m: float = 4.0
    q_vel_mps: float = 0.04
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    alpha: float = 0.2
    beta: float = 2.0
    kappa: float = 0.0
    window_size: int = 10
    nis_per_update_expected: float = 4.0
    nis_warn_ratio: float = 1.5
    nis_alarm_ratio: float = 2.0
    q_scale_warn: float = 3.0
    q_scale_alarm: float = 10.0
    q_scale_max: float = 25.0
    smoothing: float = 0.4
    angle_deweight_elev_cap_deg: float | None = None


@dataclass(frozen=True)
class AdaptiveUKFConfig:
    q_pos_m: float = 4.0
    q_vel_mps: float = 0.04
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    alpha: float = 0.2
    beta: float = 2.0
    kappa: float = 0.0
    adapt_rate: float = 0.08
    min_r_scale: float = 0.4
    max_r_scale: float = 25.0
    huber_kappa: float = 2.8
    nis_soft_gate: float = 16.0
    # See ``EKFConfig.angle_deweight_elev_cap_deg``. None => bit-for-bit
    # identical to the legacy adaptive UKF (default everywhere). The geometric
    # azimuth inflation is applied to the effective R that enters the gain
    # *after* the innovation-adaptive R clip, so it cannot be undone by the
    # adaptive mechanism (it is a geometry, not a noise-level, correction).
    angle_deweight_elev_cap_deg: float | None = None


def _ukf_weights(n: int, alpha: float, beta: float, kappa: float) -> tuple[np.ndarray, np.ndarray, float]:
    lam = alpha**2 * (n + kappa) - n
    w_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=np.float64)
    w_c = w_m.copy()
    w_m[0] = lam / (n + lam)
    w_c[0] = lam / (n + lam) + (1.0 - alpha**2 + beta)
    return w_m, w_c, lam


def _sigma_points(x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    n = x.size
    base = _stabilize_cov(p)
    scaled = (n + lam) * base
    root = None
    for jitter in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            root = np.linalg.cholesky(scaled + jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            continue
    if root is None:
        eigvals, eigvecs = np.linalg.eigh(_stabilize_cov(scaled + 1e-1 * np.eye(n)))
        eigvals = np.clip(eigvals, 1e-6, None)
        root = eigvecs @ np.diag(np.sqrt(eigvals))
    sigmas = np.zeros((2 * n + 1, n), dtype=np.float64)
    sigmas[0] = x
    for i in range(n):
        sigmas[i + 1] = x + root[:, i]
        sigmas[n + i + 1] = x - root[:, i]
    return sigmas


def _mean_with_angle(sigmas: np.ndarray, w: np.ndarray, angle_idx: int | None = None) -> np.ndarray:
    m = np.sum(sigmas * w[:, None], axis=0)
    if angle_idx is not None:
        s = float(np.sum(w * np.sin(sigmas[:, angle_idx])))
        c = float(np.sum(w * np.cos(sigmas[:, angle_idx])))
        m[angle_idx] = np.arctan2(s, c)
    return m


def _safe_sigma_propagation(
    sigma: np.ndarray,
    *,
    dt: float,
    ballistic_coeff_m2_per_kg: float,
    t_s: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    enable_third_body: bool,
    enable_srp: bool,
    srp_area_to_mass_m2_per_kg: float,
    srp_cr: float,
    sun_initial_phase_rad: float,
    moon_initial_phase_rad: float,
) -> np.ndarray:
    sigma_in = np.asarray(sigma, dtype=np.float64)
    sigma_clipped = sigma_in.copy()
    sigma_clipped[:3] = np.clip(sigma_clipped[:3], -8.0e7, 8.0e7)
    sigma_clipped[3:] = np.clip(sigma_clipped[3:], -2.0e4, 2.0e4)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            step = rk4_step(
                sigma_clipped,
                dt=dt,
                ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
                t_s=t_s,
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
    except FloatingPointError:
        step = sigma_clipped
    except OverflowError:
        step = sigma_clipped
    if not np.all(np.isfinite(step)):
        return sigma_clipped
    step = np.asarray(step, dtype=np.float64)
    step[:3] = np.clip(step[:3], -8.0e7, 8.0e7)
    step[3:] = np.clip(step[3:], -2.0e4, 2.0e4)
    return step


def run_ukf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: UKFConfig,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    t_len = measurements.shape[0]
    n = 6
    x_hist = np.zeros((t_len, n), dtype=np.float64)
    p_hist = np.zeros((t_len, n, n), dtype=np.float64)

    x = np.asarray(x0_est, dtype=np.float64).copy()
    # Reuse same covariance structure as EKF for easy comparability.
    ekf_cfg_like = type("cfg", (), {"init_pos_std_m": cfg.init_pos_std_m, "init_vel_std_mps": cfg.init_vel_std_mps})
    p = _stabilize_cov(build_init_covariance(ekf_cfg_like))  # type: ignore[arg-type]
    w_m, w_c, lam = _ukf_weights(n=n, alpha=cfg.alpha, beta=cfg.beta, kappa=cfg.kappa)
    r_meas = np.diag(np.asarray(meas_std_vector, dtype=np.float64) ** 2)

    x_hist[0] = x
    p_hist[0] = p

    q_cfg_like = type("cfg", (), {"q_pos_m": cfg.q_pos_m, "q_vel_mps": cfg.q_vel_mps})

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])
        q = build_process_covariance(q_cfg_like, dt)  # type: ignore[arg-type]

        sig = _sigma_points(x, p, lam)
        sig_pred = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            sig_pred[i] = _safe_sigma_propagation(
                sig[i],
                dt=dt,
                ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
                t_s=float(times_s[k - 1]),
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

        x = np.sum(sig_pred * w_m[:, None], axis=0)
        p = q.copy()
        for i in range(sig_pred.shape[0]):
            d = sig_pred[i] - x
            p += w_c[i] * np.outer(d, d)
        p = _stabilize_cov(p)

        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue

            z = measurements[k, s_idx]
            z_sig = np.zeros((sig_pred.shape[0], 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                z_i, _ = line_of_sight_measurement(sig_pred[i], station, times_s[k])
                z_sig[i] = z_i

            z_mean = _mean_with_angle(z_sig, w_m, angle_idx=1)
            s_mat = r_meas.copy()
            if cfg.angle_deweight_elev_cap_deg is not None:
                s_mat[1, 1] *= azimuth_deweight_factor(
                    float(z_mean[2]), cfg.angle_deweight_elev_cap_deg
                )
            p_xz = np.zeros((n, 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                dz = z_sig[i] - z_mean
                dz[1] = wrap_angle_pi(float(dz[1]))
                dx = sig_pred[i] - x
                s_mat += w_c[i] * np.outer(dz, dz)
                p_xz += w_c[i] * np.outer(dx, dz)

            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)
            k_gain = p_xz @ s_inv
            y = z - z_mean
            y[1] = wrap_angle_pi(float(y[1]))
            x = x + k_gain @ y
            p = p - k_gain @ s_mat @ k_gain.T
            p = _stabilize_cov(p)

        x_hist[k] = x
        p_hist[k] = p

    return x_hist, p_hist


def run_adaptive_ukf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: AdaptiveUKFConfig,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Innovation-adaptive and robustified UKF."""
    t_len = measurements.shape[0]
    n = 6
    x_hist = np.zeros((t_len, n), dtype=np.float64)
    p_hist = np.zeros((t_len, n, n), dtype=np.float64)

    x = np.asarray(x0_est, dtype=np.float64).copy()
    ekf_cfg_like = type("cfg", (), {"init_pos_std_m": cfg.init_pos_std_m, "init_vel_std_mps": cfg.init_vel_std_mps})
    p = _stabilize_cov(build_init_covariance(ekf_cfg_like))  # type: ignore[arg-type]
    w_m, w_c, lam = _ukf_weights(n=n, alpha=cfg.alpha, beta=cfg.beta, kappa=cfg.kappa)

    base_r_diag = np.asarray(meas_std_vector, dtype=np.float64) ** 2
    n_stations = len(stations)
    station_r_diag = np.repeat(base_r_diag[None, :], n_stations, axis=0)
    r_diag_min = cfg.min_r_scale * base_r_diag
    r_diag_max = cfg.max_r_scale * base_r_diag

    x_hist[0] = x
    p_hist[0] = p

    q_cfg_like = type("cfg", (), {"q_pos_m": cfg.q_pos_m, "q_vel_mps": cfg.q_vel_mps})

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])
        q = build_process_covariance(q_cfg_like, dt)  # type: ignore[arg-type]

        sig = _sigma_points(x, p, lam)
        sig_pred = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            sig_pred[i] = _safe_sigma_propagation(
                sig[i],
                dt=dt,
                ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
                t_s=float(times_s[k - 1]),
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

        x = np.sum(sig_pred * w_m[:, None], axis=0)
        p = q.copy()
        for i in range(sig_pred.shape[0]):
            d = sig_pred[i] - x
            p += w_c[i] * np.outer(d, d)
        p = _stabilize_cov(p)

        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue

            z = measurements[k, s_idx]
            z_sig = np.zeros((sig_pred.shape[0], 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                z_i, _ = line_of_sight_measurement(sig_pred[i], station, times_s[k])
                z_sig[i] = z_i

            z_mean = _mean_with_angle(z_sig, w_m, angle_idx=1)
            s_pred = np.zeros((4, 4), dtype=np.float64)
            p_xz = np.zeros((n, 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                dz = z_sig[i] - z_mean
                dz[1] = wrap_angle_pi(float(dz[1]))
                dx = sig_pred[i] - x
                s_pred += w_c[i] * np.outer(dz, dz)
                p_xz += w_c[i] * np.outer(dx, dz)

            r_diag = station_r_diag[s_idx]
            s_mat = s_pred + np.diag(r_diag)
            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)

            y = z - z_mean
            y[1] = wrap_angle_pi(float(y[1]))
            nis = float(y.T @ s_inv @ y)
            robust_scale = 1.0
            if nis > cfg.nis_soft_gate:
                robust_scale = np.sqrt(max(nis / max(cfg.nis_soft_gate, 1e-9), 1.0))
            if cfg.huber_kappa > 1e-9:
                robust_scale = max(robust_scale, np.sqrt(max(nis, 1e-12)) / cfg.huber_kappa)

            innovation_diag = np.maximum(y**2 - np.diag(s_pred), 1e-12)
            proposal = np.clip(innovation_diag, r_diag_min, r_diag_max)
            r_diag_new = (1.0 - cfg.adapt_rate) * r_diag + cfg.adapt_rate * proposal
            station_r_diag[s_idx] = np.clip(r_diag_new, r_diag_min, r_diag_max)
            r_eff_diag = np.clip(station_r_diag[s_idx] * (robust_scale**2), r_diag_min, r_diag_max)
            if cfg.angle_deweight_elev_cap_deg is not None:
                r_eff_diag = r_eff_diag.copy()
                r_eff_diag[1] *= azimuth_deweight_factor(
                    float(z_mean[2]), cfg.angle_deweight_elev_cap_deg
                )

            s_mat = s_pred + np.diag(r_eff_diag)
            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)
            k_gain = p_xz @ s_inv
            x = x + k_gain @ y
            p = p - k_gain @ s_mat @ k_gain.T
            p = _stabilize_cov(p)

        x_hist[k] = x
        p_hist[k] = p

    return x_hist, p_hist


def run_adaptive_ukf_instrumented(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: AdaptiveUKFConfig,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Instrumented twin of :func:`run_adaptive_ukf`.

    The numerical recursion is a line-for-line copy of :func:`run_adaptive_ukf`
    (so ``x_hist``/``p_hist`` are bit-for-bit identical and this never alters
    the baseline outputs that the default path produces); the only addition is
    a per-visible-update diagnostic record that quantifies the adaptive
    measurement-noise mechanism. Each record captures, for one
    (time step, station) update:

    * ``pre_adapt_nis`` -- the normalized innovation squared computed against
      the carried (pre-adaptation) station ``R`` diagonal, and whether it
      exceeded the soft gate ``cfg.nis_soft_gate``;
    * ``robust_scale`` -- the Huber/soft-gate robustification factor;
    * the measurement-noise *scale* relative to the nominal ``R`` diagonal
      before adaptation, the clipped *proposal*, the post-adaptation scale,
      and the *effective* scale that actually enters the gain (adapted scale
      times ``robust_scale**2``), per channel and channel-averaged;
    * the resulting state-update norm (full 6D, plus position/velocity parts).

    Returns ``(x_hist, p_hist, records)`` where ``records`` is a list of plain
    dicts (one per visible update) suitable for direct ``pandas`` ingestion.
    """
    t_len = measurements.shape[0]
    n = 6
    x_hist = np.zeros((t_len, n), dtype=np.float64)
    p_hist = np.zeros((t_len, n, n), dtype=np.float64)
    records: list[dict[str, Any]] = []

    x = np.asarray(x0_est, dtype=np.float64).copy()
    ekf_cfg_like = type("cfg", (), {"init_pos_std_m": cfg.init_pos_std_m, "init_vel_std_mps": cfg.init_vel_std_mps})
    p = _stabilize_cov(build_init_covariance(ekf_cfg_like))  # type: ignore[arg-type]
    w_m, w_c, lam = _ukf_weights(n=n, alpha=cfg.alpha, beta=cfg.beta, kappa=cfg.kappa)

    base_r_diag = np.asarray(meas_std_vector, dtype=np.float64) ** 2
    n_stations = len(stations)
    station_r_diag = np.repeat(base_r_diag[None, :], n_stations, axis=0)
    r_diag_min = cfg.min_r_scale * base_r_diag
    r_diag_max = cfg.max_r_scale * base_r_diag
    safe_base_r = np.where(base_r_diag > 0.0, base_r_diag, 1.0)

    x_hist[0] = x
    p_hist[0] = p

    q_cfg_like = type("cfg", (), {"q_pos_m": cfg.q_pos_m, "q_vel_mps": cfg.q_vel_mps})

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])
        q = build_process_covariance(q_cfg_like, dt)  # type: ignore[arg-type]

        sig = _sigma_points(x, p, lam)
        sig_pred = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            sig_pred[i] = _safe_sigma_propagation(
                sig[i],
                dt=dt,
                ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
                t_s=float(times_s[k - 1]),
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

        x = np.sum(sig_pred * w_m[:, None], axis=0)
        p = q.copy()
        for i in range(sig_pred.shape[0]):
            d = sig_pred[i] - x
            p += w_c[i] * np.outer(d, d)
        p = _stabilize_cov(p)

        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue

            z = measurements[k, s_idx]
            z_sig = np.zeros((sig_pred.shape[0], 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                z_i, _ = line_of_sight_measurement(sig_pred[i], station, times_s[k])
                z_sig[i] = z_i

            z_mean = _mean_with_angle(z_sig, w_m, angle_idx=1)
            s_pred = np.zeros((4, 4), dtype=np.float64)
            p_xz = np.zeros((n, 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                dz = z_sig[i] - z_mean
                dz[1] = wrap_angle_pi(float(dz[1]))
                dx = sig_pred[i] - x
                s_pred += w_c[i] * np.outer(dz, dz)
                p_xz += w_c[i] * np.outer(dx, dz)

            r_diag = station_r_diag[s_idx]
            r_diag_pre = r_diag.copy()
            s_mat = s_pred + np.diag(r_diag)
            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)

            y = z - z_mean
            y[1] = wrap_angle_pi(float(y[1]))
            nis = float(y.T @ s_inv @ y)
            robust_scale = 1.0
            if nis > cfg.nis_soft_gate:
                robust_scale = np.sqrt(max(nis / max(cfg.nis_soft_gate, 1e-9), 1.0))
            if cfg.huber_kappa > 1e-9:
                robust_scale = max(robust_scale, np.sqrt(max(nis, 1e-12)) / cfg.huber_kappa)

            innovation_diag = np.maximum(y**2 - np.diag(s_pred), 1e-12)
            proposal = np.clip(innovation_diag, r_diag_min, r_diag_max)
            r_diag_new = (1.0 - cfg.adapt_rate) * r_diag + cfg.adapt_rate * proposal
            station_r_diag[s_idx] = np.clip(r_diag_new, r_diag_min, r_diag_max)
            r_eff_diag = np.clip(station_r_diag[s_idx] * (robust_scale**2), r_diag_min, r_diag_max)
            if cfg.angle_deweight_elev_cap_deg is not None:
                r_eff_diag = r_eff_diag.copy()
                r_eff_diag[1] *= azimuth_deweight_factor(
                    float(z_mean[2]), cfg.angle_deweight_elev_cap_deg
                )

            s_mat = s_pred + np.diag(r_eff_diag)
            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)
            k_gain = p_xz @ s_inv
            state_update = k_gain @ y
            x = x + state_update
            p = p - k_gain @ s_mat @ k_gain.T
            p = _stabilize_cov(p)

            scale_pre = r_diag_pre / safe_base_r
            scale_proposal = proposal / safe_base_r
            scale_post = station_r_diag[s_idx] / safe_base_r
            scale_eff = r_eff_diag / safe_base_r
            record: dict[str, Any] = {
                "time_step": int(k),
                "dt_s": float(dt),
                "station_index": int(s_idx),
                "station_name": str(getattr(station, "name", s_idx)),
                "pre_adapt_nis": float(nis),
                "nis_soft_gate": float(cfg.nis_soft_gate),
                "nis_exceeds_soft_gate": bool(nis > cfg.nis_soft_gate),
                "whitened_innovation_norm": float(np.sqrt(max(nis, 0.0))),
                "robust_scale": float(robust_scale),
                "r_scale_pre_mean": float(np.mean(scale_pre)),
                "r_proposal_scale_mean": float(np.mean(scale_proposal)),
                "r_scale_post_mean": float(np.mean(scale_post)),
                "r_eff_scale_mean": float(np.mean(scale_eff)),
                "state_update_norm": float(np.linalg.norm(state_update)),
                "state_update_pos_norm_m": float(np.linalg.norm(state_update[:3])),
                "state_update_vel_norm_mps": float(np.linalg.norm(state_update[3:])),
            }
            for ch_idx, ch_name in enumerate(AUKF_MEAS_CHANNELS):
                record[f"r_scale_pre_{ch_name}"] = float(scale_pre[ch_idx])
                record[f"r_proposal_scale_{ch_name}"] = float(scale_proposal[ch_idx])
                record[f"r_scale_post_{ch_name}"] = float(scale_post[ch_idx])
                record[f"r_eff_scale_{ch_name}"] = float(scale_eff[ch_idx])
            records.append(record)

        x_hist[k] = x
        p_hist[k] = p

    return x_hist, p_hist, records


def predicted_innovation_nis(
    pred_states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    meas_std_vector: np.ndarray,
) -> list[dict[str, Any]]:
    """Comparable R-only normalized-innovation diagnostic for any predictor.

    Given a single trajectory's filtered/predicted state path (``pred_states``
    with shape ``[t_len, 6]``), compute, for every visible update, the
    whitened measurement residual ``(z - h(x_pred)) / sigma`` and the
    associated R-only NIS (sum of squared whitened residuals over the four
    measurement channels, distributed as chi-square with 4 dof under a
    correctly specified model). Unlike the AUKF's own NIS this uses only the
    fixed measurement-noise diagonal -- no innovation covariance ``S`` -- so it
    is directly comparable across cached EKF/UKF/AUKF predictions, which do not
    carry ``S``. Returns one dict per visible update.
    """
    pred = np.asarray(pred_states, dtype=np.float64)
    meas = np.asarray(measurements, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    std = np.asarray(meas_std_vector, dtype=np.float64).reshape(4)
    inv_std = 1.0 / np.clip(std, 1e-12, None)

    t_len = meas.shape[0]
    out: list[dict[str, Any]] = []
    for t in range(t_len):
        for s_idx, station in enumerate(stations):
            if vis[t, s_idx] < 0.5:
                continue
            z_pred, _ = line_of_sight_measurement(pred[t], station, float(times[t]))
            resid = np.asarray(meas[t, s_idx], dtype=np.float64) - z_pred
            resid[1] = wrap_angle_pi(float(resid[1]))
            resid[2] = wrap_angle_pi(float(resid[2]))
            whitened = resid * inv_std
            nis_r = float(np.dot(whitened, whitened))
            out.append(
                {
                    "time_step": int(t),
                    "station_index": int(s_idx),
                    "station_name": str(getattr(station, "name", s_idx)),
                    "nis_r": nis_r,
                    "whitened_norm": float(np.sqrt(max(nis_r, 0.0))),
                }
            )
    return out


def run_process_noise_adaptive_ukf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: ProcessNoiseAdaptiveUKFConfig,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Process-noise-adaptive UKF.

    Measurement noise ``R`` is held at nominal; the predicted ``Q`` is
    multiplicatively inflated when a sliding-window estimate of the R-only NIS
    suggests the dominant residual is dynamics bias rather than measurement
    noise (predeclared rule, loop 41).

    Returns ``(x_hist, p_hist, records)`` where ``records`` are one dict per
    predict step with the windowed-NIS-driven ``q_scale_used`` actually
    applied to that step's process covariance.
    """
    t_len = measurements.shape[0]
    n = 6
    x_hist = np.zeros((t_len, n), dtype=np.float64)
    p_hist = np.zeros((t_len, n, n), dtype=np.float64)
    records: list[dict[str, Any]] = []

    x = np.asarray(x0_est, dtype=np.float64).copy()
    ekf_cfg_like = type(
        "cfg",
        (),
        {"init_pos_std_m": cfg.init_pos_std_m, "init_vel_std_mps": cfg.init_vel_std_mps},
    )
    p = _stabilize_cov(build_init_covariance(ekf_cfg_like))  # type: ignore[arg-type]
    w_m, w_c, lam = _ukf_weights(n=n, alpha=cfg.alpha, beta=cfg.beta, kappa=cfg.kappa)
    r_meas = np.diag(np.asarray(meas_std_vector, dtype=np.float64) ** 2)
    inv_std = 1.0 / np.clip(np.asarray(meas_std_vector, dtype=np.float64), 1e-12, None)

    x_hist[0] = x
    p_hist[0] = p

    q_cfg_like = type("cfg", (), {"q_pos_m": cfg.q_pos_m, "q_vel_mps": cfg.q_vel_mps})

    # Sliding window of recent R-only NIS values (loop41 predeclared rule).
    nis_window: list[float] = []
    smoothed_q_scale: float = 1.0
    expected = max(float(cfg.nis_per_update_expected), 1e-9)

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])

        # Decide Q-scale from the *previous* window (no use of step-k innovations).
        if nis_window:
            mean_nis = float(np.mean(np.asarray(nis_window, dtype=np.float64)))
        else:
            mean_nis = expected
        ratio = mean_nis / expected
        if ratio >= cfg.nis_alarm_ratio:
            q_scale_target = float(cfg.q_scale_alarm)
        elif ratio >= cfg.nis_warn_ratio:
            q_scale_target = float(cfg.q_scale_warn)
        else:
            q_scale_target = 1.0
        smoothed_q_scale = (1.0 - cfg.smoothing) * smoothed_q_scale + cfg.smoothing * q_scale_target
        q_scale_used = float(min(max(smoothed_q_scale, 1.0), cfg.q_scale_max))

        q = build_process_covariance(q_cfg_like, dt) * q_scale_used  # type: ignore[arg-type]

        sig = _sigma_points(x, p, lam)
        sig_pred = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            sig_pred[i] = _safe_sigma_propagation(
                sig[i],
                dt=dt,
                ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
                t_s=float(times_s[k - 1]),
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

        x = np.sum(sig_pred * w_m[:, None], axis=0)
        p = q.copy()
        for i in range(sig_pred.shape[0]):
            d = sig_pred[i] - x
            p += w_c[i] * np.outer(d, d)
        p = _stabilize_cov(p)

        step_nis_values: list[float] = []
        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue

            z = measurements[k, s_idx]
            z_sig = np.zeros((sig_pred.shape[0], 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                z_i, _ = line_of_sight_measurement(sig_pred[i], station, times_s[k])
                z_sig[i] = z_i

            z_mean = _mean_with_angle(z_sig, w_m, angle_idx=1)
            s_mat = r_meas.copy()
            if cfg.angle_deweight_elev_cap_deg is not None:
                s_mat[1, 1] *= azimuth_deweight_factor(
                    float(z_mean[2]), cfg.angle_deweight_elev_cap_deg
                )
            p_xz = np.zeros((n, 4), dtype=np.float64)
            for i in range(sig_pred.shape[0]):
                dz = z_sig[i] - z_mean
                dz[1] = wrap_angle_pi(float(dz[1]))
                dx = sig_pred[i] - x
                s_mat += w_c[i] * np.outer(dz, dz)
                p_xz += w_c[i] * np.outer(dx, dz)

            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)
            k_gain = p_xz @ s_inv
            y = z - z_mean
            y[1] = wrap_angle_pi(float(y[1]))
            x = x + k_gain @ y
            p = p - k_gain @ s_mat @ k_gain.T
            p = _stabilize_cov(p)

            whitened = y * inv_std
            nis_r = float(np.dot(whitened, whitened))
            step_nis_values.append(nis_r)

        # Update the windowed-NIS estimator with this step's R-only NIS values.
        for nis_val in step_nis_values:
            nis_window.append(nis_val)
        while len(nis_window) > cfg.window_size:
            nis_window.pop(0)

        records.append(
            {
                "time_step": int(k),
                "dt_s": float(dt),
                "windowed_mean_nis_r": float(mean_nis),
                "windowed_nis_ratio": float(ratio),
                "q_scale_target": float(q_scale_target),
                "q_scale_used": float(q_scale_used),
                "n_visible_updates": int(len(step_nis_values)),
            }
        )

        x_hist[k] = x
        p_hist[k] = p

    return x_hist, p_hist, records
