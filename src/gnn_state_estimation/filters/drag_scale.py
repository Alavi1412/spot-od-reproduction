"""Drag-Scale Adaptive Extended Kalman Filter (DSA-EKF).

This estimator extends the compact six-dimensional orbital state with one
multiplicative drag-scaling parameter that scales the deterministic drag
acceleration. It is the textbook ballistic-coefficient/drag-scale estimation
approach prescribed by classical statistical OD when the dominant residual is
a drag/density/ballistic-coefficient mismatch (Tapley, Schutz, and Born 2004,
Section 10.5; Vallado 2013, Section 9.6.1). The augmented state ``[r, v, beta]``
absorbs the drag-acceleration scaling channel directly, rather than the
generic empirical-acceleration channel that the loop 44 DMC-EKF found to be
structurally inert under sparse visibility.

The dynamics in the augmented state are

.. math::

    \\dot r = v,
    \\qquad
    \\dot v = a_\\text{grav}(r) + a_{J_2}(r) + \\beta\\, a_\\text{drag}(r, v),
    \\qquad
    \\dot \\beta = -(\\beta - 1) / \\tau_\\beta,

with the drag scale modelled as a first-order Gauss--Markov process about its
nominal value of one. The discrete-time process covariance for the drag-scale
block under the Gauss--Markov model is
``Q_beta(dt) = sigma_beta^2 * (1 - exp(-2*dt/tau_beta))`` so the steady-state
variance equals ``sigma_beta^2``.

The estimator's deterministic flow uses the same compact two-body+J2+drag
acceleration as every other classical baseline in this study, with the drag
acceleration scaled by ``beta``; the only structural difference from the
EKF/UKF/AUKF/PUKF/DMC family is the multiplicative drag channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import EARTH_ROTATION_RATE, J2, MU_EARTH, R_EARTH
from ..coordinates import StationGeometry, line_of_sight_measurement
from ..dynamics import (
    atmospheric_density_exponential,
    moon_position_eci,
    numerical_jacobian,
    srp_acceleration,
    sun_position_eci,
    third_body_acceleration,
)
from ..constants import MU_MOON, MU_SUN
from .ekf import (
    azimuth_deweight_factor,
    wrap_angle_pi,
)
from .ukf import _stabilize_cov, _ukf_weights


@dataclass(frozen=True)
class DragScaleAEKFConfig:
    """Drag-Scale Adaptive EKF configuration.

    Attributes
    ----------
    q_pos_m, q_vel_mps:
        Compact-state continuous-time process-noise standard deviations
        applied to the position and velocity blocks per second; match the
        EKF/UKF/AUKF baselines so the only structural difference is the drag
        scaling channel.
    init_pos_std_m, init_vel_std_mps:
        Initial-state standard deviations (identical to the EKF baseline).
    init_drag_scale_std:
        Initial standard deviation of the drag-scale parameter beta around 1.
    drag_scale_sigma_ss:
        Steady-state Gauss--Markov standard deviation of beta.
    drag_scale_tau_s:
        First-order Gauss--Markov decorrelation time for beta.
    gating_threshold:
        Chi-square gating threshold for innovation Mahalanobis distance.
    angle_deweight_elev_cap_deg:
        Optional topocentric-azimuth de-weighting cap.
    """

    q_pos_m: float = 5.0
    q_vel_mps: float = 0.05
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    init_drag_scale_std: float = 0.5
    drag_scale_sigma_ss: float = 0.5
    drag_scale_tau_s: float = 600.0
    gating_threshold: float = 18.47
    angle_deweight_elev_cap_deg: float | None = None


def _build_init_covariance(cfg: DragScaleAEKFConfig) -> np.ndarray:
    p = np.array(
        [
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
            cfg.init_drag_scale_std**2,
        ],
        dtype=np.float64,
    )
    return np.diag(p)


def _build_process_covariance(cfg: DragScaleAEKFConfig, dt: float) -> np.ndarray:
    dt_pos = max(dt, 1.0)
    q_diag = np.array(
        [
            cfg.q_pos_m**2 * dt_pos,
            cfg.q_pos_m**2 * dt_pos,
            cfg.q_pos_m**2 * dt_pos,
            cfg.q_vel_mps**2 * dt_pos,
            cfg.q_vel_mps**2 * dt_pos,
            cfg.q_vel_mps**2 * dt_pos,
            0.0,
        ],
        dtype=np.float64,
    )
    q = np.diag(q_diag)
    tau = max(cfg.drag_scale_tau_s, 1e-3)
    s2 = cfg.drag_scale_sigma_ss**2
    q[6, 6] = s2 * (1.0 - float(np.exp(-2.0 * dt / tau)))
    return q


def _acceleration_scaled_drag(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    beta: float,
    *,
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
    r2 = float(np.dot(r_eci, r_eci))
    r = float(np.sqrt(r2))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    x, y, z = r_eci[0], r_eci[1], r_eci[2]
    a_grav = -MU_EARTH * r_eci / (r**3)
    z2 = z * z
    factor = 1.5 * J2 * MU_EARTH * (R_EARTH**2) / (r**5)
    common = 5.0 * z2 / r2
    a_j2 = factor * np.array(
        [x * (common - 1.0), y * (common - 1.0), z * (common - 3.0)], dtype=np.float64
    )
    altitude = r - R_EARTH
    rho = atmospheric_density_exponential(
        altitude_m=altitude,
        rho_ref=drag_rho_ref,
        h_ref_m=drag_h_ref_m,
        scale_height_m=drag_scale_height_m,
    )
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE], dtype=np.float64)
    v_atm = np.cross(omega, r_eci)
    v_rel = v_eci - v_atm
    v_rel_norm = float(np.linalg.norm(v_rel))
    a_drag = -0.5 * ballistic_coeff_m2_per_kg * rho * v_rel_norm * v_rel * beta
    a_total = a_grav + a_j2 + a_drag
    if enable_third_body or enable_srp:
        r_sun = sun_position_eci(t_s=t_s, initial_phase_rad=sun_initial_phase_rad)
    if enable_third_body:
        r_moon = moon_position_eci(t_s=t_s, initial_phase_rad=moon_initial_phase_rad)
        a_total = a_total + third_body_acceleration(r_sat=r_eci, r_body=r_sun, mu_body=MU_SUN)
        a_total = a_total + third_body_acceleration(r_sat=r_eci, r_body=r_moon, mu_body=MU_MOON)
    if enable_srp:
        a_total = a_total + srp_acceleration(
            r_sat=r_eci,
            r_sun=r_sun,
            area_to_mass_m2_per_kg=srp_area_to_mass_m2_per_kg,
            cr=srp_cr,
        )
    return a_total


def _dsa_state_derivative(
    state: np.ndarray,
    *,
    t_s: float,
    tau_beta_s: float,
    ballistic_coeff_m2_per_kg: float,
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
    r = state[:3]
    v = state[3:6]
    beta = float(state[6])
    a = _acceleration_scaled_drag(
        r,
        v,
        beta,
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
    beta_dot = -(beta - 1.0) / max(tau_beta_s, 1e-3)
    return np.hstack([v, a, np.array([beta_dot], dtype=np.float64)]).astype(np.float64)


def _dsa_rk4_step(
    state: np.ndarray,
    dt: float,
    **kwargs,
) -> np.ndarray:
    k1 = _dsa_state_derivative(state, **kwargs)
    kw_mid = dict(kwargs)
    kw_mid["t_s"] = kwargs["t_s"] + 0.5 * dt
    k2 = _dsa_state_derivative(state + 0.5 * dt * k1, **kw_mid)
    k3 = _dsa_state_derivative(state + 0.5 * dt * k2, **kw_mid)
    kw_end = dict(kwargs)
    kw_end["t_s"] = kwargs["t_s"] + dt
    k4 = _dsa_state_derivative(state + dt * k3, **kw_end)
    return (state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float64)


def run_drag_scale_aekf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: DragScaleAEKFConfig,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run the 7-state DSA-EKF on one trajectory.

    Returns the 6-component state history so the DSA-EKF can be scored
    against the same observed-step position-RMSE metric as every other
    estimator. The drag-scale history is returned in the diagnostics.
    """
    t_len = measurements.shape[0]
    x_hat = np.zeros((t_len, 6), dtype=np.float64)
    p_hat = np.zeros((t_len, 6, 6), dtype=np.float64)
    beta_history = np.zeros(t_len, dtype=np.float64)

    x = np.zeros(7, dtype=np.float64)
    x[:6] = np.asarray(x0_est, dtype=np.float64)
    x[6] = 1.0
    p = _build_init_covariance(cfg)
    x_hat[0] = x[:6]
    p_hat[0] = p[:6, :6]
    beta_history[0] = x[6]

    r_meas = np.diag(np.asarray(meas_std_vector, dtype=np.float64) ** 2)

    step_kwargs = dict(
        tau_beta_s=cfg.drag_scale_tau_s,
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

    h_pad = np.zeros((4, 7), dtype=np.float64)

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])
        kw_step = dict(step_kwargs)
        kw_step["t_s"] = float(times_s[k - 1])

        def f_fn(state_in: np.ndarray) -> np.ndarray:
            return _dsa_rk4_step(state_in, dt=dt, **kw_step)

        f_jac = numerical_jacobian(f_fn, x, eps=1e-3)
        x = f_fn(x)
        q = _build_process_covariance(cfg, dt)
        p = f_jac @ p @ f_jac.T + q
        p = 0.5 * (p + p.T)

        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue
            z = measurements[k, s_idx]

            def h_fn(state_in: np.ndarray) -> np.ndarray:
                z_pred, _ = line_of_sight_measurement(state_in[:6], station, times_s[k])
                return z_pred

            h_jac6 = numerical_jacobian(lambda y: h_fn(np.hstack([y, x[6:7]])), x[:6], eps=1e-2)
            h_jac = h_pad.copy()
            h_jac[:, :6] = h_jac6
            z_pred = h_fn(x)
            y = z - z_pred
            y[1] = wrap_angle_pi(float(y[1]))
            r_base = r_meas
            if cfg.angle_deweight_elev_cap_deg is not None:
                r_base = r_meas.copy()
                r_base[1, 1] *= azimuth_deweight_factor(
                    float(z_pred[2]), cfg.angle_deweight_elev_cap_deg
                )
            s_mat = h_jac @ p @ h_jac.T + r_base
            r_eff = r_base
            try:
                s_inv = np.linalg.inv(s_mat)
            except np.linalg.LinAlgError:
                s_inv = np.linalg.pinv(s_mat)
            maha = float(y.T @ s_inv @ y)
            if maha > cfg.gating_threshold:
                r_eff = 9.0 * r_base
                s_mat = h_jac @ p @ h_jac.T + r_eff
                try:
                    s_inv = np.linalg.inv(s_mat)
                except np.linalg.LinAlgError:
                    s_inv = np.linalg.pinv(s_mat)
            k_gain = p @ h_jac.T @ s_inv
            x = x + k_gain @ y
            i_kh = np.eye(7) - k_gain @ h_jac
            p = i_kh @ p @ i_kh.T + k_gain @ r_eff @ k_gain.T
            p = 0.5 * (p + p.T)

        # Soft bound to keep beta physically positive.
        if x[6] < 0.05:
            x[6] = 0.05
        if x[6] > 5.0:
            x[6] = 5.0

        x_hat[k] = x[:6]
        p_hat[k] = p[:6, :6]
        beta_history[k] = x[6]

    diagnostics = {"drag_scale_history": beta_history}
    return x_hat, p_hat, diagnostics


# ---------------------------------------------------------------------------
# Drag-Scale Adaptive Unscented Kalman Filter (DSA-UKF)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DragScaleAUKFConfig:
    """Drag-Scale Adaptive UKF configuration.

    The DSA-UKF augments the six-dimensional orbital state with one
    multiplicative drag-scaling parameter beta, like the DSA-EKF, but performs
    the predict step by deterministic sigma-point propagation through the same
    augmented-state RK4 integrator. The motivation is that on a slice with a
    pure multiplicative drag-scale truth-side bias, the structural channel is
    correct by construction; the EKF's analytic linearisation of the
    seven-dimensional flow at large beta perturbations and at very low
    altitudes can introduce a substantial mean-trajectory error that the
    sigma-point family avoids by deterministic propagation.

    Attributes
    ----------
    q_pos_m, q_vel_mps:
        Continuous-time process-noise standard deviations applied per second
        to the position and velocity blocks; match the UKF baseline so the
        only structural difference is the multiplicative drag channel.
    init_pos_std_m, init_vel_std_mps:
        Initial-state standard deviations (identical to the UKF baseline).
    init_drag_scale_std:
        Initial standard deviation of beta around 1.
    drag_scale_sigma_ss:
        Steady-state Gauss--Markov standard deviation of beta.
    drag_scale_tau_s:
        First-order Gauss--Markov decorrelation time for beta.
    alpha, beta, kappa:
        Standard scaled-unscented-transform parameters; the parameter name
        ``beta`` here is the UKF sigma-point distribution parameter, not the
        drag scale.
    beta_min, beta_max:
        Hard saturation bounds applied to the drag-scale state after every
        update so the channel cannot run away to nonphysical values.
    angle_deweight_elev_cap_deg:
        Optional topocentric-azimuth de-weighting cap, applied identically
        to every estimator in the study.
    """

    q_pos_m: float = 4.0
    q_vel_mps: float = 0.04
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    init_drag_scale_std: float = 0.5
    drag_scale_sigma_ss: float = 0.5
    drag_scale_tau_s: float = 600.0
    alpha: float = 0.2
    beta: float = 2.0
    kappa: float = 0.0
    beta_min: float = 0.05
    beta_max: float = 5.0
    angle_deweight_elev_cap_deg: float | None = None


def _build_init_covariance_dsa_ukf(cfg: DragScaleAUKFConfig) -> np.ndarray:
    p = np.array(
        [
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
            cfg.init_drag_scale_std**2,
        ],
        dtype=np.float64,
    )
    return np.diag(p)


def _build_process_covariance_dsa_ukf(cfg: DragScaleAUKFConfig, dt: float) -> np.ndarray:
    dt_pos = max(dt, 1.0)
    q = np.diag(
        np.array(
            [
                cfg.q_pos_m**2 * dt_pos,
                cfg.q_pos_m**2 * dt_pos,
                cfg.q_pos_m**2 * dt_pos,
                cfg.q_vel_mps**2 * dt_pos,
                cfg.q_vel_mps**2 * dt_pos,
                cfg.q_vel_mps**2 * dt_pos,
                0.0,
            ],
            dtype=np.float64,
        )
    )
    tau = max(cfg.drag_scale_tau_s, 1e-3)
    s2 = cfg.drag_scale_sigma_ss**2
    q[6, 6] = s2 * (1.0 - float(np.exp(-2.0 * dt / tau)))
    return q


def _safe_dsa_sigma_step(
    sigma: np.ndarray,
    *,
    dt: float,
    t_s: float,
    tau_beta_s: float,
    beta_min: float,
    beta_max: float,
    ballistic_coeff_m2_per_kg: float,
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
    """Numerically-guarded augmented-state RK4 step for one UKF sigma point.

    The sigma point's position, velocity, and drag-scale components are each
    clipped to physically reasonable ranges both before integration (so a
    momentarily ill-conditioned sigma-point covariance cannot produce an
    Inf-acceleration evaluation) and after integration (so a single sigma
    point that diverges cannot corrupt the predicted mean and covariance via
    a non-finite contribution). If the RK4 step itself emits non-finite or
    over/underflow signals, the sigma point is returned at its pre-step
    clipped value, which is the conservative fallback used elsewhere in this
    project's sigma-point machinery (see :func:`_safe_sigma_propagation`).
    """
    s_in = np.asarray(sigma, dtype=np.float64)
    s_clip = s_in.copy()
    s_clip[:3] = np.clip(s_clip[:3], -8.0e7, 8.0e7)
    s_clip[3:6] = np.clip(s_clip[3:6], -2.0e4, 2.0e4)
    s_clip[6] = float(np.clip(s_clip[6], beta_min, beta_max))
    kwargs = dict(
        t_s=t_s,
        tau_beta_s=tau_beta_s,
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
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            step = _dsa_rk4_step(s_clip, dt=dt, **kwargs)
    except FloatingPointError:
        step = s_clip
    except OverflowError:
        step = s_clip
    if not np.all(np.isfinite(step)):
        step = s_clip
    step = np.asarray(step, dtype=np.float64)
    step[:3] = np.clip(step[:3], -8.0e7, 8.0e7)
    step[3:6] = np.clip(step[3:6], -2.0e4, 2.0e4)
    step[6] = float(np.clip(step[6], beta_min, beta_max))
    return step


def run_drag_scale_aukf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: DragScaleAUKFConfig,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
    enable_third_body: bool = False,
    enable_srp: bool = False,
    srp_area_to_mass_m2_per_kg: float = 0.02,
    srp_cr: float = 1.3,
    sun_initial_phase_rad: float = 0.0,
    moon_initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run the seven-state Drag-Scale Adaptive UKF on one trajectory.

    Returns ``(x_hat[:, 6], p_hat[:, 6, 6], diagnostics)`` so the DSA-UKF can
    be scored against the same observed-step position-RMSE metric as every
    other estimator. The drag-scale history is returned in the diagnostics.
    """
    t_len = measurements.shape[0]
    n = 7
    x_hat = np.zeros((t_len, 6), dtype=np.float64)
    p_hat = np.zeros((t_len, 6, 6), dtype=np.float64)
    beta_history = np.zeros(t_len, dtype=np.float64)

    x = np.zeros(n, dtype=np.float64)
    x[:6] = np.asarray(x0_est, dtype=np.float64)
    x[6] = 1.0
    p = _stabilize_cov(_build_init_covariance_dsa_ukf(cfg))

    w_m, w_c, lam = _ukf_weights(
        n=n, alpha=cfg.alpha, beta=cfg.beta, kappa=cfg.kappa
    )
    r_meas = np.diag(np.asarray(meas_std_vector, dtype=np.float64) ** 2)

    x_hat[0] = x[:6]
    p_hat[0] = p[:6, :6]
    beta_history[0] = x[6]

    step_kwargs = dict(
        tau_beta_s=cfg.drag_scale_tau_s,
        beta_min=cfg.beta_min,
        beta_max=cfg.beta_max,
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
        q = _build_process_covariance_dsa_ukf(cfg, dt)

        # Sigma-point propagation in the augmented seven-dimensional state.
        sig = _sigma_points_n(x, p, lam)
        sig_pred = np.zeros_like(sig)
        for i in range(sig.shape[0]):
            sig_pred[i] = _safe_dsa_sigma_step(
                sig[i],
                dt=dt,
                t_s=float(times_s[k - 1]),
                **step_kwargs,
            )

        x = np.sum(sig_pred * w_m[:, None], axis=0)
        # Defensive: if the mean somehow ends up non-finite, snap the offending
        # component back to a safe value drawn from the carried estimate.
        if not np.all(np.isfinite(x)):
            x = np.where(np.isfinite(x), x, 0.0)
        x[6] = float(np.clip(x[6], cfg.beta_min, cfg.beta_max))

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
                z_i, _ = line_of_sight_measurement(
                    sig_pred[i, :6], station, times_s[k]
                )
                z_sig[i] = z_i

            # Angle-aware mean for the azimuth channel; range, elevation,
            # and range-rate use the standard weighted mean.
            z_mean = np.sum(z_sig * w_m[:, None], axis=0)
            s_az = float(np.sum(w_m * np.sin(z_sig[:, 1])))
            c_az = float(np.sum(w_m * np.cos(z_sig[:, 1])))
            z_mean[1] = float(np.arctan2(s_az, c_az))

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

        x[6] = float(np.clip(x[6], cfg.beta_min, cfg.beta_max))
        if not np.all(np.isfinite(x)):
            x = np.where(np.isfinite(x), x, 0.0)
            x[6] = float(np.clip(x[6], cfg.beta_min, cfg.beta_max))

        x_hat[k] = x[:6]
        p_hat[k] = p[:6, :6]
        beta_history[k] = x[6]

    diagnostics = {"drag_scale_history": beta_history}
    return x_hat, p_hat, diagnostics


def _sigma_points_n(x: np.ndarray, p: np.ndarray, lam: float) -> np.ndarray:
    """Augmented-state sigma-point generator.

    Identical in form to :func:`gnn_state_estimation.filters.ukf._sigma_points`
    but agnostic to the state dimension; the augmented seven-dimensional state
    of the DSA-UKF makes the six-dimensional helper inapplicable directly.
    """
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
