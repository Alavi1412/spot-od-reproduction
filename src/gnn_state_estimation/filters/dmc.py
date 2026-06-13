"""Dynamic Model Compensation (DMC) Extended Kalman Filter.

This estimator extends the compact six-dimensional orbital state with three
first-order Gauss--Markov empirical-acceleration components per axis
(:math:`\\tau`-decorrelated, zero-mean, stationary), and integrates that
empirical acceleration as part of the deterministic flow. The 9-state DMC-EKF
absorbs unmodelled accelerations through a structural model channel rather
than through innovation-driven measurement-noise reweighting, which is the
operational response that statistical OD prescribes when the dominant
residual is a dynamics/force-model bias (Tapley, Schutz, and Born 2004,
\\S10; Wright 1981; Stacey and D'Amico 2021).

The dynamics in the augmented state :math:`X = [r, v, w]^T` are

.. math::

    \\dot r = v, \\qquad
    \\dot v = a_\\text{compact}(r, v) + w, \\qquad
    \\dot w = -w/\\tau,

with a continuous-time Gauss--Markov process driving :math:`w`. The
estimator's deterministic flow uses the same compact two-body+J2+drag
acceleration as every other classical baseline in this study, so the only
difference from the EKF/UKF/AUKF/PUKF family is the structural
empirical-acceleration channel.

The discrete-time process covariance for the empirical-acceleration block
under the Gauss--Markov model is

.. math::

    Q_w(\\Delta t) = \\sigma_w^2 \\cdot (1 - \\exp(-2 \\Delta t / \\tau)) \\cdot I_3,

so the steady-state empirical-acceleration variance equals :math:`\\sigma_w^2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coordinates import StationGeometry, line_of_sight_measurement
from ..dynamics import numerical_jacobian, rk4_step
from .ekf import (
    azimuth_deweight_factor,
    wrap_angle_pi,
)


@dataclass(frozen=True)
class DMCEKFConfig:
    """DMC-EKF configuration.

    All thresholds and process-noise parameters are predeclared in
    ``release/predeclarations/dmc_ekf_rule_loop44.json`` and are not chosen
    by inspecting any held-out result.

    Attributes
    ----------
    q_pos_m, q_vel_mps:
        Compact-state continuous-time process-noise standard deviations
        applied to the position and velocity blocks per second; these match
        the EKF/UKF/AUKF baselines so the only structural difference is the
        empirical-acceleration channel.
    init_pos_std_m, init_vel_std_mps:
        Initial-state standard deviations (identical to the EKF baseline).
    init_emp_accel_std_mps2:
        Initial standard deviation of the empirical-acceleration block.
    emp_accel_sigma_mps2:
        Steady-state Gauss--Markov standard deviation of the
        empirical-acceleration channel.
    emp_accel_tau_s:
        First-order Gauss--Markov decorrelation time. Order of magnitude is
        a small fraction of the orbital period; 300 s is the operational
        default for LEO drag/atmospheric-density residuals.
    gating_threshold:
        Chi-square gating threshold for innovation Mahalanobis distance,
        identical to the EKF baseline.
    angle_deweight_elev_cap_deg:
        Optional topocentric-azimuth de-weighting cap, identical to the EKF
        baseline.
    """

    q_pos_m: float = 5.0
    q_vel_mps: float = 0.05
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    init_emp_accel_std_mps2: float = 1.0e-6
    emp_accel_sigma_mps2: float = 1.0e-6
    emp_accel_tau_s: float = 300.0
    gating_threshold: float = 18.47
    angle_deweight_elev_cap_deg: float | None = None


def _build_dmc_init_covariance(cfg: DMCEKFConfig) -> np.ndarray:
    p = np.array(
        [
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
            cfg.init_emp_accel_std_mps2**2,
            cfg.init_emp_accel_std_mps2**2,
            cfg.init_emp_accel_std_mps2**2,
        ],
        dtype=np.float64,
    )
    return np.diag(p)


def _build_dmc_process_covariance(cfg: DMCEKFConfig, dt: float) -> np.ndarray:
    """Discrete-time process covariance for the DMC-augmented state.

    Position and velocity blocks follow the same diagonal continuous-time
    spectral-density model as the EKF baseline so that with the empirical
    block disabled (sigma==0) the recursion reduces to a standard EKF; the
    empirical block uses the Gauss--Markov stationary increment.
    """
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
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    q = np.diag(q_diag)
    tau = max(cfg.emp_accel_tau_s, 1e-3)
    s2 = cfg.emp_accel_sigma_mps2**2
    gm_var = s2 * (1.0 - float(np.exp(-2.0 * dt / tau)))
    q[6, 6] = gm_var
    q[7, 7] = gm_var
    q[8, 8] = gm_var
    return q


def _dmc_step(
    state: np.ndarray,
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
    emp_accel_tau_s: float,
) -> np.ndarray:
    """One DMC integration step.

    Splits the step into a base compact-state RK4 (driven by the
    instantaneous empirical acceleration treated as a constant additive force
    over the step) and an analytic Gauss--Markov decay of the empirical
    block.
    """
    r = state[:3]
    v = state[3:6]
    w = state[6:9]
    base_state = np.hstack([r, v]).astype(np.float64)

    base_kwargs = dict(
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
    base_next = rk4_step(base_state, dt=dt, t_s=t_s, **base_kwargs)

    # Add the empirical-acceleration impulse to velocity using an
    # exponentially-decaying step-average (integral of w(t) = w0 exp(-t/tau)
    # from 0 to dt is w0 tau (1-exp(-dt/tau))).
    tau = max(emp_accel_tau_s, 1e-3)
    factor = tau * (1.0 - float(np.exp(-dt / tau)))
    velocity_kick = w * factor
    position_kick = w * (dt * tau - tau**2 * (1.0 - float(np.exp(-dt / tau))))

    next_state = np.zeros(9, dtype=np.float64)
    next_state[:3] = base_next[:3] + position_kick
    next_state[3:6] = base_next[3:6] + velocity_kick
    next_state[6:9] = w * float(np.exp(-dt / tau))
    return next_state


def run_dmc_ekf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: DMCEKFConfig,
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
    """Run the 9-state DMC-EKF on one trajectory.

    Returns the 6-component state history (position and velocity only) so the
    DMC-EKF can be scored against the same observed-step position-RMSE metric
    as every other estimator. The empirical-acceleration estimates are
    returned in the diagnostics dictionary so they can be audited.
    """
    t_len = measurements.shape[0]
    x_hat = np.zeros((t_len, 6), dtype=np.float64)
    p_hat = np.zeros((t_len, 6, 6), dtype=np.float64)
    emp_accel_history = np.zeros((t_len, 3), dtype=np.float64)

    x = np.zeros(9, dtype=np.float64)
    x[:6] = np.asarray(x0_est, dtype=np.float64)
    p = _build_dmc_init_covariance(cfg)
    x_hat[0] = x[:6]
    p_hat[0] = p[:6, :6]
    emp_accel_history[0] = x[6:9]

    r_meas = np.diag(np.asarray(meas_std_vector, dtype=np.float64) ** 2)
    step_kwargs = dict(
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
        emp_accel_tau_s=cfg.emp_accel_tau_s,
    )

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])

        def f_fn(state_in: np.ndarray) -> np.ndarray:
            return _dmc_step(state_in, dt=dt, t_s=float(times_s[k - 1]), **step_kwargs)

        f_jac = numerical_jacobian(f_fn, x, eps=1e-3)
        x = f_fn(x)
        q = _build_dmc_process_covariance(cfg, dt)
        p = f_jac @ p @ f_jac.T + q
        p = 0.5 * (p + p.T)

        h_pad = np.zeros((4, 9), dtype=np.float64)
        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue
            z = measurements[k, s_idx]

            def h_fn(state_in: np.ndarray) -> np.ndarray:
                z_pred, _ = line_of_sight_measurement(state_in[:6], station, times_s[k])
                return z_pred

            h_jac6 = numerical_jacobian(lambda y: h_fn(np.hstack([y, x[6:9]])), x[:6], eps=1e-2)
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
            i_kh = np.eye(9) - k_gain @ h_jac
            p = i_kh @ p @ i_kh.T + k_gain @ r_eff @ k_gain.T
            p = 0.5 * (p + p.T)

        x_hat[k] = x[:6]
        p_hat[k] = p[:6, :6]
        emp_accel_history[k] = x[6:9]

    diagnostics = {"empirical_acceleration_mps2": emp_accel_history}
    return x_hat, p_hat, diagnostics
