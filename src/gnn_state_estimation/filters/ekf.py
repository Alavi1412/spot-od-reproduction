"""Extended Kalman Filter for nonlinear orbit determination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..coordinates import StationGeometry, line_of_sight_measurement
from ..dynamics import numerical_jacobian, rk4_step


def wrap_angle_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def azimuth_deweight_factor(elevation_rad: float, elev_cap_deg: float) -> float:
    """Topocentric-azimuth measurement-variance inflation, capped.

    A topocentric azimuth conveys almost no cross-track position information
    near the station zenith: a fixed cross-track position error subtends an
    azimuth angle that grows as ``1/cos(elevation)`` (``az`` is geometrically
    singular at zenith). Weighting the azimuth channel with a fixed small
    angular variance therefore makes the measurement update near-singular at
    high elevation and drives Kalman-gain blow-up and filter divergence.
    Inflating the azimuth variance by ``1/cos^2(elevation)`` restores the
    correct (cross-track) information content; the elevation cap bounds the
    factor so a numerical zenith crossing cannot produce an unbounded weight.
    This is the standard topocentric-angle weighting and is applied
    identically to every estimator, so it cannot create a learned advantage.
    """
    cap_rad = np.deg2rad(min(max(float(elev_cap_deg), 0.0), 89.9))
    c = float(np.cos(elevation_rad))
    c_cap = float(np.cos(cap_rad))
    denom = max(c * c, c_cap * c_cap, 1e-12)
    return 1.0 / denom


@dataclass(frozen=True)
class EKFConfig:
    q_pos_m: float = 5.0
    q_vel_mps: float = 0.05
    init_pos_std_m: float = 2e3
    init_vel_std_mps: float = 5.0
    gating_threshold: float = 18.47  # Chi-square approx, dof=4, p=0.001
    # Optional, default-off topocentric-azimuth de-weighting. When None the
    # filter is bit-for-bit identical to the legacy recursion (every existing
    # scenario, table, and pinned number is unchanged). When set to an
    # elevation cap in degrees, the azimuth measurement variance is inflated by
    # the capped 1/cos^2(elevation) factor (see ``azimuth_deweight_factor``),
    # the astrodynamically standard fix for the near-zenith azimuth
    # conditioning defect.
    angle_deweight_elev_cap_deg: float | None = None


def build_process_covariance(cfg: EKFConfig, dt: float) -> np.ndarray:
    q = np.array(
        [
            cfg.q_pos_m**2,
            cfg.q_pos_m**2,
            cfg.q_pos_m**2,
            cfg.q_vel_mps**2,
            cfg.q_vel_mps**2,
            cfg.q_vel_mps**2,
        ],
        dtype=np.float64,
    )
    return np.diag(q * max(dt, 1.0))


def build_init_covariance(cfg: EKFConfig) -> np.ndarray:
    p = np.array(
        [
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_pos_std_m**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
            cfg.init_vel_std_mps**2,
        ],
        dtype=np.float64,
    )
    return np.diag(p)


def _measurement_model(x: np.ndarray, station: StationGeometry, t_s: float) -> np.ndarray:
    z, _ = line_of_sight_measurement(x, station, t_s)
    return z


def run_ekf(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    ballistic_coeff_m2_per_kg: float,
    meas_std_vector: np.ndarray,
    x0_est: np.ndarray,
    cfg: EKFConfig,
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
    """Run EKF on one trajectory.

    Returns:
        x_hat: [T, 6]
        p_hat: [T, 6, 6]
    """
    t_len = measurements.shape[0]
    x_hat = np.zeros((t_len, 6), dtype=np.float64)
    p_hat = np.zeros((t_len, 6, 6), dtype=np.float64)

    x = np.asarray(x0_est, dtype=np.float64).copy()
    p = build_init_covariance(cfg)
    x_hat[0] = x
    p_hat[0] = p

    r_meas = np.diag(np.asarray(meas_std_vector, dtype=np.float64) ** 2)

    for k in range(1, t_len):
        dt = float(times_s[k] - times_s[k - 1])

        def f_fn(x_in: np.ndarray) -> np.ndarray:
            return rk4_step(
                x_in,
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

        f_jac = numerical_jacobian(f_fn, x, eps=1e-3)
        x = f_fn(x)
        q = build_process_covariance(cfg, dt)
        p = f_jac @ p @ f_jac.T + q

        for s_idx, station in enumerate(stations):
            if visibility[k, s_idx] < 0.5:
                continue
            z = measurements[k, s_idx]

            def h_fn(x_in: np.ndarray) -> np.ndarray:
                return _measurement_model(x_in, station, times_s[k])

            h_jac = numerical_jacobian(h_fn, x, eps=1e-2)
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
                # Soft robustification against severe outliers.
                r_eff = 9.0 * r_base
                s_mat = h_jac @ p @ h_jac.T + r_eff
                try:
                    s_inv = np.linalg.inv(s_mat)
                except np.linalg.LinAlgError:
                    s_inv = np.linalg.pinv(s_mat)
            k_gain = p @ h_jac.T @ s_inv
            x = x + k_gain @ y
            i_kh = np.eye(6) - k_gain @ h_jac
            p = i_kh @ p @ i_kh.T + k_gain @ r_eff @ k_gain.T
            p = 0.5 * (p + p.T)

        x_hat[k] = x
        p_hat[k] = p

    return x_hat, p_hat
