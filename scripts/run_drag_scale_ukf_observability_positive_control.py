#!/usr/bin/env python
"""Observability-supporting constructive positive control for the DSA-UKF (loop 56).

This driver implements the predeclared observability-positive constructive
control whose predeclared rule, scope, and interpretation are recorded in
``release/predeclarations/drag_scale_ukf_observability_positive_control_loop56.json``.

The candidate is the loop 55 Drag-Scale Adaptive UKF (DSA-UKF) construction
unchanged; the filter-side hyperparameters are pinned at the loop 55 selected
operating point. The slice geometry is selected from a small predeclared grid
of observability-supporting configurations (denser station network, longer
arc length, and an extended-arc variant with larger truth-side drag-scale bias
and a lower measurement-noise floor). The grid is iterated in the predeclared
order most-parsimonious-first; the selected grid point is the first grid point
that satisfies the validation-side predeclared decision predicate, or (if no
grid point satisfies it) the grid point with the most-favourable validation
margin. The held-out test seed evaluates only the selected grid point.

The script is non-paper-facing. The paper-facing summary is rendered into the
supplementary table by ``build_paper_assets.py`` from the JSON.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hifi_force_mismatch as base

from gnn_state_estimation.coordinates import StationGeometry
from gnn_state_estimation.dynamics import acceleration_eci, kepler_to_cartesian
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import (
    AdaptiveUKFConfig,
    DMCEKFConfig,
    DragScaleAEKFConfig,
    DragScaleAUKFConfig,
    EKFConfig,
    ProcessNoiseAdaptiveUKFConfig,
    UKFConfig,
    run_adaptive_ukf,
    run_dmc_ekf,
    run_drag_scale_aekf,
    run_drag_scale_aukf,
    run_ekf,
    run_process_noise_adaptive_ukf,
    run_ukf,
)
from gnn_state_estimation.simulation import (
    MeasurementNoiseConfig,
    parse_dataset_config,
)
from gnn_state_estimation.utils.io import load_yaml


# ---------------------------------------------------------------------------
# Twenty-station globally distributed ground network. The roster mirrors the
# loop 33 credible-dense-tracking probe (configs/experiment.yaml splits.
# credible_dense_od_test.overrides.stations) so the slice is fully
# self-contained and the rule predeclaration can pin the network exactly.
# ---------------------------------------------------------------------------
_DENSE_TWENTY_GLOBAL_STATIONS: tuple[dict[str, float | str], ...] = (
    {"name": "California", "lat_deg": 34.74, "lon_deg": -120.57, "alt_m": 120.0},
    {"name": "Florida", "lat_deg": 28.52, "lon_deg": -80.65, "alt_m": 5.0},
    {"name": "Texas", "lat_deg": 30.60, "lon_deg": -97.67, "alt_m": 150.0},
    {"name": "Hawaii", "lat_deg": 19.70, "lon_deg": -155.08, "alt_m": 10.0},
    {"name": "Chile", "lat_deg": -33.15, "lon_deg": -70.67, "alt_m": 570.0},
    {"name": "Brasilia", "lat_deg": -15.79, "lon_deg": -47.88, "alt_m": 1170.0},
    {"name": "Argentina", "lat_deg": -34.60, "lon_deg": -58.38, "alt_m": 25.0},
    {"name": "Peru", "lat_deg": -12.04, "lon_deg": -77.03, "alt_m": 150.0},
    {"name": "Spain", "lat_deg": 40.43, "lon_deg": -4.25, "alt_m": 700.0},
    {"name": "Germany", "lat_deg": 50.00, "lon_deg": 8.50, "alt_m": 110.0},
    {"name": "UK", "lat_deg": 51.48, "lon_deg": -0.10, "alt_m": 25.0},
    {"name": "Azores", "lat_deg": 37.74, "lon_deg": -25.66, "alt_m": 100.0},
    {"name": "SouthAfrica", "lat_deg": -25.89, "lon_deg": 27.69, "alt_m": 1400.0},
    {"name": "Kenya", "lat_deg": -1.29, "lon_deg": 36.82, "alt_m": 1660.0},
    {"name": "India", "lat_deg": 17.41, "lon_deg": 78.55, "alt_m": 540.0},
    {"name": "Japan", "lat_deg": 36.00, "lon_deg": 140.00, "alt_m": 30.0},
    {"name": "Singapore", "lat_deg": 1.35, "lon_deg": 103.82, "alt_m": 15.0},
    {"name": "Australia", "lat_deg": -31.80, "lon_deg": 115.89, "alt_m": 40.0},
    {"name": "NewZealand", "lat_deg": -43.52, "lon_deg": 172.62, "alt_m": 10.0},
    {"name": "Alaska", "lat_deg": 64.84, "lon_deg": -147.72, "alt_m": 140.0},
)


def _build_dense_twenty_stations(min_elevation_deg: float = 8.0) -> tuple[StationGeometry, ...]:
    return tuple(
        StationGeometry(
            name=str(s["name"]),
            lat_deg=float(s["lat_deg"]),
            lon_deg=float(s["lon_deg"]),
            alt_m=float(s["alt_m"]),
            min_elevation_deg=float(min_elevation_deg),
        )
        for s in _DENSE_TWENTY_GLOBAL_STATIONS
    )


def _build_measurement_noise(profile: str, base_noise: MeasurementNoiseConfig) -> MeasurementNoiseConfig:
    if profile == "nominal":
        return base_noise
    if profile == "lower_noise":
        return MeasurementNoiseConfig(
            range_std_m=15.0,
            az_std_deg=0.010,
            el_std_deg=0.010,
            range_rate_std_mps=0.04,
            outlier_prob=base_noise.outlier_prob,
            outlier_scale=base_noise.outlier_scale,
            range_bias_std_m=base_noise.range_bias_std_m,
            az_bias_std_deg=base_noise.az_bias_std_deg,
            el_bias_std_deg=base_noise.el_bias_std_deg,
            range_rate_bias_std_mps=base_noise.range_rate_bias_std_mps,
            clock_bias_std_s=base_noise.clock_bias_std_s,
            clock_jitter_std_s=base_noise.clock_jitter_std_s,
            random_dropout_prob=base_noise.random_dropout_prob,
        )
    raise ValueError(f"Unknown measurement_noise_profile {profile!r}")


# ---------------------------------------------------------------------------
# Compact-dynamics propagator (identical structure to the loop 55 control).
# ---------------------------------------------------------------------------


def _compact_state_derivative(
    state: np.ndarray,
    t_s: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    a = acceleration_eci(
        r_eci=r,
        v_eci=v,
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        t_s=t_s,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
        enable_third_body=False,
        enable_srp=False,
    )
    return np.hstack([v, a]).astype(np.float64)


def _compact_rk4_step(
    state: np.ndarray,
    dt: float,
    t_s: float,
    **kwargs,
) -> np.ndarray:
    k1 = _compact_state_derivative(state, t_s=t_s, **kwargs)
    k2 = _compact_state_derivative(state + 0.5 * dt * k1, t_s=t_s + 0.5 * dt, **kwargs)
    k3 = _compact_state_derivative(state + 0.5 * dt * k2, t_s=t_s + 0.5 * dt, **kwargs)
    k4 = _compact_state_derivative(state + dt * k3, t_s=t_s + dt, **kwargs)
    return (state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float64)


def _propagate_compact_trajectory(
    x0: np.ndarray,
    dt: float,
    steps: int,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
) -> np.ndarray:
    states = np.zeros((steps, 6), dtype=np.float64)
    states[0] = np.asarray(x0, dtype=np.float64)
    for k in range(1, steps):
        states[k] = _compact_rk4_step(
            states[k - 1],
            dt=dt,
            t_s=(k - 1) * dt,
            ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
            drag_rho_ref=drag_rho_ref,
            drag_h_ref_m=drag_h_ref_m,
            drag_scale_height_m=drag_scale_height_m,
        )
    return states


def _sample_orbit(
    altitude_min_km: float,
    altitude_max_km: float,
    eccentricity_min: float,
    eccentricity_max: float,
    inclination_min_deg: float,
    inclination_max_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    earth_radius_m = 6378.1363e3
    altitude_m = rng.uniform(altitude_min_km, altitude_max_km) * 1e3
    semi_major_axis_m = earth_radius_m + altitude_m
    eccentricity = rng.uniform(eccentricity_min, eccentricity_max)
    inclination_rad = np.deg2rad(
        rng.uniform(inclination_min_deg, inclination_max_deg)
    )
    raan_rad = rng.uniform(0.0, 2.0 * np.pi)
    arg_perigee_rad = rng.uniform(0.0, 2.0 * np.pi)
    true_anomaly_rad = rng.uniform(0.0, 2.0 * np.pi)
    return kepler_to_cartesian(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=eccentricity,
        inclination_rad=inclination_rad,
        raan_rad=raan_rad,
        arg_perigee_rad=arg_perigee_rad,
        true_anomaly_rad=true_anomaly_rad,
    )


def _build_population(
    *,
    n_traj: int,
    steps: int,
    dt: float,
    truth_beta: float,
    altitude_min: float,
    altitude_max: float,
    stations: tuple[StationGeometry, ...],
    meas_noise: MeasurementNoiseConfig,
    seed: int,
    dataset_cfg,
    init_pos_sigma: float,
    init_vel_sigma: float,
):
    dyn = dataset_cfg.dynamics
    orbit_cfg = dataset_cfg.orbit_sampling
    meas_std = meas_noise.std_vector
    rng = np.random.default_rng(seed)
    times = np.tile(np.arange(steps, dtype=np.float64) * dt, (n_traj, 1))

    states_all = np.zeros((n_traj, steps, 6), dtype=np.float64)
    meas_all = np.zeros((n_traj, steps, len(stations), 4), dtype=np.float64)
    vis_all = np.zeros((n_traj, steps, len(stations)), dtype=np.float64)
    x0_est_all = np.zeros((n_traj, 6), dtype=np.float64)

    truth_bc = float(dyn.ballistic_coeff_m2_per_kg) * truth_beta
    for i in range(n_traj):
        x0 = _sample_orbit(
            altitude_min_km=altitude_min,
            altitude_max_km=altitude_max,
            eccentricity_min=orbit_cfg.eccentricity_min,
            eccentricity_max=orbit_cfg.eccentricity_max,
            inclination_min_deg=orbit_cfg.inclination_min_deg,
            inclination_max_deg=orbit_cfg.inclination_max_deg,
            rng=rng,
        )
        states = _propagate_compact_trajectory(
            x0,
            dt=dt,
            steps=steps,
            ballistic_coeff_m2_per_kg=truth_bc,
            drag_rho_ref=dyn.drag_rho_ref,
            drag_h_ref_m=dyn.drag_h_ref_m,
            drag_scale_height_m=dyn.drag_scale_height_m,
        )
        meas, vis = base._generate_observations(
            states=states,
            times=times[i],
            stations=stations,
            noise_std=meas_std,
            outlier_prob=meas_noise.outlier_prob,
            outlier_scale=meas_noise.outlier_scale,
            dropout_prob=meas_noise.random_dropout_prob,
            rng=rng,
        )
        x0_perturb = np.array(
            [
                rng.normal(0.0, init_pos_sigma),
                rng.normal(0.0, init_pos_sigma),
                rng.normal(0.0, init_pos_sigma),
                rng.normal(0.0, init_vel_sigma),
                rng.normal(0.0, init_vel_sigma),
                rng.normal(0.0, init_vel_sigma),
            ]
        )
        states_all[i] = states
        meas_all[i] = meas
        vis_all[i] = vis
        x0_est_all[i] = states[0] + x0_perturb
    return states_all, meas_all, vis_all, x0_est_all, times


def _drag_scale_separation_diagnostic(
    states_all: np.ndarray,
    *,
    dt: float,
    dyn,
) -> dict[str, float]:
    """Mean position separation (m) between the truth-side beta-scaled flow
    already in ``states_all`` and the nominal beta=1.0 flow integrated from the
    same initial state. This is an upfront observability diagnostic that does
    not depend on any estimator: it is the raw drag-scale signal available to
    the line-of-sight measurements over the slice.
    """
    steps = states_all.shape[1]
    seps_final = []
    seps_mean = []
    for i in range(states_all.shape[0]):
        x0 = states_all[i, 0]
        nominal = _propagate_compact_trajectory(
            x0,
            dt=dt,
            steps=steps,
            ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg),
            drag_rho_ref=dyn.drag_rho_ref,
            drag_h_ref_m=dyn.drag_h_ref_m,
            drag_scale_height_m=dyn.drag_scale_height_m,
        )
        diff = states_all[i, :, :3] - nominal[:, :3]
        norm = np.linalg.norm(diff, axis=-1)
        seps_final.append(float(norm[-1]))
        seps_mean.append(float(np.mean(norm)))
    return {
        "median_final_separation_m": float(np.median(seps_final)),
        "median_mean_separation_m": float(np.median(seps_mean)),
    }


# ---------------------------------------------------------------------------
# Filter execution for the seven-estimator comparator set.
# ---------------------------------------------------------------------------


def _build_filter_configs(baseline_cfg, dyn, th_pukf, th_dmc, th_dsa, dsa_hyperparams):
    ekf_cfg = EKFConfig(
        q_pos_m=baseline_cfg.ekf.q_pos_m,
        q_vel_mps=baseline_cfg.ekf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
        gating_threshold=baseline_cfg.ekf.gating_threshold,
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ekf, "angle_deweight_elev_cap_deg", None
        ),
    )
    ukf_cfg = UKFConfig(
        q_pos_m=baseline_cfg.ukf.q_pos_m,
        q_vel_mps=baseline_cfg.ukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
        ),
    )
    aukf_cfg = AdaptiveUKFConfig(
        q_pos_m=baseline_cfg.aukf.q_pos_m,
        q_vel_mps=baseline_cfg.aukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.aukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.aukf.init_vel_std_mps,
        alpha=baseline_cfg.aukf.alpha,
        beta=baseline_cfg.aukf.beta,
        kappa=baseline_cfg.aukf.kappa,
        adapt_rate=baseline_cfg.aukf.adapt_rate,
        min_r_scale=baseline_cfg.aukf.min_r_scale,
        max_r_scale=baseline_cfg.aukf.max_r_scale,
        huber_kappa=baseline_cfg.aukf.huber_kappa,
        nis_soft_gate=baseline_cfg.aukf.nis_soft_gate,
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.aukf, "angle_deweight_elev_cap_deg", None
        ),
    )
    pukf_cfg = ProcessNoiseAdaptiveUKFConfig(
        q_pos_m=baseline_cfg.ukf.q_pos_m,
        q_vel_mps=baseline_cfg.ukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        window_size=int(th_pukf["window_size"]),
        nis_per_update_expected=float(th_pukf["nis_per_update_expected"]),
        nis_warn_ratio=float(th_pukf["nis_warn_ratio"]),
        nis_alarm_ratio=float(th_pukf["nis_alarm_ratio"]),
        q_scale_warn=float(th_pukf["q_scale_warn"]),
        q_scale_alarm=float(th_pukf["q_scale_alarm"]),
        q_scale_max=float(th_pukf["q_scale_max"]),
        smoothing=float(th_pukf["smoothing"]),
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
        ),
    )
    dmc_cfg = DMCEKFConfig(
        q_pos_m=baseline_cfg.ekf.q_pos_m,
        q_vel_mps=baseline_cfg.ekf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
        init_emp_accel_std_mps2=float(th_dmc["init_emp_accel_std_mps2"]),
        emp_accel_sigma_mps2=float(th_dmc["emp_accel_sigma_mps2"]),
        emp_accel_tau_s=float(th_dmc["emp_accel_tau_s"]),
        gating_threshold=float(th_dmc["gating_threshold"]),
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ekf, "angle_deweight_elev_cap_deg", None
        ),
    )
    dsa_ekf_cfg = DragScaleAEKFConfig(
        q_pos_m=float(th_dsa["q_pos_m"]),
        q_vel_mps=float(th_dsa["q_vel_mps"]),
        init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
        init_drag_scale_std=float(th_dsa["default_init_drag_scale_std"]),
        drag_scale_sigma_ss=float(th_dsa["default_drag_scale_sigma_ss"]),
        drag_scale_tau_s=float(th_dsa["default_drag_scale_tau_s"]),
        gating_threshold=float(th_dsa["gating_threshold"]),
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ekf, "angle_deweight_elev_cap_deg", None
        ),
    )
    dsa_ukf_cfg = DragScaleAUKFConfig(
        q_pos_m=baseline_cfg.ukf.q_pos_m,
        q_vel_mps=baseline_cfg.ukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
        init_drag_scale_std=float(dsa_hyperparams["init_drag_scale_std"]),
        drag_scale_sigma_ss=float(dsa_hyperparams["drag_scale_sigma_ss"]),
        drag_scale_tau_s=float(dsa_hyperparams["drag_scale_tau_s"]),
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
        ),
    )
    return {
        "EKF": ekf_cfg,
        "UKF": ukf_cfg,
        "AUKF": aukf_cfg,
        "PUKF": pukf_cfg,
        "DMC_EKF": dmc_cfg,
        "DSA_EKF": dsa_ekf_cfg,
        "DSA_UKF": dsa_ukf_cfg,
    }


def _run_all_filters(
    *,
    states_all: np.ndarray,
    meas_all: np.ndarray,
    vis_all: np.ndarray,
    x0_est_all: np.ndarray,
    times: np.ndarray,
    stations: tuple[StationGeometry, ...],
    meas_noise: MeasurementNoiseConfig,
    dyn,
    cfgs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_traj = states_all.shape[0]
    base_kwargs = dict(
        ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg),
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
        enable_third_body=False,
        enable_srp=False,
        srp_area_to_mass_m2_per_kg=dyn.srp_area_to_mass_m2_per_kg,
        srp_cr=dyn.srp_cr,
        sun_initial_phase_rad=dyn.sun_initial_phase_rad,
        moon_initial_phase_rad=dyn.moon_initial_phase_rad,
    )
    meas_std = meas_noise.std_vector
    preds = {name: np.zeros_like(states_all) for name in cfgs}
    dsa_ekf_beta_final: list[float] = []
    dsa_ukf_beta_mean: list[float] = []
    dsa_ukf_beta_max: list[float] = []
    dsa_ukf_beta_final: list[float] = []
    for i in range(n_traj):
        preds["EKF"][i], _ = run_ekf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=cfgs["EKF"], **base_kwargs,
        )
        preds["UKF"][i], _ = run_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=cfgs["UKF"], **base_kwargs,
        )
        preds["AUKF"][i], _ = run_adaptive_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=cfgs["AUKF"], **base_kwargs,
        )
        preds["PUKF"][i], _, _ = run_process_noise_adaptive_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=cfgs["PUKF"], **base_kwargs,
        )
        preds["DMC_EKF"][i], _, _ = run_dmc_ekf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=cfgs["DMC_EKF"], **base_kwargs,
        )
        preds["DSA_EKF"][i], _, dsa_ekf_diag = run_drag_scale_aekf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=cfgs["DSA_EKF"], **base_kwargs,
        )
        bh_ekf = dsa_ekf_diag.get("drag_scale_history")
        if bh_ekf is not None and bh_ekf.size:
            dsa_ekf_beta_final.append(float(bh_ekf[-1]))
        preds["DSA_UKF"][i], _, dsa_ukf_diag = run_drag_scale_aukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=cfgs["DSA_UKF"], **base_kwargs,
        )
        bh = dsa_ukf_diag.get("drag_scale_history")
        if bh is not None and bh.size:
            dsa_ukf_beta_max.append(float(np.max(bh)))
            dsa_ukf_beta_mean.append(float(np.mean(bh)))
            dsa_ukf_beta_final.append(float(bh[-1]))
    diagnostics = {
        "dsa_ekf_median_final_beta": (
            float(np.median(dsa_ekf_beta_final)) if dsa_ekf_beta_final else float("nan")
        ),
        "dsa_ukf_median_mean_beta": (
            float(np.median(dsa_ukf_beta_mean)) if dsa_ukf_beta_mean else float("nan")
        ),
        "dsa_ukf_median_max_abs_beta_deviation": (
            float(np.median([abs(b - 1.0) for b in dsa_ukf_beta_max]))
            if dsa_ukf_beta_max else float("nan")
        ),
        "dsa_ukf_median_final_beta": (
            float(np.median(dsa_ukf_beta_final)) if dsa_ukf_beta_final else float("nan")
        ),
    }
    return preds, diagnostics


def _evaluate_predicate(
    preds: dict[str, np.ndarray],
    states_all: np.ndarray,
    vis_all: np.ndarray,
    eval_start: int,
    bootstrap_samples: int,
    seed: int,
    floor_fraction: float,
) -> dict[str, Any]:
    metrics = {
        name: base._per_traj_observed_pos_rmse(
            states_all, arr, vis_all, eval_start
        )
        for name, arr in preds.items()
    }
    means = {k: float(np.nanmean(v)) for k, v in metrics.items()}
    non_candidate_names = ["EKF", "UKF", "AUKF", "PUKF", "DMC_EKF", "DSA_EKF"]
    best_non_candidate = min(non_candidate_names, key=lambda k: means[k])
    best_non_candidate_mean = means[best_non_candidate]
    floor_abs = floor_fraction * best_non_candidate_mean
    gap_diffs = metrics["DSA_UKF"] - metrics[best_non_candidate]
    gap_mean, gap_lo, gap_hi = base._paired_bootstrap_ci(
        gap_diffs, n_boot=bootstrap_samples, seed=seed
    )
    candidate_is_lowest = means["DSA_UKF"] == min(means.values())
    ci_strictly_negative = bool(np.isfinite(gap_hi) and gap_hi < 0.0)
    floor_exceeded = bool(
        np.isfinite(gap_mean) and gap_mean < 0.0 and -gap_mean > floor_abs
    )
    is_positive = bool(candidate_is_lowest and ci_strictly_negative and floor_exceeded)
    paired = {}
    for name in non_candidate_names:
        diffs = metrics["DSA_UKF"] - metrics[name]
        m, lo, hi = base._paired_bootstrap_ci(
            diffs, n_boot=bootstrap_samples, seed=seed + 1
        )
        p = base._one_sided_wilcoxon_candidate_better(diffs)
        finite = diffs[np.isfinite(diffs)]
        paired[f"DSA_UKF_minus_{name}"] = {
            "candidate": "DSA_UKF",
            "baseline": name,
            "mean_diff_m": m,
            "ci_lo_m": lo,
            "ci_hi_m": hi,
            "wilcoxon_p_one_sided_candidate_better": p,
            "n_paired": int(finite.size),
            "candidate_better_count": int(np.sum(finite < 0.0)),
        }
    return {
        "means_m": means,
        "paired": paired,
        "best_non_candidate_estimator": best_non_candidate,
        "best_non_candidate_mean_m": best_non_candidate_mean,
        "practical_significance_floor_abs_m": floor_abs,
        "dsa_ukf_minus_best_non_candidate_mean_m": gap_mean,
        "dsa_ukf_minus_best_non_candidate_ci_lo_m": gap_lo,
        "dsa_ukf_minus_best_non_candidate_ci_hi_m": gap_hi,
        "dsa_ukf_is_strictly_lowest_mean": bool(candidate_is_lowest),
        "ci_strictly_negative_for_dsa_ukf": ci_strictly_negative,
        "floor_exceeded": floor_exceeded,
        "predeclared_positive_criterion_met": is_positive,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--predeclared-rule",
        default=(
            "release/predeclarations/"
            "drag_scale_ukf_observability_positive_control_loop56.json"
        ),
    )
    p.add_argument(
        "--pukf-predeclared-rule",
        default="release/predeclarations/pukf_q_adaptive_rule_loop41.json",
    )
    p.add_argument(
        "--dmc-predeclared-rule",
        default="release/predeclarations/dmc_ekf_rule_loop44.json",
    )
    p.add_argument(
        "--dsa-predeclared-rule",
        default="release/predeclarations/drag_scale_aekf_rule_loop45.json",
    )
    p.add_argument(
        "--output-json",
        default=(
            "results/drag_scale_ukf_observability_positive_control/"
            "drag_scale_ukf_observability_positive_control.json"
        ),
    )
    p.add_argument(
        "--output-csv",
        default=(
            "results/drag_scale_ukf_observability_positive_control/"
            "drag_scale_ukf_observability_positive_control.csv"
        ),
    )
    p.add_argument(
        "--validation-output-json",
        default=(
            "results/drag_scale_ukf_observability_positive_control/"
            "drag_scale_ukf_observability_validation.json"
        ),
    )
    return p


def _stations_for_label(label: str, min_elevation_deg: float = 8.0) -> tuple[StationGeometry, ...]:
    if label == "dense_twenty_global":
        return _build_dense_twenty_stations(min_elevation_deg=min_elevation_deg)
    raise ValueError(f"Unknown station_network_label {label!r}")


def _meas_for_profile(profile: str, base_noise: MeasurementNoiseConfig) -> MeasurementNoiseConfig:
    return _build_measurement_noise(profile, base_noise)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    base_noise = dataset_cfg.measurement_noise

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    rule = json.loads(Path(args.predeclared_rule).read_text())
    pukf_rule = json.loads(Path(args.pukf_predeclared_rule).read_text())
    dmc_rule = json.loads(Path(args.dmc_predeclared_rule).read_text())
    dsa_rule = json.loads(Path(args.dsa_predeclared_rule).read_text())
    th_pukf = pukf_rule["thresholds"]
    th_dmc = dmc_rule["thresholds"]
    th_dsa = dsa_rule["thresholds"]

    dsa_hyperparams = rule["candidate"]["hyperparameters_pinned_from_loop55"]
    cfgs = _build_filter_configs(
        baseline_cfg, dyn, th_pukf, th_dmc, th_dsa, dsa_hyperparams
    )
    floor_fraction = float(
        rule["decision_predicate"]["practical_significance_floor_fraction"]
    )
    bootstrap_samples_val = int(
        rule["validation_grid"]["bootstrap_samples_validation"]
    )
    bootstrap_samples_test = int(
        rule["decision_predicate"]["bootstrap_samples"]
    )
    val_seed = int(rule["validation_grid"]["validation_seed"])
    test_seed = int(rule["validation_grid"]["test_seed"])
    n_val = int(rule["validation_grid"]["validation_n_trajectories"])
    n_test = int(rule["validation_grid"]["test_n_trajectories"])
    eval_start = int(rule["validation_grid"]["eval_start_step"])
    epoch_unix = float(rule["validation_grid"]["epoch_unix"])
    init_pos_sigma = baseline_cfg.ukf.init_pos_std_m
    init_vel_sigma = baseline_cfg.ukf.init_vel_std_mps

    # ---- Validation grid sweep on the disjoint validation seed ----
    validation_records: list[dict[str, Any]] = []
    val_total_t = time.perf_counter()
    for grid_idx, grid_point in enumerate(rule["validation_grid"]["grid_points_ordered"]):
        label = grid_point["label"]
        steps = int(grid_point["arc_length_steps"])
        dt = float(grid_point["propagator_step_s"])
        truth_beta = float(grid_point["truth_beta_value"])
        altitude_min = float(grid_point["orbit_altitude_min_km"])
        altitude_max = float(grid_point["orbit_altitude_max_km"])
        stations = _stations_for_label(grid_point["station_network_label"])
        meas_noise = _meas_for_profile(grid_point["measurement_noise_profile"], base_noise)

        # Each grid point uses a unique seed offset to keep the populations
        # disjoint across grid points while still being deterministic.
        grid_val_seed = val_seed + grid_idx * 1000
        t_gen = time.perf_counter()
        val_states, val_meas, val_vis, val_x0_est, val_times = _build_population(
            n_traj=n_val,
            steps=steps,
            dt=dt,
            truth_beta=truth_beta,
            altitude_min=altitude_min,
            altitude_max=altitude_max,
            stations=stations,
            meas_noise=meas_noise,
            seed=grid_val_seed,
            dataset_cfg=dataset_cfg,
            init_pos_sigma=init_pos_sigma,
            init_vel_sigma=init_vel_sigma,
        )
        t_gen = time.perf_counter() - t_gen
        sep_diag = _drag_scale_separation_diagnostic(val_states, dt=dt, dyn=dyn)
        visible_frac = float(
            np.mean(np.sum(val_vis[:, eval_start:], axis=-1) >= 0.5)
        )

        t_filt = time.perf_counter()
        val_preds, val_filt_diag = _run_all_filters(
            states_all=val_states,
            meas_all=val_meas,
            vis_all=val_vis,
            x0_est_all=val_x0_est,
            times=val_times,
            stations=stations,
            meas_noise=meas_noise,
            dyn=dyn,
            cfgs=cfgs,
        )
        t_filt = time.perf_counter() - t_filt
        decision = _evaluate_predicate(
            val_preds, val_states, val_vis, eval_start,
            bootstrap_samples=bootstrap_samples_val,
            seed=grid_val_seed,
            floor_fraction=floor_fraction,
        )
        validation_records.append({
            "label": label,
            "grid_index": grid_idx,
            "arc_length_steps": steps,
            "arc_length_hours_approximate": float(grid_point["arc_length_hours_approximate"]),
            "propagator_step_s": dt,
            "orbit_altitude_min_km": altitude_min,
            "orbit_altitude_max_km": altitude_max,
            "truth_beta_value": truth_beta,
            "station_network_label": grid_point["station_network_label"],
            "n_stations": len(stations),
            "measurement_noise_profile": grid_point["measurement_noise_profile"],
            "n_trajectories": n_val,
            "validation_seed": grid_val_seed,
            "visible_step_fraction_eval": visible_frac,
            "drag_scale_separation_diagnostic_m": sep_diag,
            "decision": decision,
            "dsa_diagnostics": val_filt_diag,
            "elapsed_seconds": {
                "truth_generation": float(t_gen),
                "filters": float(t_filt),
            },
        })
    val_total_t = time.perf_counter() - val_total_t

    # ---- Selection rule ----
    selected_idx = None
    selection_rationale = None
    for rec in validation_records:
        if rec["decision"]["predeclared_positive_criterion_met"]:
            selected_idx = rec["grid_index"]
            selection_rationale = (
                f"Grid point {rec['label']} (index {rec['grid_index']}) "
                f"is the first grid point in the predeclared order that "
                f"satisfies the validation-side decision predicate."
            )
            break
    if selected_idx is None:
        # Pick the grid point with the most-favourable validation margin
        # (smallest mean DSA-UKF minus best non-candidate gap).
        margins = [
            rec["decision"]["dsa_ukf_minus_best_non_candidate_mean_m"]
            for rec in validation_records
        ]
        finite_indices = [i for i, m in enumerate(margins) if np.isfinite(m)]
        if not finite_indices:
            selected_idx = 0
            selection_rationale = (
                "No grid point produced a finite validation margin; "
                "selecting the first predeclared grid point so the held-out "
                "test still reports a deterministic outcome."
            )
        else:
            selected_idx = min(finite_indices, key=lambda i: margins[i])
            selection_rationale = (
                "No grid point satisfied the validation-side predicate; "
                f"selecting the grid point {validation_records[selected_idx]['label']} "
                f"with the most-favourable validation margin "
                f"({margins[selected_idx]:.3f} m, lower is more favourable to the candidate)."
            )

    selected_grid_point = rule["validation_grid"]["grid_points_ordered"][selected_idx]
    selected_record = validation_records[selected_idx]

    val_out_path = Path(args.validation_output_json)
    val_out_path.parent.mkdir(parents=True, exist_ok=True)
    val_payload: dict[str, Any] = {
        "scenario": "drag_scale_ukf_observability_validation",
        "schema_version": "drag_scale_ukf_observability_validation_v1",
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "validation_seed_base": val_seed,
        "n_trajectories": n_val,
        "bootstrap_samples": bootstrap_samples_val,
        "validation_records": validation_records,
        "selected_grid_index": selected_idx,
        "selected_grid_label": selected_grid_point["label"],
        "selection_rationale": selection_rationale,
        "elapsed_seconds_total_validation": float(val_total_t),
    }
    val_out_path.write_text(json.dumps(val_payload, indent=2))

    # ---- Held-out test on the disjoint test seed for the selected grid point ----
    steps = int(selected_grid_point["arc_length_steps"])
    dt = float(selected_grid_point["propagator_step_s"])
    truth_beta = float(selected_grid_point["truth_beta_value"])
    altitude_min = float(selected_grid_point["orbit_altitude_min_km"])
    altitude_max = float(selected_grid_point["orbit_altitude_max_km"])
    stations = _stations_for_label(selected_grid_point["station_network_label"])
    meas_noise = _meas_for_profile(selected_grid_point["measurement_noise_profile"], base_noise)

    t_test_gen = time.perf_counter()
    test_states, test_meas, test_vis, test_x0_est, test_times = _build_population(
        n_traj=n_test,
        steps=steps,
        dt=dt,
        truth_beta=truth_beta,
        altitude_min=altitude_min,
        altitude_max=altitude_max,
        stations=stations,
        meas_noise=meas_noise,
        seed=test_seed,
        dataset_cfg=dataset_cfg,
        init_pos_sigma=init_pos_sigma,
        init_vel_sigma=init_vel_sigma,
    )
    t_test_gen = time.perf_counter() - t_test_gen
    test_sep_diag = _drag_scale_separation_diagnostic(test_states, dt=dt, dyn=dyn)
    test_visible_frac = float(
        np.mean(np.sum(test_vis[:, eval_start:], axis=-1) >= 0.5)
    )

    t_test_filt = time.perf_counter()
    test_preds, test_filt_diag = _run_all_filters(
        states_all=test_states,
        meas_all=test_meas,
        vis_all=test_vis,
        x0_est_all=test_x0_est,
        times=test_times,
        stations=stations,
        meas_noise=meas_noise,
        dyn=dyn,
        cfgs=cfgs,
    )
    t_test_filt = time.perf_counter() - t_test_filt
    test_decision = _evaluate_predicate(
        test_preds, test_states, test_vis, eval_start,
        bootstrap_samples=bootstrap_samples_test,
        seed=test_seed,
        floor_fraction=floor_fraction,
    )

    # Cross-filter R-only NIS at the test population for the diagnostic
    # alongside the mean RMSE.
    nis_per_filter: dict[str, dict[str, float]] = {}
    for name, preds in test_preds.items():
        nis_per_filter[name] = base._cross_filter_r_only_nis(
            test_states, preds, test_meas, test_vis, test_times, stations,
            meas_noise.std_vector, eval_start
        )

    metrics = {
        name: base._per_traj_observed_pos_rmse(
            test_states, arr, test_vis, eval_start
        )
        for name, arr in test_preds.items()
    }

    payload: dict[str, Any] = {
        "scenario": "drag_scale_ukf_observability_positive_control",
        "schema_version": "drag_scale_ukf_observability_positive_control_v1",
        "scope": (
            "Observability-supporting constructive positive control for the "
            "loop 55 Drag-Scale Adaptive UKF. The candidate's filter-side "
            "hyperparameters are pinned at the loop 55 selected operating "
            "point. The slice geometry is the grid point selected by the "
            "predeclared validation-side selection rule. The held-out test "
            "seed is disjoint from the validation seed and from every other "
            "test seed used in this manuscript. The loop 54 EKF-based and "
            "loop 55 UKF-based constructive-control outcomes are preserved "
            "unchanged."
        ),
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "selected_grid_index": selected_idx,
        "selected_grid_point": selected_grid_point,
        "selection_rationale": selection_rationale,
        "validation_artifact_source": str(val_out_path),
        "validation_artifact_sha256": hashlib.sha256(
            val_out_path.read_bytes()
        ).hexdigest(),
        "n_trajectories": int(n_test),
        "validation_n_trajectories": int(n_val),
        "steps": int(steps),
        "dt_s": float(dt),
        "eval_start_step": int(eval_start),
        "epoch_unix": float(epoch_unix),
        "test_seed": int(test_seed),
        "validation_seed_base": int(val_seed),
        "truth_beta_value": float(truth_beta),
        "orbit_altitude_min_km": float(altitude_min),
        "orbit_altitude_max_km": float(altitude_max),
        "n_stations": int(len(stations)),
        "station_network_label": selected_grid_point["station_network_label"],
        "measurement_noise_profile": selected_grid_point["measurement_noise_profile"],
        "bootstrap_samples": int(bootstrap_samples_test),
        "visible_step_fraction_eval": test_visible_frac,
        "drag_scale_separation_diagnostic_m": test_sep_diag,
        "truth_acceleration": (
            "compact two-body + J2 + exponential drag with truth-side "
            "ballistic coefficient scaled by truth_beta"
        ),
        "estimator_acceleration": (
            "compact two-body + J2 + exponential drag at nominal ballistic "
            "coefficient (DSA-EKF and DSA-UKF estimate beta online via their "
            "respective recursions)"
        ),
        "observed_step_rmse_mean_m": test_decision["means_m"],
        "paired": test_decision["paired"],
        "decision": {
            "best_non_candidate_estimator": test_decision["best_non_candidate_estimator"],
            "best_non_candidate_mean_m": test_decision["best_non_candidate_mean_m"],
            "practical_significance_floor_fraction": floor_fraction,
            "practical_significance_floor_abs_m": test_decision["practical_significance_floor_abs_m"],
            "dsa_ukf_minus_best_non_candidate_mean_m": test_decision["dsa_ukf_minus_best_non_candidate_mean_m"],
            "dsa_ukf_minus_best_non_candidate_ci_lo_m": test_decision["dsa_ukf_minus_best_non_candidate_ci_lo_m"],
            "dsa_ukf_minus_best_non_candidate_ci_hi_m": test_decision["dsa_ukf_minus_best_non_candidate_ci_hi_m"],
            "dsa_ukf_is_strictly_lowest_mean": test_decision["dsa_ukf_is_strictly_lowest_mean"],
            "ci_strictly_negative_for_dsa_ukf": test_decision["ci_strictly_negative_for_dsa_ukf"],
            "floor_exceeded": test_decision["floor_exceeded"],
            "predeclared_positive_criterion_met": test_decision["predeclared_positive_criterion_met"],
        },
        "dsa_ukf_diagnostics": {
            "median_mean_beta": test_filt_diag["dsa_ukf_median_mean_beta"],
            "median_max_abs_beta_deviation": test_filt_diag["dsa_ukf_median_max_abs_beta_deviation"],
            "median_final_beta": test_filt_diag["dsa_ukf_median_final_beta"],
            "truth_beta_target": float(truth_beta),
        },
        "dsa_ekf_diagnostics": {
            "median_final_beta": test_filt_diag["dsa_ekf_median_final_beta"],
            "truth_beta_target": float(truth_beta),
        },
        "cross_filter_r_only_nis": nis_per_filter,
        "elapsed_seconds": {
            "validation_total": float(val_total_t),
            "truth_generation_test": float(t_test_gen),
            "filters_test": float(t_test_filt),
        },
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    rows = []
    for i in range(n_test):
        row: dict[str, Any] = {"trajectory_index": i}
        for k, arr in metrics.items():
            val = float(arr[i]) if np.isfinite(arr[i]) else None
            row[f"{k}_observed_pos_rmse_m"] = val
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
