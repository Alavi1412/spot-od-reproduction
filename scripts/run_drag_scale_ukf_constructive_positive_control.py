#!/usr/bin/env python
"""Constructive positive control for the Drag-Scale Adaptive UKF (loop 55).

This driver implements the predeclared additional targeted constructive positive
control whose predeclared rule, scope, and interpretation are recorded in
``release/predeclarations/drag_scale_ukf_constructive_positive_control_loop55.json``.

The slice fixes the unmodelled physics to be a pure multiplicative drag-scale
mismatch (no extra zonal geopotential, no luni-solar third body, no SRP, no
time-varying density modulation): the truth-side ballistic coefficient is
``truth_beta_value`` times the nominal value used by every estimator. The
DSA-UKF estimates the multiplicative scaling factor online via deterministic
sigma-point propagation of the augmented seven-dimensional flow. EKF / UKF /
AUKF / PUKF / DMC-EKF / DSA-EKF retain their loop 54 configuration so the
loop 54 constructive-control outcome is preserved unchanged when the
``--include-loop54-candidates`` flag is set.

The script is non-paper-facing. The paper-facing summary is rendered into
the supplementary table by ``build_paper_assets.py`` from the JSON.
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
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.utils.io import load_yaml


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


def _sample_low_altitude_orbit(
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
    seed: int,
    dataset_cfg,
    init_pos_sigma: float,
    init_vel_sigma: float,
):
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector
    orbit_cfg = dataset_cfg.orbit_sampling
    rng = np.random.default_rng(seed)
    times = np.tile(np.arange(steps, dtype=np.float64) * dt, (n_traj, 1))

    states_all = np.zeros((n_traj, steps, 6), dtype=np.float64)
    meas_all = np.zeros((n_traj, steps, len(stations), 4), dtype=np.float64)
    vis_all = np.zeros((n_traj, steps, len(stations)), dtype=np.float64)
    x0_est_all = np.zeros((n_traj, 6), dtype=np.float64)

    truth_bc = float(dyn.ballistic_coeff_m2_per_kg) * truth_beta
    for i in range(n_traj):
        x0 = _sample_low_altitude_orbit(
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
            outlier_prob=dataset_cfg.measurement_noise.outlier_prob,
            outlier_scale=dataset_cfg.measurement_noise.outlier_scale,
            dropout_prob=dataset_cfg.measurement_noise.random_dropout_prob,
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


def _run_dsa_ukf_population(
    *,
    states_all: np.ndarray,
    meas_all: np.ndarray,
    vis_all: np.ndarray,
    x0_est_all: np.ndarray,
    times: np.ndarray,
    cfg: DragScaleAUKFConfig,
    dataset_cfg,
    baseline_cfg,
    eval_start: int,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], list[float]]:
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector
    n_traj = states_all.shape[0]
    preds = np.zeros_like(states_all)
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
    beta_max: list[float] = []
    beta_mean: list[float] = []
    beta_final: list[float] = []
    for i in range(n_traj):
        preds[i], _, diag = run_drag_scale_aukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=cfg,
            **base_kwargs,
        )
        bh = diag.get("drag_scale_history")
        if bh is not None and bh.size:
            beta_max.append(float(np.max(bh)))
            beta_mean.append(float(np.mean(bh)))
            beta_final.append(float(bh[-1]))
    rmse = base._per_traj_observed_pos_rmse(states_all, preds, vis_all, eval_start)
    return preds, rmse, beta_max, beta_mean, beta_final


def _validation_sweep(
    *,
    grid: list[dict[str, float]],
    states_all: np.ndarray,
    meas_all: np.ndarray,
    vis_all: np.ndarray,
    x0_est_all: np.ndarray,
    times: np.ndarray,
    dataset_cfg,
    baseline_cfg,
    eval_start: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, point in grid:
        cfg = DragScaleAUKFConfig(
            q_pos_m=baseline_cfg.ukf.q_pos_m,
            q_vel_mps=baseline_cfg.ukf.q_vel_mps,
            init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
            init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
            init_drag_scale_std=float(point["init_drag_scale_std"]),
            drag_scale_sigma_ss=float(point["drag_scale_sigma_ss"]),
            drag_scale_tau_s=float(point["drag_scale_tau_s"]),
            alpha=baseline_cfg.ukf.alpha,
            beta=baseline_cfg.ukf.beta,
            kappa=baseline_cfg.ukf.kappa,
            angle_deweight_elev_cap_deg=getattr(
                baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
            ),
        )
        _, rmse, beta_max, beta_mean, beta_final = _run_dsa_ukf_population(
            states_all=states_all,
            meas_all=meas_all,
            vis_all=vis_all,
            x0_est_all=x0_est_all,
            times=times,
            cfg=cfg,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            eval_start=eval_start,
        )
        records.append(
            {
                "label": label,
                "init_drag_scale_std": float(point["init_drag_scale_std"]),
                "drag_scale_sigma_ss": float(point["drag_scale_sigma_ss"]),
                "drag_scale_tau_s": float(point["drag_scale_tau_s"]),
                "observed_step_pos_rmse_mean_m": float(np.nanmean(rmse)),
                "observed_step_pos_rmse_median_m": float(np.nanmedian(rmse)),
                "observed_step_pos_rmse_max_m": float(np.nanmax(rmse)),
                "median_max_abs_beta_deviation": (
                    float(np.median([abs(b - 1.0) for b in beta_max]))
                    if beta_max
                    else float("nan")
                ),
                "median_mean_beta": (
                    float(np.median(beta_mean)) if beta_mean else float("nan")
                ),
                "median_final_beta": (
                    float(np.median(beta_final)) if beta_final else float("nan")
                ),
            }
        )
    return records


def _select_grid_point(
    records: list[dict[str, Any]], tie_tol_m: float = 1.0
) -> dict[str, Any]:
    if not records:
        raise RuntimeError("Empty validation grid; cannot select a hyperparameter point.")
    ordered = sorted(
        records,
        key=lambda r: (
            float(r["observed_step_pos_rmse_mean_m"]),
            float(r["init_drag_scale_std"]),
            float(r["drag_scale_sigma_ss"]),
            float(r["drag_scale_tau_s"]),
        ),
    )
    best_mean = float(ordered[0]["observed_step_pos_rmse_mean_m"])
    tied = [r for r in ordered if float(r["observed_step_pos_rmse_mean_m"]) - best_mean <= tie_tol_m]
    tied.sort(
        key=lambda r: (
            float(r["init_drag_scale_std"]),
            float(r["drag_scale_sigma_ss"]),
            float(r["drag_scale_tau_s"]),
        )
    )
    return tied[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--predeclared-rule",
        default=(
            "release/predeclarations/"
            "drag_scale_ukf_constructive_positive_control_loop55.json"
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
        "--loop54-selection",
        default=(
            "results/drag_scale_aekf_validation_long_arc/"
            "drag_scale_aekf_validation_long_arc.json"
        ),
        help="Loop 54 DSA-EKF validation selection (long arc), used for the DSA-EKF candidate only.",
    )
    p.add_argument(
        "--loop54-selection-fallback",
        default=(
            "results/drag_scale_aekf_validation/"
            "drag_scale_aekf_validation.json"
        ),
    )
    p.add_argument(
        "--output-json",
        default=(
            "results/drag_scale_ukf_constructive_positive_control/"
            "drag_scale_ukf_constructive_positive_control.json"
        ),
    )
    p.add_argument(
        "--output-csv",
        default=(
            "results/drag_scale_ukf_constructive_positive_control/"
            "drag_scale_ukf_constructive_positive_control.csv"
        ),
    )
    p.add_argument(
        "--validation-output-json",
        default=(
            "results/drag_scale_ukf_constructive_positive_control/"
            "drag_scale_ukf_validation.json"
        ),
    )
    return p


def _loop54_dsa_selection(
    selection_path: Path,
    fallback_path: Path,
    dsa_rule: dict[str, Any],
) -> dict[str, Any]:
    target = None
    if selection_path.is_file():
        target = selection_path
    elif fallback_path.is_file():
        target = fallback_path
    if target is not None:
        sel = json.loads(target.read_text())
        chosen = sel.get("selected_grid_point")
        if chosen is not None:
            return {
                "selected_label": chosen.get("label"),
                "init_drag_scale_std": float(chosen["init_drag_scale_std"]),
                "drag_scale_sigma_ss": float(chosen["drag_scale_sigma_ss"]),
                "drag_scale_tau_s": float(chosen["drag_scale_tau_s"]),
                "validation_artifact_source": str(target),
                "validation_artifact_sha256": hashlib.sha256(
                    target.read_bytes()
                ).hexdigest(),
            }
    th = dsa_rule["thresholds"]
    return {
        "selected_label": "default-from-rule",
        "init_drag_scale_std": float(th["default_init_drag_scale_std"]),
        "drag_scale_sigma_ss": float(th["default_drag_scale_sigma_ss"]),
        "drag_scale_tau_s": float(th["default_drag_scale_tau_s"]),
        "validation_artifact_source": None,
        "validation_artifact_sha256": None,
    }


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    rule = json.loads(Path(args.predeclared_rule).read_text())
    pukf_rule = json.loads(Path(args.pukf_predeclared_rule).read_text())
    dmc_rule = json.loads(Path(args.dmc_predeclared_rule).read_text())
    dsa_rule = json.loads(Path(args.dsa_predeclared_rule).read_text())
    th_pukf = pukf_rule["thresholds"]
    th_dmc = dmc_rule["thresholds"]
    th_dsa = dsa_rule["thresholds"]

    design = rule["constructive_control_design"]
    sel_cfg = rule["hyperparameter_selection"]
    grid_cfg = sel_cfg["grid_predeclared"]
    grid: list[tuple[str, dict[str, float]]] = []
    counter = 0
    for init_std in grid_cfg["init_drag_scale_std"]:
        for sigma_ss in grid_cfg["drag_scale_sigma_ss"]:
            for tau_s in grid_cfg["drag_scale_tau_s"]:
                counter += 1
                grid.append(
                    (
                        f"G{counter:02d}",
                        {
                            "init_drag_scale_std": float(init_std),
                            "drag_scale_sigma_ss": float(sigma_ss),
                            "drag_scale_tau_s": float(tau_s),
                        },
                    )
                )

    n_traj_val = int(design["validation_n_trajectories"])
    n_traj_test = int(design["n_trajectories_planned"])
    steps = int(design["arc_length_steps"])
    dt = float(design["propagator_step_s"])
    truth_beta = float(design["truth_beta_value"])
    altitude_min = float(design["orbit_altitude_min_km"])
    altitude_max = float(design["orbit_altitude_max_km"])
    test_seed = int(design["test_seed"])
    val_seed = int(design["validation_seed"])
    epoch_unix = float(design["epoch_unix"])
    eval_start = 11

    init_pos_sigma = baseline_cfg.ukf.init_pos_std_m
    init_vel_sigma = baseline_cfg.ukf.init_vel_std_mps

    # ---- Validation sweep on the disjoint validation seed ----
    t_val_gen = time.perf_counter()
    val_states, val_meas, val_vis, val_x0_est, val_times = _build_population(
        n_traj=n_traj_val,
        steps=steps,
        dt=dt,
        truth_beta=truth_beta,
        altitude_min=altitude_min,
        altitude_max=altitude_max,
        seed=val_seed,
        dataset_cfg=dataset_cfg,
        init_pos_sigma=init_pos_sigma,
        init_vel_sigma=init_vel_sigma,
    )
    t_val_gen = time.perf_counter() - t_val_gen
    t_val_filt = time.perf_counter()
    val_records = _validation_sweep(
        grid=grid,
        states_all=val_states,
        meas_all=val_meas,
        vis_all=val_vis,
        x0_est_all=val_x0_est,
        times=val_times,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        eval_start=eval_start,
    )
    t_val_filt = time.perf_counter() - t_val_filt
    selected = _select_grid_point(val_records)
    val_payload = {
        "scenario": "drag_scale_ukf_validation",
        "schema_version": "drag_scale_ukf_validation_v1",
        "validation_seed": val_seed,
        "n_trajectories": n_traj_val,
        "steps": steps,
        "dt_s": dt,
        "truth_beta_value": truth_beta,
        "orbit_altitude_min_km": altitude_min,
        "orbit_altitude_max_km": altitude_max,
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "grid_points": [{"label": lab, **pt} for lab, pt in grid],
        "validation_records": val_records,
        "selected_grid_point": selected,
        "tie_breaking_rule": (
            "Lowest validation mean; ties within 1 m broken by smallest "
            "init_drag_scale_std, then smallest drag_scale_sigma_ss, then "
            "smallest drag_scale_tau_s."
        ),
        "elapsed_seconds": {
            "truth_generation": float(t_val_gen),
            "filters": float(t_val_filt),
        },
    }
    out_val_json = Path(args.validation_output_json)
    out_val_json.parent.mkdir(parents=True, exist_ok=True)
    out_val_json.write_text(json.dumps(val_payload, indent=2))

    # ---- Test population on the disjoint held-out test seed ----
    t_test_gen = time.perf_counter()
    states_all, meas_all, vis_all, x0_est_all, times = _build_population(
        n_traj=n_traj_test,
        steps=steps,
        dt=dt,
        truth_beta=truth_beta,
        altitude_min=altitude_min,
        altitude_max=altitude_max,
        seed=test_seed,
        dataset_cfg=dataset_cfg,
        init_pos_sigma=init_pos_sigma,
        init_vel_sigma=init_vel_sigma,
    )
    t_test_gen = time.perf_counter() - t_test_gen

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
    loop54_dsa_sel = _loop54_dsa_selection(
        Path(args.loop54_selection), Path(args.loop54_selection_fallback), dsa_rule
    )
    dsa_ekf_cfg = DragScaleAEKFConfig(
        q_pos_m=float(th_dsa["q_pos_m"]),
        q_vel_mps=float(th_dsa["q_vel_mps"]),
        init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
        init_drag_scale_std=float(loop54_dsa_sel["init_drag_scale_std"]),
        drag_scale_sigma_ss=float(loop54_dsa_sel["drag_scale_sigma_ss"]),
        drag_scale_tau_s=float(loop54_dsa_sel["drag_scale_tau_s"]),
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
        init_drag_scale_std=float(selected["init_drag_scale_std"]),
        drag_scale_sigma_ss=float(selected["drag_scale_sigma_ss"]),
        drag_scale_tau_s=float(selected["drag_scale_tau_s"]),
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
        ),
    )

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

    ekf_pred = np.zeros_like(states_all)
    ukf_pred = np.zeros_like(states_all)
    aukf_pred = np.zeros_like(states_all)
    pukf_pred = np.zeros_like(states_all)
    dmc_pred = np.zeros_like(states_all)
    dsa_ekf_pred = np.zeros_like(states_all)
    dsa_ukf_pred = np.zeros_like(states_all)
    dsa_ekf_beta_final: list[float] = []
    dsa_ukf_beta_max: list[float] = []
    dsa_ukf_beta_mean: list[float] = []
    dsa_ukf_beta_final: list[float] = []

    t_test_filt = time.perf_counter()
    for i in range(n_traj_test):
        ekf_pred[i], _ = run_ekf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=ekf_cfg, **base_kwargs,
        )
        ukf_pred[i], _ = run_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=ukf_cfg, **base_kwargs,
        )
        aukf_pred[i], _ = run_adaptive_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=aukf_cfg, **base_kwargs,
        )
        pukf_pred[i], _, _ = run_process_noise_adaptive_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=pukf_cfg, **base_kwargs,
        )
        dmc_pred[i], _, _ = run_dmc_ekf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=dmc_cfg, **base_kwargs,
        )
        dsa_ekf_pred[i], _, dsa_ekf_diag = run_drag_scale_aekf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=dsa_ekf_cfg, **base_kwargs,
        )
        bh_ekf = dsa_ekf_diag.get("drag_scale_history")
        if bh_ekf is not None and bh_ekf.size:
            dsa_ekf_beta_final.append(float(bh_ekf[-1]))
        dsa_ukf_pred[i], _, dsa_ukf_diag = run_drag_scale_aukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=dsa_ukf_cfg, **base_kwargs,
        )
        bh = dsa_ukf_diag.get("drag_scale_history")
        if bh is not None and bh.size:
            dsa_ukf_beta_max.append(float(np.max(bh)))
            dsa_ukf_beta_mean.append(float(np.mean(bh)))
            dsa_ukf_beta_final.append(float(bh[-1]))
    t_test_filt = time.perf_counter() - t_test_filt

    metrics = {
        "EKF": base._per_traj_observed_pos_rmse(states_all, ekf_pred, vis_all, eval_start),
        "UKF": base._per_traj_observed_pos_rmse(states_all, ukf_pred, vis_all, eval_start),
        "AUKF": base._per_traj_observed_pos_rmse(states_all, aukf_pred, vis_all, eval_start),
        "PUKF": base._per_traj_observed_pos_rmse(states_all, pukf_pred, vis_all, eval_start),
        "DMC_EKF": base._per_traj_observed_pos_rmse(states_all, dmc_pred, vis_all, eval_start),
        "DSA_EKF": base._per_traj_observed_pos_rmse(states_all, dsa_ekf_pred, vis_all, eval_start),
        "DSA_UKF": base._per_traj_observed_pos_rmse(states_all, dsa_ukf_pred, vis_all, eval_start),
    }
    means = {k: float(np.nanmean(v)) for k, v in metrics.items()}

    pair_specs = [
        ("DSA_UKF", "EKF"),
        ("DSA_UKF", "UKF"),
        ("DSA_UKF", "AUKF"),
        ("DSA_UKF", "PUKF"),
        ("DSA_UKF", "DMC_EKF"),
        ("DSA_UKF", "DSA_EKF"),
    ]
    bootstrap_samples = int(rule["decision_predicate"]["bootstrap_samples"])
    paired: dict[str, dict[str, float]] = {}
    rng_offset = 0
    for cand, baseline_name in pair_specs:
        diffs = metrics[cand] - metrics[baseline_name]
        mean_d, lo, hi = base._paired_bootstrap_ci(
            diffs, n_boot=bootstrap_samples, seed=test_seed + rng_offset
        )
        rng_offset += 1
        p = base._one_sided_wilcoxon_candidate_better(diffs)
        finite = diffs[np.isfinite(diffs)]
        paired[f"{cand}_minus_{baseline_name}"] = {
            "candidate": cand,
            "baseline": baseline_name,
            "mean_diff_m": mean_d,
            "ci_lo_m": lo,
            "ci_hi_m": hi,
            "wilcoxon_p_one_sided_candidate_better": p,
            "n_paired": int(finite.size),
            "candidate_better_count": int(np.sum(finite < 0.0)),
        }

    floor = float(rule["decision_predicate"]["practical_significance_floor_fraction"])
    non_candidate_names = ["EKF", "UKF", "AUKF", "PUKF", "DMC_EKF", "DSA_EKF"]
    best_non_candidate = min(non_candidate_names, key=lambda k: means[k])
    best_non_candidate_mean = means[best_non_candidate]
    floor_abs = floor * best_non_candidate_mean
    gap_diffs = metrics["DSA_UKF"] - metrics[best_non_candidate]
    gap_mean, gap_lo, gap_hi = base._paired_bootstrap_ci(
        gap_diffs, n_boot=bootstrap_samples, seed=test_seed + 999
    )
    candidate_is_lowest = means["DSA_UKF"] == min(means.values())
    ci_strictly_negative = gap_hi < 0.0
    floor_exceeded = -gap_mean > floor_abs and gap_mean < 0.0
    is_positive = bool(candidate_is_lowest and ci_strictly_negative and floor_exceeded)

    decision = {
        "best_non_candidate_estimator": best_non_candidate,
        "best_non_candidate_mean_m": best_non_candidate_mean,
        "practical_significance_floor_fraction": floor,
        "practical_significance_floor_abs_m": floor_abs,
        "dsa_ukf_minus_best_non_candidate_mean_m": gap_mean,
        "dsa_ukf_minus_best_non_candidate_ci_lo_m": gap_lo,
        "dsa_ukf_minus_best_non_candidate_ci_hi_m": gap_hi,
        "dsa_ukf_is_strictly_lowest_mean": bool(candidate_is_lowest),
        "ci_strictly_negative_for_dsa_ukf": bool(ci_strictly_negative),
        "floor_exceeded": bool(floor_exceeded),
        "predeclared_positive_criterion_met": is_positive,
    }

    nis_per_filter: dict[str, dict[str, float]] = {}
    for name, preds in (
        ("EKF", ekf_pred), ("UKF", ukf_pred), ("AUKF", aukf_pred),
        ("PUKF", pukf_pred), ("DMC_EKF", dmc_pred),
        ("DSA_EKF", dsa_ekf_pred), ("DSA_UKF", dsa_ukf_pred),
    ):
        nis_per_filter[name] = base._cross_filter_r_only_nis(
            states_all, preds, meas_all, vis_all, times, stations,
            meas_std, eval_start
        )

    payload: dict[str, Any] = {
        "scenario": "drag_scale_ukf_constructive_positive_control",
        "schema_version": "drag_scale_ukf_constructive_positive_control_v1",
        "scope": (
            "UKF-based drag-scale adaptive constructive positive control. "
            "Truth-side dynamics: compact two-body + J2 + exponential-atmosphere "
            "drag with the truth-side ballistic coefficient multiplied by "
            f"truth_beta={truth_beta:.2f}. Estimator-side dynamics: same compact "
            "two-body + J2 + drag at the nominal ballistic coefficient. "
            "DSA-UKF estimates beta online via deterministic sigma-point "
            "propagation of the augmented seven-dimensional flow. "
            "Every other estimator keeps the unscaled compact dynamics. The "
            "loop 54 DSA-EKF constructive-control outcome is preserved unchanged."
        ),
        "n_trajectories": int(n_traj_test),
        "validation_n_trajectories": int(n_traj_val),
        "steps": int(steps),
        "dt_s": float(dt),
        "eval_start_step": int(eval_start),
        "epoch_unix": float(epoch_unix),
        "test_seed": int(test_seed),
        "validation_seed": int(val_seed),
        "truth_beta_value": float(truth_beta),
        "orbit_altitude_min_km": float(altitude_min),
        "orbit_altitude_max_km": float(altitude_max),
        "bootstrap_samples": int(bootstrap_samples),
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "selected_grid_point": selected,
        "validation_artifact_source": str(out_val_json),
        "validation_artifact_sha256": hashlib.sha256(
            out_val_json.read_bytes()
        ).hexdigest(),
        "loop54_dsa_ekf_selection": loop54_dsa_sel,
        "truth_acceleration": (
            "compact two-body + J2 + exponential drag with truth-side "
            "ballistic coefficient scaled by truth_beta"
        ),
        "estimator_acceleration": (
            "compact two-body + J2 + exponential drag at nominal ballistic "
            "coefficient (DSA-EKF and DSA-UKF estimate beta online via their "
            "respective recursions)"
        ),
        "observed_step_rmse_mean_m": means,
        "paired": paired,
        "decision": decision,
        "dsa_ukf_diagnostics": {
            "median_max_abs_beta_deviation": (
                float(np.median([abs(b - 1.0) for b in dsa_ukf_beta_max]))
                if dsa_ukf_beta_max else float("nan")
            ),
            "median_mean_beta": (
                float(np.median(dsa_ukf_beta_mean)) if dsa_ukf_beta_mean else float("nan")
            ),
            "median_final_beta": (
                float(np.median(dsa_ukf_beta_final)) if dsa_ukf_beta_final else float("nan")
            ),
            "truth_beta_target": float(truth_beta),
        },
        "dsa_ekf_diagnostics": {
            "median_final_beta": (
                float(np.median(dsa_ekf_beta_final)) if dsa_ekf_beta_final else float("nan")
            ),
            "truth_beta_target": float(truth_beta),
        },
        "cross_filter_r_only_nis": nis_per_filter,
        "elapsed_seconds": {
            "truth_generation_validation": float(t_val_gen),
            "filters_validation": float(t_val_filt),
            "truth_generation_test": float(t_test_gen),
            "filters_test": float(t_test_filt),
        },
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    rows = []
    for i in range(n_traj_test):
        row: dict[str, Any] = {"trajectory_index": i}
        for k, arr in metrics.items():
            row[f"{k}_observed_pos_rmse_m"] = (
                float(arr[i]) if np.isfinite(arr[i]) else None
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
