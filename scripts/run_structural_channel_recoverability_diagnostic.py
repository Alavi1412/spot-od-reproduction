#!/usr/bin/env python
"""Cheap structural-channel recoverability diagnostic for SPOT-OD.

This driver runs two deliberately favorable implementation sanity checks:

* DSA-EKF is given an all-visible, low-noise drag-scale mismatch where the
  truth-side ballistic coefficient is scaled by a known beta.
* DMC-EKF is given an all-visible, low-noise compact-dynamics trajectory with a
  known constant inertial acceleration added to the truth.

The diagnostic is scoped to recoverability of already-implemented structural
channels. It is not an operational orbit-determination scenario and is not a
primary endpoint.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_hifi_force_mismatch as base

from gnn_state_estimation.coordinates import StationGeometry
from gnn_state_estimation.dynamics import acceleration_eci, kepler_to_cartesian
from gnn_state_estimation.filters import (
    DMCEKFConfig,
    DragScaleAEKFConfig,
    EKFConfig,
    run_dmc_ekf,
    run_drag_scale_aekf,
    run_ekf,
)
from gnn_state_estimation.simulation import MeasurementNoiseConfig, parse_dataset_config
from gnn_state_estimation.utils.io import load_yaml


_PERTURBATION = np.array(
    [10.0, -8.0, 5.0, 0.02, -0.01, 0.015], dtype=np.float64
)


def _all_visible_stations() -> tuple[StationGeometry, ...]:
    return (
        StationGeometry("Equator0", 0.0, 0.0, 0.0, -90.0),
        StationGeometry("Equator90E", 0.0, 90.0, 0.0, -90.0),
        StationGeometry("NorthMid", 45.0, -90.0, 0.0, -90.0),
        StationGeometry("SouthMid", -45.0, 150.0, 0.0, -90.0),
    )


def _measurement_noise(spec: dict[str, float]) -> MeasurementNoiseConfig:
    return MeasurementNoiseConfig(
        range_std_m=float(spec["range_std_m"]),
        az_std_deg=float(spec["az_std_deg"]),
        el_std_deg=float(spec["el_std_deg"]),
        range_rate_std_mps=float(spec["range_rate_std_mps"]),
        outlier_prob=0.0,
        outlier_scale=1.0,
        random_dropout_prob=0.0,
    )


def _initial_state(spec: dict[str, float]) -> np.ndarray:
    earth_radius_m = 6378.1363e3
    semi_major_axis_m = earth_radius_m + float(spec["altitude_km"]) * 1e3
    return kepler_to_cartesian(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=float(spec["eccentricity"]),
        inclination_rad=np.deg2rad(float(spec["inclination_deg"])),
        raan_rad=np.deg2rad(float(spec["raan_deg"])),
        arg_perigee_rad=np.deg2rad(float(spec["arg_perigee_deg"])),
        true_anomaly_rad=np.deg2rad(float(spec["true_anomaly_deg"])),
    )


def _state_derivative(
    state: np.ndarray,
    *,
    t_s: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    extra_accel_mps2: np.ndarray | None = None,
) -> np.ndarray:
    a = acceleration_eci(
        r_eci=state[:3],
        v_eci=state[3:],
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        t_s=t_s,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
        enable_third_body=False,
        enable_srp=False,
    )
    if extra_accel_mps2 is not None:
        a = a + np.asarray(extra_accel_mps2, dtype=np.float64)
    return np.hstack([state[3:], a]).astype(np.float64)


def _rk4_step(
    state: np.ndarray,
    *,
    dt: float,
    t_s: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    extra_accel_mps2: np.ndarray | None = None,
) -> np.ndarray:
    kwargs = dict(
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
        extra_accel_mps2=extra_accel_mps2,
    )
    k1 = _state_derivative(state, t_s=t_s, **kwargs)
    k2 = _state_derivative(state + 0.5 * dt * k1, t_s=t_s + 0.5 * dt, **kwargs)
    k3 = _state_derivative(state + 0.5 * dt * k2, t_s=t_s + 0.5 * dt, **kwargs)
    k4 = _state_derivative(state + dt * k3, t_s=t_s + dt, **kwargs)
    return (state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(
        np.float64
    )


def _propagate(
    x0: np.ndarray,
    *,
    steps: int,
    dt: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    extra_accel_mps2: np.ndarray | None = None,
) -> np.ndarray:
    states = np.zeros((steps, 6), dtype=np.float64)
    states[0] = np.asarray(x0, dtype=np.float64)
    for k in range(1, steps):
        states[k] = _rk4_step(
            states[k - 1],
            dt=dt,
            t_s=(k - 1) * dt,
            ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
            drag_rho_ref=drag_rho_ref,
            drag_h_ref_m=drag_h_ref_m,
            drag_scale_height_m=drag_scale_height_m,
            extra_accel_mps2=extra_accel_mps2,
        )
    return states


def _ekf_config(spec: dict[str, float]) -> EKFConfig:
    return EKFConfig(
        q_pos_m=float(spec["q_pos_m"]),
        q_vel_mps=float(spec["q_vel_mps"]),
        init_pos_std_m=float(spec["init_pos_std_m"]),
        init_vel_std_mps=float(spec["init_vel_std_mps"]),
        gating_threshold=float(spec["gating_threshold"]),
    )


def _dsa_config(ekf_spec: dict[str, float], dsa_spec: dict[str, float]) -> DragScaleAEKFConfig:
    return DragScaleAEKFConfig(
        q_pos_m=float(ekf_spec["q_pos_m"]),
        q_vel_mps=float(ekf_spec["q_vel_mps"]),
        init_pos_std_m=float(ekf_spec["init_pos_std_m"]),
        init_vel_std_mps=float(ekf_spec["init_vel_std_mps"]),
        init_drag_scale_std=float(dsa_spec["init_drag_scale_std"]),
        drag_scale_sigma_ss=float(dsa_spec["drag_scale_sigma_ss"]),
        drag_scale_tau_s=float(dsa_spec["drag_scale_tau_s"]),
        gating_threshold=float(ekf_spec["gating_threshold"]),
    )


def _dmc_config(ekf_spec: dict[str, float], dmc_spec: dict[str, float]) -> DMCEKFConfig:
    return DMCEKFConfig(
        q_pos_m=float(ekf_spec["q_pos_m"]),
        q_vel_mps=float(ekf_spec["q_vel_mps"]),
        init_pos_std_m=float(ekf_spec["init_pos_std_m"]),
        init_vel_std_mps=float(ekf_spec["init_vel_std_mps"]),
        init_emp_accel_std_mps2=float(dmc_spec["init_emp_accel_std_mps2"]),
        emp_accel_sigma_mps2=float(dmc_spec["emp_accel_sigma_mps2"]),
        emp_accel_tau_s=float(dmc_spec["emp_accel_tau_s"]),
        gating_threshold=float(ekf_spec["gating_threshold"]),
    )


def _rmse_m(truth: np.ndarray, pred: np.ndarray, eval_start: int) -> float:
    err = truth[eval_start:, :3] - pred[eval_start:, :3]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def _common_filter_kwargs(dyn) -> dict[str, Any]:
    return {
        "drag_rho_ref": dyn.drag_rho_ref,
        "drag_h_ref_m": dyn.drag_h_ref_m,
        "drag_scale_height_m": dyn.drag_scale_height_m,
        "enable_third_body": False,
        "enable_srp": False,
        "srp_area_to_mass_m2_per_kg": dyn.srp_area_to_mass_m2_per_kg,
        "srp_cr": dyn.srp_cr,
        "sun_initial_phase_rad": dyn.sun_initial_phase_rad,
        "moon_initial_phase_rad": dyn.moon_initial_phase_rad,
    }


def _run_dsa_case(rule: dict[str, Any], dyn, stations: tuple[StationGeometry, ...]) -> dict[str, Any]:
    spec = rule["diagnostic_cases"]["drag_scale"]
    steps = int(spec["steps"])
    dt = float(spec["dt_s"])
    eval_start = int(spec["eval_start_step"])
    times = np.arange(steps, dtype=np.float64) * dt
    meas_noise = _measurement_noise(spec["measurement_noise"])
    rng = np.random.default_rng(int(spec["seed"]))
    truth_beta = float(spec["truth_beta"])
    x0 = _initial_state(spec["orbit"])
    states = _propagate(
        x0,
        steps=steps,
        dt=dt,
        ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg) * truth_beta,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
    )
    measurements, visibility = base._generate_observations(
        states=states,
        times=times,
        stations=stations,
        noise_std=meas_noise.std_vector,
        outlier_prob=0.0,
        outlier_scale=1.0,
        dropout_prob=0.0,
        rng=rng,
    )
    x0_est = states[0] + _PERTURBATION
    filter_kwargs = _common_filter_kwargs(dyn)
    ekf_cfg = _ekf_config(spec["ekf"])
    dsa_cfg = _dsa_config(spec["ekf"], spec["dsa_ekf"])

    t0 = time.perf_counter()
    ekf_pred, _ = run_ekf(
        measurements=measurements,
        visibility=visibility,
        times_s=times,
        stations=stations,
        ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg),
        meas_std_vector=meas_noise.std_vector,
        x0_est=x0_est,
        cfg=ekf_cfg,
        **filter_kwargs,
    )
    dsa_pred, _, dsa_diag = run_drag_scale_aekf(
        measurements=measurements,
        visibility=visibility,
        times_s=times,
        stations=stations,
        ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg),
        meas_std_vector=meas_noise.std_vector,
        x0_est=x0_est,
        cfg=dsa_cfg,
        **filter_kwargs,
    )
    elapsed = time.perf_counter() - t0
    beta_history = np.asarray(dsa_diag["drag_scale_history"], dtype=np.float64)
    last100 = beta_history[-100:]
    ekf_rmse = _rmse_m(states, ekf_pred, eval_start)
    dsa_rmse = _rmse_m(states, dsa_pred, eval_start)
    beta_mean = float(np.mean(last100))
    return {
        "target": "drag_scale_beta",
        "truth_beta": truth_beta,
        "last100_beta_mean": beta_mean,
        "last100_beta_std": float(np.std(last100)),
        "beta_abs_error": float(abs(beta_mean - truth_beta)),
        "beta_relative_error": float(abs(beta_mean - truth_beta) / truth_beta),
        "beta_final": float(beta_history[-1]),
        "beta_min": float(np.min(beta_history)),
        "beta_max": float(np.max(beta_history)),
        "rmse_m": {"EKF": ekf_rmse, "DSA_EKF": dsa_rmse},
        "candidate_improves_over_ekf": bool(dsa_rmse < ekf_rmse),
        "visible_fraction": float(np.mean(visibility)),
        "elapsed_seconds": float(elapsed),
        "n_steps": steps,
        "dt_s": dt,
        "eval_start_step": eval_start,
        "seed": int(spec["seed"]),
        "n_stations": len(stations),
        "measurement_noise": spec["measurement_noise"],
        "orbit": spec["orbit"],
    }


def _run_dmc_case(rule: dict[str, Any], dyn, stations: tuple[StationGeometry, ...]) -> dict[str, Any]:
    spec = rule["diagnostic_cases"]["empirical_acceleration"]
    steps = int(spec["steps"])
    dt = float(spec["dt_s"])
    eval_start = int(spec["eval_start_step"])
    times = np.arange(steps, dtype=np.float64) * dt
    meas_noise = _measurement_noise(spec["measurement_noise"])
    rng = np.random.default_rng(int(spec["seed"]))
    truth_accel = np.asarray(spec["truth_empirical_acceleration_mps2"], dtype=np.float64)
    x0 = _initial_state(spec["orbit"])
    states = _propagate(
        x0,
        steps=steps,
        dt=dt,
        ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg),
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
        extra_accel_mps2=truth_accel,
    )
    measurements, visibility = base._generate_observations(
        states=states,
        times=times,
        stations=stations,
        noise_std=meas_noise.std_vector,
        outlier_prob=0.0,
        outlier_scale=1.0,
        dropout_prob=0.0,
        rng=rng,
    )
    x0_est = states[0] + _PERTURBATION
    filter_kwargs = _common_filter_kwargs(dyn)
    ekf_cfg = _ekf_config(spec["ekf"])
    dmc_cfg = _dmc_config(spec["ekf"], spec["dmc_ekf"])

    t0 = time.perf_counter()
    ekf_pred, _ = run_ekf(
        measurements=measurements,
        visibility=visibility,
        times_s=times,
        stations=stations,
        ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg),
        meas_std_vector=meas_noise.std_vector,
        x0_est=x0_est,
        cfg=ekf_cfg,
        **filter_kwargs,
    )
    dmc_pred, _, dmc_diag = run_dmc_ekf(
        measurements=measurements,
        visibility=visibility,
        times_s=times,
        stations=stations,
        ballistic_coeff_m2_per_kg=float(dyn.ballistic_coeff_m2_per_kg),
        meas_std_vector=meas_noise.std_vector,
        x0_est=x0_est,
        cfg=dmc_cfg,
        **filter_kwargs,
    )
    elapsed = time.perf_counter() - t0
    accel_history = np.asarray(dmc_diag["empirical_acceleration_mps2"], dtype=np.float64)
    last100_mean = np.mean(accel_history[-100:], axis=0)
    ekf_rmse = _rmse_m(states, ekf_pred, eval_start)
    dmc_rmse = _rmse_m(states, dmc_pred, eval_start)
    rel_error = float(
        np.linalg.norm(last100_mean - truth_accel) / np.linalg.norm(truth_accel)
    )
    return {
        "target": "constant_inertial_empirical_acceleration",
        "truth_empirical_acceleration_mps2": truth_accel.tolist(),
        "last100_empirical_acceleration_mean_mps2": last100_mean.tolist(),
        "empirical_acceleration_relative_l2_error": rel_error,
        "empirical_acceleration_abs_l2_error_mps2": float(
            np.linalg.norm(last100_mean - truth_accel)
        ),
        "empirical_acceleration_final_mps2": accel_history[-1].tolist(),
        "rmse_m": {"EKF": ekf_rmse, "DMC_EKF": dmc_rmse},
        "candidate_improves_over_ekf": bool(dmc_rmse < ekf_rmse),
        "visible_fraction": float(np.mean(visibility)),
        "elapsed_seconds": float(elapsed),
        "n_steps": steps,
        "dt_s": dt,
        "eval_start_step": eval_start,
        "seed": int(spec["seed"]),
        "n_stations": len(stations),
        "measurement_noise": spec["measurement_noise"],
        "orbit": spec["orbit"],
    }


def _fmt_float(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return "nan"
    if abs(x) >= 1e4 or (0 < abs(x) < 0.01):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def _fmt_vec_mps2(v: list[float]) -> str:
    scaled = [float(x) / 1e-5 for x in v]
    return r"$[" + ", ".join(f"{x:.2f}" for x in scaled) + r"]\times10^{-5}$"


def _render_table(payload: dict[str, Any]) -> str:
    dsa = payload["dsa_drag_scale"]
    dmc = payload["dmc_empirical_acceleration"]
    lines = [
        r"\begin{table}[!htbp]",
        r"  \centering\small",
        (
            r"  \caption{Favorable-geometry implementation sanity diagnostic for "
            r"structural channels. In deliberately all-visible, low-noise geometry, "
            r"the drag-scale adaptive channel and empirical-acceleration channel "
            r"are each evaluated against a matched synthetic signal. This diagnostic "
            r"verifies implementation recoverability only; it is not an operational "
            r"orbit-determination setting and not a primary endpoint result.}"
        ),
        r"  \label{tab:structural_channel_recoverability}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Case & Known signal & Last-100 mean estimate & EKF$\to$channel RMSE [m] \\",
        r"    \midrule",
        (
            r"    DSA-EKF drag scale $\beta$ & "
            f"{dsa['truth_beta']:.2f} & {dsa['last100_beta_mean']:.3f} "
            f"& {_fmt_float(dsa['rmse_m']['EKF'])}$\\to$"
            f"{_fmt_float(dsa['rmse_m']['DSA_EKF'])} " + r"\\"
        ),
        (
            r"    DMC-EKF empirical acceleration [m/s$^2$] & "
            f"{_fmt_vec_mps2(dmc['truth_empirical_acceleration_mps2'])} & "
            f"{_fmt_vec_mps2(dmc['last100_empirical_acceleration_mean_mps2'])} "
            f"& {_fmt_float(dmc['rmse_m']['EKF'])}$\\to$"
            f"{_fmt_float(dmc['rmse_m']['DMC_EKF'])} " + r"\\"
        ),
        r"    \midrule",
        (
            r"    Drag-scale relative recovery error & "
            r"\multicolumn{3}{c}{"
            f"{100.0 * dsa['beta_relative_error']:.2f}" + r"\%} \\"
        ),
        (
            r"    Empirical-acceleration relative recovery error & "
            r"\multicolumn{3}{c}{"
            f"{100.0 * dmc['empirical_acceleration_relative_l2_error']:.2f}"
            + r"\%} \\"
        ),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        (
            r"  \\[2pt] {\footnotesize The geometry uses four ground stations with "
            r"minimum elevation $-90^\circ$, no dropout, no outliers, and "
            r"lower-than-nominal measurement noise. Favorable conditions are "
            r"intentional and bound the interpretation to recoverability of the "
            r"implemented structural channels.}"
        ),
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--predeclared-rule",
        default="release/predeclarations/structural_channel_recoverability_loop70.json",
    )
    parser.add_argument(
        "--output-json",
        default=(
            "results/structural_channel_recoverability/"
            "structural_channel_recoverability.json"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=(
            "results/structural_channel_recoverability/"
            "structural_channel_recoverability.csv"
        ),
    )
    parser.add_argument(
        "--output-table",
        default="paper/tables/structural_channel_recoverability.tex",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    rule_path = Path(args.predeclared_rule)
    rule = json.loads(rule_path.read_text())
    stations = _all_visible_stations()

    total_t = time.perf_counter()
    dsa = _run_dsa_case(rule, dyn, stations)
    dmc = _run_dmc_case(rule, dyn, stations)
    elapsed = time.perf_counter() - total_t

    payload = {
        "scenario": "structural_channel_recoverability",
        "schema_version": "structural_channel_recoverability_v1",
        "scope": (
            "Cheap favorable-geometry implementation sanity diagnostic for "
            "recoverability of two structural channels. The diagnostic uses "
            "matched synthetic signals and all-visible low-noise geometry; it "
            "is not an operational OD setting and not a primary endpoint."
        ),
        "predeclared_rule_path": str(rule_path),
        "predeclared_rule_digest_sha256": hashlib.sha256(
            rule_path.read_bytes()
        ).hexdigest(),
        "nominal_ballistic_coeff_m2_per_kg": float(
            dyn.ballistic_coeff_m2_per_kg
        ),
        "dsa_drag_scale": dsa,
        "dmc_empirical_acceleration": dmc,
        "decision": {
            "dsa_beta_recovery_within_tolerance": bool(
                dsa["beta_relative_error"]
                <= float(rule["acceptance"]["dsa_beta_relative_tolerance"])
            ),
            "dsa_improves_over_ekf": bool(dsa["candidate_improves_over_ekf"]),
            "dmc_empirical_acceleration_recovery_within_tolerance": bool(
                dmc["empirical_acceleration_relative_l2_error"]
                <= float(rule["acceptance"]["dmc_acceleration_relative_l2_tolerance"])
            ),
            "dmc_improves_over_ekf": bool(dmc["candidate_improves_over_ekf"]),
        },
        "elapsed_seconds_total": float(elapsed),
    }
    payload["decision"]["all_checks_pass"] = bool(
        all(payload["decision"].values())
    )

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_table = Path(args.output_table)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    out_table.write_text(_render_table(payload))

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "ekf_rmse_m",
                "channel_rmse_m",
                "known_signal",
                "last100_mean_estimate",
                "relative_recovery_error",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case": "drag_scale_beta",
                "ekf_rmse_m": dsa["rmse_m"]["EKF"],
                "channel_rmse_m": dsa["rmse_m"]["DSA_EKF"],
                "known_signal": dsa["truth_beta"],
                "last100_mean_estimate": dsa["last100_beta_mean"],
                "relative_recovery_error": dsa["beta_relative_error"],
            }
        )
        writer.writerow(
            {
                "case": "empirical_acceleration",
                "ekf_rmse_m": dmc["rmse_m"]["EKF"],
                "channel_rmse_m": dmc["rmse_m"]["DMC_EKF"],
                "known_signal": json.dumps(dmc["truth_empirical_acceleration_mps2"]),
                "last100_mean_estimate": json.dumps(
                    dmc["last100_empirical_acceleration_mean_mps2"]
                ),
                "relative_recovery_error": dmc[
                    "empirical_acceleration_relative_l2_error"
                ],
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
