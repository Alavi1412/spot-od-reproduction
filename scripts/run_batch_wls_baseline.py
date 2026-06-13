#!/usr/bin/env python
"""Compute a robust batch weighted least-squares OD reference baseline.

The baseline estimates one initial Cartesian state per trajectory from the
full visible measurement arc, using the same deterministic force model and
line-of-sight measurement model as the recursive filters. It is an offline
postfit OD reference, not a causal filter.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from gnn_state_estimation.coordinates import line_of_sight_measurement
from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.dynamics import propagate_orbit
from gnn_state_estimation.evaluation import score_predictions
from gnn_state_estimation.filters.ekf import wrap_angle_pi
from gnn_state_estimation.scenarios import estimator_sim_config
from gnn_state_estimation.simulation import DatasetConfig, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output-dir", type=str, default="results/batch_wls_baseline")
    parser.add_argument("--scenarios", type=str, default="test,stress_test")
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--huber-f-scale", type=float, default=2.5)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def scenario_dataset_config(cfg: dict[str, Any], scenario: str) -> DatasetConfig:
    sim_cfg = copy.deepcopy(cfg["simulation"])
    if scenario == "stress_test":
        sim_cfg = deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg:
        # Batch WLS is an estimator/OD reference: use the estimator-side config
        # (== truth config unless the scenario declares estimator_overrides).
        sim_cfg = estimator_sim_config(sim_cfg, scenario_cfg)
    return parse_dataset_config(sim_cfg)


def _propagate_from_initial(x0_est: np.ndarray, dataset_cfg: DatasetConfig) -> np.ndarray:
    dyn = dataset_cfg.dynamics
    return propagate_orbit(
        initial_state_eci=x0_est,
        dt=dyn.dt_s,
        steps=dyn.steps,
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        process_noise_std=0.0,
        rng=None,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
        enable_third_body=dyn.enable_third_body,
        enable_srp=dyn.enable_srp,
        srp_area_to_mass_m2_per_kg=dyn.srp_area_to_mass_m2_per_kg,
        srp_cr=dyn.srp_cr,
        sun_initial_phase_rad=dyn.sun_initial_phase_rad,
        moon_initial_phase_rad=dyn.moon_initial_phase_rad,
    )


def _measurement_residuals(
    states: np.ndarray,
    measurements: np.ndarray,
    visible_pairs: np.ndarray,
    times_s: np.ndarray,
    dataset_cfg: DatasetConfig,
) -> np.ndarray:
    meas_std = dataset_cfg.measurement_noise.std_vector
    residuals = np.empty((visible_pairs.shape[0], 4), dtype=np.float64)
    for out_idx, (t_idx, s_idx) in enumerate(visible_pairs):
        z_pred, _ = line_of_sight_measurement(
            states[int(t_idx)],
            dataset_cfg.stations[int(s_idx)],
            float(times_s[int(t_idx)]),
        )
        y = np.asarray(measurements[int(t_idx), int(s_idx)], dtype=np.float64) - z_pred
        y[1] = wrap_angle_pi(float(y[1]))
        residuals[out_idx] = y / meas_std
    return residuals.reshape(-1)


def fit_batch_wls_trajectory(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    x0_est: np.ndarray,
    dataset_cfg: DatasetConfig,
    initial_scale: np.ndarray,
    *,
    max_nfev: int,
    huber_f_scale: float,
    prior_weight: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    visible_pairs = np.argwhere(visibility >= 0.5)
    if visible_pairs.size == 0:
        prediction = _propagate_from_initial(x0_est, dataset_cfg)
        return prediction, {
            "success": False,
            "status": 0,
            "message": "no_visible_measurements",
            "nfev": 0,
            "num_measurements": 0,
            "cost": float("nan"),
            "optimality": float("nan"),
            "initial_state_correction_norm": 0.0,
        }

    x0_base = np.asarray(x0_est, dtype=np.float64)
    initial_scale = np.asarray(initial_scale, dtype=np.float64)

    def residual_fn(u: np.ndarray) -> np.ndarray:
        candidate_x0 = x0_base + np.asarray(u, dtype=np.float64) * initial_scale
        states = _propagate_from_initial(candidate_x0, dataset_cfg)
        prior_residual = float(prior_weight) * np.asarray(u, dtype=np.float64)
        measurement_residual = _measurement_residuals(
            states=states,
            measurements=measurements,
            visible_pairs=visible_pairs,
            times_s=times_s,
            dataset_cfg=dataset_cfg,
        )
        if not np.all(np.isfinite(measurement_residual)):
            measurement_residual = np.nan_to_num(
                measurement_residual,
                nan=1.0e6,
                posinf=1.0e6,
                neginf=-1.0e6,
            )
        return np.concatenate([prior_residual, measurement_residual])

    result = least_squares(
        residual_fn,
        x0=np.zeros(6, dtype=np.float64),
        bounds=(-12.0 * np.ones(6, dtype=np.float64), 12.0 * np.ones(6, dtype=np.float64)),
        loss="huber",
        f_scale=float(huber_f_scale),
        x_scale=np.ones(6, dtype=np.float64),
        max_nfev=int(max_nfev),
        ftol=1.0e-7,
        xtol=1.0e-7,
        gtol=1.0e-7,
    )
    fitted_x0 = x0_base + result.x * initial_scale
    prediction = _propagate_from_initial(fitted_x0, dataset_cfg)
    return prediction, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "num_measurements": int(visible_pairs.shape[0]),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "initial_state_correction_norm": float(np.linalg.norm(result.x * initial_scale)),
        "initial_position_correction_m": float(np.linalg.norm(result.x[:3] * initial_scale[:3])),
        "initial_velocity_correction_mps": float(np.linalg.norm(result.x[3:] * initial_scale[3:])),
    }


def _fit_single_worker(payload: dict[str, Any]) -> tuple[int, np.ndarray, dict[str, Any]]:
    idx = int(payload["idx"])
    pred, info = fit_batch_wls_trajectory(
        measurements=payload["measurements"],
        visibility=payload["visibility"],
        times_s=payload["times_s"],
        x0_est=payload["x0_est"],
        dataset_cfg=payload["dataset_cfg"],
        initial_scale=payload["initial_scale"],
        max_nfev=int(payload["max_nfev"]),
        huber_f_scale=float(payload["huber_f_scale"]),
        prior_weight=float(payload["prior_weight"]),
    )
    info["trajectory"] = idx
    return idx, pred, info


def _masked_pos_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    err = y_true[mask, :3] - y_pred[mask, :3]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def _observed_metrics(states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int) -> dict[str, float]:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    zero_visible = ~observed
    y_true = states[:, eval_start:]
    y_pred = preds[:, eval_start:]
    return {
        "observed_step_pos_rmse_m": _masked_pos_rmse(y_true, y_pred, observed),
        "zero_visible_pos_rmse_m": _masked_pos_rmse(y_true, y_pred, zero_visible),
        "observed_steps": int(np.sum(observed)),
        "zero_visible_steps": int(np.sum(zero_visible)),
    }


def _trajectory_rmse_values(y_true: np.ndarray, y_pred: np.ndarray, eval_start: int) -> np.ndarray:
    err = y_true[:, eval_start:, :3] - y_pred[:, eval_start:, :3]
    return np.sqrt(np.mean(np.sum(err * err, axis=-1), axis=1))


def _method_summary(
    states: np.ndarray,
    preds: np.ndarray,
    visibility: np.ndarray,
    eval_start: int,
) -> dict[str, float]:
    all_step = score_predictions(states[:, eval_start:], preds[:, eval_start:])
    obs = _observed_metrics(states, preds, visibility, eval_start)
    traj = _trajectory_rmse_values(states, preds, eval_start)
    return {
        "all_step_pos_rmse_m": float(all_step["pos_rmse_m"]),
        "all_step_vel_rmse_mps": float(all_step["vel_rmse_mps"]),
        "median_traj_pos_rmse_m": float(np.median(traj)),
        "max_traj_pos_rmse_m": float(np.max(traj)),
        "failure_rate_100km": float(np.mean(traj > 100_000.0)),
        **obs,
    }


def run_scenario(
    cfg: dict[str, Any],
    scenario: str,
    output_dir: Path,
    *,
    eval_start: int,
    max_nfev: int,
    huber_f_scale: float,
    prior_weight: float,
    trajectory_limit: int,
    workers: int,
) -> dict[str, Any]:
    data_dir = Path(cfg["data"]["output_dir"])
    arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
    if arrays.x0_estimates is None:
        raise ValueError(f"{scenario} is missing x0_estimates.")
    if arrays.ekf_prior is None or arrays.ukf_prior is None or arrays.aukf_prior is None:
        raise ValueError(f"{scenario} is missing recursive filter prior arrays.")

    dataset_cfg = scenario_dataset_config(cfg, scenario)
    n_traj = arrays.states.shape[0] if trajectory_limit <= 0 else min(trajectory_limit, arrays.states.shape[0])
    initial_scale = np.array(
        [
            float(cfg["baselines"]["init_pos_std_m"]),
            float(cfg["baselines"]["init_pos_std_m"]),
            float(cfg["baselines"]["init_pos_std_m"]),
            float(cfg["baselines"]["init_vel_std_mps"]),
            float(cfg["baselines"]["init_vel_std_mps"]),
            float(cfg["baselines"]["init_vel_std_mps"]),
        ],
        dtype=np.float64,
    )

    preds = np.zeros_like(arrays.states[:n_traj])
    fit_rows: list[dict[str, Any]] = []
    payloads = [
        {
            "idx": i,
            "measurements": arrays.measurements[i],
            "visibility": arrays.visibility[i],
            "times_s": arrays.times[i],
            "x0_est": arrays.x0_estimates[i],
            "dataset_cfg": dataset_cfg,
            "initial_scale": initial_scale,
            "max_nfev": max_nfev,
            "huber_f_scale": huber_f_scale,
            "prior_weight": prior_weight,
        }
        for i in range(n_traj)
    ]
    completed = 0
    if workers <= 1:
        for payload in payloads:
            idx, pred, info = _fit_single_worker(payload)
            preds[idx] = pred
            fit_rows.append(info)
            completed += 1
            if completed % 8 == 0 or completed == n_traj:
                print(f"{scenario}: fitted {completed}/{n_traj} trajectories")
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_fit_single_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                idx, pred, info = future.result()
                preds[idx] = pred
                fit_rows.append(info)
                completed += 1
                if completed % 8 == 0 or completed == n_traj:
                    print(f"{scenario}: fitted {completed}/{n_traj} trajectories")

    state = arrays.states[:n_traj]
    vis = arrays.visibility[:n_traj]
    method_summaries = {
        "BatchWLS": _method_summary(state, preds, vis, eval_start),
        "EKF": _method_summary(state, arrays.ekf_prior[:n_traj], vis, eval_start),
        "UKF": _method_summary(state, arrays.ukf_prior[:n_traj], vis, eval_start),
        "AUKF": _method_summary(state, arrays.aukf_prior[:n_traj], vis, eval_start),
    }
    best_recursive_method = min(
        ("EKF", "UKF", "AUKF"),
        key=lambda name: method_summaries[name]["observed_step_pos_rmse_m"],
    )
    best_recursive = method_summaries[best_recursive_method]
    wls = method_summaries["BatchWLS"]
    observed_gain = (
        100.0
        * (best_recursive["observed_step_pos_rmse_m"] - wls["observed_step_pos_rmse_m"])
        / best_recursive["observed_step_pos_rmse_m"]
    )
    all_step_gain = (
        100.0
        * (best_recursive["all_step_pos_rmse_m"] - wls["all_step_pos_rmse_m"])
        / best_recursive["all_step_pos_rmse_m"]
    )

    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scenario_dir / "batch_wls_predictions.npz", batch_wls=preds)
    pd.DataFrame(fit_rows).to_csv(scenario_dir / "batch_wls_fit_diagnostics.csv", index=False)
    summary = {
        "scenario": scenario,
        "trajectories": int(n_traj),
        "eval_start_step": int(eval_start),
        "max_nfev": int(max_nfev),
        "huber_f_scale": float(huber_f_scale),
        "prior_weight": float(prior_weight),
        "mean_visible_measurements_per_traj": float(np.mean([row["num_measurements"] for row in fit_rows])),
        "fit_success_rate": float(np.mean([bool(row["success"]) for row in fit_rows])),
        "best_recursive_observed_method": best_recursive_method,
        "wls_gain_vs_best_recursive_observed_percent": float(observed_gain),
        "wls_gain_vs_best_recursive_all_step_percent": float(all_step_gain),
        "methods": method_summaries,
    }
    (scenario_dir / "batch_wls_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def flatten_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        base = {
            "scenario": summary["scenario"],
            "trajectories": summary["trajectories"],
            "eval_start_step": summary["eval_start_step"],
            "mean_visible_measurements_per_traj": summary["mean_visible_measurements_per_traj"],
            "fit_success_rate": summary["fit_success_rate"],
            "best_recursive_observed_method": summary["best_recursive_observed_method"],
            "wls_gain_vs_best_recursive_observed_percent": summary["wls_gain_vs_best_recursive_observed_percent"],
            "wls_gain_vs_best_recursive_all_step_percent": summary["wls_gain_vs_best_recursive_all_step_percent"],
        }
        for method, metrics in summary["methods"].items():
            prefix = method.lower()
            for key, value in metrics.items():
                base[f"{prefix}_{key}"] = value
        rows.append(base)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    summaries = []
    for scenario in [name.strip() for name in args.scenarios.split(",") if name.strip()]:
        summaries.append(
            run_scenario(
                cfg,
                scenario,
                output_dir,
                eval_start=eval_start,
                max_nfev=args.max_nfev,
                huber_f_scale=args.huber_f_scale,
                prior_weight=args.prior_weight,
                trajectory_limit=args.trajectory_limit,
                workers=args.workers,
            )
        )
    rows = flatten_summary_rows(summaries)
    pd.DataFrame(rows).to_csv(output_dir / "batch_wls_summary.csv", index=False)
    (output_dir / "batch_wls_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(output_dir / "batch_wls_summary.csv"), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
