#!/usr/bin/env python
"""Compute open-loop propagation baselines for manuscript diagnostics."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.dynamics import propagate_orbit
from gnn_state_estimation.evaluation import score_predictions
from gnn_state_estimation.simulation import DatasetConfig, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output-dir", type=str, default="results/propagation_baseline")
    parser.add_argument("--scenarios", type=str, default="test,stress_test")
    return parser


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def scenario_dataset_config(cfg: dict[str, Any], scenario: str) -> DatasetConfig:
    sim_cfg = cfg["simulation"]
    if scenario == "stress_test":
        sim_cfg = deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    return parse_dataset_config(sim_cfg)


def open_loop_predictions(arrays, dataset_cfg: DatasetConfig) -> np.ndarray:
    if arrays.x0_estimates is None:
        raise ValueError("x0_estimates are required for open-loop propagation diagnostics.")
    dyn = dataset_cfg.dynamics
    preds = np.zeros_like(arrays.states)
    for idx, x0_est in enumerate(arrays.x0_estimates):
        preds[idx] = propagate_orbit(
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
    return preds


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["data"]["output_dir"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    rows: list[dict[str, Any]] = []
    for scenario in [name.strip() for name in args.scenarios.split(",") if name.strip()]:
        arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
        dataset_cfg = scenario_dataset_config(cfg, scenario)
        open_loop = open_loop_predictions(arrays, dataset_cfg)
        if arrays.ekf_prior is None or arrays.ukf_prior is None or arrays.aukf_prior is None:
            raise ValueError(f"{scenario} is missing stored EKF/UKF/AUKF prior arrays.")
        initial_pos_rmse = float(
            np.sqrt(np.mean(np.sum((arrays.x0_estimates[:, :3] - arrays.states[:, 0, :3]) ** 2, axis=1)))
        )
        open_loop_metric = score_predictions(arrays.states[:, eval_start:], open_loop[:, eval_start:])
        ekf_metric = score_predictions(arrays.states[:, eval_start:], arrays.ekf_prior[:, eval_start:])
        ukf_metric = score_predictions(arrays.states[:, eval_start:], arrays.ukf_prior[:, eval_start:])
        aukf_metric = score_predictions(arrays.states[:, eval_start:], arrays.aukf_prior[:, eval_start:])
        best_filter_rmse = min(
            float(ekf_metric["pos_rmse_m"]),
            float(ukf_metric["pos_rmse_m"]),
            float(aukf_metric["pos_rmse_m"]),
        )
        open_loop_rmse = float(open_loop_metric["pos_rmse_m"])
        rows.append(
            {
                "scenario": scenario,
                "trajectories": int(arrays.states.shape[0]),
                "eval_start_step": int(eval_start),
                "evaluated_steps": int(arrays.states.shape[1] - eval_start),
                "initial_pos_rmse_m": initial_pos_rmse,
                "open_loop_pos_rmse_m": open_loop_rmse,
                "ekf_pos_rmse_m": float(ekf_metric["pos_rmse_m"]),
                "ukf_pos_rmse_m": float(ukf_metric["pos_rmse_m"]),
                "aukf_pos_rmse_m": float(aukf_metric["pos_rmse_m"]),
                "best_filter_pos_rmse_m": best_filter_rmse,
                "best_filter_gain_vs_open_loop_percent": 100.0 * (open_loop_rmse - best_filter_rmse) / open_loop_rmse
                if open_loop_rmse > 0.0
                else float("nan"),
            }
        )
        np.savez_compressed(output_dir / f"{scenario}_open_loop_predictions.npz", open_loop=open_loop)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "propagation_baseline_summary.csv", index=False)
    (output_dir / "propagation_baseline_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(output_dir / "propagation_baseline_summary.csv"), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
