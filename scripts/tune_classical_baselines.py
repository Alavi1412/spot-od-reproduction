#!/usr/bin/env python
"""Tune EKF/UKF/AUKF baselines and emit an auditable search ledger."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from gnn_state_estimation.dataset import DatasetArrays, load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config, score_predictions
from gnn_state_estimation.filters.ekf import run_ekf
from gnn_state_estimation.filters.ukf import run_adaptive_ukf, run_ukf
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.runtime import build_run_manifest, duration_metadata, resolve_device, utc_now_iso


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--output-dir", type=str, default="results/classical_tuning")
    p.add_argument("--device", type=str, default=None)
    return p


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _candidate_grid(values: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(values.keys())
    combos = []
    for row in itertools.product(*(values[key] for key in keys)):
        combos.append({key: value for key, value in zip(keys, row, strict=True)})
    return combos


def _default_search_space(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    ekf = cfg["baselines"]["ekf"]
    ukf = cfg["baselines"]["ukf"]
    aukf = cfg["baselines"]["aukf"]
    return {
        "ekf": _candidate_grid(
            {
                "q_pos_m": [round(float(ekf["q_pos_m"]) * s, 4) for s in (0.8, 1.0, 1.2, 1.4)],
                "q_vel_mps": [round(float(ekf["q_vel_mps"]) * s, 4) for s in (0.8, 1.0)],
                "gating_threshold": [round(float(ekf["gating_threshold"]) + d, 4) for d in (-2.0, 0.0)],
            }
        ),
        "ukf": _candidate_grid(
            {
                "q_pos_m": [round(float(ukf["q_pos_m"]) * s, 4) for s in (0.8, 1.0, 1.2, 1.4)],
                "q_vel_mps": [round(float(ukf["q_vel_mps"]) * s, 4) for s in (0.8, 1.0)],
                "alpha": [0.1, float(ukf["alpha"])],
            }
        ),
        "aukf": _candidate_grid(
            {
                "q_pos_m": [round(float(aukf["q_pos_m"]) * s, 4) for s in (0.8, 1.0)],
                "q_vel_mps": [round(float(aukf["q_vel_mps"]) * s, 4) for s in (0.8, 1.0)],
                "adapt_rate": [0.05, float(aukf["adapt_rate"])],
                "nis_soft_gate": [12.0, float(aukf["nis_soft_gate"])],
            }
        ),
    }


def _configured_search_space(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    classical_tuning = cfg.get("baselines", {}).get("classical_tuning", {})
    search_space = classical_tuning.get("search_space")
    if not isinstance(search_space, dict):
        return _default_search_space(cfg)
    out: dict[str, list[dict[str, Any]]] = {}
    for method in ("ekf", "ukf", "aukf"):
        method_space = search_space.get(method)
        if isinstance(method_space, dict):
            out[method] = _candidate_grid(method_space)
    if not out:
        return _default_search_space(cfg)
    return out


def _subset_arrays(arrays: DatasetArrays, limit: int | None) -> DatasetArrays:
    if limit is None or limit <= 0 or arrays.states.shape[0] <= limit:
        return arrays
    sl = slice(0, int(limit))
    return replace(
        arrays,
        states=arrays.states[sl],
        measurements=arrays.measurements[sl],
        visibility=arrays.visibility[sl],
        times=arrays.times[sl],
        ekf_prior=arrays.ekf_prior[sl] if arrays.ekf_prior is not None else None,
        ukf_prior=arrays.ukf_prior[sl] if arrays.ukf_prior is not None else None,
        aukf_prior=arrays.aukf_prior[sl] if arrays.aukf_prior is not None else None,
        secondary_prior=arrays.secondary_prior[sl] if arrays.secondary_prior is not None else None,
        x0_estimates=arrays.x0_estimates[sl] if arrays.x0_estimates is not None else None,
        innovation_features=arrays.innovation_features[sl] if arrays.innovation_features is not None else None,
        prior_bank_stats=arrays.prior_bank_stats[sl] if arrays.prior_bank_stats is not None else None,
        sample_weight=arrays.sample_weight[sl] if arrays.sample_weight is not None else None,
        regime_name=arrays.regime_name[sl] if arrays.regime_name is not None else None,
    )


def _score_method(
    *,
    method_key: str,
    candidate_cfg: dict[str, Any],
    cfg: dict[str, Any],
    nominal_search_arrays,
    stress_search_arrays,
    nominal_dataset_cfg,
    stress_dataset_cfg,
    eval_start: int,
) -> dict[str, Any]:
    base_cfg = deep_update(cfg["baselines"], {method_key: candidate_cfg})
    baseline_cfg = parse_baseline_config(base_cfg)
    nominal_pred = _run_single_method(
        method_key=method_key,
        arrays=nominal_search_arrays,
        dataset_cfg=nominal_dataset_cfg,
        baseline_cfg=baseline_cfg,
    )
    stress_pred = _run_single_method(
        method_key=method_key,
        arrays=stress_search_arrays,
        dataset_cfg=stress_dataset_cfg,
        baseline_cfg=baseline_cfg,
    )
    nominal_metric = score_predictions(nominal_search_arrays.states[:, eval_start:], nominal_pred[:, eval_start:])
    stress_metric = score_predictions(stress_search_arrays.states[:, eval_start:], stress_pred[:, eval_start:])
    stress_weight = float(cfg.get("benchmark_suite", {}).get("stress_sample_weight", 1.0))
    objective = float(nominal_metric["pos_rmse_m"] + stress_weight * stress_metric["pos_rmse_m"])
    return {
        "method": method_key.upper(),
        "objective": objective,
        "nominal_pos_rmse_m": float(nominal_metric["pos_rmse_m"]),
        "nominal_vel_rmse_mps": float(nominal_metric["vel_rmse_mps"]),
        "stress_pos_rmse_m": float(stress_metric["pos_rmse_m"]),
        "stress_vel_rmse_mps": float(stress_metric["vel_rmse_mps"]),
        "params": candidate_cfg,
    }


def _full_eval_metrics(
    *,
    method_key: str,
    candidate_cfg: dict[str, Any],
    cfg: dict[str, Any],
    nominal_full_arrays,
    stress_full_arrays,
    nominal_dataset_cfg,
    stress_dataset_cfg,
    eval_start: int,
) -> dict[str, float]:
    base_cfg = deep_update(cfg["baselines"], {method_key: candidate_cfg})
    baseline_cfg = parse_baseline_config(base_cfg)
    full_nominal_pred = _run_single_method(
        method_key=method_key,
        arrays=nominal_full_arrays,
        dataset_cfg=nominal_dataset_cfg,
        baseline_cfg=baseline_cfg,
    )
    full_stress_pred = _run_single_method(
        method_key=method_key,
        arrays=stress_full_arrays,
        dataset_cfg=stress_dataset_cfg,
        baseline_cfg=baseline_cfg,
    )
    full_nominal_metric = score_predictions(nominal_full_arrays.states[:, eval_start:], full_nominal_pred[:, eval_start:])
    full_stress_metric = score_predictions(stress_full_arrays.states[:, eval_start:], full_stress_pred[:, eval_start:])
    return {
        "full_nominal_pos_rmse_m": float(full_nominal_metric["pos_rmse_m"]),
        "full_stress_pos_rmse_m": float(full_stress_metric["pos_rmse_m"]),
    }


def _run_single_method(*, method_key: str, arrays, dataset_cfg, baseline_cfg) -> np.ndarray:
    n_traj = arrays.states.shape[0]
    preds = np.zeros_like(arrays.states)
    dyn = dataset_cfg.dynamics
    meas_std = dataset_cfg.measurement_noise.std_vector
    for i in range(n_traj):
        x0_est = arrays.x0_estimates[i]
        common = {
            "measurements": arrays.measurements[i],
            "visibility": arrays.visibility[i],
            "times_s": arrays.times[i],
            "stations": dataset_cfg.stations,
            "ballistic_coeff_m2_per_kg": dyn.ballistic_coeff_m2_per_kg,
            "meas_std_vector": meas_std,
            "x0_est": x0_est,
            "drag_rho_ref": dyn.drag_rho_ref,
            "drag_h_ref_m": dyn.drag_h_ref_m,
            "drag_scale_height_m": dyn.drag_scale_height_m,
            "enable_third_body": dyn.enable_third_body,
            "enable_srp": dyn.enable_srp,
            "srp_area_to_mass_m2_per_kg": dyn.srp_area_to_mass_m2_per_kg,
            "srp_cr": dyn.srp_cr,
            "sun_initial_phase_rad": dyn.sun_initial_phase_rad,
            "moon_initial_phase_rad": dyn.moon_initial_phase_rad,
        }
        if method_key == "ekf":
            pred, _ = run_ekf(cfg=baseline_cfg.ekf, **common)
        elif method_key == "ukf":
            pred, _ = run_ukf(cfg=baseline_cfg.ukf, **common)
        elif method_key == "aukf":
            if baseline_cfg.aukf is None:
                raise ValueError("AUKF tuning requested but no AUKF config is present.")
            pred, _ = run_adaptive_ukf(cfg=baseline_cfg.aukf, **common)
        else:
            raise ValueError(f"Unsupported baseline method: {method_key}")
        preds[i] = pred
    return preds


def main() -> None:
    run_started_at = utc_now_iso()
    run_perf_start = time.perf_counter()
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    cfg_text = cfg_path.read_text(encoding="utf-8")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "classical_tuning_progress.jsonl"
    device = resolve_device(args.device or cfg.get("device", {}).get("eval", "cpu"))

    data_dir = Path(cfg["data"]["output_dir"])
    nominal_arrays = load_dataset_npz(data_dir / "val.npz")
    stress_arrays = load_dataset_npz(data_dir / "stress_val.npz")
    tuning_cfg = cfg.get("baselines", {}).get("classical_tuning", {})
    trajectory_limit = tuning_cfg.get("search_trajectory_limit_per_split")
    nominal_search_arrays = _subset_arrays(nominal_arrays, int(trajectory_limit) if trajectory_limit is not None else None)
    stress_search_arrays = _subset_arrays(stress_arrays, int(trajectory_limit) if trajectory_limit is not None else None)
    nominal_dataset_cfg = parse_dataset_config(cfg["simulation"])
    stress_dataset_cfg = parse_dataset_config(deep_update(cfg["simulation"], cfg["stress_simulation_overrides"]))
    search_space = _configured_search_space(cfg)
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    ledger_rows: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, int]] = set()
    if progress_path.exists():
        deduped_rows: list[dict[str, Any]] = []
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (str(row["method"]).lower(), int(row["candidate_index_within_method"]))
                if key in completed_keys:
                    continue
                completed_keys.add(key)
                deduped_rows.append(row)
        ledger_rows = deduped_rows
        with progress_path.open("w", encoding="utf-8") as f:
            for row in ledger_rows:
                f.write(json.dumps(row) + "\n")
    best_by_method: dict[str, dict[str, Any]] = {}
    trial_counter = len(ledger_rows)
    for method_key in ("ekf", "ukf", "aukf"):
        candidates = search_space.get(method_key, [])
        if not candidates:
            continue
        existing_method_trials = [row for row in ledger_rows if str(row["method"]).lower() == method_key]
        best_trial: dict[str, Any] | None = min(existing_method_trials, key=lambda row: float(row["objective"])) if existing_method_trials else None
        for idx, candidate in enumerate(candidates):
            if (method_key, idx) in completed_keys:
                continue
            trial_started_at = utc_now_iso()
            trial_perf_start = time.perf_counter()
            trial = _score_method(
                method_key=method_key,
                candidate_cfg=candidate,
                cfg=cfg,
                nominal_search_arrays=nominal_search_arrays,
                stress_search_arrays=stress_search_arrays,
                nominal_dataset_cfg=nominal_dataset_cfg,
                stress_dataset_cfg=stress_dataset_cfg,
                eval_start=eval_start,
            )
            trial["trial_index"] = int(trial_counter)
            trial["candidate_index_within_method"] = int(idx)
            trial["timing"] = duration_metadata(trial_perf_start, started_at_utc=trial_started_at)
            ledger_rows.append(trial)
            completed_keys.add((method_key, idx))
            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trial) + "\n")
            print(
                f"[{method_key.upper()} candidate {idx + 1}/{len(candidates)}] "
                f"objective={trial['objective']:.2f} nominal={trial['nominal_pos_rmse_m']:.2f} "
                f"stress={trial['stress_pos_rmse_m']:.2f}"
            )
            if best_trial is None or float(trial["objective"]) < float(best_trial["objective"]):
                best_trial = trial
            trial_counter += 1
        if best_trial is not None:
            best_trial.update(
                _full_eval_metrics(
                    method_key=method_key,
                    candidate_cfg=best_trial["params"],
                    cfg=cfg,
                    nominal_full_arrays=nominal_arrays,
                    stress_full_arrays=stress_arrays,
                    nominal_dataset_cfg=nominal_dataset_cfg,
                    stress_dataset_cfg=stress_dataset_cfg,
                    eval_start=eval_start,
                )
            )
            best_by_method[method_key.upper()] = best_trial

    ledger_rows = sorted(ledger_rows, key=lambda row: (row["method"], row["objective"]))
    ledger_payload = {
        "config_path": str(cfg_path),
        "search_budget_total": int(len(ledger_rows)),
        "search_budget_per_method": {method: int(len(search_space.get(method.lower(), []))) for method in best_by_method},
        "objective": "nominal_val_pos_rmse + stress_weight * stress_val_pos_rmse",
        "stress_weight": float(cfg.get("benchmark_suite", {}).get("stress_sample_weight", 1.0)),
        "best_trials": best_by_method,
        "trials": ledger_rows,
    }
    dump_json(ledger_payload, out_dir / "classical_tuning_summary.json")

    import pandas as pd

    flat_rows = []
    for row in ledger_rows:
        flat_rows.append(
            {
                "trial_index": row["trial_index"],
                "candidate_index_within_method": row["candidate_index_within_method"],
                "method": row["method"],
                "objective": row["objective"],
                "nominal_pos_rmse_m": row["nominal_pos_rmse_m"],
                "nominal_vel_rmse_mps": row["nominal_vel_rmse_mps"],
                "stress_pos_rmse_m": row["stress_pos_rmse_m"],
                "stress_vel_rmse_mps": row["stress_vel_rmse_mps"],
                "full_nominal_pos_rmse_m": row.get("full_nominal_pos_rmse_m", float("nan")),
                "full_stress_pos_rmse_m": row.get("full_stress_pos_rmse_m", float("nan")),
                "params_json": str(row["params"]),
                "duration_sec": row["timing"]["duration_sec"],
            }
        )
    pd.DataFrame(flat_rows).to_csv(out_dir / "classical_tuning_ledger.csv", index=False)

    build_run_manifest(
        command=["tune_classical_baselines.py", "--config", str(cfg_path), "--output-dir", str(out_dir), "--device", str(device)],
        config_text=cfg_text,
        config_path=cfg_path,
        output_path=Path(cfg["output"]["manifest_dir"]) / "classical_tuning.json",
        device=device,
        seed=int(cfg["seed"]),
        dataset_paths={
            "val": data_dir / "val.npz",
            "stress_val": data_dir / "stress_val.npz",
        },
        extra={
            "summary_path": str(out_dir / "classical_tuning_summary.json"),
            "ledger_path": str(out_dir / "classical_tuning_ledger.csv"),
            "search_budget_total": int(len(ledger_rows)),
        },
        repo_root=cfg_path.parent.parent,
        timing=duration_metadata(run_perf_start, started_at_utc=run_started_at),
    )
    print(f"Classical tuning complete. {len(ledger_rows)} trials written to {out_dir}.")


if __name__ == "__main__":
    main()
