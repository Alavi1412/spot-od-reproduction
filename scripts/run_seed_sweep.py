#!/usr/bin/env python
"""Run multi-seed hybrid training/evaluation for robustness reporting."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import (
    parse_baseline_config,
    relative_improvement_percent,
    run_filter_baselines,
    run_model_inference,
    score_predictions,
)
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.seeding import seed_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--seeds", type=str, default="7,13,23,37,42")
    p.add_argument("--output-dir", type=str, default="results/seed_sweep")
    p.add_argument("--baseline-cache-dir", type=str, default="results/baseline_cache")
    return p


def parse_seed_list(seed_str: str) -> list[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def deep_update(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_or_compute_baselines(cfg: dict, baseline_cache_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_dir = Path(cfg["data"]["output_dir"])
    test_arr = load_dataset_npz(data_dir / "test.npz")
    stress_arr = load_dataset_npz(data_dir / "stress_test.npz")
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    test_cache = baseline_cache_dir / "test_baselines.npz"
    stress_cache = baseline_cache_dir / "stress_test_baselines.npz"
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)

    if test_cache.exists():
        d = np.load(test_cache)
        test_pred = {"ekf": d["ekf"], "ukf": d["ukf"]}
        if test_pred["ekf"].shape != test_arr.states.shape or test_pred["ukf"].shape != test_arr.states.shape:
            test_pred = run_filter_baselines(
                states=test_arr.states,
                measurements=test_arr.measurements,
                visibility=test_arr.visibility,
                times=test_arr.times,
                dataset_cfg=parse_dataset_config(cfg["simulation"]),
                baseline_cfg=baseline_cfg,
                seed=int(cfg["seed"]) + 3,
                x0_estimates=test_arr.x0_estimates,
            )
            np.savez_compressed(test_cache, **test_pred)
    else:
        test_pred = run_filter_baselines(
            states=test_arr.states,
            measurements=test_arr.measurements,
            visibility=test_arr.visibility,
            times=test_arr.times,
            dataset_cfg=parse_dataset_config(cfg["simulation"]),
            baseline_cfg=baseline_cfg,
            seed=int(cfg["seed"]) + 3,
            x0_estimates=test_arr.x0_estimates,
        )
        np.savez_compressed(test_cache, **test_pred)

    stress_cfg = deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])
    if stress_cache.exists():
        d = np.load(stress_cache)
        stress_pred = {"ekf": d["ekf"], "ukf": d["ukf"]}
        if stress_pred["ekf"].shape != stress_arr.states.shape or stress_pred["ukf"].shape != stress_arr.states.shape:
            stress_pred = run_filter_baselines(
                states=stress_arr.states,
                measurements=stress_arr.measurements,
                visibility=stress_arr.visibility,
                times=stress_arr.times,
                dataset_cfg=parse_dataset_config(stress_cfg),
                baseline_cfg=baseline_cfg,
                seed=int(cfg["seed"]) + 17,
                x0_estimates=stress_arr.x0_estimates,
            )
            np.savez_compressed(stress_cache, **stress_pred)
    else:
        stress_pred = run_filter_baselines(
            states=stress_arr.states,
            measurements=stress_arr.measurements,
            visibility=stress_arr.visibility,
            times=stress_arr.times,
            dataset_cfg=parse_dataset_config(stress_cfg),
            baseline_cfg=baseline_cfg,
            seed=int(cfg["seed"]) + 17,
            x0_estimates=stress_arr.x0_estimates,
        )
        np.savez_compressed(stress_cache, **stress_pred)

    return test_pred, stress_pred


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    seeds = parse_seed_list(args.seeds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    data_dir = Path(cfg["data"]["output_dir"])
    train_arr = load_dataset_npz(data_dir / "train.npz")
    val_arr = load_dataset_npz(data_dir / "val.npz")
    test_arr = load_dataset_npz(data_dir / "test.npz")
    stress_arr = load_dataset_npz(data_dir / "stress_test.npz")

    test_baseline, stress_baseline = load_or_compute_baselines(cfg, Path(args.baseline_cache_dir))
    ukf_test_pos_rmse = float(
        score_predictions(test_arr.states[:, eval_start:], test_baseline["ukf"][:, eval_start:])["pos_rmse_m"]
    )
    ukf_stress_pos_rmse = float(
        score_predictions(stress_arr.states[:, eval_start:], stress_baseline["ukf"][:, eval_start:])["pos_rmse_m"]
    )

    rows = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        seed_all(seed)
        run_dir = out_dir / f"seed_{seed}"
        model, hist, ckpt = train_model(
            train_arrays=train_arr,
            val_arrays=val_arr,
            cfg=train_cfg,
            output_dir=run_dir,
            seed=seed,
            use_ekf_prior=True,
            model_kwargs={"residual_scale": 0.02, "use_gating": True, "bounded_residual": True},
            dataset_cfg=parse_dataset_config(cfg["simulation"]),
            baseline_cfg=parse_baseline_config(cfg["baselines"]),
        )

        test_pred = run_model_inference(
            model=model,
            states=test_arr.states,
            measurements=test_arr.measurements,
            visibility=test_arr.visibility,
            station_ecef=test_arr.station_ecef,
            window_size=train_cfg.window_size,
            ekf_prior=test_baseline["ekf"],
        )
        stress_pred = run_model_inference(
            model=model,
            states=stress_arr.states,
            measurements=stress_arr.measurements,
            visibility=stress_arr.visibility,
            station_ecef=stress_arr.station_ecef,
            window_size=train_cfg.window_size,
            ekf_prior=stress_baseline["ekf"],
        )

        test_metric = score_predictions(test_arr.states[:, eval_start:], test_pred[:, eval_start:])
        stress_metric = score_predictions(stress_arr.states[:, eval_start:], stress_pred[:, eval_start:])
        test_impr = relative_improvement_percent(ukf_test_pos_rmse, test_metric["pos_rmse_m"])
        stress_impr = relative_improvement_percent(ukf_stress_pos_rmse, stress_metric["pos_rmse_m"])

        rows.append(
            {
                "seed": seed,
                "checkpoint": str(ckpt),
                "test_pos_rmse_m": test_metric["pos_rmse_m"],
                "test_vel_rmse_mps": test_metric["vel_rmse_mps"],
                "stress_pos_rmse_m": stress_metric["pos_rmse_m"],
                "stress_vel_rmse_mps": stress_metric["vel_rmse_mps"],
                "test_improvement_vs_ukf_percent": test_impr,
                "stress_improvement_vs_ukf_percent": stress_impr,
                "epochs_trained": len(hist["train_loss"]),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "seed_sweep_metrics.csv", index=False)

    summary = {
        "seeds": seeds,
        "count": int(len(seeds)),
        "test_pos_rmse_mean": float(df["test_pos_rmse_m"].mean()),
        "test_pos_rmse_std": float(df["test_pos_rmse_m"].std(ddof=1)),
        "stress_pos_rmse_mean": float(df["stress_pos_rmse_m"].mean()),
        "stress_pos_rmse_std": float(df["stress_pos_rmse_m"].std(ddof=1)),
        "test_improvement_vs_ukf_mean_percent": float(df["test_improvement_vs_ukf_percent"].mean()),
        "stress_improvement_vs_ukf_mean_percent": float(df["stress_improvement_vs_ukf_percent"].mean()),
    }
    dump_json(summary, out_dir / "seed_sweep_summary.json")
    print("\nSeed sweep complete.")


if __name__ == "__main__":
    main()
