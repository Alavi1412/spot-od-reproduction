#!/usr/bin/env python
"""Run a measurement-corruption robustness sweep on the test set."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import (
    parse_baseline_config,
    relative_improvement_percent,
    run_filter_baselines,
    run_model_inference,
    score_predictions,
)
from gnn_state_estimation.models import TemporalGraphEstimator
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import dump_json, load_yaml


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_list_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_model(
    checkpoint: Path,
    train_cfg,
    use_prior_default: bool,
    device: torch.device,
) -> TemporalGraphEstimator:
    ckpt = torch.load(checkpoint, map_location=device)
    model_kwargs = ckpt.get("model_kwargs", {})
    use_prior = bool(ckpt.get("use_ekf_prior", use_prior_default))
    model = TemporalGraphEstimator(
        hidden_dim=train_cfg.hidden_dim,
        gnn_layers=train_cfg.gnn_layers,
        gru_layers=train_cfg.gru_layers,
        dropout=train_cfg.dropout,
        use_ekf_prior=use_prior,
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def inject_measurement_corruption(
    measurements_true: np.ndarray,
    visibility: np.ndarray,
    std_vector: np.ndarray,
    outlier_prob: float,
    outlier_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(0.0, std_vector, size=measurements_true.shape)
    if outlier_prob > 0.0:
        outlier_mask = rng.uniform(size=visibility.shape) < outlier_prob
        noise[outlier_mask] *= outlier_scale
    meas = measurements_true + noise
    meas[..., 1] = np.mod(meas[..., 1], 2.0 * np.pi)
    meas[..., 2] = np.clip(meas[..., 2], -0.5 * np.pi, 0.5 * np.pi)
    meas *= visibility[..., None]
    return meas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--gnn-checkpoint", type=str, default=None)
    p.add_argument("--hybrid-checkpoint", type=str, default=None)
    p.add_argument("--range-scales", type=str, default="0.7,1.0,1.4,1.8")
    p.add_argument("--outlier-probs", type=str, default="0.00,0.01,0.03,0.05")
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--max-trajectories", type=int, default=0)
    p.add_argument("--repeats-per-setting", type=int, default=3)
    p.add_argument("--output-csv", type=str, default="results/robustness/robustness_grid.csv")
    p.add_argument("--summary-json", type=str, default="results/robustness/robustness_summary.json")
    p.add_argument("--figure-dir", type=str, default="results/robustness/figures")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--skip-hybrid", action="store_true")
    return p


def select_subset_indices(n_total: int, max_trajectories: int, seed: int) -> np.ndarray:
    if max_trajectories <= 0 or max_trajectories >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=int(max_trajectories), replace=False)
    return np.sort(idx.astype(np.int64))


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    gnn_ckpt = Path(args.gnn_checkpoint) if args.gnn_checkpoint else ckpt_dir / "best_gnn.pt"
    hybrid_ckpt = Path(args.hybrid_checkpoint) if args.hybrid_checkpoint else ckpt_dir / "best_hybrid.pt"

    models: dict[str, TemporalGraphEstimator] = {}
    if not args.skip_gnn and gnn_ckpt.exists():
        models["GNN"] = load_model(gnn_ckpt, train_cfg, use_prior_default=False, device=device)
    if not args.skip_hybrid and hybrid_ckpt.exists():
        models["HybridGNN"] = load_model(hybrid_ckpt, train_cfg, use_prior_default=True, device=device)

    data_dir = Path(cfg["data"]["output_dir"])
    arr = load_dataset_npz(data_dir / "test.npz")
    raw = np.load(data_dir / "test.npz")
    measurements_true = raw["measurements_true"]
    states = arr.states
    visibility = arr.visibility
    times = arr.times
    x0_estimates = arr.x0_estimates

    n_total = states.shape[0]
    subset_idx = select_subset_indices(n_total=n_total, max_trajectories=int(args.max_trajectories), seed=int(args.seed) + 53)
    n_use = int(subset_idx.size)
    if n_use < n_total:
        states = states[subset_idx]
        measurements_true = measurements_true[subset_idx]
        visibility = visibility[subset_idx]
        times = times[subset_idx]
        if x0_estimates is not None:
            x0_estimates = x0_estimates[subset_idx]

    base_sim_cfg = cfg["simulation"]
    base_noise = base_sim_cfg["measurement_noise"]
    base_std = np.array(
        [
            float(base_noise["range_std_m"]),
            np.deg2rad(float(base_noise["az_std_deg"])),
            np.deg2rad(float(base_noise["el_std_deg"])),
            float(base_noise["range_rate_std_mps"]),
        ],
        dtype=np.float64,
    )
    outlier_scale = float(base_noise["outlier_scale"])

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    range_scales = parse_list_floats(args.range_scales)
    outlier_probs = parse_list_floats(args.outlier_probs)
    rng_master = np.random.default_rng(args.seed)

    rows: list[dict[str, float | str]] = []
    setting_id = 0
    for rs in range_scales:
        for op in outlier_probs:
            setting_id += 1
            print(
                f"Setting {setting_id}/{len(range_scales) * len(outlier_probs)}: "
                f"scale={rs:.2f}, outlier_prob={op:.3f}, repeats={args.repeats_per_setting}"
            )
            for repeat_id in range(int(args.repeats_per_setting)):
                seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                std_vec = base_std * rs
                noisy_meas = inject_measurement_corruption(
                    measurements_true=measurements_true,
                    visibility=visibility,
                    std_vector=std_vec,
                    outlier_prob=op,
                    outlier_scale=outlier_scale,
                    rng=rng,
                )

                sim_cfg = deep_update(
                    base_sim_cfg,
                    {
                        "measurement_noise": {
                            "range_std_m": float(base_noise["range_std_m"]) * rs,
                            "az_std_deg": float(base_noise["az_std_deg"]) * rs,
                            "el_std_deg": float(base_noise["el_std_deg"]) * rs,
                            "range_rate_std_mps": float(base_noise["range_rate_std_mps"]) * rs,
                            "outlier_prob": op,
                            "outlier_scale": outlier_scale,
                        }
                    },
                )
                ds_cfg = parse_dataset_config(sim_cfg)

                filters = run_filter_baselines(
                    states=states,
                    measurements=noisy_meas,
                    visibility=visibility,
                    times=times,
                    dataset_cfg=ds_cfg,
                    baseline_cfg=baseline_cfg,
                    seed=seed + 7,
                    x0_estimates=x0_estimates,
                )
                preds: dict[str, np.ndarray] = {"EKF": filters["ekf"], "UKF": filters["ukf"]}
                if "aukf" in filters:
                    preds["AUKF"] = filters["aukf"]
                if "GNN" in models:
                    preds["GNN"] = run_model_inference(
                        model=models["GNN"],
                        states=states,
                        measurements=noisy_meas,
                        visibility=visibility,
                        station_ecef=arr.station_ecef,
                        window_size=train_cfg.window_size,
                    )
                if "HybridGNN" in models:
                    preds["HybridGNN"] = run_model_inference(
                        model=models["HybridGNN"],
                        states=states,
                        measurements=noisy_meas,
                        visibility=visibility,
                        station_ecef=arr.station_ecef,
                        window_size=train_cfg.window_size,
                        ekf_prior=filters["ekf"],
                    )

                metric_by_method: dict[str, dict[str, float]] = {}
                for method, pred in preds.items():
                    metric_by_method[method] = score_predictions(states[:, eval_start:], pred[:, eval_start:])

                ukf_pos = metric_by_method["UKF"]["pos_rmse_m"]
                for method, metric in metric_by_method.items():
                    imp = 0.0 if method == "UKF" else relative_improvement_percent(ukf_pos, metric["pos_rmse_m"])
                    rows.append(
                        {
                            "range_scale": rs,
                            "outlier_prob": op,
                            "setting_id": setting_id,
                            "repeat_id": repeat_id,
                            "seed": seed,
                            "method": method,
                            "pos_rmse_m": float(metric["pos_rmse_m"]),
                            "vel_rmse_mps": float(metric["vel_rmse_mps"]),
                            "improvement_vs_ukf_percent": float(imp),
                        }
                    )

    df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = {}

    def summarize_method(method: str) -> dict[str, float] | None:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            return None
        setting_mean = (
            sub.groupby(["range_scale", "outlier_prob"], as_index=False)["improvement_vs_ukf_percent"]
            .mean()
            .rename(columns={"improvement_vs_ukf_percent": "setting_mean_improvement_vs_ukf_percent"})
        )
        vals = setting_mean["setting_mean_improvement_vs_ukf_percent"].to_numpy(dtype=np.float64)
        return {
            "num_settings": int(setting_mean.shape[0]),
            "mean_improvement_vs_ukf_percent": float(np.mean(vals)),
            "std_improvement_vs_ukf_percent": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "min_improvement_vs_ukf_percent": float(np.min(vals)),
            "max_improvement_vs_ukf_percent": float(np.max(vals)),
            "fraction_settings_better_than_ukf": float(np.mean(vals > 0.0)),
        }

    hyb_summary = summarize_method("HybridGNN")
    if hyb_summary is not None:
        summary["hybrid"] = hyb_summary
    ekf_summary = summarize_method("EKF")
    if ekf_summary is not None:
        summary["ekf"] = ekf_summary
    aukf_summary = summarize_method("AUKF")
    if aukf_summary is not None:
        summary["aukf"] = aukf_summary
    summary["grid"] = {"range_scales": range_scales, "outlier_probs": outlier_probs}
    summary["num_trajectories"] = int(n_use)
    summary["trajectory_subset_indices"] = [int(i) for i in subset_idx.tolist()]
    summary["repeats_per_setting"] = int(args.repeats_per_setting)
    dump_json(summary, Path(args.summary_json))

    fig_dir = Path(args.figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    def save_heatmap(method: str, filename: str) -> None:
        sub = df[df["method"] == method]
        if sub.empty:
            return
        sub_mean = sub.groupby(["outlier_prob", "range_scale"], as_index=False)["improvement_vs_ukf_percent"].mean()
        pivot = sub_mean.pivot(index="outlier_prob", columns="range_scale", values="improvement_vs_ukf_percent")
        plt.figure(figsize=(6.8, 5.3))
        sns.heatmap(pivot.sort_index(), annot=True, fmt=".2f", cmap="RdYlGn", center=0.0, cbar_kws={"label": "Improvement vs UKF [%]"})
        plt.xlabel("Range/angle/range-rate noise scale")
        plt.ylabel("Outlier probability")
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=220)
        plt.close()

    save_heatmap("HybridGNN", "hybrid_vs_ukf_heatmap.png")
    save_heatmap("EKF", "ekf_vs_ukf_heatmap.png")
    save_heatmap("AUKF", "aukf_vs_ukf_heatmap.png")

    line_df = df[df["method"].isin(["EKF", "UKF", "AUKF", "HybridGNN", "GNN"])].copy()
    if not line_df.empty:
        line_df["noise_outlier_index"] = line_df["range_scale"] + 10.0 * line_df["outlier_prob"]
        plt.figure(figsize=(8.5, 5.1))
        sns.lineplot(
            data=line_df,
            x="noise_outlier_index",
            y="pos_rmse_m",
            hue="method",
            style="outlier_prob",
            markers=True,
            dashes=False,
        )
        plt.xlabel("Corruption index (range-scale + 10*outlier-prob)")
        plt.ylabel("Position RMSE [m]")
        plt.tight_layout()
        plt.savefig(fig_dir / "robustness_profile.png", dpi=220)
        plt.close()

    print("Robustness sweep complete.")


if __name__ == "__main__":
    main()
