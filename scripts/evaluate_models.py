#!/usr/bin/env python
"""Evaluate EKF/UKF/GNN/Hybrid on nominal and stress test sets."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import binomtest, norm, wilcoxon

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import (
    apply_logvar_shift,
    build_innovation_features,
    build_prior_bank_feature_array,
    build_scorecard,
    bootstrap_mean_ci,
    compute_method_activity,
    fit_logvar_shift,
    metric_entry_diverged,
    paired_bootstrap_mean_diff_ci,
    parse_baseline_config,
    relative_improvement_percent,
    run_filter_baselines,
    run_model_inference,
    score_predictions,
    trajectory_rmse,
)
from gnn_state_estimation.models import TemporalGraphEstimator
from gnn_state_estimation.scenarios import estimator_sim_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.runtime import build_run_manifest, duration_metadata, resolve_device, utc_now_iso


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _default_if_none(arg: str | None, fallback: str) -> str:
    return arg if arg is not None else fallback


def scenario_sim_config(base_sim: dict[str, Any], scenario_cfg: dict[str, Any]) -> dict[str, Any]:
    # Evaluation runs the recursive-filter baselines, innovation features, and
    # prior banks, so it must use the estimator-side config (== truth config
    # unless the scenario declares estimator_overrides).
    return estimator_sim_config(base_sim, scenario_cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--gnn-checkpoint", type=str, default=None)
    p.add_argument("--hybrid-checkpoint", type=str, default=None)
    p.add_argument("--innovation-hybrid-checkpoint", type=str, default=None)
    p.add_argument("--metrics-path", type=str, default=None)
    p.add_argument("--per-step-path", type=str, default=None)
    p.add_argument("--trajectory-path", type=str, default=None)
    p.add_argument("--improvement-path", type=str, default=None)
    p.add_argument("--calibration-path", type=str, default=None)
    p.add_argument("--predictions-path", type=str, default=None)
    p.add_argument("--figure-dir", type=str, default=None)
    p.add_argument("--scorecard-out", type=str, default=None)
    p.add_argument("--baseline-cache-dir", type=str, default="results/baseline_cache")
    p.add_argument("--scenarios", type=str, default=None)
    p.add_argument("--disable-uncertainty-calibration", action="store_true")
    p.add_argument("--calibration-bootstrap-samples", type=int, default=1000)
    p.add_argument("--calibration-bootstrap-ci-percent", type=float, default=95.0)
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--skip-hybrid", action="store_true")
    p.add_argument("--skip-innovation-hybrid", action="store_true")
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated model names to evaluate. Classical baselines are always evaluated.",
    )
    return p


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


def per_step_position_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true[..., :3] - y_pred[..., :3]) ** 2, axis=(0, 2)))


def visibility_bucket_rmse(
    states: np.ndarray,
    pred: np.ndarray,
    visibility: np.ndarray,
) -> dict[str, float]:
    vis_count = visibility.sum(axis=2)
    pos_err = np.linalg.norm(pred[..., :3] - states[..., :3], axis=-1)
    out = {}
    buckets = {
        "vis_0": vis_count == 0,
        "vis_1": vis_count == 1,
        "vis_2plus": vis_count >= 2,
    }
    for name, mask in buckets.items():
        if np.any(mask):
            out[f"{name}_pos_rmse_m"] = float(np.sqrt(np.mean(pos_err[mask] ** 2)))
            out[f"{name}_count"] = int(np.sum(mask))
        else:
            out[f"{name}_pos_rmse_m"] = float("nan")
            out[f"{name}_count"] = 0
    return out


def paired_improvement_summary(
    baseline_traj: np.ndarray,
    candidate_traj: np.ndarray,
    seed: int = 123,
) -> tuple[dict[str, float | int | dict[str, float]], np.ndarray]:
    diff = baseline_traj - candidate_traj
    n = int(diff.size)
    wins = int(np.sum(diff > 0.0))
    losses = int(np.sum(diff < 0.0))
    ties = int(n - wins - losses)
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohens_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0
    ci = paired_bootstrap_mean_diff_ci(a=baseline_traj, b=candidate_traj, seed=seed)
    try:
        stat, p_w = wilcoxon(baseline_traj, candidate_traj, alternative="greater")
        stat_f = float(stat)
        p_w_f = float(p_w)
    except ValueError:
        stat_f = float("nan")
        p_w_f = float("nan")
    p_sign = float(binomtest(k=wins, n=n, p=0.5, alternative="greater").pvalue) if n > 0 else float("nan")
    summary = {
        "n_trajectories": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_percent": float(100.0 * wins / max(n, 1)),
        "mean_improvement_m": mean_diff,
        "median_improvement_m": median_diff,
        "cohens_dz": cohens_dz,
        "mean_improvement_m_bootstrap_ci": ci,
        "wilcoxon_greater_stat": stat_f,
        "wilcoxon_greater_pvalue": p_w_f,
        "binomial_sign_pvalue": p_sign,
    }
    return summary, diff


def divergence_summary(traj_pos_rmse: np.ndarray, traj_vel_rmse: np.ndarray) -> dict[str, Any]:
    traj_pos = np.asarray(traj_pos_rmse, dtype=np.float64).reshape(-1)
    traj_vel = np.asarray(traj_vel_rmse, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(traj_pos) & np.isfinite(traj_vel)
    finite_pos = traj_pos[finite_mask]
    finite_vel = traj_vel[finite_mask]
    median_pos = float(np.median(finite_pos)) if finite_pos.size else float("nan")
    median_vel = float(np.median(finite_vel)) if finite_vel.size else float("nan")
    max_pos = float(np.max(finite_pos)) if finite_pos.size else float("inf")
    max_vel = float(np.max(finite_vel)) if finite_vel.size else float("inf")
    ratio_pos = float(max_pos / max(median_pos, 1e-9)) if np.isfinite(max_pos) and np.isfinite(median_pos) else float("inf")
    ratio_vel = float(max_vel / max(median_vel, 1e-9)) if np.isfinite(max_vel) and np.isfinite(median_vel) else float("inf")
    extreme_mask = (~finite_mask) | (np.abs(traj_pos) > 1.0e8) | (np.abs(traj_vel) > 1.0e5)
    diverged = bool(
        np.any(extreme_mask)
        or (np.isfinite(ratio_pos) and ratio_pos > 1.0e4)
        or (np.isfinite(ratio_vel) and ratio_vel > 1.0e4)
    )
    reasons: list[str] = []
    if np.any(~finite_mask):
        reasons.append("nonfinite_trajectory_rmse")
    if np.any((np.abs(traj_pos) > 1.0e8) | (np.abs(traj_vel) > 1.0e5)):
        reasons.append("extreme_trajectory_rmse")
    if np.isfinite(ratio_pos) and ratio_pos > 1.0e4:
        reasons.append("position_rmse_outlier_ratio")
    if np.isfinite(ratio_vel) and ratio_vel > 1.0e4:
        reasons.append("velocity_rmse_outlier_ratio")
    return {
        "diverged": diverged,
        "divergence_reason": ",".join(reasons) if reasons else "",
        "num_diverged_trajectories": int(np.sum(extreme_mask)),
        "max_traj_pos_rmse_m": max_pos,
        "median_traj_pos_rmse_m": median_pos,
        "max_traj_vel_rmse_mps": max_vel,
        "median_traj_vel_rmse_mps": median_vel,
        "max_to_median_traj_pos_rmse_ratio": ratio_pos,
        "max_to_median_traj_vel_rmse_ratio": ratio_vel,
    }


def uncertainty_diagnostics(
    states: np.ndarray,
    pred: np.ndarray,
    logvar: np.ndarray,
    n_bootstrap: int,
    ci_percent: float,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    # Analyze per-component calibration for position channels.
    pos_err = pred[..., :3] - states[..., :3]
    pos_logvar = logvar[..., :3]
    valid = np.isfinite(pos_logvar)
    if not np.any(valid):
        return {}, pd.DataFrame(columns=["nominal_coverage", "empirical_coverage"])

    def flatten_stats(err_arr: np.ndarray, lv_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ok = np.isfinite(lv_arr) & np.isfinite(err_arr)
        if not np.any(ok):
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        err_flat = err_arr[ok]
        lv_flat = lv_arr[ok]
        var_flat = np.exp(np.clip(lv_flat, -30.0, 60.0))
        std_flat = np.sqrt(var_flat)
        z_flat = np.abs(err_flat) / np.clip(std_flat, 1e-9, None)
        nll_flat = 0.5 * (np.log(2.0 * np.pi) + np.log(var_flat) + (err_flat**2) / var_flat)
        return z_flat.astype(np.float64), nll_flat.astype(np.float64), std_flat.astype(np.float64)

    z, nll, std = flatten_stats(pos_err, pos_logvar)
    if z.size == 0:
        return {}, pd.DataFrame(columns=["nominal_coverage", "empirical_coverage"])

    quantiles = [0.50, 0.68, 0.80, 0.90, 0.95]
    q_arr = np.asarray(quantiles, dtype=np.float64)
    thresholds = np.asarray([float(norm.ppf((1.0 + q) / 2.0)) for q in quantiles], dtype=np.float64)
    rows = []
    abs_gaps = []
    for q, thr in zip(quantiles, thresholds, strict=True):
        emp = float(np.mean(z <= thr))
        rows.append(
            {
                "nominal_coverage": q,
                "empirical_coverage": emp,
            }
        )
        abs_gaps.append(abs(emp - q))

    alpha = (100.0 - float(ci_percent)) / 2.0
    boot_cov68 = np.array([], dtype=np.float64)
    boot_cov95 = np.array([], dtype=np.float64)
    boot_ece = np.array([], dtype=np.float64)
    boot_nll = np.array([], dtype=np.float64)
    boot_sigma = np.array([], dtype=np.float64)
    boot_cov_quantiles = np.empty((0, len(quantiles)), dtype=np.float64)
    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        n_traj = int(pos_err.shape[0])
        boot_cov68 = np.empty(n_bootstrap, dtype=np.float64)
        boot_cov95 = np.empty(n_bootstrap, dtype=np.float64)
        boot_ece = np.empty(n_bootstrap, dtype=np.float64)
        boot_nll = np.empty(n_bootstrap, dtype=np.float64)
        boot_sigma = np.empty(n_bootstrap, dtype=np.float64)
        boot_cov_quantiles = np.empty((n_bootstrap, len(quantiles)), dtype=np.float64)
        for i in range(n_bootstrap):
            traj_idx = rng.integers(0, n_traj, size=n_traj)
            z_b, nll_b, std_b = flatten_stats(pos_err[traj_idx], pos_logvar[traj_idx])
            if z_b.size == 0:
                boot_cov_quantiles[i] = np.nan
                boot_cov68[i] = np.nan
                boot_cov95[i] = np.nan
                boot_ece[i] = np.nan
                boot_nll[i] = np.nan
                boot_sigma[i] = np.nan
                continue
            covs = np.asarray([np.mean(z_b <= thr) for thr in thresholds], dtype=np.float64)
            boot_cov_quantiles[i] = covs
            boot_cov68[i] = float(np.mean(z_b <= 1.0))
            boot_cov95[i] = float(np.mean(z_b <= 1.96))
            boot_ece[i] = float(np.mean(np.abs(covs - q_arr)))
            boot_nll[i] = float(np.mean(nll_b))
            boot_sigma[i] = float(np.mean(std_b))

        for j, row in enumerate(rows):
            finite = boot_cov_quantiles[:, j]
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                row["empirical_coverage_ci_low"] = float("nan")
                row["empirical_coverage_ci_high"] = float("nan")
            else:
                row["empirical_coverage_ci_low"] = float(np.percentile(finite, alpha))
                row["empirical_coverage_ci_high"] = float(np.percentile(finite, 100.0 - alpha))

    def ci_dict(point: float, samples: np.ndarray) -> dict[str, float]:
        finite = samples[np.isfinite(samples)]
        if finite.size == 0:
            return {"mean": float(point), "ci_low": float("nan"), "ci_high": float("nan")}
        return {
            "mean": float(point),
            "ci_low": float(np.percentile(finite, alpha)),
            "ci_high": float(np.percentile(finite, 100.0 - alpha)),
        }

    cov68 = float(np.mean(z <= 1.0))
    cov95 = float(np.mean(z <= 1.96))
    ece = float(np.mean(abs_gaps))
    nll_mean = float(np.mean(nll))
    sigma_mean = float(np.mean(std))
    metrics = {
        "pos_uncertainty_nll": nll_mean,
        "pos_uncertainty_sigma_mean_m": sigma_mean,
        "pos_uncertainty_cov68": cov68,
        "pos_uncertainty_cov95": cov95,
        "pos_uncertainty_ece": ece,
        "pos_uncertainty_nll_bootstrap_ci": ci_dict(nll_mean, boot_nll),
        "pos_uncertainty_sigma_mean_m_bootstrap_ci": ci_dict(sigma_mean, boot_sigma),
        "pos_uncertainty_cov68_bootstrap_ci": ci_dict(cov68, boot_cov68),
        "pos_uncertainty_cov95_bootstrap_ci": ci_dict(cov95, boot_cov95),
        "pos_uncertainty_ece_bootstrap_ci": ci_dict(ece, boot_ece),
        "pos_uncertainty_bootstrap_samples": int(max(n_bootstrap, 0)),
        "pos_uncertainty_bootstrap_ci_percent": float(ci_percent),
    }
    return metrics, pd.DataFrame(rows)


def make_plots(
    error_df: pd.DataFrame,
    per_step_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # ECDF by method and scenario.
    g = sns.FacetGrid(error_df, col="scenario", hue="method", height=4.2, aspect=1.15, sharex=True, sharey=True)
    g.map_dataframe(sns.ecdfplot, x="pos_error_m")
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_xscale("log")
        ax.set_xlabel("Instantaneous position error [m] (log-scale)")
        ax.set_ylabel("ECDF")
    g.fig.tight_layout()
    g.savefig(fig_dir / "position_error_ecdf.png", dpi=220)
    plt.close(g.fig)

    # Per-step profile.
    g = sns.FacetGrid(per_step_df, col="scenario", hue="method", height=4.1, aspect=1.2, sharey=False)
    g.map_dataframe(sns.lineplot, x="step", y="pos_rmse_m")
    g.add_legend()
    for ax in g.axes.flat:
        ax.set_xlabel("Time step")
        ax.set_ylabel("Position RMSE [m]")
    g.fig.tight_layout()
    g.savefig(fig_dir / "per_step_rmse.png", dpi=220)
    plt.close(g.fig)

    # Boxplot summary (sampled for speed).
    sample_df = error_df.sample(min(len(error_df), 24000), random_state=42)
    plt.figure(figsize=(8.8, 5.2))
    sns.boxplot(data=sample_df, x="method", y="pos_error_m", hue="scenario")
    plt.yscale("log")
    plt.xlabel("")
    plt.ylabel("Position error [m] (log-scale)")
    plt.tight_layout()
    plt.savefig(fig_dir / "position_error_boxplot.png", dpi=220)
    plt.close()

    # Visibility-conditioned RMSE bars.
    vis_df = (
        error_df.groupby(["scenario", "method", "vis_bucket"], as_index=False)["pos_error_m"]
        .apply(lambda x: np.sqrt(np.mean(np.square(x))))
        .rename(columns={"pos_error_m": "pos_rmse_m"})
    )
    g = sns.catplot(
        data=vis_df,
        kind="bar",
        x="vis_bucket",
        y="pos_rmse_m",
        hue="method",
        col="scenario",
        order=["0", "1", "2+"],
        height=4.1,
        aspect=1.15,
        sharey=False,
    )
    g.set_axis_labels("Visible stations", "Position RMSE [m]")
    g.fig.tight_layout()
    g.savefig(fig_dir / "visibility_bucket_rmse.png", dpi=220)
    plt.close(g.fig)

    # Trajectory-level improvement distribution for hybrid vs UKF.
    if not improvement_df.empty:
        hybrid_imp = improvement_df[improvement_df["comparison"] == "HybridGNN_vs_UKF"].copy()
        if not hybrid_imp.empty:
            plt.figure(figsize=(8.8, 5.2))
            sns.histplot(
                data=hybrid_imp,
                x="delta_pos_rmse_m",
                hue="scenario",
                bins=32,
                element="step",
                stat="density",
                common_norm=False,
            )
            plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
            plt.xlabel("Trajectory improvement [m] (UKF RMSE - Hybrid RMSE)")
            plt.ylabel("Density")
            plt.tight_layout()
            plt.savefig(fig_dir / "hybrid_vs_ukf_improvement_hist.png", dpi=220)
            plt.close()

    # Calibration curve.
    if not calibration_df.empty:
        plt.figure(figsize=(7.2, 6.0))
        sns.lineplot(
            data=calibration_df,
            x="nominal_coverage",
            y="empirical_coverage",
            hue="method",
            style="scenario",
            markers=True,
            dashes=False,
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="Ideal")
        plt.xlim(0.45, 1.0)
        plt.ylim(0.45, 1.0)
        plt.xlabel("Nominal central coverage")
        plt.ylabel("Empirical central coverage")
        plt.tight_layout()
        plt.savefig(fig_dir / "uncertainty_calibration.png", dpi=220)
        plt.close()


def load_or_compute_baselines(
    cache_path: Path,
    states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times: np.ndarray,
    dataset_cfg,
    baseline_cfg,
    seed: int,
    x0_estimates: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], float]:
    if cache_path.exists():
        data = np.load(cache_path)
        cached = {k: data[k] for k in ("ekf", "ukf", "aukf") if k in data.files}
        shape_ok = all(v.shape == states.shape for v in cached.values())
        needs_aukf = baseline_cfg.aukf is not None and "aukf" not in cached
        if "ekf" in cached and "ukf" in cached and not needs_aukf and shape_ok:
            return cached, 0.0

    t0 = time.perf_counter()
    preds = run_filter_baselines(
        states=states,
        measurements=measurements,
        visibility=visibility,
        times=times,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        seed=seed,
        x0_estimates=x0_estimates,
    )
    runtime = time.perf_counter() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **preds)
    return preds, runtime


def scenario_eval(
    scenario_name: str,
    test_npz: Path,
    sim_cfg_dict: dict[str, Any],
    cfg: dict[str, Any],
    models: dict[str, TemporalGraphEstimator],
    train_cfg,
    cache_path: Path,
    logvar_calibration: dict[str, dict[str, float]] | None = None,
    calibration_bootstrap_samples: int = 0,
    calibration_ci_percent: float = 95.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    arrays = load_dataset_npz(test_npz)
    dataset_cfg = parse_dataset_config(sim_cfg_dict)
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    states = arrays.states
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    states_eval = states[:, eval_start:]
    vis_eval = arrays.visibility[:, eval_start:]
    baseline_seed = int(cfg["seed"]) + (17 if scenario_name == "stress_test" else 3)
    filters, baseline_runtime = load_or_compute_baselines(
        cache_path=cache_path,
        states=states,
        measurements=arrays.measurements,
        visibility=arrays.visibility,
        times=arrays.times,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        seed=baseline_seed,
        x0_estimates=arrays.x0_estimates,
    )

    preds: dict[str, np.ndarray] = {"EKF": filters["ekf"], "UKF": filters["ukf"]}
    if "aukf" in filters:
        preds["AUKF"] = filters["aukf"]
    pred_logvars: dict[str, np.ndarray] = {}
    innovation_features_full: np.ndarray | None = None
    if any(getattr(model, "use_innovation_features", False) for model in models.values()):
        if arrays.innovation_features is not None:
            innovation_features_full = arrays.innovation_features
        else:
            innovation_features_full = build_innovation_features(
                dataset_cfg=dataset_cfg,
                measurements=arrays.measurements,
                visibility=arrays.visibility,
                times=arrays.times,
                ekf_prior=filters["ekf"],
            )
    secondary_prior_full: np.ndarray | None = None
    if any(getattr(model, "use_dual_prior_fusion", False) for model in models.values()):
        secondary_prior_full = filters.get("aukf")
        if secondary_prior_full is None:
            raise ValueError("Dual-prior fusion requires AUKF baseline outputs.")
    prior_bank_stats_cache: dict[bool, np.ndarray] = {}
    if any(getattr(model, "use_prior_bank_fusion", False) for model in models.values()):
        if "ukf" not in filters or "aukf" not in filters:
            raise ValueError("Prior-bank fusion requires EKF, UKF, and AUKF outputs.")

    def prior_stats_for_model(model: TemporalGraphEstimator) -> np.ndarray | None:
        if not getattr(model, "use_prior_bank_fusion", False):
            return None
        use_obs = bool(getattr(model, "use_observability_context", False))
        if not use_obs and arrays.prior_bank_stats is not None:
            return arrays.prior_bank_stats
        if use_obs not in prior_bank_stats_cache:
            prior_bank_stats_cache[use_obs] = build_prior_bank_feature_array(
                filters["ekf"],
                filters["ukf"],
                filters["aukf"],
                dataset_cfg=dataset_cfg,
                station_ecef=arrays.station_ecef,
                visibility=arrays.visibility,
                times=arrays.times,
                use_observability_context=use_obs,
            )
        return prior_bank_stats_cache[use_obs]
    runtime_sec = {
        "baseline_cache_build_sec": baseline_runtime,
        "EKF_UKF_cache_build_sec": baseline_runtime,  # backward-compat key
    }

    for model_name, model in models.items():
        t0 = time.perf_counter()
        pred, logvar = run_model_inference(
            model=model,
            states=states,
            measurements=arrays.measurements,
            visibility=arrays.visibility,
            station_ecef=arrays.station_ecef,
            window_size=train_cfg.window_size,
            ekf_prior=filters["ekf"] if (model.use_ekf_prior or getattr(model, "use_prior_bank_fusion", False)) else None,
            ukf_prior=filters.get("ukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            aukf_prior=filters.get("aukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            secondary_prior=secondary_prior_full if model.use_dual_prior_fusion else None,
            innovation_features=innovation_features_full if model.use_innovation_features else None,
            prior_bank_stats=prior_stats_for_model(model),
            return_logvar=True,
        )
        if logvar_calibration and model_name in logvar_calibration:
            c = logvar_calibration[model_name]
            logvar = apply_logvar_shift(
                logvar,
                pos_shift=float(c.get("pos_logvar_shift", 0.0)),
                vel_shift=float(c.get("vel_logvar_shift", 0.0)),
            )
        preds[model_name] = pred
        pred_logvars[model_name] = logvar
        runtime_sec[f"{model_name}_inference_sec"] = time.perf_counter() - t0

    summary: dict[str, Any] = {}
    flat_rows: list[pd.DataFrame] = []
    per_step_rows: list[pd.DataFrame] = []
    traj_rows: list[pd.DataFrame] = []
    calibration_rows: list[pd.DataFrame] = []
    traj_cache: dict[str, np.ndarray] = {}

    vis_count_eval = vis_eval.sum(axis=2).reshape(-1).astype(int)
    vis_bucket_eval = np.where(vis_count_eval == 0, "0", np.where(vis_count_eval == 1, "1", "2+"))

    for method_idx, (method, pred) in enumerate(preds.items()):
        pred_eval = pred[:, eval_start:]
        metric = score_predictions(states_eval, pred_eval)
        metric.update(visibility_bucket_rmse(states=states_eval, pred=pred_eval, visibility=vis_eval))
        if method in pred_logvars:
            lv_eval = pred_logvars[method][:, eval_start:]
            unc_metrics, cal_df = uncertainty_diagnostics(
                states=states_eval,
                pred=pred_eval,
                logvar=lv_eval,
                n_bootstrap=max(int(calibration_bootstrap_samples), 0),
                ci_percent=float(calibration_ci_percent),
                seed=baseline_seed + 1000 + method_idx * 17,
            )
            metric.update(unc_metrics)
            if not cal_df.empty:
                cal_df = cal_df.assign(scenario=scenario_name, method=method)
                calibration_rows.append(cal_df)
        summary[method] = metric

        pos_err = np.linalg.norm(pred_eval[..., :3] - states_eval[..., :3], axis=-1).reshape(-1)
        flat_rows.append(
            pd.DataFrame(
                {
                    "scenario": scenario_name,
                    "method": method,
                    "pos_error_m": pos_err,
                    "vis_count": vis_count_eval,
                    "vis_bucket": vis_bucket_eval,
                }
            )
        )

        rmse_t = per_step_position_rmse(states_eval, pred_eval)
        per_step_rows.append(
            pd.DataFrame(
                {
                    "scenario": scenario_name,
                    "method": method,
                    "step": np.arange(eval_start, eval_start + rmse_t.size),
                    "pos_rmse_m": rmse_t,
                }
            )
        )

        tr = trajectory_rmse(states_eval, pred_eval)
        traj_cache[method] = tr["pos_rmse_m"]
        metric.update(divergence_summary(tr["pos_rmse_m"], tr["vel_rmse_mps"]))
        traj_rows.append(
            pd.DataFrame(
                {
                    "scenario": scenario_name,
                    "method": method,
                    "traj_id": np.arange(tr["pos_rmse_m"].size),
                    "traj_pos_rmse_m": tr["pos_rmse_m"],
                    "traj_vel_rmse_mps": tr["vel_rmse_mps"],
                }
            )
        )

    if "UKF" in summary:
        ukf_pos = float(summary["UKF"]["pos_rmse_m"])
        ukf_valid = not metric_entry_diverged(summary["UKF"])
        for method in summary.keys():
            if method == "UKF":
                continue
            if ukf_valid and not metric_entry_diverged(summary[method]):
                summary[method]["improvement_vs_ukf_pos_rmse_percent"] = relative_improvement_percent(
                    ukf_pos, float(summary[method]["pos_rmse_m"])
                )
            else:
                summary[method]["improvement_vs_ukf_pos_rmse_percent"] = float("nan")

    classical_methods = [m for m in ("EKF", "UKF", "AUKF") if m in summary and not metric_entry_diverged(summary[m])]
    if classical_methods:
        best_classical = min(classical_methods, key=lambda m: float(summary[m]["pos_rmse_m"]))
        best_classical_rmse = float(summary[best_classical]["pos_rmse_m"])
        for method in summary.keys():
            if method == best_classical:
                continue
            if not metric_entry_diverged(summary[method]):
                summary[method]["improvement_vs_best_classical_pos_rmse_percent"] = relative_improvement_percent(
                    best_classical_rmse, float(summary[method]["pos_rmse_m"])
                )
            else:
                summary[method]["improvement_vs_best_classical_pos_rmse_percent"] = float("nan")
        summary["_best_classical_method"] = best_classical
        summary["_best_classical_pos_rmse_m"] = best_classical_rmse

    significance: dict[str, Any] = {}
    improvement_rows: list[pd.DataFrame] = []
    traj_df = pd.concat(traj_rows, ignore_index=True)

    if "UKF" in traj_cache and not metric_entry_diverged(summary["UKF"]):
        ukf_tr = traj_cache["UKF"]
        ukf_ci = bootstrap_mean_ci(ukf_tr, seed=123)
        for method in sorted(m for m in traj_cache.keys() if m != "UKF"):
            if method not in traj_cache:
                continue
            if metric_entry_diverged(summary[method]):
                continue
            method_tr = traj_cache[method]
            summary_stats, delta = paired_improvement_summary(ukf_tr, method_tr, seed=123)
            summary_stats["ukf_pos_rmse_bootstrap"] = ukf_ci
            summary_stats[f"{method.lower()}_pos_rmse_bootstrap"] = bootstrap_mean_ci(method_tr, seed=123)
            key = f"{method.lower()}_vs_ukf"
            significance[key] = summary_stats
            if method == "HybridGNN":
                significance["hybrid_vs_ukf"] = summary_stats
            if method == "InnovationHybridGNN":
                significance["innovation_hybrid_vs_ukf"] = summary_stats
            improvement_rows.append(
                pd.DataFrame(
                    {
                        "scenario": scenario_name,
                        "comparison": f"{method}_vs_UKF",
                        "traj_id": np.arange(delta.size),
                        "delta_pos_rmse_m": delta,
                        "is_improved": delta > 0.0,
                    }
                )
            )

    if "AUKF" in traj_cache and not metric_entry_diverged(summary["AUKF"]):
        aukf_tr = traj_cache["AUKF"]
        for method in sorted(m for m in traj_cache.keys() if m not in {"UKF", "AUKF"}):
            if metric_entry_diverged(summary[method]):
                continue
            method_tr = traj_cache[method]
            s2, delta2 = paired_improvement_summary(aukf_tr, method_tr, seed=321)
            s2["aukf_pos_rmse_bootstrap"] = bootstrap_mean_ci(aukf_tr, seed=321)
            s2[f"{method.lower()}_pos_rmse_bootstrap"] = bootstrap_mean_ci(method_tr, seed=321)
            significance[f"{method.lower()}_vs_aukf"] = s2
            if method == "HybridGNN":
                significance["hybrid_vs_aukf"] = s2
            if method == "InnovationHybridGNN":
                significance["innovation_hybrid_vs_aukf"] = s2
            improvement_rows.append(
                pd.DataFrame(
                    {
                        "scenario": scenario_name,
                        "comparison": f"{method}_vs_AUKF",
                        "traj_id": np.arange(delta2.size),
                        "delta_pos_rmse_m": delta2,
                        "is_improved": delta2 > 0.0,
                    }
                )
            )

    coverage = arrays.visibility.sum(axis=2)
    coverage_stats = {
        "mean_visible_stations_per_step": float(np.mean(coverage)),
        "fraction_steps_zero_visibility": float(np.mean(coverage == 0)),
        "fraction_steps_one_visibility": float(np.mean(coverage == 1)),
        "fraction_steps_two_plus_visibility": float(np.mean(coverage >= 2)),
    }

    scenario_meta = {"runtime_sec": runtime_sec, "coverage": coverage_stats, "significance": significance}
    scenario_meta["evaluation_window"] = {
        "start_step_inclusive": eval_start,
        "evaluated_steps": int(states_eval.shape[1]),
        "total_steps": int(states.shape[1]),
    }
    if logvar_calibration:
        scenario_meta["uncertainty_calibration"] = logvar_calibration
    scenario_meta["method_activity"] = {
        method: compute_method_activity(
            prediction=preds[method],
            ekf_prior=filters["ekf"],
            aukf_prior=filters.get("aukf"),
            eval_start=eval_start,
        )
        for method in preds
        if method not in {"EKF", "UKF", "AUKF"}
    }
    summary["_meta"] = scenario_meta
    scorecard_thresholds = dict(cfg.get("scorecard_thresholds", {}))
    for method in [m for m in preds if m not in {"UKF"}]:
        summary[method]["scorecard"] = build_scorecard(
            summary,
            method_name=method,
            scenario_name=scenario_name,
            thresholds=scorecard_thresholds,
        )

    cal_out = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    imp_out = pd.concat(improvement_rows, ignore_index=True) if improvement_rows else pd.DataFrame()

    return (
        summary,
        pd.concat(flat_rows, ignore_index=True),
        pd.concat(per_step_rows, ignore_index=True),
        traj_df,
        cal_out,
        imp_out,
    )


def fit_uncertainty_calibration(
    cfg: dict[str, Any],
    models: dict[str, TemporalGraphEstimator],
    train_cfg,
    cache_dir: Path,
) -> dict[str, dict[str, float]]:
    if not models:
        return {}
    data_dir = Path(cfg["data"]["output_dir"])
    val_path = data_dir / "val.npz"
    if not val_path.exists():
        return {}

    arrays = load_dataset_npz(val_path)
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    ds_cfg = parse_dataset_config(cfg["simulation"])
    filters, _ = load_or_compute_baselines(
        cache_path=cache_dir / "val_baselines.npz",
        states=arrays.states,
        measurements=arrays.measurements,
        visibility=arrays.visibility,
        times=arrays.times,
        dataset_cfg=ds_cfg,
        baseline_cfg=baseline_cfg,
        seed=int(cfg["seed"]) + 5,
        x0_estimates=arrays.x0_estimates,
    )
    innovation_features = None
    if any(getattr(model, "use_innovation_features", False) for model in models.values()):
        innovation_features = arrays.innovation_features
        if innovation_features is None:
            innovation_features = build_innovation_features(
                dataset_cfg=ds_cfg,
                measurements=arrays.measurements,
                visibility=arrays.visibility,
                times=arrays.times,
                ekf_prior=filters["ekf"],
            )
    secondary_prior: np.ndarray | None = None
    if any(getattr(model, "use_dual_prior_fusion", False) for model in models.values()):
        secondary_prior = filters.get("aukf")
        if secondary_prior is None:
            raise ValueError("Dual-prior fusion requires AUKF baseline outputs.")
    prior_bank_stats_cache: dict[bool, np.ndarray] = {}

    def prior_stats_for_model(model: TemporalGraphEstimator) -> np.ndarray | None:
        if not getattr(model, "use_prior_bank_fusion", False):
            return None
        use_obs = bool(getattr(model, "use_observability_context", False))
        if not use_obs and arrays.prior_bank_stats is not None:
            return arrays.prior_bank_stats
        if use_obs not in prior_bank_stats_cache:
            prior_bank_stats_cache[use_obs] = build_prior_bank_feature_array(
                filters["ekf"],
                filters["ukf"],
                filters["aukf"],
                dataset_cfg=ds_cfg,
                station_ecef=arrays.station_ecef,
                visibility=arrays.visibility,
                times=arrays.times,
                use_observability_context=use_obs,
            )
        return prior_bank_stats_cache[use_obs]

    out: dict[str, dict[str, float]] = {}
    for model_name, model in models.items():
        pred, logvar = run_model_inference(
            model=model,
            states=arrays.states,
            measurements=arrays.measurements,
            visibility=arrays.visibility,
            station_ecef=arrays.station_ecef,
            window_size=train_cfg.window_size,
            ekf_prior=filters["ekf"] if (model.use_ekf_prior or getattr(model, "use_prior_bank_fusion", False)) else None,
            ukf_prior=filters.get("ukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            aukf_prior=filters.get("aukf") if getattr(model, "use_prior_bank_fusion", False) else None,
            secondary_prior=secondary_prior if model.use_dual_prior_fusion else None,
            innovation_features=innovation_features if model.use_innovation_features else None,
            prior_bank_stats=prior_stats_for_model(model),
            return_logvar=True,
        )
        out[model_name] = {
            "pos_logvar_shift": fit_logvar_shift(arrays.states, pred, logvar, slice(0, 3)),
            "vel_logvar_shift": fit_logvar_shift(arrays.states, pred, logvar, slice(3, 6)),
        }
    return out


def main() -> None:
    run_started_at = utc_now_iso()
    run_perf_start = time.perf_counter()
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    train_cfg = parse_train_config(cfg["training"])
    device = resolve_device(args.device or cfg.get("device", {}).get("eval", train_cfg.device))

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    models: dict[str, TemporalGraphEstimator] = {}
    models_cfg = cfg.get("models", {})
    requested_models = None
    if args.models:
        requested_models = {item.strip() for item in args.models.split(",") if item.strip()}
    legacy_overrides = {
        "GNN": args.gnn_checkpoint,
        "HybridGNN": args.hybrid_checkpoint,
        "InnovationHybridGNN": args.innovation_hybrid_checkpoint,
    }
    for model_name, spec in models_cfg.items():
        if not bool(spec.get("enabled", False)):
            continue
        if requested_models is not None and model_name not in requested_models:
            continue
        if model_name == "GNN" and args.skip_gnn:
            continue
        if model_name == "HybridGNN" and args.skip_hybrid:
            continue
        if model_name == "InnovationHybridGNN" and args.skip_innovation_hybrid:
            continue
        ckpt_path = Path(_default_if_none(legacy_overrides.get(model_name), str(ckpt_dir / spec["checkpoint_name"])))
        if ckpt_path.exists():
            models[model_name] = load_model(
                ckpt_path,
                train_cfg,
                use_prior_default=bool(spec.get("use_ekf_prior", False)),
                device=device,
            )

    data_dir = Path(cfg["data"]["output_dir"])
    cache_dir = Path(args.baseline_cache_dir)
    logvar_cal = {}
    if not args.disable_uncertainty_calibration:
        logvar_cal = fit_uncertainty_calibration(cfg, models, train_cfg, cache_dir)
        if logvar_cal:
            print(f"Uncertainty calibration shifts: {logvar_cal}")

    all_scenarios: list[tuple[str, dict[str, Any]]] = [
        ("test", cfg["simulation"]),
        ("stress_test", deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])),
    ]
    for scenario_name, scenario_cfg in cfg.get("benchmark_suite", {}).get("scenarios", {}).items():
        all_scenarios.append((scenario_name, scenario_sim_config(cfg["simulation"], scenario_cfg)))
    if args.scenarios:
        requested = [item.strip() for item in args.scenarios.split(",") if item.strip()]
        requested_set = set(requested)
        scenarios = [(name, sim_cfg) for name, sim_cfg in all_scenarios if name in requested_set]
        if "val" in requested_set:
            scenarios.append(("val", cfg["simulation"]))
        if "stress_val" in requested_set:
            scenarios.append(("stress_val", deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])))
        missing = requested_set.difference({name for name, _ in scenarios})
        if missing:
            raise ValueError(f"Unknown scenarios requested: {sorted(missing)}")
    else:
        scenarios = all_scenarios

    metrics: dict[str, Any] = {}
    all_errors_list: list[pd.DataFrame] = []
    all_steps_list: list[pd.DataFrame] = []
    all_traj_list: list[pd.DataFrame] = []
    all_cal_list: list[pd.DataFrame] = []
    all_imp_list: list[pd.DataFrame] = []
    for scenario_name, scenario_cfg in scenarios:
        summary, errors, steps, traj, cal, imp = scenario_eval(
            scenario_name=scenario_name,
            test_npz=data_dir / f"{scenario_name}.npz",
            sim_cfg_dict=scenario_cfg,
            cfg=cfg,
            models=models,
            train_cfg=train_cfg,
            cache_path=cache_dir / f"{scenario_name}_baselines.npz",
            logvar_calibration=logvar_cal,
            calibration_bootstrap_samples=args.calibration_bootstrap_samples,
            calibration_ci_percent=args.calibration_bootstrap_ci_percent,
        )
        metrics[scenario_name] = summary
        all_errors_list.append(errors)
        all_steps_list.append(steps)
        all_traj_list.append(traj)
        if not cal.empty:
            all_cal_list.append(cal)
        if not imp.empty:
            all_imp_list.append(imp)

    all_errors = pd.concat(all_errors_list, ignore_index=True)
    all_steps = pd.concat(all_steps_list, ignore_index=True)
    all_traj = pd.concat(all_traj_list, ignore_index=True)
    all_cal = pd.concat(all_cal_list, ignore_index=True) if all_cal_list else pd.DataFrame()
    all_imp = pd.concat(all_imp_list, ignore_index=True) if all_imp_list else pd.DataFrame()
    metrics_path = Path(_default_if_none(args.metrics_path, cfg["output"]["metrics_path"]))
    per_step_path = Path(_default_if_none(args.per_step_path, cfg["output"]["per_step_path"]))
    traj_path = Path(
        _default_if_none(
            args.trajectory_path,
            str(Path(cfg["output"]["per_step_path"]).with_name("trajectory_errors.csv")),
        )
    )
    improvement_path = Path(
        _default_if_none(
            args.improvement_path,
            str(Path(cfg["output"]["per_step_path"]).with_name("trajectory_improvement.csv")),
        )
    )
    calibration_path = Path(
        _default_if_none(
            args.calibration_path,
            str(Path(cfg["output"]["per_step_path"]).with_name("uncertainty_calibration.csv")),
        )
    )
    scorecard_path = Path(_default_if_none(args.scorecard_out, cfg["output"].get("scorecard_path", "results/scorecard_summary.json")))
    pred_path = Path(_default_if_none(args.predictions_path, "results/predictions_test.npz"))
    fig_dir = Path(_default_if_none(args.figure_dir, cfg["output"]["figure_dir"]))

    dump_json(metrics, metrics_path)
    per_step_path.parent.mkdir(parents=True, exist_ok=True)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    improvement_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)

    all_steps.to_csv(per_step_path, index=False)
    all_traj.to_csv(traj_path, index=False)
    if not all_imp.empty:
        all_imp.to_csv(improvement_path, index=False)
    else:
        pd.DataFrame(
            columns=["scenario", "comparison", "traj_id", "delta_pos_rmse_m", "is_improved"]
        ).to_csv(improvement_path, index=False)
    if not all_cal.empty:
        all_cal.to_csv(calibration_path, index=False)
    else:
        pd.DataFrame(columns=["nominal_coverage", "empirical_coverage", "scenario", "method"]).to_csv(
            calibration_path, index=False
        )

    make_plots(all_errors, all_steps, all_imp, all_cal, fig_dir)

    scorecard_summary = {
        "_thresholds": dict(cfg.get("scorecard_thresholds", {})),
        **{
            scenario: {
                method: payload.get("scorecard")
                for method, payload in scenario_metrics.items()
                if isinstance(payload, dict) and "scorecard" in payload
            }
            for scenario, scenario_metrics in metrics.items()
        },
    }
    dump_json(scorecard_summary, scorecard_path)

    np.savez_compressed(
        pred_path,
        errors=all_errors.to_records(index=False),
        per_step=all_steps.to_records(index=False),
        trajectories=all_traj.to_records(index=False),
        improvements=(
            all_imp.to_records(index=False)
            if not all_imp.empty
            else pd.DataFrame(
                columns=["scenario", "comparison", "traj_id", "delta_pos_rmse_m", "is_improved"]
            ).to_records(index=False)
        ),
        calibration=(
            all_cal.to_records(index=False)
            if not all_cal.empty
            else pd.DataFrame(columns=["nominal_coverage", "empirical_coverage", "scenario", "method"]).to_records(
                index=False
            )
        ),
    )
    manifest_command = ["evaluate_models.py", "--config", args.config, "--device", str(device)]
    if args.scenarios:
        manifest_command.extend(["--scenarios", args.scenarios])
    build_run_manifest(
        command=manifest_command,
        config_text=Path(args.config).read_text(encoding="utf-8"),
        config_path=args.config,
        output_path=Path(cfg["output"]["manifest_dir"]) / "evaluation.json",
        device=device,
        seed=int(cfg["seed"]),
        dataset_paths={name: data_dir / f"{name}.npz" for name in metrics.keys()},
        extra={"metrics_path": str(metrics_path), "scorecard_path": str(scorecard_path), "selected_scenarios": args.scenarios},
        repo_root=Path(args.config).parent.parent,
        timing=duration_metadata(run_perf_start, started_at_utc=run_started_at),
    )
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
