#!/usr/bin/env python
"""Compute seed-pooled paired diagnostics from learned seed-suite checkpoints."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from gnn_state_estimation.dataset import scale_measurements, scale_state, unscale_state
from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import (
    build_innovation_features,
    build_prior_bank_feature_array,
    parse_baseline_config,
)
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml
from gnn_state_estimation.utils.runtime import resolve_device

try:
    from run_benchmark_seed_sweep import deep_update, load_model_from_checkpoint, load_or_compute_baselines
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts.run_benchmark_seed_sweep import deep_update, load_model_from_checkpoint, load_or_compute_baselines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed-pooled paired significance diagnostic.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-csv", default="results/seed_pooled_significance.csv")
    parser.add_argument("--output-json", default="results/seed_pooled_significance.json")
    parser.add_argument("--distinctness-csv", default="results/seed_suite_distinctness.csv")
    parser.add_argument("--scenario", default="stress_test")
    parser.add_argument("--bootstrap-samples", type=int, default=3000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260518)
    return parser


def trajectory_pos_rmse(states: np.ndarray, pred: np.ndarray, eval_start: int) -> np.ndarray:
    pos_err = np.linalg.norm(pred[:, eval_start:, :3] - states[:, eval_start:, :3], axis=-1)
    return np.sqrt(np.mean(pos_err**2, axis=1))


@torch.no_grad()
def run_model_inference_batched(
    *,
    model,
    states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    station_ecef: np.ndarray,
    window_size: int,
    ekf_prior: np.ndarray | None = None,
    ukf_prior: np.ndarray | None = None,
    aukf_prior: np.ndarray | None = None,
    secondary_prior: np.ndarray | None = None,
    innovation_features: np.ndarray | None = None,
    prior_bank_stats: np.ndarray | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Batched equivalent of run_model_inference for seed-suite audit speed."""
    device = next(model.parameters()).device
    model.eval()
    n_traj, t_steps = states.shape[:2]
    preds = np.full_like(states, np.nan, dtype=np.float64)
    station = (station_ecef / 6_378_137.0).astype(np.float32)
    station_tensor = torch.from_numpy(station).to(device)

    for t in range(window_size - 1, t_steps):
        t0 = t - window_size + 1
        for start in range(0, n_traj, batch_size):
            stop = min(start + batch_size, n_traj)
            sl = slice(start, stop)
            meas = scale_measurements(measurements[sl, t0 : t + 1].astype(np.float32))
            vis = visibility[sl, t0 : t + 1].astype(np.float32)[..., None]
            meas_t = torch.from_numpy(meas).to(device)
            vis_t = torch.from_numpy(vis).to(device)
            prior_t = None
            if model.use_ekf_prior:
                if ekf_prior is None:
                    raise ValueError("Model requires EKF prior but none provided.")
                prior_t = torch.from_numpy(scale_state(ekf_prior[sl, t].astype(np.float32))).to(device)
            secondary_prior_t = None
            if model.use_dual_prior_fusion:
                if secondary_prior is None:
                    raise ValueError("Model requires secondary prior but none was provided.")
                secondary_prior_t = torch.from_numpy(scale_state(secondary_prior[sl, t].astype(np.float32))).to(device)
            innov_t = None
            if model.use_innovation_features:
                if innovation_features is None:
                    raise ValueError("Model requires innovation features but none were provided.")
                innov_t = torch.from_numpy(innovation_features[sl, t0 : t + 1].astype(np.float32)).to(device)
            prior_bank_t = None
            prior_stats_t = None
            if getattr(model, "use_prior_bank_fusion", False):
                if ekf_prior is None or ukf_prior is None or aukf_prior is None:
                    raise ValueError("Prior-bank fusion requires EKF, UKF, and AUKF priors.")
                prior_bank_scaled = np.stack(
                    [
                        scale_state(ekf_prior[sl, t].astype(np.float32)),
                        scale_state(ukf_prior[sl, t].astype(np.float32)),
                        scale_state(aukf_prior[sl, t].astype(np.float32)),
                    ],
                    axis=1,
                )
                prior_bank_t = torch.from_numpy(prior_bank_scaled).to(device)
                if getattr(model, "prior_stats_dim", 0) > 0:
                    if prior_bank_stats is None:
                        raise ValueError("prior_bank_stats are required for prior-bank fusion.")
                    prior_stats_t = torch.from_numpy(prior_bank_stats[sl, t].astype(np.float32)).to(device)

            out = model(
                measurements=meas_t,
                visibility=vis_t,
                station_xyz=station_tensor,
                ekf_prior=prior_t,
                secondary_prior=secondary_prior_t,
                innovation_features=innov_t,
                prior_bank=prior_bank_t,
                prior_bank_stats=prior_stats_t,
            )
            preds[sl, t] = unscale_state(out["state"].detach().cpu().numpy())
    return preds


def infer_candidate_trajectory_rmse(
    *,
    cfg: dict[str, Any],
    model_name: str,
    checkpoint_path: Path,
    arrays,
    dataset_cfg,
    baseline_cfg,
    train_cfg,
    device,
    scenario: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    spec = cfg["models"][model_name]
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        train_cfg=train_cfg,
        spec=spec,
        device=device,
    )
    filters = load_or_compute_baselines(
        cache_path=Path("results/baseline_cache") / f"{scenario}_baselines.npz",
        arrays=arrays,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        seed=int(cfg["seed"]) + 101,
    )
    innovation_features = arrays.innovation_features
    if getattr(model, "use_innovation_features", False) and innovation_features is None:
        innovation_features = build_innovation_features(
            dataset_cfg=dataset_cfg,
            measurements=arrays.measurements,
            visibility=arrays.visibility,
            times=arrays.times,
            ekf_prior=filters["ekf"],
        )
    prior_bank_stats = arrays.prior_bank_stats
    if getattr(model, "use_prior_bank_fusion", False):
        if prior_bank_stats is None or bool(getattr(model, "use_observability_context", False)):
            prior_bank_stats = build_prior_bank_feature_array(
                filters["ekf"],
                filters["ukf"],
                filters["aukf"],
                dataset_cfg=dataset_cfg,
                station_ecef=arrays.station_ecef,
                visibility=arrays.visibility,
                times=arrays.times,
                use_observability_context=bool(getattr(model, "use_observability_context", False)),
            )
    pred = run_model_inference_batched(
        model=model,
        states=arrays.states,
        measurements=arrays.measurements,
        visibility=arrays.visibility,
        station_ecef=arrays.station_ecef,
        window_size=train_cfg.window_size,
        ekf_prior=filters["ekf"] if (model.use_ekf_prior or getattr(model, "use_prior_bank_fusion", False)) else None,
        ukf_prior=filters["ukf"] if getattr(model, "use_prior_bank_fusion", False) else None,
        aukf_prior=filters["aukf"] if getattr(model, "use_prior_bank_fusion", False) else None,
        secondary_prior=filters["aukf"] if getattr(model, "use_dual_prior_fusion", False) else None,
        innovation_features=innovation_features if getattr(model, "use_innovation_features", False) else None,
        prior_bank_stats=prior_bank_stats if getattr(model, "use_prior_bank_fusion", False) else None,
    )
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    return trajectory_pos_rmse(arrays.states, pred, eval_start), filters


def two_level_bootstrap_ci(
    diffs_by_seed: dict[int, np.ndarray],
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    seed_ids = np.asarray(sorted(diffs_by_seed), dtype=int)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sampled_seeds = rng.choice(seed_ids, size=seed_ids.size, replace=True)
        values: list[np.ndarray] = []
        for seed_id in sampled_seeds:
            diffs = diffs_by_seed[int(seed_id)]
            idx = rng.integers(0, diffs.size, size=diffs.size)
            values.append(diffs[idx])
        boot[i] = float(np.mean(np.concatenate(values)))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def format_float(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)


def checkpoint_states_equal(left: Path, right: Path) -> bool:
    left_ckpt = torch.load(left, map_location="cpu")
    right_ckpt = torch.load(right, map_location="cpu")
    if left_ckpt.get("model_kwargs") != right_ckpt.get("model_kwargs"):
        return False
    left_state = left_ckpt["model_state_dict"]
    right_state = right_ckpt["model_state_dict"]
    if set(left_state) != set(right_state):
        return False
    return all(torch.equal(left_state[key], right_state[key]) for key in left_state)


def trajectory_array_provenance(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n_trajectory_rmse": int(arr.size),
        "trajectory_rmse_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "trajectory_rmse_min_m": format_float(float(np.min(arr))) if arr.size else None,
        "trajectory_rmse_mean_m": format_float(float(np.mean(arr))) if arr.size else None,
        "trajectory_rmse_max_m": format_float(float(np.max(arr))) if arr.size else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    scenario = args.scenario
    scenario_cfg = cfg["simulation"]
    if scenario == "stress_test":
        scenario_cfg = deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])
    elif scenario != "test":
        scenario_spec = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
        if scenario_spec is None:
            raise ValueError(f"Unknown scenario {scenario!r}")
        scenario_cfg = deep_update(cfg["simulation"], scenario_spec.get("overrides", {}))

    train_cfg = parse_train_config(cfg["training"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(train_cfg, device=str(cfg.get("device", {}).get("eval", train_cfg.device)))
    device = resolve_device(train_cfg.device)
    data_dir = Path(cfg["data"]["output_dir"])
    arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
    dataset_cfg = parse_dataset_config(scenario_cfg)
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    specs = [
        ("RGR-GF", "HybridGNN", Path("results/seed_suite_hybrid_public/benchmark_seed_metrics.csv")),
    ]
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {"scenario": scenario, "device": str(device), "methods": {}}

    baseline_traj_cache: dict[str, np.ndarray] | None = None
    for method_label, model_name, metrics_path in specs:
        if not metrics_path.exists():
            continue
        metrics_df = pd.read_csv(metrics_path)
        focus = metrics_df[metrics_df["scenario"] == scenario].copy()
        if focus.empty:
            continue
        diffs_by_baseline: dict[str, dict[int, np.ndarray]] = {"UKF": {}, "AUKF": {}}
        method_raw: dict[str, Any] = {}
        for item in focus.sort_values("seed").itertuples(index=False):
            seed_id = int(item.seed)
            checkpoint_path = Path(str(item.checkpoint))
            candidate_traj, filters = infer_candidate_trajectory_rmse(
                cfg=cfg,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                arrays=arrays,
                dataset_cfg=dataset_cfg,
                baseline_cfg=baseline_cfg,
                train_cfg=train_cfg,
                device=device,
                scenario=scenario,
            )
            if baseline_traj_cache is None:
                baseline_traj_cache = {
                    "UKF": trajectory_pos_rmse(arrays.states, filters["ukf"], eval_start),
                    "AUKF": trajectory_pos_rmse(arrays.states, filters["aukf"], eval_start),
                }
            method_raw[str(seed_id)] = {
                "checkpoint": str(checkpoint_path),
                "candidate_traj_pos_rmse_m": candidate_traj.tolist(),
            }
            for baseline_label in ("UKF", "AUKF"):
                diffs_by_baseline[baseline_label][seed_id] = baseline_traj_cache[baseline_label] - candidate_traj

        raw["methods"][method_label] = method_raw
        for baseline_label, diffs_by_seed in diffs_by_baseline.items():
            if not diffs_by_seed:
                continue
            pooled = np.concatenate([diffs_by_seed[seed_id] for seed_id in sorted(diffs_by_seed)])
            ci_low, ci_high = two_level_bootstrap_ci(
                diffs_by_seed,
                seed=int(args.bootstrap_seed) + len(rows),
                n_bootstrap=int(args.bootstrap_samples),
            )
            try:
                stat, p_value = wilcoxon(pooled, alternative="greater")
                stat_f = float(stat)
                p_f = float(p_value)
            except ValueError:
                stat_f = float("nan")
                p_f = float("nan")
            seed_means = np.asarray([float(np.mean(diffs_by_seed[seed_id])) for seed_id in sorted(diffs_by_seed)])
            rows.append(
                {
                    "method": method_label,
                    "scenario": scenario,
                    "comparison": f"vs {baseline_label}",
                    "baseline": baseline_label,
                    "n_seeds": int(len(diffs_by_seed)),
                    "n_seed_trajectory_pairs": int(pooled.size),
                    "mean_gain_m": float(np.mean(pooled)),
                    "two_level_bootstrap_ci_low_m": ci_low,
                    "two_level_bootstrap_ci_high_m": ci_high,
                    "pooled_wilcoxon_stat": stat_f,
                    "pooled_wilcoxon_p": p_f,
                    "pooled_win_rate_percent": float(100.0 * np.mean(pooled > 0.0)),
                    "seed_win_rate_percent": float(100.0 * np.mean(seed_means > 0.0)),
                    "seed_mean_gain_min_m": float(np.min(seed_means)),
                    "seed_mean_gain_mean_m": float(np.mean(seed_means)),
                    "seed_mean_gain_max_m": float(np.max(seed_means)),
                    "bootstrap_type": "two_level_resample_seeds_then_trajectories",
                    "wilcoxon_backend": "scipy.stats.wilcoxon",
                    "wilcoxon_note": "pooled seed-trajectory diagnostic; repeated trajectory identities across seeds are not independent held-out samples",
                }
            )

    valid_rows = [row for row in rows if row.get("method") == "RGR-GF"]
    withdrawn_rows = []
    for row in rows:
        if row.get("method") == "RGR-GF":
            continue
        withdrawn_rows.append(
            {
                "method": row.get("method"),
                "scenario": row.get("scenario"),
                "comparison": row.get("comparison"),
                "baseline": row.get("baseline"),
                "n_seeds": row.get("n_seeds"),
                "n_seed_trajectory_pairs": row.get("n_seed_trajectory_pairs"),
                "withdrawn": True,
                "numeric_summary_removed": True,
                "suppressed_numeric_fields": [
                    "mean_gain_m",
                    "two_level_bootstrap_ci_low_m",
                    "two_level_bootstrap_ci_high_m",
                    "pooled_wilcoxon_stat",
                    "pooled_wilcoxon_p",
                    "pooled_win_rate_percent",
                    "seed_win_rate_percent",
                    "seed_mean_gain_min_m",
                    "seed_mean_gain_mean_m",
                    "seed_mean_gain_max_m",
                ],
                "withdrawal_reason": (
                    "IDP-RGR-GF repeated-seed artifact is not independent: "
                    "distinctness audit shows checkpoint-state and trajectory-RMSE duplication "
                    "with RGR-GF for seeds 41 and 43. Numeric withdrawn summary fields are "
                    "suppressed so the withdrawn result is not transcribable from this JSON."
                ),
                "wilcoxon_note": (
                    "withdrawn diagnostic; in addition to recurring trajectory identities across seeds, "
                    "the distinctness audit shows IDP-RGR-GF duplicates RGR-GF checkpoint state and "
                    "trajectory-RMSE arrays for seeds 41 and 43"
                ),
            }
        )
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(valid_rows).to_csv(out_csv, index=False)
    distinctness_rows = []
    if "RGR-GF" in raw["methods"] and "IDP-RGR-GF" in raw["methods"]:
        common_seeds = sorted(set(raw["methods"]["RGR-GF"]) & set(raw["methods"]["IDP-RGR-GF"]), key=int)
        for seed_text in common_seeds:
            left = raw["methods"]["RGR-GF"][seed_text]
            right = raw["methods"]["IDP-RGR-GF"][seed_text]
            left_arr = np.asarray(left["candidate_traj_pos_rmse_m"], dtype=np.float64)
            right_arr = np.asarray(right["candidate_traj_pos_rmse_m"], dtype=np.float64)
            state_equal = checkpoint_states_equal(Path(left["checkpoint"]), Path(right["checkpoint"]))
            traj_exact = bool(np.array_equal(left_arr, right_arr))
            max_abs_diff = float(np.max(np.abs(left_arr - right_arr))) if left_arr.size else float("nan")
            distinctness_rows.append(
                {
                    "scenario": scenario,
                    "seed": int(seed_text),
                    "left_method": "RGR-GF",
                    "right_method": "IDP-RGR-GF",
                    "left_checkpoint": left["checkpoint"],
                    "right_checkpoint": right["checkpoint"],
                    "model_state_dict_identical": state_equal,
                    "trajectory_rmse_exactly_identical": traj_exact,
                    "trajectory_rmse_max_abs_diff_m": max_abs_diff,
                    "independent_repeated_seed_corollary": bool(not state_equal and not traj_exact),
                }
            )
    distinctness_csv = Path(args.distinctness_csv)
    pd.DataFrame(distinctness_rows).to_csv(distinctness_csv, index=False)
    withdrawn_method_arrays: dict[str, Any] = {}
    if "IDP-RGR-GF" in raw["methods"]:
        withdrawn_seeds = {}
        for seed_text, payload in raw["methods"].pop("IDP-RGR-GF").items():
            withdrawn_seeds[seed_text] = {
                "checkpoint": payload["checkpoint"],
                **trajectory_array_provenance(payload.get("candidate_traj_pos_rmse_m", [])),
            }
        withdrawn_method_arrays["IDP-RGR-GF"] = {
            "withdrawn": True,
            "withdrawal_reason": (
                "Not a canonical recomputation source: distinctness audit shows checkpoint-state "
                "and trajectory-RMSE duplication with RGR-GF for seeds 41 and 43. Raw withdrawn "
                "trajectory arrays are replaced by hashes and summary statistics for provenance."
            ),
            "seeds": withdrawn_seeds,
        }
    raw["rows"] = valid_rows
    raw["withdrawn_rows"] = withdrawn_rows
    raw["withdrawn_method_arrays"] = withdrawn_method_arrays
    raw["distinctness_rows_path"] = str(distinctness_csv)
    raw["distinctness_rows"] = distinctness_rows
    out_json.write_text(json.dumps(raw, indent=2, default=format_float) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(valid_rows),
                "withdrawn_rows": len(withdrawn_rows),
                "csv": str(out_csv),
                "json": str(out_json),
                "distinctness_csv": str(distinctness_csv),
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
