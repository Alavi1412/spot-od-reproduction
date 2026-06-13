#!/usr/bin/env python
"""Statistically harden the controlled force-model-mismatch slice.

The single-realization force-model-mismatch table reports a point ordering
(causal EKF best observed-step, the tuned AUKF worst, no learned residual
ahead of EKF). A reviewer correctly noted that a point ordering with a
small EKF-vs-learned margin and no uncertainty quantification is not a
defensible discriminative result.

This script adds two paired diagnostics on the controlled-mismatch slice,
reusing the proven seed-suite inference/baseline helpers:

1. Deterministic classical pairs (EKF/UKF/AUKF). The classical filters are
   deterministic on the controlled-mismatch realization, so the paired unit
   is a trajectory's observed-step position RMSE over the trajectory
   population. We report the mean paired gain, a 95% percentile *paired*
   bootstrap CI (resampling trajectory indices), and a one-sided paired
   Wilcoxon signed-rank diagnostic. This is the significance test for the
   "EKF best / AUKF counterproductive under true dynamics mismatch" claim.

2. The canonical 15-seed RGR-GF cohort (the same training-seed cohort that
   carries the headline learned-correction comparison) evaluated on the
   controlled-mismatch slice. For each trained seed we recompute the
   learned residual model's observed-step RMSE and its paired gain versus
   the (deterministic) EKF and AUKF baselines, then report the seed-level
   mean gain with a seed bootstrap CI and seed win rate, and a two-level
   (seeds-then-trajectories) bootstrap CI plus a pooled Wilcoxon
   diagnostic. This replaces "single-seed ~2 m margin" with a 15-seed
   robustness statement on whether the learned residual ever beats EKF
   under controlled force-model mismatch.

No positive is invented: the script reports whatever the paired diagnostics
show. The observed-step convention is identical to the main evaluator.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.utils.io import load_yaml
from gnn_state_estimation.utils.runtime import resolve_device

try:
    from run_benchmark_seed_sweep import deep_update
    from compute_seed_observed_significance import infer_candidate_predictions
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts.run_benchmark_seed_sweep import deep_update
    from scripts.compute_seed_observed_significance import infer_candidate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Force-mismatch paired significance hardening.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scenario", default="force_model_mismatch_test")
    parser.add_argument("--model-name", default="HybridGNN")
    parser.add_argument("--method-label", default="RGR-GF")
    parser.add_argument(
        "--metrics-csv",
        default="results/seed_suite_hybrid_public/benchmark_seed_metrics.csv",
    )
    parser.add_argument("--output-json", default="results/force_mismatch_seed_significance.json")
    parser.add_argument("--output-csv", default="results/force_mismatch_seed_significance.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260519)
    return parser


def per_trajectory_observed_rmse(
    states: np.ndarray,
    pred: np.ndarray,
    visibility: np.ndarray,
    eval_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trajectory observed-step position RMSE.

    A (trajectory, step) pair from ``eval_start`` onward is observed when at
    least one station is visible (summed station visibility >= 0.5), the same
    convention as the pooled main-evaluator observed-step metric. Returns the
    per-trajectory RMSE array (NaN where a trajectory has no observed step) and
    the boolean trajectory mask of trajectories with at least one observed
    step. The observed mask depends only on data visibility, so it is shared
    across every estimator and the pairs stay aligned.
    """
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5  # (n_traj, n_eval_steps)
    y_true = states[:, eval_start:, :3]
    y_pred = pred[:, eval_start:, :3]
    sq = np.sum((y_true - y_pred) ** 2, axis=-1)  # (n_traj, n_eval_steps)
    n_traj = states.shape[0]
    out = np.full(n_traj, np.nan, dtype=np.float64)
    has_obs = observed.any(axis=1)
    for i in range(n_traj):
        if has_obs[i]:
            out[i] = float(np.sqrt(np.mean(sq[i, observed[i]])))
    return out, has_obs


def pooled_observed_rmse(
    states: np.ndarray,
    pred: np.ndarray,
    visibility: np.ndarray,
    eval_start: int,
) -> float:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    if not np.any(observed):
        return float("nan")
    err = states[:, eval_start:, :3][observed] - pred[:, eval_start:, :3][observed]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def paired_bootstrap_ci(
    diffs: np.ndarray, *, seed: int, n_bootstrap: int
) -> tuple[float, float]:
    """Percentile paired bootstrap over the trajectory population."""
    if diffs.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        boot[i] = float(np.mean(diffs[idx]))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def two_level_bootstrap_ci(
    diffs_by_seed: dict[int, np.ndarray], *, seed: int, n_bootstrap: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    seed_ids = np.asarray(sorted(diffs_by_seed), dtype=int)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sampled = rng.choice(seed_ids, size=seed_ids.size, replace=True)
        vals: list[np.ndarray] = []
        for sid in sampled:
            d = diffs_by_seed[int(sid)]
            idx = rng.integers(0, d.size, size=d.size)
            vals.append(d[idx])
        boot[i] = float(np.mean(np.concatenate(vals)))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def one_sided_wilcoxon(diffs: np.ndarray) -> tuple[float, float, str]:
    """One-sided paired signed-rank, alternative = candidate better.

    ``diffs`` is baseline - candidate, so positive favours the candidate and
    the one-sided 'greater' alternative tests candidate superiority.
    Descriptive diagnostic only.
    """
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0 or np.allclose(finite, 0.0):
        return float("nan"), float("nan"), "unavailable"
    for kwargs, backend in (
        ({"alternative": "greater", "method": "exact"}, "scipy.stats.wilcoxon[exact]"),
        ({"alternative": "greater"}, "scipy.stats.wilcoxon"),
    ):
        try:
            stat, p = wilcoxon(finite, **kwargs)
            return float(stat), float(p), backend
        except (ValueError, TypeError):
            continue
    return float("nan"), float("nan"), "unavailable"


def fmt(value: float) -> float | None:
    return None if not (isinstance(value, (int, float)) and math.isfinite(value)) else float(value)


def scenario_simulation_cfg(cfg: dict[str, Any], scenario: str) -> dict[str, Any]:
    if scenario == "stress_test":
        return deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])
    if scenario == "test":
        return cfg["simulation"]
    spec = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if spec is None:
        raise ValueError(f"Unknown scenario {scenario!r}")
    return deep_update(cfg["simulation"], spec.get("overrides", {}))


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    scenario = args.scenario

    train_cfg = parse_train_config(cfg["training"])
    train_cfg = replace(train_cfg, device=args.device)
    device = resolve_device(train_cfg.device)
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    data_dir = Path(cfg["data"]["output_dir"])
    arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
    scenario_cfg = scenario_simulation_cfg(cfg, scenario)
    dataset_cfg = parse_dataset_config(scenario_cfg)
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Seed-suite metrics CSV not found: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)
    # The seed checkpoint is shared across scenarios; use the 'test' rows to
    # enumerate the canonical 15-seed cohort and their checkpoint paths.
    seed_rows = (
        metrics_df[metrics_df["scenario"] == "test"][["seed", "checkpoint"]]
        .drop_duplicates()
        .sort_values("seed")
    )
    if seed_rows.empty:
        raise RuntimeError("No seed-cohort checkpoints found in metrics CSV.")

    classical: dict[str, np.ndarray] = {}
    classical_traj: dict[str, np.ndarray] = {}
    obs_mask: np.ndarray | None = None
    pooled_obs: dict[str, float] = {}

    per_seed_records: list[dict[str, Any]] = []
    seed_obs: list[float] = []
    diffs_vs: dict[str, dict[int, np.ndarray]] = {"EKF": {}, "AUKF": {}}
    seed_gain_vs: dict[str, list[float]] = {"EKF": [], "AUKF": []}

    for item in seed_rows.itertuples(index=False):
        seed_id = int(item.seed)
        ckpt = Path(str(item.checkpoint))
        pred, filters = infer_candidate_predictions(
            cfg=cfg,
            model_name=args.model_name,
            checkpoint_path=ckpt,
            arrays=arrays,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            train_cfg=train_cfg,
            device=device,
            scenario=scenario,
        )
        if not classical:
            for label, key in (("EKF", "ekf"), ("UKF", "ukf"), ("AUKF", "aukf")):
                t_rmse, mask = per_trajectory_observed_rmse(
                    arrays.states, filters[key], arrays.visibility, eval_start
                )
                classical_traj[label] = t_rmse
                pooled_obs[label] = pooled_observed_rmse(
                    arrays.states, filters[key], arrays.visibility, eval_start
                )
                obs_mask = mask if obs_mask is None else (obs_mask & mask)
            classical = {k: classical_traj[k] for k in classical_traj}

        cand_traj, cand_mask = per_trajectory_observed_rmse(
            arrays.states, pred, arrays.visibility, eval_start
        )
        seed_obs_rmse = pooled_observed_rmse(
            arrays.states, pred, arrays.visibility, eval_start
        )
        seed_obs.append(seed_obs_rmse)
        common = obs_mask & cand_mask
        for base_label in ("EKF", "AUKF"):
            base_traj = classical_traj[base_label]
            diff = base_traj[common] - cand_traj[common]  # >0 favours learned
            diffs_vs[base_label][seed_id] = diff
            seed_gain = pooled_obs[base_label] - seed_obs_rmse
            seed_gain_vs[base_label].append(seed_gain)
            per_seed_records.append(
                {
                    "scenario": scenario,
                    "seed": seed_id,
                    "method": args.method_label,
                    "baseline": base_label,
                    "comparison": f"vs {base_label}",
                    "candidate_observed_pos_rmse_m": fmt(seed_obs_rmse),
                    "baseline_observed_pos_rmse_m": fmt(pooled_obs[base_label]),
                    "seed_observed_step_gain_m": fmt(seed_gain),
                    "n_paired_trajectories": int(np.sum(common)),
                    "seed_win": bool(seed_gain > 0.0),
                    "checkpoint": str(ckpt),
                }
            )

    n_traj = int(np.sum(obs_mask)) if obs_mask is not None else 0

    # --- Deterministic classical paired diagnostics over the trajectory pop.
    classical_pairs = [("EKF", "AUKF"), ("EKF", "UKF"), ("UKF", "AUKF")]
    classical_rows: list[dict[str, Any]] = []
    for cand_label, base_label in classical_pairs:
        base = classical_traj[base_label][obs_mask]
        cand = classical_traj[cand_label][obs_mask]
        diff = base - cand  # >0 favours the candidate (the better-claimed filter)
        ci_low, ci_high = paired_bootstrap_ci(
            diff, seed=int(args.bootstrap_seed) + len(classical_rows), n_bootstrap=int(args.bootstrap_samples)
        )
        stat, p, backend = one_sided_wilcoxon(diff)
        classical_rows.append(
            {
                "scenario": scenario,
                "comparison": f"{cand_label} vs {base_label}",
                "candidate": cand_label,
                "baseline": base_label,
                "metric": "observed_step_pos_rmse_m",
                "statistical_unit": "per-trajectory observed-step position RMSE",
                "n_trajectories": n_traj,
                "candidate_pooled_observed_pos_rmse_m": fmt(pooled_obs[cand_label]),
                "baseline_pooled_observed_pos_rmse_m": fmt(pooled_obs[base_label]),
                "mean_paired_gain_m": fmt(float(np.mean(diff))),
                "paired_bootstrap_ci_low_m": fmt(ci_low),
                "paired_bootstrap_ci_high_m": fmt(ci_high),
                "paired_win_rate_percent": fmt(float(100.0 * np.mean(diff > 0.0))),
                "wilcoxon_stat": fmt(stat),
                "wilcoxon_p": fmt(p),
                "wilcoxon_backend": backend,
                "deterministic_note": (
                    "classical filters are deterministic on the controlled-mismatch "
                    "realization; the paired unit is a trajectory's observed-step RMSE "
                    "over the trajectory population"
                ),
            }
        )

    # --- 15-seed learned cohort paired diagnostics vs deterministic EKF/AUKF.
    cohort_rows: list[dict[str, Any]] = []
    seed_obs_arr = np.asarray(seed_obs, dtype=np.float64)
    for base_label in ("EKF", "AUKF"):
        seed_gains = np.asarray(seed_gain_vs[base_label], dtype=np.float64)
        diffs_by_seed = diffs_vs[base_label]
        pooled = np.concatenate([diffs_by_seed[s] for s in sorted(diffs_by_seed)])
        seed_ci = paired_bootstrap_ci(
            seed_gains,
            seed=int(args.bootstrap_seed) + 100 + len(cohort_rows),
            n_bootstrap=int(args.bootstrap_samples),
        )
        tl_ci = two_level_bootstrap_ci(
            diffs_by_seed,
            seed=int(args.bootstrap_seed) + 200 + len(cohort_rows),
            n_bootstrap=int(args.bootstrap_samples),
        )
        stat, p, backend = one_sided_wilcoxon(pooled)
        s_stat, s_p, s_backend = one_sided_wilcoxon(seed_gains)
        cohort_rows.append(
            {
                "scenario": scenario,
                "method": args.method_label,
                "comparison": f"vs {base_label}",
                "baseline": base_label,
                "metric": "observed_step_pos_rmse_m",
                "statistical_unit": "training seed (15-seed cohort)",
                "n_seeds": int(seed_gains.size),
                "n_seed_trajectory_pairs": int(pooled.size),
                "candidate_observed_pos_rmse_mean_m": fmt(float(np.mean(seed_obs_arr))),
                "candidate_observed_pos_rmse_min_m": fmt(float(np.min(seed_obs_arr))),
                "candidate_observed_pos_rmse_max_m": fmt(float(np.max(seed_obs_arr))),
                "baseline_observed_pos_rmse_m": fmt(pooled_obs[base_label]),
                "mean_seed_observed_step_gain_m": fmt(float(np.mean(seed_gains))),
                "seed_bootstrap_ci_low_m": fmt(seed_ci[0]),
                "seed_bootstrap_ci_high_m": fmt(seed_ci[1]),
                "two_level_bootstrap_ci_low_m": fmt(tl_ci[0]),
                "two_level_bootstrap_ci_high_m": fmt(tl_ci[1]),
                "seed_wins": int(np.sum(seed_gains > 0.0)),
                "seed_win_rate_percent": fmt(float(100.0 * np.mean(seed_gains > 0.0))),
                "pooled_win_rate_percent": fmt(float(100.0 * np.mean(pooled > 0.0))),
                "pooled_wilcoxon_stat": fmt(stat),
                "pooled_wilcoxon_p": fmt(p),
                "seed_wilcoxon_p": fmt(s_p),
                "wilcoxon_backend": backend,
                "seed_wilcoxon_backend": s_backend,
                "wilcoxon_note": (
                    "descriptive diagnostic, not a confirmatory test; trajectory "
                    "identities recur across seeds so pooled pairs are not "
                    "independent held-out samples"
                ),
            }
        )

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario,
        "device": str(device),
        "model_name": args.model_name,
        "method_label": args.method_label,
        "window_size": int(train_cfg.window_size),
        "eval_start_step": int(eval_start),
        "n_trajectories_with_observed_step": n_traj,
        "observed_step_definition": (
            "evaluated steps from window start onward with >=1 visible station; "
            "same convention as the main evaluator"
        ),
        "classical_pooled_observed_pos_rmse_m": {k: fmt(v) for k, v in pooled_obs.items()},
        "classical_paired_rows": classical_rows,
        "seed_cohort_rows": cohort_rows,
        "per_seed_rows": per_seed_records,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    out_json.write_text(json.dumps(payload, indent=2, default=fmt) + "\n", encoding="utf-8")
    pd.DataFrame(per_seed_records).to_csv(out_csv, index=False)
    print(
        json.dumps(
            {
                "scenario": scenario,
                "device": str(device),
                "n_trajectories": n_traj,
                "n_seeds": int(seed_obs_arr.size),
                "classical_rows": len(classical_rows),
                "cohort_rows": len(cohort_rows),
                "json": str(out_json),
                "csv": str(out_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
