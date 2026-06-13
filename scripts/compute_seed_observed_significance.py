#!/usr/bin/env python
"""Seed-level observed-step significance diagnostic from learned seed-suite checkpoints.

The manuscript declares observed-step position RMSE (evaluated steps, from the
window start onward, on which at least one station is visible) as the primary
estimator-skill metric, while all-step RMSE is a propagation-gap reference. The
15-seed seed-aware diagnostic in :func:`build_seed_aware_significance_table`
operates on the all-step ``pos_rmse_m`` recorded in the seed-suite metrics CSV.
This script recomputes, from the same retained 15 seed checkpoints, the
observed-step analogue so the primary-metric headline is backed by a
reproducible 15-seed observed-step diagnostic rather than only the all-step
aggregate.

Unit of analysis is a training seed's observed-step aggregate position RMSE
(one scalar per seed/scenario, not a trajectory-paired pool): for each
seed/scenario the candidate observed-step RMSE is compared against the UKF and
AUKF observed-step RMSE computed on the same scenario data with the same window
and observed-step scoring convention. Trajectory identities are not pooled
across seeds.
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
import torch
from scipy.stats import wilcoxon

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
    from compute_seed_pooled_significance import run_model_inference_batched
    from run_benchmark_seed_sweep import deep_update, load_model_from_checkpoint, load_or_compute_baselines
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts.compute_seed_pooled_significance import run_model_inference_batched
    from scripts.run_benchmark_seed_sweep import (
        deep_update,
        load_model_from_checkpoint,
        load_or_compute_baselines,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed-level observed-step significance diagnostic.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--metrics-csv",
        default="results/seed_suite_hybrid_public/benchmark_seed_metrics.csv",
        help="Seed-suite metrics CSV providing the retained seed checkpoints.",
    )
    parser.add_argument("--model-name", default="HybridGNN")
    parser.add_argument("--method-label", default="RGR-GF")
    parser.add_argument("--scenarios", default="test,stress_test")
    parser.add_argument("--output-csv", default="results/seed_observed_significance.csv")
    parser.add_argument("--output-summary-csv", default="results/seed_observed_significance_summary.csv")
    parser.add_argument("--output-summary-json", default="results/seed_observed_significance_summary.json")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260518)
    return parser


def observed_step_pos_rmse(
    states: np.ndarray,
    pred: np.ndarray,
    visibility: np.ndarray,
    eval_start: int,
) -> tuple[float, int]:
    """Observed-step position RMSE using the main evaluator's convention.

    Mirrors ``_observed_metrics`` / ``_masked_pos_rmse`` in
    ``scripts/run_batch_wls_baseline.py``: from the window start (``eval_start``)
    onward, a (trajectory, step) pair is "observed" when at least one station is
    visible (summed visibility over stations >= 0.5). RMSE is the root mean
    squared 3D position error pooled over all observed (trajectory, step) pairs.
    """
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    if not np.any(observed):
        return float("nan"), 0
    y_true = states[:, eval_start:]
    y_pred = pred[:, eval_start:]
    err = y_true[observed, :3] - y_pred[observed, :3]
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))
    return rmse, int(np.sum(observed))


def infer_candidate_predictions(
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
    """Load a seed checkpoint and return raw candidate predictions plus filters.

    This reuses the seed-suite loading/inference helpers; it is the prediction
    half of ``compute_seed_pooled_significance.infer_candidate_trajectory_rmse``
    without collapsing to a trajectory-RMSE vector, so that the observed-step
    mask can be applied with the main evaluator's convention.
    """
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
    return pred, filters


def seed_bootstrap_ci(values: np.ndarray, *, seed: int, n_bootstrap: int) -> tuple[float, float]:
    """Single-level percentile bootstrap over the per-seed gain values."""
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, values.size, size=values.size)
        boot[i] = float(np.mean(values[idx]))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def diagnostic_wilcoxon(gains: np.ndarray) -> tuple[float, str]:
    """One-sided seed-level signed-rank p (diagnostic only).

    With a small repeated-training-seed sample and gains that may be all one
    sign, this is sign-test-like and is reported only as diagnostic direction
    evidence, never as a headline significance claim. Returns NaN if scipy
    cannot compute it (e.g. all-zero differences).
    """
    if gains.size == 0:
        return float("nan"), "unavailable"
    for kwargs, backend in (
        ({"alternative": "greater", "method": "exact"}, "scipy.stats.wilcoxon[exact]"),
        ({"alternative": "greater", "mode": "exact"}, "scipy.stats.wilcoxon[exact]"),
        ({"alternative": "greater"}, "scipy.stats.wilcoxon"),
    ):
        try:
            _, p_value = wilcoxon(gains, **kwargs)
            return float(p_value), backend
        except (ValueError, TypeError):
            continue
    return float("nan"), "unavailable"


def scenario_simulation_cfg(cfg: dict[str, Any], scenario: str) -> dict[str, Any]:
    scenario_cfg = cfg["simulation"]
    if scenario == "stress_test":
        return deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])
    if scenario == "test":
        return scenario_cfg
    scenario_spec = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_spec is None:
        raise ValueError(f"Unknown scenario {scenario!r}")
    return deep_update(cfg["simulation"], scenario_spec.get("overrides", {}))


def format_float(value: float) -> float | None:
    return None if not (isinstance(value, (int, float)) and math.isfinite(value)) else float(value)


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]

    train_cfg = parse_train_config(cfg["training"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(train_cfg, device=str(cfg.get("device", {}).get("eval", train_cfg.device)))
    device = resolve_device(train_cfg.device)
    data_dir = Path(cfg["data"]["output_dir"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Seed-suite metrics CSV not found: {metrics_path}")
    metrics_df = pd.read_csv(metrics_path)

    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        focus = metrics_df[metrics_df["scenario"] == scenario].copy()
        if focus.empty:
            continue
        scenario_cfg = scenario_simulation_cfg(cfg, scenario)
        arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
        dataset_cfg = parse_dataset_config(scenario_cfg)

        baseline_obs: dict[str, float] = {}
        observed_steps_count = 0
        per_seed_gain: dict[str, list[float]] = {"UKF": [], "AUKF": []}

        for item in focus.sort_values("seed").itertuples(index=False):
            seed_id = int(item.seed)
            checkpoint_path = Path(str(item.checkpoint))
            pred, filters = infer_candidate_predictions(
                cfg=cfg,
                model_name=args.model_name,
                checkpoint_path=checkpoint_path,
                arrays=arrays,
                dataset_cfg=dataset_cfg,
                baseline_cfg=baseline_cfg,
                train_cfg=train_cfg,
                device=device,
                scenario=scenario,
            )
            candidate_obs, observed_steps = observed_step_pos_rmse(
                arrays.states, pred, arrays.visibility, eval_start
            )
            if not baseline_obs:
                observed_steps_count = observed_steps
                for blabel, key in (("UKF", "ukf"), ("AUKF", "aukf")):
                    baseline_obs[blabel], _ = observed_step_pos_rmse(
                        arrays.states, filters[key], arrays.visibility, eval_start
                    )
            for baseline_label in ("UKF", "AUKF"):
                base_val = baseline_obs[baseline_label]
                gain = base_val - candidate_obs
                pct = 100.0 * gain / base_val if math.isfinite(base_val) and base_val != 0.0 else float("nan")
                per_seed_gain[baseline_label].append(gain)
                per_seed_rows.append(
                    {
                        "method": args.method_label,
                        "scenario": scenario,
                        "seed": seed_id,
                        "comparison": f"vs {baseline_label}",
                        "baseline": baseline_label,
                        "metric": "observed_step_pos_rmse_m",
                        "window_size": int(train_cfg.window_size),
                        "eval_start_step": int(eval_start),
                        "observed_steps": int(observed_steps),
                        "candidate_observed_pos_rmse_m": candidate_obs,
                        "baseline_observed_pos_rmse_m": base_val,
                        "observed_step_gain_m": gain,
                        "observed_step_percent_improvement": pct,
                        "seed_win": bool(gain > 0.0),
                        "checkpoint": str(checkpoint_path),
                    }
                )

        # Recover per-seed candidate observed RMSE from collected rows for summary stats.
        cand_vals = [
            r["candidate_observed_pos_rmse_m"]
            for r in per_seed_rows
            if r["scenario"] == scenario and r["baseline"] == "UKF"
        ]
        cand_arr = np.asarray(cand_vals, dtype=np.float64)

        for baseline_label in ("UKF", "AUKF"):
            gains = np.asarray(per_seed_gain[baseline_label], dtype=np.float64)
            if gains.size == 0:
                continue
            ci_low, ci_high = seed_bootstrap_ci(
                gains,
                seed=int(args.bootstrap_seed) + len(summary_rows),
                n_bootstrap=int(args.bootstrap_samples),
            )
            p_value, backend = diagnostic_wilcoxon(gains)
            seed_wins = int(np.sum(gains > 0.0))
            base_val = baseline_obs[baseline_label]
            pct_arr = (
                100.0 * gains / base_val
                if math.isfinite(base_val) and base_val != 0.0
                else np.full(gains.shape, np.nan)
            )
            summary_rows.append(
                {
                    "method": args.method_label,
                    "scenario": scenario,
                    "comparison": f"vs {baseline_label}",
                    "baseline": baseline_label,
                    "metric": "observed_step_pos_rmse_m",
                    "n_seeds": int(gains.size),
                    "candidate_observed_pos_rmse_mean_m": float(np.mean(cand_arr)) if cand_arr.size else float("nan"),
                    "candidate_observed_pos_rmse_min_m": float(np.min(cand_arr)) if cand_arr.size else float("nan"),
                    "candidate_observed_pos_rmse_max_m": float(np.max(cand_arr)) if cand_arr.size else float("nan"),
                    "baseline_observed_pos_rmse_m": float(base_val),
                    "observed_steps": int(observed_steps_count),
                    "mean_observed_step_gain_m": float(np.mean(gains)),
                    "seed_bootstrap_ci_low_m": ci_low,
                    "seed_bootstrap_ci_high_m": ci_high,
                    "seed_wins": seed_wins,
                    "seed_win_rate_percent": float(100.0 * seed_wins / gains.size),
                    "mean_percent_improvement": float(np.mean(pct_arr)),
                    "diagnostic_wilcoxon_p": p_value,
                    "wilcoxon_backend": backend,
                    "observed_step_definition": (
                        "evaluated steps from window start onward with >=1 visible "
                        "station; same convention as the main evaluator"
                    ),
                    "wilcoxon_note": (
                        "seed-level observed-step aggregate diagnostic; with a small "
                        "repeated-training-seed sample this signed-rank p is "
                        "sign-test-like and is direction evidence only, not a "
                        "headline significance claim"
                    ),
                }
            )

    out_csv = Path(args.output_csv)
    out_summary_csv = Path(args.output_summary_csv)
    out_summary_json = Path(args.output_summary_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_seed_rows).to_csv(out_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(out_summary_csv, index=False)
    out_summary_json.write_text(
        json.dumps(
            {
                "scenarios": scenarios,
                "device": str(device),
                "metrics_csv": str(metrics_path),
                "window_size": int(train_cfg.window_size),
                "eval_start_step": int(eval_start),
                "rows": summary_rows,
            },
            indent=2,
            default=format_float,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "per_seed_rows": len(per_seed_rows),
                "summary_rows": len(summary_rows),
                "csv": str(out_csv),
                "summary_csv": str(out_summary_csv),
                "summary_json": str(out_summary_json),
                "device": str(device),
                "summary": summary_rows,
            },
            indent=2,
            default=format_float,
        )
    )


if __name__ == "__main__":
    main()
