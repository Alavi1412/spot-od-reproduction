#!/usr/bin/env python
"""Mechanistic diagnostic for the controlled force-model-mismatch result.

The manuscript's controlled dynamics/process-noise-mismatch split
(``force_model_mismatch_test``) reports EKF best and AUKF worst under true
dynamics mismatch, but does not analyse *why*. The truth is synthesized with a
heavier drag area-to-mass, denser reference atmosphere, larger SRP, and nonzero
process noise, while the estimators see the nominal compact model (the scenario
declares an explicit empty ``estimator_overrides``, so this script resolves the
estimator-side config exactly as the recursive filters / batch WLS / RFIS do).

This script quantifies the adaptive measurement-noise mechanism. It re-runs the
AUKF with an instrumented twin (numerically identical to the baseline, verified
against the cached ``aukf_prior``) and records, per visible update:

* the pre-adaptation NIS and whether it tripped the soft gate,
* the robust (Huber/soft-gate) inflation factor,
* the adapted / proposed / effective measurement-noise scale, and
* the resulting state-update norm.

It also computes a comparable R-only normalized-innovation NIS for the cached
EKF/UKF/AUKF predictions, and observed-step position RMSE for the three
recursive filters, so the summary is table-ready and focuses on the mechanism:
under true dynamics mismatch the innovations are dominated by an unmodelled
*dynamics* bias, not measurement noise; the AUKF's response is to inflate R
and damp its own corrections, which is exactly the wrong move when the prior
(not the measurement) is biased -- hence AUKF worst, EKF best.

Outputs:
* ``results/force_model_mismatch_adaptation_updates.csv`` -- one row per visible
  AUKF update with the full per-update diagnostic.
* ``results/force_model_mismatch_adaptation_summary.json`` / ``.csv`` -- the
  table-ready mechanism summary.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters.ukf import predicted_innovation_nis, run_adaptive_ukf_instrumented
from gnn_state_estimation.scenarios import (
    estimator_sim_config,
    has_estimator_overrides,
    truth_sim_config,
)
from gnn_state_estimation.simulation import DatasetConfig, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--scenario", default="force_model_mismatch_test")
    parser.add_argument(
        "--output-csv",
        default="results/force_model_mismatch_adaptation_updates.csv",
    )
    parser.add_argument(
        "--output-summary-json",
        default="results/force_model_mismatch_adaptation_summary.json",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="results/force_model_mismatch_adaptation_summary.csv",
    )
    parser.add_argument(
        "--trajectory-limit",
        type=int,
        default=0,
        help="If > 0, only process the first N trajectories (smoke runs).",
    )
    parser.add_argument(
        "--reconstruction-tol-m",
        type=float,
        default=1.0e-3,
        help="Max abs position diff (m) tolerated between the instrumented "
        "AUKF and the cached aukf_prior before the run is flagged.",
    )
    return parser


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def resolve_sim_configs(
    cfg: dict[str, Any], scenario: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Return (estimator_sim, truth_sim, scenario_cfg) for the scenario.

    Uses the same estimator-side resolution as the recursive filters / batch
    WLS (``estimator_sim_config``): equal to the truth config unless the
    scenario declares ``estimator_overrides`` (the force-mismatch split
    declares an explicit empty mapping -> estimators see the nominal base sim).
    """
    sim_cfg = copy.deepcopy(cfg["simulation"])
    if scenario == "stress_test":
        sim_cfg = deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        # Plain scenarios (test / stress_test): estimator == truth == sim_cfg.
        return sim_cfg, sim_cfg, None
    est_sim = estimator_sim_config(sim_cfg, scenario_cfg)
    truth_sim = truth_sim_config(sim_cfg, scenario_cfg)
    return est_sim, truth_sim, scenario_cfg


def dynamics_provenance(
    est_sim: dict[str, Any], truth_sim: dict[str, Any]
) -> dict[str, Any]:
    """Key truth-vs-estimator dynamics deltas (mechanism provenance only).

    Diagnostic provenance for the results JSON; never written to paper files.
    """
    keys = (
        "ballistic_coeff_m2_per_kg",
        "process_noise_std",
        "drag_rho_ref",
        "srp_area_to_mass_m2_per_kg",
        "srp_cr",
    )
    est_dyn = est_sim.get("dynamics", {})
    truth_dyn = truth_sim.get("dynamics", {})
    out: dict[str, Any] = {}
    for key in keys:
        out[key] = {
            "estimator": est_dyn.get(key),
            "truth": truth_dyn.get(key),
            "mismatched": est_dyn.get(key) != truth_dyn.get(key),
        }
    return out


def masked_pos_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    err = y_true[mask, :3] - y_pred[mask, :3]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def observed_step_metrics(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> dict[str, float]:
    """Observed-step / all-step position RMSE (main-evaluator convention).

    Mirrors ``_observed_metrics`` in ``scripts/run_batch_wls_baseline.py``:
    from the window start onward, a (trajectory, step) pair is observed when at
    least one station is visible.
    """
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    y_true = states[:, eval_start:]
    y_pred = preds[:, eval_start:]
    all_mask = np.ones(observed.shape, dtype=bool)
    return {
        "observed_step_pos_rmse_m": masked_pos_rmse(y_true, y_pred, observed),
        "all_step_pos_rmse_m": masked_pos_rmse(y_true, y_pred, all_mask),
        "observed_steps": int(np.sum(observed)),
    }


def finite_or_none(value: float) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def summarize_records(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"n_visible_updates": 0}
    nis = df["pre_adapt_nis"].to_numpy(dtype=np.float64)
    return {
        "n_visible_updates": int(len(df)),
        "mean_pre_adapt_nis": float(np.mean(nis)),
        "median_pre_adapt_nis": float(np.median(nis)),
        "p90_pre_adapt_nis": float(np.percentile(nis, 90.0)),
        "max_pre_adapt_nis": float(np.max(nis)),
        "percent_updates_exceeding_soft_gate": float(
            100.0 * np.mean(df["nis_exceeds_soft_gate"].to_numpy(dtype=bool))
        ),
        "nis_soft_gate": float(df["nis_soft_gate"].iloc[0]),
        "mean_robust_scale": float(np.mean(df["robust_scale"].to_numpy(dtype=np.float64))),
        "median_robust_scale": float(np.median(df["robust_scale"].to_numpy(dtype=np.float64))),
        "mean_r_scale_pre": float(np.mean(df["r_scale_pre_mean"].to_numpy(dtype=np.float64))),
        "mean_r_proposal_scale": float(
            np.mean(df["r_proposal_scale_mean"].to_numpy(dtype=np.float64))
        ),
        "mean_r_scale_post": float(np.mean(df["r_scale_post_mean"].to_numpy(dtype=np.float64))),
        "mean_r_eff_scale": float(np.mean(df["r_eff_scale_mean"].to_numpy(dtype=np.float64))),
        "mean_state_update_norm": float(
            np.mean(df["state_update_norm"].to_numpy(dtype=np.float64))
        ),
        "mean_state_update_pos_norm_m": float(
            np.mean(df["state_update_pos_norm_m"].to_numpy(dtype=np.float64))
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    scenario = str(args.scenario)

    est_sim, truth_sim, scenario_cfg = resolve_sim_configs(cfg, scenario)
    dataset_cfg: DatasetConfig = parse_dataset_config(est_sim)
    dyn = dataset_cfg.dynamics
    meas_std = dataset_cfg.measurement_noise.std_vector

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    if baseline_cfg.aukf is None:
        raise ValueError("Baseline config does not define an AUKF; cannot run the diagnostic.")
    aukf_cfg = baseline_cfg.aukf

    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    data_dir = Path(cfg["data"]["output_dir"])
    arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
    if arrays.x0_estimates is None:
        raise ValueError(
            f"{scenario}.npz is missing x0_estimates; the AUKF must reuse the "
            "same initial estimates as the cached baseline to be faithful."
        )
    states = arrays.states
    measurements = arrays.measurements
    visibility = arrays.visibility
    times = arrays.times
    n_traj = states.shape[0]
    if args.trajectory_limit > 0:
        n_traj = min(args.trajectory_limit, n_traj)

    update_rows: list[dict[str, Any]] = []
    aukf_recon = np.zeros_like(states[:n_traj])
    max_recon_diff_pos_m = 0.0
    max_recon_diff_vel_mps = 0.0
    cross_filter_nis: dict[str, list[float]] = {"EKF": [], "UKF": [], "AUKF": []}
    prior_arrays = {
        "EKF": arrays.ekf_prior,
        "UKF": arrays.ukf_prior,
        "AUKF": arrays.aukf_prior,
    }

    start = time.perf_counter()
    for i in range(n_traj):
        x_hist, _p_hist, records = run_adaptive_ukf_instrumented(
            measurements=measurements[i],
            visibility=visibility[i],
            times_s=times[i],
            stations=dataset_cfg.stations,
            ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
            meas_std_vector=meas_std,
            x0_est=arrays.x0_estimates[i],
            cfg=aukf_cfg,
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
        aukf_recon[i] = x_hist
        if arrays.aukf_prior is not None:
            diff = x_hist - arrays.aukf_prior[i]
            max_recon_diff_pos_m = max(
                max_recon_diff_pos_m, float(np.max(np.abs(diff[:, :3])))
            )
            max_recon_diff_vel_mps = max(
                max_recon_diff_vel_mps, float(np.max(np.abs(diff[:, 3:])))
            )
        for rec in records:
            rec_out = {"scenario": scenario, "trajectory": int(i)}
            rec_out.update(rec)
            update_rows.append(rec_out)

        for method, prior in prior_arrays.items():
            if prior is None:
                continue
            for entry in predicted_innovation_nis(
                pred_states=prior[i],
                measurements=measurements[i],
                visibility=visibility[i],
                times_s=times[i],
                stations=dataset_cfg.stations,
                meas_std_vector=meas_std,
            ):
                cross_filter_nis[method].append(entry["nis_r"])
    elapsed_s = time.perf_counter() - start

    updates_df = pd.DataFrame(update_rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    updates_df.to_csv(out_csv, index=False)

    mechanism = summarize_records(updates_df)

    observed_step: dict[str, Any] = {}
    for method, prior in prior_arrays.items():
        if prior is None:
            continue
        observed_step[method] = observed_step_metrics(
            states[:n_traj], prior[:n_traj], visibility[:n_traj], eval_start
        )

    cross_filter_summary: dict[str, Any] = {}
    for method, vals in cross_filter_nis.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        cross_filter_summary[method] = {
            "n_visible_updates": int(arr.size),
            "mean_r_only_nis": float(np.mean(arr)),
            "median_r_only_nis": float(np.median(arr)),
            "p90_r_only_nis": float(np.percentile(arr, 90.0)),
        }

    reconstruction = {
        "max_abs_pos_diff_vs_cached_aukf_m": max_recon_diff_pos_m,
        "max_abs_vel_diff_vs_cached_aukf_mps": max_recon_diff_vel_mps,
        "tolerance_m": float(args.reconstruction_tol_m),
        "matches_cached_aukf_prior": bool(
            arrays.aukf_prior is not None
            and max_recon_diff_pos_m <= float(args.reconstruction_tol_m)
        ),
        "cached_aukf_prior_available": bool(arrays.aukf_prior is not None),
    }

    summary = {
        "scenario": scenario,
        "trajectories_processed": int(n_traj),
        "total_trajectories_in_split": int(states.shape[0]),
        "window_size": int(train_cfg.window_size),
        "eval_start_step": int(eval_start),
        "runtime_s": float(elapsed_s),
        "estimator_truth_model_mismatch": bool(
            scenario_cfg is not None and has_estimator_overrides(scenario_cfg)
        ),
        "dynamics_provenance": dynamics_provenance(est_sim, truth_sim),
        "aukf_config": {
            "adapt_rate": float(aukf_cfg.adapt_rate),
            "nis_soft_gate": float(aukf_cfg.nis_soft_gate),
            "huber_kappa": float(aukf_cfg.huber_kappa),
            "min_r_scale": float(aukf_cfg.min_r_scale),
            "max_r_scale": float(aukf_cfg.max_r_scale),
        },
        "aukf_adaptation_mechanism": mechanism,
        "cross_filter_r_only_nis": cross_filter_summary,
        "observed_step_pos_rmse": observed_step,
        "aukf_reconstruction": reconstruction,
        "interpretation": (
            "Under true force-model/process-noise mismatch the visible-update "
            "innovations are dominated by an unmodelled dynamics bias, not by "
            "measurement noise. The AUKF reacts by inflating its measurement-"
            "noise diagonal (mean effective R scale > 1) and applying robust "
            "down-weighting, which shrinks the very state corrections needed to "
            "track the biased prior. EKF keeps a tighter gain and so tracks the "
            "drifting truth best; AUKF damps itself the most and is worst."
        ),
    }

    out_summary_json = Path(args.output_summary_json)
    out_summary_csv = Path(args.output_summary_csv)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.write_text(
        json.dumps(summary, indent=2, default=finite_or_none) + "\n", encoding="utf-8"
    )

    flat: dict[str, Any] = {
        "scenario": scenario,
        "trajectories_processed": int(n_traj),
        "estimator_truth_model_mismatch": summary["estimator_truth_model_mismatch"],
        "aukf_matches_cached_prior": reconstruction["matches_cached_aukf_prior"],
        "max_abs_pos_diff_vs_cached_aukf_m": max_recon_diff_pos_m,
    }
    for key, value in mechanism.items():
        flat[f"aukf_{key}"] = value
    for method, metrics in observed_step.items():
        flat[f"{method.lower()}_observed_step_pos_rmse_m"] = metrics["observed_step_pos_rmse_m"]
        flat[f"{method.lower()}_all_step_pos_rmse_m"] = metrics["all_step_pos_rmse_m"]
    for method, metrics in cross_filter_summary.items():
        flat[f"{method.lower()}_mean_r_only_nis"] = metrics["mean_r_only_nis"]
        flat[f"{method.lower()}_median_r_only_nis"] = metrics["median_r_only_nis"]
    pd.DataFrame([flat]).to_csv(out_summary_csv, index=False)

    print(
        json.dumps(
            {
                "scenario": scenario,
                "trajectories_processed": int(n_traj),
                "update_rows": int(len(updates_df)),
                "runtime_s": round(float(elapsed_s), 2),
                "updates_csv": str(out_csv),
                "summary_json": str(out_summary_json),
                "summary_csv": str(out_summary_csv),
                "aukf_reconstruction": reconstruction,
                "aukf_adaptation_mechanism": mechanism,
                "cross_filter_r_only_nis": cross_filter_summary,
                "observed_step_pos_rmse": observed_step,
            },
            indent=2,
            default=finite_or_none,
        )
    )


if __name__ == "__main__":
    main()
