#!/usr/bin/env python
"""Constrained-AUKF mechanism-control experiment on the force-model-mismatch split.

Additional targeted mechanism-control: re-runs the AUKF with effective
measurement-noise scale capped at 2.0 (max_r_scale=2.0, all other parameters
unchanged) on the same 48 force_model_mismatch_test trajectories and compares
observed-step RMSE to cached EKF, UKF, and standard AUKF (max_r_scale=30.0).
The cap is implemented via an AdaptiveUKFConfig copy with max_r_scale=2.0;
run_adaptive_ukf_instrumented clamps r_eff_diag to r_diag_max before the gain,
so the cap is effective and faithful to the existing implementation semantics.

This is an additional targeted mechanism-control diagnostic, not a predeclared
positive criterion and not part of the original frozen learned-estimator claim
audit. It tests whether limiting R-inflation to 2x nominal (versus the standard
30x ceiling) partially restores correction gain under dynamics-bias conditions,
without training or tuning any new estimator.

Paired-difference convention: AUKF-Rcap minus comparator.  Negative values mean
AUKF-Rcap achieves lower RMSE (better); positive values mean AUKF-Rcap is worse.

Outputs:
  results/constrained_aukf_mechanism_control/constrained_aukf_mechanism_control.json
  results/constrained_aukf_mechanism_control/constrained_aukf_mechanism_control.csv
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters.ukf import run_adaptive_ukf_instrumented
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml

try:
    from analyze_force_mismatch_adaptation import (
        resolve_sim_configs,
        summarize_records,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.analyze_force_mismatch_adaptation import (
        resolve_sim_configs,
        summarize_records,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def per_trajectory_observed_rmse(
    states: np.ndarray,
    pred: np.ndarray,
    visibility: np.ndarray,
    eval_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trajectory observed-step position RMSE and trajectory-observed mask.

    Mirrors ``per_trajectory_observed_rmse`` in run_force_mismatch_seed_significance.py.
    """
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5  # (n_traj, n_eval_steps)
    y_true = states[:, eval_start:, :3]
    y_pred = pred[:, eval_start:, :3]
    sq = np.sum((y_true - y_pred) ** 2, axis=-1)
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
    diffs: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> tuple[float, float]:
    """Percentile paired bootstrap 95% CI over the trajectory population."""
    if diffs.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, diffs.size, size=diffs.size)
        boot[i] = float(np.mean(diffs[idx]))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def finite_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--scenario", default="force_model_mismatch_test")
    p.add_argument(
        "--rcap-max-r-scale",
        type=float,
        default=2.0,
        help="Effective R-scale cap for the constrained AUKF variant (default 2.0).",
    )
    p.add_argument(
        "--output-dir",
        default="results/constrained_aukf_mechanism_control",
    )
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=20260526)
    p.add_argument(
        "--trajectory-limit",
        type=int,
        default=0,
        help="If > 0, only process the first N trajectories (smoke runs).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    scenario = str(args.scenario)
    rcap = float(args.rcap_max_r_scale)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    est_sim, truth_sim, scenario_cfg = resolve_sim_configs(cfg, scenario)
    dataset_cfg = parse_dataset_config(est_sim)
    dyn = dataset_cfg.dynamics
    meas_std = dataset_cfg.measurement_noise.std_vector

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    if baseline_cfg.aukf is None:
        raise ValueError("Baseline config does not define an AUKF.")
    aukf_cfg_standard = baseline_cfg.aukf
    # Build constrained variant: cap effective R scale at rcap (all other params identical)
    aukf_cfg_rcap = replace(aukf_cfg_standard, max_r_scale=rcap)

    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    data_dir = Path(cfg["data"]["output_dir"])
    arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
    if arrays.x0_estimates is None:
        raise ValueError(
            f"{scenario}.npz is missing x0_estimates; the AUKF must reuse the "
            "same initial estimates as the cached baseline."
        )
    states = arrays.states
    measurements = arrays.measurements
    visibility = arrays.visibility
    times = arrays.times
    n_traj = states.shape[0]
    if args.trajectory_limit > 0:
        n_traj = min(args.trajectory_limit, n_traj)

    # -----------------------------------------------------------------------
    # Run constrained AUKF (AUKF-Rcap) with instrumentation
    # -----------------------------------------------------------------------
    update_rows: list[dict[str, Any]] = []
    rcap_preds = np.zeros((n_traj, states.shape[1], 6), dtype=np.float64)

    t_start = time.perf_counter()
    for i in range(n_traj):
        x_hist, _p_hist, records = run_adaptive_ukf_instrumented(
            measurements=measurements[i],
            visibility=visibility[i],
            times_s=times[i],
            stations=dataset_cfg.stations,
            ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
            meas_std_vector=meas_std,
            x0_est=arrays.x0_estimates[i],
            cfg=aukf_cfg_rcap,
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
        rcap_preds[i] = x_hist
        for rec in records:
            rec_out: dict[str, Any] = {"scenario": scenario, "trajectory": int(i)}
            rec_out.update(rec)
            update_rows.append(rec_out)
    elapsed_s = time.perf_counter() - t_start

    # -----------------------------------------------------------------------
    # AUKF-Rcap mechanism summary
    # -----------------------------------------------------------------------
    updates_df = pd.DataFrame(update_rows)
    rcap_mechanism = summarize_records(updates_df)

    # -----------------------------------------------------------------------
    # Observed-step RMSE: AUKF-Rcap + cached EKF / UKF / AUKF
    # -----------------------------------------------------------------------
    cached = {
        "EKF": arrays.ekf_prior,
        "UKF": arrays.ukf_prior,
        "AUKF": arrays.aukf_prior,
    }

    # Pooled observed-step RMSE
    pooled: dict[str, float] = {
        "AUKF_Rcap": pooled_observed_rmse(states[:n_traj], rcap_preds, visibility[:n_traj], eval_start),
    }
    for name, prior in cached.items():
        if prior is None:
            continue
        pooled[name] = pooled_observed_rmse(states[:n_traj], prior[:n_traj], visibility[:n_traj], eval_start)

    # Per-trajectory observed-step RMSE
    traj_rcap, has_obs_rcap = per_trajectory_observed_rmse(
        states[:n_traj], rcap_preds, visibility[:n_traj], eval_start
    )

    traj_cached: dict[str, np.ndarray] = {}
    has_obs_cached: dict[str, np.ndarray] = {}
    for name, prior in cached.items():
        if prior is None:
            continue
        t, h = per_trajectory_observed_rmse(
            states[:n_traj], prior[:n_traj], visibility[:n_traj], eval_start
        )
        traj_cached[name] = t
        has_obs_cached[name] = h

    # Common observed mask across all methods
    common_mask = has_obs_rcap.copy()
    for h in has_obs_cached.values():
        common_mask &= h

    n_paired = int(np.sum(common_mask))

    # -----------------------------------------------------------------------
    # Paired comparisons: AUKF-Rcap minus comparator
    # Convention: negative = AUKF-Rcap lower RMSE (better)
    # -----------------------------------------------------------------------
    paired_results: dict[str, Any] = {}
    for comp_name, comp_traj in traj_cached.items():
        diffs = traj_rcap[common_mask] - comp_traj[common_mask]  # AUKF-Rcap - comparator
        mean_gap = float(np.mean(diffs))
        ci_lo, ci_hi = paired_bootstrap_ci(
            diffs,
            seed=args.bootstrap_seed + len(paired_results),
            n_bootstrap=args.bootstrap_samples,
        )
        win_rate = float(100.0 * np.mean(diffs < 0.0))  # % trajectories AUKF-Rcap lower
        paired_results[comp_name] = {
            "mean_paired_gap_m": mean_gap,
            "ci_lo_m": ci_lo,
            "ci_hi_m": ci_hi,
            "win_rate_pct": win_rate,
            "n_paired_trajectories": n_paired,
            "convention": "AUKF_Rcap minus comparator; negative = AUKF_Rcap lower RMSE",
        }

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    summary: dict[str, Any] = {
        "scenario": scenario,
        "schema_version": "constrained_aukf_mechanism_control_v1",
        "trajectories_processed": int(n_traj),
        "total_trajectories_in_split": int(states.shape[0]),
        "eval_start_step": int(eval_start),
        "runtime_s": float(elapsed_s),
        "rcap_config": {
            "max_r_scale": float(aukf_cfg_rcap.max_r_scale),
            "min_r_scale": float(aukf_cfg_rcap.min_r_scale),
            "adapt_rate": float(aukf_cfg_rcap.adapt_rate),
            "nis_soft_gate": float(aukf_cfg_rcap.nis_soft_gate),
            "huber_kappa": float(aukf_cfg_rcap.huber_kappa),
        },
        "standard_aukf_max_r_scale": float(aukf_cfg_standard.max_r_scale),
        "rcap_mechanism": rcap_mechanism,
        "pooled_observed_step_rmse_m": pooled,
        "paired_comparisons": paired_results,
        "n_paired_trajectories": n_paired,
        "paired_difference_convention": (
            "AUKF_Rcap minus comparator; negative means AUKF_Rcap achieves lower RMSE"
        ),
        "interpretation": (
            "Capping the effective R scale at 2x nominal (vs the standard 30x ceiling) "
            "limits the R-inflation damping diagnosed in the AUKF mechanism table. "
            "If AUKF-Rcap is better than standard AUKF, the mechanism connection is confirmed. "
            "If it is still worse than EKF, the R-inflation cap alone does not fully rescue "
            "AUKF under dynamics-bias conditions, consistent with the known covariance-matching "
            "limitation. The result is reported honestly with paired bootstrap CIs and does not "
            "change the primary evidence hierarchy."
        ),
    }

    out_json = out_dir / "constrained_aukf_mechanism_control.json"
    out_json.write_text(
        json.dumps(summary, indent=2, default=finite_or_none) + "\n", encoding="utf-8"
    )

    flat: dict[str, Any] = {
        "scenario": scenario,
        "trajectories_processed": int(n_traj),
        "rcap_max_r_scale": float(rcap),
        "standard_aukf_max_r_scale": float(aukf_cfg_standard.max_r_scale),
        "rcap_mean_r_eff_scale": rcap_mechanism.get("mean_r_eff_scale", float("nan")),
        "rcap_mean_pre_adapt_nis": rcap_mechanism.get("mean_pre_adapt_nis", float("nan")),
        "rcap_median_pre_adapt_nis": rcap_mechanism.get("median_pre_adapt_nis", float("nan")),
        "rcap_p90_pre_adapt_nis": rcap_mechanism.get("p90_pre_adapt_nis", float("nan")),
        "rcap_mean_state_update_pos_norm_m": rcap_mechanism.get(
            "mean_state_update_pos_norm_m", float("nan")
        ),
    }
    for key, val in pooled.items():
        flat[f"pooled_rmse_{key}_m"] = val
    for comp, pr in paired_results.items():
        flat[f"gap_{comp}_mean_m"] = pr["mean_paired_gap_m"]
        flat[f"gap_{comp}_ci_lo_m"] = pr["ci_lo_m"]
        flat[f"gap_{comp}_ci_hi_m"] = pr["ci_hi_m"]
        flat[f"gap_{comp}_win_rate_pct"] = pr["win_rate_pct"]
    pd.DataFrame([flat]).to_csv(
        out_dir / "constrained_aukf_mechanism_control.csv", index=False
    )

    print(
        json.dumps(
            {
                "scenario": scenario,
                "trajectories_processed": int(n_traj),
                "rcap_max_r_scale": float(rcap),
                "runtime_s": round(float(elapsed_s), 2),
                "pooled_observed_step_rmse_m": {
                    k: round(v, 2) if math.isfinite(v) else None
                    for k, v in pooled.items()
                },
                "paired_comparisons_summary": {
                    comp: {
                        "mean_gap_m": round(float(pr["mean_paired_gap_m"]), 2)
                        if math.isfinite(float(pr["mean_paired_gap_m"]))
                        else None,
                        "ci_lo_m": round(float(pr["ci_lo_m"]), 2)
                        if math.isfinite(float(pr["ci_lo_m"]))
                        else None,
                        "ci_hi_m": round(float(pr["ci_hi_m"]), 2)
                        if math.isfinite(float(pr["ci_hi_m"]))
                        else None,
                        "win_rate_pct": round(float(pr["win_rate_pct"]), 1),
                    }
                    for comp, pr in paired_results.items()
                },
                "rcap_mechanism_summary": {
                    "mean_r_eff_scale": round(
                        float(rcap_mechanism.get("mean_r_eff_scale", float("nan"))), 3
                    )
                    if math.isfinite(
                        float(rcap_mechanism.get("mean_r_eff_scale", float("nan")))
                    )
                    else None,
                    "mean_state_update_pos_norm_m": round(
                        float(
                            rcap_mechanism.get(
                                "mean_state_update_pos_norm_m", float("nan")
                            )
                        ),
                        1,
                    )
                    if math.isfinite(
                        float(
                            rcap_mechanism.get(
                                "mean_state_update_pos_norm_m", float("nan")
                            )
                        )
                    )
                    else None,
                },
                "output_dir": str(out_dir),
            },
            indent=2,
            default=finite_or_none,
        )
    )


if __name__ == "__main__":
    main()
