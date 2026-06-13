#!/usr/bin/env python
"""Bounded Ensemble Kalman Filter (EnKF) comparator (Task EVID-ENKF, loop 163).

Closes the Ensemble-Kalman-family exclusion risk in the claim audit. The
manuscript compares EKF/UKF/AUKF/PUKF and several structural-channel
augmentations, but the ensemble Gaussian filter -- a major nonlinear-filtering
family -- was not represented. This driver evaluates a bounded, untuned,
canonical *stochastic* (perturbed-observation) EnKF on the manuscript's own
trajectory populations against its own cached EKF/UKF/AUKF priors, under the
frozen predeclared rule recorded in
``results/loop163_enkf_comparator/enkf_predeclared_rule_loop163.json``.

For each scenario (force-mismatch primary; nominal/stress as context) the EnKF
is run fresh with the *estimator-side* compact dynamics (the same resolution
used by every recursive filter in this study), the per-trajectory observed-step
position RMSE is computed for the EnKF and for the cached EKF/UKF/AUKF priors,
and the EnKF-vs-comparator gaps are summarized with paired trajectory-bootstrap
95% CIs and a one-sided Wilcoxon signed-rank descriptive p-value. The
predeclared positive criterion is applied to the force-mismatch slice; any
non-positive outcome is reported as a bounded Ensemble-Kalman-family negative.

This script is non-paper-facing. Outputs:
- results/loop163_enkf_comparator/enkf_comparator.json
- results/loop163_enkf_comparator/enkf_comparator.csv
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon  # type: ignore[import-untyped]

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import EnKFConfig, run_enkf
from gnn_state_estimation.scenarios import estimator_sim_config
from gnn_state_estimation.simulation import DatasetConfig, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


SCENARIOS = (
    ("force_model_mismatch_test", "force_mismatch", True),
    ("test", "nominal", False),
    ("stress_test", "stress", False),
)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_estimator_sim(cfg: dict[str, Any], scenario: str) -> dict[str, Any]:
    """Estimator-side simulation config, replicating the convention used by
    scripts/run_pukf_force_mismatch.py so the EnKF sees exactly the same
    compact model the cached EKF/UKF/AUKF priors were computed with."""
    sim_cfg = copy.deepcopy(cfg["simulation"])
    if scenario == "stress_test":
        sim_cfg = _deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        return sim_cfg
    return estimator_sim_config(sim_cfg, scenario_cfg)


def _per_traj_observed_pos_rmse(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> np.ndarray:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    y_true = states[:, eval_start:, :3]
    y_pred = preds[:, eval_start:, :3]
    out = np.full(states.shape[0], np.nan, dtype=np.float64)
    for i in range(states.shape[0]):
        mask = observed[i]
        if not np.any(mask):
            continue
        err = y_true[i, mask] - y_pred[i, mask]
        out[i] = float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))
    return out


def _paired_bootstrap_ci(
    diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    finite = np.asarray(diffs, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = finite.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = float(np.mean(finite[idx]))
    return (
        float(np.mean(finite)),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1.0 - alpha / 2)),
    )


def _one_sided_wilcoxon_candidate_better(diffs: np.ndarray) -> float:
    finite = diffs[np.isfinite(diffs)]
    if finite.size == 0 or np.allclose(finite, 0.0):
        return float("nan")
    try:
        return float(wilcoxon(finite, alternative="less", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--predeclared-rule",
        default="results/loop163_enkf_comparator/enkf_predeclared_rule_loop163.json",
    )
    p.add_argument(
        "--output-json",
        default="results/loop163_enkf_comparator/enkf_comparator.json",
    )
    p.add_argument(
        "--output-csv",
        default="results/loop163_enkf_comparator/enkf_comparator.csv",
    )
    p.add_argument("--trajectory-limit", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    return p


def _score_scenario(
    cfg: dict[str, Any],
    scenario: str,
    th: dict[str, Any],
    eval_start: int,
    baseline_cfg: Any,
    n_boot: int,
    traj_limit: int,
) -> dict[str, Any]:
    est_sim = _resolve_estimator_sim(cfg, scenario)
    dataset_cfg: DatasetConfig = parse_dataset_config(est_sim)
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector

    arrays = load_dataset_npz(Path(cfg["data"]["output_dir"]) / f"{scenario}.npz")
    if arrays.x0_estimates is None:
        raise ValueError(f"{scenario}.npz missing x0_estimates")

    states = arrays.states
    meas = arrays.measurements
    vis = arrays.visibility
    times = arrays.times
    n_traj = states.shape[0]
    if traj_limit > 0:
        n_traj = min(traj_limit, n_traj)

    base_seed = int(th["seed"])
    enkf_pred = np.zeros_like(states[:n_traj])
    pos_spread_means: list[float] = []

    t0 = time.perf_counter()
    for i in range(n_traj):
        enkf_cfg = EnKFConfig(
            q_pos_m=float(th["q_pos_m"]),
            q_vel_mps=float(th["q_vel_mps"]),
            init_pos_std_m=float(th["init_pos_std_m"]),
            init_vel_std_mps=float(th["init_vel_std_mps"]),
            ensemble_size=int(th["ensemble_size"]),
            inflation=float(th["inflation"]),
            seed=base_seed + i,
            angle_deweight_elev_cap_deg=getattr(
                baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
            ),
        )
        x_hist, _p_hist, diag = run_enkf(
            measurements=meas[i],
            visibility=vis[i],
            times_s=times[i],
            stations=stations,
            ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
            meas_std_vector=meas_std,
            x0_est=arrays.x0_estimates[i],
            cfg=enkf_cfg,
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
        enkf_pred[i] = x_hist
        pos_spread_means.append(float(diag["mean_pos_spread_m"]))
    elapsed = time.perf_counter() - t0

    per_traj: dict[str, np.ndarray] = {
        "EnKF": _per_traj_observed_pos_rmse(states[:n_traj], enkf_pred, vis[:n_traj], eval_start)
    }
    cached = {"EKF": arrays.ekf_prior, "UKF": arrays.ukf_prior, "AUKF": arrays.aukf_prior}
    for name, prior in cached.items():
        if prior is None:
            continue
        per_traj[name] = _per_traj_observed_pos_rmse(
            states[:n_traj], prior[:n_traj], vis[:n_traj], eval_start
        )
    means = {k: float(np.nanmean(v)) for k, v in per_traj.items()}

    # EnKF vs each comparator (EnKF - other; negative => EnKF better).
    paired: dict[str, dict[str, float]] = {}
    rng_offset = 0
    for name, arr in per_traj.items():
        if name == "EnKF":
            continue
        diffs = per_traj["EnKF"] - arr
        mean_d, lo, hi = _paired_bootstrap_ci(
            diffs, n_boot=n_boot, seed=base_seed + rng_offset
        )
        rng_offset += 1
        wp = _one_sided_wilcoxon_candidate_better(diffs)
        finite = diffs[np.isfinite(diffs)]
        paired[name] = {
            "mean_enkf_minus_other_m": mean_d,
            "ci_lo_m": lo,
            "ci_hi_m": hi,
            "wilcoxon_p_one_sided_enkf_better": wp,
            "n_paired": int(finite.size),
            "enkf_better_count": int(np.sum(finite < 0.0)),
        }

    floor = float(th["practical_significance_floor"])
    non_enkf = {k: v for k, v in means.items() if k != "EnKF"}
    best_non_enkf = min(non_enkf, key=lambda k: non_enkf[k]) if non_enkf else ""
    best_non_enkf_mean = non_enkf.get(best_non_enkf, float("nan"))
    gap_diffs = per_traj["EnKF"] - per_traj[best_non_enkf] if best_non_enkf else np.array([])
    gap_mean, gap_lo, gap_hi = _paired_bootstrap_ci(
        gap_diffs, n_boot=n_boot, seed=base_seed + 999
    )
    enkf_is_lowest = bool(means["EnKF"] == min(means.values()))
    ci_strictly_negative = bool(np.isfinite(gap_hi) and gap_hi < 0.0)
    floor_abs = floor * best_non_enkf_mean
    floor_exceeded = bool(np.isfinite(gap_mean) and gap_mean < 0.0 and -gap_mean > floor_abs)
    is_positive = bool(enkf_is_lowest and ci_strictly_negative and floor_exceeded)

    decision = {
        "best_non_enkf_estimator": best_non_enkf,
        "best_non_enkf_mean_m": best_non_enkf_mean,
        "practical_significance_floor_fraction": floor,
        "practical_significance_floor_abs_m": floor_abs,
        "enkf_minus_best_non_enkf_mean_m": gap_mean,
        "enkf_minus_best_non_enkf_ci_lo_m": gap_lo,
        "enkf_minus_best_non_enkf_ci_hi_m": gap_hi,
        "enkf_is_strictly_lowest_mean": enkf_is_lowest,
        "ci_strictly_negative_for_enkf": ci_strictly_negative,
        "floor_exceeded": floor_exceeded,
        "predeclared_positive_criterion_met": is_positive,
    }

    return {
        "scenario": scenario,
        "n_trajectories": int(n_traj),
        "eval_start_step": int(eval_start),
        "stations": int(len(stations)),
        "observed_step_rmse_mean_m": means,
        "paired_enkf_vs_other": paired,
        "decision": decision,
        "enkf_diagnostics": {
            "ensemble_size": int(th["ensemble_size"]),
            "inflation": float(th["inflation"]),
            "mean_pos_spread_m": float(np.mean(pos_spread_means)) if pos_spread_means else float("nan"),
        },
        "elapsed_seconds": float(elapsed),
        "_per_traj": {k: v for k, v in per_traj.items()},
    }


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    rule = json.loads(Path(args.predeclared_rule).read_text())
    th = rule["thresholds"]

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    scenario_results: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for scenario, label, _is_primary in SCENARIOS:
        res = _score_scenario(
            cfg,
            scenario,
            th,
            eval_start,
            baseline_cfg,
            int(args.bootstrap_samples),
            int(args.trajectory_limit),
        )
        per_traj = res.pop("_per_traj")
        n = res["n_trajectories"]
        for i in range(n):
            row: dict[str, Any] = {"scenario": scenario, "label": label, "trajectory_index": i}
            for k, arr in per_traj.items():
                row[f"{k}_observed_pos_rmse_m"] = (
                    float(arr[i]) if np.isfinite(arr[i]) else None
                )
            csv_rows.append(row)
        scenario_results[scenario] = res
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "label": label,
                    "observed_step_rmse_mean_m": res["observed_step_rmse_mean_m"],
                    "predeclared_positive": res["decision"]["predeclared_positive_criterion_met"],
                },
                indent=2,
            )
        )

    primary = scenario_results["force_model_mismatch_test"]
    overall_positive = bool(primary["decision"]["predeclared_positive_criterion_met"])

    payload: dict[str, Any] = {
        "task_id": "EVID-ENKF",
        "scenario": "enkf_comparator_loop163",
        "schema_version": "enkf_comparator_v1",
        "scope": rule["scope"],
        "primary_scenario": "force_model_mismatch_test",
        "context_scenarios": ["test", "stress_test"],
        "comparator_source": rule["evaluation_protocol"]["comparator_source"],
        "eval_start_step": int(eval_start),
        "bootstrap_samples": int(args.bootstrap_samples),
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "estimator_acceleration": "compact two-body + J2 + exponential-atmosphere drag (estimator-side resolution; identical to the cached EKF/UKF/AUKF priors)",
        "enkf_representation": "stochastic perturbed-observation ensemble (Evensen 1994/2003; Burgers et al. 1998)",
        "scenarios": scenario_results,
        "primary_decision": primary["decision"],
        "overall_predeclared_positive": overall_positive,
        "outcome": "positive" if overall_positive else "bounded_enkf_family_negative",
        "exclusion_risk_closed": True,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    print(
        json.dumps(
            {
                "outcome": payload["outcome"],
                "primary_means_m": primary["observed_step_rmse_mean_m"],
                "primary_decision": primary["decision"],
                "json": str(out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
