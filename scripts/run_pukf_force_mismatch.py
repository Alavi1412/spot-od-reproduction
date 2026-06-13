#!/usr/bin/env python
"""Process-noise-adaptive UKF (PUKF) evaluation on the controlled force-mismatch slice.

Implements the loop 41 predeclared rule (release/predeclarations/
pukf_q_adaptive_rule_loop41.json) and scores it against the cached
EKF/UKF/AUKF priors for the controlled force-model mismatch scenario, on the
same trajectory population and observed-step convention as the AUKF
mechanism diagnostic.

This script is non-paper-facing: outputs go to results/pukf_force_mismatch_loop41.json
and a table-ready summary CSV. The paper-facing wording is honest about the
predeclared rule and the outcome of the evaluation.
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
from scipy.stats import wilcoxon  # type: ignore[import-untyped]

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import (
    ProcessNoiseAdaptiveUKFConfig,
    run_process_noise_adaptive_ukf,
)
from gnn_state_estimation.scenarios import (
    estimator_sim_config,
    truth_sim_config,
)
from gnn_state_estimation.simulation import DatasetConfig, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_sim_configs(cfg: dict[str, Any], scenario: str):
    sim_cfg = copy.deepcopy(cfg["simulation"])
    if scenario == "stress_test":
        sim_cfg = _deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        return sim_cfg, sim_cfg, None
    return (
        estimator_sim_config(sim_cfg, scenario_cfg),
        truth_sim_config(sim_cfg, scenario_cfg),
        scenario_cfg,
    )


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
    diffs: np.ndarray, n_boot: int = 5000, seed: int = 20260520, alpha: float = 0.05
) -> tuple[float, float, float]:
    finite = np.asarray(diffs, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    n = finite.size
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[b] = float(np.mean(finite[idx]))
    return (
        float(np.mean(finite)),
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1.0 - alpha / 2)),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--scenario", default="force_model_mismatch_test")
    p.add_argument("--predeclared-rule", default="release/predeclarations/pukf_q_adaptive_rule_loop41.json")
    p.add_argument("--output-json", default="results/pukf_force_mismatch_loop41.json")
    p.add_argument("--trajectory-limit", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    scenario = args.scenario

    rule = json.loads(Path(args.predeclared_rule).read_text())
    th = rule["thresholds"]

    est_sim, truth_sim, _scenario_cfg = _resolve_sim_configs(cfg, scenario)
    dataset_cfg: DatasetConfig = parse_dataset_config(est_sim)
    dyn = dataset_cfg.dynamics
    meas_std = dataset_cfg.measurement_noise.std_vector

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    arrays = load_dataset_npz(Path(cfg["data"]["output_dir"]) / f"{scenario}.npz")
    if arrays.x0_estimates is None:
        raise ValueError(f"{scenario}.npz missing x0_estimates")

    states = arrays.states
    meas = arrays.measurements
    vis = arrays.visibility
    times = arrays.times
    n_traj = states.shape[0]
    if args.trajectory_limit > 0:
        n_traj = min(args.trajectory_limit, n_traj)

    pukf_cfg_kwargs = dict(
        q_pos_m=baseline_cfg.ukf.q_pos_m,
        q_vel_mps=baseline_cfg.ukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        window_size=int(th["window_size"]),
        nis_per_update_expected=float(th["nis_per_update_expected"]),
        nis_warn_ratio=float(th["nis_warn_ratio"]),
        nis_alarm_ratio=float(th["nis_alarm_ratio"]),
        q_scale_warn=float(th["q_scale_warn"]),
        q_scale_alarm=float(th["q_scale_alarm"]),
        q_scale_max=float(th["q_scale_max"]),
        smoothing=float(th["smoothing"]),
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
        ),
    )
    pukf_cfg = ProcessNoiseAdaptiveUKFConfig(**pukf_cfg_kwargs)

    pukf_pred = np.zeros_like(states[:n_traj])
    q_scale_records: list[float] = []

    t0 = time.perf_counter()
    for i in range(n_traj):
        x_hist, _p_hist, recs = run_process_noise_adaptive_ukf(
            measurements=meas[i],
            visibility=vis[i],
            times_s=times[i],
            stations=dataset_cfg.stations,
            ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
            meas_std_vector=meas_std,
            x0_est=arrays.x0_estimates[i],
            cfg=pukf_cfg,
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
        pukf_pred[i] = x_hist
        for rec in recs:
            q_scale_records.append(float(rec["q_scale_used"]))
    elapsed = time.perf_counter() - t0

    pukf_rmse = _per_traj_observed_pos_rmse(states[:n_traj], pukf_pred, vis[:n_traj], eval_start)
    cached = {
        "EKF": arrays.ekf_prior,
        "UKF": arrays.ukf_prior,
        "AUKF": arrays.aukf_prior,
    }
    per_traj = {"PUKF": pukf_rmse}
    means = {"PUKF": float(np.nanmean(pukf_rmse))}
    for name, prior in cached.items():
        if prior is None:
            continue
        per_traj[name] = _per_traj_observed_pos_rmse(
            states[:n_traj], prior[:n_traj], vis[:n_traj], eval_start
        )
        means[name] = float(np.nanmean(per_traj[name]))

    # PUKF vs each cached comparator: paired difference (PUKF - other);
    # negative => PUKF better.
    paired = {}
    for name, arr in per_traj.items():
        if name == "PUKF":
            continue
        diffs = pukf_rmse - arr
        mean, lo, hi = _paired_bootstrap_ci(
            diffs, n_boot=int(args.bootstrap_samples), seed=20260520
        )
        valid = diffs[np.isfinite(diffs)]
        try:
            wstat = wilcoxon(valid, alternative="less", zero_method="wilcox")
            wp = float(wstat.pvalue)
        except Exception:
            wp = float("nan")
        paired[name] = {
            "mean_pukf_minus_other_m": mean,
            "ci_lo_m": lo,
            "ci_hi_m": hi,
            "wilcoxon_p_one_sided_pukf_better": wp,
            "n_paired": int(valid.size),
            "pukf_better_count": int(np.sum(valid < 0)),
        }

    best_other = min((m for n, m in means.items() if n != "PUKF"), default=float("nan"))
    best_other_name = (
        min(
            ((n, m) for n, m in means.items() if n != "PUKF"),
            key=lambda kv: kv[1],
            default=("", float("nan")),
        )[0]
    )
    pukf_beats_all = (
        all(means["PUKF"] < m for n, m in means.items() if n != "PUKF") if means else False
    )
    # Predeclared decision: positive iff PUKF mean is strictly best AND its
    # paired CI vs every comparator is strictly negative.
    ci_strict = all(
        p["ci_hi_m"] is not None and p["ci_hi_m"] < 0.0 for p in paired.values()
    )
    predeclared_positive = bool(pukf_beats_all and ci_strict)

    payload = {
        "scenario": scenario,
        "n_trajectories": int(n_traj),
        "eval_start_step": int(eval_start),
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": None,
        "elapsed_seconds": float(elapsed),
        "q_scale_summary": {
            "n_steps_recorded": int(len(q_scale_records)),
            "mean_q_scale": float(np.mean(q_scale_records)) if q_scale_records else float("nan"),
            "median_q_scale": float(np.median(q_scale_records))
            if q_scale_records
            else float("nan"),
            "p90_q_scale": float(np.percentile(q_scale_records, 90.0))
            if q_scale_records
            else float("nan"),
            "max_q_scale": float(np.max(q_scale_records)) if q_scale_records else float("nan"),
        },
        "observed_step_rmse_mean_m": means,
        "paired_pukf_vs_other": paired,
        "best_non_pukf": {"name": best_other_name, "mean_rmse_m": float(best_other)},
        "predeclared_positive": predeclared_positive,
        "outcome": (
            "positive" if predeclared_positive else "bounded_negative"
        ),
    }

    import hashlib
    payload["predeclared_rule_digest_sha256"] = hashlib.sha256(
        Path(args.predeclared_rule).read_bytes()
    ).hexdigest()

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
