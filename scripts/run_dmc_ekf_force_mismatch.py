#!/usr/bin/env python
"""Dynamic Model Compensation (DMC) EKF higher-fidelity force-mismatch (Loop 44).

The Loop 41 mechanism diagnostic showed the AUKF misreads dynamics-bias
innovations as measurement-noise evidence under a controlled compact-model
mismatch; Loop 42 extended that to a J3+J4+luni-solar slice and Loop 42 (M3)
further to a J2..J6 + luni-solar + diurnal-bulge slice. Both higher-fidelity
slices fired the cross-filter R-only NIS signature but did not flip the
EKF/AUKF ordering, so the compact-model AUKF prescription is scoped to the
compact dynamics. The reviewer concern for Loop 44 is that the surviving
positive contribution is therefore non-operational.

The operational response prescribed by classical statistical OD when the
dominant residual is a dynamics/force-model bias is a structural
empirical-acceleration channel (dynamic model compensation; Tapley, Schutz,
and Born 2004, Section 10; Wright 1981; Stacey and D'Amico 2021). This
driver evaluates the predeclared 9-state DMC-EKF (compact two-body+J2+drag
deterministic flow augmented with a per-axis first-order Gauss-Markov
empirical-acceleration channel) on the higher-fidelity force-mismatch slice
against EKF/UKF/AUKF/PUKF on the same population.

The decision predicate is predeclared in
``release/predeclarations/dmc_ekf_rule_loop44.json`` BEFORE running this
experiment: DMC-EKF is a positive contribution if and only if (i) its mean
observed-step position RMSE is strictly the lowest among
{EKF, UKF, AUKF, PUKF, DMC-EKF}, AND (ii) the paired-bootstrap 95% CI for
the DMC-EKF-minus-best-non-DMC gap is strictly negative, AND (iii) the gap
exceeds the predeclared 3% practical-significance floor of the best-non-DMC
mean.

Outputs (non-paper-facing):
- results/dmc_ekf_force_mismatch/dmc_ekf_force_mismatch.json
- results/dmc_ekf_force_mismatch/dmc_ekf_force_mismatch.csv

The paper-facing table is rendered by ``scripts/build_paper_assets.py`` from
the JSON.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hifi_force_mismatch as base
import run_hifi_force_mismatch_extended as ext

from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import (
    AdaptiveUKFConfig,
    DMCEKFConfig,
    EKFConfig,
    ProcessNoiseAdaptiveUKFConfig,
    UKFConfig,
    run_adaptive_ukf,
    run_dmc_ekf,
    run_ekf,
    run_process_noise_adaptive_ukf,
    run_ukf,
)
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.utils.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--predeclared-rule",
        default="release/predeclarations/dmc_ekf_rule_loop44.json",
    )
    p.add_argument(
        "--pukf-predeclared-rule",
        default="release/predeclarations/pukf_q_adaptive_rule_loop41.json",
    )
    p.add_argument(
        "--output-json",
        default="results/dmc_ekf_force_mismatch/dmc_ekf_force_mismatch.json",
    )
    p.add_argument(
        "--output-csv",
        default="results/dmc_ekf_force_mismatch/dmc_ekf_force_mismatch.csv",
    )
    p.add_argument("--trajectories", type=int, default=48)
    p.add_argument("--seed", type=int, default=20260544)
    p.add_argument(
        "--epoch-unix",
        type=float,
        default=1_736_640_000.0,
        help="Absolute UTC epoch for the luni-solar ephemerides.",
    )
    p.add_argument(
        "--diurnal-alpha",
        type=float,
        default=0.30,
        help="Diurnal density-bulge amplitude (default 0.30).",
    )
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    dmc_rule = json.loads(Path(args.predeclared_rule).read_text())
    pukf_rule = json.loads(Path(args.pukf_predeclared_rule).read_text())
    th = pukf_rule["thresholds"]
    dmc_th = dmc_rule["thresholds"]

    eval_start = 11
    rng = np.random.default_rng(args.seed)
    n_traj = int(args.trajectories)
    steps = dyn.steps
    dt = dyn.dt_s
    times = np.tile(np.arange(steps, dtype=np.float64) * dt, (n_traj, 1))

    truth_ballistic = dyn.ballistic_coeff_m2_per_kg
    truth_rho_ref = dyn.drag_rho_ref
    truth_h_ref = dyn.drag_h_ref_m
    truth_scale_h = dyn.drag_scale_height_m
    diurnal_alpha = float(args.diurnal_alpha)

    states_all = np.zeros((n_traj, steps, 6), dtype=np.float64)
    meas_all = np.zeros((n_traj, steps, len(stations), 4), dtype=np.float64)
    vis_all = np.zeros((n_traj, steps, len(stations)), dtype=np.float64)
    x0_est_all = np.zeros((n_traj, 6), dtype=np.float64)

    init_pos_sigma = baseline_cfg.ukf.init_pos_std_m
    init_vel_sigma = baseline_cfg.ukf.init_vel_std_mps

    t_gen = time.perf_counter()
    for i in range(n_traj):
        x0 = base._sample_orbit(dataset_cfg.orbit_sampling, rng)
        states = ext._propagate_hifi_extended_trajectory(
            x0,
            dt=dt,
            steps=steps,
            epoch0_unix=args.epoch_unix,
            ballistic_coeff_m2_per_kg=truth_ballistic,
            drag_rho_ref=truth_rho_ref,
            drag_h_ref_m=truth_h_ref,
            drag_scale_height_m=truth_scale_h,
            diurnal_alpha=diurnal_alpha,
        )
        meas, vis = base._generate_observations(
            states=states,
            times=times[i],
            stations=stations,
            noise_std=meas_std,
            outlier_prob=dataset_cfg.measurement_noise.outlier_prob,
            outlier_scale=dataset_cfg.measurement_noise.outlier_scale,
            dropout_prob=dataset_cfg.measurement_noise.random_dropout_prob,
            rng=rng,
        )
        x0_perturb = np.array(
            [
                rng.normal(0.0, init_pos_sigma),
                rng.normal(0.0, init_pos_sigma),
                rng.normal(0.0, init_pos_sigma),
                rng.normal(0.0, init_vel_sigma),
                rng.normal(0.0, init_vel_sigma),
                rng.normal(0.0, init_vel_sigma),
            ]
        )
        states_all[i] = states
        meas_all[i] = meas
        vis_all[i] = vis
        x0_est_all[i] = states[0] + x0_perturb
    t_gen = time.perf_counter() - t_gen

    base_kwargs = dict(
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
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
    ekf_cfg = EKFConfig(
        q_pos_m=baseline_cfg.ekf.q_pos_m,
        q_vel_mps=baseline_cfg.ekf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
        gating_threshold=baseline_cfg.ekf.gating_threshold,
        angle_deweight_elev_cap_deg=getattr(baseline_cfg.ekf, "angle_deweight_elev_cap_deg", None),
    )
    ukf_cfg = UKFConfig(
        q_pos_m=baseline_cfg.ukf.q_pos_m,
        q_vel_mps=baseline_cfg.ukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        angle_deweight_elev_cap_deg=getattr(baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None),
    )
    aukf_cfg = AdaptiveUKFConfig(
        q_pos_m=baseline_cfg.aukf.q_pos_m,
        q_vel_mps=baseline_cfg.aukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.aukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.aukf.init_vel_std_mps,
        alpha=baseline_cfg.aukf.alpha,
        beta=baseline_cfg.aukf.beta,
        kappa=baseline_cfg.aukf.kappa,
        adapt_rate=baseline_cfg.aukf.adapt_rate,
        min_r_scale=baseline_cfg.aukf.min_r_scale,
        max_r_scale=baseline_cfg.aukf.max_r_scale,
        huber_kappa=baseline_cfg.aukf.huber_kappa,
        nis_soft_gate=baseline_cfg.aukf.nis_soft_gate,
        angle_deweight_elev_cap_deg=getattr(baseline_cfg.aukf, "angle_deweight_elev_cap_deg", None),
    )
    pukf_cfg = ProcessNoiseAdaptiveUKFConfig(
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
        angle_deweight_elev_cap_deg=getattr(baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None),
    )
    dmc_cfg = DMCEKFConfig(
        q_pos_m=baseline_cfg.ekf.q_pos_m,
        q_vel_mps=baseline_cfg.ekf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
        init_emp_accel_std_mps2=float(dmc_th["init_emp_accel_std_mps2"]),
        emp_accel_sigma_mps2=float(dmc_th["emp_accel_sigma_mps2"]),
        emp_accel_tau_s=float(dmc_th["emp_accel_tau_s"]),
        gating_threshold=float(dmc_th["gating_threshold"]),
        angle_deweight_elev_cap_deg=getattr(baseline_cfg.ekf, "angle_deweight_elev_cap_deg", None),
    )

    ekf_pred = np.zeros_like(states_all)
    ukf_pred = np.zeros_like(states_all)
    aukf_pred = np.zeros_like(states_all)
    pukf_pred = np.zeros_like(states_all)
    dmc_pred = np.zeros_like(states_all)
    dmc_emp_max: list[float] = []
    pukf_q_scales: list[float] = []

    t_filt = time.perf_counter()
    for i in range(n_traj):
        ekf_pred[i], _ = run_ekf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=ekf_cfg,
            **base_kwargs,
        )
        ukf_pred[i], _ = run_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=ukf_cfg,
            **base_kwargs,
        )
        aukf_pred[i], _ = run_adaptive_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=aukf_cfg,
            **base_kwargs,
        )
        pukf_pred[i], _, pukf_recs = run_process_noise_adaptive_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=pukf_cfg,
            **base_kwargs,
        )
        for rec in pukf_recs:
            pukf_q_scales.append(float(rec["q_scale_used"]))
        dmc_pred[i], _, dmc_diag = run_dmc_ekf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=dmc_cfg,
            **base_kwargs,
        )
        emp_history = dmc_diag.get("empirical_acceleration_mps2")
        if emp_history is not None and emp_history.size:
            dmc_emp_max.append(float(np.max(np.abs(emp_history))))
    t_filt = time.perf_counter() - t_filt

    metrics = {
        "EKF": base._per_traj_observed_pos_rmse(states_all, ekf_pred, vis_all, eval_start),
        "UKF": base._per_traj_observed_pos_rmse(states_all, ukf_pred, vis_all, eval_start),
        "AUKF": base._per_traj_observed_pos_rmse(states_all, aukf_pred, vis_all, eval_start),
        "PUKF": base._per_traj_observed_pos_rmse(states_all, pukf_pred, vis_all, eval_start),
        "DMC_EKF": base._per_traj_observed_pos_rmse(states_all, dmc_pred, vis_all, eval_start),
    }
    means = {k: float(np.nanmean(v)) for k, v in metrics.items()}

    pair_specs = [
        ("DMC_EKF", "EKF"),
        ("DMC_EKF", "UKF"),
        ("DMC_EKF", "AUKF"),
        ("DMC_EKF", "PUKF"),
        ("EKF", "AUKF"),
        ("UKF", "AUKF"),
    ]
    paired: dict[str, dict[str, float]] = {}
    rng_offset = 0
    for cand, baseline in pair_specs:
        diffs = metrics[cand] - metrics[baseline]
        mean_d, lo, hi = base._paired_bootstrap_ci(
            diffs, n_boot=int(args.bootstrap_samples), seed=args.seed + rng_offset
        )
        rng_offset += 1
        p = base._one_sided_wilcoxon_candidate_better(diffs)
        finite = diffs[np.isfinite(diffs)]
        paired[f"{cand}_minus_{baseline}"] = {
            "candidate": cand,
            "baseline": baseline,
            "mean_diff_m": mean_d,
            "ci_lo_m": lo,
            "ci_hi_m": hi,
            "wilcoxon_p_one_sided_candidate_better": p,
            "n_paired": int(finite.size),
            "candidate_better_count": int(np.sum(finite < 0.0)),
        }

    # Decision rule application.
    floor = float(dmc_th["practical_significance_floor"])
    best_non_dmc = min(
        ("EKF", "UKF", "AUKF", "PUKF"), key=lambda k: means[k]
    )
    best_non_dmc_mean = means[best_non_dmc]
    floor_abs = floor * best_non_dmc_mean
    gap_diffs = metrics["DMC_EKF"] - metrics[best_non_dmc]
    gap_mean, gap_lo, gap_hi = base._paired_bootstrap_ci(
        gap_diffs, n_boot=int(args.bootstrap_samples), seed=args.seed + 999
    )
    dmc_is_lowest = means["DMC_EKF"] == min(means.values())
    ci_strictly_negative = gap_hi < 0.0
    floor_exceeded = -gap_mean > floor_abs and gap_mean < 0.0
    is_positive = bool(dmc_is_lowest and ci_strictly_negative and floor_exceeded)

    decision = {
        "best_non_dmc_estimator": best_non_dmc,
        "best_non_dmc_mean_m": best_non_dmc_mean,
        "practical_significance_floor_fraction": floor,
        "practical_significance_floor_abs_m": floor_abs,
        "dmc_minus_best_non_dmc_mean_m": gap_mean,
        "dmc_minus_best_non_dmc_ci_lo_m": gap_lo,
        "dmc_minus_best_non_dmc_ci_hi_m": gap_hi,
        "dmc_is_strictly_lowest_mean": bool(dmc_is_lowest),
        "ci_strictly_negative_for_dmc": bool(ci_strictly_negative),
        "floor_exceeded": bool(floor_exceeded),
        "predeclared_positive_criterion_met": is_positive,
    }

    nis_per_filter: dict[str, dict[str, float]] = {}
    for name, preds in (
        ("EKF", ekf_pred),
        ("UKF", ukf_pred),
        ("AUKF", aukf_pred),
        ("PUKF", pukf_pred),
        ("DMC_EKF", dmc_pred),
    ):
        nis_per_filter[name] = base._cross_filter_r_only_nis(
            states_all,
            preds,
            meas_all,
            vis_all,
            times,
            stations,
            meas_std,
            eval_start,
        )

    payload: dict[str, Any] = {
        "scenario": "dmc_ekf_force_mismatch",
        "schema_version": "dmc_ekf_force_mismatch_v1",
        "scope": (
            "Dynamic-model-compensation EKF on higher-fidelity force-mismatch: "
            "truth uses two-body + J2..J6 zonal geopotential + luni-solar third-body "
            "acceleration + exponential drag with a one-sided diurnal-bulge density "
            f"modulation (alpha={diurnal_alpha:.2f}); EKF/UKF/AUKF/PUKF/DMC-EKF all "
            "use the compact two-body+J2+drag deterministic flow; DMC-EKF augments "
            "the state with a per-axis first-order Gauss-Markov empirical-acceleration "
            f"channel (sigma={dmc_th['emp_accel_sigma_mps2']:.1e} m/s^2, "
            f"tau={dmc_th['emp_accel_tau_s']:.0f} s)."
        ),
        "n_trajectories": int(n_traj),
        "steps": int(steps),
        "dt_s": float(dt),
        "eval_start_step": int(eval_start),
        "epoch_unix": float(args.epoch_unix),
        "rng_seed": int(args.seed),
        "diurnal_alpha": diurnal_alpha,
        "bootstrap_samples": int(args.bootstrap_samples),
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "pukf_predeclared_rule_path": args.pukf_predeclared_rule,
        "truth_acceleration": (
            "accel_hifi_extended(r, epoch_unix) (J2..J6 + luni-solar) + "
            "exponential drag with diurnal-bulge density modulation"
        ),
        "estimator_acceleration": "compact two-body + J2 + time-invariant exponential drag",
        "dmc_augmented_state": "[r, v, w] with w a per-axis Gauss-Markov empirical-acceleration",
        "observed_step_rmse_mean_m": means,
        "paired": paired,
        "decision": decision,
        "dmc_diagnostics": {
            "max_abs_empirical_acceleration_mps2": (
                float(np.max(dmc_emp_max)) if dmc_emp_max else float("nan")
            ),
            "median_max_abs_empirical_acceleration_mps2": (
                float(np.median(dmc_emp_max)) if dmc_emp_max else float("nan")
            ),
        },
        "pukf_diagnostics": {
            "mean_q_scale": float(np.mean(pukf_q_scales)) if pukf_q_scales else float("nan"),
            "median_q_scale": float(np.median(pukf_q_scales)) if pukf_q_scales else float("nan"),
            "max_q_scale": float(np.max(pukf_q_scales)) if pukf_q_scales else float("nan"),
        },
        "cross_filter_r_only_nis": nis_per_filter,
        "elapsed_seconds": {"truth_generation": float(t_gen), "filters": float(t_filt)},
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    rows = []
    for i in range(n_traj):
        row: dict[str, Any] = {"trajectory_index": i}
        for k, arr in metrics.items():
            row[f"{k}_observed_pos_rmse_m"] = float(arr[i]) if np.isfinite(arr[i]) else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(json.dumps(
        {
            "n_trajectories": int(n_traj),
            "observed_step_rmse_mean_m": means,
            "decision": decision,
            "json": str(out_json),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
