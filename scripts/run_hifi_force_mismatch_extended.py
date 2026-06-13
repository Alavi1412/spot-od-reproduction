#!/usr/bin/env python
"""Extended higher-fidelity force-mismatch experiment.

The earlier higher-fidelity force-mismatch slice extended the truth-side
dynamics above the compact-J2 ceiling by adding the J3 and J4 zonal
geopotential terms and the luni-solar third-body acceleration. A reviewer
asked for one yet richer fidelity tier so the AUKF mechanism diagnostic can
be probed beyond a single second-fidelity slice. This driver implements an
additive next step:

* The truth-side conservative gravity is extended from J2..J4 to J2..J6
  (the well-established EGM-class nominal zonal coefficients) using the
  analytic ``zonal_acceleration_extended`` gradient (independently validated
  against a finite-difference of the corresponding zonal potential).
* The truth-side drag adds a diurnal-bulge density modulation:
  ``rho_truth = rho_exp(altitude) * (1 + alpha * max(0, n_sun . r_hat))``,
  with ``alpha = 0.3``. The bulge is one-sided so it cannot be absorbed
  into a constant rescaling. The estimator continues to use the nominal
  time-invariant exponential-density drag.
* The estimator-side compact two-body+J2+drag model is unchanged.

Outputs (non-paper-facing):
- ``results/hifi_force_mismatch_extended/hifi_force_mismatch_extended.json``
- ``results/hifi_force_mismatch_extended/hifi_force_mismatch_extended.csv``

This script reuses every other component (orbit sampling, measurements,
classical filter set, paired bootstrap, AUKF cross-filter NIS) from the
earlier higher-fidelity driver so the new slice is directly comparable to
the earlier higher-fidelity table.
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
from scipy.stats import wilcoxon

import run_hifi_force_mismatch as base

from gnn_state_estimation.constants import EARTH_ROTATION_RATE
from gnn_state_estimation.coordinates import line_of_sight_measurement
from gnn_state_estimation.dynamics import atmospheric_density_exponential
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import (
    EKFConfig,
    UKFConfig,
    AdaptiveUKFConfig,
    ProcessNoiseAdaptiveUKFConfig,
    run_ekf,
    run_ukf,
    run_adaptive_ukf,
    run_process_noise_adaptive_ukf,
)
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.sp3 import accel_hifi_extended, sun_position_eci_m
from gnn_state_estimation.utils.io import load_yaml


# --- Truth-side extended higher-fidelity propagator ----------------------


def _drag_diurnal_acceleration(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    epoch_unix: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    diurnal_alpha: float,
) -> np.ndarray:
    """Exponential-atmosphere drag with a one-sided diurnal density bulge.

    The bulge factor ``(1 + alpha * max(0, n_sun . r_hat))`` adds a day-side
    density excess that cannot be absorbed by a constant rescaling because
    it changes sign through the orbit; the estimator uses the time-invariant
    exponential density and so cannot capture this perturbation from inside
    its model class.
    """
    r = float(np.linalg.norm(r_eci))
    if r < 1.0:
        return np.zeros(3, dtype=np.float64)
    altitude = r - 6378.1363e3
    rho_base = atmospheric_density_exponential(
        altitude_m=altitude,
        rho_ref=drag_rho_ref,
        h_ref_m=drag_h_ref_m,
        scale_height_m=drag_scale_height_m,
    )
    r_sun = sun_position_eci_m(epoch_unix)
    sun_norm = float(np.linalg.norm(r_sun))
    if sun_norm < 1.0:
        bulge = 0.0
    else:
        cos_theta = float(np.dot(r_eci, r_sun)) / (r * sun_norm)
        bulge = max(0.0, cos_theta)
    rho = rho_base * (1.0 + diurnal_alpha * bulge)
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE], dtype=np.float64)
    v_atm = np.cross(omega, r_eci)
    v_rel = v_eci - v_atm
    v_rel_norm = float(np.linalg.norm(v_rel))
    return -0.5 * ballistic_coeff_m2_per_kg * rho * v_rel_norm * v_rel


def _hifi_extended_state_derivative(
    state: np.ndarray,
    epoch_unix: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    diurnal_alpha: float,
) -> np.ndarray:
    r = state[:3]
    v = state[3:]
    a = accel_hifi_extended(r, epoch_unix)
    a = a + _drag_diurnal_acceleration(
        r,
        v,
        epoch_unix,
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
        diurnal_alpha=diurnal_alpha,
    )
    return np.hstack([v, a]).astype(np.float64)


def _hifi_extended_rk4_step(
    state: np.ndarray,
    dt: float,
    epoch_unix: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    diurnal_alpha: float,
) -> np.ndarray:
    kwargs = dict(
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
        diurnal_alpha=diurnal_alpha,
    )
    k1 = _hifi_extended_state_derivative(state, epoch_unix, **kwargs)
    k2 = _hifi_extended_state_derivative(state + 0.5 * dt * k1, epoch_unix + 0.5 * dt, **kwargs)
    k3 = _hifi_extended_state_derivative(state + 0.5 * dt * k2, epoch_unix + 0.5 * dt, **kwargs)
    k4 = _hifi_extended_state_derivative(state + dt * k3, epoch_unix + dt, **kwargs)
    return (state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float64)


def _propagate_hifi_extended_trajectory(
    x0: np.ndarray,
    dt: float,
    steps: int,
    epoch0_unix: float,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float,
    drag_h_ref_m: float,
    drag_scale_height_m: float,
    diurnal_alpha: float,
) -> np.ndarray:
    states = np.zeros((steps, 6), dtype=np.float64)
    states[0] = np.asarray(x0, dtype=np.float64)
    for k in range(1, steps):
        states[k] = _hifi_extended_rk4_step(
            states[k - 1],
            dt=dt,
            epoch_unix=epoch0_unix + (k - 1) * dt,
            ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
            drag_rho_ref=drag_rho_ref,
            drag_h_ref_m=drag_h_ref_m,
            drag_scale_height_m=drag_scale_height_m,
            diurnal_alpha=diurnal_alpha,
        )
    return states


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--predeclared-rule",
        default="release/predeclarations/pukf_q_adaptive_rule_loop41.json",
    )
    p.add_argument(
        "--output-json",
        default="results/hifi_force_mismatch_extended/hifi_force_mismatch_extended.json",
    )
    p.add_argument(
        "--output-csv",
        default="results/hifi_force_mismatch_extended/hifi_force_mismatch_extended.csv",
    )
    p.add_argument("--trajectories", type=int, default=48)
    p.add_argument("--seed", type=int, default=20260530)
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
        help="Diurnal density-bulge amplitude (default 0.30 = +-30 percent dayside density excess).",
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
    rule = json.loads(Path(args.predeclared_rule).read_text())
    th = rule["thresholds"]

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
        states = _propagate_hifi_extended_trajectory(
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

    ekf_kwargs = dict(
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

    ekf_pred = np.zeros_like(states_all)
    ukf_pred = np.zeros_like(states_all)
    aukf_pred = np.zeros_like(states_all)
    pukf_pred = np.zeros_like(states_all)
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
            **ekf_kwargs,
        )
        ukf_pred[i], _ = run_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=ukf_cfg,
            **ekf_kwargs,
        )
        aukf_pred[i], _ = run_adaptive_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=aukf_cfg,
            **ekf_kwargs,
        )
        pukf_pred[i], _, pukf_recs = run_process_noise_adaptive_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=pukf_cfg,
            **ekf_kwargs,
        )
        for rec in pukf_recs:
            pukf_q_scales.append(float(rec["q_scale_used"]))
    t_filt = time.perf_counter() - t_filt

    metrics = {
        "EKF": base._per_traj_observed_pos_rmse(states_all, ekf_pred, vis_all, eval_start),
        "UKF": base._per_traj_observed_pos_rmse(states_all, ukf_pred, vis_all, eval_start),
        "AUKF": base._per_traj_observed_pos_rmse(states_all, aukf_pred, vis_all, eval_start),
        "PUKF": base._per_traj_observed_pos_rmse(states_all, pukf_pred, vis_all, eval_start),
    }
    means = {k: float(np.nanmean(v)) for k, v in metrics.items()}

    paired: dict[str, dict[str, float]] = {}
    pair_specs = [
        ("EKF", "AUKF"),
        ("EKF", "UKF"),
        ("UKF", "AUKF"),
        ("PUKF", "EKF"),
        ("PUKF", "UKF"),
        ("PUKF", "AUKF"),
    ]
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

    nis_per_filter: dict[str, dict[str, float]] = {}
    for name, preds in (
        ("EKF", ekf_pred),
        ("UKF", ukf_pred),
        ("AUKF", aukf_pred),
        ("PUKF", pukf_pred),
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
        "scenario": "hifi_force_mismatch_extended",
        "schema_version": "hifi_force_mismatch_extended_v1",
        "scope": (
            "Extended higher-fidelity force-model mismatch: truth uses two-body + "
            "J2..J6 zonal geopotential + luni-solar third-body acceleration + "
            "exponential-atmosphere drag with a one-sided diurnal-bulge "
            f"density modulation (alpha={diurnal_alpha:.2f}); estimators use the "
            "nominal compact two-body+J2+drag model with a time-invariant "
            "exponential density. The missing-physics gap is exclusively in "
            "the conservative force field and the time-varying density and is "
            "physically richer than the earlier J3+J4+luni-solar slice."
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
        "truth_acceleration": (
            "accel_hifi_extended(r, epoch_unix) (J2..J6 + luni-solar) + "
            "exponential drag with diurnal-bulge density modulation"
        ),
        "estimator_acceleration": "compact two-body + J2 + time-invariant exponential drag",
        "observed_step_rmse_mean_m": means,
        "paired": paired,
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
            "ekf_minus_aukf_m": paired["EKF_minus_AUKF"]["mean_diff_m"],
            "ekf_minus_aukf_ci": [
                paired["EKF_minus_AUKF"]["ci_lo_m"],
                paired["EKF_minus_AUKF"]["ci_hi_m"],
            ],
            "median_r_only_nis": {k: v["median"] for k, v in nis_per_filter.items()},
            "json": str(out_json),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
