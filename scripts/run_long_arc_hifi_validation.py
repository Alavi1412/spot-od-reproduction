#!/usr/bin/env python
"""Long-arc DSA-EKF validation tuning over the predeclared grid (Loop 47).

Runs the Drag-Scale Adaptive EKF on a long-arc validation seed disjoint from
the held-out test seed and from every other split used in the manuscript.
Evaluates the predeclared small grid of Gauss-Markov hyperparameters and
writes a selection artefact recording the lowest-mean-observed-step-RMSE
grid point. The selection artefact is later loaded by the held-out test
harness ``scripts/run_long_arc_hifi_force_mismatch.py`` so the test
population never enters the hyperparameter selection.
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

import run_hifi_force_mismatch as base
import run_long_arc_hifi_force_mismatch as la

from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import DragScaleAEKFConfig, run_drag_scale_aekf
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.utils.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--predeclared-rule",
        default="release/predeclarations/long_arc_hifi_rule_loop47.json",
    )
    p.add_argument(
        "--output-json",
        default="results/long_arc_hifi_validation/long_arc_hifi_validation.json",
    )
    p.add_argument("--trajectories", type=int, default=None)
    p.add_argument("--epoch-unix", type=float, default=1_736_640_000.0)
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
    vt = rule["validation_tuning"]
    th = rule["thresholds"]
    arc = rule["arc"]
    grid = vt["tuning_grid"]
    seed = int(vt["validation_seed"])
    n_traj = int(args.trajectories) if args.trajectories else int(vt["validation_n_trajectories"])
    steps = int(arc["steps"])
    dt = float(arc["dt_s"])
    eval_start = int(arc["eval_start_step"])
    diurnal_alpha = float(th["alpha_diurnal"])
    longitudinal_alpha = float(th["alpha_longitudinal"])
    longitudinal_wavenumber = float(th["longitudinal_wavenumber"])

    rng = np.random.default_rng(seed)
    times = np.tile(np.arange(steps, dtype=np.float64) * dt, (n_traj, 1))
    init_pos_sigma = baseline_cfg.ukf.init_pos_std_m
    init_vel_sigma = baseline_cfg.ukf.init_vel_std_mps

    states_all = np.zeros((n_traj, steps, 6), dtype=np.float64)
    meas_all = np.zeros((n_traj, steps, len(stations), 4), dtype=np.float64)
    vis_all = np.zeros((n_traj, steps, len(stations)), dtype=np.float64)
    x0_est_all = np.zeros((n_traj, 6), dtype=np.float64)

    t_gen = time.perf_counter()
    for i in range(n_traj):
        x0 = base._sample_orbit(dataset_cfg.orbit_sampling, rng)
        states = la.propagate_long_arc_trajectory(
            x0,
            dt=dt,
            steps=steps,
            epoch0_unix=args.epoch_unix,
            ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
            drag_rho_ref=dyn.drag_rho_ref,
            drag_h_ref_m=dyn.drag_h_ref_m,
            drag_scale_height_m=dyn.drag_scale_height_m,
            diurnal_alpha=diurnal_alpha,
            longitudinal_alpha=longitudinal_alpha,
            longitudinal_wavenumber=longitudinal_wavenumber,
        )
        meas, vis = base._generate_observations(
            states=states, times=times[i], stations=stations,
            noise_std=meas_std,
            outlier_prob=dataset_cfg.measurement_noise.outlier_prob,
            outlier_scale=dataset_cfg.measurement_noise.outlier_scale,
            dropout_prob=dataset_cfg.measurement_noise.random_dropout_prob,
            rng=rng,
        )
        x0_perturb = np.array(
            [rng.normal(0.0, init_pos_sigma) for _ in range(3)]
            + [rng.normal(0.0, init_vel_sigma) for _ in range(3)]
        )
        states_all[i] = states
        meas_all[i] = meas
        vis_all[i] = vis
        x0_est_all[i] = states[0] + x0_perturb
        print(f"[validation] traj {i+1}/{n_traj} truth generated", flush=True)
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

    results = []
    t_grid = time.perf_counter()
    for gp in grid:
        cfg_dsa = DragScaleAEKFConfig(
            q_pos_m=float(th["q_pos_m"]),
            q_vel_mps=float(th["q_vel_mps"]),
            init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
            init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
            init_drag_scale_std=float(gp["init_drag_scale_std"]),
            drag_scale_sigma_ss=float(gp["drag_scale_sigma_ss"]),
            drag_scale_tau_s=float(gp["drag_scale_tau_s"]),
            gating_threshold=float(th["gating_threshold"]),
            angle_deweight_elev_cap_deg=getattr(baseline_cfg.ekf, "angle_deweight_elev_cap_deg", None),
        )
        preds = np.zeros_like(states_all)
        for i in range(n_traj):
            preds[i], _, _ = run_drag_scale_aekf(
                measurements=meas_all[i], visibility=vis_all[i], times_s=times[i],
                stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
                cfg=cfg_dsa, **base_kwargs,
            )
        per_traj = base._per_traj_observed_pos_rmse(states_all, preds, vis_all, eval_start)
        finite = per_traj[np.isfinite(per_traj)]
        mean_rmse = float(np.mean(finite)) if finite.size else float("nan")
        results.append({
            "label": gp["label"],
            "init_drag_scale_std": float(gp["init_drag_scale_std"]),
            "drag_scale_sigma_ss": float(gp["drag_scale_sigma_ss"]),
            "drag_scale_tau_s": float(gp["drag_scale_tau_s"]),
            "validation_mean_observed_step_rmse_m": mean_rmse,
            "validation_n_finite": int(finite.size),
        })
        print(
            f"[validation] {gp['label']}: mean={mean_rmse:.4f} m  "
            f"n_finite={int(finite.size)}",
            flush=True,
        )
    t_grid = time.perf_counter() - t_grid

    chosen = min(results, key=lambda r: r["validation_mean_observed_step_rmse_m"])

    payload: dict[str, Any] = {
        "scenario": "long_arc_hifi_validation",
        "schema_version": "long_arc_hifi_validation_v1",
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "validation_seed": int(seed),
        "validation_n_trajectories": int(n_traj),
        "diurnal_alpha": float(diurnal_alpha),
        "longitudinal_alpha": float(longitudinal_alpha),
        "longitudinal_wavenumber": float(longitudinal_wavenumber),
        "steps": int(steps),
        "dt_s": float(dt),
        "selection_rule": vt["selection_rule"],
        "grid_results": results,
        "selected_grid_point": {
            "label": chosen["label"],
            "init_drag_scale_std": chosen["init_drag_scale_std"],
            "drag_scale_sigma_ss": chosen["drag_scale_sigma_ss"],
            "drag_scale_tau_s": chosen["drag_scale_tau_s"],
            "validation_mean_observed_step_rmse_m": chosen["validation_mean_observed_step_rmse_m"],
        },
        "elapsed_seconds": {"truth_generation": float(t_gen), "grid_evaluation": float(t_grid)},
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(json.dumps({
        "selected": payload["selected_grid_point"],
        "json": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
