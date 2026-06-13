#!/usr/bin/env python
"""Faithful KalmanNet to SPOT-OD learning-curve experiment.

A reviewer concern on the earlier faithful transposition was that the
40-optimizer-step budget could not distinguish a genuine architectural
limitation from a training-budget artefact. This driver runs the same faithful
transposition under a materially larger optimizer-step budget (default 320
steps, an 8x extension of the earlier 40-step budget) and snapshots
held-out test observed-step position RMSE at a predeclared schedule of
optimizer-step milestones (40, 80, 160, 320). The same disjoint-seed train/CV
and test split is used; no cherry-picking of an intermediate snapshot is
allowed because all milestones are pre-listed in the schedule.

Outputs:
- ``results/kalmannet_spot_od/learning_curve.json`` (full payload, schema v1)
- ``results/kalmannet_spot_od/learning_curve.csv`` (per-trajectory test RMSE
  at each milestone)
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
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse pieces from the single-budget transposition driver.
import run_kalmannet_spot_od_transposition as base

import torch  # noqa: E402  (after base import to share torch.load patch)
import torch.nn as nn  # noqa: E402

from gnn_state_estimation.dynamics import rk4_step  # noqa: E402
from gnn_state_estimation.evaluation import parse_baseline_config  # noqa: E402
from gnn_state_estimation.filters import (  # noqa: E402
    EKFConfig,
    UKFConfig,
    AdaptiveUKFConfig,
    ProcessNoiseAdaptiveUKFConfig,
    run_ekf,
    run_ukf,
    run_adaptive_ukf,
    run_process_noise_adaptive_ukf,
)
from gnn_state_estimation.simulation import parse_dataset_config  # noqa: E402
from gnn_state_estimation.utils.io import load_yaml  # noqa: E402


# --- Staged training loop -------------------------------------------------


def _train_with_snapshots(
    knet: base.KNetSPOTODWrapper,
    sys_model: base.SPOTODSystemModel,
    sys_funcs: base.SPOTODSystemFunctions,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    train_vis: torch.Tensor,
    train_x0: torch.Tensor,
    cv_input: torch.Tensor,
    cv_target: torch.Tensor,
    cv_vis: torch.Tensor,
    cv_x0: torch.Tensor,
    *,
    n_steps: int,
    n_batch: int,
    lr: float,
    wd: float,
    device: torch.device,
    eval_start: int,
    snapshot_steps: list[int],
) -> dict[str, Any]:
    """Train for ``n_steps`` optimizer steps; deep-copy the model parameters
    after the optimizer step at each milestone in ``snapshot_steps`` (so
    snapshot ``k`` is the model state immediately after ``k`` optimizer
    updates). CV-best checkpointing is preserved on top of the milestones so
    the final-model state is the best-CV model over the full schedule.
    """
    knet.to(device)
    optimizer = torch.optim.Adam(knet.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss(reduction="mean")
    best_cv = float("inf")
    best_state = None
    history: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    snapshot_set = set(int(s) for s in snapshot_steps)
    n_train = train_input.shape[0]
    n_cv = cv_input.shape[0]
    T = train_input.shape[-1]
    m = sys_model.m

    rng_py = np.random.default_rng(0)
    for step_i in range(1, n_steps + 1):
        # --- Training batch ---
        idx = rng_py.choice(n_train, size=min(n_batch, n_train), replace=False)
        y_batch = train_input[idx].to(device)
        x_batch = train_target[idx].to(device)
        v_batch = train_vis[idx].to(device)
        x0_batch = train_x0[idx].to(device)
        b = y_batch.shape[0]

        knet.train()
        knet.batch_size = b
        knet.init_hidden_KNet()
        sys_funcs.visibility_mask = None
        sys_funcs.step_idx = 0
        knet.InitSequence(x0_batch, T)
        sys_funcs.reset()
        x_out = torch.zeros((b, m, T), device=device, dtype=y_batch.dtype)
        for t in range(T):
            sys_funcs.step_idx = t
            sys_funcs.set_visibility(v_batch[:, :, t].detach().cpu().numpy())
            y_t = y_batch[:, :, t].unsqueeze(2)
            x_t = knet(y_t)
            x_out[:, :, t] = torch.squeeze(x_t, 2)
        loss = loss_fn(x_out[:, :3, eval_start:], x_batch[:, :3, eval_start:])
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            train_loss = float("inf")
        else:
            loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(knet.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss = float(loss.item())

        # --- CV ---
        knet.eval()
        knet.batch_size = n_cv
        knet.init_hidden_KNet()
        sys_funcs.visibility_mask = None
        sys_funcs.step_idx = 0
        knet.InitSequence(cv_x0.to(device), T)
        sys_funcs.reset()
        with torch.no_grad():
            x_cv = torch.zeros((n_cv, m, T), device=device, dtype=cv_input.dtype)
            for t in range(T):
                sys_funcs.step_idx = t
                sys_funcs.set_visibility(cv_vis[:, :, t].detach().cpu().numpy())
                y_t = cv_input[:, :, t].to(device).unsqueeze(2)
                x_t = knet(y_t)
                x_cv[:, :, t] = torch.squeeze(x_t, 2)
            cv_loss = float(
                loss_fn(x_cv[:, :3, eval_start:], cv_target[:, :3, eval_start:].to(device)).item()
            )
        history.append({"step": step_i, "train_mse": train_loss, "cv_mse": cv_loss})
        if cv_loss < best_cv:
            best_cv = cv_loss
            best_state = {k: v.clone().detach().cpu() for k, v in knet.state_dict().items()}

        if step_i in snapshot_set:
            snapshots[step_i] = {
                k: v.clone().detach().cpu() for k, v in knet.state_dict().items()
            }
            print(
                f"[KNet-LC] snapshot at step {step_i}: train_mse={train_loss:.6f}, "
                f"cv_mse={cv_loss:.6f}, best_cv={best_cv:.6f}",
                flush=True,
            )
        elif step_i % 10 == 0 or step_i == 1:
            print(
                f"[KNet-LC] step {step_i}: train_mse={train_loss:.6f}, "
                f"cv_mse={cv_loss:.6f}, best={best_cv:.6f}",
                flush=True,
            )

    return {
        "history": history,
        "best_cv": best_cv,
        "best_state": best_state,
        "snapshots": snapshots,
    }


def _eval_snapshot_rmse(
    knet: base.KNetSPOTODWrapper,
    sys_funcs: base.SPOTODSystemFunctions,
    state_dict: dict[str, torch.Tensor],
    test_input: torch.Tensor,
    test_vis: torch.Tensor,
    test_x0: torch.Tensor,
    device: torch.device,
    T: int,
    m: int,
    states_all: np.ndarray,
    vis_all: np.ndarray,
    eval_start: int,
) -> np.ndarray:
    knet.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    knet.batch_size = test_input.shape[0]
    preds = base._predict(knet, sys_funcs, test_input, test_vis, test_x0, device, T, m)
    return base._per_traj_observed_pos_rmse(states_all, preds, vis_all, eval_start)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--predeclared-rule",
        default="release/predeclarations/pukf_q_adaptive_rule_loop41.json",
    )
    p.add_argument("--output-dir", default="results/kalmannet_spot_od")
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-cv", type=int, default=16)
    p.add_argument("--n-test", type=int, default=32)
    p.add_argument(
        "--n-steps",
        type=int,
        default=320,
        help="total optimizer steps (default 320 = 8x the earlier 40-step budget)",
    )
    p.add_argument(
        "--snapshot-steps",
        type=str,
        default="40,80,160,320",
        help="comma-separated optimizer-step milestones at which to evaluate test RMSE",
    )
    p.add_argument("--n-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--in-mult", type=int, default=5)
    p.add_argument("--out-mult", type=int, default=4)
    p.add_argument("--train-seed", type=int, default=20260601)
    p.add_argument("--test-seed", type=int, default=20260701)
    p.add_argument("--bootstrap-samples", type=int, default=3000)
    p.add_argument("--device", default="auto")
    return p


def main() -> int:
    args = build_parser().parse_args()
    snapshot_steps = sorted(set(int(x.strip()) for x in args.snapshot_steps.split(",") if x.strip()))
    if not snapshot_steps:
        raise SystemExit("--snapshot-steps must contain at least one milestone")
    if max(snapshot_steps) > args.n_steps:
        raise SystemExit("Final snapshot exceeds --n-steps")

    cfg = load_yaml(Path(args.config))
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    rule = json.loads(Path(args.predeclared_rule).read_text())
    th = rule["thresholds"]

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    n_stations = len(stations)
    eval_start = 11

    init_pos_sigma = baseline_cfg.ukf.init_pos_std_m
    init_vel_sigma = baseline_cfg.ukf.init_vel_std_mps

    def _gen(rng: np.random.Generator, n: int) -> list[dict[str, np.ndarray]]:
        return [
            base._generate_one(
                rng,
                dataset_cfg.orbit_sampling,
                stations,
                dyn,
                meas_std,
                dataset_cfg.measurement_noise.outlier_prob,
                dataset_cfg.measurement_noise.outlier_scale,
                dataset_cfg.measurement_noise.random_dropout_prob,
                init_pos_sigma,
                init_vel_sigma,
            )
            for _ in range(n)
        ]

    print(f"[KNet-LC] device={device}, generating data", flush=True)
    t0 = time.perf_counter()
    train_rng = np.random.default_rng(args.train_seed)
    test_rng = np.random.default_rng(args.test_seed)
    train_bundles = _gen(train_rng, args.n_train)
    cv_bundles = _gen(train_rng, args.n_cv)
    test_bundles = _gen(test_rng, args.n_test)
    t_gen = time.perf_counter() - t0

    train_input, train_target, train_vis, train_x0 = base._prepare_input_target(
        train_bundles, n_stations
    )
    cv_input, cv_target, cv_vis, cv_x0 = base._prepare_input_target(cv_bundles, n_stations)
    test_input, test_target, test_vis, test_x0 = base._prepare_input_target(test_bundles, n_stations)

    T = dyn.steps
    sys_funcs = base.SPOTODSystemFunctions(
        stations=stations,
        times_s=np.arange(T, dtype=np.float64) * dyn.dt_s,
        dt_s=dyn.dt_s,
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
    )
    m1x_0 = torch.zeros(6, 1)
    m2x_0 = torch.eye(6) * (init_pos_sigma ** 2)
    Q = torch.eye(6)
    R = torch.eye(sys_funcs.obs_dim)
    sys_model = base.SPOTODSystemModel(
        sys_funcs, T=T, T_test=T, m1x_0=m1x_0, m2x_0=m2x_0, Q=Q, R=R
    )

    knet = base.KNetSPOTODWrapper()

    class _Args:
        pass

    knet_args = _Args()
    knet_args.use_cuda = device.type == "cuda"
    knet_args.n_batch = max(args.n_batch, args.n_cv)
    knet_args.in_mult_KNet = args.in_mult
    knet_args.out_mult_KNet = args.out_mult
    knet.NNBuild(sys_model, knet_args)
    knet.attach_sys_funcs(sys_funcs)
    n_params = sum(p.numel() for p in knet.parameters() if p.requires_grad)
    print(
        f"[KNet-LC] params={n_params}, T={T}, m={sys_model.m}, n={sys_model.n}, "
        f"n_steps={args.n_steps}, snapshots={snapshot_steps}",
        flush=True,
    )

    t0 = time.perf_counter()
    train_out = _train_with_snapshots(
        knet=knet,
        sys_model=sys_model,
        sys_funcs=sys_funcs,
        train_input=train_input,
        train_target=train_target,
        train_vis=train_vis,
        train_x0=train_x0,
        cv_input=cv_input,
        cv_target=cv_target,
        cv_vis=cv_vis,
        cv_x0=cv_x0,
        n_steps=args.n_steps,
        n_batch=args.n_batch,
        lr=args.lr,
        wd=args.wd,
        device=device,
        eval_start=eval_start,
        snapshot_steps=snapshot_steps,
    )
    t_train = time.perf_counter() - t0

    states_all = np.stack([b["states"] for b in test_bundles], axis=0)
    meas_all = np.stack([b["measurements"] for b in test_bundles], axis=0)
    vis_all = np.stack([b["visibility"] for b in test_bundles], axis=0)
    times_all = np.tile(np.arange(T, dtype=np.float64) * dyn.dt_s, (len(test_bundles), 1))
    x0_est_all = np.stack([b["x0_est"] for b in test_bundles], axis=0)

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

    ekf_pred = np.zeros_like(states_all)
    ukf_pred = np.zeros_like(states_all)
    aukf_pred = np.zeros_like(states_all)
    for i in range(len(test_bundles)):
        ekf_pred[i], _ = run_ekf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times_all[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=ekf_cfg,
            **ekf_kwargs,
        )
        ukf_pred[i], _ = run_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times_all[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=ukf_cfg,
            **ekf_kwargs,
        )
        aukf_pred[i], _ = run_adaptive_ukf(
            measurements=meas_all[i],
            visibility=vis_all[i],
            times_s=times_all[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0_est_all[i],
            cfg=aukf_cfg,
            **ekf_kwargs,
        )
    classical_rmse = {
        "EKF": base._per_traj_observed_pos_rmse(states_all, ekf_pred, vis_all, eval_start),
        "UKF": base._per_traj_observed_pos_rmse(states_all, ukf_pred, vis_all, eval_start),
        "AUKF": base._per_traj_observed_pos_rmse(states_all, aukf_pred, vis_all, eval_start),
    }
    classical_means = {k: float(np.nanmean(v)) for k, v in classical_rmse.items()}
    best_classical_name = min(classical_means, key=lambda k: classical_means[k])

    snapshots_payload: list[dict[str, Any]] = []
    snapshots_state = train_out["snapshots"]
    per_traj_rows: list[dict[str, Any]] = []
    for step in snapshot_steps:
        state_dict = snapshots_state.get(int(step))
        if state_dict is None:
            continue
        per_traj = _eval_snapshot_rmse(
            knet,
            sys_funcs,
            state_dict,
            test_input,
            test_vis,
            test_x0,
            device,
            T,
            sys_model.m,
            states_all,
            vis_all,
            eval_start,
        )
        diffs = per_traj - classical_rmse[best_classical_name]
        mean_d, lo, hi = base._paired_bootstrap_ci(
            diffs, n_boot=int(args.bootstrap_samples), seed=args.test_seed + step
        )
        finite = diffs[np.isfinite(diffs)]
        snapshots_payload.append(
            {
                "optimizer_step": int(step),
                "test_observed_step_rmse_mean_m": float(np.nanmean(per_traj)),
                "test_observed_step_rmse_median_m": float(np.nanmedian(per_traj)),
                "best_classical_name": best_classical_name,
                "best_classical_mean_m": classical_means[best_classical_name],
                "knet_minus_best_mean_m": mean_d,
                "knet_minus_best_ci_lo_m": lo,
                "knet_minus_best_ci_hi_m": hi,
                "knet_better_count": int(np.sum(finite < 0.0)) if finite.size else 0,
                "n_paired": int(finite.size),
            }
        )
        for traj_idx in range(states_all.shape[0]):
            per_traj_rows.append(
                {
                    "optimizer_step": int(step),
                    "trajectory_index": traj_idx,
                    "test_observed_step_pos_rmse_m": float(per_traj[traj_idx])
                    if np.isfinite(per_traj[traj_idx])
                    else None,
                }
            )

    # Add a classical-only per-trajectory row for the test split
    for traj_idx in range(states_all.shape[0]):
        per_traj_rows.append(
            {
                "optimizer_step": -1,
                "trajectory_index": traj_idx,
                "EKF_pos_rmse_m": float(classical_rmse["EKF"][traj_idx])
                if np.isfinite(classical_rmse["EKF"][traj_idx])
                else None,
                "UKF_pos_rmse_m": float(classical_rmse["UKF"][traj_idx])
                if np.isfinite(classical_rmse["UKF"][traj_idx])
                else None,
                "AUKF_pos_rmse_m": float(classical_rmse["AUKF"][traj_idx])
                if np.isfinite(classical_rmse["AUKF"][traj_idx])
                else None,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "kalmannet_spot_od_learning_curve_v1",
        "scenario": "kalmannet_spot_od_learning_curve",
        "vendor_commit": base.VENDOR_COMMIT,
        "device": str(device),
        "n_params_kalmannet": int(n_params),
        "config": {
            "n_train": int(args.n_train),
            "n_cv": int(args.n_cv),
            "n_test": int(args.n_test),
            "n_steps": int(args.n_steps),
            "snapshot_steps": [int(s) for s in snapshot_steps],
            "n_batch": int(args.n_batch),
            "lr": float(args.lr),
            "wd": float(args.wd),
            "in_mult": int(args.in_mult),
            "out_mult": int(args.out_mult),
            "train_seed": int(args.train_seed),
            "test_seed": int(args.test_seed),
            "T": int(T),
            "m": int(sys_model.m),
            "n": int(sys_model.n),
            "eval_start_step": int(eval_start),
        },
        "classical_baselines_mean_observed_step_rmse_m": classical_means,
        "best_classical_baseline": best_classical_name,
        "snapshots": snapshots_payload,
        "training_history": train_out["history"],
        "elapsed_seconds": {
            "data_generation": float(t_gen),
            "training_and_snapshots": float(t_train + (time.perf_counter() - t0 - t_train)),
        },
        "predeclared_rule_path": args.predeclared_rule,
        "predeclared_rule_digest_sha256": hashlib.sha256(
            Path(args.predeclared_rule).read_bytes()
        ).hexdigest(),
        "adaptations_from_upstream": [
            "state dim m=2 -> m=6 (ECI position+velocity)",
            "obs dim n=2 -> n=32 (8 stations x 4 channels)",
            "linear f, h -> SPOT-OD nonlinear orbital RK4 propagator and station observation",
            "step-counter held on a wrapper so f/h see the correct absolute time",
            "invisible station blocks are zeroed identically in observation and h(x)",
            "training loss restricted to observed-step window (eval_start onward), same as classical scoring",
            "predeclared optimizer-step milestones at 40/80/160/320 (no cherry-picking)",
        ],
        "notes": (
            "Same train/CV/test seeds and protocol as the single-budget faithful "
            "transposition; the only difference is the longer optimizer-step "
            "budget and the predeclared evaluation milestones."
        ),
    }

    out_json = out_dir / "learning_curve.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))
    pd.DataFrame(per_traj_rows).to_csv(out_dir / "learning_curve.csv", index=False)

    print(json.dumps({"snapshots": snapshots_payload, "classical": classical_means}, indent=2, default=float), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
