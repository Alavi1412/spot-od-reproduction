#!/usr/bin/env python
"""Loop-46 KalmanNet SPOT-OD diagnostics and diagnostic-control comparator.

Reviewer concern (M1): the faithful KalmanNet-to-SPOT-OD transposition's
observed-step RMSE magnitudes (six- to nine-figure metres at every predeclared
optimizer-step snapshot) invite an implementation/scaling-artefact explanation
rather than a fundamental learned-OD limit. This script answers that concern
by

(a) Recording per-channel state error breakdowns, per-step gradient norms,
    loss-curve statistics, and visibility-bucket-conditional RMSE on a fresh
    faithful-transposition training run on the same disjoint-seed splits as
    the existing learning-curve artefact; and

(b) Training a clearly labelled normalized/tuned KalmanNet **diagnostic
    control (KNet-DC)** with two minimal, scientifically defensible
    modifications, documented honestly as deviations from the faithful
    transposition:

      DC-1. The training MSE loss includes velocity channels alongside
            position channels (scaled by STATE_SCALE). The faithful upstream
            release minimises position-only loss in the linear-canonical
            benchmark; we extend this to the SPOT-OD state because velocity
            error feeds back through f(x) into subsequent position error.

      DC-2. The observation vector is augmented with the per-step
            per-station visibility mask (eight extra channels carrying 0/1
            visibility flags). h(x) reproduces the visibility vector exactly,
            so the innovation on the eight visibility channels is identically
            zero by construction; the only effect is that the L2-normalized
            obs_diff and obs_innov_diff vectors carry visibility information
            the upstream gain-RNN can use to distinguish "zero because not
            visible" from "zero because perfect fit". This changes the
            observation dimension from n=32 to n=40 and rescales the
            upstream FC/GRU layers proportionally.

The diagnostic control is **explicitly not** the faithful transposition. The
faithful transposition remains the protocol's external published learned-OD
audit case (Table~\\ref{tab:kalmannet_spot_od_transposition} and the 1000-step
learning curve), and the KNet-DC is reported alongside as a diagnostic
sanity check: if KNet-DC reaches sub-kilometre RMSE then the regime is
learnable and the faithful negative is shown to be implementation/scaling
specific rather than fundamental; if KNet-DC also attrites then the bounded
negative on this measurement setting is strengthened.

Outputs (non-paper-facing JSON/CSV under ``results/kalmannet_spot_od/``):
- diagnostic.json  : full payload with both faithful and DC diagnostics
- diagnostic.csv   : per-trajectory RMSE rows
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
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse the faithful transposition driver for data generation, system-functions,
# and per-trajectory RMSE/bootstrap helpers (single source of truth so the
# faithful baseline reproduces exactly).
import run_kalmannet_spot_od_transposition as base

import torch  # noqa: E402  (after base import so torch.load patch is applied)
import torch.nn as nn  # noqa: E402

from gnn_state_estimation.coordinates import StationGeometry  # noqa: E402
from gnn_state_estimation.dataset import MEAS_SCALE, STATE_SCALE  # noqa: E402
from gnn_state_estimation.evaluation import parse_baseline_config  # noqa: E402
from gnn_state_estimation.filters import (  # noqa: E402
    EKFConfig,
    UKFConfig,
    AdaptiveUKFConfig,
    run_ekf,
    run_ukf,
    run_adaptive_ukf,
)
from gnn_state_estimation.simulation import parse_dataset_config  # noqa: E402
from gnn_state_estimation.utils.io import load_yaml  # noqa: E402


# --- Visibility-augmented system functions (diagnostic control) ----------


class VisibilityAugmentedSPOTODSystemFunctions(base.SPOTODSystemFunctions):
    """Diagnostic-control system functions: append visibility flags to h(x).

    The first 4*n_stations channels reproduce the faithful SPOT-OD line-of-sight
    observation with the zero-padding-for-invisible convention; the final
    n_stations channels carry the per-station visibility flag (0/1) so the
    L2-normalized obs_diff and obs_innov_diff vectors flowing into the gain
    RNN distinguish "zero because invisible" from "zero because perfect fit".
    h(x) reproduces the visibility flag exactly, so the innovation on those
    eight channels is identically zero (no spurious learning signal); the only
    effect is on the normalized-direction inputs to the upstream gain network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_dim = 4 * self.n_stations + self.n_stations  # 32 + 8 = 40

    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        was_2d = x.dim() == 2
        if was_2d:
            x = x.unsqueeze(0)
        x_np = x.detach().cpu().numpy()  # (B, m, 1) in SCALED coordinates
        b = x_np.shape[0]
        out = np.zeros((b, self.obs_dim, 1), dtype=np.float64)
        step_clamped = min(max(self.step_idx, 0), len(self.times_s) - 1)
        t_s = float(self.times_s[step_clamped])
        mask = self.visibility_mask
        n_st = self.n_stations
        for i in range(b):
            state = x_np[i, :, 0] * STATE_SCALE
            if not np.all(np.isfinite(state)):
                continue
            for s_idx, station in enumerate(self.stations):
                if mask is not None and mask[i, s_idx] < 0.5:
                    # masked invisible -> zero observation, visibility flag 0
                    continue
                from gnn_state_estimation.coordinates import line_of_sight_measurement  # local import to avoid cycles
                z, _ = line_of_sight_measurement(state, station, t_s)
                z = z.copy()
                z[1] = base._wrap_az(float(z[1]))
                out[i, s_idx * 4 : s_idx * 4 + 4, 0] = z / MEAS_SCALE
                # Append visibility flag in the trailing per-station block:
                out[i, 4 * n_st + s_idx, 0] = 1.0
        t_out = torch.from_numpy(out).to(device=device, dtype=x.dtype)
        if was_2d:
            t_out = t_out.squeeze(0)
        return t_out


def _prepare_input_target_augmented(
    bundles: list[dict[str, np.ndarray]], n_stations: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as base._prepare_input_target but appends per-station visibility flags."""
    n = len(bundles)
    steps = bundles[0]["states"].shape[0]
    obs_dim = 4 * n_stations + n_stations
    y = np.zeros((n, obs_dim, steps), dtype=np.float32)
    x = np.zeros((n, 6, steps), dtype=np.float32)
    v = np.zeros((n, n_stations, steps), dtype=np.float32)
    x0 = np.zeros((n, 6, 1), dtype=np.float32)
    meas_scale_block = np.tile(MEAS_SCALE.astype(np.float64), n_stations)
    for i, b in enumerate(bundles):
        meas = b["measurements"]
        vis = b["visibility"]
        masked = meas * vis[..., None]
        flat = masked.reshape(steps, 4 * n_stations)
        scaled_flat = flat / meas_scale_block
        y[i, : 4 * n_stations] = np.transpose(scaled_flat, (1, 0)).astype(np.float32)
        # Visibility-flag channels (already in [0,1]):
        y[i, 4 * n_stations :] = np.transpose(vis, (1, 0)).astype(np.float32)
        x[i] = np.transpose(b["states"] / STATE_SCALE, (1, 0)).astype(np.float32)
        v[i] = np.transpose(vis, (1, 0)).astype(np.float32)
        x0[i, :, 0] = (b["x0_est"] / STATE_SCALE).astype(np.float32)
    return (
        torch.from_numpy(y),
        torch.from_numpy(x),
        torch.from_numpy(v),
        torch.from_numpy(x0),
    )


# --- Diagnostics: per-channel error, gradient norms, visibility buckets ---


def _per_traj_per_axis_pos_rmse(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> np.ndarray:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    y_true = states[:, eval_start:, :3]
    y_pred = preds[:, eval_start:, :3]
    out = np.full((states.shape[0], 3), np.nan, dtype=np.float64)
    for i in range(states.shape[0]):
        mask = observed[i]
        if not np.any(mask):
            continue
        err = y_true[i, mask] - y_pred[i, mask]
        out[i] = np.sqrt(np.mean(err * err, axis=0))
    return out


def _per_traj_per_axis_vel_rmse(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> np.ndarray:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    y_true = states[:, eval_start:, 3:]
    y_pred = preds[:, eval_start:, 3:]
    out = np.full((states.shape[0], 3), np.nan, dtype=np.float64)
    for i in range(states.shape[0]):
        mask = observed[i]
        if not np.any(mask):
            continue
        err = y_true[i, mask] - y_pred[i, mask]
        out[i] = np.sqrt(np.mean(err * err, axis=0))
    return out


def _visibility_bucket_rmse(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> dict[str, float]:
    """Pooled (over trajectory, step) position RMSE conditioned on the number of
    visible stations at the evaluated step.
    """
    eval_vis = visibility[:, eval_start:]
    n_vis = np.sum(eval_vis, axis=-1)  # (N, T_eval)
    err = states[:, eval_start:, :3] - preds[:, eval_start:, :3]
    se = np.sum(err * err, axis=-1)  # (N, T_eval)
    out: dict[str, float] = {}
    for label, predicate in (("zero", n_vis < 0.5), ("one", (n_vis >= 0.5) & (n_vis < 1.5)), ("ge_two", n_vis >= 1.5)):
        if not np.any(predicate):
            out[f"{label}_pooled_pos_rmse_m"] = float("nan")
            out[f"{label}_count"] = 0
            continue
        out[f"{label}_pooled_pos_rmse_m"] = float(np.sqrt(np.mean(se[predicate])))
        out[f"{label}_count"] = int(np.sum(predicate))
    return out


# --- Training loop with gradient-norm capture ----------------------------


def _train_with_diagnostics(
    knet: base.KNetSPOTODWrapper,
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
    m: int,
    n_steps: int,
    n_batch: int,
    lr: float,
    wd: float,
    device: torch.device,
    eval_start: int,
    include_velocity_in_loss: bool,
    label: str,
) -> dict[str, Any]:
    knet.to(device)
    optimizer = torch.optim.Adam(knet.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss(reduction="mean")
    best_cv = float("inf")
    best_state = None
    history: list[dict[str, Any]] = []
    n_train = train_input.shape[0]
    n_cv = cv_input.shape[0]
    T = train_input.shape[-1]

    rng_py = np.random.default_rng(0)
    for step_i in range(1, n_steps + 1):
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
        if include_velocity_in_loss:
            loss_components = {
                "pos": loss_fn(x_out[:, :3, eval_start:], x_batch[:, :3, eval_start:]),
                "vel": loss_fn(x_out[:, 3:, eval_start:], x_batch[:, 3:, eval_start:]),
            }
            loss = loss_components["pos"] + loss_components["vel"]
        else:
            loss_components = {
                "pos": loss_fn(x_out[:, :3, eval_start:], x_batch[:, :3, eval_start:]),
            }
            loss = loss_components["pos"]
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            train_loss = float("inf")
            grad_norm = float("nan")
        else:
            loss.backward(retain_graph=False)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(knet.parameters(), max_norm=1.0).item()
            )
            optimizer.step()
            train_loss = float(loss.item())

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
            cv_loss_pos = float(
                loss_fn(x_cv[:, :3, eval_start:], cv_target[:, :3, eval_start:].to(device)).item()
            )
            if include_velocity_in_loss:
                cv_loss_vel = float(
                    loss_fn(x_cv[:, 3:, eval_start:], cv_target[:, 3:, eval_start:].to(device)).item()
                )
                cv_loss = cv_loss_pos + cv_loss_vel
            else:
                cv_loss_vel = float("nan")
                cv_loss = cv_loss_pos
        history.append(
            {
                "step": step_i,
                "train_mse": train_loss,
                "train_mse_pos": float(loss_components["pos"].item()) if torch.isfinite(loss_components["pos"]) else float("inf"),
                "train_mse_vel": float(loss_components["vel"].item()) if include_velocity_in_loss and torch.isfinite(loss_components["vel"]) else float("nan"),
                "cv_mse": cv_loss,
                "cv_mse_pos": cv_loss_pos,
                "cv_mse_vel": cv_loss_vel,
                "grad_norm": grad_norm,
            }
        )
        if cv_loss < best_cv:
            best_cv = cv_loss
            best_state = {k: v.clone().detach().cpu() for k, v in knet.state_dict().items()}
        if step_i % 10 == 0 or step_i == 1:
            print(
                f"[{label}] step {step_i}: train_pos={loss_components['pos']:.6f} "
                f"cv_pos={cv_loss_pos:.6f} grad_norm={grad_norm:.4f} best_cv={best_cv:.6f}",
                flush=True,
            )

    if best_state is not None:
        knet.load_state_dict(best_state)
    return {"history": history, "best_cv": best_cv}


def _diagnose_predictions(
    knet: base.KNetSPOTODWrapper,
    sys_funcs: base.SPOTODSystemFunctions,
    test_input: torch.Tensor,
    test_vis: torch.Tensor,
    test_x0: torch.Tensor,
    device: torch.device,
    T: int,
    m: int,
    states_all: np.ndarray,
    vis_all: np.ndarray,
    eval_start: int,
) -> dict[str, Any]:
    preds = base._predict(knet, sys_funcs, test_input, test_vis, test_x0, device, T, m)
    per_traj_obs_rmse = base._per_traj_observed_pos_rmse(states_all, preds, vis_all, eval_start)
    per_axis_pos = _per_traj_per_axis_pos_rmse(states_all, preds, vis_all, eval_start)
    per_axis_vel = _per_traj_per_axis_vel_rmse(states_all, preds, vis_all, eval_start)
    vis_buckets = _visibility_bucket_rmse(states_all, preds, vis_all, eval_start)
    return {
        "preds": preds,
        "per_traj_observed_pos_rmse_m": per_traj_obs_rmse,
        "observed_step_rmse_mean_m": float(np.nanmean(per_traj_obs_rmse)),
        "observed_step_rmse_median_m": float(np.nanmedian(per_traj_obs_rmse)),
        "per_axis_pos_rmse_m": {
            "rx": float(np.nanmean(per_axis_pos[:, 0])),
            "ry": float(np.nanmean(per_axis_pos[:, 1])),
            "rz": float(np.nanmean(per_axis_pos[:, 2])),
        },
        "per_axis_vel_rmse_mps": {
            "vx": float(np.nanmean(per_axis_vel[:, 0])),
            "vy": float(np.nanmean(per_axis_vel[:, 1])),
            "vz": float(np.nanmean(per_axis_vel[:, 2])),
        },
        "visibility_bucket_pooled_pos_rmse_m": vis_buckets,
    }


# --- Driver ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--output-dir", default="results/kalmannet_spot_od")
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-cv", type=int, default=16)
    p.add_argument("--n-test", type=int, default=32)
    p.add_argument("--faithful-n-steps", type=int, default=160)
    p.add_argument("--dc-n-steps", type=int, default=1000)
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


def _build_knet(
    sys_funcs: base.SPOTODSystemFunctions,
    T: int,
    device: torch.device,
    *,
    in_mult: int,
    out_mult: int,
    n_batch: int,
    init_pos_sigma: float,
) -> tuple[base.KNetSPOTODWrapper, base.SPOTODSystemModel]:
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
    knet_args.n_batch = n_batch
    knet_args.in_mult_KNet = in_mult
    knet_args.out_mult_KNet = out_mult
    knet.NNBuild(sys_model, knet_args)
    knet.attach_sys_funcs(sys_funcs)
    return knet, sys_model


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[KNet-DIAG] device={device}", flush=True)

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

    print("[KNet-DIAG] generating data", flush=True)
    t0 = time.perf_counter()
    train_rng = np.random.default_rng(args.train_seed)
    test_rng = np.random.default_rng(args.test_seed)
    train_bundles = _gen(train_rng, args.n_train)
    cv_bundles = _gen(train_rng, args.n_cv)
    test_bundles = _gen(test_rng, args.n_test)
    t_gen = time.perf_counter() - t0

    # Faithful (n=32) inputs
    train_input_f, train_target, train_vis, train_x0 = base._prepare_input_target(
        train_bundles, n_stations
    )
    cv_input_f, cv_target, cv_vis, cv_x0 = base._prepare_input_target(cv_bundles, n_stations)
    test_input_f, test_target, test_vis, test_x0 = base._prepare_input_target(
        test_bundles, n_stations
    )
    # Diagnostic-control (n=40) inputs (same train/target/visibility tensors)
    train_input_dc, _, _, _ = _prepare_input_target_augmented(train_bundles, n_stations)
    cv_input_dc, _, _, _ = _prepare_input_target_augmented(cv_bundles, n_stations)
    test_input_dc, _, _, _ = _prepare_input_target_augmented(test_bundles, n_stations)

    T = dyn.steps

    # --- Classical references on the test population (once) ---
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
    best_name = min(classical_means, key=lambda k: classical_means[k])
    print(f"[KNet-DIAG] classical observed-step RMSE means: {classical_means}", flush=True)
    print(f"[KNet-DIAG] best classical reference: {best_name}", flush=True)

    # --- Faithful baseline ---
    print(
        f"[KNet-DIAG] training faithful baseline ({args.faithful_n_steps} steps)",
        flush=True,
    )
    sys_funcs_f = base.SPOTODSystemFunctions(
        stations=stations,
        times_s=np.arange(T, dtype=np.float64) * dyn.dt_s,
        dt_s=dyn.dt_s,
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
    )
    knet_f, _ = _build_knet(
        sys_funcs_f, T, device,
        in_mult=args.in_mult,
        out_mult=args.out_mult,
        n_batch=max(args.n_batch, args.n_cv),
        init_pos_sigma=init_pos_sigma,
    )
    t0 = time.perf_counter()
    train_f = _train_with_diagnostics(
        knet_f, sys_funcs_f,
        train_input_f, train_target, train_vis, train_x0,
        cv_input_f, cv_target, cv_vis, cv_x0,
        m=6,
        n_steps=args.faithful_n_steps,
        n_batch=args.n_batch,
        lr=args.lr, wd=args.wd, device=device,
        eval_start=eval_start,
        include_velocity_in_loss=False,
        label="KNet-FAITH",
    )
    t_faith = time.perf_counter() - t0
    diag_f = _diagnose_predictions(
        knet_f, sys_funcs_f,
        test_input_f, test_vis, test_x0,
        device, T, 6, states_all, vis_all, eval_start,
    )
    n_params_f = int(sum(p.numel() for p in knet_f.parameters() if p.requires_grad))
    print(
        f"[KNet-DIAG] faithful: test observed RMSE mean={diag_f['observed_step_rmse_mean_m']:.2f}m "
        f"median={diag_f['observed_step_rmse_median_m']:.2f}m",
        flush=True,
    )

    # --- Diagnostic control (n=40 augmented obs + position+velocity loss) ---
    print(
        f"[KNet-DIAG] training diagnostic control ({args.dc_n_steps} steps)", flush=True
    )
    sys_funcs_dc = VisibilityAugmentedSPOTODSystemFunctions(
        stations=stations,
        times_s=np.arange(T, dtype=np.float64) * dyn.dt_s,
        dt_s=dyn.dt_s,
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
    )
    knet_dc, _ = _build_knet(
        sys_funcs_dc, T, device,
        in_mult=args.in_mult,
        out_mult=args.out_mult,
        n_batch=max(args.n_batch, args.n_cv),
        init_pos_sigma=init_pos_sigma,
    )
    t0 = time.perf_counter()
    train_dc = _train_with_diagnostics(
        knet_dc, sys_funcs_dc,
        train_input_dc, train_target, train_vis, train_x0,
        cv_input_dc, cv_target, cv_vis, cv_x0,
        m=6,
        n_steps=args.dc_n_steps,
        n_batch=args.n_batch,
        lr=args.lr, wd=args.wd, device=device,
        eval_start=eval_start,
        include_velocity_in_loss=True,
        label="KNet-DC",
    )
    t_dc = time.perf_counter() - t0
    diag_dc = _diagnose_predictions(
        knet_dc, sys_funcs_dc,
        test_input_dc, test_vis, test_x0,
        device, T, 6, states_all, vis_all, eval_start,
    )
    n_params_dc = int(sum(p.numel() for p in knet_dc.parameters() if p.requires_grad))
    print(
        f"[KNet-DIAG] DC: test observed RMSE mean={diag_dc['observed_step_rmse_mean_m']:.2f}m "
        f"median={diag_dc['observed_step_rmse_median_m']:.2f}m",
        flush=True,
    )

    # --- Paired diffs vs best classical ---
    def _paired_block(knet_per_traj: np.ndarray, seed: int) -> dict[str, Any]:
        diffs = knet_per_traj - classical_rmse[best_name]
        mean_d, lo, hi = base._paired_bootstrap_ci(
            diffs, n_boot=int(args.bootstrap_samples), seed=seed
        )
        finite = diffs[np.isfinite(diffs)]
        return {
            "best_classical": best_name,
            "best_classical_mean_m": classical_means[best_name],
            "mean_minus_best_m": mean_d,
            "ci_lo_m": lo,
            "ci_hi_m": hi,
            "n_paired": int(finite.size),
            "better_count": int(np.sum(finite < 0.0)) if finite.size else 0,
        }

    paired_f = _paired_block(diag_f["per_traj_observed_pos_rmse_m"], args.test_seed + 100)
    paired_dc = _paired_block(diag_dc["per_traj_observed_pos_rmse_m"], args.test_seed + 200)

    # --- Serialize ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "kalmannet_spot_od_diagnostic_v1",
        "scenario": "kalmannet_spot_od_diagnostic",
        "vendor_commit": base.VENDOR_COMMIT,
        "device": str(device),
        "n_params": {
            "faithful": n_params_f,
            "diagnostic_control": n_params_dc,
        },
        "config": {
            "n_train": int(args.n_train),
            "n_cv": int(args.n_cv),
            "n_test": int(args.n_test),
            "faithful_n_steps": int(args.faithful_n_steps),
            "dc_n_steps": int(args.dc_n_steps),
            "n_batch": int(args.n_batch),
            "lr": float(args.lr),
            "wd": float(args.wd),
            "in_mult": int(args.in_mult),
            "out_mult": int(args.out_mult),
            "train_seed": int(args.train_seed),
            "test_seed": int(args.test_seed),
            "T": int(T),
            "m": 6,
            "n_faithful": 32,
            "n_diagnostic_control": 40,
            "eval_start_step": int(eval_start),
        },
        "classical_baselines_mean_observed_step_rmse_m": classical_means,
        "best_classical_baseline": best_name,
        "faithful_diagnostics": {
            "best_cv_mse_pos_only": float(train_f["best_cv"]),
            "observed_step_rmse_mean_m": diag_f["observed_step_rmse_mean_m"],
            "observed_step_rmse_median_m": diag_f["observed_step_rmse_median_m"],
            "per_axis_pos_rmse_m": diag_f["per_axis_pos_rmse_m"],
            "per_axis_vel_rmse_mps": diag_f["per_axis_vel_rmse_mps"],
            "visibility_bucket_pooled_pos_rmse_m": diag_f["visibility_bucket_pooled_pos_rmse_m"],
            "paired_vs_best_classical": paired_f,
            "training_history": train_f["history"],
            "training_time_s": float(t_faith),
        },
        "diagnostic_control_diagnostics": {
            "best_cv_mse_pos_plus_vel": float(train_dc["best_cv"]),
            "observed_step_rmse_mean_m": diag_dc["observed_step_rmse_mean_m"],
            "observed_step_rmse_median_m": diag_dc["observed_step_rmse_median_m"],
            "per_axis_pos_rmse_m": diag_dc["per_axis_pos_rmse_m"],
            "per_axis_vel_rmse_mps": diag_dc["per_axis_vel_rmse_mps"],
            "visibility_bucket_pooled_pos_rmse_m": diag_dc["visibility_bucket_pooled_pos_rmse_m"],
            "paired_vs_best_classical": paired_dc,
            "training_history": train_dc["history"],
            "training_time_s": float(t_dc),
        },
        "diagnostic_control_modifications": [
            "DC-1: training MSE loss includes velocity channels alongside positions (scaled coordinates)",
            "DC-2: observation vector augmented with per-station visibility flags (n=32->40); h(x) reproduces the flags exactly so innovation on visibility channels is identically zero by construction",
        ],
        "elapsed_seconds": {
            "data_generation": float(t_gen),
            "faithful_training": float(t_faith),
            "diagnostic_control_training": float(t_dc),
        },
        "labelling": (
            "The faithful transposition keeps the upstream KalmanNet architecture, "
            "training pipeline, n=32 SPOT-OD observation, position-only loss, and "
            "zero-pad-invisible visibility convention unchanged from the linear-"
            "canonical sanity check up to dimension. The diagnostic control "
            "(KNet-DC) is explicitly NOT a faithful transposition: it is a "
            "scientifically defensible normalized/tuned KalmanNet variant whose "
            "two modifications (position+velocity loss, visibility-mask "
            "augmented observation) probe whether the faithful negative is "
            "implementation-specific or fundamental."
        ),
    }

    out_json = out_dir / "diagnostic.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))

    rows = []
    for i in range(len(test_bundles)):
        rows.append(
            {
                "trajectory_index": i,
                "KalmanNet_faithful_obs_pos_rmse_m": float(
                    diag_f["per_traj_observed_pos_rmse_m"][i]
                )
                if np.isfinite(diag_f["per_traj_observed_pos_rmse_m"][i])
                else None,
                "KalmanNet_diagnostic_control_obs_pos_rmse_m": float(
                    diag_dc["per_traj_observed_pos_rmse_m"][i]
                )
                if np.isfinite(diag_dc["per_traj_observed_pos_rmse_m"][i])
                else None,
                "EKF_obs_pos_rmse_m": float(classical_rmse["EKF"][i])
                if np.isfinite(classical_rmse["EKF"][i])
                else None,
                "UKF_obs_pos_rmse_m": float(classical_rmse["UKF"][i])
                if np.isfinite(classical_rmse["UKF"][i])
                else None,
                "AUKF_obs_pos_rmse_m": float(classical_rmse["AUKF"][i])
                if np.isfinite(classical_rmse["AUKF"][i])
                else None,
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "diagnostic.csv", index=False)
    print(
        json.dumps(
            {
                "best_classical": best_name,
                "faithful": {
                    "mean": diag_f["observed_step_rmse_mean_m"],
                    "median": diag_f["observed_step_rmse_median_m"],
                },
                "diagnostic_control": {
                    "mean": diag_dc["observed_step_rmse_mean_m"],
                    "median": diag_dc["observed_step_rmse_median_m"],
                },
            },
            indent=2,
            default=float,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
