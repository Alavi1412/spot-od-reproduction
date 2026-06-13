#!/usr/bin/env python
"""Adapted KalmanNet transposition to the SPOT-OD measurement setting (loop163).

This is the *adapted* counterpart of the faithful Loop-42/57 KalmanNet
transposition (``run_kalmannet_spot_od_transposition.py``). It keeps the
upstream KalmanNet gain network (``KNet.KalmanNet_nn.KalmanNetNN``: the
(Q,Sigma,S)-GRU stack and seven FC blocks) UNCHANGED, but it addresses the four
adaptation gaps that the faithful transposition deliberately left open, and --
most importantly -- it repairs the single defect that the loop53 design-gap
quarantine note identified as the dominant numerical failure: the state
transition ``f`` and measurement ``h`` are now native torch-autograd functions
(``gnn_state_estimation.kalmannet_adapted.torch_dynamics``) instead of numpy
bridges that ``.detach()`` and sever backpropagation-through-time.

Adaptations (each toggleable for ablation):

* **R3 / BPTT (the decisive one).** ``f`` (RK4 two-body+J2+drag) and ``h``
  (range/az/el/range-rate) are differentiable torch ops, so the recurrent
  rollout keeps a graph from the position+velocity loss back through every
  propagation and measurement step to the gain network. ``--detach-dynamics``
  reproduces the broken-BPTT faithful behaviour for a controlled ablation.
* **R1 / orbital-scale normalization.** The gain-network input innovation
  features are reweighted to noise-equivalent (per-channel ``1/sigma``) units
  before the upstream L2 direction-normalization, so range/range-rate (precise)
  are not drowned out by the angle channels. The *actual* innovation used in
  ``KGain @ dy`` stays in ``MEAS_SCALE`` units so the gain magnitude is
  unchanged (avoiding the loop52 inflation-to-1e4 failure).
* **R2 / bounded posterior.** The learned correction is passed through a smooth
  ``tanh`` envelope (generous, ~hundreds of km) in scaled state coordinates, so
  a transient bad gain cannot drive the propagator through overflow.
* **gap2 / curriculum.** Two-stage visibility-stratified curriculum (train on
  the top-visibility trajectories first, then the full pool).
* **gap3 / sparse observation.** Per-station visibility-flag augmentation
  (n=32 -> 40), and angle-innovation wrapping into (-pi, pi] so azimuth wraps
  near 0/2pi do not inject spurious innovations.
* **gap4 / learning rate + budget.** Cosine schedule with warm-up and an
  extended optimizer budget.

The classical EKF/UKF/AUKF/PUKF references are scored on the SAME truth
trajectories under the SAME compact force model that generated them
(``--classical-model matched``; the truth uses compact two-body+J2+drag with no
third-body/SRP, so this is the fair shared-model comparison). The manuscript
harness instead runs the classical filters with third-body+SRP enabled; that
variant is also recorded for cross-reference under ``classical_manuscript``.

Outputs (non-paper-facing) under ``results/kalmannet_adapted/``. Nothing here
is paper-facing and this driver owns only the ``kalmannet_adapted`` namespace.
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
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse the faithful transposition driver for data generation and the exact
# per-trajectory observed-step RMSE / paired-bootstrap helpers, so the adapted
# baseline is scored on an identical protocol.
import run_kalmannet_spot_od_transposition as base

import torch  # noqa: E402  (after base import so its torch.load patch is applied)
import torch.nn as nn  # noqa: E402

from gnn_state_estimation.dataset import MEAS_SCALE, STATE_SCALE  # noqa: E402
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
from gnn_state_estimation.kalmannet_adapted.torch_dynamics import (  # noqa: E402
    SPOTODTorchDynamics,
)
from gnn_state_estimation.simulation import parse_dataset_config  # noqa: E402
from gnn_state_estimation.utils.io import load_yaml  # noqa: E402

from KNet.KalmanNet_nn import KalmanNetNN  # noqa: E402


# --- Stateful differentiable system functions -----------------------------


class AdaptedSysFuncs:
    """Holds the differentiable dynamics + obs builder and the per-step time /
    visibility context the upstream KalmanNet forward pass needs.

    ``f`` and ``h`` operate on scaled state ``[B, 6, 1]`` and return scaled
    state / ``MEAS_SCALE``-unit observations ``[B, n, 1]`` while keeping a torch
    graph. The visibility mask and step index are set by the training/predict
    loop before each forward call (same stateful pattern as the faithful
    transposition, but with native torch f/h).
    """

    def __init__(
        self,
        dyn: SPOTODTorchDynamics,
        *,
        times_s: np.ndarray,
        augment_visibility: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.dyn = dyn
        self.times_s = np.asarray(times_s, dtype=np.float64)
        self.n_stations = dyn.n_stations
        self.augment = bool(augment_visibility)
        self.obs_dim = 4 * self.n_stations + (self.n_stations if self.augment else 0)
        self.device = device
        self.dtype = dtype
        self.step_idx = 0
        self.visibility_mask: torch.Tensor | None = None  # (B, S) float {0,1}
        meas_scale = torch.as_tensor(
            np.asarray(MEAS_SCALE, dtype=np.float64), dtype=dtype, device=device
        )
        self.meas_scale = meas_scale  # (4,)

    def reset(self) -> None:
        self.step_idx = 0

    def set_visibility(self, mask: torch.Tensor) -> None:
        self.visibility_mask = mask

    def f_torch(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 6, 1] scaled -> [B, 6, 1] scaled
        xb = x.squeeze(-1)
        out = self.dyn.f_scaled(xb)
        return out.unsqueeze(-1)

    def h_torch(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 6, 1] scaled -> [B, obs_dim, 1] in MEAS_SCALE units, masked.
        xb = x.squeeze(-1)
        b = xb.shape[0]
        step = min(max(self.step_idx, 0), len(self.times_s) - 1)
        t_s = float(self.times_s[step])
        meas_phys = self.dyn.h_scaled_all_stations(xb, t_s)  # (B, S, 4) physical
        meas_scaled = meas_phys / self.meas_scale  # (B, S, 4)
        mask = self.visibility_mask
        if mask is None:
            mask = torch.ones(b, self.n_stations, dtype=self.dtype, device=xb.device)
        mask = mask.to(meas_scaled.dtype)
        meas_scaled = meas_scaled * mask.unsqueeze(-1)  # zero invisible blocks
        flat = meas_scaled.reshape(b, 4 * self.n_stations)
        if self.augment:
            flat = torch.cat([flat, mask], dim=1)  # append visibility flags
        return flat.unsqueeze(-1)


# --- Adapted KalmanNet wrapper (gain RNN unchanged) -----------------------


class AdaptedKNet(KalmanNetNN):
    """Upstream gain network with: optional BPTT detach ablation, noise-equiv
    innovation-feature reweighting, angle-innovation wrapping, and a bounded
    posterior projection. The KGain network itself (``KGain_step`` and all
    FC/GRU weights) is untouched.
    """

    def configure_adaptation(
        self,
        *,
        sys_funcs: AdaptedSysFuncs,
        obs_feat_weight: torch.Tensor,
        az_channel_index: torch.Tensor,
        az_wrap_period: float,
        posterior_clip: torch.Tensor | None,
        detach_dynamics: bool,
        noise_equiv_features: bool,
        wrap_azimuth: bool,
        gain_scale: float = 1.0,
        straight_through_clip: bool = True,
    ) -> None:
        self.sys_funcs = sys_funcs
        self.register_buffer("obs_feat_weight", obs_feat_weight, persistent=False)
        self.register_buffer("az_channel_index", az_channel_index, persistent=False)
        self.az_wrap_period = float(az_wrap_period)
        self.gain_scale = float(gain_scale)
        if posterior_clip is not None:
            self.register_buffer("posterior_clip", posterior_clip, persistent=False)
        else:
            self.posterior_clip = None
        self.detach_dynamics = bool(detach_dynamics)
        self.noise_equiv_features = bool(noise_equiv_features)
        self.wrap_azimuth = bool(wrap_azimuth)
        self.straight_through_clip = bool(straight_through_clip)

    def reset_steps(self) -> None:
        self.sys_funcs.reset()

    def zero_init_gain(self) -> None:
        """Zero the final layer of the gain network (FC2) so the initial Kalman
        gain is exactly zero. The filter therefore *starts* as pure propagation
        (a stable ~20 km no-skill estimate) and learns small beneficial
        corrections from there, instead of starting from a random large gain in
        a steep, divergence-prone region of the loss. This is the standard
        zero-init trick for residual/gain output heads."""
        last = None
        for mod in self.FC2.modules():
            if isinstance(mod, nn.Linear):
                last = mod
        if last is not None:
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)

    def detach_recurrent(self) -> None:
        """Truncated-BPTT boundary: detach every recurrent carrier (posterior
        state, prior history, last observation, and the three GRU hidden
        states) so the autograd graph does not extend past this point. The
        differentiable f/h inside each window keep BPTT intact within the
        window, which bounds gradient depth and prevents the exploding
        gradients of full 120-step BPTT through the chaotic dynamics."""
        for attr in (
            "m1x_posterior",
            "m1x_posterior_previous",
            "m1x_prior",
            "m1x_prior_previous",
            "y_previous",
            "h_Q",
            "h_Sigma",
            "h_S",
        ):
            t = getattr(self, attr, None)
            if t is not None and torch.is_tensor(t):
                setattr(self, attr, t.detach())

    def _wrap_az_channels(self, vec: torch.Tensor) -> torch.Tensor:
        """Wrap the azimuth channels of a (B, n) innovation-like vector into
        (-period/2, period/2]. Other channels untouched."""
        if not self.wrap_azimuth or self.az_channel_index.numel() == 0:
            return vec
        period = self.az_wrap_period
        out = vec.clone()
        idx = self.az_channel_index
        az = out[:, idx]
        out[:, idx] = az - period * torch.round(az / period)
        return out

    def step_prior(self):
        post = self.m1x_posterior
        if self.detach_dynamics:
            post = post.detach()
        # Alignment fix: the rollout initializes the posterior to x0_est (the
        # estimate of the state at t=0). The upstream KalmanNet always
        # propagates the posterior *before* the first measurement update, which
        # makes posterior_t estimate states[t+1] while it is scored against
        # states[t] -- a one-step (~orbital-velocity*dt ~ 150 km) systematic
        # offset. Skipping propagation on the first step makes posterior_0
        # estimate states[0], so posterior_t aligns with states[t] exactly like
        # the classical filters. With a zero gain the rollout then equals the
        # aligned pure-propagation reference.
        if int(self.sys_funcs.step_idx) <= 0:
            self.m1x_prior = post
        else:
            self.m1x_prior = self.f(post)
        self.m1y = self.h(self.m1x_prior)

    def step_KGain_est(self, y):
        obs_diff = torch.squeeze(y, 2) - torch.squeeze(self.y_previous, 2)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)
        fw_evol_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(
            self.m1x_posterior_previous, 2
        )
        fw_update_diff = torch.squeeze(self.m1x_posterior, 2) - torch.squeeze(
            self.m1x_prior_previous, 2
        )

        # gap3: wrap azimuth-channel differences before normalization.
        obs_diff = self._wrap_az_channels(obs_diff)
        obs_innov_diff = self._wrap_az_channels(obs_innov_diff)

        # R1: noise-equivalent per-channel reweighting of the obs features only
        # (direction is then L2-normalized; the actual innovation dy used in
        # KGain @ dy is NOT reweighted, so the gain magnitude is unchanged).
        if self.noise_equiv_features:
            w = self.obs_feat_weight.view(1, -1)
            obs_diff = obs_diff * w
            obs_innov_diff = obs_innov_diff * w

        obs_diff = nn.functional.normalize(obs_diff, p=2, dim=1, eps=1e-12)
        obs_innov_diff = nn.functional.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12)
        fw_evol_diff = nn.functional.normalize(fw_evol_diff, p=2, dim=1, eps=1e-12)
        fw_update_diff = nn.functional.normalize(fw_update_diff, p=2, dim=1, eps=1e-12)

        KG = self.KGain_step(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def KNet_step(self, y):
        self.step_prior()
        self.step_KGain_est(y)

        dy = y - self.m1y  # [B, n, 1]
        if self.wrap_azimuth and self.az_channel_index.numel() > 0:
            dy = self._wrap_az_channels(torch.squeeze(dy, 2)).unsqueeze(2)

        INOV = self.gain_scale * torch.bmm(self.KGain, dy)  # [B, m, 1]
        if self.posterior_clip is not None:
            clip = self.posterior_clip.view(1, self.m, 1)
            if self.straight_through_clip:
                # Straight-through clamp: the forward posterior correction is
                # hard-bounded for propagation stability, but the gradient passes
                # through unbounded. A saturating tanh bound instead zeroes the
                # gradient to the gain network on exactly the large-innovation
                # steps that matter, collapsing training (gnorm -> 0); the
                # straight-through estimator keeps the learning signal alive.
                clamped = torch.minimum(torch.maximum(INOV, -clip), clip)
                INOV = INOV + (clamped - INOV).detach()
            else:
                INOV = clip * torch.tanh(INOV / clip)

        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y
        return self.m1x_posterior


# --- SystemModel facade ----------------------------------------------------


class _SysModel:
    def __init__(self, sys_funcs: AdaptedSysFuncs, m: int, n: int) -> None:
        self.f = sys_funcs.f_torch
        self.h = sys_funcs.h_torch
        self.m = m
        self.n = n
        self.prior_Q = torch.eye(m)
        self.prior_Sigma = torch.zeros(m, m)
        self.prior_S = torch.eye(n)


# --- Data preparation (MEAS_SCALE units, optional visibility augmentation) --


def _prepare(
    bundles: list[dict[str, np.ndarray]],
    n_stations: int,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(bundles)
    steps = bundles[0]["states"].shape[0]
    los_dim = 4 * n_stations
    obs_dim = los_dim + (n_stations if augment else 0)
    y = np.zeros((n, obs_dim, steps), dtype=np.float64)
    x = np.zeros((n, 6, steps), dtype=np.float64)
    v = np.zeros((n, n_stations, steps), dtype=np.float64)
    x0 = np.zeros((n, 6, 1), dtype=np.float64)
    meas_scale_block = np.tile(np.asarray(MEAS_SCALE, dtype=np.float64), n_stations)
    for i, b in enumerate(bundles):
        meas = b["measurements"]  # (T, S, 4)
        vis = b["visibility"]  # (T, S)
        masked = meas * vis[..., None]
        flat = masked.reshape(steps, los_dim)
        scaled = flat / meas_scale_block
        y[i, :los_dim] = np.transpose(scaled, (1, 0))
        if augment:
            y[i, los_dim:] = np.transpose(vis, (1, 0))
        x[i] = np.transpose(b["states"] / STATE_SCALE, (1, 0))
        v[i] = np.transpose(vis, (1, 0))
        x0[i, :, 0] = b["x0_est"] / STATE_SCALE
    return (
        torch.from_numpy(y),
        torch.from_numpy(x),
        torch.from_numpy(v),
        torch.from_numpy(x0),
    )


# --- Cosine LR with warm-up -----------------------------------------------


def _cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    warm = max(1, warmup)
    if step <= warm:
        return base_lr * step / warm
    progress = (step - warm) / max(1, total - warm)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def _select_high_visibility_indices(
    train_vis: torch.Tensor, eval_start: int, tercile: float = 1.0 / 3.0
) -> np.ndarray:
    vis_per_step = (train_vis.sum(dim=1) > 0.5).float()  # (N, T)
    frac = vis_per_step[:, eval_start:].mean(dim=1).cpu().numpy()
    n = frac.shape[0]
    k = max(1, int(math.ceil(tercile * n)))
    return np.argsort(-frac)[:k]


# --- Rollout (shared by train / cv / predict) -----------------------------


def _rollout(
    knet: AdaptedKNet,
    sys_funcs: AdaptedSysFuncs,
    y: torch.Tensor,
    vis: torch.Tensor,
    x0: torch.Tensor,
    device: torch.device,
    m: int,
    tbptt_window: int = 0,
) -> torch.Tensor:
    """Run the full T-step rollout; returns predicted scaled state (B, m, T).

    If ``tbptt_window > 0`` and gradients are enabled, the recurrent carriers
    are detached every ``tbptt_window`` steps (windowed truncated BPTT) so the
    gradient depth is bounded while f/h stay differentiable within each window.
    """
    b = y.shape[0]
    T = y.shape[-1]
    y = y.to(device)
    vis = vis.to(device)
    knet.batch_size = b
    knet.init_hidden_KNet()
    sys_funcs.visibility_mask = None
    sys_funcs.step_idx = 0
    knet.InitSequence(x0.to(device), T)
    sys_funcs.reset()
    truncate = tbptt_window > 0 and torch.is_grad_enabled()
    x_out = torch.zeros((b, m, T), device=device, dtype=y.dtype)
    for t in range(T):
        sys_funcs.step_idx = t
        sys_funcs.set_visibility(vis[:, :, t])
        y_t = y[:, :, t].unsqueeze(2)
        x_t = knet(y_t)
        x_out[:, :, t] = torch.squeeze(x_t, 2)
        if truncate and (t + 1) % tbptt_window == 0 and t < T - 1:
            knet.detach_recurrent()
    return x_out


def _train(
    knet: AdaptedKNet,
    sys_funcs: AdaptedSysFuncs,
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
    warmup: int,
    curriculum_stage_a_steps: int,
    use_curriculum: bool,
    n_batch: int,
    base_lr: float,
    wd: float,
    use_cosine: bool,
    device: torch.device,
    eval_start: int,
    cv_every: int,
    vel_loss_weight: float,
    tbptt_window: int,
) -> dict[str, Any]:
    knet.to(device)
    optimizer = torch.optim.Adam(knet.parameters(), lr=base_lr, weight_decay=wd)
    loss_fn = nn.MSELoss(reduction="mean")
    best_cv = float("inf")
    best_state = None
    history: list[dict[str, Any]] = []
    n_train = train_input.shape[0]

    stage_a_idx = (
        _select_high_visibility_indices(train_vis, eval_start)
        if use_curriculum
        else np.arange(n_train)
    )
    rng_py = np.random.default_rng(0)
    last_cv = float("inf")
    for step_i in range(1, n_steps + 1):
        if use_curriculum and step_i <= curriculum_stage_a_steps:
            pool, stage = stage_a_idx, "A"
        else:
            pool, stage = np.arange(n_train), "B"
        bs = min(n_batch, pool.size)
        idx = rng_py.choice(pool, size=bs, replace=False)
        y_b = train_input[idx]
        x_b = train_target[idx].to(device)
        v_b = train_vis[idx]
        x0_b = train_x0[idx]

        lr_t = _cosine_lr(step_i, n_steps, base_lr, warmup) if use_cosine else base_lr
        for g in optimizer.param_groups:
            g["lr"] = lr_t

        knet.train()
        x_out = _rollout(knet, sys_funcs, y_b, v_b, x0_b, device, m, tbptt_window=tbptt_window)
        loss_pos = loss_fn(x_out[:, :3, eval_start:], x_b[:, :3, eval_start:])
        loss_vel = loss_fn(x_out[:, 3:, eval_start:], x_b[:, 3:, eval_start:])
        loss = loss_pos + vel_loss_weight * loss_vel
        optimizer.zero_grad()
        if not torch.isfinite(loss):
            train_loss = float("inf")
            grad_norm = float("nan")
            # Safety net: a non-finite loss should not happen with the dynamics
            # overflow guards, but if it does, roll back to the best checkpoint
            # so a single bad step cannot corrupt the weights for the rest of
            # training.
            if best_state is not None:
                knet.load_state_dict(best_state)
        else:
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(knet.parameters(), max_norm=1.0).item()
            )
            optimizer.step()
            train_loss = float(loss.item())

        run_cv = (step_i % cv_every == 0) or (step_i == n_steps) or (step_i == 1)
        if run_cv:
            knet.eval()
            with torch.no_grad():
                x_cv = _rollout(knet, sys_funcs, cv_input, cv_vis, cv_x0, device, m)
                cv_pos = float(
                    loss_fn(x_cv[:, :3, eval_start:], cv_target[:, :3, eval_start:].to(device)).item()
                )
                cv_vel = float(
                    loss_fn(x_cv[:, 3:, eval_start:], cv_target[:, 3:, eval_start:].to(device)).item()
                )
                cv_loss = cv_pos + vel_loss_weight * cv_vel
            last_cv = cv_loss
            if cv_loss < best_cv:
                best_cv = cv_loss
                best_state = {
                    k: val.clone().detach().cpu() for k, val in knet.state_dict().items()
                }
        else:
            cv_loss = last_cv

        history.append(
            {
                "step": step_i,
                "stage": stage,
                "lr": lr_t,
                "train_mse": train_loss,
                "cv_mse": cv_loss,
                "grad_norm": grad_norm,
            }
        )
        if step_i % 25 == 0 or step_i == 1 or step_i == n_steps:
            print(
                f"[KNet-A] step {step_i} stage={stage} lr={lr_t:.2e} "
                f"train={train_loss:.6e} cv={cv_loss:.6e} best={best_cv:.6e} "
                f"gnorm={grad_norm:.3f}",
                flush=True,
            )

    if best_state is not None:
        knet.load_state_dict(best_state)
    return {"history": history, "best_cv": best_cv}


def _predict(
    knet: AdaptedKNet,
    sys_funcs: AdaptedSysFuncs,
    test_input: torch.Tensor,
    test_vis: torch.Tensor,
    test_x0: torch.Tensor,
    device: torch.device,
    m: int,
) -> np.ndarray:
    knet.eval()
    with torch.no_grad():
        x_out = _rollout(knet, sys_funcs, test_input, test_vis, test_x0, device, m)
    arr = x_out.detach().cpu().numpy().transpose(0, 2, 1)  # (n, T, m) scaled
    return arr * STATE_SCALE.astype(np.float64)


# --- Classical baselines ---------------------------------------------------


def _run_classical(
    test_bundles, stations, meas_std, baseline_cfg, dyn, T, *, enable_perturbations: bool
) -> dict[str, np.ndarray]:
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
        enable_third_body=dyn.enable_third_body if enable_perturbations else False,
        enable_srp=dyn.enable_srp if enable_perturbations else False,
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
    pukf_rule = json.loads(
        Path("release/predeclarations/pukf_q_adaptive_rule_loop41.json").read_text()
    )["thresholds"]
    pukf_cfg = ProcessNoiseAdaptiveUKFConfig(
        q_pos_m=baseline_cfg.ukf.q_pos_m,
        q_vel_mps=baseline_cfg.ukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        window_size=int(pukf_rule["window_size"]),
        nis_per_update_expected=float(pukf_rule["nis_per_update_expected"]),
        nis_warn_ratio=float(pukf_rule["nis_warn_ratio"]),
        nis_alarm_ratio=float(pukf_rule["nis_alarm_ratio"]),
        q_scale_warn=float(pukf_rule["q_scale_warn"]),
        q_scale_alarm=float(pukf_rule["q_scale_alarm"]),
        q_scale_max=float(pukf_rule["q_scale_max"]),
        smoothing=float(pukf_rule["smoothing"]),
        angle_deweight_elev_cap_deg=getattr(baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None),
    )

    preds = {k: np.zeros_like(states_all) for k in ("EKF", "UKF", "AUKF", "PUKF")}
    for i in range(len(test_bundles)):
        preds["EKF"][i], _ = run_ekf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times_all[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=ekf_cfg, **ekf_kwargs,
        )
        preds["UKF"][i], _ = run_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times_all[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=ukf_cfg, **ekf_kwargs,
        )
        preds["AUKF"][i], _ = run_adaptive_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times_all[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=aukf_cfg, **ekf_kwargs,
        )
        preds["PUKF"][i], _, _ = run_process_noise_adaptive_ukf(
            measurements=meas_all[i], visibility=vis_all[i], times_s=times_all[i],
            stations=stations, meas_std_vector=meas_std, x0_est=x0_est_all[i],
            cfg=pukf_cfg, **ekf_kwargs,
        )
    return preds


# --- Main -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--output-dir", default="results/kalmannet_adapted")
    p.add_argument("--tag", default="loop163")
    p.add_argument("--n-train", type=int, default=160)
    p.add_argument("--n-cv", type=int, default=24)
    p.add_argument("--n-test", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=80)
    p.add_argument("--curriculum-stage-a-steps", type=int, default=200)
    p.add_argument("--n-batch", type=int, default=24)
    p.add_argument("--base-lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--in-mult", type=int, default=5)
    p.add_argument("--out-mult", type=int, default=4)
    p.add_argument("--cv-every", type=int, default=10)
    p.add_argument("--vel-loss-weight", type=float, default=1.0)
    p.add_argument("--tbptt-window", type=int, default=24,
                   help="truncated-BPTT window (steps); 0 = full-sequence BPTT. "
                        "Bounds gradient depth to prevent exploding gradients.")
    p.add_argument("--posterior-clip-pos", type=float, default=5e-3,
                   help="bounded-posterior clamp on position correction (scaled; 5e-3=50 km)")
    p.add_argument("--posterior-clip-vel", type=float, default=5e-3,
                   help="bounded-posterior clamp on velocity correction (scaled; 5e-3=50 m/s)")
    p.add_argument("--gain-scale", type=float, default=1.0,
                   help="multiplier on the learned Kalman-gain correction. <1 starts "
                        "the filter nearer pure propagation (cf. in-house KalmanNetGain 1e-3).")
    p.add_argument("--tanh-clip", action="store_true",
                   help="use a saturating tanh posterior bound instead of the default "
                        "straight-through clamp (ablation; tanh collapses the gain gradient).")
    p.add_argument("--no-zero-init-gain", action="store_true",
                   help="ABLATION: do NOT zero-initialize the gain output head. The default "
                        "(zero-init) starts the filter at pure propagation for stable learning.")
    p.add_argument("--train-seed", type=int, default=20265701)
    p.add_argument("--val-seed", type=int, default=20265702)
    p.add_argument("--test-seed", type=int, default=20265703)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--device", default="auto")
    p.add_argument("--classical-model", default="both",
                   choices=["matched", "manuscript", "both"])
    # Ablation toggles (default = all adaptations ON).
    p.add_argument("--no-augment-visibility", action="store_true")
    p.add_argument("--no-noise-equiv", action="store_true")
    p.add_argument("--no-wrap-azimuth", action="store_true")
    p.add_argument("--no-bound-posterior", action="store_true")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--no-cosine", action="store_true")
    p.add_argument("--detach-dynamics", action="store_true",
                   help="ABLATION: detach the posterior fed to f each step, "
                        "emulating the faithful numpy-bridge broken-BPTT behaviour.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    n_stations = len(stations)
    eval_start = 11
    T = dyn.steps
    augment = not args.no_augment_visibility

    init_pos_sigma = baseline_cfg.ukf.init_pos_std_m
    init_vel_sigma = baseline_cfg.ukf.init_vel_std_mps

    def _gen(rng, n):
        return [
            base._generate_one(
                rng, dataset_cfg.orbit_sampling, stations, dyn, meas_std,
                dataset_cfg.measurement_noise.outlier_prob,
                dataset_cfg.measurement_noise.outlier_scale,
                dataset_cfg.measurement_noise.random_dropout_prob,
                init_pos_sigma, init_vel_sigma,
            )
            for _ in range(n)
        ]

    print(f"[KNet-A] device={device} dtype={args.dtype} augment={augment}", flush=True)
    t0 = time.perf_counter()
    train_bundles = _gen(np.random.default_rng(args.train_seed), args.n_train)
    cv_bundles = _gen(np.random.default_rng(args.val_seed), args.n_cv)
    test_bundles = _gen(np.random.default_rng(args.test_seed), args.n_test)
    t_gen = time.perf_counter() - t0

    train_input, train_target, train_vis, train_x0 = _prepare(train_bundles, n_stations, augment)
    cv_input, cv_target, cv_vis, cv_x0 = _prepare(cv_bundles, n_stations, augment)
    test_input, test_target, test_vis, test_x0 = _prepare(test_bundles, n_stations, augment)
    # Cast to working dtype, move static tensors to device.
    for t in (train_input, train_target, train_vis, train_x0,
              cv_input, cv_target, cv_vis, cv_x0,
              test_input, test_target, test_vis, test_x0):
        pass
    train_input = train_input.to(dtype); train_target = train_target.to(dtype)
    train_vis = train_vis.to(dtype); train_x0 = train_x0.to(dtype)
    cv_input = cv_input.to(dtype).to(device); cv_target = cv_target.to(dtype)
    cv_vis = cv_vis.to(dtype).to(device); cv_x0 = cv_x0.to(dtype).to(device)
    test_input = test_input.to(dtype); test_target = test_target.to(dtype)
    test_vis = test_vis.to(dtype); test_x0 = test_x0.to(dtype)

    # --- Build differentiable dynamics + sys funcs ---
    torch_dyn = SPOTODTorchDynamics(
        stations,
        dt_s=dyn.dt_s,
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
        state_scale=np.asarray(STATE_SCALE, dtype=np.float64),
        device=device,
        dtype=dtype,
    )
    sys_funcs = AdaptedSysFuncs(
        torch_dyn,
        times_s=np.arange(T, dtype=np.float64) * dyn.dt_s,
        augment_visibility=augment,
        device=device,
        dtype=dtype,
    )
    obs_dim = sys_funcs.obs_dim

    # noise-equivalent per-channel feature weights (scaled units), unit-mean
    # over the LoS channels so the visibility-flag channels remain comparable.
    sigma_scaled = np.asarray(meas_std, dtype=np.float64) / np.asarray(MEAS_SCALE, dtype=np.float64)
    los_w = 1.0 / np.clip(sigma_scaled, 1e-12, None)  # (4,)
    los_w = los_w / np.mean(los_w)  # unit mean over the 4 LoS channels
    feat_w = np.tile(los_w, n_stations)
    if augment:
        feat_w = np.concatenate([feat_w, np.ones(n_stations)])
    obs_feat_weight = torch.as_tensor(feat_w, dtype=dtype, device=device)

    az_idx = torch.as_tensor(
        [4 * s + 1 for s in range(n_stations)], dtype=torch.long, device=device
    )
    az_wrap_period = 2.0 * np.pi / float(MEAS_SCALE[1])  # az scaled by pi -> period 2.0

    posterior_clip = None
    if not args.no_bound_posterior:
        clip_vec = np.array(
            [args.posterior_clip_pos] * 3 + [args.posterior_clip_vel] * 3, dtype=np.float64
        )
        posterior_clip = torch.as_tensor(clip_vec, dtype=dtype, device=device)

    sys_model = _SysModel(sys_funcs, m=6, n=obs_dim)
    knet = AdaptedKNet()

    class _Args:
        pass

    knet_args = _Args()
    knet_args.use_cuda = device.type == "cuda"
    knet_args.n_batch = max(args.n_batch, args.n_cv, args.n_test)
    knet_args.in_mult_KNet = args.in_mult
    knet_args.out_mult_KNet = args.out_mult
    knet.NNBuild(sys_model, knet_args)
    knet.configure_adaptation(
        sys_funcs=sys_funcs,
        obs_feat_weight=obs_feat_weight,
        az_channel_index=az_idx,
        az_wrap_period=az_wrap_period,
        posterior_clip=posterior_clip,
        detach_dynamics=bool(args.detach_dynamics),
        noise_equiv_features=not args.no_noise_equiv,
        wrap_azimuth=not args.no_wrap_azimuth,
        gain_scale=args.gain_scale,
        straight_through_clip=not args.tanh_clip,
    )
    if not args.no_zero_init_gain:
        knet.zero_init_gain()
    knet = knet.to(dtype).to(device)
    # prior buffers used in init_hidden_KNet must match dtype/device.
    knet.prior_Q = knet.prior_Q.to(dtype).to(device)
    knet.prior_Sigma = knet.prior_Sigma.to(dtype).to(device)
    knet.prior_S = knet.prior_S.to(dtype).to(device)
    n_params = int(sum(p.numel() for p in knet.parameters() if p.requires_grad))
    print(f"[KNet-A] params={n_params} m=6 n={obs_dim} T={T}", flush=True)

    # --- Train ---
    t0 = time.perf_counter()
    train_out = _train(
        knet, sys_funcs,
        train_input, train_target, train_vis, train_x0,
        cv_input, cv_target, cv_vis, cv_x0,
        m=6,
        n_steps=args.n_steps,
        warmup=args.warmup,
        curriculum_stage_a_steps=args.curriculum_stage_a_steps,
        use_curriculum=not args.no_curriculum,
        n_batch=args.n_batch,
        base_lr=args.base_lr,
        wd=args.wd,
        use_cosine=not args.no_cosine,
        device=device,
        eval_start=eval_start,
        cv_every=args.cv_every,
        vel_loss_weight=args.vel_loss_weight,
        tbptt_window=args.tbptt_window,
    )
    t_train = time.perf_counter() - t0

    # --- Predict ---
    knet_pred = _predict(knet, sys_funcs, test_input, test_vis, test_x0, device, 6)

    states_all = np.stack([b["states"] for b in test_bundles], axis=0)
    vis_all = np.stack([b["visibility"] for b in test_bundles], axis=0)
    knet_rmse = base._per_traj_observed_pos_rmse(states_all, knet_pred, vis_all, eval_start)

    # Pure-propagation reference (no measurements): propagate the noisy x0 with
    # the same torch f for T steps. This is the no-skill ceiling -- any estimator
    # must beat it by using measurements. Scored on the same observed-step rule.
    with torch.no_grad():
        x_prop = test_x0.to(device).squeeze(2)  # (N, 6) scaled
        prop_states = torch.zeros((x_prop.shape[0], 6, T), device=device, dtype=x_prop.dtype)
        prop_states[:, :, 0] = x_prop
        for t in range(1, T):
            x_prop = torch_dyn.f_scaled(x_prop)
            prop_states[:, :, t] = x_prop
        prop_pred = prop_states.cpu().numpy().transpose(0, 2, 1) * STATE_SCALE.astype(np.float64)
    prop_rmse = base._per_traj_observed_pos_rmse(states_all, prop_pred, vis_all, eval_start)
    prop_mean = float(np.nanmean(prop_rmse))

    # --- Classical baselines ---
    classical_blocks: dict[str, dict[str, Any]] = {}
    variants = []
    if args.classical_model in ("matched", "both"):
        variants.append(("matched", False))
    if args.classical_model in ("manuscript", "both"):
        variants.append(("manuscript", True))
    classical_rmse_by_variant: dict[str, dict[str, np.ndarray]] = {}
    for vname, enable in variants:
        preds = _run_classical(
            test_bundles, stations, meas_std, baseline_cfg, dyn, T,
            enable_perturbations=enable,
        )
        rmse = {
            k: base._per_traj_observed_pos_rmse(states_all, preds[k], vis_all, eval_start)
            for k in preds
        }
        classical_rmse_by_variant[vname] = rmse
        classical_blocks[vname] = {
            "observed_step_rmse_mean_m": {k: float(np.nanmean(v)) for k, v in rmse.items()},
            "observed_step_rmse_median_m": {k: float(np.nanmedian(v)) for k, v in rmse.items()},
            "enable_third_body_srp": enable,
        }

    # Primary comparison: against the fair matched-model classical baseline if
    # available, else manuscript.
    primary_variant = "matched" if "matched" in classical_rmse_by_variant else "manuscript"
    primary_rmse = classical_rmse_by_variant[primary_variant]
    primary_means = {k: float(np.nanmean(v)) for k, v in primary_rmse.items()}
    best_classical = min(primary_means, key=lambda k: primary_means[k])
    knet_mean = float(np.nanmean(knet_rmse))
    diffs = knet_rmse - primary_rmse[best_classical]
    mean_d, lo, hi = base._paired_bootstrap_ci(
        diffs, n_boot=int(args.bootstrap_samples), seed=args.test_seed
    )
    finite = diffs[np.isfinite(diffs)]
    paired = {
        "primary_variant": primary_variant,
        "best_classical": best_classical,
        "best_classical_mean_m": primary_means[best_classical],
        "knet_mean_m": knet_mean,
        "mean_knet_minus_best_m": mean_d,
        "ci_lo_m": lo,
        "ci_hi_m": hi,
        "n_paired": int(finite.size),
        "knet_better_count": int(np.sum(finite < 0.0)) if finite.size else 0,
        "knet_better_rate_percent": float(100.0 * np.mean(finite < 0.0)) if finite.size else float("nan"),
    }

    # Fixed success criterion (declared in this script before the run; this is
    # an exploratory engineering study, not a pre-registered confirmatory test):
    # the adapted KalmanNet is a *win* iff it is strictly the lowest-mean
    # estimator AND the paired-bootstrap CI vs the best classical is strictly
    # below zero AND the gap exceeds the manuscript-wide 3% practical floor.
    floor_pct = 3.0
    floor_m = floor_pct / 100.0 * primary_means[best_classical]
    all_means = dict(primary_means); all_means["KalmanNet-Adapted"] = knet_mean
    knet_strictly_lowest = all_means["KalmanNet-Adapted"] == min(all_means.values())
    ci_strictly_negative = hi < 0.0
    floor_exceeded = (-mean_d > floor_m) and (mean_d < 0.0)
    is_win = bool(knet_strictly_lowest and ci_strictly_negative and floor_exceeded)
    if is_win:
        outcome = "win_above_floor"
    elif ci_strictly_negative:
        outcome = "win_below_floor"
    elif lo > 0:
        outcome = "loss_ci_strictly_positive"
    else:
        outcome = "inconclusive_ci_contains_zero"

    decision = {
        "success_criterion": (
            "strictly lowest mean AND paired CI strictly below 0 vs best classical "
            "AND |gap| > 3% practical floor"
        ),
        "best_classical": best_classical,
        "best_classical_mean_m": primary_means[best_classical],
        "knet_mean_m": knet_mean,
        "knet_minus_best_mean_m": mean_d,
        "ci_lo_m": lo,
        "ci_hi_m": hi,
        "practical_floor_percent": floor_pct,
        "practical_floor_m": floor_m,
        "knet_is_strictly_lowest_mean": bool(knet_strictly_lowest),
        "ci_strictly_negative": bool(ci_strictly_negative),
        "floor_exceeded": bool(floor_exceeded),
        "is_win": is_win,
        "outcome_class": outcome,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adaptations = {
        "differentiable_torch_f_h_bptt": not bool(args.detach_dynamics),
        "noise_equivalent_obs_features": not args.no_noise_equiv,
        "bounded_posterior": not args.no_bound_posterior,
        "angle_innovation_wrapping": not args.no_wrap_azimuth,
        "visibility_flag_augmentation": augment,
        "visibility_curriculum": not args.no_curriculum,
        "cosine_lr_warmup": not args.no_cosine,
    }
    payload: dict[str, Any] = {
        "schema_version": "kalmannet_adapted_spot_od_v1",
        "scenario": "kalmannet_adapted_spot_od",
        "tag": args.tag,
        "vendor_commit": base.VENDOR_COMMIT,
        "vendor_path": str(base.VENDOR_ROOT.relative_to(base.REPO_ROOT)),
        "device": str(device),
        "dtype": args.dtype,
        "n_params_kalmannet": n_params,
        "config": {
            "n_train": args.n_train, "n_cv": args.n_cv, "n_test": args.n_test,
            "n_steps": args.n_steps, "warmup": args.warmup,
            "curriculum_stage_a_steps": args.curriculum_stage_a_steps,
            "n_batch": args.n_batch, "base_lr": args.base_lr, "wd": args.wd,
            "in_mult": args.in_mult, "out_mult": args.out_mult,
            "vel_loss_weight": args.vel_loss_weight,
            "tbptt_window": args.tbptt_window,
            "gain_scale": args.gain_scale,
            "posterior_clip_pos": args.posterior_clip_pos,
            "posterior_clip_vel": args.posterior_clip_vel,
            "posterior_clip_mode": "tanh" if args.tanh_clip else "straight_through",
            "zero_init_gain": not args.no_zero_init_gain,
            "train_seed": args.train_seed, "val_seed": args.val_seed,
            "test_seed": args.test_seed, "T": T, "m": 6, "n": obs_dim,
            "eval_start_step": eval_start,
        },
        "adaptations_enabled": adaptations,
        "elapsed_seconds": {"data_generation": t_gen, "training": t_train},
        "best_cv_mse": float(train_out["best_cv"]),
        "knet_observed_step_rmse_mean_m": knet_mean,
        "knet_observed_step_rmse_median_m": float(np.nanmedian(knet_rmse)),
        "pure_propagation_reference_mean_m": prop_mean,
        "pure_propagation_reference_median_m": float(np.nanmedian(prop_rmse)),
        "classical_baselines": classical_blocks,
        "paired_vs_best_classical": paired,
        "decision": decision,
        "training_history": train_out["history"],
    }
    out_json = out_dir / f"kalmannet_adapted_{args.tag}.json"
    out_json.write_text(json.dumps(payload, indent=2, default=float))

    rows = []
    for i in range(len(test_bundles)):
        row: dict[str, Any] = {"trajectory_index": i}
        row["KalmanNet-Adapted_observed_pos_rmse_m"] = (
            float(knet_rmse[i]) if np.isfinite(knet_rmse[i]) else None
        )
        for vname, rmse in classical_rmse_by_variant.items():
            for k, arr in rmse.items():
                row[f"{k}_{vname}_observed_pos_rmse_m"] = (
                    float(arr[i]) if np.isfinite(arr[i]) else None
                )
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"kalmannet_adapted_{args.tag}.csv", index=False)

    print(json.dumps(
        {
            "knet_mean_m": knet_mean,
            "knet_median_m": float(np.nanmedian(knet_rmse)),
            "pure_propagation_mean_m": prop_mean,
            "classical_means": {v: classical_blocks[v]["observed_step_rmse_mean_m"] for v in classical_blocks},
            "paired_vs_best_classical": paired,
            "decision_outcome": outcome,
            "adaptations_enabled": adaptations,
            "elapsed_seconds": payload["elapsed_seconds"],
        },
        indent=2, default=float,
    ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
