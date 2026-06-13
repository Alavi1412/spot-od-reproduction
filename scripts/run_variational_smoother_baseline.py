#!/usr/bin/env python
"""Compute a robust fixed-interval variational smoother OD reference.

The smoother (RFIS) estimates the complete Cartesian state arc for each
trajectory by jointly minimizing measurement residuals, one-step dynamics
residuals, and a soft initial-state prior under a robust loss. It is an
offline baseline: future measurements are available to every state in the
arc, so it is reported separately from causal filters.

Design notes
------------
* The optimization variable is the per-step state *correction* expressed in
  units of the initial-state prior std (the same well-conditioned
  parametrization used by ``run_batch_wls_baseline.py``). Baking the raw
  1e7 m state scale into the variable, as an earlier draft did, made the
  Jacobian columns span ~10 orders of magnitude and starved the optimizer of
  usable steps -- the cause of the 0.0 ``fit_success_rate`` smoke symptom.
* ``scipy`` reports ``success`` only for ftol/xtol/gtol termination. A
  fixed-interval smoother with hundreds of variables routinely hits the
  evaluation budget while still having materially reduced the objective and
  produced a usable arc. Outputs therefore classify each fit
  (``converged`` / ``max_nfev_improved`` / ``max_nfev_stalled`` /
  ``diverged`` / ``no_measurements`` / ``exception``) and define
  ``fit_success_rate`` as the fraction of *usable* fits, with the stricter
  converged rate reported alongside.
* Usability is decided from intrinsic quantities only (finite, physically
  plausible state, reduced objective) -- never from ground truth.
* Paper-facing JSON/CSV contain only scenario, parameters, metrics, and an
  honest verdict: no local paths, environment, or code-structure prose.
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
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Keep BLAS single-threaded so process-level parallelism does not oversubscribe
# CPU cores (each worker solves a small dense/sparse LM system).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from gnn_state_estimation.coordinates import line_of_sight_measurement
from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.dynamics import rk4_step
from gnn_state_estimation.evaluation import score_predictions
from gnn_state_estimation.filters.ekf import wrap_angle_pi
from gnn_state_estimation.scenarios import estimator_sim_config
from gnn_state_estimation.simulation import DatasetConfig, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


# Plausible LEO/MEO state magnitudes used only to reject a diverged solve
# (radius from slightly inside Earth radius out to well past GEO, speed cap).
_R_MIN_M = 6.0e6
_R_MAX_M = 6.0e7
_V_MAX_MPS = 2.0e4
# A fit counts as having improved the objective when the raw sum of squared
# residuals drops by at least this fraction relative to the warm start.
_MIN_OBJECTIVE_REDUCTION = 0.02

# Warm starts the smoother can be initialized from. ``EKF``/``UKF``/``AUKF``
# read recursive-filter prior arrays carried in the dataset; ``BatchWLS`` reads
# a sidecar prediction file and is only offered when that file is present.
_RECURSIVE_WARM_STARTS = ("EKF", "UKF", "AUKF")
_ALL_WARM_STARTS = _RECURSIVE_WARM_STARTS + ("BatchWLS",)


def _parse_warm_starts(raw: str) -> list[str]:
    """Normalize the ``--warm-starts`` CSV to an ordered, de-duplicated list.

    Matching is case-insensitive; unknown tokens raise so a typo never
    silently collapses the multistart down to a single arc.
    """
    canonical = {name.lower(): name for name in _ALL_WARM_STARTS}
    out: list[str] = []
    for token in (t.strip() for t in raw.split(",")):
        if not token:
            continue
        key = token.lower()
        if key not in canonical:
            raise ValueError(
                f"unknown warm start {token!r}; choose from {list(_ALL_WARM_STARTS)}"
            )
        name = canonical[key]
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError("--warm-starts resolved to an empty list")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output-dir", type=str, default="results/variational_smoother_baseline")
    parser.add_argument("--scenarios", type=str, default="process_noise_shift_test,maneuver_shift_test")
    parser.add_argument("--max-nfev", type=int, default=240)
    parser.add_argument("--loss", type=str, default="huber", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--huber-f-scale", type=float, default=2.5)
    parser.add_argument("--prior-weight", type=float, default=0.20)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--process-pos-std-m", type=float, default=75.0)
    parser.add_argument("--process-vel-std-mps", type=float, default=0.75)
    parser.add_argument("--correction-bound", type=float, default=40.0)
    parser.add_argument("--trajectory-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--warm-starts",
        type=str,
        default="AUKF",
        help="Comma list of warm starts to try per trajectory; the arc with "
        "the lowest internal final SSR is selected (never truth). Accepted: "
        "EKF, UKF, AUKF, and BatchWLS when WLS predictions are available.",
    )
    parser.add_argument(
        "--wls-baseline-dir",
        type=str,
        default="results/batch_wls_baseline",
        help="Directory holding batch WLS predictions; the matching scenario "
        "is added to comparisons only if its predictions file exists.",
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


def scenario_dataset_config(cfg: dict[str, Any], scenario: str) -> DatasetConfig:
    sim_cfg = copy.deepcopy(cfg["simulation"])
    if scenario == "stress_test":
        sim_cfg = deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg:
        # RFIS is an offline estimator/OD reference: use the estimator-side
        # config (== truth config unless the scenario declares
        # estimator_overrides).
        sim_cfg = estimator_sim_config(sim_cfg, scenario_cfg)
    return parse_dataset_config(sim_cfg)


def _rk4_step_cfg(x: np.ndarray, dataset_cfg: DatasetConfig, t_s: float, dt: float) -> np.ndarray:
    dyn = dataset_cfg.dynamics
    return rk4_step(
        x,
        dt=dt,
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        t_s=t_s,
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


def _visible_pairs(visibility: np.ndarray) -> np.ndarray:
    pairs = np.argwhere(np.asarray(visibility, dtype=np.float64) >= 0.5)
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return pairs.astype(np.int64, copy=False)


def _make_jacobian_fn(
    *,
    t_len: int,
    visible_pairs: np.ndarray,
    times_s: np.ndarray,
    dt: np.ndarray,
    x_ref: np.ndarray,
    opt_scale_row: np.ndarray,
    init_scale: np.ndarray,
    process_scale: np.ndarray,
    meas_std: np.ndarray,
    dataset_cfg: DatasetConfig,
    prior_weight: float,
    anchor_weight: float,
    anchor_on: bool,
):
    """Build an analytic, block-structured sparse Jacobian callable.

    The residual is the concatenation of (i) a soft initial-state prior, (ii)
    one-step dynamics defects ``x_{k+1} - f(x_k)``, (iii) line-of-sight
    measurement residuals, and (iv) an optional anchor. Its Jacobian is
    therefore sparse with only three nontrivial dense kernels: the 6x6
    one-step state-transition ``F_k`` and the 4x6 measurement sensitivity
    ``H_p``, both obtained by *local* central differences (one RK4 step / one
    line-of-sight evaluation), and constant diagonal prior/anchor blocks.
    This replaces SciPy's global finite-difference Jacobian, which re-ran the
    full arc per color group and was the dominant cost / convergence stall.
    """
    s = np.asarray(opt_scale_row, dtype=np.float64).reshape(-1)
    inv_proc = 1.0 / np.asarray(process_scale, dtype=np.float64)
    inv_meas = 1.0 / np.asarray(meas_std, dtype=np.float64)
    init_scale = np.asarray(init_scale, dtype=np.float64)
    n_vars = t_len * 6
    n_dyn = (t_len - 1) * 6
    n_pairs = int(visible_pairs.shape[0])
    base_dyn = 6
    base_meas = base_dyn + n_dyn
    n_res = base_meas + n_pairs * 4 + (t_len * 6 if anchor_on else 0)

    def jac(u: np.ndarray):
        states = x_ref + np.asarray(u, dtype=np.float64).reshape(t_len, 6) * opt_scale_row
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        # Prior block (diagonal): d/d u[0,i] = prior_weight * s[i] / init_scale[i].
        for i in range(6):
            rows.append(i)
            cols.append(i)
            vals.append(float(prior_weight) * s[i] / init_scale[i])
        # Dynamics defect blocks.
        for k in range(t_len - 1):
            xk = states[k]
            tk = float(times_s[k])
            dtk = float(dt[k])
            f_k = np.empty((6, 6), dtype=np.float64)
            for j in range(6):
                hj = 1.0e-4 * (abs(xk[j]) + 1.0)
                xp = xk.copy()
                xp[j] += hj
                xm = xk.copy()
                xm[j] -= hj
                f_k[:, j] = (
                    _rk4_step_cfg(xp, dataset_cfg, tk, dtk)
                    - _rk4_step_cfg(xm, dataset_cfg, tk, dtk)
                ) / (2.0 * hj)
            r0 = base_dyn + 6 * k
            blk = -(f_k * s[None, :]) * inv_proc[:, None]  # d defect / d u[k]
            for a in range(6):
                for b in range(6):
                    rows.append(r0 + a)
                    cols.append(6 * k + b)
                    vals.append(blk[a, b])
                rows.append(r0 + a)
                cols.append(6 * (k + 1) + a)
                vals.append(s[a] * inv_proc[a])  # d defect / d u[k+1]
        # Measurement blocks.
        for p in range(n_pairs):
            ti = int(visible_pairs[p, 0])
            si = int(visible_pairs[p, 1])
            xt = states[ti]
            tt = float(times_s[ti])
            stn = dataset_cfg.stations[si]
            h_p = np.empty((4, 6), dtype=np.float64)
            for j in range(6):
                hj = 1.0e-4 * (abs(xt[j]) + 1.0)
                xp = xt.copy()
                xp[j] += hj
                xm = xt.copy()
                xm[j] -= hj
                zp, _ = line_of_sight_measurement(xp, stn, tt)
                zm, _ = line_of_sight_measurement(xm, stn, tt)
                dz = np.asarray(zp, dtype=np.float64) - np.asarray(zm, dtype=np.float64)
                dz[1] = wrap_angle_pi(float(zp[1] - zm[1]))
                h_p[:, j] = dz / (2.0 * hj)
            r0 = base_meas + 4 * p
            blk = -(h_p * s[None, :]) * inv_meas[:, None]  # residual = (meas - z)/std
            for a in range(4):
                for b in range(6):
                    rows.append(r0 + a)
                    cols.append(6 * ti + b)
                    vals.append(blk[a, b])
        # Anchor block (diagonal): residual = anchor_weight * u.
        if anchor_on:
            r0 = base_meas + n_pairs * 4
            for k in range(t_len * 6):
                rows.append(r0 + k)
                cols.append(k)
                vals.append(float(anchor_weight))
        return csr_matrix(
            (np.asarray(vals, dtype=np.float64), (rows, cols)),
            shape=(n_res, n_vars),
        )

    return jac


def _masked_pos_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    err = y_true[mask, :3] - y_pred[mask, :3]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def _observed_metrics(states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int) -> dict[str, float]:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    zero_visible = ~observed
    y_true = states[:, eval_start:]
    y_pred = preds[:, eval_start:]
    return {
        "observed_step_pos_rmse_m": _masked_pos_rmse(y_true, y_pred, observed),
        "zero_visible_pos_rmse_m": _masked_pos_rmse(y_true, y_pred, zero_visible),
        "observed_steps": int(np.sum(observed)),
        "zero_visible_steps": int(np.sum(zero_visible)),
    }


def _trajectory_rmse_values(y_true: np.ndarray, y_pred: np.ndarray, eval_start: int) -> np.ndarray:
    err = y_true[:, eval_start:, :3] - y_pred[:, eval_start:, :3]
    return np.sqrt(np.mean(np.sum(err * err, axis=-1), axis=1))


def _method_summary(states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int) -> dict[str, float]:
    all_step = score_predictions(states[:, eval_start:], preds[:, eval_start:])
    obs = _observed_metrics(states, preds, visibility, eval_start)
    traj = _trajectory_rmse_values(states, preds, eval_start)
    return {
        "all_step_pos_rmse_m": float(all_step["pos_rmse_m"]),
        "all_step_vel_rmse_mps": float(all_step["vel_rmse_mps"]),
        "median_traj_pos_rmse_m": float(np.median(traj)),
        "max_traj_pos_rmse_m": float(np.max(traj)),
        "failure_rate_100km": float(np.mean(traj > 100_000.0)),
        **obs,
    }


def _ssr(residual: np.ndarray) -> float:
    """Raw sum of squared residuals (loss-agnostic, comparable across fits)."""
    r = np.asarray(residual, dtype=np.float64)
    if not np.all(np.isfinite(r)):
        return float("inf")
    return float(np.dot(r, r))


def _state_arc_is_sane(arc: np.ndarray) -> bool:
    if not np.all(np.isfinite(arc)):
        return False
    radius = np.linalg.norm(arc[:, :3], axis=1)
    speed = np.linalg.norm(arc[:, 3:], axis=1)
    return bool(
        np.all(radius > _R_MIN_M)
        and np.all(radius < _R_MAX_M)
        and np.all(speed < _V_MAX_MPS)
    )


def fit_smoother_trajectory(
    *,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    x0_est: np.ndarray,
    warm_start: np.ndarray,
    dataset_cfg: DatasetConfig,
    meas_std: np.ndarray,
    init_scale: np.ndarray,
    process_scale: np.ndarray,
    max_nfev: int,
    loss: str,
    huber_f_scale: float,
    prior_weight: float,
    anchor_weight: float,
    correction_bound: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    t_len = int(warm_start.shape[0])
    x_ref = np.asarray(warm_start, dtype=np.float64)
    visible_pairs = _visible_pairs(visibility)
    if visible_pairs.shape[0] == 0:
        return x_ref.copy(), {
            "success": False,
            "usable": False,
            "outcome": "no_measurements",
            "status": 0,
            "message": "no_visible_measurements",
            "nfev": 0,
            "num_measurements": 0,
            "cost": float("nan"),
            "optimality": float("nan"),
            "ssr_initial": float("nan"),
            "ssr_final": float("nan"),
            "ssr_selected": float("inf"),
            "objective_reduction_ratio": float("nan"),
            "mean_correction_norm_m": 0.0,
            "max_correction_norm_m": 0.0,
        }

    x0_est = np.asarray(x0_est, dtype=np.float64)
    process_scale = np.asarray(process_scale, dtype=np.float64)
    init_scale = np.asarray(init_scale, dtype=np.float64)
    meas_std = np.asarray(meas_std, dtype=np.float64)
    # Optimization variable: per-step correction in units of the prior std.
    # u ~ O(1) for realistic warm-start errors -> well-conditioned Jacobian.
    opt_scale = init_scale.reshape(1, 6)
    anchor_on = anchor_weight > 0.0
    dt = np.diff(times_s).astype(np.float64)

    def unpack(u: np.ndarray) -> np.ndarray:
        return x_ref + np.asarray(u, dtype=np.float64).reshape(t_len, 6) * opt_scale

    def residual_fn(u: np.ndarray) -> np.ndarray:
        states = unpack(u)
        residuals: list[np.ndarray] = [
            float(prior_weight) * (states[0] - x0_est) / init_scale
        ]
        dyn_res = np.empty((t_len - 1, 6), dtype=np.float64)
        for k in range(t_len - 1):
            pred_next = _rk4_step_cfg(states[k], dataset_cfg, float(times_s[k]), float(dt[k]))
            dyn_res[k] = (states[k + 1] - pred_next) / process_scale
        residuals.append(dyn_res.reshape(-1))
        meas_res = np.empty((visible_pairs.shape[0], 4), dtype=np.float64)
        for out_idx, (t_idx, s_idx) in enumerate(visible_pairs):
            t_i = int(t_idx)
            s_i = int(s_idx)
            z_pred, _ = line_of_sight_measurement(states[t_i], dataset_cfg.stations[s_i], float(times_s[t_i]))
            y = np.asarray(measurements[t_i, s_i], dtype=np.float64) - z_pred
            y[1] = wrap_angle_pi(float(y[1]))
            meas_res[out_idx] = y / meas_std
        residuals.append(meas_res.reshape(-1))
        if anchor_on:
            residuals.append(float(anchor_weight) * np.asarray(u, dtype=np.float64))
        out = np.concatenate(residuals)
        return np.nan_to_num(out, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)

    jacobian_fn = _make_jacobian_fn(
        t_len=t_len,
        visible_pairs=visible_pairs,
        times_s=np.asarray(times_s, dtype=np.float64),
        dt=dt,
        x_ref=x_ref,
        opt_scale_row=opt_scale,
        init_scale=init_scale,
        process_scale=process_scale,
        meas_std=meas_std,
        dataset_cfg=dataset_cfg,
        prior_weight=prior_weight,
        anchor_weight=anchor_weight,
        anchor_on=anchor_on,
    )

    u0 = np.zeros(t_len * 6, dtype=np.float64)
    ssr_initial = _ssr(residual_fn(u0))
    bound = float(abs(correction_bound))
    result = least_squares(
        residual_fn,
        x0=u0,
        jac=jacobian_fn,
        bounds=(-bound, bound),
        loss=loss,
        f_scale=float(huber_f_scale),
        max_nfev=int(max_nfev),
        tr_solver="lsmr",
        x_scale=1.0,
        ftol=1.0e-8,
        xtol=1.0e-8,
        gtol=1.0e-8,
        verbose=0,
    )
    prediction = unpack(result.x)
    final_residual = residual_fn(result.x)
    ssr_final = _ssr(final_residual)
    finite_residual = bool(np.all(np.isfinite(final_residual)))

    if np.isfinite(ssr_initial) and ssr_initial > 0.0 and np.isfinite(ssr_final):
        objective_reduction_ratio = float(1.0 - ssr_final / ssr_initial)
    else:
        objective_reduction_ratio = float("nan")
    improved = (
        np.isfinite(objective_reduction_ratio)
        and objective_reduction_ratio >= _MIN_OBJECTIVE_REDUCTION
    )
    sane = _state_arc_is_sane(prediction) and finite_residual

    if not sane:
        # Diverged solve: fall back to the warm start so downstream metrics
        # reflect a usable arc rather than an exploded one.
        prediction = x_ref.copy()
        outcome = "diverged"
        usable = False
    elif int(result.status) > 0:
        outcome = "converged"
        usable = True
    elif improved:
        outcome = "max_nfev_improved"
        usable = True
    else:
        outcome = "max_nfev_stalled"
        usable = False

    # SSR of the arc we will actually emit: the optimized residual unless the
    # solve diverged and we fell back to the warm start (then it is the
    # warm-start SSR, == ssr_initial). This is the loss-agnostic, truth-free
    # quantity multistart selection compares across warm starts.
    ssr_selected = ssr_initial if outcome == "diverged" else ssr_final

    pos_corr = np.linalg.norm(prediction[:, :3] - x_ref[:, :3], axis=1)
    return prediction, {
        "success": bool(result.success),
        "usable": bool(usable),
        "outcome": outcome,
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "num_measurements": int(visible_pairs.shape[0]),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "ssr_initial": float(ssr_initial),
        "ssr_final": float(ssr_final),
        "ssr_selected": float(ssr_selected),
        "objective_reduction_ratio": objective_reduction_ratio,
        "mean_correction_norm_m": float(np.mean(pos_corr)),
        "max_correction_norm_m": float(np.max(pos_corr)),
    }


def _one_warm_start(
    payload: dict[str, Any], name: str, arc: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        pred, info = fit_smoother_trajectory(
            measurements=payload["measurements"],
            visibility=payload["visibility"],
            times_s=payload["times_s"],
            x0_est=payload["x0_est"],
            warm_start=arc,
            dataset_cfg=payload["dataset_cfg"],
            meas_std=payload["meas_std"],
            init_scale=payload["init_scale"],
            process_scale=payload["process_scale"],
            max_nfev=int(payload["max_nfev"]),
            loss=str(payload["loss"]),
            huber_f_scale=float(payload["huber_f_scale"]),
            prior_weight=float(payload["prior_weight"]),
            anchor_weight=float(payload["anchor_weight"]),
            correction_bound=float(payload["correction_bound"]),
        )
    except Exception as exc:  # noqa: BLE001 - keep one bad arc from killing the run
        pred = np.asarray(arc, dtype=np.float64).copy()
        info = {
            "success": False,
            "usable": False,
            "outcome": "exception",
            "status": -1,
            # Type only: never leak paths/tracebacks into paper-facing CSV.
            "message": f"exception:{type(exc).__name__}",
            "nfev": 0,
            "num_measurements": 0,
            "cost": float("nan"),
            "optimality": float("nan"),
            "ssr_initial": float("nan"),
            "ssr_final": float("nan"),
            "ssr_selected": float("inf"),
            "objective_reduction_ratio": float("nan"),
            "mean_correction_norm_m": 0.0,
            "max_correction_norm_m": 0.0,
        }
    info["warm_start"] = name
    return pred, info


def _fit_single_worker(
    payload: dict[str, Any],
) -> tuple[int, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Run every requested warm start for one trajectory and select by SSR.

    Selection compares ``ssr_selected`` -- the internal objective of the arc
    each start would emit -- and keeps the minimum. It never consults ground
    truth. Ties / all-infeasible fall back to the first requested start so the
    output is always a usable (if unimproved) arc.
    """
    idx = int(payload["idx"])
    starts: list[dict[str, Any]] = payload["warm_starts"]
    per_start: list[tuple[np.ndarray, dict[str, Any]]] = []
    for spec in starts:
        name = str(spec["name"])
        arc = np.asarray(spec["arc"], dtype=np.float64)
        pred, info = _one_warm_start(payload, name, arc)
        info["trajectory"] = idx
        per_start.append((pred, info))

    best_pos = 0
    best_ssr = float("inf")
    for pos, (_, info) in enumerate(per_start):
        ssr = float(info.get("ssr_selected", float("inf")))
        if np.isfinite(ssr) and ssr < best_ssr:
            best_ssr = ssr
            best_pos = pos

    n_usable = int(sum(1 for _, info in per_start if bool(info["usable"])))
    for pos, (_, info) in enumerate(per_start):
        info["selected"] = bool(pos == best_pos)

    selected_pred, selected_src = per_start[best_pos]
    selected_info = dict(selected_src)
    selected_info["selected_warm_start"] = selected_info["warm_start"]
    selected_info["n_warm_starts_evaluated"] = int(len(per_start))
    selected_info["n_warm_starts_usable"] = n_usable
    selected_info["selected_ssr"] = float(best_ssr)
    all_infos = [dict(info) for _, info in per_start]
    return idx, selected_pred, selected_info, all_infos


def _gain_percent(reference: float, candidate: float) -> float:
    """Signed % improvement of candidate over reference (positive = better)."""
    if not np.isfinite(reference) or abs(reference) < 1e-9 or not np.isfinite(candidate):
        return float("nan")
    return float(100.0 * (reference - candidate) / reference)


def _load_wls_predictions(wls_dir: Path, scenario: str, n_traj: int, ref_shape: tuple[int, ...]) -> np.ndarray | None:
    path = wls_dir / scenario / "batch_wls_predictions.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        if "batch_wls" not in data.files:
            return None
        preds = np.asarray(data["batch_wls"], dtype=np.float64)
    except Exception:  # noqa: BLE001 - a malformed sidecar must not break the run
        return None
    if preds.ndim != 3 or preds.shape[0] < n_traj or preds.shape[1:] != ref_shape[1:]:
        return None
    return preds[:n_traj]


def run_scenario(
    cfg: dict[str, Any],
    scenario: str,
    output_dir: Path,
    *,
    eval_start: int,
    max_nfev: int,
    loss: str,
    huber_f_scale: float,
    prior_weight: float,
    anchor_weight: float,
    process_pos_std_m: float,
    process_vel_std_mps: float,
    correction_bound: float,
    trajectory_limit: int,
    workers: int,
    wls_dir: Path,
    warm_starts: list[str],
) -> dict[str, Any]:
    data_dir = Path(cfg["data"]["output_dir"])
    arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
    if arrays.x0_estimates is None:
        raise ValueError(f"{scenario} is missing x0_estimates.")
    if arrays.ekf_prior is None or arrays.ukf_prior is None or arrays.aukf_prior is None:
        raise ValueError(f"{scenario} is missing recursive filter prior arrays.")

    dataset_cfg = scenario_dataset_config(cfg, scenario)
    n_traj = arrays.states.shape[0] if trajectory_limit <= 0 else min(trajectory_limit, arrays.states.shape[0])
    init_scale = np.array(
        [
            float(cfg["baselines"]["init_pos_std_m"]),
            float(cfg["baselines"]["init_pos_std_m"]),
            float(cfg["baselines"]["init_pos_std_m"]),
            float(cfg["baselines"]["init_vel_std_mps"]),
            float(cfg["baselines"]["init_vel_std_mps"]),
            float(cfg["baselines"]["init_vel_std_mps"]),
        ],
        dtype=np.float64,
    )
    process_scale = np.array(
        [
            float(process_pos_std_m),
            float(process_pos_std_m),
            float(process_pos_std_m),
            float(process_vel_std_mps),
            float(process_vel_std_mps),
            float(process_vel_std_mps),
        ],
        dtype=np.float64,
    )

    state = arrays.states[:n_traj]
    vis = arrays.visibility[:n_traj]

    # Resolve the requested warm starts against what this scenario can supply.
    # BatchWLS depends on a sidecar prediction file; if it is absent we drop it
    # cleanly and record the skip rather than failing the run.
    wls_preds = _load_wls_predictions(wls_dir, scenario, n_traj, state.shape)
    wls_available = wls_preds is not None
    warm_start_arrays: dict[str, np.ndarray] = {
        "EKF": arrays.ekf_prior[:n_traj],
        "UKF": arrays.ukf_prior[:n_traj],
        "AUKF": arrays.aukf_prior[:n_traj],
    }
    if wls_available:
        warm_start_arrays["BatchWLS"] = wls_preds
    requested_warm_starts = list(warm_starts)
    available_warm_starts = [w for w in requested_warm_starts if w in warm_start_arrays]
    skipped_warm_starts = [w for w in requested_warm_starts if w not in warm_start_arrays]
    if not available_warm_starts:
        # Only reachable if the sole requested start was BatchWLS and the
        # sidecar is missing. Degrade to AUKF (always present) so a long
        # multi-scenario run survives; the degradation is recorded in summary.
        available_warm_starts = ["AUKF"]
        if "AUKF" not in skipped_warm_starts:
            skipped_warm_starts = [w for w in skipped_warm_starts]
    print(
        f"{scenario}: warm starts requested={requested_warm_starts} "
        f"available={available_warm_starts} skipped={skipped_warm_starts}",
        flush=True,
    )

    preds = np.zeros_like(arrays.states[:n_traj])
    fit_rows: list[dict[str, Any]] = []
    multistart_rows: list[dict[str, Any]] = []
    payloads = [
        {
            "idx": i,
            "measurements": arrays.measurements[i],
            "visibility": arrays.visibility[i],
            "times_s": arrays.times[i],
            "x0_est": arrays.x0_estimates[i],
            "warm_starts": [
                {"name": name, "arc": warm_start_arrays[name][i]}
                for name in available_warm_starts
            ],
            "dataset_cfg": dataset_cfg,
            "meas_std": dataset_cfg.measurement_noise.std_vector,
            "init_scale": init_scale,
            "process_scale": process_scale,
            "max_nfev": max_nfev,
            "loss": loss,
            "huber_f_scale": huber_f_scale,
            "prior_weight": prior_weight,
            "anchor_weight": anchor_weight,
            "correction_bound": correction_bound,
        }
        for i in range(n_traj)
    ]
    completed = 0
    if workers <= 1:
        for payload in payloads:
            idx, pred, info, all_infos = _fit_single_worker(payload)
            preds[idx] = pred
            fit_rows.append(info)
            multistart_rows.extend(all_infos)
            completed += 1
            if completed % 4 == 0 or completed == n_traj:
                print(f"{scenario}: smoothed {completed}/{n_traj} trajectories", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(_fit_single_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                idx, pred, info, all_infos = future.result()
                preds[idx] = pred
                fit_rows.append(info)
                multistart_rows.extend(all_infos)
                completed += 1
                if completed % 4 == 0 or completed == n_traj:
                    print(f"{scenario}: smoothed {completed}/{n_traj} trajectories", flush=True)

    method_summaries: dict[str, dict[str, float]] = {
        "RFIS": _method_summary(state, preds, vis, eval_start),
        "EKF": _method_summary(state, arrays.ekf_prior[:n_traj], vis, eval_start),
        "UKF": _method_summary(state, arrays.ukf_prior[:n_traj], vis, eval_start),
        "AUKF": _method_summary(state, arrays.aukf_prior[:n_traj], vis, eval_start),
    }
    if wls_available:
        method_summaries["BatchWLS"] = _method_summary(state, wls_preds, vis, eval_start)

    recursive_methods = ["EKF", "UKF", "AUKF"]
    classical_methods = recursive_methods + (["BatchWLS"] if wls_available else [])
    smoother = method_summaries["RFIS"]

    def best_by(methods: list[str], metric: str) -> tuple[str, float]:
        name = min(methods, key=lambda m: method_summaries[m][metric])
        return name, float(method_summaries[name][metric])

    best_rec_obs, best_rec_obs_val = best_by(recursive_methods, "observed_step_pos_rmse_m")
    best_rec_all, best_rec_all_val = best_by(recursive_methods, "all_step_pos_rmse_m")
    best_cls_obs, best_cls_obs_val = best_by(classical_methods, "observed_step_pos_rmse_m")
    best_cls_all, best_cls_all_val = best_by(classical_methods, "all_step_pos_rmse_m")

    gain_rec_obs = _gain_percent(best_rec_obs_val, smoother["observed_step_pos_rmse_m"])
    gain_rec_all = _gain_percent(best_rec_all_val, smoother["all_step_pos_rmse_m"])
    gain_cls_obs = _gain_percent(best_cls_obs_val, smoother["observed_step_pos_rmse_m"])
    gain_cls_all = _gain_percent(best_cls_all_val, smoother["all_step_pos_rmse_m"])
    gain_aukf_obs = _gain_percent(
        method_summaries["AUKF"]["observed_step_pos_rmse_m"], smoother["observed_step_pos_rmse_m"]
    )
    gain_aukf_all = _gain_percent(
        method_summaries["AUKF"]["all_step_pos_rmse_m"], smoother["all_step_pos_rmse_m"]
    )

    improves_observed = bool(np.isfinite(gain_cls_obs) and gain_cls_obs > 0.0)
    improves_all_step = bool(np.isfinite(gain_cls_all) and gain_cls_all > 0.0)
    materially_improves_observed = bool(np.isfinite(gain_cls_obs) and gain_cls_obs >= 1.0)
    materially_improves_all_step = bool(np.isfinite(gain_cls_all) and gain_cls_all >= 1.0)
    if improves_observed and improves_all_step:
        verdict = "improves_both_observed_and_all_step_vs_best_classical"
    elif improves_observed:
        verdict = "improves_observed_step_only_vs_best_classical"
    elif improves_all_step:
        verdict = "improves_all_step_only_vs_best_classical"
    else:
        verdict = "no_improvement_vs_best_classical"

    outcomes = [str(row["outcome"]) for row in fit_rows]
    outcome_counts = {name: int(outcomes.count(name)) for name in sorted(set(outcomes))}
    usable_flags = [bool(row["usable"]) for row in fit_rows]
    obj_ratios = np.array(
        [float(row["objective_reduction_ratio"]) for row in fit_rows], dtype=np.float64
    )
    obj_ratios = obj_ratios[np.isfinite(obj_ratios)]
    selected_names = [str(row["selected_warm_start"]) for row in fit_rows]
    warm_start_selection_counts = {
        name: int(selected_names.count(name)) for name in sorted(set(selected_names))
    }

    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scenario_dir / "rfis_predictions.npz", rfis=preds)
    pd.DataFrame(fit_rows).sort_values("trajectory").to_csv(
        scenario_dir / "rfis_fit_diagnostics.csv", index=False
    )
    pd.DataFrame(multistart_rows).sort_values(["trajectory", "warm_start"]).to_csv(
        scenario_dir / "rfis_multistart_diagnostics.csv", index=False
    )
    summary = {
        "scenario": scenario,
        "trajectories": int(n_traj),
        "eval_start_step": int(eval_start),
        "max_nfev": int(max_nfev),
        "loss": loss,
        "huber_f_scale": float(huber_f_scale),
        "prior_weight": float(prior_weight),
        "anchor_weight": float(anchor_weight),
        "process_pos_std_m": float(process_pos_std_m),
        "process_vel_std_mps": float(process_vel_std_mps),
        "correction_bound": float(correction_bound),
        "mean_visible_measurements_per_traj": float(np.mean([row["num_measurements"] for row in fit_rows])),
        "fit_success_rate": float(np.mean(usable_flags)),
        "fit_converged_rate": float(np.mean([o == "converged" for o in outcomes])),
        "fit_max_nfev_improved_rate": float(np.mean([o == "max_nfev_improved" for o in outcomes])),
        "fit_unusable_rate": float(np.mean([not u for u in usable_flags])),
        "fit_diverged_rate": float(np.mean([o == "diverged" for o in outcomes])),
        "fit_outcome_counts": outcome_counts,
        "median_objective_reduction_ratio": float(np.median(obj_ratios)) if obj_ratios.size else float("nan"),
        "requested_warm_starts": requested_warm_starts,
        "available_warm_starts": available_warm_starts,
        "skipped_warm_starts": skipped_warm_starts,
        "multistart_enabled": bool(len(available_warm_starts) > 1),
        "warm_start_selection_counts": warm_start_selection_counts,
        "wls_available": bool(wls_available),
        "best_recursive_observed_method": best_rec_obs,
        "best_recursive_all_step_method": best_rec_all,
        "best_classical_observed_method": best_cls_obs,
        "best_classical_all_step_method": best_cls_all,
        "rfis_gain_vs_best_recursive_observed_percent": gain_rec_obs,
        "rfis_gain_vs_best_recursive_all_step_percent": gain_rec_all,
        "rfis_gain_vs_best_classical_observed_percent": gain_cls_obs,
        "rfis_gain_vs_best_classical_all_step_percent": gain_cls_all,
        "rfis_gain_vs_aukf_observed_percent": gain_aukf_obs,
        "rfis_gain_vs_aukf_all_step_percent": gain_aukf_all,
        "improves_observed_step": improves_observed,
        "improves_all_step": improves_all_step,
        "materially_improves_observed_step": materially_improves_observed,
        "materially_improves_all_step": materially_improves_all_step,
        "honest_verdict": verdict,
        "methods": method_summaries,
    }
    (scenario_dir / "rfis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"{scenario}: usable={summary['fit_success_rate']:.2f} "
        f"converged={summary['fit_converged_rate']:.2f} "
        f"verdict={verdict} "
        f"obs_gain_vs_classical={gain_cls_obs:.2f}% "
        f"all_gain_vs_classical={gain_cls_all:.2f}% "
        f"selected={warm_start_selection_counts}",
        flush=True,
    )
    return summary


def flatten_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scalar_keys = [
        "scenario",
        "trajectories",
        "eval_start_step",
        "mean_visible_measurements_per_traj",
        "fit_success_rate",
        "fit_converged_rate",
        "fit_max_nfev_improved_rate",
        "fit_unusable_rate",
        "fit_diverged_rate",
        "median_objective_reduction_ratio",
        "requested_warm_starts",
        "available_warm_starts",
        "skipped_warm_starts",
        "multistart_enabled",
        "warm_start_selection_counts",
        "wls_available",
        "best_recursive_observed_method",
        "best_recursive_all_step_method",
        "best_classical_observed_method",
        "best_classical_all_step_method",
        "rfis_gain_vs_best_recursive_observed_percent",
        "rfis_gain_vs_best_recursive_all_step_percent",
        "rfis_gain_vs_best_classical_observed_percent",
        "rfis_gain_vs_best_classical_all_step_percent",
        "rfis_gain_vs_aukf_observed_percent",
        "rfis_gain_vs_aukf_all_step_percent",
        "improves_observed_step",
        "improves_all_step",
        "materially_improves_observed_step",
        "materially_improves_all_step",
        "honest_verdict",
    ]
    def _flat(value: Any) -> Any:
        # Keep CSV/JSON cells compact and scalar: lists -> "a|b", dict ->
        # "k:v;k:v". No paths/env/code prose ever reaches these fields.
        if isinstance(value, (list, tuple)):
            return "|".join(str(v) for v in value)
        if isinstance(value, dict):
            return ";".join(f"{k}:{value[k]}" for k in value)
        return value

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        base = {key: _flat(summary[key]) for key in scalar_keys}
        for method, metrics in summary["methods"].items():
            prefix = method.lower()
            for key, value in metrics.items():
                base[f"{prefix}_{key}"] = value
        rows.append(base)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wls_dir = Path(args.wls_baseline_dir)
    warm_starts = _parse_warm_starts(args.warm_starts)
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    summaries = []
    for scenario in [name.strip() for name in args.scenarios.split(",") if name.strip()]:
        summaries.append(
            run_scenario(
                cfg,
                scenario,
                output_dir,
                eval_start=eval_start,
                max_nfev=args.max_nfev,
                loss=args.loss,
                huber_f_scale=args.huber_f_scale,
                prior_weight=args.prior_weight,
                anchor_weight=args.anchor_weight,
                process_pos_std_m=args.process_pos_std_m,
                process_vel_std_mps=args.process_vel_std_mps,
                correction_bound=args.correction_bound,
                trajectory_limit=args.trajectory_limit,
                workers=args.workers,
                wls_dir=wls_dir,
                warm_starts=warm_starts,
            )
        )
    rows = flatten_summary_rows(summaries)
    pd.DataFrame(rows).to_csv(output_dir / "rfis_summary.csv", index=False)
    (output_dir / "rfis_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": "rfis_summary.csv", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
