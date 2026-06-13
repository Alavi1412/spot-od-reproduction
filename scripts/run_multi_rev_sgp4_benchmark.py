#!/usr/bin/env python
"""Multi-revolution SGP4-truth state-error benchmark with real station geometry.

This benchmark is the operationally longer, model-mismatched counterpart to the
compact perfect-shared-model 40-minute synthetic suite. For each of several
distinct real public LEO targets it:

* propagates the archived public CelesTrak two-line element set with SGP4 to
  produce the *truth* state path over a multi-revolution arc (default 12 hours,
  i.e. several orbital revolutions per target);
* generates sparse line-of-sight measurements from those true SGP4 states using
  the real eight-station ground geometry of the main study;
* runs the same compact two-body+J2+drag(+third-body/SRP) recursive filters
  (EKF, UKF, tuned AUKF), an offline robust batch weighted-least-squares fit,
  and a hard-bounded learned residual correction of the robust UKF estimate
  -- every estimator uses the *compact* OD model, so SGP4 truth is a genuine
  multi-revolution dynamics-model mismatch rather than the perfect-shared-model
  synthetic split;
* scores every estimator against the SGP4 *truth state* (position RMSE) on a
  deterministic earlier-fit / later-held-out time split, pooled across targets.

SGP4 is an analytic mean-element propagation, not a precise numerically
integrated OD truth, so absolute magnitudes are not operational OD accuracy.
But it is a substantially stronger multi-revolution model-mismatch and
real-station-geometry stress test than the 40-minute perfect-model suite, and
unlike the real ILRS SLR range-residual probe it provides a genuine truth
*state* over operationally longer arcs. The Earth-rotation transform used to
form measurements is the same simplified constant-rate transform for the truth
measurement generation and for the estimators, so the evaluated stress is the
dynamics model mismatch over multiple revolutions, not a measurement-frame
discrepancy. This is a bounded multi-revolution model-mismatch benchmark, not
an operational orbit determination or a flight-readiness validation.

The benchmark is deterministic and fully offline: it reads only the archived
public TLE catalog already in the repository (no network access).
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from gnn_state_estimation.coordinates import (
    StationGeometry,
    line_of_sight_measurement,
)
from gnn_state_estimation.dynamics import propagate_orbit, rk4_step
from gnn_state_estimation.filters.ekf import EKFConfig, run_ekf, wrap_angle_pi
from gnn_state_estimation.filters.ukf import (
    AdaptiveUKFConfig,
    UKFConfig,
    run_adaptive_ukf,
    run_ukf,
)
from gnn_state_estimation.semireal import (
    filter_tle_catalog,
    load_tle_catalog,
)
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso
from sgp4.api import Satrec


@dataclass(frozen=True)
class CompactModel:
    """Compact estimator-side force model (shared by every estimator)."""

    ballistic_coeff_m2_per_kg: float
    drag_rho_ref: float
    drag_h_ref_m: float
    drag_scale_height_m: float
    enable_third_body: bool
    enable_srp: bool
    srp_area_to_mass_m2_per_kg: float
    srp_cr: float
    sun_initial_phase_rad: float
    moon_initial_phase_rad: float

    def kwargs(self) -> dict[str, Any]:
        return {
            "ballistic_coeff_m2_per_kg": self.ballistic_coeff_m2_per_kg,
            "drag_rho_ref": self.drag_rho_ref,
            "drag_h_ref_m": self.drag_h_ref_m,
            "drag_scale_height_m": self.drag_scale_height_m,
            "enable_third_body": self.enable_third_body,
            "enable_srp": self.enable_srp,
            "srp_area_to_mass_m2_per_kg": self.srp_area_to_mass_m2_per_kg,
            "srp_cr": self.srp_cr,
            "sun_initial_phase_rad": self.sun_initial_phase_rad,
            "moon_initial_phase_rad": self.moon_initial_phase_rad,
        }


ESTIMATOR_KINDS = {
    "EKF (compact)": "classical recursive",
    "UKF (compact)": "classical recursive",
    "AUKF (compact, tuned)": "classical recursive",
    "Batch WLS (compact)": "classical offline",
    "Learned residual (UKF prior)": "learned",
}


def _stations_from_cfg(cfg: dict[str, Any]) -> tuple[StationGeometry, ...]:
    return tuple(
        StationGeometry(
            name=str(s["name"]),
            lat_deg=float(s["lat_deg"]),
            lon_deg=float(s["lon_deg"]),
            alt_m=float(s["alt_m"]),
            min_elevation_deg=float(s.get("min_elevation_deg", 8.0)),
        )
        for s in cfg["simulation"]["stations"]
    )


def _compact_model_from_cfg(cfg: dict[str, Any]) -> CompactModel:
    dyn = cfg["simulation"]["dynamics"]
    return CompactModel(
        ballistic_coeff_m2_per_kg=float(dyn["ballistic_coeff_m2_per_kg"]),
        drag_rho_ref=float(dyn["drag_rho_ref"]),
        drag_h_ref_m=float(dyn["drag_h_ref_m"]),
        drag_scale_height_m=float(dyn["drag_scale_height_m"]),
        enable_third_body=bool(dyn["enable_third_body"]),
        enable_srp=bool(dyn["enable_srp"]),
        srp_area_to_mass_m2_per_kg=float(dyn["srp_area_to_mass_m2_per_kg"]),
        srp_cr=float(dyn["srp_cr"]),
        sun_initial_phase_rad=float(dyn["sun_initial_phase_rad"]),
        moon_initial_phase_rad=float(dyn["moon_initial_phase_rad"]),
    )


def _sgp4_truth_states(sat: Satrec, times_s: np.ndarray) -> np.ndarray:
    """SGP4 TEME truth states (m, m/s) on the relative-time grid."""
    jd = np.full(times_s.shape, sat.jdsatepoch, dtype=np.float64)
    fr = np.full(times_s.shape, sat.jdsatepochF, dtype=np.float64) + times_s / 86400.0
    err, r_km, v_kmps = sat.sgp4_array(jd, fr)
    if np.any(err != 0):
        raise RuntimeError(f"SGP4 propagation failed: errors={np.unique(err)}")
    return np.hstack([r_km * 1e3, v_kmps * 1e3]).astype(np.float64)


def _generate_measurements(
    truth_states: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    noise_std: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse line-of-sight measurements generated from SGP4 truth states."""
    steps = truth_states.shape[0]
    n_st = len(stations)
    meas = np.zeros((steps, n_st, 4), dtype=np.float64)
    vis = np.zeros((steps, n_st), dtype=np.float64)
    for t in range(steps):
        for s_idx, station in enumerate(stations):
            z_true, visible = line_of_sight_measurement(
                truth_states[t], station, float(times_s[t])
            )
            if not visible:
                continue
            eps = rng.normal(0.0, noise_std)
            z = z_true + eps
            if z[1] < 0.0:
                z[1] += 2.0 * np.pi
            elif z[1] >= 2.0 * np.pi:
                z[1] -= 2.0 * np.pi
            z[2] = float(np.clip(z[2], -0.5 * np.pi, 0.5 * np.pi))
            meas[t, s_idx] = z
            vis[t, s_idx] = 1.0
    return meas, vis


def _pos_rmse(truth: np.ndarray, est: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    err = truth[mask, :3] - est[mask, :3]
    if not np.all(np.isfinite(err)):
        return float("inf")
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def _fit_batch_wls(
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    x0_est: np.ndarray,
    model: CompactModel,
    noise_std: np.ndarray,
    fit_steps: np.ndarray,
    *,
    max_nfev: int,
) -> np.ndarray:
    """Offline robust WLS: fit x0 to fit-window visible measurements only."""
    vis_pairs = [
        (int(t), int(s))
        for t in fit_steps
        for s in range(visibility.shape[1])
        if visibility[t, s] >= 0.5
    ]
    if len(vis_pairs) < 6:
        return propagate_orbit(
            x0_est, dt=float(times_s[1] - times_s[0]),
            steps=times_s.size, process_noise_std=0.0, **model.kwargs()
        )
    scale = np.array([2500.0, 2500.0, 2500.0, 6.0, 6.0, 6.0], dtype=np.float64)
    uniq_t = sorted({t for t, _ in vis_pairs})
    t_index = {t: i for i, t in enumerate(uniq_t)}
    rel = np.array([times_s[t] for t in uniq_t], dtype=np.float64)
    dt0 = float(times_s[1] - times_s[0])
    # The offline WLS fit objective integrates the compact model on a slightly
    # coarser substep than the measurement cadence; this keeps the postfit OD
    # reference tractable over multi-revolution arcs without changing the
    # estimator-side force model. Final scoring still uses the dt0 grid.
    wls_step = max(dt0, 60.0)

    def _propagate_to(state0: np.ndarray) -> np.ndarray:
        out = np.zeros((len(uniq_t), 6), dtype=np.float64)
        state = np.asarray(state0, dtype=np.float64).copy()
        t_cur = 0.0
        for i, target_t in enumerate(rel):
            span = float(target_t) - t_cur
            if span > 0.0:
                n_sub = max(1, int(np.ceil(span / wls_step)))
                sub = span / n_sub
                for _ in range(n_sub):
                    state = rk4_step(state, dt=sub, t_s=t_cur, **model.kwargs())
                    t_cur += sub
            out[i] = state
        return out

    def residual_fn(u: np.ndarray) -> np.ndarray:
        cand = x0_est + u * scale
        states = _propagate_to(cand)
        res = [u]
        for t, s in vis_pairs:
            z_pred, _ = line_of_sight_measurement(
                states[t_index[t]], _STATIONS[s], float(times_s[t])
            )
            y = np.asarray(measurements[t, s], dtype=np.float64) - z_pred
            y[1] = wrap_angle_pi(float(y[1]))
            res.append(y / noise_std)
        out = np.concatenate(res)
        return np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=-1e6)

    try:
        sol = least_squares(
            residual_fn, np.zeros(6), method="trf", loss="huber",
            f_scale=2.5, x_scale=np.ones(6), max_nfev=int(max_nfev),
        )
        fitted = x0_est + sol.x * scale
    except Exception:  # pragma: no cover - numerical safety net
        fitted = np.asarray(x0_est, dtype=np.float64).copy()
    return propagate_orbit(
        fitted, dt=dt0, steps=times_s.size, process_noise_std=0.0,
        **model.kwargs()
    )


# Hard absolute per-axis bound on the learned position correction (metres).
# Fixed (not scaled by the fit-window error) so a learned temporal
# extrapolation can never blow up the held-out comparison, mirroring the
# bounded-and-deterministic learned correction used in the real SLR audit.
LEARNED_CORR_CAP_M = 8000.0
# Training targets are clipped to this robust range so the small bounded MLP
# does not chase divergent fit-window transients of the underlying filter.
LEARNED_TARGET_CLIP_M = 50_000.0


def _learned_residual_estimate(
    base_est: np.ndarray,
    truth_states: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    period_s: float,
    fit_steps: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Hard-bounded learned residual correction of a recursive-filter prior.

    The model is trained only on the earlier fit window (supervised on the
    base filter's position error against SGP4 truth) and applied to every
    step, so the later held-out window is a genuine temporal-extrapolation
    test. Features are leak-free (time-since-start, orbital phase,
    visible-station count, prior position/speed magnitude). The per-axis
    correction is hard-bounded to a fixed absolute cap and the base prior is
    finite-guarded, so a learned extrapolation can never blow up the held-out
    comparison; this is a deterministic bounded residual estimator, not an
    unconstrained regressor.
    """
    span = float(times_s[-1] - times_s[0]) or 1.0
    rel = (times_s - times_s[0]) / span
    phase = 2.0 * np.pi * (times_s - times_s[0]) / (period_s if period_s > 0 else 1.0)
    vis_count = visibility.sum(axis=1)
    safe_base = np.where(np.isfinite(base_est), base_est, 0.0)
    pos_mag = np.linalg.norm(safe_base[:, :3], axis=1) / 7.0e6
    spd_mag = np.linalg.norm(safe_base[:, 3:], axis=1) / 8.0e3
    feats = np.column_stack([
        rel, np.sin(phase), np.cos(phase),
        np.clip(vis_count, 0.0, 8.0) / 8.0,
        (vis_count > 0.5).astype(np.float64),
        np.clip(pos_mag, 0.0, 5.0), np.clip(spd_mag, 0.0, 5.0),
    ]).astype(np.float64)
    target = np.clip(
        truth_states[:, :3] - safe_base[:, :3],
        -LEARNED_TARGET_CLIP_M, LEARNED_TARGET_CLIP_M,
    )

    tr = np.asarray(fit_steps, dtype=int)
    finite_base = np.all(np.isfinite(base_est[:, :3]), axis=1)
    tr = tr[finite_base[tr]] if tr.size else tr
    out = base_est.copy()
    if tr.size < 8:
        return out, "skipped"
    mean = feats[tr].mean(axis=0)
    std = feats[tr].std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    x_tr = (feats[tr] - mean) / std
    x_all = (feats - mean) / std
    y_tr = target[tr]

    corr, backend = _fit_residual_torch(
        x_tr, y_tr, x_all, seed=seed, bound=LEARNED_CORR_CAP_M
    )
    if corr is None:
        corr, backend = _fit_residual_ridge(
            x_tr, y_tr, x_all, bound=LEARNED_CORR_CAP_M
        )
    corr = np.clip(corr, -LEARNED_CORR_CAP_M, LEARNED_CORR_CAP_M)
    # Apply the bounded correction only where the base prior is finite; where
    # the base filter has diverged the learned row inherits that divergence
    # honestly rather than masking it with an unbounded correction.
    out[finite_base, :3] = base_est[finite_base, :3] + corr[finite_base]
    return out, backend


def _fit_residual_torch(x_tr, y_tr, x_all, *, seed, bound):
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - optional backend
        return None, "none"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_scale = float(np.sqrt(np.mean(y_tr ** 2))) or 1.0
    xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr / y_scale, dtype=torch.float32, device=device)
    xa = torch.tensor(x_all, dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_tr.shape[1], 48),
        torch.nn.Tanh(),
        torch.nn.Linear(48, 48),
        torch.nn.Tanh(),
        torch.nn.Linear(48, 3),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()
    model.train()
    for _ in range(500):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        raw = model(xa).cpu().numpy() * y_scale
    corr = bound * np.tanh(raw / bound)
    return corr.astype(np.float64), f"torch:{device.type}"


def _fit_residual_ridge(x_tr, y_tr, x_all, *, bound, alpha=1.0):
    xb_tr = np.column_stack([x_tr, np.ones(x_tr.shape[0])])
    xb_all = np.column_stack([x_all, np.ones(x_all.shape[0])])
    reg = alpha * np.eye(xb_tr.shape[1])
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(xb_tr.T @ xb_tr + reg, xb_tr.T @ y_tr)
    raw = xb_all @ beta
    corr = bound * np.tanh(raw / bound)
    return corr.astype(np.float64), "ridge"


# Module-level station handle so the WLS residual closure can resolve geometry.
_STATIONS: tuple[StationGeometry, ...] = ()


def run_target(
    name: str,
    sat: Satrec,
    catalog_sha256: str,
    cfg: dict[str, Any],
    model: CompactModel,
    *,
    arc_hours: float,
    dt_s: float,
    train_frac: float,
    max_nfev: int,
    seed: int,
) -> dict[str, Any]:
    steps = int(round(arc_hours * 3600.0 / dt_s))
    times_s = np.arange(steps, dtype=np.float64) * dt_s
    truth = _sgp4_truth_states(sat, times_s)

    mm_rev_per_day = float(sat.no_kozai * 1440.0 / (2.0 * np.pi))
    period_s = 86400.0 / mm_rev_per_day if mm_rev_per_day > 0 else float("nan")
    revolutions = float(arc_hours * 3600.0 / period_s) if period_s > 0 else float("nan")

    stations = _STATIONS
    mn = cfg["simulation"]["measurement_noise"]
    noise_std = np.array(
        [
            float(mn["range_std_m"]),
            np.deg2rad(float(mn["az_std_deg"])),
            np.deg2rad(float(mn["el_std_deg"])),
            float(mn["range_rate_std_mps"]),
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed + abs(hash(name)) % 100_000)
    meas, vis = _generate_measurements(truth, times_s, stations, noise_std, rng)

    x0_est = truth[0].copy()
    x0_est[:3] += rng.normal(0.0, float(cfg["baselines"]["init_pos_std_m"]), size=3)
    x0_est[3:] += rng.normal(0.0, float(cfg["baselines"]["init_vel_std_mps"]), size=3)

    bl = cfg["baselines"]
    ekf_cfg = EKFConfig(
        q_pos_m=float(bl["ekf"]["q_pos_m"]), q_vel_mps=float(bl["ekf"]["q_vel_mps"]),
        init_pos_std_m=float(bl["ekf"]["init_pos_std_m"]),
        init_vel_std_mps=float(bl["ekf"]["init_vel_std_mps"]),
        gating_threshold=float(bl["ekf"]["gating_threshold"]),
    )
    ukf_cfg = UKFConfig(
        q_pos_m=float(bl["ukf"]["q_pos_m"]), q_vel_mps=float(bl["ukf"]["q_vel_mps"]),
        init_pos_std_m=float(bl["ukf"]["init_pos_std_m"]),
        init_vel_std_mps=float(bl["ukf"]["init_vel_std_mps"]),
        alpha=float(bl["ukf"]["alpha"]), beta=float(bl["ukf"]["beta"]),
        kappa=float(bl["ukf"]["kappa"]),
    )
    aukf_cfg = AdaptiveUKFConfig(
        q_pos_m=float(bl["aukf"]["q_pos_m"]), q_vel_mps=float(bl["aukf"]["q_vel_mps"]),
        init_pos_std_m=float(bl["aukf"]["init_pos_std_m"]),
        init_vel_std_mps=float(bl["aukf"]["init_vel_std_mps"]),
        alpha=float(bl["aukf"]["alpha"]), beta=float(bl["aukf"]["beta"]),
        kappa=float(bl["aukf"]["kappa"]), adapt_rate=float(bl["aukf"]["adapt_rate"]),
        min_r_scale=float(bl["aukf"]["min_r_scale"]),
        max_r_scale=float(bl["aukf"]["max_r_scale"]),
        huber_kappa=float(bl["aukf"]["huber_kappa"]),
        nis_soft_gate=float(bl["aukf"]["nis_soft_gate"]),
    )
    fkw = dict(
        measurements=meas, visibility=vis, times_s=times_s, stations=stations,
        ballistic_coeff_m2_per_kg=model.ballistic_coeff_m2_per_kg,
        meas_std_vector=noise_std, x0_est=x0_est,
        drag_rho_ref=model.drag_rho_ref, drag_h_ref_m=model.drag_h_ref_m,
        drag_scale_height_m=model.drag_scale_height_m,
        enable_third_body=model.enable_third_body, enable_srp=model.enable_srp,
        srp_area_to_mass_m2_per_kg=model.srp_area_to_mass_m2_per_kg,
        srp_cr=model.srp_cr, sun_initial_phase_rad=model.sun_initial_phase_rad,
        moon_initial_phase_rad=model.moon_initial_phase_rad,
    )
    ekf_est, _ = run_ekf(cfg=ekf_cfg, **fkw)
    ukf_est, _ = run_ukf(cfg=ukf_cfg, **fkw)
    aukf_est, _ = run_adaptive_ukf(cfg=aukf_cfg, **fkw)

    n_train = max(6, int(np.floor(train_frac * steps)))
    n_train = min(n_train, steps - 1)
    fit_steps = np.arange(0, n_train)
    held_steps = np.arange(n_train, steps)
    held_mask = np.zeros(steps, dtype=bool)
    held_mask[held_steps] = True
    fit_mask = np.zeros(steps, dtype=bool)
    fit_mask[fit_steps] = True
    observed = vis.sum(axis=1) >= 0.5
    held_obs_mask = held_mask & observed

    wls_est = _fit_batch_wls(
        meas, vis, times_s, x0_est, model, noise_std, fit_steps,
        max_nfev=max_nfev,
    )
    learned_est, learned_backend = _learned_residual_estimate(
        ukf_est, truth, vis, times_s, period_s, fit_steps, seed=seed,
    )

    est_by_name = {
        "EKF (compact)": ekf_est,
        "UKF (compact)": ukf_est,
        "AUKF (compact, tuned)": aukf_est,
        "Batch WLS (compact)": wls_est,
        "Learned residual (UKF prior)": learned_est,
    }
    estimators: dict[str, dict[str, float]] = {}
    held_err_arrays: dict[str, np.ndarray] = {}
    for nm, est in est_by_name.items():
        estimators[nm] = {
            "fit_pos_rmse_m": _pos_rmse(truth, est, fit_mask),
            "held_out_pos_rmse_m": _pos_rmse(truth, est, held_mask),
            "held_out_observed_pos_rmse_m": _pos_rmse(truth, est, held_obs_mask),
        }
        e = truth[held_mask, :3] - est[held_mask, :3]
        held_err_arrays[nm] = np.sqrt(np.sum(e * e, axis=-1))

    return {
        "target": name,
        "norad_catalog_id": int(sat.satnum),
        "tle_catalog_sha256": catalog_sha256,
        "mean_motion_rev_per_day": mm_rev_per_day,
        "orbital_period_min": period_s / 60.0,
        "arc_hours": arc_hours,
        "revolutions": revolutions,
        "steps": steps,
        "num_train_steps": int(n_train),
        "num_held_out_steps": int(steps - n_train),
        "num_visible_measurements": int(vis.sum()),
        "observed_step_fraction": float(np.mean(observed)),
        "altitude_km": float(np.linalg.norm(truth[0, :3]) - 6378136.3) / 1e3,
        "inclination_deg": float(np.rad2deg(sat.inclo)),
        "estimators": estimators,
        "_held_err": held_err_arrays,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument(
        "--tle-catalog", type=str, default="configs/archived_tles.json"
    )
    p.add_argument(
        "--out-dir", type=str, default="results/multi_rev_sgp4"
    )
    p.add_argument("--arc-hours", type=float, default=12.0)
    p.add_argument("--dt-s", type=float, default=30.0)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--max-nfev", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-targets", type=int, default=6)
    return p


def main() -> None:
    global _STATIONS
    import yaml

    args = build_parser().parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _STATIONS = _stations_from_cfg(cfg)
    model = _compact_model_from_cfg(cfg)

    catalog_path = Path(args.tle_catalog)
    catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    catalog = load_tle_catalog(catalog_path)
    # Restrict to genuine multi-revolution LEO targets so each arc spans
    # several real orbital revolutions of a distinct public object.
    leo = filter_tle_catalog(
        catalog,
        min_altitude_km=200.0,
        max_altitude_km=2000.0,
        max_eccentricity=0.05,
        min_mean_motion_rev_per_day=11.0,
    )[: args.max_targets]
    if not leo:
        raise SystemExit("No multi-revolution LEO targets after filtering.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"multi-rev SGP4 benchmark: {len(leo)} targets, "
        f"arc {args.arc_hours} h, dt {args.dt_s} s",
        flush=True,
    )
    per_target: list[dict[str, Any]] = []
    for entry in leo:
        sat = Satrec.twoline2rv(entry.line1, entry.line2)
        block = run_target(
            entry.name, sat, catalog_sha256, cfg, model,
            arc_hours=args.arc_hours, dt_s=args.dt_s,
            train_frac=args.train_frac, max_nfev=args.max_nfev,
            seed=args.seed,
        )
        per_target.append(block)
        print(
            f"  done {entry.name}: {block['revolutions']:.1f} revs, "
            f"held-out EKF {block['estimators']['EKF (compact)']['held_out_pos_rmse_m']:.1f} m, "
            f"AUKF {block['estimators']['AUKF (compact, tuned)']['held_out_pos_rmse_m']:.1f} m, "
            f"WLS {block['estimators']['Batch WLS (compact)']['held_out_pos_rmse_m']:.1f} m, "
            f"learned {block['estimators']['Learned residual (UKF prior)']['held_out_pos_rmse_m']:.1f} m",
            flush=True,
        )

    # Pool held-out per-step position errors across targets per estimator.
    summary: list[dict[str, Any]] = []
    best_count = {nm: 0 for nm in ESTIMATOR_KINDS}
    for blk in per_target:
        finite = {
            nm: v["held_out_pos_rmse_m"]
            for nm, v in blk["estimators"].items()
            if np.isfinite(v["held_out_pos_rmse_m"])
        }
        if finite:
            best_count[min(finite, key=lambda k: finite[k])] += 1

    for nm, kind in ESTIMATOR_KINDS.items():
        held = np.concatenate(
            [b["_held_err"][nm] for b in per_target if nm in b["_held_err"]]
        )
        held = held[np.isfinite(held)]
        fit_vals = [
            b["estimators"][nm]["fit_pos_rmse_m"] for b in per_target
            if np.isfinite(b["estimators"][nm]["fit_pos_rmse_m"])
        ]
        obs_vals = [
            b["estimators"][nm]["held_out_observed_pos_rmse_m"]
            for b in per_target
            if np.isfinite(b["estimators"][nm]["held_out_observed_pos_rmse_m"])
        ]
        summary.append({
            "name": nm,
            "kind": kind,
            "pooled_fit_pos_rmse_m": float(np.mean(fit_vals)) if fit_vals else float("nan"),
            "pooled_held_out_pos_rmse_m": float(
                np.sqrt(np.mean(held ** 2))) if held.size else float("nan"),
            "pooled_held_out_observed_pos_rmse_m": float(
                np.mean(obs_vals)) if obs_vals else float("nan"),
            "pooled_held_out_p95_pos_err_m": float(
                np.percentile(held, 95.0)) if held.size else float("nan"),
            "targets_best_of": best_count[nm],
        })

    finite_summary = [
        e for e in summary if np.isfinite(e["pooled_held_out_pos_rmse_m"])
    ]
    best_estimator = (
        min(finite_summary, key=lambda e: e["pooled_held_out_pos_rmse_m"])["name"]
        if finite_summary else None
    )

    def _hr(nm: str) -> float:
        for e in summary:
            if e["name"] == nm:
                return e["pooled_held_out_pos_rmse_m"]
        return float("nan")

    learned_rms = _hr("Learned residual (UKF prior)")
    best_classical = min(
        (_hr(n) for n in (
            "EKF (compact)", "UKF (compact)", "AUKF (compact, tuned)",
            "Batch WLS (compact)")),
        default=float("nan"),
    )
    learned_beats_best_classical = bool(
        np.isfinite(learned_rms) and np.isfinite(best_classical)
        and learned_rms < best_classical
    )

    revs = [b["revolutions"] for b in per_target if np.isfinite(b["revolutions"])]
    serial_targets = [
        {k: v for k, v in b.items() if k != "_held_err"} for b in per_target
    ]
    caveats = (
        "Bounded multi-revolution model-mismatch benchmark. Truth is an "
        "analytic SGP4 mean-element propagation of an archived public CelesTrak "
        "TLE, not a precise numerically integrated OD truth, so absolute "
        "magnitudes are not operational OD accuracy. Every estimator uses the "
        "compact two-body+J2+drag(+third-body/SRP) OD model, so SGP4 truth is a "
        "genuine multi-revolution dynamics-model mismatch over operationally "
        "longer arcs with the real eight-station ground geometry of the main "
        "study. The Earth-rotation transform used to form measurements is the "
        "same simplified constant-rate transform for the truth measurement "
        "generation and for the estimators, so the evaluated stress is the "
        "dynamics model mismatch over multiple revolutions, not a "
        "measurement-frame discrepancy. Position RMSE is scored against the "
        "SGP4 truth state on a deterministic earlier-fit / later-held-out time "
        "split, pooled across targets. Not an operational orbit determination "
        "or a flight-readiness validation."
    )
    result = {
        "schema_version": "sgp4_truth_multirev_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed" if per_target else "no_targets",
        "arc_hours": float(args.arc_hours),
        "dt_s": float(args.dt_s),
        "steps": int(round(args.arc_hours * 3600.0 / args.dt_s)),
        "train_frac": float(args.train_frac),
        "targets": [b["target"] for b in per_target],
        "num_targets": len(per_target),
        "min_revolutions": float(np.min(revs)) if revs else float("nan"),
        "max_revolutions": float(np.max(revs)) if revs else float("nan"),
        "station_subset": [s.name for s in _STATIONS],
        "num_stations": len(_STATIONS),
        "tle_catalog": str(catalog_path),
        "tle_catalog_sha256": catalog_sha256,
        "num_observations_total": int(
            sum(b["num_visible_measurements"] for b in per_target)),
        "num_held_out_steps_total": int(
            sum(b["num_held_out_steps"] for b in per_target)),
        "best_held_out_estimator": best_estimator,
        "learned_beats_best_classical": learned_beats_best_classical,
        "estimators_summary": summary,
        "per_target": serial_targets,
        "input_digests": [
            {
                "kind": "archived_tle_catalog",
                "path": str(catalog_path),
                "sha256": catalog_sha256,
                "bytes": catalog_path.stat().st_size,
            }
        ],
        "caveats": caveats,
    }
    dump_json(result, out_dir / "multi_rev_sgp4_benchmark.json")
    print(json.dumps({
        "status": result["status"],
        "num_targets": result["num_targets"],
        "min_revolutions": result["min_revolutions"],
        "max_revolutions": result["max_revolutions"],
        "num_observations_total": result["num_observations_total"],
        "best_held_out_estimator": best_estimator,
        "learned_beats_best_classical": learned_beats_best_classical,
        "estimators_summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
