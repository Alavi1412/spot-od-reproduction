"""Evaluation for EKF/UKF/GNN estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .dataset import STATE_SCALE, compute_prior_bank_stats, scale_measurements, scale_state, unscale_state
from .filters.ekf import EKFConfig, run_ekf
from .filters.ukf import AdaptiveUKFConfig, UKFConfig, run_adaptive_ukf, run_ukf
from .innovation import compute_innovation_features
from .models.graph_estimator import TemporalGraphEstimator
from .observability import compute_observability_context_features, stations_from_ecef
from .simulation import DatasetConfig
from .utils.metrics import median_absolute_error, position_velocity_rmse


@dataclass(frozen=True)
class BaselineConfig:
    ekf: EKFConfig
    ukf: UKFConfig
    aukf: AdaptiveUKFConfig | None
    init_pos_std_m: float
    init_vel_std_mps: float


def parse_baseline_config(cfg: dict[str, Any]) -> BaselineConfig:
    aukf_cfg = cfg.get("aukf")
    return BaselineConfig(
        ekf=EKFConfig(**cfg["ekf"]),
        ukf=UKFConfig(**cfg["ukf"]),
        aukf=AdaptiveUKFConfig(**aukf_cfg) if isinstance(aukf_cfg, dict) else None,
        init_pos_std_m=float(cfg["init_pos_std_m"]),
        init_vel_std_mps=float(cfg["init_vel_std_mps"]),
    )


def generate_noisy_init(x0_true: np.ndarray, rng: np.random.Generator, pos_std: float, vel_std: float) -> np.ndarray:
    noise = np.hstack([rng.normal(0.0, pos_std, size=3), rng.normal(0.0, vel_std, size=3)])
    return x0_true + noise


def run_filter_baselines(
    states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times: np.ndarray,
    dataset_cfg: DatasetConfig,
    baseline_cfg: BaselineConfig,
    seed: int,
    x0_estimates: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_traj = states.shape[0]
    ekf_preds = np.zeros_like(states)
    ukf_preds = np.zeros_like(states)
    aukf_preds = np.zeros_like(states) if baseline_cfg.aukf is not None else None

    meas_std = dataset_cfg.measurement_noise.std_vector
    dyn = dataset_cfg.dynamics
    for i in range(n_traj):
        if x0_estimates is None:
            x0_est = generate_noisy_init(
                states[i, 0], rng, baseline_cfg.init_pos_std_m, baseline_cfg.init_vel_std_mps
            )
        else:
            x0_est = x0_estimates[i]
        ekf_x, _ = run_ekf(
            measurements=measurements[i],
            visibility=visibility[i],
            times_s=times[i],
            stations=dataset_cfg.stations,
            ballistic_coeff_m2_per_kg=dataset_cfg.dynamics.ballistic_coeff_m2_per_kg,
            meas_std_vector=meas_std,
            x0_est=x0_est,
            cfg=baseline_cfg.ekf,
            drag_rho_ref=dataset_cfg.dynamics.drag_rho_ref,
            drag_h_ref_m=dataset_cfg.dynamics.drag_h_ref_m,
            drag_scale_height_m=dataset_cfg.dynamics.drag_scale_height_m,
            enable_third_body=dyn.enable_third_body,
            enable_srp=dyn.enable_srp,
            srp_area_to_mass_m2_per_kg=dyn.srp_area_to_mass_m2_per_kg,
            srp_cr=dyn.srp_cr,
            sun_initial_phase_rad=dyn.sun_initial_phase_rad,
            moon_initial_phase_rad=dyn.moon_initial_phase_rad,
        )
        ukf_x, _ = run_ukf(
            measurements=measurements[i],
            visibility=visibility[i],
            times_s=times[i],
            stations=dataset_cfg.stations,
            ballistic_coeff_m2_per_kg=dataset_cfg.dynamics.ballistic_coeff_m2_per_kg,
            meas_std_vector=meas_std,
            x0_est=x0_est,
            cfg=baseline_cfg.ukf,
            drag_rho_ref=dataset_cfg.dynamics.drag_rho_ref,
            drag_h_ref_m=dataset_cfg.dynamics.drag_h_ref_m,
            drag_scale_height_m=dataset_cfg.dynamics.drag_scale_height_m,
            enable_third_body=dyn.enable_third_body,
            enable_srp=dyn.enable_srp,
            srp_area_to_mass_m2_per_kg=dyn.srp_area_to_mass_m2_per_kg,
            srp_cr=dyn.srp_cr,
            sun_initial_phase_rad=dyn.sun_initial_phase_rad,
            moon_initial_phase_rad=dyn.moon_initial_phase_rad,
        )
        ekf_preds[i] = ekf_x
        ukf_preds[i] = ukf_x
        if baseline_cfg.aukf is not None and aukf_preds is not None:
            aukf_x, _ = run_adaptive_ukf(
                measurements=measurements[i],
                visibility=visibility[i],
                times_s=times[i],
                stations=dataset_cfg.stations,
                ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
                meas_std_vector=meas_std,
                x0_est=x0_est,
                cfg=baseline_cfg.aukf,
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
            aukf_preds[i] = aukf_x
    out = {"ekf": ekf_preds, "ukf": ukf_preds}
    if aukf_preds is not None:
        out["aukf"] = aukf_preds
    return out


@torch.no_grad()
def run_model_inference(
    model: TemporalGraphEstimator,
    states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    station_ecef: np.ndarray,
    window_size: int,
    ekf_prior: np.ndarray | None = None,
    ukf_prior: np.ndarray | None = None,
    aukf_prior: np.ndarray | None = None,
    secondary_prior: np.ndarray | None = None,
    innovation_features: np.ndarray | None = None,
    prior_bank_stats: np.ndarray | None = None,
    return_logvar: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    n_traj, t_steps = states.shape[:2]
    preds = np.full_like(states, np.nan, dtype=np.float64)
    logvars = np.full_like(states, np.nan, dtype=np.float64)

    station = (station_ecef / 6_378_137.0).astype(np.float32)
    station_tensor = torch.from_numpy(station).to(device)
    log_scale = np.log(STATE_SCALE.astype(np.float64))

    for i in range(n_traj):
        for t in range(window_size - 1, t_steps):
            t0 = t - window_size + 1
            meas = scale_measurements(measurements[i, t0 : t + 1].astype(np.float32))
            vis = visibility[i, t0 : t + 1].astype(np.float32)[..., None]
            meas_t = torch.from_numpy(meas).unsqueeze(0).to(device)
            vis_t = torch.from_numpy(vis).unsqueeze(0).to(device)
            prior_t = None
            if model.use_ekf_prior:
                if ekf_prior is None:
                    raise ValueError("Model requires EKF prior but none provided.")
                prior_scaled = scale_state(ekf_prior[i, t].astype(np.float32))
                prior_t = torch.from_numpy(prior_scaled).unsqueeze(0).to(device)
            secondary_prior_t = None
            if model.use_dual_prior_fusion:
                if secondary_prior is None:
                    raise ValueError("Model requires secondary prior but none was provided.")
                secondary_prior_scaled = scale_state(secondary_prior[i, t].astype(np.float32))
                secondary_prior_t = torch.from_numpy(secondary_prior_scaled).unsqueeze(0).to(device)
            innov_t = None
            if model.use_innovation_features:
                if innovation_features is None:
                    raise ValueError("Model requires innovation features but none were provided.")
                innov_t = torch.from_numpy(innovation_features[i, t0 : t + 1].astype(np.float32)).unsqueeze(0).to(device)
            prior_bank_t = None
            prior_stats_t = None
            if getattr(model, "use_prior_bank_fusion", False):
                if ekf_prior is None or ukf_prior is None or aukf_prior is None:
                    raise ValueError("Prior-bank fusion requires EKF, UKF, and AUKF priors.")
                prior_bank_scaled = np.stack(
                    [
                        scale_state(ekf_prior[i, t].astype(np.float32)),
                        scale_state(ukf_prior[i, t].astype(np.float32)),
                        scale_state(aukf_prior[i, t].astype(np.float32)),
                    ],
                    axis=0,
                )
                prior_bank_t = torch.from_numpy(prior_bank_scaled).unsqueeze(0).to(device)
                if getattr(model, "prior_stats_dim", 0) > 0:
                    if prior_bank_stats is None:
                        raise ValueError("prior_bank_stats are required for prior-bank fusion.")
                    prior_stats_t = torch.from_numpy(prior_bank_stats[i, t].astype(np.float32)).unsqueeze(0).to(device)

            out = model(
                measurements=meas_t,
                visibility=vis_t,
                station_xyz=station_tensor,
                ekf_prior=prior_t,
                secondary_prior=secondary_prior_t,
                innovation_features=innov_t,
                prior_bank=prior_bank_t,
                prior_bank_stats=prior_stats_t,
            )
            pred_scaled = out["state"].squeeze(0).detach().cpu().numpy()
            preds[i, t] = unscale_state(pred_scaled)
            if return_logvar:
                # Model predicts log-variance in scaled space; map to physical units.
                logvar_scaled = out["logvar"].squeeze(0).detach().cpu().numpy().astype(np.float64)
                logvars[i, t] = logvar_scaled + 2.0 * log_scale
    if return_logvar:
        return preds, logvars
    return preds


@torch.no_grad()
def run_model_inference_batched(
    model: TemporalGraphEstimator,
    states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    station_ecef: np.ndarray,
    window_size: int,
    ekf_prior: np.ndarray | None = None,
    ukf_prior: np.ndarray | None = None,
    aukf_prior: np.ndarray | None = None,
    secondary_prior: np.ndarray | None = None,
    innovation_features: np.ndarray | None = None,
    prior_bank_stats: np.ndarray | None = None,
    return_logvar: bool = False,
    batch_size: int = 64,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Batched trajectory inference equivalent to run_model_inference.

    This keeps the same per-time-step windowing semantics while evaluating a
    batch of trajectories at each step. It is primarily used by repeated-seed
    audits where active local/message-passing stacks make per-trajectory loops
    unnecessarily slow.
    """
    device = next(model.parameters()).device
    model.eval()
    n_traj, t_steps = states.shape[:2]
    preds = np.full_like(states, np.nan, dtype=np.float64)
    logvars = np.full_like(states, np.nan, dtype=np.float64)

    station = (station_ecef / 6_378_137.0).astype(np.float32)
    station_tensor = torch.from_numpy(station).to(device)
    log_scale = np.log(STATE_SCALE.astype(np.float64))

    for t in range(window_size - 1, t_steps):
        t0 = t - window_size + 1
        for start in range(0, n_traj, batch_size):
            stop = min(start + batch_size, n_traj)
            sl = slice(start, stop)
            meas = scale_measurements(measurements[sl, t0 : t + 1].astype(np.float32))
            vis = visibility[sl, t0 : t + 1].astype(np.float32)[..., None]
            meas_t = torch.from_numpy(meas).to(device)
            vis_t = torch.from_numpy(vis).to(device)

            prior_t = None
            if model.use_ekf_prior:
                if ekf_prior is None:
                    raise ValueError("Model requires EKF prior but none provided.")
                prior_t = torch.from_numpy(scale_state(ekf_prior[sl, t].astype(np.float32))).to(device)

            secondary_prior_t = None
            if model.use_dual_prior_fusion:
                if secondary_prior is None:
                    raise ValueError("Model requires secondary prior but none was provided.")
                secondary_prior_t = torch.from_numpy(scale_state(secondary_prior[sl, t].astype(np.float32))).to(device)

            innov_t = None
            if model.use_innovation_features:
                if innovation_features is None:
                    raise ValueError("Model requires innovation features but none were provided.")
                innov_t = torch.from_numpy(innovation_features[sl, t0 : t + 1].astype(np.float32)).to(device)

            prior_bank_t = None
            prior_stats_t = None
            if getattr(model, "use_prior_bank_fusion", False):
                if ekf_prior is None or ukf_prior is None or aukf_prior is None:
                    raise ValueError("Prior-bank fusion requires EKF, UKF, and AUKF priors.")
                prior_bank_scaled = np.stack(
                    [
                        scale_state(ekf_prior[sl, t].astype(np.float32)),
                        scale_state(ukf_prior[sl, t].astype(np.float32)),
                        scale_state(aukf_prior[sl, t].astype(np.float32)),
                    ],
                    axis=1,
                )
                prior_bank_t = torch.from_numpy(prior_bank_scaled).to(device)
                if getattr(model, "prior_stats_dim", 0) > 0:
                    if prior_bank_stats is None:
                        raise ValueError("prior_bank_stats are required for prior-bank fusion.")
                    prior_stats_t = torch.from_numpy(prior_bank_stats[sl, t].astype(np.float32)).to(device)

            out = model(
                measurements=meas_t,
                visibility=vis_t,
                station_xyz=station_tensor,
                ekf_prior=prior_t,
                secondary_prior=secondary_prior_t,
                innovation_features=innov_t,
                prior_bank=prior_bank_t,
                prior_bank_stats=prior_stats_t,
            )
            preds[sl, t] = unscale_state(out["state"].detach().cpu().numpy())
            if return_logvar:
                logvar_scaled = out["logvar"].detach().cpu().numpy().astype(np.float64)
                logvars[sl, t] = logvar_scaled + 2.0 * log_scale

    if return_logvar:
        return preds, logvars
    return preds


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = {}
    result.update(position_velocity_rmse(y_true, y_pred))
    result.update(median_absolute_error(y_true, y_pred))
    return result


def relative_improvement_percent(baseline: float, candidate: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0
    return 100.0 * (baseline - candidate) / baseline


def metric_entry_diverged(
    entry: dict[str, Any],
    *,
    pos_rmse_threshold_m: float = 1.0e8,
    vel_rmse_threshold_mps: float = 1.0e5,
    rmse_to_median_ratio_threshold: float = 1.0e4,
) -> bool:
    if "diverged" in entry:
        return bool(entry["diverged"])

    pos_rmse = float(entry.get("pos_rmse_m", np.nan))
    vel_rmse = float(entry.get("vel_rmse_mps", np.nan))
    med_abs_pos = float(entry.get("med_abs_pos_m", np.nan))
    med_abs_vel = float(entry.get("med_abs_vel_mps", np.nan))

    if not np.isfinite(pos_rmse) or not np.isfinite(vel_rmse):
        return True
    if abs(pos_rmse) > pos_rmse_threshold_m or abs(vel_rmse) > vel_rmse_threshold_mps:
        return True
    if np.isfinite(med_abs_pos) and med_abs_pos > 1e-9:
        if abs(pos_rmse) / med_abs_pos > rmse_to_median_ratio_threshold:
            return True
    if np.isfinite(med_abs_vel) and med_abs_vel > 1e-9:
        if abs(vel_rmse) / med_abs_vel > rmse_to_median_ratio_threshold:
            return True
    return False


def trajectory_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return per-trajectory RMSE arrays for position and velocity."""
    pos_sq = (y_true[..., :3] - y_pred[..., :3]) ** 2
    vel_sq = (y_true[..., 3:] - y_pred[..., 3:]) ** 2
    pos = np.sqrt(np.mean(pos_sq, axis=(1, 2)))
    vel = np.sqrt(np.mean(vel_sq, axis=(1, 2)))
    return {"pos_rmse_m": pos, "vel_rmse_mps": vel}


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    ci_percent: float = 95.0,
    seed: int = 42,
) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    n = x.size
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(x[idx])
    alpha = (100.0 - ci_percent) / 2.0
    return {
        "mean": float(np.mean(x)),
        "ci_low": float(np.percentile(means, alpha)),
        "ci_high": float(np.percentile(means, 100.0 - alpha)),
    }


def paired_bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 3000,
    ci_percent: float = 95.0,
    seed: int = 42,
) -> dict[str, float]:
    """CI for mean(a - b), preserving pairing via shared bootstrap indices."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("Paired arrays must have equal size.")
    rng = np.random.default_rng(seed)
    n = x.size
    diffs = x - y
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(diffs[idx])
    alpha = (100.0 - ci_percent) / 2.0
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(boot, alpha)),
        "ci_high": float(np.percentile(boot, 100.0 - alpha)),
    }


def fit_logvar_shift(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logvar: np.ndarray,
    channel_slice: slice,
) -> float:
    """Fit additive shift for log-variance by closed-form NLL optimum."""
    err_sq = (y_pred[..., channel_slice] - y_true[..., channel_slice]) ** 2
    lv = logvar[..., channel_slice]
    valid = np.isfinite(lv)
    if not np.any(valid):
        return 0.0
    ratio = err_sq[valid] * np.exp(-np.clip(lv[valid], -40.0, 40.0))
    ratio_mean = float(np.mean(np.clip(ratio, 1e-12, 1e12)))
    return float(np.clip(np.log(ratio_mean), -30.0, 30.0))


def apply_logvar_shift(
    logvar: np.ndarray,
    pos_shift: float,
    vel_shift: float,
) -> np.ndarray:
    out = np.array(logvar, copy=True)
    out[..., :3] = out[..., :3] + pos_shift
    out[..., 3:] = out[..., 3:] + vel_shift
    return out


def build_innovation_features(
    dataset_cfg: DatasetConfig,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times: np.ndarray,
    ekf_prior: np.ndarray,
) -> np.ndarray:
    return compute_innovation_features(
        prior_states=ekf_prior,
        measurements=measurements,
        visibility=visibility,
        times_s=times,
        stations=dataset_cfg.stations,
        meas_std_vector=dataset_cfg.measurement_noise.std_vector,
    )


def build_prior_bank_feature_array(
    ekf_prior: np.ndarray,
    ukf_prior: np.ndarray,
    aukf_prior: np.ndarray,
    *,
    dataset_cfg: DatasetConfig | None = None,
    station_ecef: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    times: np.ndarray | None = None,
    use_observability_context: bool = False,
) -> np.ndarray:
    stats = compute_prior_bank_stats(ekf_prior, ukf_prior, aukf_prior)
    if use_observability_context:
        if dataset_cfg is None or visibility is None or times is None:
            raise ValueError("Observability context requires dataset_cfg, visibility, and times.")
        stations = dataset_cfg.stations
        if station_ecef is not None and len(stations) != int(np.asarray(station_ecef).shape[0]):
            stations = stations_from_ecef(station_ecef)
        obs = compute_observability_context_features(
            prior_states=ekf_prior,
            visibility=visibility,
            times_s=times,
            stations=stations,
            meas_std_vector=dataset_cfg.measurement_noise.std_vector,
        )
        stats = np.concatenate([stats, obs], axis=-1)
    return stats


def compute_method_activity(
    prediction: np.ndarray,
    ekf_prior: np.ndarray,
    aukf_prior: np.ndarray | None = None,
    eval_start: int = 0,
) -> dict[str, float]:
    pred = prediction[:, eval_start:]
    ekf = ekf_prior[:, eval_start:]
    delta_ekf = np.linalg.norm(pred - ekf, axis=-1)
    stats = {
        "mean_abs_delta_vs_ekf": float(np.mean(delta_ekf)),
        "median_abs_delta_vs_ekf": float(np.median(delta_ekf)),
        "fraction_steps_delta_vs_ekf_gt_1m": float(np.mean(delta_ekf > 1.0)),
        "fraction_steps_delta_vs_ekf_gt_10m": float(np.mean(delta_ekf > 10.0)),
    }
    if aukf_prior is not None:
        aukf = aukf_prior[:, eval_start:]
        delta_aukf = np.linalg.norm(pred - aukf, axis=-1)
        stats["mean_abs_delta_vs_aukf"] = float(np.mean(delta_aukf))
        stats["median_abs_delta_vs_aukf"] = float(np.median(delta_aukf))
    return stats


def build_scorecard(
    scenario_metrics: dict[str, Any],
    *,
    method_name: str,
    scenario_name: str | None = None,
    ukf_name: str = "UKF",
    classical_best_name: str | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = scenario_metrics.get(method_name, {})
    ukf = scenario_metrics.get(ukf_name, {})
    best_name = classical_best_name or scenario_metrics.get("_best_classical_method")
    best = scenario_metrics.get(best_name, {}) if best_name else {}
    meta = scenario_metrics.get("_meta", {}) if isinstance(scenario_metrics.get("_meta", {}), dict) else {}
    method_activity = meta.get("method_activity", {}).get(method_name, {})
    thresholds = thresholds or {}
    entry_diverged = metric_entry_diverged(entry)
    ukf_diverged = metric_entry_diverged(ukf)
    best_diverged = metric_entry_diverged(best) if best else True
    ukf_valid = bool(ukf) and not ukf_diverged
    best_valid = bool(best) and not best_diverged
    improvement_vs_ukf = float(entry.get("improvement_vs_ukf_pos_rmse_percent", np.nan))
    improvement_vs_best = float(entry.get("improvement_vs_best_classical_pos_rmse_percent", np.nan))
    mean_abs_delta_vs_ekf = float(method_activity.get("mean_abs_delta_vs_ekf", np.nan))
    frac_delta_gt_10m = float(method_activity.get("fraction_steps_delta_vs_ekf_gt_10m", np.nan))
    min_delta_vs_ekf = float(thresholds.get("min_delta_vs_ekf_mean_m", 0.0))
    min_frac_delta_gt_10m = float(thresholds.get("min_fraction_steps_delta_vs_ekf_gt_10m", 0.0))
    min_ukf_improvement = float(thresholds.get("min_stress_improvement_vs_ukf_percent", 0.0))
    max_nominal_degradation = float(thresholds.get("max_nominal_degradation_vs_best_classical_percent", np.inf))
    activity_nontrivial = bool(
        np.isfinite(mean_abs_delta_vs_ekf)
        and np.isfinite(frac_delta_gt_10m)
        and mean_abs_delta_vs_ekf >= min_delta_vs_ekf
        and frac_delta_gt_10m >= min_frac_delta_gt_10m
    )
    scorecard = {
        "scenario": scenario_name,
        "candidate_diverged": entry_diverged,
        "ukf_valid_for_comparison": ukf_valid,
        "best_classical_valid_for_comparison": best_valid,
        "beats_ukf": bool(
            ukf_valid and not entry_diverged and float(entry.get("pos_rmse_m", np.inf)) < float(ukf.get("pos_rmse_m", -np.inf))
        ),
        "beats_best_classical": bool(
            best_valid and not entry_diverged and float(entry.get("pos_rmse_m", np.inf)) <= float(best.get("pos_rmse_m", -np.inf))
        ),
        "improvement_vs_ukf_percent": improvement_vs_ukf,
        "improvement_vs_best_classical_percent": improvement_vs_best,
        "mean_abs_delta_vs_ekf_m": mean_abs_delta_vs_ekf,
        "fraction_steps_delta_vs_ekf_gt_10m": frac_delta_gt_10m,
        "activity_nontrivial": activity_nontrivial,
        "meets_min_ukf_improvement_threshold": bool(
            ukf_valid and not entry_diverged and np.isfinite(improvement_vs_ukf) and improvement_vs_ukf >= min_ukf_improvement
        ),
        "meets_max_nominal_degradation_threshold": bool(
            best_valid
            and not entry_diverged
            and np.isfinite(improvement_vs_best)
            and improvement_vs_best >= -max_nominal_degradation
        ),
        "thresholds_applied": {
            "min_delta_vs_ekf_mean_m": min_delta_vs_ekf,
            "min_fraction_steps_delta_vs_ekf_gt_10m": min_frac_delta_gt_10m,
            "min_stress_improvement_vs_ukf_percent": min_ukf_improvement,
            "max_nominal_degradation_vs_best_classical_percent": max_nominal_degradation,
        },
    }
    return scorecard
