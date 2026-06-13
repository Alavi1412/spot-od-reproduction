"""UNCORRECTED densified-visibility control (a conditioning-artifact probe).

This script densifies only the observation network (a 20-station global
ground network at an 8-degree elevation mask) while holding the compact force
model and orbit sampling fixed, with the legacy fixed angular measurement
weight (no topocentric-azimuth de-weighting). It is retained ONLY as the
uncorrected control: its multi-thousand-kilometre medians and high
gross-failure rate are a near-zenith topocentric-azimuth measurement-
conditioning artifact shared by every estimator (the azimuth is geometrically
singular at the station zenith and an over-tight angular weight drives
Kalman-gain blow-up), NOT an estimator-skill or regime-difficulty statement.
The astrodynamically meaningful conclusion is drawn instead from the corrected
credible dense-tracking probe (scripts/run_credible_dense_od_probe.py), which
applies the standard azimuth de-weighting, shares one perfect dynamics model
with third-body and SRP enabled, and uses a predeclared primary metric.

For K independently seeded realizations it runs the recursive classical
filters (EKF/UKF/AUKF, no training) and the fixed, previously trained RGR-GF
residual estimator in inference, and reports BOTH all-step and observed-step
position RMSE symmetrically with the realized visibility-bucket fractions. No
model is retrained; this regime is NOT a pre-registered endpoint.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from gnn_state_estimation.evaluation import (
    build_innovation_features,
    build_prior_bank_feature_array,
    generate_noisy_init,
    parse_baseline_config,
    run_filter_baselines,
    run_model_inference,
)
from gnn_state_estimation.models import TemporalGraphEstimator
from gnn_state_estimation.scenarios import estimator_sim_config, truth_sim_config
from gnn_state_estimation.simulation import generate_dataset, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "dense_visibility_probe"
OUT_PATH = OUT_DIR / "dense_visibility_probe.json"

SCENARIO_KEY = "dense_visibility_test"


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def resolve_sim_configs(cfg: dict, scenario: str):
    sim_cfg = copy.deepcopy(cfg["simulation"])
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        raise SystemExit(f"scenario {scenario!r} not found in benchmark_suite")
    return estimator_sim_config(sim_cfg, scenario_cfg), truth_sim_config(
        sim_cfg, scenario_cfg
    )


def load_rgr_gf(cfg: dict, train_cfg, device):
    spec = cfg["models"]["HybridGNN"]
    ckpt = torch.load(
        ROOT / "results" / "checkpoints" / spec["checkpoint_name"], map_location=device
    )
    model_kwargs = ckpt.get("model_kwargs", dict(spec.get("model_kwargs", {})))
    model = TemporalGraphEstimator(
        hidden_dim=train_cfg.hidden_dim,
        gnn_layers=train_cfg.gnn_layers,
        gru_layers=train_cfg.gru_layers,
        dropout=train_cfg.dropout,
        use_ekf_prior=bool(ckpt.get("use_ekf_prior", spec.get("use_ekf_prior", True))),
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# Numerical-divergence guardrail, identical to the manuscript protocol:
# "a trajectory is treated as numerically divergent when its trajectory-level
# position RMSE is non-finite or exceeds 1e8 m".
DIVERGENCE_POS_RMSE_M = 1.0e8
# The manuscript's stricter physical-adequacy ("engineering-failure") audit
# threshold: a trajectory estimate above 100 km position RMSE is a gross
# (physically inadequate) failure. In a measurement-update-dominant regime the
# distribution is heavy-tailed, so the estimator-skill comparison is reported
# robustly: median trajectory RMSE over all trajectories, plus the pooled RMSE
# restricted to physically adequate trajectories and the gross-failure rate.
ENGINEERING_ADEQUATE_POS_RMSE_M = 1.0e5


def per_traj_pos_rmse(states, preds, eval_start, observed_mask) -> tuple:
    """Per-trajectory all-step and observed-step position RMSE.

    observed_mask is (n_traj, n_eval) boolean (>=1 visible station at step).
    Returns (all_step[n_traj], observed_step[n_traj]) with NaN where a
    trajectory has no observed step.
    """
    s = states[:, eval_start:, :3]
    p = preds[:, eval_start:, :3]
    sq = np.sum((s - p) ** 2, axis=-1)  # (n_traj, n_eval) squared pos error
    all_step = np.sqrt(np.mean(sq, axis=1))
    obs = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        m = observed_mask[i]
        if np.any(m):
            obs[i] = np.sqrt(np.mean(sq[i][m]))
    return all_step, obs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--num-realizations", type=int, default=12)
    ap.add_argument("--trajectories", type=int, default=32)
    ap.add_argument("--base-seed", type=int, default=260000)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    device = torch.device("cpu")
    model = load_rgr_gf(cfg, train_cfg, device)

    K = int(args.num_realizations)
    N = int(args.trajectories)
    method_keys = ["EKF", "UKF", "AUKF", "RGR-GF"]

    est_sim, truth_sim = resolve_sim_configs(cfg, SCENARIO_KEY)
    truth_dc = parse_dataset_config(truth_sim)
    est_dc = parse_dataset_config(est_sim)
    n_stations = len(truth_sim["stations"])

    # Per-trajectory pooled accumulators (pooled over all realizations).
    traj_all = {m: [] for m in method_keys}
    traj_obs = {m: [] for m in method_keys}
    zero_vis_fracs = []
    one_vis_fracs = []
    multi_vis_fracs = []

    for r in range(K):
        seed = args.base_seed + r
        data = generate_dataset(truth_dc, N, seed=seed)
        states = data["states"]
        meas = data["measurements"]
        vis = data["visibility"]
        times = data["times"]

        eval_vis = vis[:, eval_start:]
        vis_count = np.sum(eval_vis >= 0.5, axis=-1)
        observed_mask = vis_count >= 1  # (n_traj, n_eval)
        zero_vis_fracs.append(float(np.mean(vis_count == 0)))
        one_vis_fracs.append(float(np.mean(vis_count == 1)))
        multi_vis_fracs.append(float(np.mean(vis_count >= 2)))

        rng = np.random.default_rng(seed)
        x0 = np.stack(
            [
                generate_noisy_init(
                    states[i, 0],
                    rng,
                    baseline_cfg.init_pos_std_m,
                    baseline_cfg.init_vel_std_mps,
                )
                for i in range(states.shape[0])
            ]
        )
        pf = run_filter_baselines(
            states=states,
            measurements=meas,
            visibility=vis,
            times=times,
            dataset_cfg=est_dc,
            baseline_cfg=baseline_cfg,
            seed=seed,
            x0_estimates=x0,
        )
        inno = build_innovation_features(
            dataset_cfg=est_dc,
            measurements=meas,
            visibility=vis,
            times=times,
            ekf_prior=pf["ekf"],
        )
        pbank = build_prior_bank_feature_array(
            pf["ekf"],
            pf["ukf"],
            pf["aukf"],
            dataset_cfg=est_dc,
            station_ecef=data["station_ecef"],
            visibility=vis,
            times=times,
            use_observability_context=False,
        )
        with torch.no_grad():
            rgr = run_model_inference(
                model=model,
                states=states,
                measurements=meas,
                visibility=vis,
                station_ecef=data["station_ecef"],
                window_size=int(train_cfg.window_size),
                ekf_prior=pf["ekf"],
                ukf_prior=pf["ukf"],
                aukf_prior=pf["aukf"],
                innovation_features=inno,
                prior_bank_stats=pbank,
            )
        preds = {
            "EKF": pf["ekf"],
            "UKF": pf["ukf"],
            "AUKF": pf["aukf"],
            "RGR-GF": rgr,
        }
        for m in method_keys:
            a, o = per_traj_pos_rmse(states, preds[m], eval_start, observed_mask)
            traj_all[m].append(a)
            traj_obs[m].append(o)
        print(
            f"realization {r}: zero-vis={zero_vis_fracs[-1]:.3f} done",
            flush=True,
        )

    # Concatenate per-trajectory arrays over all realizations.
    all_arr = {m: np.concatenate(traj_all[m]) for m in method_keys}
    obs_arr = {m: np.concatenate(traj_obs[m]) for m in method_keys}
    n_traj_total = int(all_arr[method_keys[0]].size)

    def finite_ok(x):
        return np.isfinite(x) & (x <= DIVERGENCE_POS_RMSE_M)

    # Per-method divergence-avoidance (all-step trajectory RMSE within guard).
    survive = {m: finite_ok(all_arr[m]) for m in method_keys}
    div_avoid = {
        m: round(float(np.mean(survive[m])), 4) for m in method_keys
    }

    def rmse_over(idx_arr, vals):
        v = vals[idx_arr]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return round(float(np.sqrt(np.mean(v ** 2))), 2)

    # Common-survivor set: trajectories non-divergent for EVERY method, so the
    # all-step / observed-step comparison is a fair paired comparison.
    common = np.ones(n_traj_total, dtype=bool)
    for m in method_keys:
        common &= survive[m]
    n_common = int(np.sum(common))

    per_method_finite_all = {
        m: rmse_over(survive[m], all_arr[m]) for m in method_keys
    }
    per_method_finite_obs = {
        m: rmse_over(survive[m], obs_arr[m]) for m in method_keys
    }
    common_all = {m: rmse_over(common, all_arr[m]) for m in method_keys}
    common_obs = {m: rmse_over(common, obs_arr[m]) for m in method_keys}

    # Robust primary view: median trajectory RMSE over ALL trajectories
    # (heavy-tail-insensitive), reported all-step and observed-step.
    def med(vals):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return None
        return round(float(np.median(v)), 2)

    median_all = {m: med(all_arr[m]) for m in method_keys}
    median_obs = {m: med(obs_arr[m]) for m in method_keys}

    # Gross-failure rate (manuscript engineering-adequacy threshold) and the
    # pooled RMSE over trajectories physically adequate for EVERY method.
    adequate = {
        m: np.isfinite(all_arr[m]) & (all_arr[m] <= ENGINEERING_ADEQUATE_POS_RMSE_M)
        for m in method_keys
    }
    gross_failure_rate = {
        m: round(float(np.mean(~adequate[m])), 4) for m in method_keys
    }
    common_adeq = np.ones(n_traj_total, dtype=bool)
    for m in method_keys:
        common_adeq &= adequate[m]
    n_common_adeq = int(np.sum(common_adeq))
    adeq_all = {m: rmse_over(common_adeq, all_arr[m]) for m in method_keys}
    adeq_obs = {m: rmse_over(common_adeq, obs_arr[m]) for m in method_keys}

    def best_of(d):
        cand = {k: v for k, v in d.items() if v is not None}
        return min(cand, key=cand.get) if cand else None

    best_div = max(method_keys, key=lambda m: div_avoid[m])
    best_all = best_of(median_all)
    best_obs = best_of(median_obs)
    best_adeq_obs = best_of(adeq_obs)
    zero_vis_mean = round(float(np.mean(zero_vis_fracs)), 4)

    def safe_diff(d, a, b):
        if d.get(a) is None or d.get(b) is None:
            return None
        return round(d[a] - d[b], 2)

    rgr_obs_minus_aukf = safe_diff(median_obs, "RGR-GF", "AUKF")
    rgr_obs_minus_ukf = safe_diff(median_obs, "RGR-GF", "UKF")

    # Is the regime estimator-discriminative at all? Only meaningful on the
    # physically adequate subset (where estimates are not catastrophic). If
    # the spread there is below a trivial margin, the difficulty is
    # estimator-independent and NO method (classical or learned) can be
    # called best -- the honest reading is "indistinguishable".
    MARGIN = 0.05
    adeq_obs_vals = [v for v in adeq_obs.values() if v is not None]
    if len(adeq_obs_vals) == len(method_keys) and min(adeq_obs_vals) > 0:
        spread = (max(adeq_obs_vals) - min(adeq_obs_vals)) / min(adeq_obs_vals)
    else:
        spread = 0.0
    discriminative = bool(spread > MARGIN)
    # A learned advantage "changes the conclusion" only if the regime is
    # discriminative AND RGR-GF is strictly best on the adequate-subset
    # observed-step metric by more than the trivial margin over every
    # classical reference. In a divergence-dominated, non-discriminative
    # regime this is False regardless of noisy median ordering.
    if discriminative and best_adeq_obs == "RGR-GF":
        classical_best = min(
            adeq_obs[m] for m in ("EKF", "UKF", "AUKF") if adeq_obs[m] is not None
        )
        conclusion_changes = bool(
            adeq_obs["RGR-GF"] is not None
            and classical_best > 0
            and (classical_best - adeq_obs["RGR-GF"]) / classical_best > MARGIN
        )
    else:
        conclusion_changes = False

    result = {
        "status": "completed",
        "schema_version": "dense_visibility_probe_v2",
        "regime": (
            "UNCORRECTED densified-visibility control: perfect-shared-compact-"
            "model, 20-station mid-latitude global ground network at an "
            "8-degree elevation mask, with the legacy fixed angular "
            "measurement weight (NO topocentric-azimuth de-weighting). Its "
            "catastrophic medians are a near-zenith azimuth measurement-"
            "conditioning artifact, not an estimator-skill or regime-"
            "difficulty statement; the corrected credible probe supersedes its "
            "interpretation"
        ),
        "pre_registered": False,
        "metric_reporting": (
            "all-step and observed-step reported symmetrically; trajectories "
            "exceeding the manuscript numerical-divergence guardrail "
            "(non-finite or >1e8 m trajectory position RMSE) are reported as a "
            "divergence-avoidance rate and excluded from the pooled RMSE"
        ),
        "num_realizations": K,
        "trajectories_per_realization": N,
        "n_trajectories_total": n_traj_total,
        "n_stations": n_stations,
        "eval_start_step": eval_start,
        "divergence_guard_pos_rmse_m": DIVERGENCE_POS_RMSE_M,
        "visibility": {
            "zero_visible_fraction_mean": zero_vis_mean,
            "one_visible_fraction_mean": round(float(np.mean(one_vis_fracs)), 4),
            "two_plus_visible_fraction_mean": round(
                float(np.mean(multi_vis_fracs)), 4
            ),
            "measurement_informed_fraction_mean": round(1.0 - zero_vis_mean, 4),
            "main_split_zero_visible_fraction_reference": 0.79,
        },
        "engineering_adequate_pos_rmse_m": ENGINEERING_ADEQUATE_POS_RMSE_M,
        "divergence_avoidance_fraction": div_avoid,
        "best_method_divergence_avoidance": best_div,
        "gross_failure_rate": gross_failure_rate,
        "median_trajectory_position_rmse_m": {
            "all_step": median_all,
            "observed_step": median_obs,
            "note": (
                "median over all trajectories (heavy-tail-robust primary view)"
            ),
        },
        "engineering_adequate_pooled_rmse_m": {
            "n_common_adequate_trajectories": n_common_adeq,
            "all_step": adeq_all,
            "observed_step": adeq_obs,
            "note": (
                "pooled over trajectories physically adequate (<=100 km "
                "trajectory RMSE) for every method"
            ),
        },
        "finite_pooled_position_rmse_m": {
            "all_step": per_method_finite_all,
            "observed_step": per_method_finite_obs,
            "note": "pooled over each method's own non-divergent trajectories",
        },
        "common_survivor_position_rmse_m": {
            "n_common_trajectories": n_common,
            "all_step": common_all,
            "observed_step": common_obs,
            "note": "pooled over trajectories non-divergent for every method",
        },
        "best_method_all_step_median": best_all,
        "best_method_observed_step_median": best_obs,
        "best_method_observed_step_adequate": best_adeq_obs,
        "rgr_gf_minus_aukf_observed_step_median_m": rgr_obs_minus_aukf,
        "rgr_gf_minus_ukf_observed_step_median_m": rgr_obs_minus_ukf,
        "adequate_observed_step_spread_fraction": round(float(spread), 4),
        "regime_is_estimator_discriminative": discriminative,
        "conclusion_changes_vs_sparse_regime": conclusion_changes,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} (K={K}, N={N})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
