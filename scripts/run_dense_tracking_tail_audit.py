"""Tail-conditioned audit of the network-consistent dense-tracking probe.

A reviewer concern on the earlier reading of the dense-tracking probe was
that the predeclared all-step paired RGR-GF-minus-best-classical mean is
dominated by a small number of megametre-scale tails for the learned
estimator. This script re-runs the same deterministic 6-realization,
20-trajectory probe (same seeds, identical estimator setup) and emits an
auxiliary tail-conditioned audit alongside the previously committed artifact:

* per-trajectory all-step / observed-step RMSE for each estimator
* a per-method count of trajectories crossing the 100 km engineering-
  adequacy threshold (the "gross failure" count) and the 100,000 km
  divergence guard
* a paired RGR-GF-minus-best-classical comparison restricted to the
  jointly engineering-adequate subset (every method below 100 km on a
  given trajectory) so the mean is not dominated by a few megametre tails
* the same paired comparison on the median (heavy-tail-robust)

Outputs:
- ``results/credible_dense_od_probe/tail_audit.json`` (schema v1)
- ``results/credible_dense_od_probe/tail_audit.csv`` (per-trajectory)

The original probe artifact is left byte-identical.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gnn_state_estimation.evaluation import (
    BaselineConfig,
    build_innovation_features,
    build_prior_bank_feature_array,
    generate_noisy_init,
    parse_baseline_config,
    run_filter_baselines,
    run_model_inference,
)
from gnn_state_estimation.models import TemporalGraphEstimator
from gnn_state_estimation.scenarios import (
    estimator_sim_config,
    has_estimator_overrides,
    truth_sim_config,
)
from gnn_state_estimation.simulation import generate_dataset, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "credible_dense_od_probe"
OUT_JSON = OUT_DIR / "tail_audit.json"
OUT_CSV = OUT_DIR / "tail_audit.csv"

SCENARIO_KEY = "credible_dense_od_test"
ELEV_CAP_DEG = 80.0
DIVERGENCE_POS_RMSE_M = 1.0e8
ENGINEERING_ADEQUATE_POS_RMSE_M = 1.0e5


def conditioned_baseline(base: BaselineConfig, cap_deg: float | None) -> BaselineConfig:
    return BaselineConfig(
        ekf=dataclasses.replace(base.ekf, angle_deweight_elev_cap_deg=cap_deg),
        ukf=dataclasses.replace(base.ukf, angle_deweight_elev_cap_deg=cap_deg),
        aukf=(
            None
            if base.aukf is None
            else dataclasses.replace(base.aukf, angle_deweight_elev_cap_deg=cap_deg)
        ),
        init_pos_std_m=base.init_pos_std_m,
        init_vel_std_mps=base.init_vel_std_mps,
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


def per_traj_pos_rmse(states, preds, eval_start, observed_mask):
    s = states[:, eval_start:, :3]
    p = preds[:, eval_start:, :3]
    sq = np.sum((s - p) ** 2, axis=-1)
    all_step = np.sqrt(np.mean(sq, axis=1))
    obs = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        m = observed_mask[i]
        if np.any(m):
            obs[i] = np.sqrt(np.mean(sq[i][m]))
    return all_step, obs


def resolve_sim_configs(cfg: dict, scenario: str):
    sim_cfg = copy.deepcopy(cfg["simulation"])
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        raise SystemExit(f"scenario {scenario!r} not found in benchmark_suite")
    est = estimator_sim_config(sim_cfg, scenario_cfg)
    truth = truth_sim_config(sim_cfg, scenario_cfg)
    return est, truth, scenario_cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--num-realizations", type=int, default=6)
    p.add_argument("--trajectories", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=330000)
    p.add_argument("--bootstrap", type=int, default=3000)
    return p


def _paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int):
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = finite.size
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = float(np.mean(finite[idx]))
    return float(np.mean(finite)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    base_baseline = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    device = torch.device("cpu")
    model = load_rgr_gf(cfg, train_cfg, device)

    K = int(args.num_realizations)
    N = int(args.trajectories)
    method_keys = ["EKF", "UKF", "AUKF", "RGR-GF"]

    est_sim, truth_sim, scenario_cfg = resolve_sim_configs(cfg, SCENARIO_KEY)
    shared_dynamics = not has_estimator_overrides(scenario_cfg) and est_sim == truth_sim
    if not shared_dynamics:
        raise SystemExit("perfect shared model required")
    truth_dc = parse_dataset_config(truth_sim)
    est_dc = parse_dataset_config(est_sim)

    fixed_cfg = conditioned_baseline(base_baseline, ELEV_CAP_DEG)

    traj_all: dict[str, list[np.ndarray]] = {m: [] for m in method_keys}
    traj_obs: dict[str, list[np.ndarray]] = {m: [] for m in method_keys}
    seeds_per_realization: list[int] = []

    for r in range(K):
        seed = args.base_seed + r
        seeds_per_realization.append(seed)
        data = generate_dataset(truth_dc, N, seed=seed)
        states = data["states"]
        meas = data["measurements"]
        vis = data["visibility"]
        times = data["times"]

        eval_vis = vis[:, eval_start:]
        vis_count = np.sum(eval_vis >= 0.5, axis=-1)
        observed_mask = vis_count >= 1

        rng = np.random.default_rng(seed)
        x0 = np.stack(
            [
                generate_noisy_init(
                    states[i, 0],
                    rng,
                    base_baseline.init_pos_std_m,
                    base_baseline.init_vel_std_mps,
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
            baseline_cfg=fixed_cfg,
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
        preds = {"EKF": pf["ekf"], "UKF": pf["ukf"], "AUKF": pf["aukf"], "RGR-GF": rgr}
        for m in method_keys:
            a, o = per_traj_pos_rmse(states, preds[m], eval_start, observed_mask)
            traj_all[m].append(a)
            traj_obs[m].append(o)
        print(f"realization {r} (seed {seed}) done", flush=True)

    all_arr = {m: np.concatenate(traj_all[m]) for m in method_keys}
    obs_arr = {m: np.concatenate(traj_obs[m]) for m in method_keys}
    n_traj = int(all_arr[method_keys[0]].size)

    def adequate(x):
        return np.isfinite(x) & (x <= ENGINEERING_ADEQUATE_POS_RMSE_M)

    def survives(x):
        return np.isfinite(x) & (x <= DIVERGENCE_POS_RMSE_M)

    per_method_gross = {m: int(np.sum(~adequate(all_arr[m]))) for m in method_keys}
    per_method_divergent = {m: int(np.sum(~survives(all_arr[m]))) for m in method_keys}

    # Joint non-divergence: every method below the divergence guard.
    joint_non_div_mask = np.ones(n_traj, dtype=bool)
    for m in method_keys:
        joint_non_div_mask &= survives(all_arr[m])
    # Joint engineering-adequate (the tail-conditioned set).
    joint_adequate_mask = np.ones(n_traj, dtype=bool)
    for m in method_keys:
        joint_adequate_mask &= adequate(all_arr[m])

    classical = ["EKF", "UKF", "AUKF"]
    # Use the unconditional best classical primary (matches the committed
    # artifact's predeclared rule of pooled all-step RMSE over survivors).
    pooled_rmse_unconditional = {
        m: float(np.sqrt(np.mean(all_arr[m][survives(all_arr[m])] ** 2)))
        for m in classical
    }
    best_classical = min(pooled_rmse_unconditional, key=pooled_rmse_unconditional.get)

    def _conditioned_paired(mask: np.ndarray, kind: str) -> dict[str, Any]:
        a = all_arr["RGR-GF"][mask] if kind == "all_step" else obs_arr["RGR-GF"][mask]
        b = all_arr[best_classical][mask] if kind == "all_step" else obs_arr[best_classical][mask]
        valid = np.isfinite(a) & np.isfinite(b)
        a = a[valid]
        b = b[valid]
        diff = a - b
        # Mean-difference paired bootstrap
        mean_diff, mean_lo, mean_hi = _paired_bootstrap(
            diff, n_boot=int(args.bootstrap), seed=20260520
        )
        # Median-difference paired bootstrap
        rng = np.random.default_rng(20260521)
        if diff.size > 0:
            medians = np.empty(int(args.bootstrap), dtype=np.float64)
            for i in range(int(args.bootstrap)):
                idx = rng.integers(0, diff.size, diff.size)
                medians[i] = float(np.median(diff[idx]))
            median_diff = float(np.median(diff))
            median_lo, median_hi = (
                float(np.percentile(medians, 2.5)),
                float(np.percentile(medians, 97.5)),
            )
        else:
            median_diff = median_lo = median_hi = float("nan")
        # Pooled RMSE on the subset (the committed-artifact metric)
        pooled_rgr = float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")
        pooled_best = float(np.sqrt(np.mean(b ** 2))) if b.size else float("nan")
        rgr_better_count = int(np.sum(diff < 0.0)) if diff.size else 0
        return {
            "n_paired": int(diff.size),
            "best_classical": best_classical,
            "pooled_rgr_gf_rmse_m": round(pooled_rgr, 2) if np.isfinite(pooled_rgr) else None,
            "pooled_best_classical_rmse_m": round(pooled_best, 2) if np.isfinite(pooled_best) else None,
            "pooled_rmse_difference_m": round(pooled_rgr - pooled_best, 2)
            if np.isfinite(pooled_rgr) and np.isfinite(pooled_best)
            else None,
            "paired_mean_difference_m": round(mean_diff, 2) if np.isfinite(mean_diff) else None,
            "paired_mean_ci95_low_m": round(mean_lo, 2) if np.isfinite(mean_lo) else None,
            "paired_mean_ci95_high_m": round(mean_hi, 2) if np.isfinite(mean_hi) else None,
            "paired_median_difference_m": round(median_diff, 2)
            if np.isfinite(median_diff)
            else None,
            "paired_median_ci95_low_m": round(median_lo, 2) if np.isfinite(median_lo) else None,
            "paired_median_ci95_high_m": round(median_hi, 2) if np.isfinite(median_hi) else None,
            "rgr_gf_better_count": rgr_better_count,
            "rgr_gf_better_percent": round(100.0 * rgr_better_count / max(diff.size, 1), 2),
        }

    joint_adequate_indices = np.where(joint_adequate_mask)[0].tolist()
    joint_non_div_indices = np.where(joint_non_div_mask)[0].tolist()
    n_joint_adequate = int(np.sum(joint_adequate_mask))
    n_joint_non_div = int(np.sum(joint_non_div_mask))

    payload = {
        "schema_version": "dense_tracking_tail_audit_v1",
        "scenario": "dense_tracking_tail_audit",
        "regime": "network-consistent dense-tracking probe (same seeds and estimator setup as the committed credible_dense_od_probe_v1)",
        "num_realizations": K,
        "trajectories_per_realization": N,
        "n_trajectories_total": n_traj,
        "seeds_per_realization": seeds_per_realization,
        "engineering_adequate_pos_rmse_m": ENGINEERING_ADEQUATE_POS_RMSE_M,
        "divergence_guard_pos_rmse_m": DIVERGENCE_POS_RMSE_M,
        "per_method_gross_failure_count": per_method_gross,
        "per_method_divergent_count": per_method_divergent,
        "best_classical_unconditional": best_classical,
        "pooled_rmse_unconditional_classical_m": {
            m: round(pooled_rmse_unconditional[m], 2) for m in classical
        },
        "tail_conditioning": {
            "joint_non_divergent_count": n_joint_non_div,
            "joint_engineering_adequate_count": n_joint_adequate,
            "joint_non_divergent_paired_all_step": _conditioned_paired(
                joint_non_div_mask, "all_step"
            ),
            "joint_non_divergent_paired_observed_step": _conditioned_paired(
                joint_non_div_mask, "observed_step"
            ),
            "joint_engineering_adequate_paired_all_step": _conditioned_paired(
                joint_adequate_mask, "all_step"
            ),
            "joint_engineering_adequate_paired_observed_step": _conditioned_paired(
                joint_adequate_mask, "observed_step"
            ),
        },
        "interpretation": (
            "The paired RGR-GF-minus-best-classical comparison is reported "
            "on two subsets: (1) joint non-divergent trajectories (every "
            "method below the 100,000 km divergence guard); (2) joint "
            "engineering-adequate trajectories (every method below the 100 km "
            "engineering-adequacy threshold). On the tail-conditioned "
            "engineering-adequate subset the paired mean and median "
            "differences and the pooled-RMSE difference are reported so the "
            "comparison cannot be dominated by megametre-scale tails. "
            "RGR-GF gross failures and divergent trajectories are counted "
            "per method so the per-realization tail structure is auditable."
        ),
        "notes": (
            "This audit is deterministic for fixed seeds and uses the same "
            "fixed, previously trained learned estimator evaluated without "
            "any per-realization refitting; the committed primary probe "
            "artifact is left byte-identical."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")

    rows = []
    for i in range(n_traj):
        row = {
            "trajectory_index": i,
            "realization_index": i // N,
            "joint_engineering_adequate": bool(joint_adequate_mask[i]),
            "joint_non_divergent": bool(joint_non_div_mask[i]),
        }
        for m in method_keys:
            row[f"{m}_all_step_pos_rmse_m"] = float(all_arr[m][i]) if np.isfinite(all_arr[m][i]) else None
            row[f"{m}_observed_step_pos_rmse_m"] = float(obs_arr[m][i]) if np.isfinite(obs_arr[m][i]) else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
