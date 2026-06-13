"""Loop-37 FF1: evaluate the less-constrained learned residual comparator
(``RGR-U``, trained by ``train_unconstrained_residual.py``) against the tuned
classical references on a *fresh independent* realization set.

This directly answers the reviewer objection that the study's central negative
is "in substantial part" a design consequence of the bounded/anchored residual
architecture: if a learned estimator whose residual budget, gate, context
budget, and prior-anchoring/auxiliary penalties have all been removed *still*
does not beat the tuned classical references on the primary observed-step
endpoint, then the negative is not an artifact of that bound.

Predeclared before running (fixed in this file and in
``train_unconstrained_residual.py``):

* COMPARATOR  : the RGR-U checkpoint, run in inference only (no selection,
  tuning, or retraining on these realizations).
* PRIMARY     : observed-step position RMSE (>=1 visible station; the main
  evaluator convention).
* REFERENCE   : all-step position RMSE (propagation-dominated reference only).
* CLASSICAL   : EKF, UKF, tuned AUKF, and the offline robust batch
  weighted-least-squares OD reference, with the canonical WLS settings
  (Huber f-scale 2.5, prior weight 1.0, max 80 function evaluations).
* DECISION    : on the primary endpoint a learned positive requires RGR-U to
  be the per-scenario best method with the paired RGR-U-minus-best-classical
  percentile bootstrap CI strictly below zero.
* REALIZATION : independently seeded realizations, base seed 880000, disjoint
  from the 41--55 training/validation cohort, every model-selection split,
  the 770000 observed-step pre-registration seed, and the 90000
  scenario-resampling seed.

No positive is invented: the script reports whatever the predeclared rule
yields, including unbounded-residual numerical divergence if it occurs.
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

try:
    from run_batch_wls_baseline import fit_batch_wls_trajectory
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_batch_wls_baseline import fit_batch_wls_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "unconstrained_residual_comparator"
OUT_PATH = OUT_DIR / "unconstrained_residual_comparator.json"

SCENARIOS = [
    ("test", "Nominal", "nominal sparse-visibility synthetic split"),
    ("stress_test", "Measurement-noise stress", "inflated measurement noise/outliers"),
    (
        "force_model_mismatch_test",
        "Controlled force-model mismatch",
        "truth inflates drag/SRP/process noise; estimators keep the compact model",
    ),
]
PRIMARY_METRIC = "observed_step_position_rmse_m"
REFERENCE_METRIC = "all_step_position_rmse_m"
BASE_SEED = 880000
GROSS_FAILURE_M = 1.0e5
# Canonical batch-WLS settings (match run_batch_wls_baseline.py defaults).
WLS_MAX_NFEV = 80
WLS_HUBER_F_SCALE = 2.5
WLS_PRIOR_WEIGHT = 1.0


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
    if scenario == "stress_test":
        sim_cfg = deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        return sim_cfg, sim_cfg
    return estimator_sim_config(sim_cfg, scenario_cfg), truth_sim_config(sim_cfg, scenario_cfg)


def load_unconstrained(ckpt_path: Path, train_cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_kwargs = ckpt.get("model_kwargs", {})
    model = TemporalGraphEstimator(
        hidden_dim=train_cfg.hidden_dim,
        gnn_layers=train_cfg.gnn_layers,
        gru_layers=train_cfg.gru_layers,
        dropout=train_cfg.dropout,
        use_ekf_prior=bool(ckpt.get("use_ekf_prior", True)),
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_kwargs


def observed_step_pos_rmse(states, preds, visibility, eval_start) -> float:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    err = states[:, eval_start:][observed, :3] - preds[:, eval_start:][observed, :3]
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def all_step_pos_rmse(states, preds, eval_start) -> float:
    err = states[:, eval_start:, :3] - preds[:, eval_start:, :3]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def gross_failure_rate(states, preds, eval_start) -> float:
    err = states[:, eval_start:, :3] - preds[:, eval_start:, :3]
    per_traj = np.sqrt(np.mean(np.sum(err * err, axis=-1), axis=-1))
    bad = (~np.isfinite(per_traj)) | (per_traj > GROSS_FAILURE_M)
    return float(np.mean(bad))


def percentile_bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, finite.size, size=finite.size)
        boot[i] = float(np.mean(finite[idx]))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def run_wls(states, measurements, vis, times, est_dc, x0, baseline_cfg):
    initial_scale = np.array(
        [
            float(baseline_cfg.init_pos_std_m),
            float(baseline_cfg.init_pos_std_m),
            float(baseline_cfg.init_pos_std_m),
            float(baseline_cfg.init_vel_std_mps),
            float(baseline_cfg.init_vel_std_mps),
            float(baseline_cfg.init_vel_std_mps),
        ],
        dtype=np.float64,
    )
    preds = np.zeros_like(states)
    for i in range(states.shape[0]):
        pred, _ = fit_batch_wls_trajectory(
            measurements=measurements[i],
            visibility=vis[i],
            times_s=times[i],
            x0_est=x0[i],
            dataset_cfg=est_dc,
            initial_scale=initial_scale,
            max_nfev=WLS_MAX_NFEV,
            huber_f_scale=WLS_HUBER_F_SCALE,
            prior_weight=WLS_PRIOR_WEIGHT,
        )
        preds[i] = pred
    return preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--checkpoint", default="results/checkpoints/unconstrained_residual.pt")
    ap.add_argument("--num-realizations", type=int, default=8)
    ap.add_argument("--trajectories", type=int, default=24)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    device = torch.device("cpu")
    model, model_kwargs = load_unconstrained(Path(args.checkpoint), train_cfg, device)

    K = int(args.num_realizations)
    N = int(args.trajectories)
    classical_keys = ["EKF", "UKF", "AUKF", "WLS"]
    method_keys = classical_keys + ["RGR-U"]

    scenario_rows = []
    for s_idx, (key, label, regime) in enumerate(SCENARIOS):
        est_sim, truth_sim = resolve_sim_configs(cfg, key)
        truth_dc = parse_dataset_config(truth_sim)
        est_dc = parse_dataset_config(est_sim)
        prim = {m: [] for m in method_keys}
        ref = {m: [] for m in method_keys}
        gross = {m: [] for m in method_keys}
        for r in range(K):
            seed = args.base_seed + 1000 * (s_idx + 1) + r
            data = generate_dataset(truth_dc, N, seed=seed)
            states, meas, vis, times = (
                data["states"],
                data["measurements"],
                data["visibility"],
                data["times"],
            )
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
            wls_pred = run_wls(states, meas, vis, times, est_dc, x0, baseline_cfg)
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
                rgr_u = run_model_inference(
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
                "WLS": wls_pred,
                "RGR-U": rgr_u,
            }
            for m in method_keys:
                prim[m].append(observed_step_pos_rmse(states, preds[m], vis, eval_start))
                ref[m].append(all_step_pos_rmse(states, preds[m], eval_start))
                gross[m].append(gross_failure_rate(states, preds[m], eval_start))

        prim_arr = {m: np.asarray(prim[m], dtype=np.float64) for m in method_keys}
        ref_arr = {m: np.asarray(ref[m], dtype=np.float64) for m in method_keys}
        prim_mean = {m: float(np.nanmean(prim_arr[m])) for m in method_keys}
        prim_std = {
            m: float(np.nanstd(prim_arr[m], ddof=1)) if np.isfinite(prim_arr[m]).sum() > 1 else float("nan")
            for m in method_keys
        }
        ref_mean = {m: float(np.nanmean(ref_arr[m])) for m in method_keys}
        gross_mean = {m: float(np.mean(gross[m])) for m in method_keys}
        best_classical = min(classical_keys, key=lambda m: prim_mean[m])
        best_overall = min(method_keys, key=lambda m: prim_mean[m])
        paired_gap = prim_arr["RGR-U"] - prim_arr[best_classical]  # <0 => learned better
        gap_ci = percentile_bootstrap_ci(
            paired_gap, seed=args.base_seed + 7 * (s_idx + 1), n_boot=int(args.bootstrap_samples)
        )
        finite_gap = paired_gap[np.isfinite(paired_gap)]
        gap_mean = float(np.mean(finite_gap)) if finite_gap.size else float("nan")
        learned_positive = bool(
            best_overall == "RGR-U" and np.isfinite(gap_ci[1]) and gap_ci[1] < 0.0
        )
        scenario_rows.append(
            {
                "name": key,
                "label": label,
                "regime": regime,
                "n_realizations": K,
                "trajectories_per_realization": N,
                "primary_observed_step_pos_rmse_m": {m: round(prim_mean[m], 2) for m in method_keys},
                "primary_observed_step_pos_rmse_std_m": {
                    m: round(prim_std[m], 2) for m in method_keys
                },
                "reference_all_step_pos_rmse_m": {m: round(ref_mean[m], 2) for m in method_keys},
                "gross_failure_rate": {m: round(gross_mean[m], 4) for m in method_keys},
                "best_method_primary": best_overall,
                "best_classical_primary": best_classical,
                "rgr_u_minus_best_classical_primary_mean_m": round(gap_mean, 2),
                "rgr_u_minus_best_classical_primary_ci_low_m": round(gap_ci[0], 2),
                "rgr_u_minus_best_classical_primary_ci_high_m": round(gap_ci[1], 2),
                "learned_positive_under_predeclared_rule": learned_positive,
            }
        )
        print(
            f"{label}: PRIMARY(observed-step) best={best_overall} "
            f"EKF={prim_mean['EKF']:.1f} UKF={prim_mean['UKF']:.1f} "
            f"AUKF={prim_mean['AUKF']:.1f} WLS={prim_mean['WLS']:.1f} "
            f"RGR-U={prim_mean['RGR-U']:.1f} | RGR-U-best_classical={gap_mean:.1f} "
            f"CI[{gap_ci[0]:.1f},{gap_ci[1]:.1f}] learned_positive={learned_positive}",
            flush=True,
        )

    n = len(scenario_rows)
    learned_positive_count = sum(
        1 for r in scenario_rows if r["learned_positive_under_predeclared_rule"]
    )
    classical_best_count = sum(
        1 for r in scenario_rows if r["best_method_primary"] in classical_keys
    )
    result = {
        "status": "completed",
        "schema_version": "unconstrained_residual_comparator_v1",
        "comparator": {
            "name": "RGR-U",
            "description": (
                "less-constrained learned residual: identical architecture, "
                "capacity, inputs, training data, curriculum, and seed "
                "derivation to the canonical bounded residual, with the 0.03 "
                "tanh residual budget, the learned gate, the context budget, "
                "and the prior-anchoring/activity/entropy/visibility training "
                "penalties all removed (residual_scale 1.0, unbounded, no "
                "gate/budget, zero anchoring penalty)"
            ),
            "model_kwargs": model_kwargs,
        },
        "pre_registration": {
            "primary_metric": PRIMARY_METRIC,
            "reference_metric": REFERENCE_METRIC,
            "decision_rule": (
                "on the primary observed-step endpoint, a learned positive "
                "requires the less-constrained RGR-U to be the per-scenario "
                "best method with the paired RGR-U-minus-best-classical "
                "bootstrap CI strictly below zero; the all-step metric is the "
                "propagation-dominated reference only"
            ),
            "classical_references": classical_keys,
            "realization_base_seed": int(args.base_seed),
            "seed_disjointness": (
                "base seed 880000 is disjoint from the 41-55 training/"
                "validation cohort, every model-selection validation split, "
                "the 770000 observed-step pre-registration seed, and the "
                "90000 scenario-resampling base seed; the RGR-U checkpoint is "
                "run in inference only (no selection/tuning/retraining here)"
            ),
            "scenarios": [s[0] for s in SCENARIOS],
            "num_realizations_per_scenario": K,
            "trajectories_per_realization": N,
            "fixed_before_results": True,
        },
        "statistical_unit": (
            f"independent realization (independent trajectory population and "
            f"measurement-noise draw); per-scenario estimate is the mean over "
            f"{K} independent realizations with a percentile bootstrap CI on "
            f"the paired RGR-U-minus-best-classical primary-metric gap"
        ),
        "num_scenarios": n,
        "scenarios": scenario_rows,
        "summary": {
            "n_scenarios": n,
            "scenarios_with_learned_positive_under_predeclared_rule": learned_positive_count,
            "scenarios_with_classical_best_on_primary": classical_best_count,
            "verdict": (
                "the less-constrained learned residual does not beat the tuned "
                "classical references on the primary observed-step endpoint on "
                "any independent scenario: removing the residual budget and the "
                "prior-anchoring penalty does not overturn the negative"
                if learned_positive_count == 0
                else "less-constrained learned positive observed under the "
                "predeclared rule"
            ),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({n} scenarios, K={K})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
