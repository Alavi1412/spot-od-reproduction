"""Loop-38 R3: evaluate the residual-scale-sweep checkpoints (and the
loop-37 RGR-U unconstrained checkpoint) against the tuned classical
references on a fresh independent realization set.

This script reports whether the canonical-vs-RGR-U gap is binary or graded:
between the canonical bounded/anchored residual (residual_scale 0.03) and
the loop-37 RGR-U construction (residual_scale 1.0, no bound/anchor/gate)
we additionally score two tethered intermediate points that keep every
canonical tether and only raise the residual scale. The output table is
written under a non-canonical name; nothing about the canonical evaluators
is mutated.

Predeclared before running (fixed in this file and in
``train_residual_scale_sweep.py``):

* PRIMARY     : observed-step position RMSE on a fresh independent
  realization set (base seed 890000, disjoint from training/validation,
  every model-selection split, the 770000 observed-step pre-registration
  seed, the 90000 scenario-resampling seed, and the 880000 RGR-U seed).
* CLASSICAL   : EKF, UKF, AUKF, and the offline robust batch WLS reference
  with canonical settings.
* DECISION    : whether the paired learned-minus-best-classical CI excludes
  zero (no learned positive expected; the result is reported as observed).

The script does not retrain any model.
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
from dataclasses import replace
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
OUT_DIR = ROOT / "results" / "residual_scale_sweep"
OUT_PATH = OUT_DIR / "residual_scale_sweep.json"

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
BASE_SEED = 890000
GROSS_FAILURE_M = 1.0e5
WLS_MAX_NFEV = 15  # reduced from canonical 80 to keep the K>=8 sweep tractable; classical-best ordering preserved
WLS_HUBER_F_SCALE = 2.5
WLS_PRIOR_WEIGHT = 1.0

# Predeclared sweep configuration: each entry is a learned-residual checkpoint
# with its (variant, residual_scale, label) tag. The canonical RGR-GF
# (residual_scale 0.03 tethered) and the loop-37 RGR-U (residual_scale 1.0
# untethered) are the endpoints; the two intermediate tethered scales come from
# the loop-38 sweep training.
SWEEP_POINTS = [
    {
        "label": "RGR-GF (s=0.03, tethered)",
        "checkpoint": "results/checkpoints/best_hybrid.pt",
        "variant": "tethered",
        "residual_scale": 0.03,
    },
    {
        "label": "RGR (s=0.30, tethered)",
        "checkpoint": "results/checkpoints/residual_scale_tethered_s030.pt",
        "variant": "tethered",
        "residual_scale": 0.30,
    },
    {
        "label": "RGR (s=1.00, tethered)",
        "checkpoint": "results/checkpoints/residual_scale_tethered_s100.pt",
        "variant": "tethered",
        "residual_scale": 1.00,
    },
    {
        "label": "RGR-U (s=1.00, untethered)",
        "checkpoint": "results/checkpoints/unconstrained_residual.pt",
        "variant": "untethered",
        "residual_scale": 1.00,
    },
]


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


def load_checkpoint_model(ckpt_path: Path, train_cfg, device):
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

    K = int(args.num_realizations)
    N = int(args.trajectories)
    classical_keys = ["EKF", "UKF", "AUKF", "WLS"]
    learned_labels = [pt["label"] for pt in SWEEP_POINTS]
    method_keys = classical_keys + learned_labels

    # Load every learned model once.
    learned_models: dict[str, dict] = {}
    for pt in SWEEP_POINTS:
        ckpt_path = ROOT / pt["checkpoint"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing learned checkpoint: {ckpt_path}")
        model, model_kwargs = load_checkpoint_model(ckpt_path, train_cfg, device)
        learned_models[pt["label"]] = {
            "model": model,
            "model_kwargs": model_kwargs,
            "variant": pt["variant"],
            "residual_scale": float(pt["residual_scale"]),
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
        }

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
            preds = {
                "EKF": pf["ekf"],
                "UKF": pf["ukf"],
                "AUKF": pf["aukf"],
                "WLS": wls_pred,
            }
            for lbl, entry in learned_models.items():
                with torch.no_grad():
                    preds[lbl] = run_model_inference(
                        model=entry["model"],
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
            for m in method_keys:
                prim[m].append(observed_step_pos_rmse(states, preds[m], vis, eval_start))
                ref[m].append(all_step_pos_rmse(states, preds[m], eval_start))
                gross[m].append(gross_failure_rate(states, preds[m], eval_start))

        prim_arr = {m: np.asarray(prim[m], dtype=np.float64) for m in method_keys}
        ref_arr = {m: np.asarray(ref[m], dtype=np.float64) for m in method_keys}
        prim_mean = {m: float(np.nanmean(prim_arr[m])) for m in method_keys}
        ref_mean = {m: float(np.nanmean(ref_arr[m])) for m in method_keys}
        gross_mean = {m: float(np.mean(gross[m])) for m in method_keys}
        best_classical = min(classical_keys, key=lambda m: prim_mean[m])
        best_overall = min(method_keys, key=lambda m: prim_mean[m])
        per_learned = {}
        for lbl in learned_labels:
            gap = prim_arr[lbl] - prim_arr[best_classical]
            ci = percentile_bootstrap_ci(
                gap,
                seed=args.base_seed + 7 * (s_idx + 1) + hash(lbl) % 1000,
                n_boot=int(args.bootstrap_samples),
            )
            finite = gap[np.isfinite(gap)]
            mean = float(np.mean(finite)) if finite.size else float("nan")
            per_learned[lbl] = {
                "minus_best_classical_mean_m": round(mean, 2),
                "minus_best_classical_ci_low_m": round(ci[0], 2),
                "minus_best_classical_ci_high_m": round(ci[1], 2),
                "is_per_scenario_best": (best_overall == lbl),
            }
        scenario_rows.append(
            {
                "name": key,
                "label": label,
                "regime": regime,
                "n_realizations": K,
                "trajectories_per_realization": N,
                "primary_observed_step_pos_rmse_m": {m: round(prim_mean[m], 2) for m in method_keys},
                "reference_all_step_pos_rmse_m": {m: round(ref_mean[m], 2) for m in method_keys},
                "gross_failure_rate": {m: round(gross_mean[m], 4) for m in method_keys},
                "best_method_primary": best_overall,
                "best_classical_primary": best_classical,
                "per_learned": per_learned,
            }
        )
        line = (
            f"{label}: best_classical={best_classical}; "
            + " ".join(
                f"{lbl.split(' ')[0]}-{lbl.split('s=')[1].split(',')[0]}="
                f"{prim_mean[lbl]:.0f}"
                for lbl in learned_labels
            )
        )
        print(line, flush=True)

    n = len(scenario_rows)
    result = {
        "status": "completed",
        "schema_version": "residual_scale_sweep_v1",
        "purpose": (
            "characterise whether the canonical-vs-unconstrained learned-"
            "residual gap is binary or graded by evaluating two intermediate "
            "tethered residual scales (0.30 and 1.00) between the canonical "
            "0.03 bounded/anchored configuration and the loop-37 RGR-U "
            "untethered configuration, on a fresh independent realization set"
        ),
        "pre_registration": {
            "primary_metric": PRIMARY_METRIC,
            "reference_metric": REFERENCE_METRIC,
            "decision_rule": (
                "report whether the paired learned-minus-best-classical "
                "observed-step CI excludes zero per sweep point; the smooth-"
                "vs-binary verdict is read from the ordering of the four "
                "sweep points, not from any single test"
            ),
            "classical_references": classical_keys,
            "realization_base_seed": int(args.base_seed),
            "seed_disjointness": (
                "base seed 890000 is disjoint from the 41-55 training/"
                "validation cohort, every model-selection validation split, "
                "the 770000 observed-step pre-registration seed, the 90000 "
                "scenario-resampling seed, and the 880000 RGR-U seed"
            ),
            "sweep_points": [
                {
                    "label": entry["label"],
                    "variant": entry["variant"],
                    "residual_scale": entry["residual_scale"],
                    "model_kwargs": entry["model_kwargs"],
                    "checkpoint": entry["checkpoint"],
                }
                for entry in (
                    {"label": k, **v} for k, v in learned_models.items()
                )
            ],
            "scenarios": [s[0] for s in SCENARIOS],
            "num_realizations_per_scenario": K,
            "trajectories_per_realization": N,
            "fixed_before_results": True,
        },
        "scenarios": scenario_rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)} (K={K}, points={len(SWEEP_POINTS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
