"""Loop-35 MC-3: re-pre-register the discriminative observed-step endpoint on
a *fresh independent realization set*.

A reviewer noted that observed-step RMSE entered the manuscript as an
explicitly post-hoc sensitivity metric (adopted after the propagation-
dominated all-step metric proved weakly discriminative) while the
pre-registered primary endpoint was the weakly discriminative all-step
metric. This script converts observed-step RMSE into a *predeclared
primary endpoint on data that has never been used for anything else*.

Pre-registration record (fixed in code before any result is produced):

* PRIMARY endpoint  : observed-step position RMSE (steps from the window
  start onward with at least one visible station; the main-evaluator
  convention).
* REFERENCE endpoint : all-step position RMSE, reported only as the
  propagation-dominated reference, never as the decision metric.
* DECISION RULE     : on the PRIMARY endpoint, report the best method per
  scenario and the paired released-RGR-GF-minus-best-classical mean with a
  percentile bootstrap CI over the independent realizations; a learned
  positive requires the learned estimator to be the per-scenario best on
  the PRIMARY endpoint with the paired CI strictly below zero.
* REALIZATION SET   : independently seeded trajectory populations with
  independent measurement-noise draws, base seed 770000, disjoint from the
  training/validation seeds (41--55), from the model-selection validation
  splits (including the SatNOGS replay split), and from the loop-24/25
  scenario-resampling base seed (90000). No model is selected, tuned, or
  retrained on these realizations; the released compact RGR-GF checkpoint
  is run in inference only.

This is therefore an independent, pre-registered confirmation of the
observed-step ordering, distinct from (and not contaminating) the
disclosed post-hoc seed-cohort observed-step recomputation. No positive is
invented: the script reports whatever the predeclared rule yields.
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
OUT_DIR = ROOT / "results" / "observed_step_preregistration"
OUT_PATH = OUT_DIR / "observed_step_preregistration.json"

# Predeclared scenario set = the three protocol endpoints already
# named in the manuscript.
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
BASE_SEED = 770000


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


def percentile_bootstrap_ci(values: np.ndarray, *, seed: int, n_boot: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, values.size, size=values.size)
        boot[i] = float(np.mean(values[idx]))
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--num-realizations", type=int, default=12)
    ap.add_argument("--trajectories", type=int, default=32)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    # Optional single-scenario shard: when set to a valid scenario index the
    # run processes only that scenario (keeping its original index so the
    # per-realization seed is byte-identical to a full run) and writes the
    # one-scenario payload to --out-path. Used to parallelise the otherwise
    # sequential 3-scenario sweep across processes; the values are unchanged.
    ap.add_argument("--only-scenario-index", type=int, default=-1)
    ap.add_argument("--out-path", default=str(OUT_PATH))
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    device = torch.device("cpu")
    model = load_rgr_gf(cfg, train_cfg, device)

    K = int(args.num_realizations)
    N = int(args.trajectories)
    classical_keys = ["EKF", "UKF", "AUKF"]
    method_keys = classical_keys + ["RGR-GF"]

    scenario_rows = []
    only_idx = int(args.only_scenario_index)
    for s_idx, (key, label, regime) in enumerate(SCENARIOS):
        if only_idx >= 0 and s_idx != only_idx:
            continue
        est_sim, truth_sim = resolve_sim_configs(cfg, key)
        truth_dc = parse_dataset_config(truth_sim)
        est_dc = parse_dataset_config(est_sim)
        prim = {m: [] for m in method_keys}
        ref = {m: [] for m in method_keys}
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
                prim[m].append(observed_step_pos_rmse(states, preds[m], vis, eval_start))
                ref[m].append(all_step_pos_rmse(states, preds[m], eval_start))

        prim_arr = {m: np.asarray(prim[m], dtype=np.float64) for m in method_keys}
        ref_arr = {m: np.asarray(ref[m], dtype=np.float64) for m in method_keys}
        prim_mean = {m: float(np.mean(prim_arr[m])) for m in method_keys}
        prim_std = {m: float(np.std(prim_arr[m], ddof=1)) for m in method_keys}
        ref_mean = {m: float(np.mean(ref_arr[m])) for m in method_keys}
        best_classical = min(classical_keys, key=lambda m: prim_mean[m])
        best_overall = min(method_keys, key=lambda m: prim_mean[m])
        paired_gap = prim_arr["RGR-GF"] - prim_arr[best_classical]  # <0 => learned better
        gap_ci = percentile_bootstrap_ci(
            paired_gap, seed=args.base_seed + 7 * (s_idx + 1), n_boot=int(args.bootstrap_samples)
        )
        learned_positive = bool(
            best_overall == "RGR-GF" and gap_ci[1] < 0.0
        )

        # Strict-prefix K=8 endpoint-fixation support rollup.
        K_PRIMARY = 8
        if K >= K_PRIMARY:
            prim8 = {m: prim_arr[m][:K_PRIMARY] for m in method_keys}
            ref8 = {m: ref_arr[m][:K_PRIMARY] for m in method_keys}
            prim_mean8 = {m: float(np.mean(prim8[m])) for m in method_keys}
            prim_std8 = {m: float(np.std(prim8[m], ddof=1)) for m in method_keys}
            ref_mean8 = {m: float(np.mean(ref8[m])) for m in method_keys}
            best_classical8 = min(classical_keys, key=lambda m: prim_mean8[m])
            best_overall8 = min(method_keys, key=lambda m: prim_mean8[m])
            paired_gap8 = prim8["RGR-GF"] - prim8[best_classical8]
            gap_ci8 = percentile_bootstrap_ci(
                paired_gap8,
                seed=args.base_seed + 7 * (s_idx + 1) + 800000,
                n_boot=int(args.bootstrap_samples),
            )
            learned_positive8 = bool(
                best_overall8 == "RGR-GF" and gap_ci8[1] < 0.0
            )
            primary_k8 = {
                "n_realizations": K_PRIMARY,
                "primary_observed_step_pos_rmse_m": {
                    m: round(prim_mean8[m], 2) for m in method_keys
                },
                "primary_observed_step_pos_rmse_std_m": {
                    m: round(prim_std8[m], 2) for m in method_keys
                },
                "reference_all_step_pos_rmse_m": {
                    m: round(ref_mean8[m], 2) for m in method_keys
                },
                "best_method_primary": best_overall8,
                "best_classical_primary": best_classical8,
                "rgr_gf_minus_best_classical_primary_mean_m": round(
                    float(np.mean(paired_gap8)), 2
                ),
                "rgr_gf_minus_best_classical_primary_ci_low_m": round(gap_ci8[0], 2),
                "rgr_gf_minus_best_classical_primary_ci_high_m": round(gap_ci8[1], 2),
                "learned_positive_under_predeclared_rule": learned_positive8,
            }
        else:
            primary_k8 = None

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
                "best_method_primary": best_overall,
                "best_classical_primary": best_classical,
                "rgr_gf_minus_best_classical_primary_mean_m": round(float(np.mean(paired_gap)), 2),
                "rgr_gf_minus_best_classical_primary_ci_low_m": round(gap_ci[0], 2),
                "rgr_gf_minus_best_classical_primary_ci_high_m": round(gap_ci[1], 2),
                "learned_positive_under_predeclared_rule": learned_positive,
                "per_realization_primary_m": {
                    m: [round(v, 2) for v in prim_arr[m].tolist()] for m in method_keys
                },
                "per_realization_reference_all_step_m": {
                    m: [round(v, 2) for v in ref_arr[m].tolist()] for m in method_keys
                },
                "primary_k8_predeclared": primary_k8,
            }
        )
        print(
            f"{label}: PRIMARY(observed-step) best={best_overall} "
            f"EKF={prim_mean['EKF']:.1f} UKF={prim_mean['UKF']:.1f} "
            f"AUKF={prim_mean['AUKF']:.1f} RGR-GF={prim_mean['RGR-GF']:.1f} "
            f"| RGR-GF-best_classical={np.mean(paired_gap):.1f} "
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
        "schema_version": "observed_step_preregistration_v2",
        "pre_registration": {
            "primary_metric": PRIMARY_METRIC,
            "reference_metric": REFERENCE_METRIC,
            "decision_rule": (
                "on the primary observed-step endpoint, a learned positive "
                "requires the released RGR-GF to be the per-scenario best "
                "method with the paired RGR-GF-minus-best-classical bootstrap "
                "CI strictly below zero; the all-step metric is reported only "
                "as the propagation-dominated reference"
            ),
            "realization_base_seed": int(args.base_seed),
            "seed_disjointness": (
                "base seed 770000 is disjoint from the 41-55 training/"
                "validation cohort, from every model-selection validation "
                "split including the SatNOGS replay split, and from the "
                "loop-24/25 scenario-resampling base seed (90000); no model is "
                "selected, tuned, or retrained on these realizations"
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
            f"the paired RGR-GF-minus-best-classical primary-metric gap"
        ),
        "source": (
            "fresh independent realizations generated and scored at build time; "
            "classical filters plus the released compact RGR-GF checkpoint in "
            "inference only (no model selection on these realizations)"
        ),
        "num_scenarios": n,
        "scenarios": scenario_rows,
        "summary": {
            "n_scenarios": n,
            "scenarios_with_learned_positive_under_predeclared_rule": learned_positive_count,
            "scenarios_with_classical_best_on_primary": classical_best_count,
            "verdict": (
                "observed-step endpoint-fixation support confirms the disclosed "
                "post-hoc ordering: no learned positive under the fixed "
                "rule on any independent scenario"
                if learned_positive_count == 0
                else "learned positive observed under the predeclared rule"
            ),
            "k8_primary_summary": {
                "n_realizations": 8,
                "load_bearing_endpoint": (
                    "K=8 endpoint-fixation support result is the supporting "
                    "endpoint check; K=16 strict-extension result is disclosed as "
                    "a transparent post-hoc sensitivity, not as a "
                    "confirmatory pre-registered endpoint"
                ),
                "scenarios_with_learned_positive_under_predeclared_rule": sum(
                    1
                    for r in scenario_rows
                    if r.get("primary_k8_predeclared")
                    and r["primary_k8_predeclared"]["learned_positive_under_predeclared_rule"]
                ),
                "scenarios_with_classical_best_on_primary": sum(
                    1
                    for r in scenario_rows
                    if r.get("primary_k8_predeclared")
                    and r["primary_k8_predeclared"]["best_method_primary"] in classical_keys
                ),
            },
        },
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {out_path} ({n} scenarios, K={K})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
