"""Loop-24 M4: scenario-level resampling with *genuine independent realizations*.

Reviewer M4 objected that the previous scenario-resampling table was derived
offline from a single primary seed by pooling visibility buckets, so the
"scenario population" was not an independent draw. This script replaces that
with a real sampling design: for each deterministic scenario it generates K
*independent* realizations -- each an independently seeded trajectory
population with its own measurement-noise draw -- runs the recursive classical
filters (EKF/UKF/AUKF, no training) and the trained RGR-GF residual model on
every realization, and reports the per-scenario mean and standard deviation of
the primary observed-step position RMSE over the K realizations. The scenario
is the statistical unit and the per-scenario estimate now carries real
realization-level sampling variability rather than a single deterministic
number.

No model is retrained. RGR-GF is the released ``best_hybrid.pt`` checkpoint run
in inference on each fresh realization, so the learned/classical comparison is
apples-to-apples on the same independent draws.
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
import math
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
OUT_DIR = ROOT / "results" / "scenario_resampling"
OUT_PATH = OUT_DIR / "scenario_resampling.json"

SCENARIOS = [
    ("test", "Nominal", "nominal optical/radio sparse-visibility"),
    ("stress_test", "Measurement-noise stress", "inflated measurement noise/outliers"),
    ("high_drag_test", "High drag", "inflated ballistic coefficient + process noise"),
    ("process_noise_shift_test", "Process-noise shift", "elevated process-noise std"),
    ("maneuver_shift_test", "Maneuver-like shift", "large process noise + dropout/outliers"),
    ("low_inclination_test", "Low inclination", "low-inclination orbital regime"),
    ("sunsync_like_test", "Sun-synchronous-like", "sun-synchronous-like geometry"),
    ("high_inclination_test", "High inclination", "high-inclination orbital regime"),
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


def observed_step_pos_rmse(states, preds, visibility, eval_start) -> float:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    err = states[:, eval_start:][observed, :3] - preds[:, eval_start:][observed, :3]
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--num-realizations", type=int, default=5)
    ap.add_argument("--trajectories", type=int, default=24)
    ap.add_argument("--base-seed", type=int, default=90000)
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

    scenario_rows = []
    for s_idx, (key, label, regime) in enumerate(SCENARIOS):
        est_sim, truth_sim = resolve_sim_configs(cfg, key)
        truth_dc = parse_dataset_config(truth_sim)
        est_dc = parse_dataset_config(est_sim)
        per_real = {m: [] for m in method_keys}
        for r in range(K):
            seed = args.base_seed + 1000 * (s_idx + 1) + r
            data = generate_dataset(truth_dc, N, seed=seed)
            states = data["states"]
            meas = data["measurements"]
            vis = data["visibility"]
            times = data["times"]
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
                per_real[m].append(
                    observed_step_pos_rmse(states, preds[m], vis, eval_start)
                )

        means = {m: float(np.mean(per_real[m])) for m in method_keys}
        stds = {m: float(np.std(per_real[m], ddof=1)) for m in method_keys}
        rgr = means["RGR-GF"]
        best_method = min(means, key=means.get)
        scenario_rows.append(
            {
                "name": key,
                "label": label,
                "regime": regime,
                "n_realizations": K,
                "trajectories_per_realization": N,
                "ekf_obs_pos_rmse_m": round(means["EKF"], 2),
                "ukf_obs_pos_rmse_m": round(means["UKF"], 2),
                "aukf_obs_pos_rmse_m": round(means["AUKF"], 2),
                "rgr_gf_obs_pos_rmse_m": round(rgr, 2),
                "rgr_gf_obs_pos_rmse_std_m": round(stds["RGR-GF"], 2),
                "ekf_obs_pos_rmse_std_m": round(stds["EKF"], 2),
                "aukf_obs_pos_rmse_std_m": round(stds["AUKF"], 2),
                "rgr_minus_ekf_m": round(rgr - means["EKF"], 2),
                "rgr_minus_ukf_m": round(rgr - means["UKF"], 2),
                "rgr_minus_aukf_m": round(rgr - means["AUKF"], 2),
                "best_method": best_method,
            }
        )
        print(
            f"{label}: best={best_method} "
            f"EKF={means['EKF']:.1f} UKF={means['UKF']:.1f} "
            f"AUKF={means['AUKF']:.1f} RGR-GF={rgr:.1f} (K={K})",
            flush=True,
        )

    n = len(scenario_rows)
    beats = {
        "aukf": sum(1 for r in scenario_rows if r["rgr_minus_aukf_m"] < 0),
        "ekf": sum(1 for r in scenario_rows if r["rgr_minus_ekf_m"] < 0),
        "ukf": sum(1 for r in scenario_rows if r["rgr_minus_ukf_m"] < 0),
    }
    best_counts = {m: 0 for m in method_keys}
    for r in scenario_rows:
        best_counts[r["best_method"]] += 1

    def mean(field: str) -> float:
        return round(sum(r[field] for r in scenario_rows) / n, 2) if n else float("nan")

    result = {
        "status": "completed",
        "schema_version": "scenario_resampling_v2",
        "primary_metric": "observed_step_position_rmse_m",
        "statistical_unit": (
            f"deterministic scenario; per-scenario estimate is the mean over "
            f"{int(args.num_realizations)} independent realizations "
            f"(independent trajectory populations and measurement-noise draws)"
        ),
        "source": (
            "independent realizations generated and scored at build time "
            "(no offline bucket pooling; classical filters + released RGR-GF "
            "checkpoint in inference)"
        ),
        "num_scenarios": n,
        "num_realizations_per_scenario": int(args.num_realizations),
        "trajectories_per_realization": int(args.trajectories),
        "scenarios": scenario_rows,
        "summary": {
            "n_scenarios": n,
            "rgr_gf_beats_aukf_scenarios": beats["aukf"],
            "rgr_gf_beats_ekf_scenarios": beats["ekf"],
            "rgr_gf_beats_ukf_scenarios": beats["ukf"],
            "mean_rgr_minus_aukf_m": mean("rgr_minus_aukf_m"),
            "mean_rgr_minus_ekf_m": mean("rgr_minus_ekf_m"),
            "mean_rgr_minus_ukf_m": mean("rgr_minus_ukf_m"),
            "best_method_scenario_counts": best_counts,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} ({n} scenarios, K={args.num_realizations})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
