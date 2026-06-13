#!/usr/bin/env python
"""Run targeted station-outage robustness analysis."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import (
    parse_baseline_config,
    relative_improvement_percent,
    run_filter_baselines,
    run_model_inference,
    score_predictions,
)
from gnn_state_estimation.models import TemporalGraphEstimator
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import dump_json, load_yaml


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_model(
    checkpoint: Path,
    train_cfg,
    use_prior_default: bool,
    device: torch.device,
) -> TemporalGraphEstimator:
    ckpt = torch.load(checkpoint, map_location=device)
    model_kwargs = ckpt.get("model_kwargs", {})
    use_prior = bool(ckpt.get("use_ekf_prior", use_prior_default))
    model = TemporalGraphEstimator(
        hidden_dim=train_cfg.hidden_dim,
        gnn_layers=train_cfg.gnn_layers,
        gru_layers=train_cfg.gru_layers,
        dropout=train_cfg.dropout,
        use_ekf_prior=use_prior,
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--gnn-checkpoint", type=str, default=None)
    p.add_argument("--hybrid-checkpoint", type=str, default=None)
    p.add_argument("--scenarios", type=str, default="test,stress_test")
    p.add_argument("--topk-single-outages", type=int, default=4)
    p.add_argument("--max-trajectories", type=int, default=24)
    p.add_argument("--seed", type=int, default=31415)
    p.add_argument("--output-csv", type=str, default="results/station_outage_sweep/station_outage_grid.csv")
    p.add_argument("--summary-csv", type=str, default="results/station_outage_sweep/station_outage_summary.csv")
    p.add_argument("--summary-json", type=str, default="results/station_outage_sweep/station_outage_summary.json")
    p.add_argument("--skip-gnn", action="store_true")
    p.add_argument("--skip-hybrid", action="store_true")
    return p


def parse_scenarios(text: str) -> list[str]:
    out = [t.strip() for t in text.split(",") if t.strip()]
    valid = {"test", "stress_test"}
    for s in out:
        if s not in valid:
            raise ValueError(f"Unsupported scenario '{s}'. Expected one of {sorted(valid)}.")
    return out


def select_outage_patterns(
    visibility: np.ndarray,
    station_names: list[str],
    topk: int,
) -> list[dict[str, Any]]:
    # Rank stations by empirical visibility frequency to target the most informative outages first.
    vis_rate = np.mean(visibility, axis=(0, 1))
    order = list(np.argsort(-vis_rate))
    n = len(station_names)

    patterns: list[dict[str, Any]] = [{"pattern_label": "No outage", "drop_idx": []}]
    k = min(max(int(topk), 0), n)
    for j in range(k):
        idx = int(order[j])
        patterns.append(
            {
                "pattern_label": f"Drop {station_names[idx]}",
                "drop_idx": [idx],
            }
        )

    if n >= 2:
        pair = [int(order[0]), int(order[1])]
        patterns.append({"pattern_label": "Drop top-2 visible", "drop_idx": pair})
    if n >= 3:
        triple = [int(order[0]), int(order[1]), int(order[2])]
        patterns.append({"pattern_label": "Drop top-3 visible", "drop_idx": triple})

    # Remove accidental duplicates while preserving order.
    uniq: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for p in patterns:
        key = tuple(sorted(int(i) for i in p["drop_idx"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def apply_station_outage(
    measurements: np.ndarray,
    visibility: np.ndarray,
    drop_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    meas = np.array(measurements, copy=True)
    vis = np.array(visibility, copy=True)
    if drop_idx:
        vis[:, :, drop_idx] = 0
        meas[:, :, drop_idx, :] = 0.0
    return meas, vis


def select_subset_indices(n_total: int, max_trajectories: int, seed: int) -> np.ndarray:
    if max_trajectories <= 0 or max_trajectories >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=int(max_trajectories), replace=False)
    return np.sort(idx.astype(np.int64))


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    scenarios = parse_scenarios(args.scenarios)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    station_names = [str(s["name"]) for s in cfg["simulation"]["stations"]]

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    gnn_ckpt = Path(args.gnn_checkpoint) if args.gnn_checkpoint else ckpt_dir / "best_gnn.pt"
    hybrid_ckpt = Path(args.hybrid_checkpoint) if args.hybrid_checkpoint else ckpt_dir / "best_hybrid.pt"
    models: dict[str, TemporalGraphEstimator] = {}
    if not args.skip_gnn and gnn_ckpt.exists():
        models["GNN"] = load_model(gnn_ckpt, train_cfg, use_prior_default=False, device=device)
    if not args.skip_hybrid and hybrid_ckpt.exists():
        models["HybridGNN"] = load_model(hybrid_ckpt, train_cfg, use_prior_default=True, device=device)

    data_dir = Path(cfg["data"]["output_dir"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    pattern_source = "test" if "test" in scenarios else scenarios[0]
    pattern_npz = data_dir / ("test.npz" if pattern_source == "test" else "stress_test.npz")
    pattern_arr = load_dataset_npz(pattern_npz)
    pattern_idx = select_subset_indices(
        n_total=int(pattern_arr.states.shape[0]),
        max_trajectories=int(args.max_trajectories),
        seed=int(args.seed) + 101,
    )
    pattern_visibility = pattern_arr.visibility[pattern_idx] if pattern_idx.size < pattern_arr.states.shape[0] else pattern_arr.visibility
    patterns = select_outage_patterns(
        visibility=pattern_visibility,
        station_names=station_names,
        topk=args.topk_single_outages,
    )

    method_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    patterns_json: dict[str, Any] = {}
    subset_indices_json: dict[str, list[int]] = {}

    for scenario_idx, scenario in enumerate(scenarios):
        scenario_label = "Test" if scenario == "test" else "Stress"
        npz_path = data_dir / ("test.npz" if scenario == "test" else "stress_test.npz")
        arr = load_dataset_npz(npz_path)
        states = arr.states
        measurements = arr.measurements
        visibility = arr.visibility
        times = arr.times
        x0_estimates = arr.x0_estimates

        n_total = states.shape[0]
        subset_idx = select_subset_indices(
            n_total=n_total,
            max_trajectories=int(args.max_trajectories),
            seed=int(args.seed) + scenario_idx * 1009 + 17,
        )
        subset_indices_json[scenario] = [int(i) for i in subset_idx.tolist()]
        if subset_idx.size < n_total:
            states = states[subset_idx]
            measurements = measurements[subset_idx]
            visibility = visibility[subset_idx]
            times = times[subset_idx]
            if x0_estimates is not None:
                x0_estimates = x0_estimates[subset_idx]
        n_use = int(states.shape[0])

        patterns_json[scenario] = [
            {
                "pattern_label": p["pattern_label"],
                "drop_idx": [int(i) for i in p["drop_idx"]],
                "drop_names": [station_names[int(i)] for i in p["drop_idx"]],
            }
            for p in patterns
        ]

        if scenario == "test":
            sim_cfg = cfg["simulation"]
        else:
            sim_cfg = deep_update(cfg["simulation"], cfg.get("stress_simulation_overrides", {}))
        ds_cfg = parse_dataset_config(sim_cfg)

        for pattern_idx, pattern in enumerate(patterns):
            drop_idx = [int(i) for i in pattern["drop_idx"]]
            meas_drop, vis_drop = apply_station_outage(measurements=measurements, visibility=visibility, drop_idx=drop_idx)

            filters = run_filter_baselines(
                states=states,
                measurements=meas_drop,
                visibility=vis_drop,
                times=times,
                dataset_cfg=ds_cfg,
                baseline_cfg=baseline_cfg,
                seed=int(args.seed) + scenario_idx * 2000 + pattern_idx * 31,
                x0_estimates=x0_estimates,
            )

            preds: dict[str, np.ndarray] = {"EKF": filters["ekf"], "UKF": filters["ukf"]}
            if "aukf" in filters:
                preds["AUKF"] = filters["aukf"]
            if "GNN" in models:
                preds["GNN"] = run_model_inference(
                    model=models["GNN"],
                    states=states,
                    measurements=meas_drop,
                    visibility=vis_drop,
                    station_ecef=arr.station_ecef,
                    window_size=train_cfg.window_size,
                )
            if "HybridGNN" in models:
                preds["HybridGNN"] = run_model_inference(
                    model=models["HybridGNN"],
                    states=states,
                    measurements=meas_drop,
                    visibility=vis_drop,
                    station_ecef=arr.station_ecef,
                    window_size=train_cfg.window_size,
                    ekf_prior=filters["ekf"],
                )

            metric_by_method: dict[str, dict[str, float]] = {}
            for method, pred in preds.items():
                metric_by_method[method] = score_predictions(states[:, eval_start:], pred[:, eval_start:])

            ukf_pos = float(metric_by_method["UKF"]["pos_rmse_m"])
            vis_count_eval = vis_drop[:, eval_start:, :].sum(axis=2)
            mean_visible = float(np.mean(vis_count_eval))
            frac_zero_visible = float(np.mean(vis_count_eval == 0))
            dropped_names = [station_names[i] for i in drop_idx]
            dropped_name_text = ",".join(dropped_names) if dropped_names else "None"
            for method, m in metric_by_method.items():
                imp = 0.0 if method == "UKF" else relative_improvement_percent(ukf_pos, float(m["pos_rmse_m"]))
                method_rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario_label,
                        "pattern_label": pattern["pattern_label"],
                        "num_dropped": int(len(drop_idx)),
                        "dropped_stations": dropped_name_text,
                        "method": method,
                        "pos_rmse_m": float(m["pos_rmse_m"]),
                        "vel_rmse_mps": float(m["vel_rmse_mps"]),
                        "improvement_vs_ukf_percent": float(imp),
                        "mean_visible_stations_per_step": mean_visible,
                        "fraction_zero_visibility": frac_zero_visible,
                        "n_trajectories": int(n_use),
                    }
                )

            hybrid_pos = float(metric_by_method["HybridGNN"]["pos_rmse_m"]) if "HybridGNN" in metric_by_method else float("nan")
            aukf_pos = float(metric_by_method["AUKF"]["pos_rmse_m"]) if "AUKF" in metric_by_method else float("nan")
            summary_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": scenario_label,
                    "pattern_label": pattern["pattern_label"],
                    "num_dropped": int(len(drop_idx)),
                    "dropped_stations": dropped_name_text,
                    "mean_visible_stations_per_step": mean_visible,
                    "fraction_zero_visibility": frac_zero_visible,
                    "ukf_pos_rmse_m": ukf_pos,
                    "ekf_pos_rmse_m": float(metric_by_method["EKF"]["pos_rmse_m"]),
                    "aukf_pos_rmse_m": aukf_pos,
                    "hybrid_pos_rmse_m": hybrid_pos,
                    "hybrid_vs_ukf_percent": relative_improvement_percent(ukf_pos, hybrid_pos)
                    if np.isfinite(hybrid_pos)
                    else float("nan"),
                    "aukf_vs_ukf_percent": relative_improvement_percent(ukf_pos, aukf_pos)
                    if np.isfinite(aukf_pos)
                    else float("nan"),
                    "hybrid_vs_aukf_percent": relative_improvement_percent(aukf_pos, hybrid_pos)
                    if np.isfinite(aukf_pos) and np.isfinite(hybrid_pos)
                    else float("nan"),
                    "n_trajectories": int(n_use),
                }
            )
            print(
                f"[{scenario}] {pattern['pattern_label']}: "
                f"Hybrid-vs-UKF={summary_rows[-1]['hybrid_vs_ukf_percent']:.2f}% | "
                f"AUKF-vs-UKF={summary_rows[-1]['aukf_vs_ukf_percent']:.2f}%"
            )

    method_df = pd.DataFrame(method_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "num_dropped", "pattern_label"])
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    method_df.to_csv(out_csv, index=False)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(args.summary_csv), index=False)

    stress_df = summary_df[summary_df["scenario"] == "stress_test"]
    test_df = summary_df[summary_df["scenario"] == "test"]
    out_json = {
        "patterns": patterns_json,
        "pattern_source_scenario": pattern_source,
        "subset_indices": subset_indices_json,
        "n_rows": int(summary_df.shape[0]),
        "n_method_rows": int(method_df.shape[0]),
        "stress_hybrid_vs_ukf_mean_percent": float(stress_df["hybrid_vs_ukf_percent"].mean())
        if not stress_df.empty
        else float("nan"),
        "stress_hybrid_vs_ukf_min_percent": float(stress_df["hybrid_vs_ukf_percent"].min())
        if not stress_df.empty
        else float("nan"),
        "stress_hybrid_vs_ukf_max_percent": float(stress_df["hybrid_vs_ukf_percent"].max())
        if not stress_df.empty
        else float("nan"),
        "test_hybrid_vs_ukf_mean_percent": float(test_df["hybrid_vs_ukf_percent"].mean()) if not test_df.empty else float("nan"),
        "test_hybrid_vs_ukf_min_percent": float(test_df["hybrid_vs_ukf_percent"].min()) if not test_df.empty else float("nan"),
        "test_hybrid_vs_ukf_max_percent": float(test_df["hybrid_vs_ukf_percent"].max()) if not test_df.empty else float("nan"),
    }
    dump_json(out_json, Path(args.summary_json))
    print("Station outage sweep complete.")


if __name__ == "__main__":
    main()
