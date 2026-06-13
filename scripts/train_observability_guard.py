#!/usr/bin/env python
"""Train an observability-aware guard over candidate state estimators."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gnn_state_estimation.benchmark_tasks import DEFAULT_FEATURE_NAMES, compute_trajectory_feature_frame
from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.guarded_selection import (
    GuardedSelectorConfig,
    build_cost_matrix,
    evaluate_selection,
    fit_guarded_selector,
    predict_costs,
    save_guarded_selector,
)
from gnn_state_estimation.observation_replay import apply_public_observation_station_bank
from gnn_state_estimation.observability import compute_observability_context_features, stations_from_ecef
from gnn_state_estimation.public_data import apply_public_station_selection
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.utils.io import dump_json, load_yaml


DEFAULT_METHODS = (
    "EKF",
    "UKF",
    "AUKF",
    "KalmanNetLike",
    "NoGraphResidual",
    "LearnedNoiseAdaptive",
    "HybridGNN",
    "InnovationHybridGNN",
    "ObservabilityContextHybridGNN",
)

OBS_TRAJECTORY_FEATURES = (
    "obs_log_trace",
    "obs_log_pdet",
    "obs_log_condition",
    "obs_rank_fraction",
    "obs_visible_fraction",
    "obs_min_eig_log",
)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def scenario_sim_config(base_sim: dict[str, Any], scenario_cfg: dict[str, Any]) -> dict[str, Any]:
    kind = str(scenario_cfg.get("kind", "synthetic"))
    sim_cfg = deep_update(base_sim, scenario_cfg.get("overrides", {}))
    if kind == "public_catalog_replay":
        sim_cfg, _ = apply_public_station_selection(sim_cfg, scenario_cfg)
    elif kind == "public_observation_replay":
        sim_cfg, _ = apply_public_observation_station_bank(sim_cfg, scenario_cfg)
    return sim_cfg


def scenario_config_map(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scenarios = {
        "train": cfg["simulation"],
        "val": cfg["simulation"],
        "test": cfg["simulation"],
        "stress_train": deep_update(cfg["simulation"], cfg["stress_simulation_overrides"]),
        "stress_val": deep_update(cfg["simulation"], cfg["stress_simulation_overrides"]),
        "stress_test": deep_update(cfg["simulation"], cfg["stress_simulation_overrides"]),
    }
    for scenario_name, scenario_spec in cfg.get("benchmark_suite", {}).get("scenarios", {}).items():
        scenarios[scenario_name] = scenario_sim_config(cfg["simulation"], scenario_spec)
    return scenarios


def parse_list(value: str | list[str] | None, fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def add_observability_trajectory_features(
    frame: pd.DataFrame,
    *,
    arrays,
    dataset_cfg,
    eval_start: int,
) -> pd.DataFrame:
    out = frame.copy()
    if arrays.ekf_prior is None:
        for name in OBS_TRAJECTORY_FEATURES:
            for reducer in ("mean", "std", "min", "max"):
                out[f"{name}_{reducer}"] = 0.0
        return out

    stations = dataset_cfg.stations
    if len(stations) != int(arrays.station_ecef.shape[0]):
        stations = stations_from_ecef(arrays.station_ecef)
    obs = compute_observability_context_features(
        prior_states=arrays.ekf_prior,
        visibility=arrays.visibility,
        times_s=arrays.times,
        stations=stations,
        meas_std_vector=dataset_cfg.measurement_noise.std_vector,
    )
    obs_eval = obs[:, eval_start:, :]
    for idx, name in enumerate(OBS_TRAJECTORY_FEATURES):
        channel = obs_eval[:, :, idx]
        out[f"{name}_mean"] = channel.mean(axis=1)
        out[f"{name}_std"] = channel.std(axis=1)
        out[f"{name}_min"] = channel.min(axis=1)
        out[f"{name}_max"] = channel.max(axis=1)
    return out


def build_feature_frames(
    *,
    cfg: dict[str, Any],
    data_dir: Path,
    scenarios: list[str],
    eval_start: int,
) -> pd.DataFrame:
    scenario_cfgs = scenario_config_map(cfg)
    frames: list[pd.DataFrame] = []
    for scenario_name in scenarios:
        arrays = load_dataset_npz(data_dir / f"{scenario_name}.npz")
        base = compute_trajectory_feature_frame(arrays, scenario_name=scenario_name, eval_start=eval_start)
        if scenario_name not in scenario_cfgs:
            raise ValueError(f"No simulation config available for scenario {scenario_name!r}.")
        ds_cfg = parse_dataset_config(scenario_cfgs[scenario_name])
        frames.append(
            add_observability_trajectory_features(
                base,
                arrays=arrays,
                dataset_cfg=ds_cfg,
                eval_start=eval_start,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def method_aggregate_summary(
    trajectory_errors: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    scopes: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_df = trajectory_errors[trajectory_errors["method"] == method]
        for scope in scopes:
            group = method_df if scope == "combined" else method_df[method_df["scenario"] == scope]
            if group.empty:
                continue
            pos = group["traj_pos_rmse_m"].to_numpy(dtype=np.float64)
            vel = group["traj_vel_rmse_mps"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "method": method,
                    "scope": scope,
                    "n_trajectories": int(group.shape[0]),
                    "aggregate_pos_rmse_m": float(np.sqrt(np.mean(pos**2))),
                    "mean_traj_pos_rmse_m": float(np.mean(pos)),
                    "aggregate_vel_rmse_mps": float(np.sqrt(np.mean(vel**2))),
                    "mean_traj_vel_rmse_mps": float(np.mean(vel)),
                }
            )
    return pd.DataFrame(rows)


def learn_public_observation_vetoes(
    trajectory_errors: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    train_features: pd.DataFrame,
    max_divergence_rate: float,
    divergence_threshold_m: float,
    velocity_threshold_mps: float,
) -> tuple[str, ...]:
    public_keys = train_features.loc[
        train_features["is_public_observation"].to_numpy(dtype=np.float64) > 0.5,
        ["scenario", "traj_id"],
    ].drop_duplicates()
    if public_keys.empty:
        return tuple()
    vetoed: list[str] = []
    for method in methods:
        method_df = trajectory_errors[trajectory_errors["method"] == method].copy()
        merged = public_keys.merge(method_df, on=["scenario", "traj_id"], how="left", validate="one_to_one")
        if merged.empty:
            continue
        pos = merged["traj_pos_rmse_m"].to_numpy(dtype=np.float64)
        vel = merged["traj_vel_rmse_mps"].to_numpy(dtype=np.float64)
        bad = (~np.isfinite(pos)) | (~np.isfinite(vel)) | (pos > divergence_threshold_m) | (vel > velocity_threshold_mps)
        if float(np.mean(bad)) > max_divergence_rate:
            vetoed.append(method)
    return tuple(vetoed)


def select_with_vetoes(
    selector,
    feature_df: pd.DataFrame,
    *,
    public_observation_vetoes: tuple[str, ...],
    divergence_penalty_m: float,
) -> pd.DataFrame:
    costs = predict_costs(selector, feature_df)
    if public_observation_vetoes:
        public_mask = feature_df["is_public_observation"].to_numpy(dtype=np.float64) > 0.5
        for method in public_observation_vetoes:
            if method in selector.methods:
                method_idx = selector.methods.index(method)
                costs[public_mask, method_idx] = divergence_penalty_m
    idx = np.argmin(costs, axis=1)
    out = feature_df.loc[:, ["scenario", "traj_id"]].copy()
    out["selected_method"] = np.asarray(selector.methods, dtype=object)[idx]
    out["predicted_cost_m"] = costs[np.arange(costs.shape[0]), idx]
    for method_idx, method in enumerate(selector.methods):
        out[f"predicted_cost_{method}_m"] = costs[:, method_idx]
    return out


def write_latex_table(summary: pd.DataFrame, counts: pd.DataFrame, path: Path) -> None:
    rows = []
    display_scopes = [scope for scope in summary["scope"].tolist() if scope != "combined"]
    for scope in display_scopes:
        row = summary[summary["scope"] == scope].iloc[0]
        counts_scope = counts[counts["scope"] == scope]
        count_text = ", ".join(
            f"{item.selected_method}: {int(item.count)}" for item in counts_scope.itertuples(index=False)
        )
        rows.append(
            " & ".join(
                [
                    scope.replace("_", "\\_"),
                    f"{row.aggregate_pos_rmse_m:.2f}",
                    f"{row.mean_traj_pos_rmse_m:.2f}",
                    f"{row.divergence_rate:.3f}",
                    count_text.replace("_", "\\_"),
                ]
            )
            + r" \\"
        )
    body = "\n".join(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{lrrrl}",
                r"\toprule",
                r"Scenario & Aggregate RMSE [m] & Mean trajectory RMSE [m] & Divergence rate & Selected methods \\",
                r"\midrule",
                body,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--trajectory-path", type=str, default="results/observability_guard/candidate_trajectory_errors.csv")
    parser.add_argument("--output-dir", type=str, default="results/observability_guard")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--train-scenarios", type=str, default=None)
    parser.add_argument("--eval-scenarios", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-public-veto", action="store_true")
    parser.add_argument("--public-veto-max-divergence-rate", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_errors = pd.read_csv(args.trajectory_path)
    methods = tuple(parse_list(args.methods, list(DEFAULT_METHODS)))
    available_methods = set(str(name) for name in trajectory_errors["method"].unique())
    missing_methods = sorted(set(methods).difference(available_methods))
    if missing_methods:
        raise ValueError(f"Trajectory file is missing candidate methods: {missing_methods}")

    benchmark_cfg = cfg.get("benchmark_tasks", {})
    train_scenarios = parse_list(
        args.train_scenarios,
        benchmark_cfg.get("task_train_scenarios", ["test", "stress_test", "satnogs_observation_replay_val"]),
    )
    eval_scenarios = parse_list(
        args.eval_scenarios,
        ["test", "stress_test", "satnogs_observation_replay_test", "satnogs_observation_replay_stress_test"],
    )
    all_scenarios = list(dict.fromkeys(train_scenarios + eval_scenarios))
    eval_start = max(int(cfg["training"]["window_size"]) - 1, 0)
    data_dir = Path(cfg["data"]["output_dir"])

    features = build_feature_frames(cfg=cfg, data_dir=data_dir, scenarios=all_scenarios, eval_start=eval_start)
    feature_names = tuple(
        name
        for name in features.columns
        if name not in {"scenario", "traj_id"}
        and pd.api.types.is_numeric_dtype(features[name])
    )

    train_features = features[features["scenario"].isin(train_scenarios)].reset_index(drop=True)
    train_traj = trajectory_errors[trajectory_errors["scenario"].isin(train_scenarios)].reset_index(drop=True)
    train_aligned, train_costs = build_cost_matrix(
        train_features,
        train_traj,
        methods=methods,
        divergence_penalty_m=float(cfg.get("benchmark_tasks", {}).get("selector_divergence_penalty_m", 1.0e8)),
    )

    selector_cfg = GuardedSelectorConfig(
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        divergence_penalty_m=float(cfg.get("benchmark_tasks", {}).get("selector_divergence_penalty_m", 1.0e8)),
    )
    selector = fit_guarded_selector(
        train_aligned,
        train_costs,
        methods=methods,
        feature_names=feature_names,
        config=selector_cfg,
        device=torch.device(args.device),
    )

    eval_features = features[features["scenario"].isin(eval_scenarios)].reset_index(drop=True)
    public_vetoes = tuple()
    if not args.disable_public_veto:
        public_vetoes = learn_public_observation_vetoes(
            train_traj,
            methods=methods,
            train_features=train_features,
            max_divergence_rate=float(args.public_veto_max_divergence_rate),
            divergence_threshold_m=float(selector_cfg.divergence_penalty_m),
            velocity_threshold_mps=float(cfg.get("benchmark_tasks", {}).get("divergence_thresholds", {}).get("vel_rmse_mps", 1.0e5)),
        )
    selection = select_with_vetoes(
        selector,
        eval_features,
        public_observation_vetoes=public_vetoes,
        divergence_penalty_m=float(selector_cfg.divergence_penalty_m),
    )
    eval_traj = trajectory_errors[trajectory_errors["scenario"].isin(eval_scenarios)].reset_index(drop=True)
    detail, guarded_summary = evaluate_selection(selection, eval_traj)
    guarded_summary.insert(0, "method", "GuardedObservabilitySelector")

    scopes = ["combined"] + eval_scenarios
    candidate_summary = method_aggregate_summary(eval_traj, methods=methods, scopes=scopes)
    combined_summary = pd.concat([candidate_summary, guarded_summary], ignore_index=True)

    counts = (
        detail.groupby(["scenario", "selected_method"], as_index=False)
        .size()
        .rename(columns={"scenario": "scope", "size": "count"})
    )
    combined_counts = (
        detail.groupby(["selected_method"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .assign(scope="combined")
    )
    counts = pd.concat([combined_counts.loc[:, ["scope", "selected_method", "count"]], counts], ignore_index=True)

    save_guarded_selector(selector, output_dir / "guarded_observability_selector.pt")
    selector.history.to_csv(output_dir / "guarded_selector_training_history.csv", index=False)
    train_aligned.to_csv(output_dir / "guarded_selector_train_features.csv", index=False)
    selection.to_csv(output_dir / "guarded_selector_selection.csv", index=False)
    detail.to_csv(output_dir / "guarded_selector_details.csv", index=False)
    guarded_summary.to_csv(output_dir / "guarded_selector_metrics.csv", index=False)
    combined_summary.to_csv(output_dir / "guarded_selector_candidate_comparison.csv", index=False)
    counts.to_csv(output_dir / "guarded_selector_counts.csv", index=False)
    write_latex_table(
        guarded_summary[guarded_summary["scope"].isin(eval_scenarios)],
        counts,
        Path("paper/tables/observability_guard.tex"),
    )

    payload = {
        "methods": list(methods),
        "feature_names": list(feature_names),
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
        "selector_config": asdict(selector_cfg),
        "public_observation_veto_methods": list(public_vetoes),
        "checkpoint": str(output_dir / "guarded_observability_selector.pt"),
        "history_last": selector.history.tail(1).to_dict(orient="records"),
        "summary": guarded_summary.to_dict(orient="records"),
        "selection_counts": counts.to_dict(orient="records"),
    }
    dump_json(payload, output_dir / "guarded_selector_summary.json")
    print(f"Guarded selector trained with {len(feature_names)} features and {len(methods)} candidate methods.")


if __name__ == "__main__":
    main()
