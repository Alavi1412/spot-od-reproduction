#!/usr/bin/env python
"""Build SPOT-OD public benchmark task artifacts from released experiments."""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from gnn_state_estimation.benchmark_tasks import (
    DEFAULT_SELECTOR_METHODS,
    build_oracle_selector_targets,
    build_packet_registry,
    build_selection_detail_frame,
    build_selector_outputs,
    build_stability_labels,
    compute_trajectory_feature_frame,
    fit_stability_models,
    prepare_method_feature_frame,
    score_stability_predictors,
)
from gnn_state_estimation.utils.io import dump_json, load_yaml


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_if_empty(value: list[str] | None, fallback: list[str]) -> list[str]:
    return list(value) if value else list(fallback)


def ensure_training_eval_artifacts(
    *,
    config_path: Path,
    device: str,
    scenarios: list[str],
    output_dir: Path,
    refresh: bool,
) -> Path:
    traj_path = output_dir / "training_trajectory_errors.csv"
    if traj_path.exists() and not refresh:
        return traj_path
    if not scenarios:
        return traj_path

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/evaluate_models.py",
        "--config",
        str(config_path),
        "--device",
        device,
        "--scenarios",
        ",".join(scenarios),
        "--metrics-path",
        str(output_dir / "training_metrics.json"),
        "--per-step-path",
        str(output_dir / "training_per_step.csv"),
        "--trajectory-path",
        str(traj_path),
        "--improvement-path",
        str(output_dir / "training_improvement.csv"),
        "--calibration-path",
        str(output_dir / "training_calibration.csv"),
        "--predictions-path",
        str(output_dir / "training_predictions.npz"),
        "--figure-dir",
        str(output_dir / "figures"),
        "--scorecard-out",
        str(output_dir / "training_scorecard.json"),
        "--calibration-bootstrap-samples",
        "0",
    ]
    subprocess.run(command, check=True, cwd=config_path.parent.parent)
    return traj_path


def build_feature_frame_for_scenarios(
    *,
    data_dir: Path,
    scenario_names: list[str],
    eval_start: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario_name in scenario_names:
        arrays = load_lightweight_dataset_npz(data_dir / f"{scenario_name}.npz")
        frames.append(compute_trajectory_feature_frame(arrays, scenario_name=scenario_name, eval_start=eval_start))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_task_definition(cfg: dict[str, Any]) -> dict[str, Any]:
    task_cfg = cfg.get("benchmark_tasks", {})
    return {
        "benchmark_name": task_cfg.get("name", "SPOT-OD"),
        "tasks": [
            {
                "name": "state_estimation",
                "train_scenarios": task_cfg.get("state_estimation_train_scenarios", []),
                "eval_scenarios": task_cfg.get("state_estimation_scenarios", []),
                "metrics": ["position_rmse_m", "velocity_rmse_m", "divergence_rate"],
            },
            {
                "name": "stability_prediction",
                "train_scenarios": task_cfg.get("task_train_scenarios", []),
                "eval_scenarios": task_cfg.get("task_eval_scenarios", []),
                "metrics": ["auroc", "auprc", "brier"],
            },
            {
                "name": "method_selection",
                "train_scenarios": task_cfg.get("task_train_scenarios", []),
                "eval_scenarios": task_cfg.get("task_eval_scenarios", []),
                "metrics": ["mean_regret_m", "divergence_avoidance_rate", "oracle_match_rate"],
            },
        ],
        "selector_methods": task_cfg.get("selector_methods", list(DEFAULT_SELECTOR_METHODS)),
    }


def load_lightweight_dataset_npz(path: Path) -> SimpleNamespace:
    with np.load(path) as data:
        return SimpleNamespace(
            visibility=data["visibility"],
            innovation_features=data["innovation_features"] if "innovation_features" in data.files else None,
            ekf_prior=data["ekf_prior"] if "ekf_prior" in data.files else None,
            ukf_prior=data["ukf_prior"] if "ukf_prior" in data.files else None,
            aukf_prior=data["aukf_prior"] if "aukf_prior" in data.files else None,
        )


def build_summary_payload(
    *,
    task_definition: dict[str, Any],
    packet_registry: dict[str, Any],
    stability_summary: pd.DataFrame,
    selection_summary: pd.DataFrame,
) -> dict[str, Any]:
    top_stability = (
        stability_summary[(stability_summary["method"] == "ALL") & (stability_summary["scope"] == "combined")]
        .sort_values(["auroc", "auprc"], ascending=[False, False])
        .head(1)
    )
    top_selector = (
        selection_summary[selection_summary["scope"] == "combined"]
        .sort_values(["mean_regret_m", "divergence_avoidance_rate"], ascending=[True, False])
        .head(1)
    )
    return {
        "benchmark_name": task_definition.get("benchmark_name"),
        "top_stability_baseline": top_stability.to_dict(orient="records")[0] if not top_stability.empty else {},
        "top_selector_baseline": top_selector.to_dict(orient="records")[0] if not top_selector.empty else {},
        "packet_registry": packet_registry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/benchmark_tasks")
    parser.add_argument("--refresh-training-eval", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    benchmark_cfg = cfg.get("benchmark_tasks", {})
    selector_methods = tuple(benchmark_cfg.get("selector_methods", list(DEFAULT_SELECTOR_METHODS)))
    train_scenarios = _default_if_empty(
        benchmark_cfg.get("task_train_scenarios"),
        ["test", "stress_test", "satnogs_observation_replay_val"],
    )
    eval_scenarios = _default_if_empty(
        benchmark_cfg.get("task_eval_scenarios"),
        ["satnogs_observation_replay_test", "satnogs_observation_replay_stress_test"],
    )
    eval_start = max(int(cfg["training"]["window_size"]) - 1, 0)

    repo_root = Path(args.config).resolve().parent.parent
    data_dir = repo_root / cfg["data"]["output_dir"]
    output_dir = repo_root / args.output_dir
    dataset_manifest = load_json(data_dir / "dataset_manifest.json")
    public_manifest = load_json(repo_root / "configs/public_tracking_manifest.json")

    training_eval_dir = output_dir / "training_eval"
    canonical_traj = pd.read_csv(repo_root / "results/trajectory_errors.csv")
    canonical_train = canonical_traj[canonical_traj["scenario"].isin(train_scenarios)].copy()
    canonical_train_scenarios = set(canonical_train["scenario"].unique())
    missing_train_scenarios = [name for name in train_scenarios if name not in canonical_train_scenarios]
    train_traj_frames: list[pd.DataFrame] = [canonical_train]
    train_traj_path = ensure_training_eval_artifacts(
        config_path=Path(args.config).resolve(),
        device=args.device,
        scenarios=missing_train_scenarios,
        output_dir=training_eval_dir,
        refresh=bool(args.refresh_training_eval),
    )
    if missing_train_scenarios:
        train_traj_frames.append(pd.read_csv(train_traj_path))

    train_traj = pd.concat(train_traj_frames, ignore_index=True) if train_traj_frames else pd.DataFrame()
    eval_traj = canonical_traj
    eval_traj = eval_traj[eval_traj["scenario"].isin(eval_scenarios)].copy()

    train_features = build_feature_frame_for_scenarios(data_dir=data_dir, scenario_names=train_scenarios, eval_start=eval_start)
    eval_features = build_feature_frame_for_scenarios(data_dir=data_dir, scenario_names=eval_scenarios, eval_start=eval_start)

    train_stability = build_stability_labels(train_traj)
    train_stability = train_stability[train_stability["method"].isin(selector_methods)].copy()
    eval_stability = build_stability_labels(eval_traj)
    eval_stability = eval_stability[eval_stability["method"].isin(selector_methods)].copy()

    train_method_features = prepare_method_feature_frame(train_features, train_stability, methods=selector_methods)
    eval_method_features = prepare_method_feature_frame(eval_features, eval_stability, methods=selector_methods)

    stability_models = fit_stability_models(train_method_features, methods=selector_methods)
    stability_detail, stability_summary = score_stability_predictors(eval_method_features, stability_models, methods=selector_methods)

    selection_detail, selection_summary = build_selector_outputs(
        train_features,
        train_stability,
        eval_features,
        eval_stability,
        selector_methods=selector_methods,
        divergence_penalty_m=float(benchmark_cfg.get("selector_divergence_penalty_m", 1.0e8)),
    )

    packet_registry = build_packet_registry(cfg, dataset_manifest, public_manifest)
    task_definition = build_task_definition(cfg)
    summary_payload = build_summary_payload(
        task_definition=task_definition,
        packet_registry=packet_registry,
        stability_summary=stability_summary,
        selection_summary=selection_summary,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(output_dir / "trajectory_features_train.csv", index=False)
    eval_features.to_csv(output_dir / "trajectory_features_eval.csv", index=False)
    train_stability.to_csv(output_dir / "stability_labels_train.csv", index=False)
    eval_stability.to_csv(output_dir / "stability_labels_eval.csv", index=False)
    stability_detail.to_csv(output_dir / "stability_prediction_details.csv", index=False)
    stability_summary.to_csv(output_dir / "stability_prediction_summary.csv", index=False)
    selection_detail.to_csv(output_dir / "method_selection_details.csv", index=False)
    selection_summary.to_csv(output_dir / "method_selection_summary.csv", index=False)

    dump_json(packet_registry, output_dir / "packet_registry.json")
    dump_json(task_definition, output_dir / "task_definition.json")
    dump_json(summary_payload, output_dir / "benchmark_task_summary.json")

    readme_lines = [
        f"# {task_definition['benchmark_name']}",
        "",
        "Public benchmark task artifacts generated from the released SPOT-OD experiment packet.",
        "",
        "## Task train scenarios",
        *[f"- `{name}`" for name in train_scenarios],
        "",
        "## Task evaluation scenarios",
        *[f"- `{name}`" for name in eval_scenarios],
        "",
        "## Key outputs",
        "- `packet_registry.json`",
        "- `task_definition.json`",
        "- `stability_prediction_summary.csv`",
        "- `method_selection_summary.csv`",
        "- `benchmark_task_summary.json`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
