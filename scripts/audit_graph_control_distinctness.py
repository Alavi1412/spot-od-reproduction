#!/usr/bin/env python
"""Audit repeated-seed distinctness for graph/no-message-passing controls."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml
from gnn_state_estimation.utils.runtime import resolve_device

try:
    from compute_seed_pooled_significance import (
        checkpoint_states_equal,
        infer_candidate_trajectory_rmse,
    )
    from run_benchmark_seed_sweep import deep_update
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts.compute_seed_pooled_significance import (
        checkpoint_states_equal,
        infer_candidate_trajectory_rmse,
    )
    from scripts.run_benchmark_seed_sweep import deep_update


CONTROL_SPECS = [
    (
        "RGR-noMP",
        "MatchedNoGraphRGR",
        Path("results/seed_suite_matched_nograph_rgr/benchmark_seed_metrics.csv"),
    ),
    (
        "RGR-local",
        "CapacityMatchedNoGraphRGR",
        Path("results/seed_suite_capacity_matched_nograph_rgr/benchmark_seed_metrics.csv"),
    ),
]

REFERENCE_SPEC = (
    "RGR-GF",
    "HybridGNN",
    Path("results/seed_suite_hybrid_public/benchmark_seed_metrics.csv"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scenarios", default="test,stress_test")
    parser.add_argument("--output-csv", default="results/graph_control_distinctness.csv")
    return parser


def scenario_config(cfg: dict[str, Any], scenario: str) -> dict[str, Any]:
    if scenario == "test":
        return cfg["simulation"]
    if scenario == "stress_test":
        return deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])
    scenario_spec = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_spec is None:
        raise ValueError(f"Unknown scenario {scenario!r}")
    return deep_update(cfg["simulation"], scenario_spec.get("overrides", {}))


def load_seed_arrays(
    *,
    cfg: dict[str, Any],
    method_label: str,
    model_name: str,
    metrics_path: Path,
    scenario: str,
    arrays,
    dataset_cfg,
    baseline_cfg,
    train_cfg,
    device,
) -> dict[int, dict[str, Any]]:
    if not metrics_path.exists():
        return {}
    metrics = pd.read_csv(metrics_path)
    focus = metrics[metrics["scenario"] == scenario].copy()
    out: dict[int, dict[str, Any]] = {}
    for item in focus.sort_values("seed").itertuples(index=False):
        checkpoint_path = Path(str(item.checkpoint))
        traj_rmse, _ = infer_candidate_trajectory_rmse(
            cfg=cfg,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            arrays=arrays,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            train_cfg=train_cfg,
            device=device,
            scenario=scenario,
        )
        out[int(item.seed)] = {
            "method": method_label,
            "checkpoint": checkpoint_path,
            "trajectory_rmse": np.asarray(traj_rmse, dtype=np.float64),
        }
    return out


def append_distinctness_rows(
    rows: list[dict[str, Any]],
    *,
    comparison_label: str,
    left_by_seed: dict[int, dict[str, Any]],
    right_by_seed: dict[int, dict[str, Any]],
) -> None:
    for seed in sorted(set(left_by_seed) & set(right_by_seed)):
        left = left_by_seed[seed]
        right = right_by_seed[seed]
        left_arr = left["trajectory_rmse"]
        right_arr = right["trajectory_rmse"]
        state_equal = checkpoint_states_equal(left["checkpoint"], right["checkpoint"])
        traj_exact = bool(np.array_equal(left_arr, right_arr))
        if left_arr.shape == right_arr.shape and left_arr.size:
            max_abs_diff = float(np.max(np.abs(left_arr - right_arr)))
        else:
            max_abs_diff = float("nan")
        rows.append(
            {
                "comparison": comparison_label,
                "scenario": left.get("scenario", right.get("scenario", "")),
                "seed": seed,
                "left_method": left["method"],
                "right_method": right["method"],
                "left_checkpoint": str(left["checkpoint"]),
                "right_checkpoint": str(right["checkpoint"]),
                "model_state_dict_identical": state_equal,
                "trajectory_rmse_exactly_identical": traj_exact,
                "trajectory_rmse_max_abs_diff_m": max_abs_diff,
                "independent_repeated_seed_corollary": bool(not state_equal and not traj_exact),
            }
        )


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(Path(args.config))
    train_cfg = parse_train_config(cfg["training"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(train_cfg, device=str(cfg.get("device", {}).get("eval", train_cfg.device)))
    device = resolve_device(train_cfg.device)
    data_dir = Path(cfg["data"]["output_dir"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    scenarios = [item.strip() for item in str(args.scenarios).split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        arrays = load_dataset_npz(data_dir / f"{scenario}.npz")
        dataset_cfg = parse_dataset_config(scenario_config(cfg, scenario))
        ref_label, ref_model, ref_path = REFERENCE_SPEC
        reference = load_seed_arrays(
            cfg=cfg,
            method_label=ref_label,
            model_name=ref_model,
            metrics_path=ref_path,
            scenario=scenario,
            arrays=arrays,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            train_cfg=train_cfg,
            device=device,
        )
        for payload in reference.values():
            payload["scenario"] = scenario
        control_cache: dict[str, dict[int, dict[str, Any]]] = {}
        for control_label, control_model, control_path in CONTROL_SPECS:
            control = load_seed_arrays(
                cfg=cfg,
                method_label=control_label,
                model_name=control_model,
                metrics_path=control_path,
                scenario=scenario,
                arrays=arrays,
                dataset_cfg=dataset_cfg,
                baseline_cfg=baseline_cfg,
                train_cfg=train_cfg,
                device=device,
            )
            for payload in control.values():
                payload["scenario"] = scenario
            control_cache[control_label] = control
            append_distinctness_rows(
                rows,
                comparison_label=f"{ref_label} vs {control_label}",
                left_by_seed=reference,
                right_by_seed=control,
            )
        if "RGR-noMP" in control_cache and "RGR-local" in control_cache:
            append_distinctness_rows(
                rows,
                comparison_label="RGR-noMP vs RGR-local",
                left_by_seed=control_cache["RGR-noMP"],
                right_by_seed=control_cache["RGR-local"],
            )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print({"rows": len(rows), "csv": str(out_csv), "device": str(device)})


if __name__ == "__main__":
    main()
