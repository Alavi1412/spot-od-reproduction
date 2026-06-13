#!/usr/bin/env python
"""Run repeated-seed training and benchmark-suite evaluation for a configured model."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gnn_state_estimation.dataset import concatenate_dataset_arrays, load_dataset_npz
from gnn_state_estimation.evaluation import (
    build_innovation_features,
    build_prior_bank_feature_array,
    metric_entry_diverged,
    parse_baseline_config,
    relative_improvement_percent,
    run_filter_baselines,
    run_model_inference_batched,
    score_predictions,
)
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.runtime import build_run_manifest, duration_metadata, resolve_device, utc_now_iso
from gnn_state_estimation.utils.seeding import seed_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--model", type=str, default="InnovationHybridGNN")
    p.add_argument("--seeds", type=str, default="41,42,43")
    p.add_argument("--scenarios", type=str, default="all")
    p.add_argument("--output-dir", type=str, default="results/seed_suite")
    p.add_argument("--baseline-cache-dir", type=str, default="results/baseline_cache")
    return p


def parse_seed_list(seed_str: str) -> list[int]:
    return [int(x.strip()) for x in seed_str.split(",") if x.strip()]


def parse_scenario_list(scenario_str: str) -> list[str] | None:
    text = scenario_str.strip().lower()
    if text in {"all", "*"}:
        return None
    return [item.strip() for item in scenario_str.split(",") if item.strip()]


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_split_group(data_dir: Path, split_names: list[str]):
    return concatenate_dataset_arrays([load_dataset_npz(data_dir / f"{name}.npz") for name in split_names])


def load_split_group_or_groups(data_dir: Path, split_names: list[str]):
    arrays = [load_dataset_npz(data_dir / f"{name}.npz") for name in split_names]
    try:
        return concatenate_dataset_arrays(arrays)
    except ValueError as exc:
        if "station geometry" not in str(exc):
            raise
        return arrays


def filter_progress_rows(
    df: pd.DataFrame,
    *,
    model_name: str,
    requested_scenarios: list[str] | None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    filtered = df.copy()
    if "model" in filtered.columns:
        filtered = filtered[filtered["model"] == model_name]
    if requested_scenarios is not None and "scenario" in filtered.columns:
        requested = {str(name) for name in requested_scenarios}
        filtered = filtered[filtered["scenario"].astype(str).isin(requested)]
    return filtered.reset_index(drop=True)


def build_seed_suite_payload(
    *,
    df: pd.DataFrame,
    model_name: str,
    requested_seeds: list[int],
    requested_scenarios: list[str] | None,
    output_dir: Path,
) -> dict[str, Any]:
    completed_seeds = sorted(int(seed) for seed in df["seed"].unique()) if not df.empty else []
    if requested_scenarios is None:
        completed_scenarios = sorted(str(name) for name in df["scenario"].unique()) if not df.empty else []
    else:
        available = set(str(name) for name in df["scenario"].unique()) if not df.empty else set()
        completed_scenarios = [name for name in requested_scenarios if name in available]
    return {
        "model": model_name,
        "requested_seeds": requested_seeds,
        "completed_seeds": completed_seeds,
        "requested_scenarios": requested_scenarios or "all",
        "completed_scenarios": completed_scenarios,
        "n_rows": int(df.shape[0]),
        "rows_path": str(output_dir / "benchmark_seed_metrics.csv"),
        "summary_path": str(output_dir / "benchmark_seed_summary.csv"),
    }


def load_or_compute_baselines(
    *,
    cache_path: Path,
    arrays,
    dataset_cfg,
    baseline_cfg,
    seed: int,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path)
        cached = {k: data[k] for k in ("ekf", "ukf", "aukf") if k in data.files}
        if all(v.shape == arrays.states.shape for v in cached.values()) and {"ekf", "ukf", "aukf"}.issubset(cached):
            return cached
    preds = run_filter_baselines(
        states=arrays.states,
        measurements=arrays.measurements,
        visibility=arrays.visibility,
        times=arrays.times,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        seed=seed,
        x0_estimates=arrays.x0_estimates,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **preds)
    return preds


def load_model_from_checkpoint(*, checkpoint_path: Path, train_cfg, spec: dict[str, Any], device) -> Any:
    from gnn_state_estimation.models import TemporalGraphEstimator

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_kwargs = ckpt.get("model_kwargs", dict(spec.get("model_kwargs", {})))
    use_prior = bool(ckpt.get("use_ekf_prior", spec.get("use_ekf_prior", False)))
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


def evaluate_model_on_scenario(
    *,
    model,
    arrays,
    dataset_cfg,
    baseline_cfg,
    cache_path: Path,
    train_cfg,
    seed: int,
) -> dict[str, float]:
    filters = load_or_compute_baselines(
        cache_path=cache_path,
        arrays=arrays,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        seed=seed,
    )
    innovation_features = arrays.innovation_features
    if getattr(model, "use_innovation_features", False) and innovation_features is None:
        innovation_features = build_innovation_features(
            dataset_cfg=dataset_cfg,
            measurements=arrays.measurements,
            visibility=arrays.visibility,
            times=arrays.times,
            ekf_prior=filters["ekf"],
        )
    prior_bank_stats = arrays.prior_bank_stats
    if getattr(model, "use_prior_bank_fusion", False) and prior_bank_stats is None:
        prior_bank_stats = build_prior_bank_feature_array(
            filters["ekf"],
            filters["ukf"],
            filters["aukf"],
            dataset_cfg=dataset_cfg,
            station_ecef=arrays.station_ecef,
            visibility=arrays.visibility,
            times=arrays.times,
            use_observability_context=bool(getattr(model, "use_observability_context", False)),
        )
    elif getattr(model, "use_prior_bank_fusion", False) and bool(getattr(model, "use_observability_context", False)):
        prior_bank_stats = build_prior_bank_feature_array(
            filters["ekf"],
            filters["ukf"],
            filters["aukf"],
            dataset_cfg=dataset_cfg,
            station_ecef=arrays.station_ecef,
            visibility=arrays.visibility,
            times=arrays.times,
            use_observability_context=True,
        )
    pred = run_model_inference_batched(
        model=model,
        states=arrays.states,
        measurements=arrays.measurements,
        visibility=arrays.visibility,
        station_ecef=arrays.station_ecef,
        window_size=train_cfg.window_size,
        ekf_prior=filters["ekf"] if (model.use_ekf_prior or getattr(model, "use_prior_bank_fusion", False)) else None,
        ukf_prior=filters["ukf"] if getattr(model, "use_prior_bank_fusion", False) else None,
        aukf_prior=filters["aukf"] if getattr(model, "use_prior_bank_fusion", False) else None,
        secondary_prior=filters["aukf"] if getattr(model, "use_dual_prior_fusion", False) else None,
        innovation_features=innovation_features if getattr(model, "use_innovation_features", False) else None,
        prior_bank_stats=prior_bank_stats if getattr(model, "use_prior_bank_fusion", False) else None,
    )
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    metric = score_predictions(arrays.states[:, eval_start:], pred[:, eval_start:])
    ekf_metric = score_predictions(arrays.states[:, eval_start:], filters["ekf"][:, eval_start:])
    ukf_metric = score_predictions(arrays.states[:, eval_start:], filters["ukf"][:, eval_start:])
    aukf_metric = score_predictions(arrays.states[:, eval_start:], filters["aukf"][:, eval_start:])
    classical = {
        "EKF": ekf_metric,
        "UKF": ukf_metric,
        "AUKF": aukf_metric,
    }
    best_classical_name = min(classical, key=lambda name: float(classical[name]["pos_rmse_m"]))
    best_classical_rmse = float(classical[best_classical_name]["pos_rmse_m"])
    return {
        "pos_rmse_m": float(metric["pos_rmse_m"]),
        "vel_rmse_mps": float(metric["vel_rmse_mps"]),
        "improvement_vs_ukf_percent": float(relative_improvement_percent(float(ukf_metric["pos_rmse_m"]), float(metric["pos_rmse_m"]))),
        "improvement_vs_best_classical_percent": float(relative_improvement_percent(best_classical_rmse, float(metric["pos_rmse_m"]))),
        "best_classical_method": best_classical_name,
        "best_classical_pos_rmse_m": best_classical_rmse,
    }


def bootstrap_ci(values: np.ndarray, *, seed: int, n_bootstrap: int = 3000, ci_percent: float = 95.0) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, values.size, size=values.size)
        boot[i] = np.mean(values[idx])
    alpha = (100.0 - ci_percent) / 2.0
    return float(np.percentile(boot, alpha)), float(np.percentile(boot, 100.0 - alpha))


def summarise_rows(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for scenario, group in df.groupby("scenario", sort=False):
        for metric in ("pos_rmse_m", "vel_rmse_mps", "improvement_vs_ukf_percent", "improvement_vs_best_classical_percent"):
            values = group[metric].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_ci(values, seed=123 + len(summary_rows))
            summary_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_seeds": int(values.size),
                }
            )
    return pd.DataFrame(summary_rows)


def main() -> None:
    run_started_at = utc_now_iso()
    run_perf_start = time.perf_counter()
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    cfg_text = cfg_path.read_text(encoding="utf-8")
    seeds = parse_seed_list(args.seeds)
    requested_scenarios = parse_scenario_list(args.scenarios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(args.model)
    progress_path = output_dir / "benchmark_seed_metrics.csv"

    models_cfg = cfg.get("models", {})
    if model_name not in models_cfg or not bool(models_cfg[model_name].get("enabled", False)):
        raise ValueError(f"Model {model_name!r} is not enabled in the config.")

    train_cfg = parse_train_config(cfg["training"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(train_cfg, device=str(cfg.get("device", {}).get("train", train_cfg.device)))
    device = resolve_device(train_cfg.device)
    data_dir = Path(cfg["data"]["output_dir"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    curriculum = cfg.get("curriculum", {}).get("stages", [])
    if not curriculum:
        curriculum = [{"name": "default", "train_splits": ["train"], "val_splits": ["val"], "epochs": train_cfg.num_epochs}]

    scenario_cfgs: list[tuple[str, dict[str, Any]]] = [
        ("test", cfg["simulation"]),
        ("stress_test", deep_update(cfg["simulation"], cfg["stress_simulation_overrides"])),
    ]
    for scenario_name, scenario_spec in cfg.get("benchmark_suite", {}).get("scenarios", {}).items():
        scenario_cfgs.append((scenario_name, deep_update(cfg["simulation"], scenario_spec.get("overrides", {}))))
    if requested_scenarios is not None:
        requested = set(requested_scenarios)
        scenario_cfgs = [(name, scfg) for name, scfg in scenario_cfgs if name in requested]
        if not scenario_cfgs:
            raise ValueError(f"No matching scenarios found for {args.scenarios!r}.")

    arrays_by_scenario = {name: load_dataset_npz(data_dir / f"{name}.npz") for name, _ in scenario_cfgs}
    dataset_cfg_by_scenario = {name: parse_dataset_config(scenario_cfg) for name, scenario_cfg in scenario_cfgs}

    spec = models_cfg[model_name]
    model_kwargs = dict(spec.get("model_kwargs", {}))

    rows: list[dict[str, Any]] = []
    completed_pairs: set[tuple[int, str]] = set()
    if progress_path.exists():
        existing_df = filter_progress_rows(
            pd.read_csv(progress_path),
            model_name=model_name,
            requested_scenarios=requested_scenarios,
        )
        rows = existing_df.to_dict(orient="records")
        completed_pairs = {(int(row["seed"]), str(row["scenario"])) for row in rows}

    for seed in seeds:
        seed_all(seed)
        current_checkpoint = None
        model = None
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        final_checkpoint = seed_dir / spec["checkpoint_name"]
        if final_checkpoint.exists():
            model = load_model_from_checkpoint(
                checkpoint_path=final_checkpoint,
                train_cfg=train_cfg,
                spec=spec,
                device=device,
            )
            current_checkpoint = final_checkpoint
        else:
            for stage_idx, stage in enumerate(curriculum):
                stage_name = str(stage["name"])
                stage_cfg = replace(train_cfg, num_epochs=int(stage.get("epochs", train_cfg.num_epochs)))
                train_arrays = load_split_group(data_dir, list(stage["train_splits"]))
                val_arrays = load_split_group_or_groups(data_dir, list(stage["val_splits"]))
                checkpoint_name = (
                    spec["checkpoint_name"]
                    if stage_idx == len(curriculum) - 1
                    else f"{Path(spec['checkpoint_name']).stem}_{stage.get('checkpoint_suffix', stage_name)}.pt"
                )
                stage_checkpoint = seed_dir / checkpoint_name
                if stage_checkpoint.exists():
                    current_checkpoint = stage_checkpoint
                    model = load_model_from_checkpoint(
                        checkpoint_path=stage_checkpoint,
                        train_cfg=train_cfg,
                        spec=spec,
                        device=device,
                    )
                    continue
                model, _, current_checkpoint = train_model(
                    train_arrays=train_arrays,
                    val_arrays=val_arrays,
                    cfg=stage_cfg,
                    output_dir=seed_dir,
                    seed=seed + stage_idx,
                    use_ekf_prior=bool(spec.get("use_ekf_prior", False)),
                    model_kwargs=model_kwargs,
                    dataset_cfg=parse_dataset_config(cfg["simulation"]),
                    baseline_cfg=baseline_cfg,
                    checkpoint_name=checkpoint_name,
                    initial_checkpoint=current_checkpoint,
                    device=device,
                )
        if model is None or current_checkpoint is None:
            raise RuntimeError(f"Training or checkpoint loading failed for {model_name} seed {seed}.")
        for scenario_name, _ in scenario_cfgs:
            if (seed, scenario_name) in completed_pairs:
                continue
            metric = evaluate_model_on_scenario(
                model=model,
                arrays=arrays_by_scenario[scenario_name],
                dataset_cfg=dataset_cfg_by_scenario[scenario_name],
                baseline_cfg=baseline_cfg,
                cache_path=Path(args.baseline_cache_dir) / f"{scenario_name}_baselines.npz",
                train_cfg=train_cfg,
                seed=int(cfg["seed"]) + 101,
            )
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario_name,
                    "model": model_name,
                    **metric,
                    "checkpoint": str(current_checkpoint),
                }
            )
            completed_pairs.add((seed, scenario_name))
            pd.DataFrame(rows).to_csv(progress_path, index=False)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["seed", "scenario"]).reset_index(drop=True)
    df.to_csv(progress_path, index=False)
    summary_df = summarise_rows(df)
    summary_df.to_csv(output_dir / "benchmark_seed_summary.csv", index=False)
    payload = build_seed_suite_payload(
        df=df,
        model_name=model_name,
        requested_seeds=seeds,
        requested_scenarios=requested_scenarios,
        output_dir=output_dir,
    )
    dump_json(payload, output_dir / "benchmark_seed_summary.json")

    build_run_manifest(
        command=[
            "run_benchmark_seed_sweep.py",
            "--config",
            str(cfg_path),
            "--model",
            model_name,
            "--seeds",
            args.seeds,
            "--scenarios",
            args.scenarios,
            "--device",
            str(device),
        ],
        config_text=cfg_text,
        config_path=cfg_path,
        output_path=Path(cfg["output"]["manifest_dir"]) / f"{model_name}_seed_suite.json",
        device=device,
        seed=int(cfg["seed"]),
        dataset_paths={name: data_dir / f"{name}.npz" for name, _ in scenario_cfgs},
        extra=payload,
        repo_root=cfg_path.parent.parent,
        timing=duration_metadata(run_perf_start, started_at_utc=run_started_at),
    )
    print(f"Benchmark seed sweep complete for {model_name}: {len(seeds)} seeds.")


if __name__ == "__main__":
    main()
