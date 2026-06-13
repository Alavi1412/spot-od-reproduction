#!/usr/bin/env python
"""Generate simulated and semi-real replay datasets plus classical prior banks."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gnn_state_estimation.dataset import compute_prior_bank_stats
from gnn_state_estimation.evaluation import generate_noisy_init, parse_baseline_config, run_filter_baselines
from gnn_state_estimation.innovation import compute_innovation_features
from gnn_state_estimation.observation_replay import (
    generate_public_observation_replay_dataset,
    load_public_observation_snapshot,
)
from gnn_state_estimation.public_data import (
    generate_public_catalog_replay_dataset,
    load_public_catalog,
)
from gnn_state_estimation.scenarios import (
    estimator_sim_config,
    has_estimator_overrides,
    scenario_kind,
    truth_sim_config,
)
from gnn_state_estimation.semireal import generate_semireal_replay_dataset, load_tle_catalog
from gnn_state_estimation.simulation import generate_dataset, parse_dataset_config
from gnn_state_estimation.utils.io import dump_json, load_yaml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--benchmark-suite", type=str, default="default")
    p.add_argument("--scenarios", type=str, default=None)
    return p


def generate_x0_estimates(states: np.ndarray, baseline_cfg, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0_estimates = np.zeros((states.shape[0], 6), dtype=np.float64)
    for i in range(states.shape[0]):
        x0_estimates[i] = generate_noisy_init(
            states[i, 0],
            rng,
            baseline_cfg.init_pos_std_m,
            baseline_cfg.init_vel_std_mps,
        )
    return x0_estimates


def compute_filter_priors(
    data: dict[str, np.ndarray],
    sim_cfg: dict[str, Any],
    baseline_cfg,
    seed: int,
    x0_estimates: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    ds_cfg = parse_dataset_config(sim_cfg)
    x0 = x0_estimates if x0_estimates is not None else generate_x0_estimates(data["states"], baseline_cfg, seed)
    filters = run_filter_baselines(
        states=data["states"],
        measurements=data["measurements"],
        visibility=data["visibility"],
        times=data["times"],
        dataset_cfg=ds_cfg,
        baseline_cfg=baseline_cfg,
        seed=seed + 97,
        x0_estimates=x0,
    )
    prior_bank_stats = compute_prior_bank_stats(filters["ekf"], filters["ukf"], filters.get("aukf"))
    return filters, x0, prior_bank_stats


def coverage_summary(data: dict[str, np.ndarray]) -> dict[str, float]:
    coverage = np.asarray(data["visibility"], dtype=np.float64).sum(axis=2)
    return {
        "mean_visible_stations_per_step": float(np.mean(coverage)),
        "fraction_steps_zero_visibility": float(np.mean(coverage == 0)),
        "fraction_steps_one_visibility": float(np.mean(coverage == 1)),
        "fraction_steps_two_plus_visibility": float(np.mean(coverage >= 2)),
    }


def maybe_add_source_summary(summary: dict[str, Any], data: dict[str, np.ndarray]) -> None:
    if "source_type" not in data:
        return
    source_type_values = np.asarray(data["source_type"]).reshape(-1)
    if source_type_values.size:
        summary["source_type"] = str(source_type_values[0])
    if "source_norad_cat_id" in data:
        cats = np.asarray(data["source_norad_cat_id"]).reshape(-1)
        summary["distinct_source_satellites"] = int(np.unique(cats).shape[0])
    elif "object_id" in data:
        cats = np.asarray(data["object_id"]).reshape(-1)
        summary["distinct_source_satellites"] = int(np.unique(cats).shape[0])
    elif "tle_name" in data:
        tle_names = np.asarray(data["tle_name"]).reshape(-1)
        summary["distinct_source_satellites"] = int(np.unique(tle_names).shape[0])
    if "anchor_station_name" in data:
        stations = np.asarray(data["anchor_station_name"]).reshape(-1)
        summary["distinct_anchor_stations"] = int(np.unique(stations).shape[0])
    if "station_name" in data:
        station_names = np.asarray(data["station_name"]).reshape(-1)
        summary["distinct_station_bank_members"] = int(np.unique(station_names).shape[0])
    elif "station_ecef" in data:
        summary["distinct_station_bank_members"] = int(np.asarray(data["station_ecef"]).shape[1])
    if "source_bucket" in data:
        buckets = np.asarray(data["source_bucket"]).reshape(-1)
        bucket_counts = {str(bucket): int(np.sum(buckets == bucket)) for bucket in np.unique(buckets)}
        summary["source_bucket_counts"] = bucket_counts
    elif "object_bucket" in data:
        buckets = np.asarray(data["object_bucket"]).reshape(-1)
        bucket_counts = {str(bucket): int(np.sum(buckets == bucket)) for bucket in np.unique(buckets)}
        summary["source_bucket_counts"] = bucket_counts
    if "source_observation_id" in data:
        obs_ids = np.asarray(data["source_observation_id"]).reshape(-1)
        summary["distinct_source_observations"] = int(np.unique(obs_ids).shape[0])
    if "source_start" in data and "source_end" in data:
        starts = np.asarray(data["source_start"]).reshape(-1)
        ends = np.asarray(data["source_end"]).reshape(-1)
        durations_s = []
        for start, end in zip(starts, ends, strict=False):
            start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
            durations_s.append((end_dt - start_dt).total_seconds())
        duration_array = np.asarray(durations_s, dtype=np.float64)
        if duration_array.size:
            summary["pass_duration_sec"] = {
                "mean": float(duration_array.mean()),
                "median": float(np.median(duration_array)),
                "min": float(duration_array.min()),
                "max": float(duration_array.max()),
            }


def save_split(
    path: Path,
    data: dict[str, np.ndarray],
    filters: dict[str, np.ndarray],
    x0_estimates: np.ndarray,
    innovation_features: np.ndarray,
    prior_bank_stats: np.ndarray,
    sample_weight: float,
    regime_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_traj = data["states"].shape[0]
    np.savez_compressed(
        path,
        **data,
        ekf_prior=filters["ekf"],
        ukf_prior=filters["ukf"],
        aukf_prior=filters.get("aukf"),
        secondary_prior=filters.get("aukf"),
        x0_estimates=x0_estimates,
        innovation_features=innovation_features,
        prior_bank_stats=prior_bank_stats,
        sample_weight=np.full(n_traj, float(sample_weight), dtype=np.float32),
        regime_name=np.array([regime_name] * n_traj, dtype="<U48"),
    )


def materialize_scenario(
    *,
    name: str,
    scenario_cfg: dict[str, Any],
    base_sim: dict[str, Any],
    baseline_cfg,
    out_dir: Path,
    base_seed: int,
) -> dict[str, Any]:
    kind = scenario_kind(scenario_cfg)
    size = int(scenario_cfg["size"])
    overrides = scenario_cfg.get("overrides", {})
    scenario_seed = int(base_seed + scenario_cfg.get("seed_offset", 0))
    sample_weight = float(scenario_cfg.get("sample_weight", 1.0))
    # Truth is synthesized from base + overrides; recursive filters / prior
    # banks / innovation features instead see the estimator config (base +
    # estimator_overrides when the scenario opts in, otherwise == truth).
    truth_sim, station_meta = truth_sim_config(base_sim, scenario_cfg, with_station_meta=True)
    est_sim = estimator_sim_config(base_sim, scenario_cfg)
    extra_summary: dict[str, Any] = {}
    if kind == "public_catalog_replay":
        extra_summary["public_station_selection"] = station_meta
    elif kind == "public_observation_replay":
        extra_summary["observation_station_bank"] = station_meta
    ds_cfg = parse_dataset_config(truth_sim)
    est_ds_cfg = parse_dataset_config(est_sim)

    if kind == "semi_real_replay":
        tle_catalog_path = scenario_cfg.get("tle_catalog_path", "configs/archived_tles.json")
        tle_catalog = load_tle_catalog(tle_catalog_path)
        data = generate_semireal_replay_dataset(
            ds_cfg,
            tle_catalog=tle_catalog,
            num_trajectories=size,
            seed=scenario_seed,
            tle_filters=scenario_cfg.get("tle_filters"),
        )
    elif kind == "public_catalog_replay":
        catalog_path = scenario_cfg.get("catalog_snapshot_path", "configs/public_celestrak_active_snapshot.json")
        catalog = load_public_catalog(catalog_path)
        data = generate_public_catalog_replay_dataset(
            ds_cfg,
            catalog=catalog,
            num_trajectories=size,
            seed=scenario_seed,
            catalog_filters=scenario_cfg.get("catalog_filters"),
            sampling_strategy=str(scenario_cfg.get("sampling_strategy", "stratified_inclination")),
        )
    elif kind == "public_observation_replay":
        observation_path = scenario_cfg.get(
            "observation_snapshot_path",
            "configs/public_satnogs_recent_good_observations.json",
        )
        observations = load_public_observation_snapshot(observation_path)
        observation_filters = dict(scenario_cfg.get("observation_filters", {}))
        station_limit = scenario_cfg.get("station_filters", {}).get("count")
        if station_limit is not None:
            observation_filters["station_count"] = int(station_limit)
        data = generate_public_observation_replay_dataset(
            ds_cfg,
            observations=observations,
            num_trajectories=size,
            seed=scenario_seed,
            observation_filters=observation_filters,
            source_slice=scenario_cfg.get("source_slice"),
        )
    else:
        data = generate_dataset(ds_cfg, num_trajectories=size, seed=scenario_seed)

    filters, x0_estimates, prior_bank_stats = compute_filter_priors(
        data=data,
        sim_cfg=est_sim,
        baseline_cfg=baseline_cfg,
        seed=scenario_seed + 10_000,
    )
    innovation_features = compute_innovation_features(
        prior_states=filters["ekf"],
        measurements=data["measurements"],
        visibility=data["visibility"],
        times_s=data["times"],
        stations=est_ds_cfg.stations,
        meas_std_vector=est_ds_cfg.measurement_noise.std_vector,
    )
    save_path = out_dir / f"{name}.npz"
    save_split(
        save_path,
        data=data,
        filters=filters,
        x0_estimates=x0_estimates,
        innovation_features=innovation_features,
        prior_bank_stats=prior_bank_stats,
        sample_weight=sample_weight,
        regime_name=name,
    )
    summary = {
        "samples": size,
        "path": str(save_path),
        "kind": kind,
        "sample_weight": sample_weight,
        "overrides": overrides,
        "coverage": coverage_summary(data),
        **extra_summary,
    }
    if has_estimator_overrides(scenario_cfg):
        summary["estimator_overrides"] = scenario_cfg.get("estimator_overrides") or {}
        summary["truth_estimator_model_mismatch"] = True
    maybe_add_source_summary(summary, data)
    return summary


def build_default_suite(exp_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    train_size = int(exp_cfg["data"]["train_size"])
    val_size = int(exp_cfg["data"]["val_size"])
    test_size = int(exp_cfg["data"]["test_size"])
    stress_size = int(exp_cfg["data"]["stress_test_size"])
    stress_overrides = exp_cfg["stress_simulation_overrides"]
    suite_cfg = exp_cfg.get("benchmark_suite", {})
    extra = suite_cfg.get("scenarios", {})
    scenarios: dict[str, dict[str, Any]] = {
        "train": {"size": train_size, "seed_offset": 0, "sample_weight": 1.0},
        "val": {"size": val_size, "seed_offset": 13, "sample_weight": 1.0},
        "test": {"size": test_size, "seed_offset": 26, "sample_weight": 1.0},
        "stress_train": {
            "size": int(suite_cfg.get("stress_train_size", max(32, train_size // 2))),
            "seed_offset": 101,
            "sample_weight": float(suite_cfg.get("stress_sample_weight", 1.5)),
            "overrides": stress_overrides,
        },
        "stress_val": {
            "size": int(suite_cfg.get("stress_val_size", max(16, val_size // 2))),
            "seed_offset": 133,
            "sample_weight": float(suite_cfg.get("stress_sample_weight", 1.5)),
            "overrides": stress_overrides,
        },
        "stress_test": {"size": stress_size, "seed_offset": 999, "sample_weight": 1.0, "overrides": stress_overrides},
    }
    scenarios.update(extra)
    return scenarios


def main() -> None:
    args = build_parser().parse_args()
    exp_cfg = load_yaml(args.config)
    baseline_cfg = parse_baseline_config(exp_cfg["baselines"])
    base_sim = exp_cfg["simulation"]
    seed = int(exp_cfg["seed"])
    out_dir = Path(exp_cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_default_suite(exp_cfg)
    selected_scenarios = None
    if args.scenarios:
        selected_scenarios = {name.strip() for name in args.scenarios.split(",") if name.strip()}
        scenarios = {name: cfg for name, cfg in scenarios.items() if name in selected_scenarios}
        missing = selected_scenarios.difference(scenarios)
        if missing:
            raise SystemExit(f"Unknown scenarios requested: {sorted(missing)}")
    summary = {}
    for name, scenario_cfg in scenarios.items():
        print(f"\n=== Generating {name} ===")
        summary[name] = materialize_scenario(
            name=name,
            scenario_cfg=scenario_cfg,
            base_sim=base_sim,
            baseline_cfg=baseline_cfg,
            out_dir=out_dir,
            base_seed=seed,
        )

    manifest_path = out_dir / "dataset_manifest.json"
    if selected_scenarios and manifest_path.exists():
        existing = {}
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        existing.update(summary)
        summary = existing
    dump_json(summary, manifest_path)
    print("\nDataset generation complete.")


if __name__ == "__main__":
    main()
