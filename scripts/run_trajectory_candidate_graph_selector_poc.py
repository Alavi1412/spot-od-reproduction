#!/usr/bin/env python
"""Train a retained-output trajectory-level graph selector over candidates."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import csv
import glob
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCHEMA_VERSION = "trajectory_candidate_graph_selector_poc.v1"
BOUNDARY_STATEMENT = (
    "Retained-output compact-simulator trajectory candidate graph selector evidence only; "
    "not independent-machine, not operational, not full-rerun evidence."
)
PREDICTION_FILENAME = "adaptive_candidate_fusion_predictions.npz"
DEFAULT_SOURCE_GLOB = "results/adaptive_candidate_fusion_observed_fixed_soft_seed*_split*_20260623"
DEFAULT_OUTPUT_DIR = "results/trajectory_candidate_graph_selector_poc"
DEFAULT_SCENARIOS = "process_noise_shift_test,maneuver_shift_test"
DEFAULT_CANDIDATE_METHODS = "EKF,UKF,AUKF,BatchWLS,RFIS,VA_RFIS"
DEFAULT_BASELINE_CANDIDATE_METHODS = DEFAULT_CANDIDATE_METHODS
SUPPORTED_CANDIDATE_METHODS = ("EKF", "UKF", "AUKF", "BatchWLS", "RFIS", "VA_RFIS")
SUPPORTED_GRAPH_LAYER_TYPES = ("mean", "attention")
TIER_NAMES = (
    "development_seed_lt_67",
    "holdout_seed_ge_67",
    "future_seed_ge_109",
    "fresh_extra",
    "all_eval_non_development",
)
SEED_SPLIT_RE = re.compile(r"seed(?P<seed>\d+)_split(?P<split>\d+)")
DISTANCE_LOG_SCALE = 20.0


@dataclass(frozen=True)
class SourceRun:
    path: Path
    seed: int
    split: int
    is_extra: bool = False


@dataclass(frozen=True)
class BaselineResult:
    method: str
    index: int
    observed_step_rmse_m: float
    observed_step_sse: float
    observed_steps: int
    method_rmses: dict[str, float | None]


@dataclass
class TrajectorySample:
    source_dir: str
    source_name: str
    seed: int
    split: int
    source_is_extra: bool
    scenario: str
    trajectory_row: int
    trajectory_index: int
    node_features: np.ndarray
    edge_features: np.ndarray
    candidate_mask: np.ndarray
    label: int
    candidate_observed_rmse: np.ndarray
    candidate_all_rmse: np.ndarray
    candidate_bank: np.ndarray
    baseline_candidate_bank: np.ndarray
    states: np.ndarray
    eval_mask: np.ndarray
    observed_mask: np.ndarray
    baseline: BaselineResult | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glob", type=str, default=DEFAULT_SOURCE_GLOB)
    parser.add_argument("--extra-source-dir", action="append", default=[])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", type=str, default=DEFAULT_SCENARIOS)
    parser.add_argument("--candidate-methods", type=str, default=DEFAULT_CANDIDATE_METHODS)
    parser.add_argument("--baseline-candidate-methods", type=str, default=DEFAULT_BASELINE_CANDIDATE_METHODS)
    parser.add_argument("--development-seed-max-exclusive", type=int, default=67)
    parser.add_argument("--holdout-seed-min", type=int, default=67)
    parser.add_argument("--future-seed-min", type=int, default=109)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--graph-layer-type", choices=SUPPORTED_GRAPH_LAYER_TYPES, default="mean")
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2009)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--ensemble-seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def resolve_ensemble_member_seeds(
    *,
    base_seed: int,
    ensemble_size: int,
    ensemble_seeds: str | None,
) -> list[int]:
    size = int(ensemble_size)
    if size <= 0:
        raise ValueError("--ensemble-size must be positive.")
    if ensemble_seeds is None:
        return [int(base_seed) + idx for idx in range(size)]

    seeds: list[int] = []
    for token in parse_csv(ensemble_seeds):
        try:
            seeds.append(int(token))
        except ValueError as exc:
            raise ValueError("--ensemble-seeds must be a CSV of ints.") from exc
    if len(seeds) != size:
        raise ValueError("--ensemble-seeds must contain exactly --ensemble-size ints.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--ensemble-seeds must not contain duplicate ints.")
    return seeds


def parse_seed_split(path: str | Path) -> tuple[int, int]:
    match = SEED_SPLIT_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"could not parse seed/split from source directory name: {Path(path).name}")
    return int(match.group("seed")), int(match.group("split"))


def parse_candidate_methods(raw: str, *, option_name: str = "--candidate-methods", min_count: int = 2) -> list[str]:
    canonical = {name.lower(): name for name in SUPPORTED_CANDIDATE_METHODS}
    out: list[str] = []
    for token in parse_csv(raw):
        key = token.lower()
        if key not in canonical:
            raise ValueError(
                f"unknown candidate method {token!r} in {option_name}; "
                f"choose from {list(SUPPORTED_CANDIDATE_METHODS)}"
            )
        method = canonical[key]
        if method not in out:
            out.append(method)
    if len(out) < int(min_count):
        plural = "method" if int(min_count) == 1 else "methods"
        raise ValueError(f"{option_name} must contain at least {int(min_count)} {plural}.")
    return out


def candidate_key(method: str) -> str:
    return method.lower().replace("_", "_")


def resolve_source_runs(source_glob: str, extra_source_dirs: list[str]) -> list[SourceRun]:
    entries: dict[Path, bool] = {}
    for raw_path in glob.glob(source_glob):
        path = Path(raw_path)
        if path.is_dir():
            entries[path.resolve()] = entries.get(path.resolve(), False)
    for raw_path in extra_source_dirs:
        path = Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"--extra-source-dir does not exist or is not a directory: {path}")
        entries[path.resolve()] = True
    if not entries:
        raise FileNotFoundError(f"no source directories matched --source-glob {source_glob!r}")
    runs: list[SourceRun] = []
    for path, is_extra in entries.items():
        seed, split = parse_seed_split(path)
        runs.append(SourceRun(path=path, seed=seed, split=split, is_extra=is_extra))
    return sorted(runs, key=lambda run: str(run.path).casefold())


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_training_device(requested: str, *, allow_cpu_smoke: bool) -> torch.device:
    raw = str(requested).strip().lower()
    if raw == "cpu":
        if not allow_cpu_smoke:
            raise SystemExit("CPU training is refused; pass --allow-cpu-smoke only for tiny tests.")
        return torch.device("cpu")
    if raw.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        if allow_cpu_smoke:
            return torch.device("cpu")
        raise SystemExit("CUDA was requested but is not available; CPU fallback is refused.")
    device = torch.device(requested)
    if device.type == "cpu" and not allow_cpu_smoke:
        raise SystemExit("CPU training is refused; pass --allow-cpu-smoke only for tiny tests.")
    return device


def visibility_observed_mask(visibility: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    vis = np.asarray(visibility)
    eval_arr = np.asarray(eval_mask, dtype=bool)
    if vis.ndim == 1:
        visible = np.nan_to_num(vis, nan=0.0) >= 0.5
    elif vis.ndim == 2:
        # A [T, S] trajectory view or a [N, T] full-array view both reduce over
        # the trailing axis for trajectory-level callers.
        if vis.shape == eval_arr.shape:
            visible = np.nan_to_num(vis, nan=0.0) >= 0.5
        else:
            visible = np.any(np.nan_to_num(vis, nan=0.0) >= 0.5, axis=-1)
    else:
        axes = tuple(range(2, vis.ndim)) if vis.shape[:2] == eval_arr.shape else tuple(range(1, vis.ndim))
        visible = np.any(np.nan_to_num(vis, nan=0.0) >= 0.5, axis=axes)
    return np.asarray(visible, dtype=bool) & eval_arr


def _finite_nonnegative_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr[arr >= 0.0]


def _scaled_distance(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return float(math.log1p(value) / DISTANCE_LOG_SCALE)


def _scaled_stats(values: np.ndarray, *, include_p50: bool) -> list[float]:
    finite = _finite_nonnegative_values(values)
    if finite.size == 0:
        return [0.0, 0.0, 0.0, 0.0] if include_p50 else [0.0, 0.0, 0.0]
    stats = [float(np.mean(finite)), float(np.std(finite))]
    if include_p50:
        stats.append(float(np.percentile(finite, 50.0)))
    stats.append(float(np.percentile(finite, 90.0)))
    return [_scaled_distance(value) for value in stats]


def _step_delta_stats(candidate: np.ndarray, mask: np.ndarray) -> list[float]:
    pos = np.asarray(candidate, dtype=np.float64)[:, :3]
    step_mask = np.asarray(mask, dtype=bool)
    if pos.shape[0] < 2:
        return [0.0, 0.0, 0.0]
    valid = (
        step_mask[1:]
        & step_mask[:-1]
        & np.all(np.isfinite(pos[1:]), axis=-1)
        & np.all(np.isfinite(pos[:-1]), axis=-1)
    )
    deltas = np.linalg.norm(pos[1:] - pos[:-1], axis=-1)
    return _scaled_stats(deltas[valid], include_p50=False)


def _pair_distance_values(candidate_i: np.ndarray, candidate_j: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pos_i = np.asarray(candidate_i, dtype=np.float64)[:, :3]
    pos_j = np.asarray(candidate_j, dtype=np.float64)[:, :3]
    valid = (
        np.asarray(mask, dtype=bool)
        & np.all(np.isfinite(pos_i), axis=-1)
        & np.all(np.isfinite(pos_j), axis=-1)
    )
    if not np.any(valid):
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(pos_i[valid] - pos_j[valid], axis=-1)


def _candidate_disagreement_stats(candidate_bank: np.ndarray, candidate_index: int, mask: np.ndarray) -> list[float]:
    values = []
    for other_index in range(candidate_bank.shape[1]):
        if other_index == candidate_index:
            continue
        values.append(_pair_distance_values(candidate_bank[:, candidate_index], candidate_bank[:, other_index], mask))
    if not values:
        return [0.0, 0.0, 0.0, 0.0]
    return _scaled_stats(np.concatenate(values), include_p50=True)


def _visibility_stats(visibility: np.ndarray, eval_mask: np.ndarray, observed_mask: np.ndarray) -> list[float]:
    vis = np.asarray(visibility, dtype=np.float64)
    eval_arr = np.asarray(eval_mask, dtype=bool)
    obs_arr = np.asarray(observed_mask, dtype=bool)
    if vis.ndim == 1:
        station_visible = (np.nan_to_num(vis, nan=0.0) >= 0.5).reshape(-1, 1)
    else:
        station_visible = (np.nan_to_num(vis, nan=0.0) >= 0.5).reshape(vis.shape[0], -1)
    station_count = max(int(station_visible.shape[1]), 1)
    visible_steps = np.any(station_visible, axis=1)
    counts_norm = station_visible.sum(axis=1).astype(np.float64) / float(station_count)
    denom = max(int(eval_arr.sum()), 1)
    if np.any(eval_arr):
        eval_counts = counts_norm[eval_arr]
        visible_step_fraction = float(np.mean(visible_steps[eval_arr]))
        station_visible_fraction = float(np.mean(station_visible[eval_arr]))
        mean_count = float(np.mean(eval_counts))
        std_count = float(np.std(eval_counts))
    else:
        visible_step_fraction = 0.0
        station_visible_fraction = 0.0
        mean_count = 0.0
        std_count = 0.0
    return [
        visible_step_fraction,
        station_visible_fraction,
        float(obs_arr.sum()) / float(denom),
        mean_count,
        std_count,
        float(eval_arr.mean()) if eval_arr.size else 0.0,
    ]


def build_feature_names(scenarios: list[str], candidate_methods: list[str]) -> list[str]:
    return [
        "finite_step_fraction_eval",
        "finite_value_fraction_eval",
        "observed_step_delta_mean_log",
        "observed_step_delta_std_log",
        "observed_step_delta_p90_log",
        "all_eval_step_delta_mean_log",
        "all_eval_step_delta_std_log",
        "all_eval_step_delta_p90_log",
        "observed_disagreement_mean_log",
        "observed_disagreement_std_log",
        "observed_disagreement_p50_log",
        "observed_disagreement_p90_log",
        "all_eval_disagreement_mean_log",
        "all_eval_disagreement_std_log",
        "all_eval_disagreement_p50_log",
        "all_eval_disagreement_p90_log",
        *[f"scenario_{scenario}" for scenario in scenarios],
        "visible_step_fraction",
        "visible_station_fraction",
        "observed_eval_fraction",
        "mean_visible_station_count_norm",
        "std_visible_station_count_norm",
        "eval_step_fraction",
        *[f"method_{method}" for method in candidate_methods],
    ]


def build_edge_feature_names() -> list[str]:
    return [
        "observed_pair_distance_mean_log",
        "observed_pair_distance_std_log",
        "observed_pair_distance_p50_log",
        "observed_pair_distance_p90_log",
        "all_eval_pair_distance_mean_log",
        "all_eval_pair_distance_std_log",
        "all_eval_pair_distance_p50_log",
        "all_eval_pair_distance_p90_log",
        "observed_pair_overlap_fraction",
        "all_eval_pair_overlap_fraction",
    ]


def build_candidate_graph_features(
    *,
    candidate_bank: np.ndarray,
    visibility: np.ndarray,
    eval_mask: np.ndarray,
    observed_mask: np.ndarray,
    scenario: str,
    scenarios: list[str],
    candidate_methods: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bank = np.asarray(candidate_bank, dtype=np.float64)
    eval_arr = np.asarray(eval_mask, dtype=bool)
    obs_arr = np.asarray(observed_mask, dtype=bool)
    if bank.ndim != 3:
        raise ValueError("candidate_bank must have shape [T, C, state_dim].")
    candidate_count = int(bank.shape[1])
    scenario_one_hot = [1.0 if scenario == name else 0.0 for name in scenarios]
    vis_stats = _visibility_stats(visibility, eval_arr, obs_arr)
    node_features: list[list[float]] = []
    candidate_mask = np.zeros(candidate_count, dtype=bool)
    eval_denom = max(int(eval_arr.sum()), 1)
    value_eval_denom = max(int(eval_arr.sum()) * int(bank.shape[2]), 1)
    for candidate_index in range(candidate_count):
        candidate = bank[:, candidate_index]
        finite_steps = np.all(np.isfinite(candidate), axis=-1)
        finite_values = np.isfinite(candidate)
        candidate_mask[candidate_index] = bool(np.any(finite_steps & obs_arr))
        method_one_hot = [1.0 if idx == candidate_index else 0.0 for idx in range(candidate_count)]
        node_features.append(
            [
                float(np.sum(finite_steps & eval_arr)) / float(eval_denom),
                float(np.sum(finite_values[eval_arr])) / float(value_eval_denom),
                *_step_delta_stats(candidate, obs_arr),
                *_step_delta_stats(candidate, eval_arr),
                *_candidate_disagreement_stats(bank, candidate_index, obs_arr),
                *_candidate_disagreement_stats(bank, candidate_index, eval_arr),
                *scenario_one_hot,
                *vis_stats,
                *method_one_hot,
            ]
        )

    edge_names = build_edge_feature_names()
    edge_features = np.zeros((candidate_count, candidate_count, len(edge_names)), dtype=np.float32)
    for target_index in range(candidate_count):
        for source_index in range(candidate_count):
            if target_index == source_index:
                continue
            observed_values = _pair_distance_values(
                bank[:, target_index],
                bank[:, source_index],
                obs_arr,
            )
            all_values = _pair_distance_values(
                bank[:, target_index],
                bank[:, source_index],
                eval_arr,
            )
            finite_pair = (
                np.all(np.isfinite(bank[:, target_index, :3]), axis=-1)
                & np.all(np.isfinite(bank[:, source_index, :3]), axis=-1)
            )
            observed_overlap = float(np.sum(finite_pair & obs_arr)) / float(max(int(obs_arr.sum()), 1))
            all_overlap = float(np.sum(finite_pair & eval_arr)) / float(max(int(eval_arr.sum()), 1))
            edge_features[target_index, source_index] = np.asarray(
                [
                    *_scaled_stats(observed_values, include_p50=True),
                    *_scaled_stats(all_values, include_p50=True),
                    observed_overlap,
                    all_overlap,
                ],
                dtype=np.float32,
            )

    nodes = np.nan_to_num(np.asarray(node_features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    edges = np.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)
    return nodes, edges, candidate_mask


def position_sse_count(states: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    truth = np.asarray(states, dtype=np.float64)[..., :3]
    pred = np.asarray(prediction, dtype=np.float64)[..., :3]
    mask_arr = np.asarray(mask, dtype=bool)
    valid = mask_arr & np.all(np.isfinite(truth), axis=-1) & np.all(np.isfinite(pred), axis=-1)
    if not np.any(valid):
        return 0.0, 0
    diff = pred[valid] - truth[valid]
    sse = float(np.sum(diff * diff))
    return sse, int(valid.sum())


def rmse_from_sse_count(sse: float, count: int) -> float:
    if count <= 0:
        return float("inf")
    return float(math.sqrt(max(float(sse), 0.0) / float(count)))


def observed_step_rmse_by_candidate(
    states: np.ndarray,
    candidate_bank: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    bank = np.asarray(candidate_bank, dtype=np.float64)
    if bank.ndim == 4:
        candidate_count = int(bank.shape[2])
        rmses = []
        for candidate_index in range(candidate_count):
            sse, count = position_sse_count(states, bank[:, :, candidate_index], mask)
            rmses.append(rmse_from_sse_count(sse, count))
        return np.asarray(rmses, dtype=np.float64)
    if bank.ndim == 3:
        candidate_count = int(bank.shape[1])
        rmses = []
        for candidate_index in range(candidate_count):
            sse, count = position_sse_count(states, bank[:, candidate_index], mask)
            rmses.append(rmse_from_sse_count(sse, count))
        return np.asarray(rmses, dtype=np.float64)
    raise ValueError("candidate_bank must have shape [N, T, C, D] or [T, C, D].")


def best_single_candidate_baseline(
    *,
    states: np.ndarray,
    candidate_bank: np.ndarray,
    observed_mask: np.ndarray,
    candidate_methods: list[str],
) -> BaselineResult:
    rmses = observed_step_rmse_by_candidate(states, candidate_bank, observed_mask)
    finite = np.isfinite(rmses)
    if not np.any(finite):
        raise ValueError("no finite candidate has observed-step RMSE for run/scenario baseline.")
    best_index = int(np.nanargmin(rmses))
    best_sse, best_count = position_sse_count(
        states,
        np.asarray(candidate_bank)[:, :, best_index],
        observed_mask,
    )
    return BaselineResult(
        method=candidate_methods[best_index],
        index=best_index,
        observed_step_rmse_m=float(rmses[best_index]),
        observed_step_sse=best_sse,
        observed_steps=best_count,
        method_rmses={
            method: (float(rmses[idx]) if math.isfinite(float(rmses[idx])) else None)
            for idx, method in enumerate(candidate_methods)
        },
    )


def _trajectory_index_array(payload: np.lib.npyio.NpzFile, n_trajectories: int) -> np.ndarray:
    if "trajectory_indices" not in payload.files:
        return np.arange(n_trajectories, dtype=np.int64)
    raw = np.asarray(payload["trajectory_indices"], dtype=np.int64)
    if raw.shape[0] != n_trajectories:
        return np.arange(n_trajectories, dtype=np.int64)
    return raw


def _load_candidate_bank(
    *,
    payload: np.lib.npyio.NpzFile,
    pred_path: Path,
    states: np.ndarray,
    candidate_methods: list[str],
    purpose: str,
) -> np.ndarray:
    missing_methods = [method for method in candidate_methods if candidate_key(method) not in payload.files]
    if missing_methods:
        missing_keys = [candidate_key(method) for method in missing_methods]
        raise KeyError(
            f"{pred_path} is missing {purpose} candidate prediction keys {missing_keys} "
            f"for methods {missing_methods}."
        )
    candidate_arrays = []
    for method in candidate_methods:
        arr = np.asarray(payload[candidate_key(method)], dtype=np.float64)
        if arr.shape != states.shape:
            raise ValueError(
                f"{pred_path} {purpose} candidate {method} shape {arr.shape} != states shape {states.shape}."
            )
        candidate_arrays.append(arr)
    return np.stack(candidate_arrays, axis=2)


def load_source_scenario_samples(
    *,
    source: SourceRun,
    scenario: str,
    scenarios: list[str],
    candidate_methods: list[str],
    baseline_candidate_methods: list[str],
) -> list[TrajectorySample]:
    pred_path = source.path / scenario / PREDICTION_FILENAME
    if not pred_path.exists():
        raise FileNotFoundError(f"missing retained prediction artifact: {pred_path}")
    with np.load(pred_path) as payload:
        required = ["states", "visibility", "eval_mask"]
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{pred_path} is missing required keys: {missing}")
        states = np.asarray(payload["states"], dtype=np.float64)
        visibility = np.asarray(payload["visibility"], dtype=np.float64)
        eval_mask = np.asarray(payload["eval_mask"], dtype=bool)
        if states.ndim != 3 or states.shape[-1] < 3:
            raise ValueError(f"{pred_path} states must have shape [N, T, state_dim].")
        if eval_mask.shape != states.shape[:2]:
            raise ValueError(f"{pred_path} eval_mask shape {eval_mask.shape} does not match states {states.shape[:2]}.")
        candidate_bank = _load_candidate_bank(
            payload=payload,
            pred_path=pred_path,
            states=states,
            candidate_methods=candidate_methods,
            purpose="selector",
        )
        baseline_candidate_bank = _load_candidate_bank(
            payload=payload,
            pred_path=pred_path,
            states=states,
            candidate_methods=baseline_candidate_methods,
            purpose="baseline",
        )
        trajectory_indices = _trajectory_index_array(payload, states.shape[0])

    observed_mask = visibility_observed_mask(visibility, eval_mask)
    baseline = best_single_candidate_baseline(
        states=states,
        candidate_bank=baseline_candidate_bank,
        observed_mask=observed_mask,
        candidate_methods=baseline_candidate_methods,
    )
    samples: list[TrajectorySample] = []
    for row_index in range(states.shape[0]):
        row_observed_mask = observed_mask[row_index]
        if not np.any(row_observed_mask):
            continue
        row_candidate_bank = candidate_bank[row_index]
        candidate_observed_rmse = observed_step_rmse_by_candidate(
            states[row_index],
            row_candidate_bank,
            row_observed_mask,
        )
        finite_label = np.isfinite(candidate_observed_rmse)
        if not np.any(finite_label):
            continue
        candidate_all_rmse = observed_step_rmse_by_candidate(
            states[row_index],
            row_candidate_bank,
            eval_mask[row_index],
        )
        node_features, edge_features, candidate_mask = build_candidate_graph_features(
            candidate_bank=row_candidate_bank,
            visibility=visibility[row_index],
            eval_mask=eval_mask[row_index],
            observed_mask=row_observed_mask,
            scenario=scenario,
            scenarios=scenarios,
            candidate_methods=candidate_methods,
        )
        if not np.any(candidate_mask):
            continue
        label_mask = candidate_mask & finite_label
        if not np.any(label_mask):
            continue
        label = int(np.nanargmin(np.where(label_mask, candidate_observed_rmse, np.inf)))
        samples.append(
            TrajectorySample(
                source_dir=str(source.path),
                source_name=source.path.name,
                seed=source.seed,
                split=source.split,
                source_is_extra=source.is_extra,
                scenario=scenario,
                trajectory_row=int(row_index),
                trajectory_index=int(trajectory_indices[row_index]),
                node_features=node_features,
                edge_features=edge_features,
                candidate_mask=candidate_mask,
                label=label,
                candidate_observed_rmse=candidate_observed_rmse,
                candidate_all_rmse=candidate_all_rmse,
                candidate_bank=row_candidate_bank,
                baseline_candidate_bank=baseline_candidate_bank[row_index],
                states=states[row_index],
                eval_mask=eval_mask[row_index],
                observed_mask=row_observed_mask,
                baseline=baseline,
            )
        )
    return samples


def load_samples(
    *,
    source_runs: list[SourceRun],
    scenarios: list[str],
    candidate_methods: list[str],
    baseline_candidate_methods: list[str],
) -> list[TrajectorySample]:
    samples: list[TrajectorySample] = []
    for source in source_runs:
        for scenario in scenarios:
            samples.extend(
                load_source_scenario_samples(
                    source=source,
                    scenario=scenario,
                    scenarios=scenarios,
                    candidate_methods=candidate_methods,
                    baseline_candidate_methods=baseline_candidate_methods,
                )
            )
    return samples


class TrajectorySelectorDataset(Dataset):
    def __init__(self, samples: list[TrajectorySample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "node_features": torch.from_numpy(sample.node_features.astype(np.float32, copy=False)),
            "edge_features": torch.from_numpy(sample.edge_features.astype(np.float32, copy=False)),
            "candidate_mask": torch.from_numpy(sample.candidate_mask.astype(bool, copy=False)),
            "label": torch.tensor(sample.label, dtype=torch.long),
        }


class CandidateEdgeGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_features: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        # h: [B, C, H]; edge_features: [B, target C, source C, E]
        bsz, candidate_count, hidden_dim = h.shape
        sender = h.unsqueeze(1).expand(bsz, candidate_count, candidate_count, hidden_dim)
        edge_h = self.edge_proj(torch.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0))
        messages = self.message(torch.cat([sender, edge_h], dim=-1))
        source_mask = candidate_mask.unsqueeze(1).expand(-1, candidate_count, -1)
        self_mask = torch.eye(candidate_count, dtype=torch.bool, device=h.device).unsqueeze(0)
        message_mask = source_mask & ~self_mask
        denom = message_mask.sum(dim=2, keepdim=True).clamp_min(1).to(dtype=h.dtype)
        aggregated = (messages * message_mask.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=2) / denom
        update = self.update(torch.cat([h, aggregated], dim=-1))
        out = self.norm(h + self.dropout(update))
        return torch.where(candidate_mask.unsqueeze(-1), out, h)


class CandidateEdgeAttentionGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.message = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention_logit = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_features: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        # h: [B, C, H]; edge_features: [B, target C, source C, E]
        bsz, candidate_count, hidden_dim = h.shape
        target = h.unsqueeze(2).expand(bsz, candidate_count, candidate_count, hidden_dim)
        sender = h.unsqueeze(1).expand(bsz, candidate_count, candidate_count, hidden_dim)
        edge_h = self.edge_proj(torch.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0))
        pair_h = torch.cat([target, sender, edge_h], dim=-1)
        messages = self.message(pair_h)
        attention_logits = self.attention_logit(pair_h).squeeze(-1)
        source_mask = candidate_mask.unsqueeze(1).expand(-1, candidate_count, -1)
        self_mask = torch.eye(candidate_count, dtype=torch.bool, device=h.device).unsqueeze(0)
        message_mask = source_mask & ~self_mask
        attention_logits = attention_logits.masked_fill(~message_mask, torch.finfo(attention_logits.dtype).min)
        attention = torch.softmax(attention_logits, dim=2)
        attention = torch.where(message_mask, attention, torch.zeros_like(attention))
        aggregated = (messages * attention.unsqueeze(-1)).sum(dim=2)
        update = self.update(torch.cat([h, aggregated], dim=-1))
        out = self.norm(h + self.dropout(update))
        return torch.where(candidate_mask.unsqueeze(-1), out, h)


class TrajectoryCandidateGraphSelector(nn.Module):
    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        graph_layers: int = 2,
        graph_layer_type: str = "mean",
    ) -> None:
        super().__init__()
        if node_feature_dim <= 0 or edge_feature_dim <= 0:
            raise ValueError("node_feature_dim and edge_feature_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if graph_layers < 0:
            raise ValueError("graph_layers must be non-negative.")
        if graph_layer_type not in SUPPORTED_GRAPH_LAYER_TYPES:
            raise ValueError(f"graph_layer_type must be one of {SUPPORTED_GRAPH_LAYER_TYPES}.")
        self.graph_layer_type = graph_layer_type
        self.message_passing_enabled = bool(graph_layers > 0)
        graph_layer_cls = (
            CandidateEdgeGraphLayer if graph_layer_type == "mean" else CandidateEdgeAttentionGraphLayer
        )
        self.node_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.graph_layers = nn.ModuleList(
            [graph_layer_cls(hidden_dim, edge_feature_dim, dropout) for _ in range(graph_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = candidate_mask.to(dtype=torch.bool)
        h = self.node_proj(torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0))
        for layer in self.graph_layers:
            h = layer(h, edge_features, mask)
        logits = self.head(h).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = torch.where(mask, probabilities, torch.zeros_like(probabilities))
        norm = probabilities.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probabilities.dtype).eps)
        probabilities = probabilities / norm
        return {"logits": logits, "probabilities": probabilities}


def _save_checkpoint(
    path: Path,
    *,
    model: TrajectoryCandidateGraphSelector,
    model_kwargs: dict[str, Any],
    candidate_methods: list[str],
    feature_names: list[str],
    edge_feature_names: list[str],
    history: dict[str, list[float]],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "boundary": BOUNDARY_STATEMENT,
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
            "graph_layers": int(model_kwargs.get("graph_layers", 0)),
            "graph_layer_type": str(model_kwargs.get("graph_layer_type", "mean")),
            "message_passing_enabled": bool(int(model_kwargs.get("graph_layers", 0)) > 0),
            "candidate_methods": candidate_methods,
            "feature_names": feature_names,
            "edge_feature_names": edge_feature_names,
            "history": history,
            "config": sanitize_for_json(config),
        },
        path,
    )


def train_selector_model(
    *,
    train_samples: list[TrajectorySample],
    model: TrajectoryCandidateGraphSelector,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    model_kwargs: dict[str, Any],
    candidate_methods: list[str],
    feature_names: list[str],
    edge_feature_names: list[str],
    config: dict[str, Any],
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    if not train_samples:
        raise ValueError("no development training samples were available.")
    dataset = TrajectorySelectorDataset(train_samples)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    history: dict[str, list[float]] = {"train_loss": []}
    best_loss = float("inf")
    ckpt_dir = output_dir / "checkpoints" if checkpoint_dir is None else Path(checkpoint_dir)
    best_path = ckpt_dir / "best_selector.pt"
    last_path = ckpt_dir / "last_selector.pt"
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            node_features = batch["node_features"].to(device)
            edge_features = batch["edge_features"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(node_features, edge_features, candidate_mask)
            loss = F.cross_entropy(output["logits"], labels)
            loss.backward()
            optimizer.step()
            batch_count = int(labels.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_count
            total_count += batch_count
        epoch_loss = total_loss / float(max(total_count, 1))
        history["train_loss"].append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _save_checkpoint(
                best_path,
                model=model,
                model_kwargs=model_kwargs,
                candidate_methods=candidate_methods,
                feature_names=feature_names,
                edge_feature_names=edge_feature_names,
                history=history,
                config=config,
            )
    _save_checkpoint(
        last_path,
        model=model,
        model_kwargs=model_kwargs,
        candidate_methods=candidate_methods,
        feature_names=feature_names,
        edge_feature_names=edge_feature_names,
        history=history,
        config=config,
    )
    return {
        "history": history,
        "best_train_loss": best_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
    }


def predict_probabilities(
    *,
    model: TrajectoryCandidateGraphSelector,
    samples: list[TrajectorySample],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    dataset = TrajectorySelectorDataset(samples)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["node_features"].to(device),
                batch["edge_features"].to(device),
                batch["candidate_mask"].to(device),
            )
            chunks.append(output["probabilities"].detach().cpu().numpy())
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def predict_ensemble_probabilities(
    *,
    models: list[TrajectoryCandidateGraphSelector],
    samples: list[TrajectorySample],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not models:
        raise ValueError("at least one selector model is required for prediction.")
    member_probabilities = [
        predict_probabilities(model=model, samples=samples, device=device, batch_size=batch_size)
        for model in models
    ]
    first_shape = member_probabilities[0].shape
    if any(probs.shape != first_shape for probs in member_probabilities):
        raise ValueError("ensemble member probability shapes did not match.")
    return np.mean(np.stack(member_probabilities, axis=0), axis=0).astype(np.float32, copy=False)


def relative_gain_percent(reference_rmse: float, candidate_rmse: float) -> float | None:
    if not math.isfinite(reference_rmse) or not math.isfinite(candidate_rmse):
        return None
    if reference_rmse <= 0.0:
        return 0.0 if abs(candidate_rmse) <= 1.0e-12 else None
    return 100.0 * (reference_rmse - candidate_rmse) / reference_rmse


def row_tiers(sample: TrajectorySample, *, development_seed_max_exclusive: int, holdout_seed_min: int, future_seed_min: int) -> list[str]:
    tiers: list[str] = []
    if (not sample.source_is_extra) and sample.seed < int(development_seed_max_exclusive):
        tiers.append("development_seed_lt_67")
    if (not sample.source_is_extra) and sample.seed >= int(holdout_seed_min):
        tiers.append("holdout_seed_ge_67")
    if (not sample.source_is_extra) and sample.seed >= int(future_seed_min):
        tiers.append("future_seed_ge_109")
    if sample.source_is_extra:
        tiers.append("fresh_extra")
    if sample.source_is_extra or sample.seed >= int(development_seed_max_exclusive):
        tiers.append("all_eval_non_development")
    return tiers


def evaluate_samples_from_probabilities(
    *,
    probabilities: np.ndarray,
    samples: list[TrajectorySample],
    candidate_methods: list[str],
    baseline_candidate_methods: list[str],
    development_seed_max_exclusive: int,
    holdout_seed_min: int,
    future_seed_min: int,
) -> list[dict[str, Any]]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if len(samples) and probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [sample_count, candidate_count].")
    if len(samples) and probabilities.shape[0] != len(samples):
        raise ValueError("probability row count does not match sample count.")
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        if sample.baseline is None:
            raise ValueError("sample is missing run/scenario baseline.")
        probs = probabilities[sample_index]
        available = np.asarray(sample.candidate_mask, dtype=bool)
        if probs.shape[0] != len(candidate_methods):
            raise ValueError("probability candidate count does not match candidate methods.")
        if available.shape[0] != probs.shape[0]:
            raise ValueError("sample candidate mask size does not match probability candidate count.")
        available_indices = np.flatnonzero(available)
        if available_indices.size == 0:
            raise ValueError("sample has no model-visible candidates.")
        available_scores = np.nan_to_num(
            probs[available_indices],
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )
        selected_index = int(available_indices[int(np.argmax(available_scores))])
        selected_method = candidate_methods[selected_index]
        selected_prediction = sample.candidate_bank[:, selected_index]
        selected_sse, selected_count = position_sse_count(sample.states, selected_prediction, sample.observed_mask)
        baseline_prediction = sample.baseline_candidate_bank[:, sample.baseline.index]
        baseline_sse, baseline_count = position_sse_count(sample.states, baseline_prediction, sample.observed_mask)
        selected_rmse = rmse_from_sse_count(selected_sse, selected_count)
        baseline_rmse = rmse_from_sse_count(baseline_sse, baseline_count)
        tiers = row_tiers(
            sample,
            development_seed_max_exclusive=development_seed_max_exclusive,
            holdout_seed_min=holdout_seed_min,
            future_seed_min=future_seed_min,
        )
        row: dict[str, Any] = {
            "source_dir": sample.source_dir,
            "source_name": sample.source_name,
            "seed": sample.seed,
            "split": sample.split,
            "source_is_extra": sample.source_is_extra,
            "scenario": sample.scenario,
            "trajectory_row": sample.trajectory_row,
            "trajectory_index": sample.trajectory_index,
            "tier_flags": ";".join(tiers),
            "candidate_methods": list(candidate_methods),
            "baseline_candidate_methods": list(baseline_candidate_methods),
            "selected_candidate_method": selected_method,
            "selected_candidate_index": selected_index,
            "selected_probability": float(probs[selected_index]),
            "label_best_observed_method": candidate_methods[sample.label],
            "selected_observed_step_rmse_m": selected_rmse,
            "selected_observed_step_sse": selected_sse,
            "selected_observed_steps": selected_count,
            "best_single_candidate_method": sample.baseline.method,
            "best_single_candidate_index": sample.baseline.index,
            "best_single_run_scenario_observed_step_rmse_m": sample.baseline.observed_step_rmse_m,
            "best_single_trajectory_observed_step_rmse_m": baseline_rmse,
            "best_single_trajectory_observed_step_sse": baseline_sse,
            "best_single_trajectory_observed_steps": baseline_count,
            "gain_vs_best_single_trajectory_percent": relative_gain_percent(baseline_rmse, selected_rmse),
        }
        for idx, method in enumerate(candidate_methods):
            key = method.lower()
            value = sample.candidate_observed_rmse[idx]
            row[f"{key}_trajectory_observed_step_rmse_m"] = float(value) if math.isfinite(float(value)) else None
        rows.append(row)
    return rows


def evaluate_samples(
    *,
    model: TrajectoryCandidateGraphSelector,
    samples: list[TrajectorySample],
    device: torch.device,
    batch_size: int,
    candidate_methods: list[str],
    baseline_candidate_methods: list[str],
    development_seed_max_exclusive: int,
    holdout_seed_min: int,
    future_seed_min: int,
) -> list[dict[str, Any]]:
    probabilities = predict_probabilities(model=model, samples=samples, device=device, batch_size=batch_size)
    return evaluate_samples_from_probabilities(
        probabilities=probabilities,
        samples=samples,
        candidate_methods=candidate_methods,
        baseline_candidate_methods=baseline_candidate_methods,
        development_seed_max_exclusive=development_seed_max_exclusive,
        holdout_seed_min=holdout_seed_min,
        future_seed_min=future_seed_min,
    )


def aggregate_tier_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for tier in TIER_NAMES:
        tier_rows = [row for row in rows if tier in str(row.get("tier_flags", "")).split(";")]
        selected_sse = sum(float(row["selected_observed_step_sse"]) for row in tier_rows)
        selected_count = sum(int(row["selected_observed_steps"]) for row in tier_rows)
        baseline_sse = sum(float(row["best_single_trajectory_observed_step_sse"]) for row in tier_rows)
        baseline_count = sum(int(row["best_single_trajectory_observed_steps"]) for row in tier_rows)
        selected_rmse = rmse_from_sse_count(selected_sse, selected_count)
        baseline_rmse = rmse_from_sse_count(baseline_sse, baseline_count)
        gains = [
            float(row["gain_vs_best_single_trajectory_percent"])
            for row in tier_rows
            if row["gain_vs_best_single_trajectory_percent"] is not None
            and math.isfinite(float(row["gain_vs_best_single_trajectory_percent"]))
        ]
        wins = ties = losses = 0
        for row in tier_rows:
            selected = float(row["selected_observed_step_rmse_m"])
            baseline = float(row["best_single_trajectory_observed_step_rmse_m"])
            if not (math.isfinite(selected) and math.isfinite(baseline)):
                continue
            if selected < baseline - 1.0e-9:
                wins += 1
            elif selected > baseline + 1.0e-9:
                losses += 1
            else:
                ties += 1
        aggregates[tier] = {
            "rows": len(tier_rows),
            "source_scenarios": len({(row["source_name"], row["scenario"]) for row in tier_rows}),
            "observed_steps": selected_count,
            "selector_observed_step_rmse_m": selected_rmse,
            "best_single_observed_step_rmse_m": baseline_rmse,
            "gain_vs_best_single_percent": relative_gain_percent(baseline_rmse, selected_rmse),
            "row_wins": wins,
            "row_ties": ties,
            "row_losses": losses,
            "mean_row_gain_percent": float(np.mean(gains)) if gains else None,
            "median_row_gain_percent": float(np.median(gains)) if gains else None,
            "selected_method_counts": dict(Counter(str(row["selected_candidate_method"]) for row in tier_rows)),
            "baseline_method_counts": dict(Counter(str(row["best_single_candidate_method"]) for row in tier_rows)),
        }
    return aggregates


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_for_json(value.item())
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return sanitize_for_json(asdict(value))
    return str(value)


def write_strict_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = sanitize_for_json(payload)
    path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: sanitize_for_json(row.get(key)) for key in fieldnames})


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.6g}"


def write_summary_md(summary: dict[str, Any], path: Path) -> None:
    aggregates = summary["aggregate_tiers"]
    lines = [
        "# Trajectory Candidate Graph Selector PoC",
        "",
        f"Boundary: {BOUNDARY_STATEMENT}",
        "",
        f"Selector candidate methods: {', '.join(summary.get('candidate_methods', []))}",
        f"Best-single baseline candidate methods: {', '.join(summary.get('baseline_candidate_methods', []))}",
        f"Graph layer type: {summary.get('graph_layer_type', 'mean')}",
        f"Graph layers: {summary.get('graph_layers', 0)}",
        "",
        "Selection uses retained truth-free candidate, visibility, eval-mask, scenario, and candidate-disagreement features only. Eval truth is used for scoring rows and baselines, not for selector decisions.",
        "",
        "| Tier | Rows | Observed steps | Selector RMSE m | Best single RMSE m | Gain % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier in TIER_NAMES:
        item = aggregates[tier]
        lines.append(
            "| "
            + " | ".join(
                [
                    tier,
                    str(item["rows"]),
                    str(item["observed_steps"]),
                    _format_metric(item["selector_observed_step_rmse_m"]),
                    _format_metric(item["best_single_observed_step_rmse_m"]),
                    _format_metric(item["gain_vs_best_single_percent"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The best-single denominator is selected per retained source run and scenario from the baseline candidate methods by observed-step RMSE.",
            "This artifact is not independent-machine reproduction, not operational precise-reference validation, and not a full raw/training/all-filter rerun.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if int(args.epochs) <= 0:
        raise SystemExit("--epochs must be positive.")
    if int(args.hidden_dim) <= 0:
        raise SystemExit("--hidden-dim must be positive.")
    if int(args.graph_layers) < 0:
        raise SystemExit("--graph-layers must be non-negative.")
    if int(args.batch_size) <= 0:
        raise SystemExit("--batch-size must be positive.")
    if float(args.learning_rate) <= 0.0:
        raise SystemExit("--learning-rate must be positive.")
    if float(args.weight_decay) < 0.0:
        raise SystemExit("--weight-decay must be non-negative.")
    if not (0.0 <= float(args.dropout) < 1.0):
        raise SystemExit("--dropout must be in [0, 1).")
    if int(args.development_seed_max_exclusive) > int(args.holdout_seed_min):
        raise SystemExit("--development-seed-max-exclusive must be <= --holdout-seed-min.")
    try:
        resolve_ensemble_member_seeds(
            base_seed=int(args.seed),
            ensemble_size=int(args.ensemble_size),
            ensemble_seeds=args.ensemble_seeds,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    started = time.perf_counter()
    args = build_parser().parse_args()
    validate_args(args)
    ensemble_size = int(args.ensemble_size)
    member_seeds = resolve_ensemble_member_seeds(
        base_seed=int(args.seed),
        ensemble_size=ensemble_size,
        ensemble_seeds=args.ensemble_seeds,
    )
    seed_all(int(args.seed))
    device = resolve_training_device(args.device, allow_cpu_smoke=bool(args.allow_cpu_smoke))
    source_runs = resolve_source_runs(args.source_glob, list(args.extra_source_dir))
    scenarios = parse_csv(args.scenarios)
    candidate_methods = parse_candidate_methods(args.candidate_methods)
    baseline_candidate_methods = parse_candidate_methods(
        args.baseline_candidate_methods,
        option_name="--baseline-candidate-methods",
        min_count=1,
    )
    if not scenarios:
        raise SystemExit("--scenarios must contain at least one scenario.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        source_runs=source_runs,
        scenarios=scenarios,
        candidate_methods=candidate_methods,
        baseline_candidate_methods=baseline_candidate_methods,
    )
    train_samples = [
        sample
        for sample in samples
        if (not sample.source_is_extra) and sample.seed < int(args.development_seed_max_exclusive)
    ]
    if not train_samples:
        raise SystemExit("no development training samples found with seed < --development-seed-max-exclusive.")
    node_feature_dim = int(train_samples[0].node_features.shape[-1])
    edge_feature_dim = int(train_samples[0].edge_features.shape[-1])
    feature_names = build_feature_names(scenarios, candidate_methods)
    edge_feature_names = build_edge_feature_names()
    graph_layers = int(args.graph_layers)
    graph_layer_type = str(args.graph_layer_type)
    message_passing_enabled = bool(graph_layers > 0)
    model_kwargs = {
        "node_feature_dim": node_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "hidden_dim": int(args.hidden_dim),
        "dropout": float(args.dropout),
        "graph_layers": graph_layers,
        "graph_layer_type": graph_layer_type,
    }
    evaluation_probability_aggregation = "arithmetic_mean" if ensemble_size > 1 else "single_member"
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "boundary": BOUNDARY_STATEMENT,
        "source_glob": args.source_glob,
        "extra_source_dirs": [str(Path(path)) for path in args.extra_source_dir],
        "output_dir": str(output_dir),
        "scenarios": scenarios,
        "candidate_methods": candidate_methods,
        "baseline_candidate_methods": baseline_candidate_methods,
        "development_seed_max_exclusive": int(args.development_seed_max_exclusive),
        "holdout_seed_min": int(args.holdout_seed_min),
        "future_seed_min": int(args.future_seed_min),
        "epochs": int(args.epochs),
        "hidden_dim": int(args.hidden_dim),
        "graph_layers": graph_layers,
        "graph_layer_type": graph_layer_type,
        "message_passing_enabled": message_passing_enabled,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "ensemble_size": ensemble_size,
        "ensemble_seeds": args.ensemble_seeds,
        "ensemble_member_seeds": member_seeds,
        "evaluation_probability_aggregation": evaluation_probability_aggregation,
        "evaluation_used_averaged_probabilities": bool(ensemble_size > 1),
        "device": str(device),
        "allow_cpu_smoke": bool(args.allow_cpu_smoke),
        "batch_size": int(args.batch_size),
        "source_runs": [asdict(run) for run in source_runs],
        "feature_names": feature_names,
        "edge_feature_names": edge_feature_names,
        "model_kwargs": model_kwargs,
    }
    models: list[TrajectoryCandidateGraphSelector] = []
    member_training_summaries: list[dict[str, Any]] = []
    for member_seed in member_seeds:
        seed_all(member_seed)
        model = TrajectoryCandidateGraphSelector(**model_kwargs)
        checkpoint_dir = (
            output_dir / "checkpoints"
            if ensemble_size == 1
            else output_dir / "checkpoints" / f"member_seed{member_seed}"
        )
        member_config = {
            **run_config,
            "member_seed": int(member_seed),
            "member_checkpoint_dir": str(checkpoint_dir),
        }
        training = train_selector_model(
            train_samples=train_samples,
            model=model,
            output_dir=output_dir,
            device=device,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            seed=int(member_seed),
            model_kwargs=model_kwargs,
            candidate_methods=candidate_methods,
            feature_names=feature_names,
            edge_feature_names=edge_feature_names,
            config=member_config,
            checkpoint_dir=checkpoint_dir,
        )
        member_training_summaries.append(
            {
                "member_seed": int(member_seed),
                "checkpoint_dir": str(checkpoint_dir),
                **training,
            }
        )
        models.append(model)

    probabilities = predict_ensemble_probabilities(
        models=models,
        samples=samples,
        device=device,
        batch_size=int(args.batch_size),
    )
    rows = evaluate_samples_from_probabilities(
        probabilities=probabilities,
        samples=samples,
        candidate_methods=candidate_methods,
        baseline_candidate_methods=baseline_candidate_methods,
        development_seed_max_exclusive=int(args.development_seed_max_exclusive),
        holdout_seed_min=int(args.holdout_seed_min),
        future_seed_min=int(args.future_seed_min),
    )
    ensemble_metadata = {
        "size": ensemble_size,
        "member_seeds": member_seeds,
        "probability_aggregation": evaluation_probability_aggregation,
        "evaluation_used_averaged_probabilities": bool(ensemble_size > 1),
        "members": member_training_summaries,
    }
    summary = {
        **run_config,
        "boundary_statement": BOUNDARY_STATEMENT,
        "train_sample_count": len(train_samples),
        "total_sample_count": len(samples),
        "training": member_training_summaries[0]
        if ensemble_size == 1
        else {"members": member_training_summaries},
        "training_members": member_training_summaries,
        "ensemble": ensemble_metadata,
        "aggregate_tiers": aggregate_tier_rows(rows),
        "rows": rows,
        "duration_s": float(time.perf_counter() - started),
    }
    write_rows_csv(rows, output_dir / "rows.csv")
    write_strict_json(summary, output_dir / "summary.json")
    write_summary_md(summary, output_dir / "summary.md")
    print(f"Wrote trajectory candidate graph selector PoC outputs under {output_dir}")


if __name__ == "__main__":
    main()
