#!/usr/bin/env python
"""Analyze validation-selected global AdaptiveCandidateFusion portfolios."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import csv
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import scripts.run_adaptive_candidate_fusion_poc as poc
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import run_adaptive_candidate_fusion_poc as poc


SCHEMA_VERSION = "adaptive_candidate_fusion_global_portfolio.v1"
DEFAULT_SCENARIOS = ("process_noise_shift_test", "maneuver_shift_test")
DEFAULT_RUN_DIRS = (
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed7_split7_20260623"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed11_split11_20260623"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed13_split13_20260623"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed17_split17_20260623"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed19_split19_20260623"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed23_split23_20260623"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed29_split29_20260623"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed31_split31_20260624"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed37_split37_20260624"),
    Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed41_split41_20260624"),
)
DEFAULT_OUTPUT_DIR = Path("results/adaptive_candidate_fusion_global_portfolio_20260624")
PREDICTION_NPZ_NAME = "adaptive_candidate_fusion_predictions.npz"
RUN_CONFIG_NAME = "run_config_summary.json"
SUMMARY_JSON_NAME = "summary.json"
SUMMARY_CSV_NAME = "summary.csv"
SUMMARY_MD_NAME = "summary.md"
DEFAULT_BOOTSTRAP_SAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 12_345
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
LEARNED_COMPONENT = "learned"
LEARNED_HARD_COMPONENT = "learned_hard"
LEARNED_POLICY_COMPONENTS = frozenset({LEARNED_COMPONENT, LEARNED_HARD_COMPONENT})
POLICY_FAMILIES = ("all", "learned_including", "nonlearned_only")
BOUNDARY_LANGUAGE = (
    "Internal compact-simulator validation-selected AdaptiveCandidateFusion portfolio "
    "evidence. Policies are selected from each run's stored validation split and applied "
    "to that run's held-out compact-simulator eval rows. This is not operational "
    "precise-reference validation, independent-machine reproduction, third-party "
    "validation, or a claim of universal learned orbit-determination performance."
)
RESULT_DIR_RE = re.compile(r"seed(?P<seed>\d+)_split(?P<split>\d+)")


@dataclass(frozen=True)
class PortfolioRecord:
    run_dir: Path
    run_name: str
    seed: int | None
    split: int | None
    scenario: str
    role: str
    states: np.ndarray
    visibility: np.ndarray
    eval_mask: np.ndarray
    components: dict[str, np.ndarray]
    candidate_methods: list[str]
    trajectory_indices: list[int]


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    run_name: str
    seed: int | None
    split: int | None
    training_seed: int | None
    config: dict[str, Any]
    cfg: dict[str, Any]
    model: Any
    candidate_methods: list[str]
    model_kwargs: dict[str, Any]
    lookback: int
    lookahead: int
    candidate_residual_features: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select global per-scenario AdaptiveCandidateFusion portfolio policies "
            "from pooled validation splits and apply them to saved held-out eval rows."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        help="Run directories containing run_config_summary.json and saved eval NPZs.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=None,
        help="Additional run directory. May be repeated.",
    )
    parser.add_argument(
        "--summary-json",
        action="append",
        type=Path,
        default=None,
        help="Aggregate summary JSON containing a top-level run_dirs list. May be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--blend-grid-step", type=float, default=0.05)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Deterministic percentile-bootstrap samples for mean gain uncertainty diagnostics.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for deterministic percentile-bootstrap uncertainty diagnostics.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=poc.PORTFOLIO_SELECTION_METRIC_CHOICES,
        default="observed_step_pos_rmse_m",
    )
    return parser


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _display_path(path: Path, root: Path | None = None) -> str:
    root = Path.cwd() if root is None else Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_seed_split(run_dir: Path, config: dict[str, Any] | None = None) -> tuple[int | None, int | None]:
    match = RESULT_DIR_RE.search(run_dir.name)
    seed = int(match.group("seed")) if match is not None else None
    split = int(match.group("split")) if match is not None else None
    if config is not None:
        split_plan = config.get("scenario_trajectory_splits", {})
        if isinstance(split_plan, dict) and split_plan.get("split_seed") is not None:
            split = int(split_plan["split_seed"])
            if seed is None:
                seed = split
    return seed, split


def _training_seed_from_config(config: dict[str, Any]) -> int | None:
    return int(config["seed"]) if config.get("seed") is not None else None


def resolve_run_dirs(
    *,
    positional: Iterable[Path],
    repeated: Iterable[Path] | None,
    summary_json: Iterable[Path] | None,
) -> list[Path]:
    paths: list[Path] = [Path(path) for path in positional]
    if repeated:
        paths.extend(Path(path) for path in repeated)
    for summary_path in summary_json or []:
        data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        run_dirs = data.get("run_dirs")
        if not isinstance(run_dirs, list):
            raise ValueError(f"{summary_path}: expected a top-level run_dirs list")
        paths.extend(Path(str(path)) for path in run_dirs)
    if not paths:
        paths = list(DEFAULT_RUN_DIRS)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def observed_step_mask(visibility: np.ndarray) -> np.ndarray:
    visibility_arr = np.asarray(visibility)
    if visibility_arr.ndim < 2:
        raise ValueError("visibility must have at least trajectory and step dimensions.")
    if visibility_arr.ndim == 2:
        return np.isfinite(visibility_arr) & (visibility_arr >= 0.5)
    return np.any(np.isfinite(visibility_arr) & (visibility_arr >= 0.5), axis=-1)


def metric_sse_count(
    *,
    states: np.ndarray,
    predictions: np.ndarray,
    visibility: np.ndarray,
    eval_mask: np.ndarray,
    metric: str,
) -> tuple[float, int]:
    metric = str(metric)
    if metric not in poc.PORTFOLIO_SELECTION_METRIC_CHOICES:
        raise ValueError(f"unsupported metric {metric!r}")
    states_arr = np.asarray(states, dtype=np.float64)
    pred_arr = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(eval_mask, dtype=bool)
    if states_arr.shape != pred_arr.shape:
        raise ValueError(f"states and predictions shape mismatch: {states_arr.shape} != {pred_arr.shape}")
    if states_arr.ndim != 3 or states_arr.shape[-1] < 3:
        raise ValueError("states and predictions must have shape [N, T, >=3].")
    if mask.shape != states_arr.shape[:2]:
        raise ValueError(f"eval_mask must have shape {states_arr.shape[:2]}, got {mask.shape}.")
    if metric == "observed_step_pos_rmse_m":
        obs = observed_step_mask(visibility)
        if obs.shape != mask.shape:
            raise ValueError(f"observed mask must have shape {mask.shape}, got {obs.shape}.")
        mask = mask & obs
    finite = np.all(np.isfinite(pred_arr), axis=-1)
    mask = mask & finite
    count = int(np.sum(mask))
    if count == 0:
        return 0.0, 0
    err = states_arr[mask, :3] - pred_arr[mask, :3]
    return float(np.sum(err * err)), count


def masked_pos_rmse(
    *,
    states: np.ndarray,
    predictions: np.ndarray,
    visibility: np.ndarray,
    eval_mask: np.ndarray,
    metric: str,
) -> float:
    sse, count = metric_sse_count(
        states=states,
        predictions=predictions,
        visibility=visibility,
        eval_mask=eval_mask,
        metric=metric,
    )
    return float(math.sqrt(sse / count)) if count > 0 else float("nan")


def policy_grid(candidate_methods: list[str], blend_grid_step: float) -> list[dict[str, Any]]:
    blend_grid = [
        alpha
        for alpha in poc.portfolio_blend_grid(float(blend_grid_step))
        if 0.0 < float(alpha) < 1.0
    ]
    components = [LEARNED_COMPONENT, LEARNED_HARD_COMPONENT, *list(candidate_methods)]
    policies: list[dict[str, Any]] = []
    for rank, component in enumerate(components):
        policies.append(
            {
                "kind": "single",
                "components": [component],
                "alpha": None,
                "policy_id": f"single:{component}",
                "selection_rank": [0, rank, 0],
            }
        )
    pair_rank = 0
    for component_a, component_b in itertools.combinations(components, 2):
        for alpha_rank, alpha in enumerate(blend_grid):
            alpha_value = float(alpha)
            policies.append(
                {
                    "kind": "blend",
                    "components": [component_a, component_b],
                    "alpha": alpha_value,
                    "policy_id": f"blend:{component_a}:{component_b}:{alpha_value:.12g}",
                    "selection_rank": [1, pair_rank, alpha_rank],
                }
            )
        pair_rank += 1
    return policies


def policy_includes_learned_component(policy: dict[str, Any]) -> bool:
    components = {str(value) for value in policy.get("components", [])}
    return bool(components & LEARNED_POLICY_COMPONENTS)


def filter_policy_grid_by_family(
    policies: Iterable[dict[str, Any]],
    policy_family: str,
) -> list[dict[str, Any]]:
    family = str(policy_family)
    if family == "all":
        return list(policies)
    if family == "learned_including":
        return [policy for policy in policies if policy_includes_learned_component(policy)]
    if family == "nonlearned_only":
        return [policy for policy in policies if not policy_includes_learned_component(policy)]
    raise ValueError(f"unsupported policy family {policy_family!r}; expected one of {POLICY_FAMILIES}")


def apply_policy(policy: dict[str, Any], components: dict[str, np.ndarray]) -> np.ndarray:
    kind = str(policy.get("kind"))
    names = [str(value) for value in policy.get("components", [])]
    if kind == "single":
        if len(names) != 1 or names[0] not in components:
            raise ValueError(f"invalid single policy components: {names!r}")
        return np.asarray(components[names[0]], dtype=np.float64)
    if kind == "blend":
        if len(names) != 2 or names[0] not in components or names[1] not in components:
            raise ValueError(f"invalid blend policy components: {names!r}")
        alpha = float(policy.get("alpha"))
        if not math.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"invalid interior blend alpha: {policy.get('alpha')!r}")
        return alpha * np.asarray(components[names[0]], dtype=np.float64) + (
            1.0 - alpha
        ) * np.asarray(components[names[1]], dtype=np.float64)
    raise ValueError(f"unknown policy kind {kind!r}")


def _candidate_methods_for_records(records: list[PortfolioRecord]) -> list[str]:
    if not records:
        raise ValueError("at least one record is required")
    methods = list(records[0].candidate_methods)
    for record in records[1:]:
        if record.candidate_methods != methods:
            raise ValueError(
                "global portfolio selection requires aligned candidate_methods; "
                f"{record.run_name}/{record.scenario} has {record.candidate_methods}, expected {methods}."
            )
    return methods


def select_global_policy(
    records: list[PortfolioRecord],
    *,
    selection_metric: str,
    blend_grid_step: float,
    selection_source: str,
    policy_family: str | None = None,
) -> dict[str, Any]:
    selection_metric = poc._validate_portfolio_selection_metric(selection_metric)
    candidate_methods = _candidate_methods_for_records(records)
    policies = policy_grid(candidate_methods, blend_grid_step)
    if policy_family is not None:
        policies = filter_policy_grid_by_family(policies, policy_family)
    if not policies:
        raise ValueError(f"no portfolio policies are available for family {policy_family!r}")
    best_key: tuple[float, int, int, int] | None = None
    best_policy: dict[str, Any] | None = None
    diagnostics: list[dict[str, Any]] = []
    for policy in policies:
        total_sse = 0.0
        total_count = 0
        for record in records:
            preds = apply_policy(policy, record.components)
            sse, count = metric_sse_count(
                states=record.states,
                predictions=preds,
                visibility=record.visibility,
                eval_mask=record.eval_mask,
                metric=selection_metric,
            )
            total_sse += sse
            total_count += count
        rmse = float(math.sqrt(total_sse / total_count)) if total_count > 0 else float("nan")
        rank = [int(value) for value in policy["selection_rank"]]
        sort_rmse = rmse if np.isfinite(rmse) else float("inf")
        key = (sort_rmse, rank[0], rank[1], rank[2])
        diagnostic = {
            "policy_id": policy["policy_id"],
            "kind": policy["kind"],
            "components": list(policy["components"]),
            "alpha": policy["alpha"],
            "validation_rmse_m": rmse,
            "validation_sse": float(total_sse),
            "evaluated_validation_steps": int(total_count),
            "selection_rank": rank,
        }
        diagnostics.append(diagnostic)
        if best_key is None or key < best_key:
            best_key = key
            best_policy = diagnostic
    if best_policy is None:
        raise ValueError("no portfolio policy could be selected")
    selected = dict(best_policy)
    selected.update(
        {
            "selection_metric": selection_metric,
            "selection_source": selection_source,
            "validation_record_count": len(records),
            "validation_scenarios": sorted({record.scenario for record in records}),
            "validation_run_names": [record.run_name for record in records],
            "candidate_methods": candidate_methods,
        }
    )
    if policy_family is not None:
        selected["policy_family"] = str(policy_family)
    diagnostics.sort(
        key=lambda item: (
            float(item["validation_rmse_m"]) if np.isfinite(float(item["validation_rmse_m"])) else float("inf"),
            *[int(value) for value in item["selection_rank"]],
        )
    )
    selected["top_validation_policies"] = diagnostics[:10]
    return selected


def _rmse_summary(record: PortfolioRecord, component: str, metric: str) -> dict[str, float | int]:
    preds = record.components[component]
    sse, count = metric_sse_count(
        states=record.states,
        predictions=preds,
        visibility=record.visibility,
        eval_mask=record.eval_mask,
        metric=metric,
    )
    return {
        "rmse_m": float(math.sqrt(sse / count)) if count > 0 else float("nan"),
        "sse": float(sse),
        "steps": int(count),
    }


def best_input_candidate(record: PortfolioRecord, *, metric: str) -> dict[str, Any]:
    best_key: tuple[float, int] | None = None
    best: dict[str, Any] | None = None
    for idx, method in enumerate(record.candidate_methods):
        summary = _rmse_summary(record, method, metric)
        rmse = float(summary["rmse_m"])
        sort_rmse = rmse if np.isfinite(rmse) else float("inf")
        key = (sort_rmse, idx)
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "method": method,
                "rmse_m": rmse,
                "sse": float(summary["sse"]),
                "steps": int(summary["steps"]),
            }
    if best is None:
        raise ValueError(f"{record.run_name}/{record.scenario}: no input candidates are available")
    return best


def gain_percent(reference_rmse: float, candidate_rmse: float) -> float:
    if not np.isfinite(reference_rmse) or not np.isfinite(candidate_rmse) or abs(reference_rmse) < 1.0e-12:
        return float("nan")
    return (float(reference_rmse) - float(candidate_rmse)) / float(reference_rmse) * 100.0


def result_from_gain(gain: float) -> str:
    if not np.isfinite(gain):
        return "invalid"
    if math.isclose(float(gain), 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return "tie"
    return "win" if gain > 0.0 else "loss"


def exact_one_sided_sign_p_value(wins: int, nonpositive: int) -> float:
    """Exact P[X >= wins] for X ~ Binomial(wins + nonpositive, 0.5)."""

    wins = int(wins)
    nonpositive = int(nonpositive)
    if wins < 0 or nonpositive < 0:
        raise ValueError("wins and nonpositive counts must be nonnegative")
    trials = wins + nonpositive
    if trials == 0:
        return float("nan")
    tail_count = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return float(math.ldexp(tail_count, -trials))


def _finite_gain_array(gains: Iterable[float]) -> np.ndarray:
    arr = np.asarray([float(gain) for gain in gains], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _gain_counts(gains: np.ndarray) -> dict[str, int]:
    wins = int(np.sum(gains > 1.0e-12))
    ties = int(np.sum(np.isclose(gains, 0.0, rtol=0.0, atol=1.0e-12)))
    losses = int(np.sum(gains < -1.0e-12))
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "nonpositive": ties + losses,
    }


def bootstrap_mean_gain_percent_ci(
    gains: Iterable[float],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> list[float]:
    """Deterministic percentile bootstrap CI for the mean of finite gain values."""

    samples = int(bootstrap_samples)
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    confidence = float(confidence_level)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    finite = _finite_gain_array(gains)
    if finite.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, finite.size, size=(samples, finite.size))
    means = np.mean(finite[indices], axis=1)
    alpha = 1.0 - confidence
    quantiles = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(quantiles[0]), float(quantiles[1])]


def gain_summary_statistics(
    gains: Iterable[float],
    *,
    total_count: int | None = None,
    n_key: str = "rows",
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Summarize finite percent gains with sign-test and bootstrap diagnostics."""

    finite = _finite_gain_array(gains)
    counts = _gain_counts(finite)
    ci = bootstrap_mean_gain_percent_ci(
        finite,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    out = {
        n_key: int(finite.size if total_count is None else total_count),
        f"finite_gain_{n_key}": int(finite.size),
        "wins": counts["wins"],
        "ties": counts["ties"],
        "losses": counts["losses"],
        "nonpositive": counts["nonpositive"],
        "mean_gain_percent": float(np.mean(finite)) if finite.size else float("nan"),
        "median_gain_percent": float(np.median(finite)) if finite.size else float("nan"),
        "min_gain_percent": float(np.min(finite)) if finite.size else float("nan"),
        "max_gain_percent": float(np.max(finite)) if finite.size else float("nan"),
        "exact_one_sided_sign_p_value": exact_one_sided_sign_p_value(
            counts["wins"],
            counts["nonpositive"],
        ),
        "bootstrap_mean_gain_percent_ci95": ci,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_confidence_level": float(confidence_level),
        "bootstrap_method": "percentile",
    }
    return out


def row_gain_statistics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    gains = [float(row["gain_vs_best_input_observed_step_percent"]) for row in rows]
    return gain_summary_statistics(
        gains,
        total_count=len(rows),
        n_key="rows",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )


def scenario_gain_statistics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        scenario_rows = [row for row in rows if str(row["scenario"]) == scenario]
        out[scenario] = row_gain_statistics(
            scenario_rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
    return out


def seed_paired_gain_statistics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    gains_by_seed: dict[int, list[float]] = {}
    for row in rows:
        if row.get("seed") is None:
            raise ValueError("seed-paired statistics require every row to include seed")
        gain = float(row["gain_vs_best_input_observed_step_percent"])
        if np.isfinite(gain):
            gains_by_seed.setdefault(int(row["seed"]), []).append(gain)

    seed_mean_gains = [
        {
            "seed": seed,
            "mean_gain_percent": float(np.mean(seed_gains)),
        }
        for seed, seed_gains in sorted(gains_by_seed.items())
        if seed_gains
    ]
    seed_means = [item["mean_gain_percent"] for item in seed_mean_gains]
    summary = gain_summary_statistics(
        seed_means,
        total_count=len(seed_mean_gains),
        n_key="seeds",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    summary["seed_wins"] = summary.pop("wins")
    summary["seed_ties"] = summary.pop("ties")
    summary["seed_losses"] = summary.pop("losses")
    summary["seed_nonpositive"] = summary.pop("nonpositive")
    summary["seed_mean_gain_percent"] = summary.pop("mean_gain_percent")
    summary["seed_median_gain_percent"] = summary.pop("median_gain_percent")
    summary["seed_min_gain_percent"] = summary.pop("min_gain_percent")
    summary["seed_max_gain_percent"] = summary.pop("max_gain_percent")
    summary["bootstrap_seed_mean_gain_percent_ci95"] = summary.pop("bootstrap_mean_gain_percent_ci95")
    summary["seed_mean_gains"] = seed_mean_gains
    summary["seed_mean_gain_percent_values"] = seed_means
    return summary


def eval_gain_statistics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    return {
        "rows": row_gain_statistics(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "by_scenario": scenario_gain_statistics(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "seed_paired": seed_paired_gain_statistics(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
    }


def evaluate_policy_on_record(
    record: PortfolioRecord,
    policy: dict[str, Any],
    *,
    comparison_metric: str = "observed_step_pos_rmse_m",
) -> dict[str, Any]:
    preds = apply_policy(policy, record.components)
    policy_sse, policy_steps = metric_sse_count(
        states=record.states,
        predictions=preds,
        visibility=record.visibility,
        eval_mask=record.eval_mask,
        metric=comparison_metric,
    )
    policy_rmse = float(math.sqrt(policy_sse / policy_steps)) if policy_steps > 0 else float("nan")
    best_input = best_input_candidate(record, metric=comparison_metric)
    gain = gain_percent(float(best_input["rmse_m"]), policy_rmse)
    all_step_rmse = masked_pos_rmse(
        states=record.states,
        predictions=preds,
        visibility=record.visibility,
        eval_mask=record.eval_mask,
        metric="all_step_pos_rmse_m",
    )
    return {
        "run_name": record.run_name,
        "run_dir": _display_path(record.run_dir),
        "seed": record.seed,
        "split": record.split,
        "scenario": record.scenario,
        "policy_id": policy["policy_id"],
        "policy_kind": policy["kind"],
        "policy_components": list(policy["components"]),
        "policy_alpha": policy.get("alpha"),
        "policy_validation_rmse_m": policy.get("validation_rmse_m"),
        "policy_validation_steps": policy.get("evaluated_validation_steps"),
        "portfolio_observed_step_pos_rmse_m": policy_rmse,
        "portfolio_observed_step_sse": float(policy_sse),
        "portfolio_observed_steps": int(policy_steps),
        "portfolio_all_step_pos_rmse_m": all_step_rmse,
        "best_input_candidate_method": best_input["method"],
        "best_input_candidate_observed_step_pos_rmse_m": float(best_input["rmse_m"]),
        "best_input_candidate_observed_steps": int(best_input["steps"]),
        "gain_vs_best_input_observed_step_percent": gain,
        "result_vs_best_input": result_from_gain(gain),
        "candidate_methods": list(record.candidate_methods),
        "trajectory_indices": list(record.trajectory_indices),
    }


def aggregate_eval_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty eval rows")
    gains = [float(row["gain_vs_best_input_observed_step_percent"]) for row in rows]
    finite_gains = [gain for gain in gains if np.isfinite(gain)]
    wins = sum(1 for row in rows if row["result_vs_best_input"] == "win")
    ties = sum(1 for row in rows if row["result_vs_best_input"] == "tie")
    losses = sum(1 for row in rows if row["result_vs_best_input"] == "loss")
    return {
        "rows": len(rows),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_gain_percent": float(np.mean(finite_gains)) if finite_gains else float("nan"),
        "median_gain_percent": float(np.median(finite_gains)) if finite_gains else float("nan"),
        "min_gain_percent": float(np.min(finite_gains)) if finite_gains else float("nan"),
        "max_gain_percent": float(np.max(finite_gains)) if finite_gains else float("nan"),
    }


def scenario_eval_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}):
        out[scenario] = aggregate_eval_rows([row for row in rows if str(row["scenario"]) == scenario])
    return out


def _prediction_npz_key(method_name: str) -> str:
    if method_name == "VA_RFIS":
        return "va_rfis"
    if method_name == "BatchWLS":
        return "batchwls"
    return method_name.lower()


def _stack_candidate_bank(components: dict[str, np.ndarray], candidate_methods: list[str]) -> np.ndarray:
    return np.stack([components[method] for method in candidate_methods], axis=2)


def load_eval_record(run_dir: Path, scenario: str, config: dict[str, Any] | None = None) -> PortfolioRecord:
    run_dir = Path(run_dir)
    npz_path = run_dir / scenario / PREDICTION_NPZ_NAME
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if config is None:
        config_path = run_dir / RUN_CONFIG_NAME
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    seed, split = _parse_seed_split(run_dir, config)
    with np.load(npz_path, allow_pickle=False) as data:
        candidate_methods = [str(value) for value in data["candidate_methods"].tolist()]
        states = np.asarray(data["states"], dtype=np.float64)
        visibility = np.asarray(data["visibility"], dtype=np.float64)
        eval_mask = np.asarray(data["eval_mask"], dtype=bool)
        trajectory_source = (
            data["trajectory_indices"] if "trajectory_indices" in data.files else np.arange(states.shape[0])
        )
        trajectory_indices = [int(value) for value in trajectory_source]
        components: dict[str, np.ndarray] = {
            LEARNED_COMPONENT: np.asarray(
                data["adaptive_candidate_fusion_soft"]
                if "adaptive_candidate_fusion_soft" in data.files
                else data["adaptive_candidate_fusion"],
                dtype=np.float64,
            )
        }
        for method in candidate_methods:
            key = _prediction_npz_key(method)
            if key not in data.files:
                raise KeyError(f"{npz_path}: missing candidate prediction key {key!r} for {method}")
            components[method] = np.asarray(data[key], dtype=np.float64)
        if "adaptive_candidate_fusion_hard_argmax" in data.files:
            components[LEARNED_HARD_COMPONENT] = np.asarray(
                data["adaptive_candidate_fusion_hard_argmax"],
                dtype=np.float64,
            )
        else:
            if "fusion_weights" not in data.files:
                raise KeyError(f"{npz_path}: missing hard-argmax prediction and fusion_weights")
            components[LEARNED_HARD_COMPONENT] = poc.hard_argmax_candidate_predictions(
                weights=np.asarray(data["fusion_weights"], dtype=np.float64),
                candidate_bank=_stack_candidate_bank(components, candidate_methods),
            )
    return PortfolioRecord(
        run_dir=run_dir,
        run_name=run_dir.name,
        seed=seed,
        split=split,
        scenario=scenario,
        role="eval",
        states=states,
        visibility=visibility,
        eval_mask=eval_mask,
        components=components,
        candidate_methods=candidate_methods,
        trajectory_indices=trajectory_indices,
    )


def load_run_context(run_dir: Path, *, device: Any) -> RunContext:
    run_dir = Path(run_dir)
    config_path = run_dir / RUN_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = poc.load_yaml(Path(str(config["config_path"])))
    checkpoint_path = Path(str(config["best_checkpoint"]))
    model, ckpt = poc.load_fusion_model(checkpoint_path, device)
    candidate_methods, model_kwargs, lookback, lookahead = poc.checkpoint_run_metadata(ckpt, checkpoint_path)
    seed, split = _parse_seed_split(run_dir, config)
    return RunContext(
        run_dir=run_dir,
        run_name=run_dir.name,
        seed=seed,
        split=split,
        training_seed=_training_seed_from_config(config),
        config=config,
        cfg=cfg,
        model=model,
        candidate_methods=candidate_methods,
        model_kwargs=model_kwargs,
        lookback=int(lookback),
        lookahead=int(lookahead),
        candidate_residual_features=poc.candidate_residual_feature_mode_from_model_kwargs(model_kwargs),
    )


def recompute_validation_record(
    context: RunContext,
    scenario: str,
    *,
    eval_batch_size: int,
) -> PortfolioRecord:
    config = context.config
    split_plan = config.get("scenario_trajectory_splits")
    if not isinstance(split_plan, dict):
        raise ValueError(f"{context.run_dir}: run_config_summary.json is missing scenario_trajectory_splits")
    bundle = poc.load_fusion_scenario(
        data_dir=Path(str(config["data_dir"])),
        cfg=context.cfg,
        scenario=scenario,
        role="val",
        split_plan=split_plan,
        candidate_methods=context.candidate_methods,
        batch_wls_dirs=[Path(str(path)) for path in config.get("batch_wls_dirs", [])],
        rfis_dir=Path(str(config.get("rfis_dir", poc.DEFAULT_RFIS_DIR))),
        va_rfis_dir=Path(str(config.get("va_rfis_dir", poc.DEFAULT_VA_RFIS_DIR))),
        use_innovation_features=bool(context.model_kwargs.get("use_innovation_features", True)),
        oracle_weighting=str(config.get("oracle_weighting", "uniform")),
        oracle_weight_clip=float(config.get("oracle_weight_clip", 50.0)),
        candidate_residual_features=context.candidate_residual_features,
    )
    soft, weights = poc.run_candidate_fusion_inference_batched(
        model=context.model,
        arrays=bundle.arrays,
        candidate_bank=bundle.candidate_bank,
        candidate_residual_features=bundle.candidate_residual_features,
        lookback=context.lookback,
        lookahead=context.lookahead,
        batch_size=int(eval_batch_size),
    )
    components: dict[str, np.ndarray] = {
        LEARNED_COMPONENT: np.asarray(soft, dtype=np.float64),
        LEARNED_HARD_COMPONENT: poc.hard_argmax_candidate_predictions(
            weights=weights,
            candidate_bank=bundle.candidate_bank,
        ),
    }
    for idx, method in enumerate(bundle.candidate_methods):
        components[method] = np.asarray(bundle.candidate_bank[:, :, idx, :], dtype=np.float64)
    return PortfolioRecord(
        run_dir=context.run_dir,
        run_name=context.run_name,
        seed=context.seed,
        split=context.split,
        scenario=scenario,
        role="val",
        states=np.asarray(bundle.arrays.states, dtype=np.float64),
        visibility=np.asarray(bundle.arrays.visibility, dtype=np.float64),
        eval_mask=poc.centered_eval_mask(bundle.arrays.states.shape, context.lookback, context.lookahead),
        components=components,
        candidate_methods=list(bundle.candidate_methods),
        trajectory_indices=list(bundle.trajectory_indices),
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    seed = int(row["seed"]) if row.get("seed") is not None else 10**9
    scenario = str(row["scenario"])
    order = {scenario_name: idx for idx, scenario_name in enumerate(DEFAULT_SCENARIOS)}
    return (seed, order.get(scenario, 99), scenario, str(row["run_name"]))


def _policy_family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (POLICY_FAMILIES.index(family), family)
    except ValueError:
        return (len(POLICY_FAMILIES), family)


def select_global_scenario_policies(
    validation_records: list[PortfolioRecord],
    scenarios: list[str],
    *,
    selection_metric: str,
    blend_grid_step: float,
    selection_source: str,
    policy_family: str | None = None,
) -> dict[str, dict[str, Any]]:
    scenario_policies: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        local_validation = [record for record in validation_records if record.scenario == scenario]
        scenario_policies[scenario] = select_global_policy(
            local_validation,
            selection_metric=selection_metric,
            blend_grid_step=blend_grid_step,
            selection_source=selection_source,
            policy_family=policy_family,
        )
    return scenario_policies


def select_policy_family_scenario_policies(
    validation_records: list[PortfolioRecord],
    scenarios: list[str],
    *,
    selection_metric: str,
    blend_grid_step: float,
    policy_families: Iterable[str] = POLICY_FAMILIES,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        family: select_global_scenario_policies(
            validation_records,
            scenarios,
            selection_metric=selection_metric,
            blend_grid_step=blend_grid_step,
            selection_source=f"pooled_validation_global_scenario_policy_family_{family}",
            policy_family=family,
        )
        for family in sorted((str(value) for value in policy_families), key=_policy_family_sort_key)
    }


def evaluate_rows_for_scenario_policies(
    eval_records: list[PortfolioRecord],
    scenario_policies: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        evaluate_policy_on_record(
            record,
            scenario_policies[record.scenario],
            comparison_metric="observed_step_pos_rmse_m",
        )
        for record in eval_records
    ]
    rows.sort(key=_row_sort_key)
    return rows


def evaluate_policy_family_diagnostics(
    eval_records: list[PortfolioRecord],
    policy_family_scenario_policies: dict[str, dict[str, dict[str, Any]]],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for family in sorted(policy_family_scenario_policies, key=_policy_family_sort_key):
        rows = evaluate_rows_for_scenario_policies(eval_records, policy_family_scenario_policies[family])
        diagnostics[family] = {
            "scenario_policy_rows": rows,
            "summary": aggregate_eval_rows(rows),
            "by_scenario": scenario_eval_summary(rows),
            "statistics": eval_gain_statistics(
                rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
        }
    return diagnostics


def build_global_portfolio(
    *,
    run_dirs: list[Path],
    scenarios: list[str],
    device: Any,
    eval_batch_size: int,
    blend_grid_step: float,
    selection_metric: str,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    validation_records: list[PortfolioRecord] = []
    eval_records: list[PortfolioRecord] = []
    run_contexts: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        context = load_run_context(run_dir, device=device)
        run_contexts.append(
            {
                "run_name": context.run_name,
                "run_dir": _display_path(context.run_dir),
                "seed": context.seed,
                "split": context.split,
                "training_seed": context.training_seed,
                "best_checkpoint": str(context.config.get("best_checkpoint")),
                "lookback": context.lookback,
                "lookahead": context.lookahead,
                "candidate_methods": context.candidate_methods,
                "candidate_residual_features": context.candidate_residual_features,
            }
        )
        for scenario in scenarios:
            validation_records.append(
                recompute_validation_record(
                    context,
                    scenario,
                    eval_batch_size=eval_batch_size,
                )
            )
            eval_records.append(load_eval_record(context.run_dir, scenario, context.config))
        del context

    candidate_methods = _candidate_methods_for_records(validation_records)
    full_policy_grid = policy_grid(candidate_methods, blend_grid_step)
    scenario_policies = select_global_scenario_policies(
        validation_records,
        scenarios,
        selection_metric=selection_metric,
        blend_grid_step=blend_grid_step,
        selection_source="pooled_validation_global_scenario",
    )
    policy_family_scenario_policies = select_policy_family_scenario_policies(
        validation_records,
        scenarios,
        selection_metric=selection_metric,
        blend_grid_step=blend_grid_step,
    )

    global_all_scenarios_policy = select_global_policy(
        validation_records,
        selection_metric=selection_metric,
        blend_grid_step=blend_grid_step,
        selection_source="pooled_validation_global_all_scenarios",
    )

    eval_rows = evaluate_rows_for_scenario_policies(eval_records, scenario_policies)
    policy_family_diagnostics = evaluate_policy_family_diagnostics(
        eval_records,
        policy_family_scenario_policies,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )

    all_scenarios_rows = [
        evaluate_policy_on_record(
            record,
            global_all_scenarios_policy,
            comparison_metric="observed_step_pos_rmse_m",
        )
        for record in eval_records
    ]
    all_scenarios_rows.sort(key=_row_sort_key)

    per_split: list[dict[str, Any]] = []
    eval_by_key = {(record.run_name, record.scenario): record for record in eval_records}
    for validation_record in validation_records:
        policy = select_global_policy(
            [validation_record],
            selection_metric=selection_metric,
            blend_grid_step=blend_grid_step,
            selection_source="single_split_scenario_validation_diagnostic",
        )
        eval_record = eval_by_key[(validation_record.run_name, validation_record.scenario)]
        diagnostic = evaluate_policy_on_record(
            eval_record,
            policy,
            comparison_metric="observed_step_pos_rmse_m",
        )
        diagnostic["diagnostic_selection_source"] = policy["selection_source"]
        per_split.append(diagnostic)
    per_split.sort(key=_row_sort_key)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "boundary_language": BOUNDARY_LANGUAGE,
        "inputs": {
            "run_dirs": [_display_path(path) for path in run_dirs],
            "scenarios": list(scenarios),
            "selection_metric": selection_metric,
            "blend_grid_step": float(blend_grid_step),
            "eval_batch_size": int(eval_batch_size),
            "bootstrap_samples": int(bootstrap_samples),
            "bootstrap_seed": int(bootstrap_seed),
            "device": str(device),
        },
        "run_contexts": run_contexts,
        "validation": {
            "policy_space": {
                "components": [
                    LEARNED_COMPONENT,
                    LEARNED_HARD_COMPONENT,
                    *candidate_methods,
                ],
                "policy_count_per_candidate_set": len(full_policy_grid),
                "policy_family_counts_per_candidate_set": {
                    family: len(filter_policy_grid_by_family(full_policy_grid, family))
                    for family in POLICY_FAMILIES
                },
            },
            "global_scenario_policies": scenario_policies,
            "global_all_scenarios_policy": global_all_scenarios_policy,
            "policy_family_scenario_policies": policy_family_scenario_policies,
            "per_split_scenario_policy_diagnostics": per_split,
        },
        "eval": {
            "global_scenario_policy_rows": eval_rows,
            "global_scenario_policy_summary": aggregate_eval_rows(eval_rows),
            "global_scenario_policy_by_scenario": scenario_eval_summary(eval_rows),
            "global_scenario_policy_statistics": eval_gain_statistics(
                eval_rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "global_all_scenarios_policy_rows": all_scenarios_rows,
            "global_all_scenarios_policy_summary": aggregate_eval_rows(all_scenarios_rows),
            "global_all_scenarios_policy_by_scenario": scenario_eval_summary(all_scenarios_rows),
            "global_all_scenarios_policy_statistics": eval_gain_statistics(
                all_scenarios_rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "policy_family_diagnostics": policy_family_diagnostics,
        },
    }
    return summary


CSV_COLUMNS = [
    "seed",
    "split",
    "scenario",
    "run_name",
    "policy_id",
    "policy_kind",
    "policy_components",
    "policy_alpha",
    "policy_validation_rmse_m",
    "policy_validation_steps",
    "portfolio_observed_step_pos_rmse_m",
    "best_input_candidate_method",
    "best_input_candidate_observed_step_pos_rmse_m",
    "gain_vs_best_input_observed_step_percent",
    "result_vs_best_input",
    "portfolio_all_step_pos_rmse_m",
    "run_dir",
]


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in CSV_COLUMNS})


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fmt_gain(value: float) -> str:
    return f"{float(value):+.2f}"


def _fmt_float(value: Any) -> str:
    number = float(value)
    return f"{number:.6f}" if np.isfinite(number) else "nan"


def _fmt_pvalue(value: Any) -> str:
    number = float(value)
    return f"{number:.6g}" if np.isfinite(number) else "nan"


def _fmt_gain_ci(values: Any) -> str:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return "nan"
    return f"[{_fmt_gain(float(values[0]))}, {_fmt_gain(float(values[1]))}]"


def _policy_label(policy: dict[str, Any]) -> str:
    if policy["kind"] == "single":
        return str(policy["components"][0])
    alpha = float(policy["alpha"])
    return f"{alpha:.2f}*{policy['components'][0]} + {1.0 - alpha:.2f}*{policy['components'][1]}"


def _policy_family_markdown_label(family: str) -> str:
    if family == "nonlearned_only":
        return "`nonlearned_only` validation-selected blend baseline"
    return f"`{family}`"


def _scenario_policy_labels(scenario_policies: dict[str, dict[str, Any]]) -> str:
    return "; ".join(
        f"`{scenario}`: `{_policy_label(policy)}`" for scenario, policy in scenario_policies.items()
    )


def render_markdown(summary: dict[str, Any]) -> str:
    scenario_summary = summary["eval"]["global_scenario_policy_by_scenario"]
    overall = summary["eval"]["global_scenario_policy_summary"]
    statistics = summary["eval"].get("global_scenario_policy_statistics")
    policies = summary["validation"]["global_scenario_policies"]
    policy_family_scenario_policies = summary["validation"].get("policy_family_scenario_policies", {})
    policy_family_diagnostics = summary["eval"].get("policy_family_diagnostics", {})
    lines = [
        "# AdaptiveCandidateFusion Global Scenario Portfolio",
        "",
        f"Schema: `{summary['schema_version']}`",
        "",
        "## Boundary",
        "",
        str(summary["boundary_language"]),
        "",
        "## Selected Scenario Policies",
        "",
        "| Scenario | Policy | Validation RMSE m | Validation Steps | Eval Wins/Rows | Mean Gain % | Min Gain % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, policy in policies.items():
        scenario_eval = scenario_summary[scenario]
        lines.append(
            "| "
            f"`{scenario}` | `{_policy_label(policy)}` | "
            f"{_fmt_float(policy['validation_rmse_m'])} | "
            f"{int(policy['evaluated_validation_steps'])} | "
            f"{scenario_eval['wins']}/{scenario_eval['rows']} | "
            f"{_fmt_gain(scenario_eval['mean_gain_percent'])} | "
            f"{_fmt_gain(scenario_eval['min_gain_percent'])} |"
        )
    if isinstance(policy_family_scenario_policies, dict) and isinstance(policy_family_diagnostics, dict):
        lines.extend(
            [
                "",
                "## Policy Family Diagnostics",
                "",
                (
                    "`nonlearned_only` is a validation-selected blend baseline: the same "
                    "pooled validation RMSE selector is applied after excluding `learned` "
                    "and `learned_hard` components."
                ),
                "",
                "| Family | Scenario Policies | Wins/Rows | Mean Gain % | Seed-Paired Wins/Seeds | Seed-Paired Mean Gain % | Seed-Paired Mean Gain 95% CI % |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for family in sorted(policy_family_scenario_policies, key=_policy_family_sort_key):
            if family not in policy_family_diagnostics:
                continue
            family_summary = policy_family_diagnostics[family]["summary"]
            seed_stats = policy_family_diagnostics[family]["statistics"]["seed_paired"]
            lines.append(
                "| "
                f"{_policy_family_markdown_label(family)} | "
                f"{_scenario_policy_labels(policy_family_scenario_policies[family])} | "
                f"{family_summary['wins']}/{family_summary['rows']} | "
                f"{_fmt_gain(family_summary['mean_gain_percent'])} | "
                f"{seed_stats['seed_wins']}/{seed_stats['seeds']} | "
                f"{_fmt_gain(seed_stats['seed_mean_gain_percent'])} | "
                f"{_fmt_gain_ci(seed_stats['bootstrap_seed_mean_gain_percent_ci95'])} |"
            )
    lines.extend(
        [
            "",
            "## Eval Summary",
            "",
            (
                f"Global scenario policies: {overall['wins']}/{overall['rows']} wins vs the "
                "best input candidate per row on observed-step position RMSE; "
                f"mean gain {_fmt_gain(overall['mean_gain_percent'])}%, "
                f"median {_fmt_gain(overall['median_gain_percent'])}%, "
                f"min {_fmt_gain(overall['min_gain_percent'])}%."
            ),
        ]
    )
    if isinstance(statistics, dict):
        row_stats = statistics["rows"]
        seed_stats = statistics["seed_paired"]
        lines.extend(
            [
                "",
                "## Statistical Diagnostics",
                "",
                (
                    "Exact p-values are one-sided sign/binomial tests for positive gain "
                    "versus nonpositive gain; CIs are deterministic percentile bootstrap "
                    "intervals for mean gain."
                ),
                "",
                "| Scope | n | Wins/Ties/Losses | Mean Gain % | Median Gain % | Min/Max Gain % | Sign p | Mean Gain 95% CI % |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                (
                    "| All rows | "
                    f"{row_stats['rows']} | "
                    f"{row_stats['wins']}/{row_stats['ties']}/{row_stats['losses']} | "
                    f"{_fmt_gain(row_stats['mean_gain_percent'])} | "
                    f"{_fmt_gain(row_stats['median_gain_percent'])} | "
                    f"{_fmt_gain(row_stats['min_gain_percent'])}/{_fmt_gain(row_stats['max_gain_percent'])} | "
                    f"{_fmt_pvalue(row_stats['exact_one_sided_sign_p_value'])} | "
                    f"{_fmt_gain_ci(row_stats['bootstrap_mean_gain_percent_ci95'])} |"
                ),
            ]
        )
        for scenario, scenario_stats in statistics["by_scenario"].items():
            lines.append(
                "| "
                f"`{scenario}` | "
                f"{scenario_stats['rows']} | "
                f"{scenario_stats['wins']}/{scenario_stats['ties']}/{scenario_stats['losses']} | "
                f"{_fmt_gain(scenario_stats['mean_gain_percent'])} | "
                f"{_fmt_gain(scenario_stats['median_gain_percent'])} | "
                f"{_fmt_gain(scenario_stats['min_gain_percent'])}/{_fmt_gain(scenario_stats['max_gain_percent'])} | "
                f"{_fmt_pvalue(scenario_stats['exact_one_sided_sign_p_value'])} | "
                f"{_fmt_gain_ci(scenario_stats['bootstrap_mean_gain_percent_ci95'])} |"
            )
        lines.append(
            "| Seed-paired means | "
            f"{seed_stats['seeds']} | "
            f"{seed_stats['seed_wins']}/{seed_stats['seed_ties']}/{seed_stats['seed_losses']} | "
            f"{_fmt_gain(seed_stats['seed_mean_gain_percent'])} | "
            f"{_fmt_gain(seed_stats['seed_median_gain_percent'])} | "
            f"{_fmt_gain(seed_stats['seed_min_gain_percent'])}/{_fmt_gain(seed_stats['seed_max_gain_percent'])} | "
            f"{_fmt_pvalue(seed_stats['exact_one_sided_sign_p_value'])} | "
            f"{_fmt_gain_ci(seed_stats['bootstrap_seed_mean_gain_percent_ci95'])} |"
        )
    lines.extend(
        [
            "",
            "## One Global Policy Diagnostic",
            "",
        ]
    )
    all_policy = summary["validation"]["global_all_scenarios_policy"]
    all_summary = summary["eval"]["global_all_scenarios_policy_summary"]
    lines.extend(
        [
            (
                f"`{_policy_label(all_policy)}` selected across all scenarios: "
                f"{all_summary['wins']}/{all_summary['rows']} wins, "
                f"mean gain {_fmt_gain(all_summary['mean_gain_percent'])}%, "
                f"min {_fmt_gain(all_summary['min_gain_percent'])}%."
            ),
            "",
            "## Rows",
            "",
            "| Seed | Scenario | Policy | Best Input | Portfolio RMSE m | Best Input RMSE m | Gain % | Result |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["eval"]["global_scenario_policy_rows"]:
        lines.append(
            "| "
            f"{row['seed']} | `{row['scenario']}` | `{row['policy_id']}` | "
            f"`{row['best_input_candidate_method']}` | "
            f"{_fmt_float(row['portfolio_observed_step_pos_rmse_m'])} | "
            f"{_fmt_float(row['best_input_candidate_observed_step_pos_rmse_m'])} | "
            f"{_fmt_gain(row['gain_vs_best_input_observed_step_percent'])} | "
            f"{row['result_vs_best_input']} |"
        )
    lines.extend(["", "## Sources", ""])
    for path in summary["inputs"]["run_dirs"]:
        lines.append(f"- `{path}`")
    lines.append("")
    return "\n".join(lines)


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(summary), encoding="utf-8")


def write_artifacts(summary: dict[str, Any], *, output_dir: Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    csv_path = output_dir / SUMMARY_CSV_NAME
    json_path = output_dir / SUMMARY_JSON_NAME
    md_path = output_dir / SUMMARY_MD_NAME
    write_csv(summary["eval"]["global_scenario_policy_rows"], csv_path)
    write_json(summary, json_path)
    write_markdown(summary, md_path)
    return {
        "csv": csv_path.as_posix(),
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
    }


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = resolve_run_dirs(
        positional=args.run_dirs,
        repeated=args.run_dir,
        summary_json=args.summary_json,
    )
    scenarios = parse_csv(args.scenarios)
    if not scenarios:
        raise SystemExit("--scenarios must name at least one scenario")
    if int(args.eval_batch_size) <= 0:
        raise SystemExit("--eval-batch-size must be positive")
    if int(args.bootstrap_samples) <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    try:
        poc.portfolio_blend_grid(float(args.blend_grid_step))
    except ValueError as exc:
        raise SystemExit(f"--blend-grid-step invalid: {exc}") from exc
    device = poc.resolve_device(str(args.device))
    summary = build_global_portfolio(
        run_dirs=run_dirs,
        scenarios=scenarios,
        device=device,
        eval_batch_size=int(args.eval_batch_size),
        blend_grid_step=float(args.blend_grid_step),
        selection_metric=str(args.selection_metric),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    paths = write_artifacts(summary, output_dir=Path(args.output_dir))
    overall = summary["eval"]["global_scenario_policy_summary"]
    print(
        "AdaptiveCandidateFusion global scenario portfolio: "
        f"{overall['wins']}/{overall['rows']} observed-step wins, "
        f"mean gain {overall['mean_gain_percent']:+.2f}%, "
        f"min {overall['min_gain_percent']:+.2f}%."
    )
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['csv']}")
    print(f"Wrote {paths['markdown']}")


if __name__ == "__main__":
    main()
