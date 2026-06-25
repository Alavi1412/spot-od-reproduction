#!/usr/bin/env python
"""Compare two already-selected trajectory architecture summary artifacts."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.run_trajectory_candidate_graph_selector_poc import (
        TIER_NAMES,
        sanitize_for_json,
        write_rows_csv,
        write_strict_json,
    )
except ModuleNotFoundError:  # pragma: no cover - used when invoked as python scripts/name.py
    from run_trajectory_candidate_graph_selector_poc import (
        TIER_NAMES,
        sanitize_for_json,
        write_rows_csv,
        write_strict_json,
    )


SCHEMA_VERSION = "trajectory_candidate_architecture_summary_comparison.v1"
DEFAULT_NAME = "architecture_summary_comparison"
ROW_KEY_FIELDS = ("source_name", "scenario", "trajectory_row")
REQUIRED_ROW_FIELDS = (
    *ROW_KEY_FIELDS,
    "selected_observed_step_rmse_m",
    "selected_observed_step_sse",
    "selected_observed_steps",
    "tier_flags",
)
RMSE_TOLERANCE = 1.0e-9
CI_PERCENTILES = (2.5, 97.5)
BOUNDARY_STATEMENT = (
    "Saved-row compact-simulator comparison only; not independent-machine reproduction, "
    "not operational precise-reference validation, and not full raw/training/all-filter rerun."
)

RowKey = tuple[str, str, int]


@dataclass(frozen=True)
class LoadedSummary:
    path: Path
    label: str
    summary_name: str | None
    payload: dict[str, Any]
    rows: list[dict[str, Any]]
    rows_by_key: dict[RowKey, dict[str, Any]]


@dataclass(frozen=True)
class SelectedMetrics:
    rmse_m: float | None
    sse: float
    observed_steps: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two saved selector/architecture summary.json files on aligned rows. "
            "The summaries must already contain selected rows; this script only scores the comparison."
        )
    )
    parser.add_argument("--left-summary", required=True, help="Path to the left summary.json.")
    parser.add_argument("--right-summary", required=True, help="Path to the right summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary.json, rows.csv, and summary.md.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Label for this comparison artifact.")
    parser.add_argument("--bootstrap-samples", type=int, default=5000, help="Bootstrap resamples per tier.")
    parser.add_argument("--bootstrap-seed", type=int, default=20260625, help="Deterministic bootstrap seed.")
    return parser


def row_key(row: dict[str, Any]) -> RowKey:
    missing = [field for field in ROW_KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"row is missing key field(s): {', '.join(missing)}")
    try:
        trajectory_row = int(row["trajectory_row"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trajectory_row is not integer-compatible: {row.get('trajectory_row')!r}") from exc
    return (str(row["source_name"]), str(row["scenario"]), trajectory_row)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"summary path does not exist or is not a file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary JSON must be an object: {path}")
    return payload


def _summary_label(path: Path, payload: dict[str, Any]) -> str:
    name = payload.get("name")
    if name:
        return str(name)
    output_dir = payload.get("output_dir")
    if output_dir:
        output_name = Path(str(output_dir)).name
        if output_name:
            return output_name
    if path.name.lower() == "summary.json" and path.parent.name:
        return path.parent.name
    return path.stem


def _validate_rows(rows: list[Any], *, label: str) -> dict[RowKey, dict[str, Any]]:
    rows_by_key: dict[RowKey, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} is not an object.")
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{label} row {index} is missing required field(s): {', '.join(missing)}")
        key = row_key(row)
        if key in rows_by_key:
            raise ValueError(f"{label} contains duplicate row key: {key!r}")
        _selected_metrics(row, label=f"{label} row {key!r}")
        rows_by_key[key] = row
    return rows_by_key


def load_summary(path: str | Path) -> LoadedSummary:
    summary_path = Path(path)
    payload = _load_json_object(summary_path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{summary_path} is missing rows list.")
    label = _summary_label(summary_path, payload)
    rows_by_key = _validate_rows(rows, label=label)
    summary_name = payload.get("name")
    return LoadedSummary(
        path=summary_path,
        label=label,
        summary_name=str(summary_name) if summary_name is not None else None,
        payload=payload,
        rows=rows,
        rows_by_key=rows_by_key,
    )


def _finite_float(value: Any, *, field: str, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has non-numeric {field}: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} has non-finite {field}: {value!r}")
    return numeric


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _observed_steps(value: Any, *, field: str, label: str) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has non-integer {field}: {value!r}") from exc
    if count < 0:
        raise ValueError(f"{label} has negative {field}: {value!r}")
    return count


def _rmse_from_sse_count_or_none(sse: float, count: int) -> float | None:
    if count <= 0:
        return None
    if not math.isfinite(sse):
        return None
    return float(math.sqrt(max(float(sse), 0.0) / float(count)))


def _selected_metrics(row: dict[str, Any], *, label: str) -> SelectedMetrics:
    sse = _finite_float(row["selected_observed_step_sse"], field="selected_observed_step_sse", label=label)
    if sse < 0.0:
        raise ValueError(f"{label} has negative selected_observed_step_sse: {sse!r}")
    observed_steps = _observed_steps(row["selected_observed_steps"], field="selected_observed_steps", label=label)
    rmse = _optional_finite_float(row.get("selected_observed_step_rmse_m"))
    if rmse is None:
        rmse = _rmse_from_sse_count_or_none(sse, observed_steps)
    return SelectedMetrics(rmse_m=rmse, sse=sse, observed_steps=observed_steps)


def validate_row_keys_match(left: LoadedSummary, right: LoadedSummary) -> None:
    left_keys = set(left.rows_by_key)
    right_keys = set(right.rows_by_key)
    if left_keys == right_keys:
        return
    missing_from_right = sorted(left_keys - right_keys)[:5]
    missing_from_left = sorted(right_keys - left_keys)[:5]
    raise ValueError(
        "row-key mismatch: "
        f"right summary is missing {len(left_keys - right_keys)} row(s), "
        f"left summary is missing {len(right_keys - left_keys)} row(s); "
        f"missing_from_right examples={missing_from_right}, "
        f"missing_from_left examples={missing_from_left}."
    )


def _gain_percent(reference_rmse: float | None, candidate_rmse: float | None) -> float | None:
    if reference_rmse is None or candidate_rmse is None:
        return None
    if not (math.isfinite(reference_rmse) and math.isfinite(candidate_rmse)):
        return None
    if reference_rmse <= 0.0:
        return None
    return 100.0 * (reference_rmse - candidate_rmse) / reference_rmse


def _row_result(left_rmse: float | None, right_rmse: float | None) -> tuple[str, str]:
    if left_rmse is None or right_rmse is None:
        return "unscored", "unscored"
    if not (math.isfinite(left_rmse) and math.isfinite(right_rmse)):
        return "unscored", "unscored"
    if left_rmse < right_rmse - RMSE_TOLERANCE:
        return "win", "left"
    if left_rmse > right_rmse + RMSE_TOLERANCE:
        return "loss", "right"
    return "tie", "tie"


def build_output_rows(*, left: LoadedSummary, right: LoadedSummary) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_row in left.rows:
        key = row_key(left_row)
        right_row = right.rows_by_key[key]
        left_metrics = _selected_metrics(left_row, label=f"{left.label} row {key!r}")
        right_metrics = _selected_metrics(right_row, label=f"{right.label} row {key!r}")
        result, winner = _row_result(left_metrics.rmse_m, right_metrics.rmse_m)
        rows.append(
            {
                "source_name": key[0],
                "scenario": key[1],
                "trajectory_row": key[2],
                "tier_flags": left_row.get("tier_flags"),
                "left_summary": left.label,
                "left_summary_path": str(left.path),
                "right_summary": right.label,
                "right_summary_path": str(right.path),
                "left_selected_candidate_method": left_row.get("selected_candidate_method"),
                "left_selected_candidate_index": left_row.get("selected_candidate_index"),
                "right_selected_candidate_method": right_row.get("selected_candidate_method"),
                "right_selected_candidate_index": right_row.get("selected_candidate_index"),
                "left_selected_observed_step_rmse_m": left_metrics.rmse_m,
                "left_selected_observed_step_sse": left_metrics.sse,
                "left_selected_observed_steps": left_metrics.observed_steps,
                "right_selected_observed_step_rmse_m": right_metrics.rmse_m,
                "right_selected_observed_step_sse": right_metrics.sse,
                "right_selected_observed_steps": right_metrics.observed_steps,
                "left_gain_vs_right_row_percent": _gain_percent(right_metrics.rmse_m, left_metrics.rmse_m),
                "left_result_vs_right": result,
                "row_winner": winner,
            }
        )
    return rows


def _row_tiers(row: dict[str, Any]) -> set[str]:
    return {item for item in str(row.get("tier_flags", "")).split(";") if item}


def _finite_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (float(percentile) / 100.0) * float(len(ordered) - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return float(ordered[lower_index])
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    weight = rank - float(lower_index)
    return float(lower + (upper - lower) * weight)


def _bootstrap_gain_ci(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not rows or bootstrap_samples <= 0:
        return {
            "bootstrap_left_gain_vs_right_pooled_percent_ci95": [None, None],
            "bootstrap_left_gain_vs_right_pooled_percent_ci95_low": None,
            "bootstrap_left_gain_vs_right_pooled_percent_ci95_high": None,
            "left_gain_vs_right_pooled_percent_bootstrap_ci95": [None, None],
            "bootstrap_finite_samples": 0,
        }
    rng = random.Random(int(bootstrap_seed))
    sample_count = int(bootstrap_samples)
    gains: list[float] = []
    row_count = len(rows)
    for _ in range(sample_count):
        left_sse = 0.0
        right_sse = 0.0
        left_count = 0
        right_count = 0
        for _ in range(row_count):
            row = rows[rng.randrange(row_count)]
            left_sse += float(row["left_selected_observed_step_sse"])
            right_sse += float(row["right_selected_observed_step_sse"])
            left_count += int(row["left_selected_observed_steps"])
            right_count += int(row["right_selected_observed_steps"])
        gain = _gain_percent(
            _rmse_from_sse_count_or_none(right_sse, right_count),
            _rmse_from_sse_count_or_none(left_sse, left_count),
        )
        if gain is not None and math.isfinite(gain):
            gains.append(float(gain))
    ci_low = _percentile(gains, CI_PERCENTILES[0])
    ci_high = _percentile(gains, CI_PERCENTILES[1])
    return {
        "bootstrap_left_gain_vs_right_pooled_percent_ci95": [ci_low, ci_high],
        "bootstrap_left_gain_vs_right_pooled_percent_ci95_low": ci_low,
        "bootstrap_left_gain_vs_right_pooled_percent_ci95_high": ci_high,
        "left_gain_vs_right_pooled_percent_bootstrap_ci95": [ci_low, ci_high],
        "bootstrap_finite_samples": len(gains),
    }


def build_aggregate_tiers(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for tier_index, tier in enumerate(TIER_NAMES):
        tier_rows = [row for row in rows if tier in _row_tiers(row)]
        left_sse = sum(float(row["left_selected_observed_step_sse"]) for row in tier_rows)
        right_sse = sum(float(row["right_selected_observed_step_sse"]) for row in tier_rows)
        left_count = sum(int(row["left_selected_observed_steps"]) for row in tier_rows)
        right_count = sum(int(row["right_selected_observed_steps"]) for row in tier_rows)
        left_rmse = _rmse_from_sse_count_or_none(left_sse, left_count)
        right_rmse = _rmse_from_sse_count_or_none(right_sse, right_count)
        row_gains = _finite_values([row.get("left_gain_vs_right_row_percent") for row in tier_rows])
        aggregate: dict[str, Any] = {
            "rows": len(tier_rows),
            "observed_steps": left_count,
            "left_observed_steps": left_count,
            "right_observed_steps": right_count,
            "left_pooled_rmse_m": left_rmse,
            "right_pooled_rmse_m": right_rmse,
            "left_gain_vs_right_pooled_percent": _gain_percent(right_rmse, left_rmse),
            "row_wins": sum(1 for row in tier_rows if row["left_result_vs_right"] == "win"),
            "row_ties": sum(1 for row in tier_rows if row["left_result_vs_right"] == "tie"),
            "row_losses": sum(1 for row in tier_rows if row["left_result_vs_right"] == "loss"),
            "mean_row_gain_percent": float(statistics.fmean(row_gains)) if row_gains else None,
            "median_row_gain_percent": float(statistics.median(row_gains)) if row_gains else None,
        }
        aggregate.update(
            _bootstrap_gain_ci(
                tier_rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=int(bootstrap_seed) + tier_index * 1_000_003,
            )
        )
        aggregates[tier] = aggregate
    return aggregates


def _source_metadata(summary: LoadedSummary) -> dict[str, Any]:
    return {
        "summary_path": str(summary.path),
        "summary_name": summary.summary_name,
        "summary_label": summary.label,
        "row_count": len(summary.rows),
    }


def build_summary(
    *,
    left: LoadedSummary,
    right: LoadedSummary,
    rows: list[dict[str, Any]],
    name: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "boundary_statement": BOUNDARY_STATEMENT,
        "comparison_scope": "post_selection_saved_row_scoring_only",
        "selection_uses_evaluation_truth": False,
        "truth_fields_used_for_selection": [],
        "row_key_fields": list(ROW_KEY_FIELDS),
        "left_summary_path": str(left.path),
        "right_summary_path": str(right.path),
        "left_summary_name": left.summary_name,
        "right_summary_name": right.summary_name,
        "left_summary_label": left.label,
        "right_summary_label": right.label,
        "left_row_count": len(left.rows),
        "right_row_count": len(right.rows),
        "row_count": len(rows),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_ci_percentiles": list(CI_PERCENTILES),
        "sources": {
            "left": _source_metadata(left),
            "right": _source_metadata(right),
        },
        "aggregate_tiers": build_aggregate_tiers(
            rows,
            bootstrap_samples=int(bootstrap_samples),
            bootstrap_seed=int(bootstrap_seed),
        ),
        "rows": rows,
        "duration_s": float(duration_s),
    }


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
        "# Architecture Summary Comparison",
        "",
        f"Name: {summary.get('name', DEFAULT_NAME)}",
        "",
        f"Boundary: {summary.get('boundary_statement', BOUNDARY_STATEMENT)}",
        "",
        "Selection already happened upstream. This script does not select using truth; it only compares "
        "saved selected-output fields on aligned rows.",
        "",
        "This artifact is not independent-machine reproduction, not operational precise-reference "
        "validation, and not a full raw/training/all-filter rerun.",
        "",
        f"Left summary: {summary.get('left_summary_label')} ({summary.get('left_summary_path')})",
        f"Right summary: {summary.get('right_summary_label')} ({summary.get('right_summary_path')})",
        f"Row key fields: {', '.join(summary.get('row_key_fields', ROW_KEY_FIELDS))}",
        f"Bootstrap samples: {summary.get('bootstrap_samples')} seed: {summary.get('bootstrap_seed')}",
        "",
        "| Tier | Rows | Observed steps | Left pooled RMSE m | Right pooled RMSE m | Left gain % | 95% CI | Wins/Ties/Losses | Mean row gain % | Median row gain % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tier in TIER_NAMES:
        item = aggregates[tier]
        ci = (
            f"[{_format_metric(item.get('bootstrap_left_gain_vs_right_pooled_percent_ci95_low'))}, "
            f"{_format_metric(item.get('bootstrap_left_gain_vs_right_pooled_percent_ci95_high'))}]"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    tier,
                    str(item["rows"]),
                    str(item["observed_steps"]),
                    _format_metric(item["left_pooled_rmse_m"]),
                    _format_metric(item["right_pooled_rmse_m"]),
                    _format_metric(item["left_gain_vs_right_pooled_percent"]),
                    ci,
                    f"{item['row_wins']}/{item['row_ties']}/{item['row_losses']}",
                    _format_metric(item["mean_row_gain_percent"]),
                    _format_metric(item["median_row_gain_percent"]),
                ]
            )
            + " |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_architecture_summaries(
    *,
    left_summary_path: str | Path,
    right_summary_path: str | Path,
    output_dir: str | Path,
    name: str = DEFAULT_NAME,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260625,
) -> dict[str, Any]:
    if int(bootstrap_samples) < 0:
        raise ValueError("--bootstrap-samples must be non-negative.")
    started = time.perf_counter()
    left = load_summary(left_summary_path)
    right = load_summary(right_summary_path)
    validate_row_keys_match(left, right)
    rows = build_output_rows(left=left, right=right)
    summary = build_summary(
        left=left,
        right=right,
        rows=rows,
        name=name,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_seed=int(bootstrap_seed),
        duration_s=time.perf_counter() - started,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, output_path / "rows.csv")
    write_strict_json(summary, output_path / "summary.json")
    write_summary_md(sanitize_for_json(summary), output_path / "summary.md")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = compare_architecture_summaries(
            left_summary_path=args.left_summary,
            right_summary_path=args.right_summary,
            output_dir=args.output_dir,
            name=str(args.name),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote architecture summary comparison outputs under {args.output_dir}")
    print(f"Rows: {summary['row_count']}")


if __name__ == "__main__":
    main()
