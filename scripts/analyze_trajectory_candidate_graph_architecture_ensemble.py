#!/usr/bin/env python
"""Analyze a graph-only architecture confidence ensemble over selector summaries."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.run_trajectory_candidate_graph_selector_poc import (
        BOUNDARY_STATEMENT,
        TIER_NAMES,
        aggregate_tier_rows,
        relative_gain_percent,
        rmse_from_sse_count,
        sanitize_for_json,
        write_rows_csv,
        write_strict_json,
    )
except ModuleNotFoundError:  # pragma: no cover - used when invoked as python scripts/name.py
    from run_trajectory_candidate_graph_selector_poc import (
        BOUNDARY_STATEMENT,
        TIER_NAMES,
        aggregate_tier_rows,
        relative_gain_percent,
        rmse_from_sse_count,
        sanitize_for_json,
        write_rows_csv,
        write_strict_json,
    )


SCHEMA_VERSION = "trajectory_candidate_graph_architecture_ensemble.v1"
DEFAULT_NAME = "graph_architecture_confidence_ensemble"
SELECTION_RULE = (
    "For each aligned row, select the graph summary row with the highest selected_probability. "
    "Evaluation truth/error fields are not used for selection."
)
ROW_KEY_FIELDS = ("source_name", "scenario", "trajectory_row")
REQUIRED_ROW_FIELDS = (
    *ROW_KEY_FIELDS,
    "selected_candidate_method",
    "selected_candidate_index",
    "selected_probability",
    "selected_observed_step_rmse_m",
    "selected_observed_step_sse",
    "selected_observed_steps",
    "best_single_candidate_method",
    "best_single_candidate_index",
    "best_single_run_scenario_observed_step_rmse_m",
    "best_single_trajectory_observed_step_rmse_m",
    "best_single_trajectory_observed_step_sse",
    "best_single_trajectory_observed_steps",
    "gain_vs_best_single_trajectory_percent",
    "tier_flags",
)
ROW_COMPATIBILITY_FIELDS = (
    "source_dir",
    "seed",
    "split",
    "source_is_extra",
    "trajectory_index",
    "tier_flags",
    "candidate_methods",
    "baseline_candidate_methods",
    "selected_observed_steps",
    "best_single_candidate_method",
    "best_single_candidate_index",
    "best_single_run_scenario_observed_step_rmse_m",
    "best_single_trajectory_observed_step_rmse_m",
    "best_single_trajectory_observed_step_sse",
    "best_single_trajectory_observed_steps",
)
SUMMARY_COMPATIBILITY_FIELDS = (
    "candidate_methods",
    "baseline_candidate_methods",
    "development_seed_max_exclusive",
    "holdout_seed_min",
    "future_seed_min",
    "scenarios",
)

RowKey = tuple[str, str, int]


@dataclass
class LoadedSummary:
    path: Path
    label: str
    payload: dict[str, Any]
    rows: list[dict[str, Any]]
    rows_by_key: dict[RowKey, dict[str, Any]]
    candidate_methods: list[str]
    baseline_candidate_methods: list[str]
    member_index: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a truth-free graph architecture confidence ensemble by selecting, per row, "
            "the graph summary with the highest selected_probability."
        )
    )
    parser.add_argument("--graph-summary", action="append", default=[], help="Path to a graph selector summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary.json, rows.csv, and summary.md.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Label for the output architecture ensemble.")
    parser.add_argument(
        "--reference-summary",
        default=None,
        help="Optional local/control selector summary.json used only for post-selection comparison metrics.",
    )
    parser.add_argument(
        "--allow-local-members",
        action="store_true",
        help=(
            "Allow local/no-message selector summaries as ensemble members for control artifacts. "
            "By default, --graph-summary inputs must be graph/message-passing summaries."
        ),
    )
    return parser


def row_key(row: dict[str, Any]) -> RowKey:
    missing = [field for field in ROW_KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"row is missing key field(s): {', '.join(missing)}")
    try:
        trajectory_row = int(row["trajectory_row"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trajectory_row is not an integer-compatible value: {row.get('trajectory_row')!r}") from exc
    return (str(row["source_name"]), str(row["scenario"]), trajectory_row)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"summary path does not exist or is not a file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary JSON must be an object: {path}")
    return payload


def _summary_label(path: Path, payload: dict[str, Any]) -> str:
    output_dir = payload.get("output_dir")
    if output_dir:
        name = Path(str(output_dir)).name
        if name:
            return name
    if path.name.lower() == "summary.json" and path.parent.name:
        return path.parent.name
    return path.stem


def _list_from_payload_or_first_row(payload: dict[str, Any], rows: list[dict[str, Any]], field: str, path: Path) -> list[str]:
    value = payload.get(field)
    if value is None and rows:
        value = rows[0].get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} is missing non-empty {field}.")
    return [str(item) for item in value]


def _resolve_baseline_candidate_methods(
    *,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    candidate_methods: list[str],
) -> list[str]:
    value = payload.get("baseline_candidate_methods")
    if value is None and rows:
        value = rows[0].get("baseline_candidate_methods")
    if value is None:
        return list(candidate_methods)
    if not isinstance(value, list) or not value:
        raise ValueError("baseline_candidate_methods must be a non-empty list when present.")
    return [str(item) for item in value]


def _validate_selector_summary_kind(payload: dict[str, Any], path: Path, *, graph_required: bool) -> None:
    prediction_mode = payload.get("prediction_mode")
    if prediction_mode is not None and prediction_mode != "selector":
        kind = "--graph-summary" if graph_required else "--reference-summary"
        raise ValueError(f"{kind} must be a selector summary; {path} has prediction_mode={prediction_mode!r}.")
    if not graph_required:
        return
    if payload.get("message_passing_enabled") is False:
        raise ValueError(f"--graph-summary appears to be local/non-graph because message_passing_enabled is false: {path}")
    if payload.get("graph_layers") is not None:
        try:
            graph_layers = int(payload["graph_layers"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"--graph-summary has non-integer graph_layers: {path}") from exc
        if graph_layers <= 0:
            raise ValueError(f"--graph-summary appears to be local/non-graph because graph_layers <= 0: {path}")


def _finite_float(value: Any, *, field: str, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has non-numeric {field}: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} has non-finite {field}: {value!r}")
    return numeric


def _validate_rows(
    *,
    rows: list[Any],
    candidate_methods: list[str],
    baseline_candidate_methods: list[str],
    label: str,
) -> dict[RowKey, dict[str, Any]]:
    if not rows:
        raise ValueError(f"{label} has no rows.")
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
        row_candidate_methods = row.get("candidate_methods")
        if row_candidate_methods is None:
            row["candidate_methods"] = list(candidate_methods)
            row_candidate_methods = row["candidate_methods"]
        if not isinstance(row_candidate_methods, list) or [str(item) for item in row_candidate_methods] != candidate_methods:
            raise ValueError(f"{label} row {key!r} has incompatible candidate methods.")
        row_baseline_candidate_methods = row.get("baseline_candidate_methods")
        if row_baseline_candidate_methods is None:
            row["baseline_candidate_methods"] = list(baseline_candidate_methods)
            row_baseline_candidate_methods = row["baseline_candidate_methods"]
        if (
            not isinstance(row_baseline_candidate_methods, list)
            or [str(item) for item in row_baseline_candidate_methods] != baseline_candidate_methods
        ):
            raise ValueError(f"{label} row {key!r} has incompatible baseline candidate methods.")
        _finite_float(row["selected_probability"], field="selected_probability", label=f"{label} row {key!r}")
        _finite_float(
            row["selected_observed_step_rmse_m"],
            field="selected_observed_step_rmse_m",
            label=f"{label} row {key!r}",
        )
        _finite_float(
            row["selected_observed_step_sse"],
            field="selected_observed_step_sse",
            label=f"{label} row {key!r}",
        )
        rows_by_key[key] = row
    return rows_by_key


def load_summary(path: str | Path, *, graph_required: bool, member_index: int | None = None) -> LoadedSummary:
    summary_path = Path(path)
    payload = _load_json_object(summary_path)
    _validate_selector_summary_kind(payload, summary_path, graph_required=graph_required)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{summary_path} is missing rows list.")
    candidate_methods = _list_from_payload_or_first_row(payload, rows, "candidate_methods", summary_path)
    baseline_candidate_methods = _resolve_baseline_candidate_methods(
        payload=payload,
        rows=rows,
        candidate_methods=candidate_methods,
    )
    label = _summary_label(summary_path, payload)
    rows_by_key = _validate_rows(
        rows=rows,
        candidate_methods=candidate_methods,
        baseline_candidate_methods=baseline_candidate_methods,
        label=label,
    )
    return LoadedSummary(
        path=summary_path,
        label=label,
        payload=payload,
        rows=rows,
        rows_by_key=rows_by_key,
        candidate_methods=candidate_methods,
        baseline_candidate_methods=baseline_candidate_methods,
        member_index=member_index,
    )


def _values_compatible(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) or isinstance(right, (int, float)):
        try:
            left_float = float(left)
            right_float = float(right)
        except (TypeError, ValueError):
            return left == right
        if math.isfinite(left_float) and math.isfinite(right_float):
            return math.isclose(left_float, right_float, rel_tol=1.0e-9, abs_tol=1.0e-9)
        return left_float == right_float
    return left == right


def _compare_row_field(
    *,
    reference: LoadedSummary,
    other: LoadedSummary,
    key: RowKey,
    field: str,
) -> None:
    left = reference.rows_by_key[key]
    right = other.rows_by_key[key]
    if field not in left or field not in right:
        raise ValueError(f"incompatible summaries: row {key!r} is missing compatibility field {field!r}.")
    if not _values_compatible(left[field], right[field]):
        raise ValueError(
            "incompatible summaries: "
            f"row {key!r} field {field!r} differs between {reference.label} and {other.label} "
            f"({left[field]!r} != {right[field]!r})."
        )


def _validate_summary_level_compatibility(reference: LoadedSummary, other: LoadedSummary) -> None:
    if reference.candidate_methods != other.candidate_methods:
        raise ValueError(
            "incompatible candidate methods: "
            f"{reference.label} has {reference.candidate_methods}, {other.label} has {other.candidate_methods}."
        )
    if reference.baseline_candidate_methods != other.baseline_candidate_methods:
        raise ValueError(
            "incompatible baseline candidate methods: "
            f"{reference.label} has {reference.baseline_candidate_methods}, "
            f"{other.label} has {other.baseline_candidate_methods}."
        )
    for field in SUMMARY_COMPATIBILITY_FIELDS:
        if field not in reference.payload or field not in other.payload:
            continue
        if not _values_compatible(reference.payload[field], other.payload[field]):
            raise ValueError(
                "incompatible summaries: "
                f"summary field {field!r} differs between {reference.label} and {other.label}."
            )


def validate_compatible_summaries(reference: LoadedSummary, others: list[LoadedSummary]) -> None:
    reference_keys = set(reference.rows_by_key)
    for other in others:
        _validate_summary_level_compatibility(reference, other)
        other_keys = set(other.rows_by_key)
        if reference_keys != other_keys:
            missing = sorted(reference_keys - other_keys)[:5]
            extra = sorted(other_keys - reference_keys)[:5]
            raise ValueError(
                "incompatible row keys: "
                f"{other.label} is missing {len(reference_keys - other_keys)} row(s) and has "
                f"{len(other_keys - reference_keys)} extra row(s); missing examples={missing}, extra examples={extra}."
            )
        for key in reference_keys:
            for field in ROW_COMPATIBILITY_FIELDS:
                _compare_row_field(reference=reference, other=other, key=key, field=field)


def _selected_member_for_key(graph_summaries: list[LoadedSummary], key: RowKey) -> tuple[LoadedSummary, dict[str, Any], float]:
    best_member = graph_summaries[0]
    best_row = best_member.rows_by_key[key]
    best_probability = _finite_float(
        best_row["selected_probability"],
        field="selected_probability",
        label=f"{best_member.label} row {key!r}",
    )
    for member in graph_summaries[1:]:
        row = member.rows_by_key[key]
        probability = _finite_float(
            row["selected_probability"],
            field="selected_probability",
            label=f"{member.label} row {key!r}",
        )
        if probability > best_probability:
            best_member = member
            best_row = row
            best_probability = probability
    return best_member, best_row, best_probability


def _is_local_no_message_member(member: LoadedSummary) -> bool:
    if member.payload.get("message_passing_enabled") is False:
        return True
    graph_layers = member.payload.get("graph_layers")
    if graph_layers is None:
        return False
    try:
        return int(graph_layers) <= 0
    except (TypeError, ValueError):
        return False


def _add_reference_fields(row: dict[str, Any], reference: LoadedSummary, key: RowKey) -> None:
    reference_row = reference.rows_by_key[key]
    reference_rmse = _finite_float(
        reference_row["selected_observed_step_rmse_m"],
        field="selected_observed_step_rmse_m",
        label=f"{reference.label} row {key!r}",
    )
    selector_rmse = _finite_float(
        row["selected_observed_step_rmse_m"],
        field="selected_observed_step_rmse_m",
        label=f"architecture ensemble row {key!r}",
    )
    row.update(
        {
            "reference_summary": reference.label,
            "reference_summary_path": str(reference.path),
            "reference_selected_candidate_method": reference_row["selected_candidate_method"],
            "reference_selected_candidate_index": reference_row["selected_candidate_index"],
            "reference_selected_probability": reference_row["selected_probability"],
            "reference_selected_observed_step_rmse_m": reference_rmse,
            "reference_selected_observed_step_sse": reference_row["selected_observed_step_sse"],
            "reference_selected_observed_steps": reference_row["selected_observed_steps"],
            "gain_vs_reference_selected_trajectory_percent": relative_gain_percent(reference_rmse, selector_rmse),
        }
    )


def build_output_rows(
    *,
    graph_summaries: list[LoadedSummary],
    name: str,
    reference: LoadedSummary | None,
) -> list[dict[str, Any]]:
    base_keys = [row_key(row) for row in graph_summaries[0].rows]
    rows: list[dict[str, Any]] = []
    for key in base_keys:
        member, selected_row, selected_probability = _selected_member_for_key(graph_summaries, key)
        row = dict(selected_row)
        row["gain_vs_best_single_trajectory_percent"] = relative_gain_percent(
            _finite_float(
                row["best_single_trajectory_observed_step_rmse_m"],
                field="best_single_trajectory_observed_step_rmse_m",
                label=f"architecture ensemble row {key!r}",
            ),
            _finite_float(
                row["selected_observed_step_rmse_m"],
                field="selected_observed_step_rmse_m",
                label=f"architecture ensemble row {key!r}",
            ),
        )
        row.update(
            {
                "architecture_ensemble_name": name,
                "architecture_ensemble_member": member.label,
                "architecture_ensemble_member_index": member.member_index,
                "architecture_ensemble_member_order": None
                if member.member_index is None
                else int(member.member_index) + 1,
                "architecture_ensemble_member_summary": str(member.path),
                "architecture_ensemble_selected_probability": selected_probability,
                "architecture_ensemble_selection_rule": "max_selected_probability",
                "architecture_ensemble_graph_summary_count": len(graph_summaries),
            }
        )
        if reference is not None:
            _add_reference_fields(row, reference, key)
        rows.append(row)
    return rows


def _mean_or_none(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def add_reference_aggregate_metrics(rows: list[dict[str, Any]], aggregates: dict[str, dict[str, Any]]) -> None:
    for tier in TIER_NAMES:
        tier_rows = [row for row in rows if tier in str(row.get("tier_flags", "")).split(";")]
        reference_sse = sum(float(row["reference_selected_observed_step_sse"]) for row in tier_rows)
        reference_count = sum(int(row["reference_selected_observed_steps"]) for row in tier_rows)
        reference_rmse = rmse_from_sse_count(reference_sse, reference_count)
        selector_rmse = float(aggregates[tier]["selector_observed_step_rmse_m"])
        row_gains = [
            float(row["gain_vs_reference_selected_trajectory_percent"])
            for row in tier_rows
            if row["gain_vs_reference_selected_trajectory_percent"] is not None
            and math.isfinite(float(row["gain_vs_reference_selected_trajectory_percent"]))
        ]
        wins = ties = losses = 0
        for row in tier_rows:
            selected = float(row["selected_observed_step_rmse_m"])
            reference = float(row["reference_selected_observed_step_rmse_m"])
            if not (math.isfinite(selected) and math.isfinite(reference)):
                continue
            if selected < reference - 1.0e-9:
                wins += 1
            elif selected > reference + 1.0e-9:
                losses += 1
            else:
                ties += 1
        aggregates[tier].update(
            {
                "reference_observed_steps": reference_count,
                "reference_selector_observed_step_rmse_m": reference_rmse,
                "gain_vs_reference_selector_percent": relative_gain_percent(reference_rmse, selector_rmse),
                "row_wins_vs_reference": wins,
                "row_ties_vs_reference": ties,
                "row_losses_vs_reference": losses,
                "mean_row_gain_vs_reference_percent": _mean_or_none(row_gains),
                "median_row_gain_vs_reference_percent": _median_or_none(row_gains),
                "reference_selected_method_counts": dict(
                    Counter(str(row["reference_selected_candidate_method"]) for row in tier_rows)
                ),
            }
        )


def add_member_aggregate_metrics(rows: list[dict[str, Any]], aggregates: dict[str, dict[str, Any]]) -> None:
    for tier in TIER_NAMES:
        tier_rows = [row for row in rows if tier in str(row.get("tier_flags", "")).split(";")]
        aggregates[tier]["architecture_ensemble_member_counts"] = dict(
            Counter(str(row["architecture_ensemble_member"]) for row in tier_rows)
        )


def build_aggregate_tiers(rows: list[dict[str, Any]], *, include_reference: bool) -> dict[str, dict[str, Any]]:
    aggregates = aggregate_tier_rows(rows)
    add_member_aggregate_metrics(rows, aggregates)
    if include_reference:
        add_reference_aggregate_metrics(rows, aggregates)
    return aggregates


def _member_metadata(member: LoadedSummary) -> dict[str, Any]:
    return {
        "member": member.label,
        "member_index": member.member_index,
        "summary_path": str(member.path),
        "message_passing_enabled": member.payload.get("message_passing_enabled"),
        "is_local_no_message_member": _is_local_no_message_member(member),
        "graph_layers": member.payload.get("graph_layers"),
        "graph_layer_type": member.payload.get("graph_layer_type"),
        "hidden_dim": member.payload.get("hidden_dim"),
        "dropout": member.payload.get("dropout"),
        "learning_rate": member.payload.get("learning_rate"),
        "ensemble_size": member.payload.get("ensemble_size"),
        "ensemble_member_seeds": member.payload.get("ensemble_member_seeds"),
    }


def _common_payload_fields(summary: LoadedSummary) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in SUMMARY_COMPATIBILITY_FIELDS:
        if field in summary.payload:
            fields[field] = summary.payload[field]
    return fields


def build_summary(
    *,
    graph_summaries: list[LoadedSummary],
    rows: list[dict[str, Any]],
    name: str,
    reference: LoadedSummary | None,
    duration_s: float,
    allow_local_members: bool,
) -> dict[str, Any]:
    aggregate_tiers = build_aggregate_tiers(rows, include_reference=reference is not None)
    local_member_count = sum(1 for member in graph_summaries if _is_local_no_message_member(member))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "boundary_statement": BOUNDARY_STATEMENT,
        "selection_rule": (
            "For each aligned row, select the member summary row with the highest selected_probability. "
            "Evaluation truth/error fields are not used for selection."
            if allow_local_members
            else SELECTION_RULE
        ),
        "selection_probability_field": "selected_probability",
        "selection_uses_evaluation_truth": False,
        "truth_fields_used_for_selection": [],
        "allow_local_members": bool(allow_local_members),
        "graph_members_required": not allow_local_members,
        "local_no_message_member_count": local_member_count,
        "member_acceptance_policy": "local_members_allowed" if allow_local_members else "graph_members_required",
        "row_key_fields": list(ROW_KEY_FIELDS),
        "candidate_methods": graph_summaries[0].candidate_methods,
        "baseline_candidate_methods": graph_summaries[0].baseline_candidate_methods,
        "graph_summary_count": len(graph_summaries),
        "graph_summary_paths": [str(member.path) for member in graph_summaries],
        "graph_members": [_member_metadata(member) for member in graph_summaries],
        "reference_summary_path": None if reference is None else str(reference.path),
        "reference_summary": None if reference is None else reference.label,
        "row_count": len(rows),
        "aggregate_tiers": aggregate_tiers,
        "rows": rows,
        "duration_s": duration_s,
    }
    payload.update(_common_payload_fields(graph_summaries[0]))
    return payload


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
    has_reference = summary.get("reference_summary_path") is not None
    allow_local_members = bool(summary.get("allow_local_members", False))
    selection_subject = "member" if allow_local_members else "graph architecture member"
    selection_source = "member summaries" if allow_local_members else "graph model outputs"
    header = [
        "Tier",
        "Rows",
        "Observed steps",
        "Architecture ensemble RMSE m",
        "Best single RMSE m",
        "Gain vs best single %",
    ]
    alignment = ["---", "---:", "---:", "---:", "---:", "---:"]
    if has_reference:
        header.extend(["Reference RMSE m", "Gain vs reference %"])
        alignment.extend(["---:", "---:"])

    lines = [
        "# Trajectory Candidate Graph Architecture Confidence Ensemble",
        "",
        f"Name: {summary.get('name', DEFAULT_NAME)}",
        "",
        f"Boundary: {summary.get('boundary_statement', BOUNDARY_STATEMENT)}",
        "",
        f"Selection rule: choose the {selection_subject} with the largest `selected_probability` "
        "for each aligned row.",
        "",
        f"This selection uses only selected probabilities from {selection_source}. It does not use "
        "evaluation truth, candidate RMSE, best-single RMSE, labels, or reference/local-control outputs "
        "to choose a member. Truth/error fields are used only after selection for scoring.",
        "",
        f"Graph summaries: {summary.get('graph_summary_count', 0)}",
        f"Local/no-message members allowed: {'yes' if allow_local_members else 'no'}",
        f"Graph members required: {'yes' if summary.get('graph_members_required', True) else 'no'}",
        f"Local/no-message member count: {summary.get('local_no_message_member_count', 0)}",
        f"Reference summary: {summary.get('reference_summary') or 'none'}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    for tier in TIER_NAMES:
        item = aggregates[tier]
        row = [
            tier,
            str(item["rows"]),
            str(item["observed_steps"]),
            _format_metric(item["selector_observed_step_rmse_m"]),
            _format_metric(item["best_single_observed_step_rmse_m"]),
            _format_metric(item["gain_vs_best_single_percent"]),
        ]
        if has_reference:
            row.extend(
                [
                    _format_metric(item.get("reference_selector_observed_step_rmse_m")),
                    _format_metric(item.get("gain_vs_reference_selector_percent")),
                ]
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "The best-single denominator is inherited from the aligned selector summaries. "
            "Reference metrics, when present, are comparison-only and are not inputs to architecture selection.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_architecture_ensemble(
    *,
    graph_summary_paths: list[str | Path],
    output_dir: str | Path,
    name: str = DEFAULT_NAME,
    reference_summary_path: str | Path | None = None,
    allow_local_members: bool = False,
) -> dict[str, Any]:
    if not graph_summary_paths:
        raise ValueError("at least one --graph-summary is required.")
    started = time.perf_counter()
    graph_summaries = [
        load_summary(path, graph_required=not allow_local_members, member_index=index)
        for index, path in enumerate(graph_summary_paths)
    ]
    validate_compatible_summaries(graph_summaries[0], graph_summaries[1:])
    reference = None
    if reference_summary_path is not None:
        reference = load_summary(reference_summary_path, graph_required=False)
        validate_compatible_summaries(graph_summaries[0], [reference])

    output_rows = build_output_rows(
        graph_summaries=graph_summaries,
        name=name,
        reference=reference,
    )
    summary = build_summary(
        graph_summaries=graph_summaries,
        rows=output_rows,
        name=name,
        reference=reference,
        duration_s=float(time.perf_counter() - started),
        allow_local_members=allow_local_members,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_rows_csv(output_rows, output_path / "rows.csv")
    write_strict_json(summary, output_path / "summary.json")
    write_summary_md(sanitize_for_json(summary), output_path / "summary.md")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = analyze_architecture_ensemble(
            graph_summary_paths=list(args.graph_summary),
            output_dir=args.output_dir,
            name=str(args.name),
            reference_summary_path=args.reference_summary,
            allow_local_members=bool(args.allow_local_members),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote graph architecture confidence ensemble outputs under {args.output_dir}")
    print(f"Rows: {summary['row_count']}")


if __name__ == "__main__":
    main()
