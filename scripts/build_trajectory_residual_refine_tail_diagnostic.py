#!/usr/bin/env python
"""Build saved-row tail diagnostics for the edge-only local residual-refine control."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = "trajectory_candidate_edge_only_local_tail_diagnostic.v1"
BOUNDARY = (
    "Saved-row compact-simulator diagnostic only; not independent reproduction, "
    "not public precise-reference validation, not operational POD, not a full "
    "raw/training/all-filter rerun, and not standalone learned recursive filtering."
)

ATTENTION_ROWS = Path(
    "results/"
    "trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625/rows.csv"
)
LOCAL_ROWS = Path(
    "results/"
    "trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625/rows.csv"
)
MEAN_ROWS = Path(
    "results/"
    "trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625/rows.csv"
)
OUTPUT_DIR = Path("results/trajectory_candidate_edge_only_local_tail_diagnostic_val53_20260625")

DEFAULT_TIERS = ("all_eval_non_development", "fresh_extra")
ROW_KEY_FIELDS = ("source_name", "scenario", "trajectory_row")
QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

ROWS_FIELDNAMES = (
    "tier",
    "source_name",
    "scenario",
    "trajectory_row",
    "trajectory_index",
    "seed",
    "split",
    "source_is_extra",
    "tier_flags",
    "observed_steps",
    "attention_selected_observed_step_rmse_m",
    "local_selected_observed_step_rmse_m",
    "mean_selected_observed_step_rmse_m",
    "best_single_trajectory_observed_step_rmse_m",
    "local_minus_attention_rmse_m",
    "local_minus_mean_rmse_m",
    "attention_selected_candidate_method",
    "attention_selected_probability",
    "local_selected_candidate_method",
    "local_selected_probability",
    "mean_selected_candidate_method",
    "mean_selected_probability",
    "best_single_candidate_method",
)


@dataclass(frozen=True)
class RowTable:
    label: str
    path: Path
    rows: tuple[dict[str, str], ...]
    by_key: dict[tuple[str, ...], dict[str, str]]


def _read_rows(label: str, path: Path) -> RowTable:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = _row_key(row)
        if key in by_key:
            raise ValueError(f"{path} contains duplicate alignment key {key!r}")
        by_key[key] = row
    return RowTable(label=label, path=path, rows=rows, by_key=by_key)


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in ROW_KEY_FIELDS)


def _has_tier(row: dict[str, str], tier: str) -> bool:
    return tier in str(row.get("tier_flags", "")).split(";")


def _tier_keys(table: RowTable, tier: str) -> set[tuple[str, ...]]:
    return {_row_key(row) for row in table.rows if _has_tier(row, tier)}


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _optional_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field)
    if raw in (None, ""):
        return None
    return float(raw)


def _integer(value: float) -> int:
    rounded = int(round(value))
    if abs(value - rounded) > 1.0e-9:
        raise ValueError(f"expected integer-like value, got {value!r}")
    return rounded


def _rmse_from_sse_steps(sse_values: Sequence[float], step_values: Sequence[float]) -> float:
    steps = float(sum(step_values))
    if steps <= 0.0:
        return float("nan")
    return math.sqrt(float(sum(sse_values)) / steps)


def _gain_percent(reference_rmse: float, candidate_rmse: float) -> float:
    if reference_rmse <= 0.0 or not math.isfinite(reference_rmse):
        return float("nan")
    return 100.0 * (reference_rmse - candidate_rmse) / reference_rmse


def _win_tie_loss(candidate: Sequence[float], reference: Sequence[float]) -> dict[str, int]:
    wins = sum(1 for left, right in zip(candidate, reference) if left < right)
    ties = sum(1 for left, right in zip(candidate, reference) if math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12))
    losses = len(candidate) - wins - ties
    return {"wins": wins, "ties": ties, "losses": losses}


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {f"p{int(q * 100):02d}": float("nan") for q in QUANTILES}
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return {f"p{int(q * 100):02d}": ordered[0] for q in QUANTILES}

    result: dict[str, float] = {}
    max_index = len(ordered) - 1
    for quantile in QUANTILES:
        position = quantile * max_index
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            value = ordered[lower]
        else:
            weight = position - lower
            value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        result[f"p{int(quantile * 100):02d}"] = float(value)
    return result


def _count_methods(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        method = row.get("selected_candidate_method", "")
        if method:
            counts[method] = counts.get(method, 0) + 1
    return dict(sorted(counts.items()))


def _check_aligned_tier(
    tier: str,
    attention: RowTable,
    local: RowTable,
    mean: RowTable,
) -> tuple[tuple[dict[str, str], dict[str, str], dict[str, str]], ...]:
    key_sets = {
        attention.label: _tier_keys(attention, tier),
        local.label: _tier_keys(local, tier),
        mean.label: _tier_keys(mean, tier),
    }
    expected = key_sets[attention.label]
    for label, keys in key_sets.items():
        if keys != expected:
            missing = sorted(expected - keys)[:5]
            extra = sorted(keys - expected)[:5]
            raise ValueError(
                f"tier {tier!r} alignment mismatch for {label}: missing={missing!r}, extra={extra!r}"
            )
    if not expected:
        raise ValueError(f"no rows matched tier {tier!r}")
    return tuple(
        (attention.by_key[key], local.by_key[key], mean.by_key[key])
        for key in sorted(expected)
    )


def _check_same_steps(
    key: tuple[str, ...],
    attention_row: dict[str, str],
    local_row: dict[str, str],
    mean_row: dict[str, str],
) -> None:
    selected_steps = [
        _float(attention_row, "selected_observed_steps"),
        _float(local_row, "selected_observed_steps"),
        _float(mean_row, "selected_observed_steps"),
    ]
    best_steps = [
        _float(attention_row, "best_single_trajectory_observed_steps"),
        _float(local_row, "best_single_trajectory_observed_steps"),
        _float(mean_row, "best_single_trajectory_observed_steps"),
    ]
    if len({round(value, 9) for value in selected_steps + best_steps}) != 1:
        raise ValueError(f"observed-step mismatch for alignment key {key!r}")


def _selected_method(row: dict[str, str]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field in ("selected_candidate_method", "selected_candidate_index"):
        value = row.get(field)
        if value not in (None, ""):
            payload[field] = value
    probability = _optional_float(row, "selected_probability")
    if probability is not None:
        payload["selected_probability"] = probability
    return payload


def _best_single_method(row: dict[str, str]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for field in ("best_single_candidate_method", "best_single_candidate_index"):
        value = row.get(field)
        if value not in (None, ""):
            payload[field] = value
    return payload


def _aligned_output_row(
    tier: str,
    attention_row: dict[str, str],
    local_row: dict[str, str],
    mean_row: dict[str, str],
) -> dict[str, object]:
    key = _row_key(attention_row)
    _check_same_steps(key, attention_row, local_row, mean_row)
    attention_rmse = _float(attention_row, "selected_observed_step_rmse_m")
    local_rmse = _float(local_row, "selected_observed_step_rmse_m")
    mean_rmse = _float(mean_row, "selected_observed_step_rmse_m")
    return {
        "tier": tier,
        "source_name": attention_row.get("source_name", ""),
        "scenario": attention_row.get("scenario", ""),
        "trajectory_row": attention_row.get("trajectory_row", ""),
        "trajectory_index": attention_row.get("trajectory_index", ""),
        "seed": attention_row.get("seed", ""),
        "split": attention_row.get("split", ""),
        "source_is_extra": attention_row.get("source_is_extra", ""),
        "tier_flags": attention_row.get("tier_flags", ""),
        "observed_steps": _integer(_float(attention_row, "selected_observed_steps")),
        "attention_selected_observed_step_rmse_m": attention_rmse,
        "local_selected_observed_step_rmse_m": local_rmse,
        "mean_selected_observed_step_rmse_m": mean_rmse,
        "best_single_trajectory_observed_step_rmse_m": _float(
            attention_row, "best_single_trajectory_observed_step_rmse_m"
        ),
        "local_minus_attention_rmse_m": local_rmse - attention_rmse,
        "local_minus_mean_rmse_m": local_rmse - mean_rmse,
        "attention_selected_candidate_method": attention_row.get("selected_candidate_method", ""),
        "attention_selected_probability": attention_row.get("selected_probability", ""),
        "local_selected_candidate_method": local_row.get("selected_candidate_method", ""),
        "local_selected_probability": local_row.get("selected_probability", ""),
        "mean_selected_candidate_method": mean_row.get("selected_candidate_method", ""),
        "mean_selected_probability": mean_row.get("selected_probability", ""),
        "best_single_candidate_method": attention_row.get("best_single_candidate_method", ""),
    }


def _tail_entry(
    tier: str,
    attention_row: dict[str, str],
    local_row: dict[str, str],
    mean_row: dict[str, str],
) -> dict[str, object]:
    output_row = _aligned_output_row(tier, attention_row, local_row, mean_row)
    entry: dict[str, object] = {
        field: output_row[field]
        for field in (
            "source_name",
            "scenario",
            "trajectory_row",
            "trajectory_index",
            "seed",
            "split",
            "observed_steps",
            "attention_selected_observed_step_rmse_m",
            "local_selected_observed_step_rmse_m",
            "mean_selected_observed_step_rmse_m",
            "best_single_trajectory_observed_step_rmse_m",
            "local_minus_attention_rmse_m",
            "local_minus_mean_rmse_m",
        )
    }
    entry["attention"] = _selected_method(attention_row)
    entry["local"] = _selected_method(local_row)
    entry["mean"] = _selected_method(mean_row)
    entry["best_single"] = _best_single_method(attention_row)
    return entry


def _sort_key_for_tail(entry: dict[str, object]) -> tuple[object, ...]:
    return (
        str(entry["source_name"]),
        str(entry["scenario"]),
        str(entry["trajectory_row"]),
    )


def _tier_summary(
    tier: str,
    paired_rows: Sequence[tuple[dict[str, str], dict[str, str], dict[str, str]]],
    *,
    top_n: int,
) -> dict[str, object]:
    attention_selected_sse = [_float(attention, "selected_observed_step_sse") for attention, _, _ in paired_rows]
    local_selected_sse = [_float(local, "selected_observed_step_sse") for _, local, _ in paired_rows]
    mean_selected_sse = [_float(mean, "selected_observed_step_sse") for _, _, mean in paired_rows]
    selected_steps = [_float(attention, "selected_observed_steps") for attention, _, _ in paired_rows]
    best_single_sse = [
        _float(attention, "best_single_trajectory_observed_step_sse")
        for attention, _, _ in paired_rows
    ]
    best_single_steps = [
        _float(attention, "best_single_trajectory_observed_steps")
        for attention, _, _ in paired_rows
    ]

    attention_rmse_by_row = [
        _float(attention, "selected_observed_step_rmse_m") for attention, _, _ in paired_rows
    ]
    local_rmse_by_row = [_float(local, "selected_observed_step_rmse_m") for _, local, _ in paired_rows]
    mean_rmse_by_row = [_float(mean, "selected_observed_step_rmse_m") for _, _, mean in paired_rows]
    local_minus_attention = [
        local - attention for local, attention in zip(local_rmse_by_row, attention_rmse_by_row)
    ]
    local_minus_mean = [local - mean for local, mean in zip(local_rmse_by_row, mean_rmse_by_row)]

    tail_entries = [_tail_entry(tier, attention, local, mean) for attention, local, mean in paired_rows]
    by_local_rmse = sorted(
        tail_entries,
        key=lambda entry: (
            -float(entry["local_selected_observed_step_rmse_m"]),
            _sort_key_for_tail(entry),
        ),
    )[:top_n]
    by_local_minus_attention = sorted(
        tail_entries,
        key=lambda entry: (-float(entry["local_minus_attention_rmse_m"]), _sort_key_for_tail(entry)),
    )[:top_n]

    attention_rmse = _rmse_from_sse_steps(attention_selected_sse, selected_steps)
    local_rmse = _rmse_from_sse_steps(local_selected_sse, selected_steps)
    mean_rmse = _rmse_from_sse_steps(mean_selected_sse, selected_steps)
    best_rmse = _rmse_from_sse_steps(best_single_sse, best_single_steps)

    return {
        "rows": len(paired_rows),
        "observed_steps": _integer(sum(selected_steps)),
        "pooled_rmse_m": {
            "edge_only_attention": attention_rmse,
            "edge_only_local_no_message": local_rmse,
            "edge_only_mean_graph": mean_rmse,
            "best_single_retained": best_rmse,
        },
        "gain_vs_best_single_percent": {
            "edge_only_attention": _gain_percent(best_rmse, attention_rmse),
            "edge_only_local_no_message": _gain_percent(best_rmse, local_rmse),
            "edge_only_mean_graph": _gain_percent(best_rmse, mean_rmse),
        },
        "local_advantage_percent": {
            "attention_over_local": _gain_percent(local_rmse, attention_rmse),
            "mean_over_local": _gain_percent(local_rmse, mean_rmse),
        },
        "row_wtl_local_vs_attention": _win_tie_loss(local_rmse_by_row, attention_rmse_by_row),
        "row_wtl_local_vs_mean": _win_tie_loss(local_rmse_by_row, mean_rmse_by_row),
        "local_minus_attention_row_rmse_delta_quantiles_m": _quantiles(local_minus_attention),
        "local_minus_mean_row_rmse_delta_quantiles_m": _quantiles(local_minus_mean),
        "selected_method_counts": {
            "attention": _count_methods(attention for attention, _, _ in paired_rows),
            "local": _count_methods(local for _, local, _ in paired_rows),
            "mean": _count_methods(mean for _, _, mean in paired_rows),
        },
        "top_local_tail_rows_by_local_rmse": by_local_rmse,
        "top_local_tail_rows_by_local_minus_attention_delta": by_local_minus_attention,
    }


def build_diagnostic(
    *,
    attention_rows: Path = ATTENTION_ROWS,
    local_rows: Path = LOCAL_ROWS,
    mean_rows: Path = MEAN_ROWS,
    tiers: Iterable[str] = DEFAULT_TIERS,
    top_n: int = 10,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    attention = _read_rows("attention", attention_rows)
    local = _read_rows("local", local_rows)
    mean = _read_rows("mean", mean_rows)

    tier_summaries: dict[str, object] = {}
    output_rows: list[dict[str, object]] = []
    for tier in tiers:
        paired = _check_aligned_tier(tier, attention, local, mean)
        tier_summaries[tier] = _tier_summary(tier, paired, top_n=top_n)
        for attention_row, local_row, mean_row in paired:
            output_rows.append(_aligned_output_row(tier, attention_row, local_row, mean_row))

    report = {
        "schema_version": SCHEMA_VERSION,
        "boundary": BOUNDARY,
        "diagnostic_scope": "edge-only local/no-message residual-refinement tail failures from saved rows",
        "generated_from_saved_rows_only": True,
        "alignment_key_fields": list(ROW_KEY_FIELDS),
        "source_rows": {
            "attention": str(attention_rows),
            "local": str(local_rows),
            "mean": str(mean_rows),
        },
        "tiers": tier_summaries,
    }
    return report, output_rows


def _format_float(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _format_wtl(wtl: dict[str, int]) -> str:
    return f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']}"


def _format_tail_row(entry: dict[str, object]) -> str:
    local = entry.get("local", {})
    attention = entry.get("attention", {})
    mean = entry.get("mean", {})
    assert isinstance(local, dict)
    assert isinstance(attention, dict)
    assert isinstance(mean, dict)
    local_method = str(local.get("selected_candidate_method", ""))
    attention_method = str(attention.get("selected_candidate_method", ""))
    mean_method = str(mean.get("selected_candidate_method", ""))
    local_prob = local.get("selected_probability")
    attention_prob = attention.get("selected_probability")
    mean_prob = mean.get("selected_probability")
    return (
        f"{entry['source_name']} / {entry['scenario']} / row {entry['trajectory_row']}: "
        f"local {_format_float(entry['local_selected_observed_step_rmse_m'])} m "
        f"({local_method}, p={_format_float(local_prob) if local_prob is not None else 'NA'}), "
        f"attention {_format_float(entry['attention_selected_observed_step_rmse_m'])} m "
        f"({attention_method}, p={_format_float(attention_prob) if attention_prob is not None else 'NA'}), "
        f"mean {_format_float(entry['mean_selected_observed_step_rmse_m'])} m "
        f"({mean_method}, p={_format_float(mean_prob) if mean_prob is not None else 'NA'}), "
        f"local-attention delta {_format_float(entry['local_minus_attention_rmse_m'])} m"
    )


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Edge-only local tail diagnostic",
        "",
        f"Boundary: {report['boundary']}",
        "",
        "This diagnostic aligns saved `rows.csv` records by `(source_name, scenario, trajectory_row)`.",
        "",
        "## Sources",
    ]
    source_rows = report["source_rows"]
    assert isinstance(source_rows, dict)
    for label, path in source_rows.items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(["", "## Aggregate diagnostics", ""])
    lines.append(
        "| Tier | Rows | Obs. steps | Attention RMSE m | Local RMSE m | Mean RMSE m | Best-single RMSE m | Local vs attention W/T/L | Local vs mean W/T/L | Local-attn p50/p95/max delta m |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    tiers = report["tiers"]
    assert isinstance(tiers, dict)
    for tier, tier_summary in tiers.items():
        assert isinstance(tier_summary, dict)
        pooled = tier_summary["pooled_rmse_m"]
        quantiles = tier_summary["local_minus_attention_row_rmse_delta_quantiles_m"]
        assert isinstance(pooled, dict)
        assert isinstance(quantiles, dict)
        lines.append(
            "| "
            f"{tier} | {tier_summary['rows']} | {tier_summary['observed_steps']} | "
            f"{_format_float(pooled['edge_only_attention'])} | "
            f"{_format_float(pooled['edge_only_local_no_message'])} | "
            f"{_format_float(pooled['edge_only_mean_graph'])} | "
            f"{_format_float(pooled['best_single_retained'])} | "
            f"{_format_wtl(tier_summary['row_wtl_local_vs_attention'])} | "
            f"{_format_wtl(tier_summary['row_wtl_local_vs_mean'])} | "
            f"{_format_float(quantiles['p50'])}/"
            f"{_format_float(quantiles['p95'])}/"
            f"{_format_float(quantiles['p100'])} |"
        )

    for tier, tier_summary in tiers.items():
        assert isinstance(tier_summary, dict)
        lines.extend(["", f"## Tail rows: {tier}", ""])
        lines.append("Top rows by local selected observed-step RMSE:")
        for entry in tier_summary["top_local_tail_rows_by_local_rmse"][:5]:
            lines.append(f"- {_format_tail_row(entry)}")
        lines.append("")
        lines.append("Top rows by local-minus-attention row RMSE delta:")
        for entry in tier_summary["top_local_tail_rows_by_local_minus_attention_delta"][:5]:
            lines.append(f"- {_format_tail_row(entry)}")

    lines.extend(
        [
            "",
            "The weak local aggregate is driven by saved-row tail failures; the attention-vs-mean comparison remains weak/mixed.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    report: dict[str, object],
    output_rows: Sequence[dict[str, object]],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    rows_csv = output_dir / "rows.csv"

    summary_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(render_markdown(report), encoding="utf-8")
    with rows_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROWS_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in output_rows:
            writer.writerow({field: row.get(field, "") for field in ROWS_FIELDNAMES})
    return {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "rows_csv": str(rows_csv),
    }


def _parse_tiers(raw: str) -> tuple[str, ...]:
    tiers = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not tiers:
        raise argparse.ArgumentTypeError("at least one tier is required")
    return tiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-rows", type=Path, default=ATTENTION_ROWS)
    parser.add_argument("--local-rows", type=Path, default=LOCAL_ROWS)
    parser.add_argument("--mean-rows", type=Path, default=MEAN_ROWS)
    parser.add_argument("--tiers", type=_parse_tiers, default=DEFAULT_TIERS)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    report, output_rows = build_diagnostic(
        attention_rows=args.attention_rows,
        local_rows=args.local_rows,
        mean_rows=args.mean_rows,
        tiers=args.tiers,
        top_n=args.top_n,
    )
    outputs = write_outputs(report, output_rows, args.output_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
