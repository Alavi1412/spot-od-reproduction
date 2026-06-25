#!/usr/bin/env python
"""Rebuild edge-only residual-refinement comparison intervals from row CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = "trajectory_candidate_residual_refine_comparison_intervals.v2"
BOOTSTRAP_METHOD = (
    "paired percentile resampling of rows or source-scenario clusters with SSE/count recomputation"
)

EDGE_ATTENTION_DIR = Path(
    "results/"
    "trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
EDGE_MEAN_DIR = Path(
    "results/"
    "trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
EDGE_LOCAL_DIR = Path(
    "results/"
    "trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
ORIGINAL_ATTENTION_DIR: Path | None = None

DEFAULT_TIERS = (
    "all_eval_non_development",
    "fresh_extra",
    "holdout_seed_ge_67",
    "future_seed_ge_109",
)
ROW_KEY_FIELDS = ("source_name", "seed", "split", "scenario", "trajectory_row", "trajectory_index")
CLUSTER_KEY_FIELDS = ("source_name", "seed", "split", "scenario")


@dataclass(frozen=True)
class RowTable:
    path: Path
    rows: tuple[dict[str, str], ...]
    by_key: dict[tuple[str, ...], dict[str, str]]


def _read_rows(path: Path) -> RowTable:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))
    by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = _row_key(row)
        if key in by_key:
            raise ValueError(f"duplicate row key {key!r} in {path}")
        by_key[key] = row
    return RowTable(path=path, rows=rows, by_key=by_key)


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in ROW_KEY_FIELDS)


def _cluster_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in CLUSTER_KEY_FIELDS)


def _has_tier(row: dict[str, str], tier: str) -> bool:
    return tier in str(row.get("tier_flags", "")).split(";")


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _rmse(sse: np.ndarray, steps: np.ndarray) -> float:
    total_steps = float(np.sum(steps))
    if total_steps <= 0.0:
        return float("nan")
    return math.sqrt(float(np.sum(sse)) / total_steps)


def _gain_percent(reference_rmse: float, candidate_rmse: float) -> float:
    if reference_rmse <= 0.0 or not math.isfinite(reference_rmse) or not math.isfinite(candidate_rmse):
        return float("nan")
    return 100.0 * (reference_rmse - candidate_rmse) / reference_rmse


def _bootstrap_gain_ci(
    candidate_sse: np.ndarray,
    candidate_steps: np.ndarray,
    reference_sse: np.ndarray,
    reference_steps: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    candidate_rmse = _rmse(candidate_sse, candidate_steps)
    reference_rmse = _rmse(reference_sse, reference_steps)
    point_gain = _gain_percent(reference_rmse, candidate_rmse)
    if samples <= 0:
        return [point_gain, point_gain]
    n = int(candidate_sse.shape[0])
    if n <= 0:
        raise ValueError("cannot bootstrap empty comparison")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(samples, n))
    candidate_rmse_samples = np.sqrt(
        candidate_sse[indices].sum(axis=1) / candidate_steps[indices].sum(axis=1)
    )
    reference_rmse_samples = np.sqrt(
        reference_sse[indices].sum(axis=1) / reference_steps[indices].sum(axis=1)
    )
    gains = 100.0 * (reference_rmse_samples - candidate_rmse_samples) / reference_rmse_samples
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return [float(lo), float(hi)]


def _cluster_arrays(
    rows: Iterable[dict[str, str]],
    candidate_sse: np.ndarray,
    candidate_steps: np.ndarray,
    reference_sse: np.ndarray,
    reference_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    grouped: dict[tuple[str, ...], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(_cluster_key(row), []).append(index)
    cluster_candidate_sse = []
    cluster_candidate_steps = []
    cluster_reference_sse = []
    cluster_reference_steps = []
    for indices in grouped.values():
        cluster_candidate_sse.append(float(candidate_sse[indices].sum()))
        cluster_candidate_steps.append(float(candidate_steps[indices].sum()))
        cluster_reference_sse.append(float(reference_sse[indices].sum()))
        cluster_reference_steps.append(float(reference_steps[indices].sum()))
    return (
        np.asarray(cluster_candidate_sse, dtype=np.float64),
        np.asarray(cluster_candidate_steps, dtype=np.float64),
        np.asarray(cluster_reference_sse, dtype=np.float64),
        np.asarray(cluster_reference_steps, dtype=np.float64),
        len(grouped),
    )


def _paired_reference_rows(
    candidate_rows: list[dict[str, str]],
    reference_table: RowTable | None,
) -> list[dict[str, str]]:
    if reference_table is None:
        return candidate_rows
    paired = []
    for row in candidate_rows:
        key = _row_key(row)
        try:
            paired.append(reference_table.by_key[key])
        except KeyError as exc:
            raise ValueError(f"{reference_table.path} missing paired row key {key!r}") from exc
    return paired


def _comparison(
    candidate_table: RowTable,
    reference_table: RowTable | None,
    tier: str,
    *,
    reference_is_best_single: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    candidate_rows = [row for row in candidate_table.rows if _has_tier(row, tier)]
    if not candidate_rows:
        raise ValueError(f"{candidate_table.path} has no rows for tier {tier!r}")
    reference_rows = _paired_reference_rows(candidate_rows, reference_table)

    candidate_sse = np.asarray(
        [_float(row, "selected_observed_step_sse") for row in candidate_rows], dtype=np.float64
    )
    candidate_steps = np.asarray(
        [_float(row, "selected_observed_steps") for row in candidate_rows], dtype=np.float64
    )
    candidate_row_rmse = np.asarray(
        [_float(row, "selected_observed_step_rmse_m") for row in candidate_rows], dtype=np.float64
    )
    if reference_is_best_single:
        reference_sse = np.asarray(
            [_float(row, "best_single_trajectory_observed_step_sse") for row in candidate_rows],
            dtype=np.float64,
        )
        reference_steps = np.asarray(
            [_float(row, "best_single_trajectory_observed_steps") for row in candidate_rows],
            dtype=np.float64,
        )
        reference_row_rmse = np.asarray(
            [_float(row, "best_single_trajectory_observed_step_rmse_m") for row in candidate_rows],
            dtype=np.float64,
        )
    else:
        reference_sse = np.asarray(
            [_float(row, "selected_observed_step_sse") for row in reference_rows], dtype=np.float64
        )
        reference_steps = np.asarray(
            [_float(row, "selected_observed_steps") for row in reference_rows], dtype=np.float64
        )
        reference_row_rmse = np.asarray(
            [_float(row, "selected_observed_step_rmse_m") for row in reference_rows], dtype=np.float64
        )

    candidate_rmse = _rmse(candidate_sse, candidate_steps)
    reference_rmse = _rmse(reference_sse, reference_steps)
    delta = reference_rmse - candidate_rmse
    wins = int(np.sum(candidate_row_rmse < reference_row_rmse))
    ties = int(np.sum(np.isclose(candidate_row_rmse, reference_row_rmse, rtol=0.0, atol=1.0e-12)))
    losses = int(candidate_row_rmse.shape[0] - wins - ties)
    (
        cluster_candidate_sse,
        cluster_candidate_steps,
        cluster_reference_sse,
        cluster_reference_steps,
        source_scenarios,
    ) = _cluster_arrays(candidate_rows, candidate_sse, candidate_steps, reference_sse, reference_steps)

    return {
        "bootstrap_method": BOOTSTRAP_METHOD,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "candidate_rmse_m": candidate_rmse,
        "delta_m": delta,
        "gain_percent": _gain_percent(reference_rmse, candidate_rmse),
        "observed_steps": int(np.sum(candidate_steps)),
        "reference_rmse_m": reference_rmse,
        "row_bootstrap_gain_percent_95ci": _bootstrap_gain_ci(
            candidate_sse,
            candidate_steps,
            reference_sse,
            reference_steps,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "row_losses": losses,
        "row_ties": ties,
        "row_wins": wins,
        "rows": int(candidate_row_rmse.shape[0]),
        "source_scenario_bootstrap_gain_percent_95ci": _bootstrap_gain_ci(
            cluster_candidate_sse,
            cluster_candidate_steps,
            cluster_reference_sse,
            cluster_reference_steps,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "source_scenarios": source_scenarios,
    }


def build_intervals(
    *,
    attention_dir: Path = EDGE_ATTENTION_DIR,
    local_dir: Path = EDGE_LOCAL_DIR,
    mean_dir: Path = EDGE_MEAN_DIR,
    original_attention_dir: Path | None = ORIGINAL_ATTENTION_DIR,
    tiers: Iterable[str] = DEFAULT_TIERS,
    bootstrap_samples: int = 20000,
    bootstrap_seed: int = 20260625,
) -> dict[str, object]:
    candidate_table = _read_rows(attention_dir / "rows.csv")
    local_table = _read_rows(local_dir / "rows.csv")
    mean_table = _read_rows(mean_dir / "rows.csv")
    references: list[tuple[str, RowTable | None, bool]] = [
        ("best_single_retained", None, True),
        ("edge_only_local_residual_refine", local_table, False),
        ("edge_only_mean_residual_refine", mean_table, False),
    ]
    if original_attention_dir is not None and (original_attention_dir / "rows.csv").exists():
        references.append(
            ("original_attention_residual_refine", _read_rows(original_attention_dir / "rows.csv"), False)
        )

    comparisons: dict[str, dict[str, object]] = {}
    for tier in tiers:
        comparisons[tier] = {}
        for label, reference_table, reference_is_best_single in references:
            comparisons[tier][label] = _comparison(
                candidate_table,
                reference_table,
                tier,
                reference_is_best_single=reference_is_best_single,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )

    return {
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "candidate_label": "edge_only_attention_residual_refine",
        "candidate_run": str(attention_dir),
        "comparisons": comparisons,
        "node_disagreement_features": "omit",
        "schema_version": SCHEMA_VERSION,
    }


def _fmt(value: object, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def render_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Edge-only residual refine comparison intervals",
        "",
        f"Candidate: `{report['candidate_run']}`",
        "",
        (
            "Node-level candidate-disagreement aggregates are omitted; pairwise edge features remain "
            "available to message-passing graph layers."
        ),
        "",
    ]
    comparisons = report["comparisons"]
    assert isinstance(comparisons, dict)
    for tier, tier_comparisons in comparisons.items():
        lines.append(f"## {tier}")
        assert isinstance(tier_comparisons, dict)
        for label, comparison in tier_comparisons.items():
            assert isinstance(comparison, dict)
            row_ci = comparison["row_bootstrap_gain_percent_95ci"]
            cluster_ci = comparison["source_scenario_bootstrap_gain_percent_95ci"]
            assert isinstance(row_ci, list)
            assert isinstance(cluster_ci, list)
            lines.append(
                "- "
                f"{label}: candidate {_fmt(comparison['candidate_rmse_m'])} m vs reference "
                f"{_fmt(comparison['reference_rmse_m'])} m, gain {_fmt(comparison['gain_percent'])}% "
                f"(row CI {_fmt(row_ci[0])} to {_fmt(row_ci[1])}; "
                f"cluster CI {_fmt(cluster_ci[0])} to {_fmt(cluster_ci[1])}), "
                f"W/T/L {comparison['row_wins']}/{comparison['row_ties']}/{comparison['row_losses']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(report: dict[str, object], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def _parse_tiers(raw: str) -> tuple[str, ...]:
    tiers = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not tiers:
        raise argparse.ArgumentTypeError("at least one tier is required")
    return tiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", type=Path, default=EDGE_ATTENTION_DIR)
    parser.add_argument("--local-dir", type=Path, default=EDGE_LOCAL_DIR)
    parser.add_argument("--mean-dir", type=Path, default=EDGE_MEAN_DIR)
    parser.add_argument("--original-attention-dir", type=Path, default=ORIGINAL_ATTENTION_DIR)
    parser.add_argument("--skip-original-attention", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260625)
    parser.add_argument("--tiers", type=_parse_tiers, default=DEFAULT_TIERS)
    parser.add_argument("--output-json", type=Path, default=EDGE_ATTENTION_DIR / "comparison_intervals.json")
    parser.add_argument("--output-md", type=Path, default=EDGE_ATTENTION_DIR / "comparison_intervals.md")
    args = parser.parse_args()

    original_attention_dir = None if args.skip_original_attention else args.original_attention_dir
    report = build_intervals(
        attention_dir=args.attention_dir,
        local_dir=args.local_dir,
        mean_dir=args.mean_dir,
        original_attention_dir=original_attention_dir,
        tiers=args.tiers,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_outputs(report, args.output_json, args.output_md)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
