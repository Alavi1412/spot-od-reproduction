from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from scripts.build_trajectory_residual_refine_comparison_intervals import (
    EDGE_ATTENTION_DIR,
    EDGE_LOCAL_DIR,
    EDGE_MEAN_DIR,
    ORIGINAL_ATTENTION_DIR,
    build_intervals,
    write_outputs,
)


FIELDNAMES = [
    "source_name",
    "seed",
    "split",
    "scenario",
    "trajectory_row",
    "trajectory_index",
    "tier_flags",
    "selected_observed_step_rmse_m",
    "selected_observed_step_sse",
    "selected_observed_steps",
    "best_single_trajectory_observed_step_rmse_m",
    "best_single_trajectory_observed_step_sse",
    "best_single_trajectory_observed_steps",
]

VAL53_ATTENTION_DIR = Path(
    "results/"
    "trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
VAL53_LOCAL_DIR = Path(
    "results/"
    "trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
VAL53_MEAN_DIR = Path(
    "results/"
    "trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)


def test_default_release_paths_target_validation_selected_val53_artifacts() -> None:
    assert EDGE_ATTENTION_DIR == VAL53_ATTENTION_DIR
    assert EDGE_LOCAL_DIR == VAL53_LOCAL_DIR
    assert EDGE_MEAN_DIR == VAL53_MEAN_DIR
    assert EDGE_ATTENTION_DIR / "comparison_intervals.json" == (
        VAL53_ATTENTION_DIR / "comparison_intervals.json"
    )
    assert EDGE_ATTENTION_DIR / "comparison_intervals.md" == (
        VAL53_ATTENTION_DIR / "comparison_intervals.md"
    )
    assert ORIGINAL_ATTENTION_DIR is None


def _row(
    index: int,
    *,
    selected_sse: float,
    best_sse: float = 100.0,
    tier_flags: str = "all_eval_non_development",
) -> dict[str, str]:
    return {
        "source_name": f"source_{index}",
        "seed": str(151 + index),
        "split": str(151 + index),
        "scenario": "process_noise_shift_test",
        "trajectory_row": str(index),
        "trajectory_index": str(index),
        "tier_flags": tier_flags,
        "selected_observed_step_rmse_m": str(math.sqrt(selected_sse)),
        "selected_observed_step_sse": str(selected_sse),
        "selected_observed_steps": "1",
        "best_single_trajectory_observed_step_rmse_m": str(math.sqrt(best_sse)),
        "best_single_trajectory_observed_step_sse": str(best_sse),
        "best_single_trajectory_observed_steps": "1",
    }


def _write_rows(directory: Path, rows: list[dict[str, str]]) -> None:
    directory.mkdir(parents=True)
    with (directory / "rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_build_intervals_pairs_rows_and_recomputes_aggregate_gains(tmp_path: Path) -> None:
    attention_dir = tmp_path / "attention"
    local_dir = tmp_path / "local"
    mean_dir = tmp_path / "mean"
    tiers = "fresh_extra;all_eval_non_development"
    _write_rows(
        attention_dir,
        [
            _row(0, selected_sse=64.0, tier_flags=tiers),
            _row(1, selected_sse=36.0),
        ],
    )
    _write_rows(
        local_dir,
        [
            _row(0, selected_sse=81.0, tier_flags=tiers),
            _row(1, selected_sse=144.0),
        ],
    )
    _write_rows(
        mean_dir,
        [
            _row(0, selected_sse=49.0, tier_flags=tiers),
            _row(1, selected_sse=49.0),
        ],
    )

    report = build_intervals(
        attention_dir=attention_dir,
        local_dir=local_dir,
        mean_dir=mean_dir,
        original_attention_dir=None,
        tiers=("all_eval_non_development",),
        bootstrap_samples=0,
    )

    all_eval = report["comparisons"]["all_eval_non_development"]
    best = all_eval["best_single_retained"]
    assert best["rows"] == 2
    assert best["observed_steps"] == 2
    assert best["candidate_rmse_m"] == pytest.approx(math.sqrt(50.0))
    assert best["reference_rmse_m"] == pytest.approx(10.0)
    assert best["gain_percent"] == pytest.approx(29.289321881345245)
    assert best["row_wins"] == 2
    assert best["row_ties"] == 0
    assert best["row_losses"] == 0
    assert best["row_bootstrap_gain_percent_95ci"] == pytest.approx(
        [best["gain_percent"], best["gain_percent"]]
    )

    local = all_eval["edge_only_local_residual_refine"]
    assert local["reference_rmse_m"] == pytest.approx(math.sqrt(112.5))
    assert local["gain_percent"] == pytest.approx(100.0 / 3.0)
    assert local["row_wins"] == 2

    mean = all_eval["edge_only_mean_residual_refine"]
    assert mean["reference_rmse_m"] == pytest.approx(7.0)
    assert mean["gain_percent"] < 0.0
    assert mean["row_wins"] == 1
    assert mean["row_losses"] == 1

    output_json = tmp_path / "comparison_intervals.json"
    output_md = tmp_path / "comparison_intervals.md"
    write_outputs(report, output_json, output_md)
    parsed = json.loads(output_json.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == report["schema_version"]
    markdown = output_md.read_text(encoding="utf-8")
    assert "edge_only_local_residual_refine" in markdown
    assert "W/T/L 2/0/0" in markdown
