from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from scripts.build_trajectory_residual_refine_tail_diagnostic import (
    ATTENTION_ROWS,
    BOUNDARY,
    LOCAL_ROWS,
    MEAN_ROWS,
    OUTPUT_DIR,
    build_diagnostic,
    write_outputs,
)


FIELDNAMES = [
    "source_name",
    "seed",
    "split",
    "source_is_extra",
    "scenario",
    "trajectory_row",
    "trajectory_index",
    "tier_flags",
    "selected_candidate_method",
    "selected_candidate_index",
    "selected_probability",
    "selected_observed_step_rmse_m",
    "selected_observed_step_sse",
    "selected_observed_steps",
    "best_single_candidate_method",
    "best_single_candidate_index",
    "best_single_trajectory_observed_step_rmse_m",
    "best_single_trajectory_observed_step_sse",
    "best_single_trajectory_observed_steps",
]

VAL53_ATTENTION_ROWS = Path(
    "results/"
    "trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625/rows.csv"
)
VAL53_LOCAL_ROWS = Path(
    "results/"
    "trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625/rows.csv"
)
VAL53_MEAN_ROWS = Path(
    "results/"
    "trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625/rows.csv"
)


def test_default_release_paths_target_validation_selected_val53_artifacts() -> None:
    assert ATTENTION_ROWS == VAL53_ATTENTION_ROWS
    assert LOCAL_ROWS == VAL53_LOCAL_ROWS
    assert MEAN_ROWS == VAL53_MEAN_ROWS
    assert OUTPUT_DIR == Path("results/trajectory_candidate_edge_only_local_tail_diagnostic_val53_20260625")


def _row(
    index: int,
    *,
    selected_sse: float,
    selected_method: str,
    selected_probability: float,
    tier_flags: str = "all_eval_non_development",
    best_sse: float = 100.0,
) -> dict[str, str]:
    return {
        "source_name": f"source_{index}",
        "seed": str(151 + index),
        "split": str(151 + index),
        "source_is_extra": "True" if index < 2 else "False",
        "scenario": "process_noise_shift_test",
        "trajectory_row": str(index),
        "trajectory_index": str(20 + index),
        "tier_flags": tier_flags,
        "selected_candidate_method": selected_method,
        "selected_candidate_index": "3",
        "selected_probability": str(selected_probability),
        "selected_observed_step_rmse_m": str(math.sqrt(selected_sse)),
        "selected_observed_step_sse": str(selected_sse),
        "selected_observed_steps": "1",
        "best_single_candidate_method": "RFIS",
        "best_single_candidate_index": "4",
        "best_single_trajectory_observed_step_rmse_m": str(math.sqrt(best_sse)),
        "best_single_trajectory_observed_step_sse": str(best_sse),
        "best_single_trajectory_observed_steps": "1",
    }


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_build_diagnostic_reports_local_tail_failure_from_saved_rows(tmp_path: Path) -> None:
    attention_rows = tmp_path / "attention" / "rows.csv"
    local_rows = tmp_path / "local" / "rows.csv"
    mean_rows = tmp_path / "mean" / "rows.csv"
    fresh_and_all = "fresh_extra;all_eval_non_development"

    _write_rows(
        attention_rows,
        [
            _row(0, selected_sse=100.0, selected_method="BatchWLS", selected_probability=0.70, tier_flags=fresh_and_all),
            _row(1, selected_sse=400.0, selected_method="RFIS", selected_probability=0.80, tier_flags=fresh_and_all),
            _row(2, selected_sse=25.0, selected_method="EKF", selected_probability=0.90),
        ],
    )
    _write_rows(
        local_rows,
        [
            _row(0, selected_sse=81.0, selected_method="BatchWLS", selected_probability=0.40, tier_flags=fresh_and_all),
            _row(1, selected_sse=2500.0, selected_method="EKF", selected_probability=0.51, tier_flags=fresh_and_all),
            _row(2, selected_sse=16.0, selected_method="RFIS", selected_probability=0.60),
        ],
    )
    _write_rows(
        mean_rows,
        [
            _row(0, selected_sse=64.0, selected_method="BatchWLS", selected_probability=0.95, tier_flags=fresh_and_all),
            _row(1, selected_sse=225.0, selected_method="RFIS", selected_probability=0.96, tier_flags=fresh_and_all),
            _row(2, selected_sse=36.0, selected_method="UKF", selected_probability=0.97),
        ],
    )

    report, output_rows = build_diagnostic(
        attention_rows=attention_rows,
        local_rows=local_rows,
        mean_rows=mean_rows,
        tiers=("all_eval_non_development", "fresh_extra"),
        top_n=2,
    )

    all_eval = report["tiers"]["all_eval_non_development"]
    assert all_eval["rows"] == 3
    assert all_eval["observed_steps"] == 3
    assert all_eval["pooled_rmse_m"]["edge_only_attention"] == pytest.approx(math.sqrt(525.0 / 3.0))
    assert all_eval["pooled_rmse_m"]["edge_only_local_no_message"] == pytest.approx(math.sqrt(2597.0 / 3.0))
    assert all_eval["pooled_rmse_m"]["edge_only_mean_graph"] == pytest.approx(math.sqrt(325.0 / 3.0))
    assert all_eval["pooled_rmse_m"]["best_single_retained"] == pytest.approx(10.0)
    assert all_eval["row_wtl_local_vs_attention"] == {"wins": 2, "ties": 0, "losses": 1}
    assert all_eval["row_wtl_local_vs_mean"] == {"wins": 1, "ties": 0, "losses": 2}
    assert all_eval["local_minus_attention_row_rmse_delta_quantiles_m"]["p100"] == pytest.approx(30.0)

    top_delta = all_eval["top_local_tail_rows_by_local_minus_attention_delta"][0]
    assert top_delta["trajectory_row"] == "1"
    assert top_delta["local"]["selected_candidate_method"] == "EKF"
    assert top_delta["local"]["selected_probability"] == pytest.approx(0.51)
    assert top_delta["attention"]["selected_candidate_method"] == "RFIS"
    assert top_delta["mean"]["selected_probability"] == pytest.approx(0.96)

    fresh = report["tiers"]["fresh_extra"]
    assert fresh["rows"] == 2
    assert len(output_rows) == 5

    output_dir = tmp_path / "diagnostic"
    outputs = write_outputs(report, output_rows, output_dir)
    parsed = json.loads(Path(outputs["summary_json"]).read_text(encoding="utf-8"))
    assert parsed["schema_version"] == report["schema_version"]
    markdown = Path(outputs["summary_md"]).read_text(encoding="utf-8")
    assert BOUNDARY in markdown
    assert "weak local aggregate is driven by saved-row tail failures" in markdown
    with Path(outputs["rows_csv"]).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert rows[0]["local_minus_attention_rmse_m"] == "-1.0"


def test_build_diagnostic_requires_matching_saved_row_keys(tmp_path: Path) -> None:
    attention_rows = tmp_path / "attention" / "rows.csv"
    local_rows = tmp_path / "local" / "rows.csv"
    mean_rows = tmp_path / "mean" / "rows.csv"
    _write_rows(
        attention_rows,
        [
            _row(0, selected_sse=100.0, selected_method="BatchWLS", selected_probability=0.70),
            _row(1, selected_sse=400.0, selected_method="RFIS", selected_probability=0.80),
        ],
    )
    _write_rows(
        local_rows,
        [
            _row(0, selected_sse=81.0, selected_method="BatchWLS", selected_probability=0.40),
            _row(1, selected_sse=2500.0, selected_method="EKF", selected_probability=0.51),
        ],
    )
    _write_rows(
        mean_rows,
        [_row(0, selected_sse=64.0, selected_method="BatchWLS", selected_probability=0.95)],
    )

    with pytest.raises(ValueError, match="alignment mismatch"):
        build_diagnostic(
            attention_rows=attention_rows,
            local_rows=local_rows,
            mean_rows=mean_rows,
            tiers=("all_eval_non_development",),
        )
