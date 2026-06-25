from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.compare_trajectory_candidate_architecture_summaries as comparison


def _row(
    *,
    source_name: str = "source_seed67",
    scenario: str = "process_noise_shift_test",
    trajectory_row: int = 0,
    selected_rmse: float = 1.0,
    observed_steps: int = 2,
    tier_flags: str = "holdout_seed_ge_67;all_eval_non_development",
    selected_method: str = "EKF",
) -> dict[str, object]:
    return {
        "source_name": source_name,
        "scenario": scenario,
        "trajectory_row": trajectory_row,
        "tier_flags": tier_flags,
        "selected_candidate_method": selected_method,
        "selected_candidate_index": 0,
        "selected_observed_step_rmse_m": selected_rmse,
        "selected_observed_step_sse": selected_rmse * selected_rmse * observed_steps,
        "selected_observed_steps": observed_steps,
    }


def _summary(rows: list[dict[str, object]], *, name: str) -> dict[str, object]:
    return {
        "schema_version": "trajectory_candidate_graph_selector_poc.v1",
        "name": name,
        "output_dir": f"results/{name}",
        "rows": rows,
    }


def _write_summary(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_aligned_summaries_compute_pooled_rmse_and_gain(tmp_path: Path) -> None:
    left = _write_summary(
        tmp_path / "left" / "summary.json",
        _summary(
            [
                _row(trajectory_row=0, selected_rmse=1.0),
                _row(trajectory_row=1, selected_rmse=1.0),
            ],
            name="graph_architecture_ensemble",
        ),
    )
    right = _write_summary(
        tmp_path / "right" / "summary.json",
        _summary(
            [
                _row(trajectory_row=0, selected_rmse=2.0, selected_method="UKF"),
                _row(trajectory_row=1, selected_rmse=2.0, selected_method="UKF"),
            ],
            name="local_no_message_control",
        ),
    )

    summary = comparison.compare_architecture_summaries(
        left_summary_path=left,
        right_summary_path=right,
        output_dir=tmp_path / "out",
        bootstrap_samples=8,
        bootstrap_seed=123,
    )

    tier = summary["aggregate_tiers"]["all_eval_non_development"]
    assert tier["rows"] == 2
    assert tier["observed_steps"] == 4
    assert tier["left_pooled_rmse_m"] == pytest.approx(1.0)
    assert tier["right_pooled_rmse_m"] == pytest.approx(2.0)
    assert tier["left_gain_vs_right_pooled_percent"] == pytest.approx(50.0)
    assert tier["row_wins"] == 2
    assert tier["row_ties"] == 0
    assert tier["row_losses"] == 0
    assert summary["rows"][0]["left_gain_vs_right_row_percent"] == pytest.approx(50.0)
    assert (tmp_path / "out" / "rows.csv").exists()


def test_row_key_mismatch_raises_clear_error(tmp_path: Path) -> None:
    left = _write_summary(
        tmp_path / "left" / "summary.json",
        _summary([_row(trajectory_row=0)], name="left"),
    )
    right = _write_summary(
        tmp_path / "right" / "summary.json",
        _summary([_row(trajectory_row=1)], name="right"),
    )

    with pytest.raises(ValueError, match="row-key mismatch"):
        comparison.compare_architecture_summaries(
            left_summary_path=left,
            right_summary_path=right,
            output_dir=tmp_path / "out",
        )


def test_bootstrap_ci_fields_and_markdown_boundary_language_are_written(tmp_path: Path) -> None:
    left = _write_summary(
        tmp_path / "left" / "summary.json",
        _summary([_row(trajectory_row=0, selected_rmse=1.0)], name="left_graph"),
    )
    right = _write_summary(
        tmp_path / "right" / "summary.json",
        _summary([_row(trajectory_row=0, selected_rmse=2.0)], name="right_local"),
    )

    summary = comparison.compare_architecture_summaries(
        left_summary_path=left,
        right_summary_path=right,
        output_dir=tmp_path / "out",
        bootstrap_samples=6,
        bootstrap_seed=987,
    )

    tier = summary["aggregate_tiers"]["holdout_seed_ge_67"]
    assert "bootstrap_left_gain_vs_right_pooled_percent_ci95_low" in tier
    assert "bootstrap_left_gain_vs_right_pooled_percent_ci95_high" in tier
    assert tier["bootstrap_finite_samples"] == 6
    markdown = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "Saved-row compact-simulator comparison only" in markdown
    assert "not independent-machine reproduction" in markdown
    assert "not operational precise-reference validation" in markdown
    assert "not a full raw/training/all-filter rerun" in markdown
    assert "Selection already happened upstream" in markdown
    assert "does not select using truth" in markdown


def test_json_writer_emits_standard_json_when_row_gain_is_nonfinite(tmp_path: Path) -> None:
    left = _write_summary(
        tmp_path / "left" / "summary.json",
        _summary([_row(trajectory_row=0, selected_rmse=1.0)], name="left"),
    )
    right = _write_summary(
        tmp_path / "right" / "summary.json",
        _summary([_row(trajectory_row=0, selected_rmse=0.0)], name="right"),
    )

    comparison.compare_architecture_summaries(
        left_summary_path=left,
        right_summary_path=right,
        output_dir=tmp_path / "out",
        bootstrap_samples=4,
        bootstrap_seed=42,
    )

    text = (tmp_path / "out" / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text

    def reject_constant(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant emitted: {token}")

    parsed = json.loads(text, parse_constant=reject_constant)
    assert parsed["rows"][0]["left_gain_vs_right_row_percent"] is None
    assert parsed["aggregate_tiers"]["all_eval_non_development"]["left_gain_vs_right_pooled_percent"] is None
