from __future__ import annotations

import json
from pathlib import Path

from scripts.build_full_rerun_divergence_audit import (
    REPORT_SCHEMA_VERSION,
    build_audit,
    render_markdown,
    write_outputs,
)


def _write_minimal_inputs(root: Path) -> tuple[Path, Path, Path]:
    metrics_path = root / "metrics_summary.json"
    scorecard_path = root / "scorecard_summary.json"
    trajectory_path = root / "trajectory_errors.csv"
    metrics_path.write_text(
        json.dumps(
            {
                "test": {
                    "EKF": {"pos_rmse_m": 9.0, "diverged": False},
                    "_meta": {},
                },
                "dense_visibility_test": {
                    "UKF": {
                        "pos_rmse_m": 1000.0,
                        "vel_rmse_mps": 10.0,
                        "diverged": True,
                        "num_diverged_trajectories": 1,
                        "divergence_reason": "extreme_trajectory_rmse",
                        "median_traj_pos_rmse_m": 20.0,
                        "max_traj_pos_rmse_m": 1000.0,
                        "max_to_median_traj_pos_rmse_ratio": 50.0,
                    },
                    "LearnedNoiseAdaptive": {
                        "pos_rmse_m": 900.0,
                        "vel_rmse_mps": 9.0,
                        "diverged": True,
                        "num_diverged_trajectories": 2,
                        "divergence_reason": "position_rmse_outlier_ratio",
                        "median_traj_pos_rmse_m": 18.0,
                        "max_traj_pos_rmse_m": 900.0,
                        "max_to_median_traj_pos_rmse_ratio": 50.0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scorecard_path.write_text(
        json.dumps(
            {
                "_thresholds": {"min_stress_improvement_vs_ukf_percent": 2.0},
                "dense_visibility_test": {
                    "LearnedNoiseAdaptive": {"candidate_diverged": True},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory_path.write_text(
        "\n".join(
            [
                "scenario,method,traj_id,traj_pos_rmse_m,traj_vel_rmse_mps",
                "dense_visibility_test,UKF,0,10,1",
                "dense_visibility_test,UKF,1,20000000,2",
                "dense_visibility_test,UKF,2,1000,200000",
                "dense_visibility_test,LearnedNoiseAdaptive,0,9,1",
                "dense_visibility_test,LearnedNoiseAdaptive,1,900,90",
                "dense_visibility_test,LearnedNoiseAdaptive,2,5,200000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics_path, scorecard_path, trajectory_path


def test_build_full_rerun_divergence_audit_schema_and_boundaries(tmp_path: Path) -> None:
    metrics_path, scorecard_path, trajectory_path = _write_minimal_inputs(tmp_path)

    audit = build_audit(
        metrics_path=metrics_path,
        scorecard_path=scorecard_path,
        trajectory_errors_path=trajectory_path,
        generated_utc="2026-06-17T00:00:00Z",
        root=tmp_path,
    )

    assert audit["schema_version"] == REPORT_SCHEMA_VERSION
    assert audit["overall_counts"]["total_scenarios"] == 2
    assert audit["overall_counts"]["scenarios_with_any_divergence"] == [
        "dense_visibility_test"
    ]
    assert audit["overall_counts"]["methods_with_divergence_by_scenario"][
        "dense_visibility_test"
    ] == ["LearnedNoiseAdaptive", "UKF"]
    assert "not a canonical table replacement" in audit["claim_boundary"]
    assert "not operational validation" in audit["claim_boundary"]
    assert "not independent reproduction" in audit["claim_boundary"]
    assert audit["decision_boundary"][
        "no_learned_positive_from_raw_tiny_wins_or_failure_conditioned_rows"
    ]

    ukf = next(case for case in audit["divergence_cases"] if case["method"] == "UKF")
    learned = next(
        case for case in audit["divergence_cases"] if case["method"] == "LearnedNoiseAdaptive"
    )
    assert ukf["scorecard_candidate_diverged"] is None
    assert learned["scorecard_candidate_diverged"] is True
    assert ukf["canonical_manuscript_table_membership"] is None
    assert ukf["num_mask_flagged_trajectories"] == 1
    assert ukf["failure_conditioned_summary"]["num_mask_flagged_trajectories"] == 1
    assert (
        ukf["failure_conditioned_summary"]["num_diverged_trajectories_from_metrics"]
        == 1
    )
    assert "matches metrics num_diverged_trajectories" in ukf[
        "mask_vs_metrics_diverged_trajectory_count_note"
    ]
    assert "num_top_flagged_trajectories_excluded_for_diagnostic_mean" not in ukf[
        "failure_conditioned_summary"
    ]

    # The velocity-only outlier is excluded by row mask even though its
    # position RMSE is not the largest position value. The high position row
    # remains because it is below the evaluator-style position threshold.
    assert ukf["failure_conditioned_summary"]["pos_rmse_m"][
        "mean_pos_rmse_excluding_mask_flagged_trajectories_m"
    ] == 10000005.0
    assert ukf["failure_conditioned_summary"]["pos_rmse_m"][
        "count_excluding_mask_flagged_trajectories"
    ] == 2
    assert 20000000.0 in ukf["failure_conditioned_summary"]["pos_rmse_m"][
        "top_values_desc"
    ]

    assert learned["num_mask_flagged_trajectories"] == 1
    assert "while metrics num_diverged_trajectories reports 2" in learned[
        "mask_vs_metrics_diverged_trajectory_count_note"
    ]
    assert learned["failure_conditioned_summary"]["pos_rmse_m"][
        "mean_pos_rmse_excluding_mask_flagged_trajectories_m"
    ] == 454.5


def test_full_rerun_divergence_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    metrics_path, scorecard_path, trajectory_path = _write_minimal_inputs(tmp_path)
    audit = build_audit(
        metrics_path=metrics_path,
        scorecard_path=scorecard_path,
        trajectory_errors_path=trajectory_path,
        generated_utc="2026-06-17T00:00:00Z",
        root=tmp_path,
    )
    json_out = tmp_path / "audit.json"
    md_out = tmp_path / "audit.md"

    write_outputs(audit, json_out, md_out)

    loaded = json.loads(json_out.read_text(encoding="utf-8"))
    markdown = md_out.read_text(encoding="utf-8")
    assert loaded["schema_version"] == REPORT_SCHEMA_VERSION
    assert "Failure-conditioned rows are for inspection" in markdown
    assert "mean excl. mask flagged rows" in markdown
    assert "paired evaluator-style extreme mask" in markdown
    assert "No learned-positive claim" in render_markdown(audit)


def test_manifest_source_indexes_full_rerun_divergence_audit() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "build_supplementary_manifest.py").read_text(
        encoding="utf-8"
    )

    assert "full_rerun_divergence_audit" in source
    assert "scripts/build_full_rerun_divergence_audit.py" in source
    assert "results/validation/full_rerun_divergence_audit_20260617.json" in source
    assert "results/validation/full_rerun_divergence_audit_20260617.md" in source
