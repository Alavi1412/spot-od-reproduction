from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import scripts.analyze_trajectory_candidate_graph_architecture_ensemble as analysis


def _row(
    *,
    source_name: str,
    trajectory_row: int,
    selected_method: str,
    selected_index: int,
    selected_probability: float,
    selected_rmse: float,
    best_single_rmse: float,
    tier_flags: str = "holdout_seed_ge_67;all_eval_non_development",
) -> dict[str, object]:
    observed_steps = 2
    selected_sse = selected_rmse * selected_rmse * observed_steps
    best_single_sse = best_single_rmse * best_single_rmse * observed_steps
    return {
        "source_dir": f"results/{source_name}",
        "source_name": source_name,
        "seed": 67,
        "split": 67,
        "source_is_extra": False,
        "scenario": "process_noise_shift_test",
        "trajectory_row": trajectory_row,
        "trajectory_index": 100 + trajectory_row,
        "tier_flags": tier_flags,
        "candidate_methods": ["EKF", "UKF"],
        "baseline_candidate_methods": ["EKF", "UKF"],
        "selected_candidate_method": selected_method,
        "selected_candidate_index": selected_index,
        "selected_probability": selected_probability,
        "label_best_observed_method": selected_method,
        "selected_observed_step_rmse_m": selected_rmse,
        "selected_observed_step_sse": selected_sse,
        "selected_observed_steps": observed_steps,
        "best_single_candidate_method": "EKF",
        "best_single_candidate_index": 0,
        "best_single_run_scenario_observed_step_rmse_m": best_single_rmse,
        "best_single_trajectory_observed_step_rmse_m": best_single_rmse,
        "best_single_trajectory_observed_step_sse": best_single_sse,
        "best_single_trajectory_observed_steps": observed_steps,
        "gain_vs_best_single_trajectory_percent": 100.0 * (best_single_rmse - selected_rmse) / best_single_rmse,
        "ekf_trajectory_observed_step_rmse_m": best_single_rmse,
        "ukf_trajectory_observed_step_rmse_m": selected_rmse,
    }


def _summary(rows: list[dict[str, object]], *, output_dir: str, graph_layers: int = 2) -> dict[str, object]:
    return {
        "schema_version": "trajectory_candidate_graph_selector_poc.v1",
        "output_dir": output_dir,
        "prediction_mode": "selector",
        "message_passing_enabled": graph_layers > 0,
        "graph_layers": graph_layers,
        "graph_layer_type": "mean",
        "candidate_methods": ["EKF", "UKF"],
        "baseline_candidate_methods": ["EKF", "UKF"],
        "development_seed_max_exclusive": 67,
        "holdout_seed_min": 67,
        "future_seed_min": 109,
        "scenarios": ["process_noise_shift_test"],
        "rows": rows,
    }


def _write_summary(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_highest_probability_graph_row_is_selected(tmp_path: Path) -> None:
    graph_a = _write_summary(
        tmp_path / "graph_a" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.20,
                    selected_rmse=4.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="graph_a",
        ),
    )
    graph_b = _write_summary(
        tmp_path / "graph_b" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="UKF",
                    selected_index=1,
                    selected_probability=0.85,
                    selected_rmse=1.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="graph_b",
        ),
    )

    summary = analysis.analyze_architecture_ensemble(
        graph_summary_paths=[graph_a, graph_b],
        output_dir=tmp_path / "out",
        name="tiny_architecture_ensemble",
    )

    row = summary["rows"][0]
    assert row["selected_candidate_method"] == "UKF"
    assert row["architecture_ensemble_member"] == "graph_b"
    assert row["architecture_ensemble_selected_probability"] == pytest.approx(0.85)
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "rows.csv").exists()
    assert "does not use evaluation truth" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")


def test_default_rejects_local_non_graph_summary_member(tmp_path: Path) -> None:
    local_member = _write_summary(
        tmp_path / "local_member" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.95,
                    selected_rmse=1.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="local_member",
            graph_layers=0,
        ),
    )

    with pytest.raises(ValueError, match="local/non-graph"):
        analysis.analyze_architecture_ensemble(
            graph_summary_paths=[local_member],
            output_dir=tmp_path / "out",
        )


def test_allow_local_members_accepts_local_summaries_and_selects_by_probability(tmp_path: Path) -> None:
    local_a = _write_summary(
        tmp_path / "local_a" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.30,
                    selected_rmse=4.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="local_a",
            graph_layers=0,
        ),
    )
    local_b = _write_summary(
        tmp_path / "local_b" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="UKF",
                    selected_index=1,
                    selected_probability=0.90,
                    selected_rmse=2.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="local_b",
            graph_layers=0,
        ),
    )

    parsed = analysis.build_parser().parse_args(
        [
            "--graph-summary",
            str(local_a),
            "--graph-summary",
            str(local_b),
            "--output-dir",
            str(tmp_path / "cli_out"),
            "--allow-local-members",
        ]
    )
    assert parsed.allow_local_members is True

    summary = analysis.analyze_architecture_ensemble(
        graph_summary_paths=[local_a, local_b],
        output_dir=tmp_path / "out",
        allow_local_members=True,
    )

    row = summary["rows"][0]
    assert row["selected_candidate_method"] == "UKF"
    assert row["architecture_ensemble_member"] == "local_b"
    assert row["architecture_ensemble_selected_probability"] == pytest.approx(0.90)
    assert summary["allow_local_members"] is True
    assert summary["graph_members_required"] is False
    assert summary["local_no_message_member_count"] == 2
    assert all(member["is_local_no_message_member"] is True for member in summary["graph_members"])


def test_summary_records_member_acceptance_policy_metadata(tmp_path: Path) -> None:
    graph_member = _write_summary(
        tmp_path / "graph_member" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.55,
                    selected_rmse=3.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="graph_member",
        ),
    )

    summary = analysis.analyze_architecture_ensemble(
        graph_summary_paths=[graph_member],
        output_dir=tmp_path / "out",
    )
    written_summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    summary_md = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")

    for payload in (summary, written_summary):
        assert payload["allow_local_members"] is False
        assert payload["graph_members_required"] is True
        assert payload["local_no_message_member_count"] == 0
        assert payload["member_acceptance_policy"] == "graph_members_required"
        assert payload["graph_members"][0]["message_passing_enabled"] is True
        assert payload["graph_members"][0]["is_local_no_message_member"] is False
    assert "Local/no-message members allowed: no" in summary_md
    assert "Graph members required: yes" in summary_md


def test_aggregate_rmse_and_gain_are_computed_correctly(tmp_path: Path) -> None:
    graph_a = _write_summary(
        tmp_path / "graph_a" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.10,
                    selected_rmse=4.0,
                    best_single_rmse=4.0,
                ),
                _row(
                    source_name="source_seed67",
                    trajectory_row=1,
                    selected_method="UKF",
                    selected_index=1,
                    selected_probability=0.90,
                    selected_rmse=3.0,
                    best_single_rmse=2.0,
                ),
            ],
            output_dir="graph_a",
        ),
    )
    graph_b = _write_summary(
        tmp_path / "graph_b" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="UKF",
                    selected_index=1,
                    selected_probability=0.80,
                    selected_rmse=1.0,
                    best_single_rmse=4.0,
                ),
                _row(
                    source_name="source_seed67",
                    trajectory_row=1,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.20,
                    selected_rmse=2.0,
                    best_single_rmse=2.0,
                ),
            ],
            output_dir="graph_b",
        ),
    )

    summary = analysis.analyze_architecture_ensemble(
        graph_summary_paths=[graph_a, graph_b],
        output_dir=tmp_path / "out",
    )

    tier = summary["aggregate_tiers"]["all_eval_non_development"]
    expected_selector_rmse = math.sqrt((1.0 * 1.0 * 2 + 3.0 * 3.0 * 2) / 4)
    expected_best_single_rmse = math.sqrt((4.0 * 4.0 * 2 + 2.0 * 2.0 * 2) / 4)
    expected_gain = 100.0 * (expected_best_single_rmse - expected_selector_rmse) / expected_best_single_rmse
    assert tier["rows"] == 2
    assert tier["observed_steps"] == 4
    assert tier["selector_observed_step_rmse_m"] == pytest.approx(expected_selector_rmse)
    assert tier["best_single_observed_step_rmse_m"] == pytest.approx(expected_best_single_rmse)
    assert tier["gain_vs_best_single_percent"] == pytest.approx(expected_gain)
    assert tier["selected_method_counts"] == {"UKF": 2}
    assert tier["architecture_ensemble_member_counts"] == {"graph_b": 1, "graph_a": 1}


def test_reference_comparison_is_included_but_does_not_affect_selection(tmp_path: Path) -> None:
    graph_a = _write_summary(
        tmp_path / "graph_a" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.70,
                    selected_rmse=2.0,
                    best_single_rmse=3.0,
                )
            ],
            output_dir="graph_a",
        ),
    )
    graph_b = _write_summary(
        tmp_path / "graph_b" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="UKF",
                    selected_index=1,
                    selected_probability=0.20,
                    selected_rmse=1.0,
                    best_single_rmse=3.0,
                )
            ],
            output_dir="graph_b",
        ),
    )
    reference = _write_summary(
        tmp_path / "local_reference" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="UKF",
                    selected_index=1,
                    selected_probability=0.99,
                    selected_rmse=0.5,
                    best_single_rmse=3.0,
                )
            ],
            output_dir="local_reference",
            graph_layers=0,
        ),
    )

    summary = analysis.analyze_architecture_ensemble(
        graph_summary_paths=[graph_a, graph_b],
        output_dir=tmp_path / "out",
        reference_summary_path=reference,
    )

    row = summary["rows"][0]
    assert row["architecture_ensemble_member"] == "graph_a"
    assert row["selected_candidate_method"] == "EKF"
    assert row["reference_selected_candidate_method"] == "UKF"
    assert row["reference_selected_observed_step_rmse_m"] == pytest.approx(0.5)
    tier = summary["aggregate_tiers"]["all_eval_non_development"]
    assert tier["reference_selector_observed_step_rmse_m"] == pytest.approx(0.5)
    assert tier["gain_vs_reference_selector_percent"] == pytest.approx(-300.0)


def test_missing_baseline_candidate_methods_falls_back_to_candidate_methods(tmp_path: Path) -> None:
    graph_a_payload = _summary(
        [
            _row(
                source_name="source_seed67",
                trajectory_row=0,
                selected_method="EKF",
                selected_index=0,
                selected_probability=0.30,
                selected_rmse=4.0,
                best_single_rmse=4.0,
            )
        ],
        output_dir="graph_a",
    )
    graph_b_payload = _summary(
        [
            _row(
                source_name="source_seed67",
                trajectory_row=0,
                selected_method="UKF",
                selected_index=1,
                selected_probability=0.80,
                selected_rmse=1.0,
                best_single_rmse=4.0,
            )
        ],
        output_dir="graph_b",
    )
    for payload in (graph_a_payload, graph_b_payload):
        payload.pop("baseline_candidate_methods")
        for row in payload["rows"]:
            row.pop("baseline_candidate_methods")
    graph_a = _write_summary(tmp_path / "graph_a" / "summary.json", graph_a_payload)
    graph_b = _write_summary(tmp_path / "graph_b" / "summary.json", graph_b_payload)

    summary = analysis.analyze_architecture_ensemble(
        graph_summary_paths=[graph_a, graph_b],
        output_dir=tmp_path / "out",
    )

    assert summary["candidate_methods"] == ["EKF", "UKF"]
    assert summary["baseline_candidate_methods"] == ["EKF", "UKF"]
    assert summary["rows"][0]["baseline_candidate_methods"] == ["EKF", "UKF"]
    assert summary["rows"][0]["selected_candidate_method"] == "UKF"


def test_incompatible_row_keys_raise_clear_error(tmp_path: Path) -> None:
    graph_a = _write_summary(
        tmp_path / "graph_a" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.20,
                    selected_rmse=4.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="graph_a",
        ),
    )
    graph_b = _write_summary(
        tmp_path / "graph_b" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=1,
                    selected_method="UKF",
                    selected_index=1,
                    selected_probability=0.85,
                    selected_rmse=1.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="graph_b",
        ),
    )

    with pytest.raises(ValueError, match="incompatible row keys"):
        analysis.analyze_architecture_ensemble(
            graph_summary_paths=[graph_a, graph_b],
            output_dir=tmp_path / "out",
        )


def test_incompatible_candidate_methods_raise_clear_error(tmp_path: Path) -> None:
    graph_a = _write_summary(
        tmp_path / "graph_a" / "summary.json",
        _summary(
            [
                _row(
                    source_name="source_seed67",
                    trajectory_row=0,
                    selected_method="EKF",
                    selected_index=0,
                    selected_probability=0.20,
                    selected_rmse=4.0,
                    best_single_rmse=4.0,
                )
            ],
            output_dir="graph_a",
        ),
    )
    payload = _summary(
        [
            _row(
                source_name="source_seed67",
                trajectory_row=0,
                selected_method="UKF",
                selected_index=1,
                selected_probability=0.85,
                selected_rmse=1.0,
                best_single_rmse=4.0,
            )
        ],
        output_dir="graph_b",
    )
    payload["candidate_methods"] = ["EKF", "AUKF"]
    payload["rows"][0]["candidate_methods"] = ["EKF", "AUKF"]
    graph_b = _write_summary(tmp_path / "graph_b" / "summary.json", payload)

    with pytest.raises(ValueError, match="candidate methods"):
        analysis.analyze_architecture_ensemble(
            graph_summary_paths=[graph_a, graph_b],
            output_dir=tmp_path / "out",
        )
