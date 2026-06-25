from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.run_trajectory_candidate_graph_selector_poc as selector


@pytest.mark.parametrize(
    ("directory_name", "expected"),
    [
        ("adaptive_candidate_fusion_observed_fixed_soft_seed23_split23_20260623", (23, 23)),
        ("adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_holdout_20260623", (67, 67)),
        ("adaptive_candidate_fusion_observed_fixed_soft_seed109_split109_future_20260623", (109, 109)),
        ("adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed139_split139_20260625", (139, 139)),
        ("prefix_seed149_split149_suffix_with_more_tokens", (149, 149)),
    ],
)
def test_parse_seed_split_accepts_suffixes(directory_name: str, expected: tuple[int, int]) -> None:
    assert selector.parse_seed_split(Path(directory_name)) == expected


def test_parse_seed_split_rejects_missing_pattern() -> None:
    with pytest.raises(ValueError, match="could not parse seed/split"):
        selector.parse_seed_split(Path("adaptive_candidate_fusion_no_seed"))


def test_resolve_ensemble_member_seeds_derives_from_base_seed() -> None:
    assert selector.resolve_ensemble_member_seeds(
        base_seed=2001,
        ensemble_size=3,
        ensemble_seeds=None,
    ) == [2001, 2002, 2003]


def test_resolve_ensemble_member_seeds_accepts_csv() -> None:
    assert selector.resolve_ensemble_member_seeds(
        base_seed=2001,
        ensemble_size=2,
        ensemble_seeds="3001, 3007",
    ) == [3001, 3007]


@pytest.mark.parametrize(
    ("ensemble_size", "ensemble_seeds", "match"),
    [
        (0, None, "positive"),
        (3, "11,12", "exactly"),
        (2, "11,not_an_int", "CSV of ints"),
        (2, "11,11", "duplicate"),
    ],
)
def test_resolve_ensemble_member_seeds_rejects_invalid_inputs(
    ensemble_size: int,
    ensemble_seeds: str | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        selector.resolve_ensemble_member_seeds(
            base_seed=2001,
            ensemble_size=ensemble_size,
            ensemble_seeds=ensemble_seeds,
        )


def test_observed_step_rmse_and_best_single_baseline_are_run_scenario_level() -> None:
    states = np.zeros((2, 2, 6), dtype=np.float64)
    candidate_bank = np.zeros((2, 2, 2, 6), dtype=np.float64)
    candidate_bank[1, :, 0, 0] = 10.0
    candidate_bank[:, :, 1, 0] = 3.0
    observed_mask = np.ones((2, 2), dtype=bool)

    first_trajectory_rmses = selector.observed_step_rmse_by_candidate(
        states[0],
        candidate_bank[0],
        observed_mask[0],
    )
    baseline = selector.best_single_candidate_baseline(
        states=states,
        candidate_bank=candidate_bank,
        observed_mask=observed_mask,
        candidate_methods=["EKF", "UKF"],
    )

    np.testing.assert_allclose(first_trajectory_rmses, np.array([0.0, 3.0]))
    assert baseline.method == "UKF"
    assert baseline.index == 1
    assert baseline.observed_step_rmse_m == pytest.approx(3.0)


def test_strict_json_writer_sanitizes_nonfinite_values(tmp_path: Path) -> None:
    payload = {
        "python_nan": float("nan"),
        "python_inf": float("inf"),
        "np_scalar": np.float64("-inf"),
        "array": np.asarray([1.5, np.nan, np.inf], dtype=np.float64),
        "nested": {"tuple": (np.float32("nan"), np.int64(4), np.bool_(True))},
    }
    output_path = tmp_path / "summary.json"

    selector.write_strict_json(payload, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text

    def reject_constant(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant emitted: {token}")

    parsed = json.loads(text, parse_constant=reject_constant)
    assert parsed["python_nan"] is None
    assert parsed["python_inf"] is None
    assert parsed["np_scalar"] is None
    assert parsed["array"] == [1.5, None, None]
    assert parsed["nested"]["tuple"] == [None, 4, True]


def test_graph_selector_forward_shape_and_candidate_masking() -> None:
    model = selector.TrajectoryCandidateGraphSelector(
        node_feature_dim=5,
        edge_feature_dim=3,
        hidden_dim=8,
        dropout=0.0,
        graph_layers=1,
    )
    assert model.graph_layer_type == "mean"
    assert isinstance(model.graph_layers[0], selector.CandidateEdgeGraphLayer)

    output = model(
        node_features=torch.randn(2, 3, 5),
        edge_features=torch.randn(2, 3, 3, 3),
        candidate_mask=torch.tensor([[True, False, True], [False, True, True]]),
    )

    assert output["logits"].shape == torch.Size([2, 3])
    assert output["probabilities"].shape == torch.Size([2, 3])
    assert output["probabilities"][0, 1].item() == pytest.approx(0.0)
    assert output["probabilities"][1, 0].item() == pytest.approx(0.0)
    torch.testing.assert_close(output["probabilities"].sum(dim=-1), torch.ones(2))


def test_attention_graph_selector_forward_shape_and_candidate_masking() -> None:
    model = selector.TrajectoryCandidateGraphSelector(
        node_feature_dim=5,
        edge_feature_dim=3,
        hidden_dim=8,
        dropout=0.0,
        graph_layers=1,
        graph_layer_type="attention",
    )
    assert model.graph_layer_type == "attention"
    assert isinstance(model.graph_layers[0], selector.CandidateEdgeAttentionGraphLayer)

    output = model(
        node_features=torch.randn(2, 3, 5),
        edge_features=torch.randn(2, 3, 3, 3),
        candidate_mask=torch.tensor([[True, False, True], [False, True, True]]),
    )

    assert output["logits"].shape == torch.Size([2, 3])
    assert output["probabilities"].shape == torch.Size([2, 3])
    assert output["probabilities"][0, 1].item() == pytest.approx(0.0)
    assert output["probabilities"][1, 0].item() == pytest.approx(0.0)
    torch.testing.assert_close(output["probabilities"].sum(dim=-1), torch.ones(2))


def test_graph_selector_zero_layers_uses_local_node_head_only_with_attention_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("graph message passing should not run when graph_layers=0")

    monkeypatch.setattr(selector.CandidateEdgeGraphLayer, "forward", fail_if_called)
    monkeypatch.setattr(selector.CandidateEdgeAttentionGraphLayer, "forward", fail_if_called)
    model = selector.TrajectoryCandidateGraphSelector(
        node_feature_dim=5,
        edge_feature_dim=3,
        hidden_dim=8,
        dropout=0.0,
        graph_layers=0,
        graph_layer_type="attention",
    )

    output = model(
        node_features=torch.randn(2, 3, 5),
        edge_features=torch.randn(2, 3, 3, 3),
        candidate_mask=torch.tensor([[True, False, True], [False, True, True]]),
    )

    assert len(model.graph_layers) == 0
    assert model.graph_layer_type == "attention"
    assert model.message_passing_enabled is False
    assert output["logits"].shape == torch.Size([2, 3])
    assert output["probabilities"][0, 1].item() == pytest.approx(0.0)
    assert output["probabilities"][1, 0].item() == pytest.approx(0.0)
    torch.testing.assert_close(output["probabilities"].sum(dim=-1), torch.ones(2))


def test_graph_layer_type_defaults_to_mean_in_parser() -> None:
    args = selector.build_parser().parse_args([])

    assert args.graph_layer_type == "mean"


def test_graph_layer_type_rejects_invalid_cli_value() -> None:
    with pytest.raises(SystemExit):
        selector.build_parser().parse_args(["--graph-layer-type", "invalid"])


def test_graph_selector_rejects_invalid_graph_layer_type() -> None:
    with pytest.raises(ValueError, match="graph_layer_type"):
        selector.TrajectoryCandidateGraphSelector(
            node_feature_dim=5,
            edge_feature_dim=3,
            hidden_dim=8,
            dropout=0.0,
            graph_layers=1,
            graph_layer_type="invalid",
        )


def _candidate_with_position_error(states: np.ndarray, error: float) -> np.ndarray:
    candidate = states.copy()
    candidate[:, :, 0] += error
    return candidate


def _write_tiny_prediction_dir(
    source_dir: Path,
    *,
    scenario: str,
    ekf_error: float,
    ukf_error: float,
    extra_method_errors: dict[str, float] | None = None,
) -> None:
    scenario_dir = source_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    states = np.zeros((2, 3, 6), dtype=np.float64)
    visibility = np.ones((2, 3, 1), dtype=np.float64)
    eval_mask = np.ones((2, 3), dtype=bool)
    payload = {
        "states": states,
        "visibility": visibility,
        "eval_mask": eval_mask,
        "trajectory_indices": np.arange(2, dtype=np.int64),
        "ekf": _candidate_with_position_error(states, ekf_error),
        "ukf": _candidate_with_position_error(states, ukf_error),
    }
    for method, error in (extra_method_errors or {}).items():
        payload[selector.candidate_key(method)] = _candidate_with_position_error(states, error)
    np.savez_compressed(
        scenario_dir / selector.PREDICTION_FILENAME,
        **payload,
    )


def test_load_samples_keeps_model_visible_mask_truth_free_when_label_rmse_is_nonfinite(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed5_split5_20260623"
    scenario = "process_noise_shift_test"
    scenario_dir = source_dir / scenario
    scenario_dir.mkdir(parents=True)
    states = np.zeros((1, 2, 6), dtype=np.float64)
    states[0, 0, :3] = np.nan
    visibility = np.ones((1, 2, 1), dtype=np.float64)
    eval_mask = np.ones((1, 2), dtype=bool)
    ekf = np.full_like(states, np.nan)
    ekf[0, 0] = 0.0
    ukf = np.zeros_like(states)
    np.savez_compressed(
        scenario_dir / selector.PREDICTION_FILENAME,
        states=states,
        visibility=visibility,
        eval_mask=eval_mask,
        trajectory_indices=np.array([42], dtype=np.int64),
        ekf=ekf,
        ukf=ukf,
    )

    samples = selector.load_samples(
        source_runs=[selector.SourceRun(path=source_dir, seed=5, split=5)],
        scenarios=[scenario],
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.candidate_mask.tolist() == [True, True]
    assert not np.isfinite(sample.candidate_observed_rmse[0])
    assert np.isfinite(sample.candidate_observed_rmse[1])
    assert sample.label == 1


def test_evaluate_samples_masks_probability_selection_with_model_visible_mask(tmp_path: Path) -> None:
    source_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(source_dir, scenario=scenario, ekf_error=3.0, ukf_error=1.0)
    samples = selector.load_samples(
        source_runs=[selector.SourceRun(path=source_dir, seed=67, split=67)],
        scenarios=[scenario],
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
    )
    for sample in samples:
        sample.candidate_mask = np.asarray([False, True], dtype=bool)

    rows = selector.evaluate_samples_from_probabilities(
        probabilities=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(samples), 1)),
        samples=samples,
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
        development_seed_max_exclusive=67,
        holdout_seed_min=67,
        future_seed_min=109,
    )

    assert {row["selected_candidate_method"] for row in rows} == {"UKF"}


def test_restricted_selector_scores_against_full_baseline_denominator(tmp_path: Path) -> None:
    source_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(
        source_dir,
        scenario=scenario,
        ekf_error=3.0,
        ukf_error=5.0,
        extra_method_errors={
            "AUKF": 4.0,
            "BatchWLS": 6.0,
            "RFIS": 1.0,
            "VA_RFIS": 7.0,
        },
    )
    candidate_methods = ["EKF", "UKF"]
    baseline_candidate_methods = selector.parse_candidate_methods(
        selector.DEFAULT_BASELINE_CANDIDATE_METHODS,
        option_name="--baseline-candidate-methods",
        min_count=1,
    )
    samples = selector.load_samples(
        source_runs=[selector.SourceRun(path=source_dir, seed=67, split=67)],
        scenarios=[scenario],
        candidate_methods=candidate_methods,
        baseline_candidate_methods=baseline_candidate_methods,
    )

    assert [sample.label for sample in samples] == [0, 0]

    probabilities = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(samples), 1))
    rows = selector.evaluate_samples_from_probabilities(
        probabilities=probabilities,
        samples=samples,
        candidate_methods=candidate_methods,
        baseline_candidate_methods=baseline_candidate_methods,
        development_seed_max_exclusive=67,
        holdout_seed_min=67,
        future_seed_min=109,
    )
    aggregates = selector.aggregate_tier_rows(rows)

    assert {row["selected_candidate_method"] for row in rows} == {"EKF"}
    assert {row["best_single_candidate_method"] for row in rows} == {"RFIS"}
    assert {row["best_single_candidate_index"] for row in rows} == {
        baseline_candidate_methods.index("RFIS")
    }
    assert rows[0]["candidate_methods"] == candidate_methods
    assert rows[0]["baseline_candidate_methods"] == baseline_candidate_methods
    assert aggregates["all_eval_non_development"]["baseline_method_counts"] == {"RFIS": 2}
    assert aggregates["all_eval_non_development"]["best_single_observed_step_rmse_m"] == pytest.approx(1.0)
    assert aggregates["all_eval_non_development"]["selector_observed_step_rmse_m"] == pytest.approx(3.0)


def test_missing_baseline_candidate_method_raises_clear_error(tmp_path: Path) -> None:
    source_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(source_dir, scenario=scenario, ekf_error=3.0, ukf_error=5.0)

    with pytest.raises(KeyError, match="missing baseline candidate prediction keys.*rfis.*RFIS"):
        selector.load_samples(
            source_runs=[selector.SourceRun(path=source_dir, seed=67, split=67)],
            scenarios=[scenario],
            candidate_methods=["EKF", "UKF"],
            baseline_candidate_methods=["EKF", "UKF", "RFIS"],
        )


def test_smoke_cli_writes_artifacts_with_cpu_opt_in(tmp_path: Path) -> None:
    dev_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed5_split5_20260623"
    holdout_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    fresh_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed139_split139_20260625"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(dev_dir, scenario=scenario, ekf_error=0.0, ukf_error=4.0)
    _write_tiny_prediction_dir(holdout_dir, scenario=scenario, ekf_error=5.0, ukf_error=1.0)
    _write_tiny_prediction_dir(fresh_dir, scenario=scenario, ekf_error=2.0, ukf_error=1.0)
    output_dir = tmp_path / "out"

    cmd = [
        sys.executable,
        "scripts/run_trajectory_candidate_graph_selector_poc.py",
        "--source-glob",
        str(tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed*_split*_20260623"),
        "--extra-source-dir",
        str(fresh_dir),
        "--output-dir",
        str(output_dir),
        "--scenarios",
        scenario,
        "--candidate-methods",
        "EKF,UKF",
        "--baseline-candidate-methods",
        "EKF,UKF",
        "--epochs",
        "1",
        "--hidden-dim",
        "8",
        "--batch-size",
        "4",
        "--graph-layers",
        "0",
        "--device",
        "cpu",
        "--allow-cpu-smoke",
    ]
    subprocess.run(cmd, cwd=Path.cwd(), check=True, capture_output=True, text=True, timeout=90)

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "rows.csv").exists()
    assert (output_dir / "checkpoints" / "best_selector.pt").exists()
    summary_text = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text
    summary = json.loads(summary_text)
    assert summary["schema_version"] == selector.SCHEMA_VERSION
    assert selector.BOUNDARY_STATEMENT in summary["boundary_statement"]
    assert summary["train_sample_count"] == 2
    assert summary["aggregate_tiers"]["fresh_extra"]["rows"] == 2
    assert summary["aggregate_tiers"]["all_eval_non_development"]["rows"] == 4
    assert summary["graph_layers"] == 0
    assert summary["graph_layer_type"] == "mean"
    assert summary["message_passing_enabled"] is False
    assert summary["model_kwargs"]["graph_layers"] == 0
    assert summary["model_kwargs"]["graph_layer_type"] == "mean"
    checkpoint = torch.load(output_dir / "checkpoints" / "best_selector.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_kwargs"]["graph_layers"] == 0
    assert checkpoint["model_kwargs"]["graph_layer_type"] == "mean"
    assert checkpoint["graph_layers"] == 0
    assert checkpoint["graph_layer_type"] == "mean"
    assert checkpoint["message_passing_enabled"] is False
    assert checkpoint["config"]["graph_layers"] == 0
    assert checkpoint["config"]["graph_layer_type"] == "mean"
    assert checkpoint["config"]["message_passing_enabled"] is False


def test_smoke_cli_writes_ensemble_member_artifacts_with_cpu_opt_in(tmp_path: Path) -> None:
    dev_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed5_split5_20260623"
    holdout_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(dev_dir, scenario=scenario, ekf_error=0.0, ukf_error=4.0)
    _write_tiny_prediction_dir(holdout_dir, scenario=scenario, ekf_error=5.0, ukf_error=1.0)
    output_dir = tmp_path / "ensemble_out"

    cmd = [
        sys.executable,
        "scripts/run_trajectory_candidate_graph_selector_poc.py",
        "--source-glob",
        str(tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed*_split*_20260623"),
        "--output-dir",
        str(output_dir),
        "--scenarios",
        scenario,
        "--candidate-methods",
        "EKF,UKF",
        "--baseline-candidate-methods",
        "EKF,UKF",
        "--epochs",
        "1",
        "--hidden-dim",
        "8",
        "--batch-size",
        "4",
        "--ensemble-size",
        "2",
        "--ensemble-seeds",
        "101,103",
        "--device",
        "cpu",
        "--allow-cpu-smoke",
    ]
    subprocess.run(cmd, cwd=Path.cwd(), check=True, capture_output=True, text=True, timeout=90)

    for member_seed in (101, 103):
        checkpoint_dir = output_dir / "checkpoints" / f"member_seed{member_seed}"
        assert (checkpoint_dir / "best_selector.pt").exists()
        assert (checkpoint_dir / "last_selector.pt").exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ensemble_size"] == 2
    assert summary["ensemble_member_seeds"] == [101, 103]
    assert summary["graph_layers"] == 2
    assert summary["graph_layer_type"] == "mean"
    assert summary["message_passing_enabled"] is True
    assert summary["evaluation_used_averaged_probabilities"] is True
    assert summary["ensemble"]["probability_aggregation"] == "arithmetic_mean"
    assert summary["ensemble"]["evaluation_used_averaged_probabilities"] is True
    assert [member["member_seed"] for member in summary["training"]["members"]] == [101, 103]
    assert len(summary["rows"]) == 4
