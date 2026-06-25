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
    assert "position_offsets" not in output
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
    assert args.prediction_mode == "selector"
    assert args.node_disagreement_features == "include"
    assert args.residual_loss_weight == pytest.approx(1.0)
    assert args.development_validation_seed_min is None
    assert selector.build_feature_names(["scenario"], ["EKF", "UKF"]) == selector.build_feature_names(
        ["scenario"],
        ["EKF", "UKF"],
        node_disagreement_features="include",
    )


def test_omit_node_disagreement_features_only_reduces_node_dimension() -> None:
    scenarios = ["process_noise_shift_test"]
    candidate_methods = ["EKF", "UKF"]
    candidate_bank = np.zeros((3, 2, 6), dtype=np.float64)
    candidate_bank[:, 1, 0] = np.asarray([1.0, 2.0, 3.0])
    visibility = np.ones((3, 1), dtype=np.float64)
    eval_mask = np.asarray([True, True, True], dtype=bool)
    observed_mask = np.asarray([True, True, False], dtype=bool)

    include_names = selector.build_feature_names(
        scenarios,
        candidate_methods,
        node_disagreement_features="include",
    )
    omit_names = selector.build_feature_names(
        scenarios,
        candidate_methods,
        node_disagreement_features="omit",
    )
    include_nodes, include_edges, include_mask = selector.build_candidate_graph_features(
        candidate_bank=candidate_bank,
        visibility=visibility,
        eval_mask=eval_mask,
        observed_mask=observed_mask,
        scenario=scenarios[0],
        scenarios=scenarios,
        candidate_methods=candidate_methods,
        node_disagreement_features="include",
    )
    omit_nodes, omit_edges, omit_mask = selector.build_candidate_graph_features(
        candidate_bank=candidate_bank,
        visibility=visibility,
        eval_mask=eval_mask,
        observed_mask=observed_mask,
        scenario=scenarios[0],
        scenarios=scenarios,
        candidate_methods=candidate_methods,
        node_disagreement_features="omit",
    )
    edge_names = selector.build_edge_feature_names()

    assert set(selector.NODE_DISAGREEMENT_FEATURE_NAMES).issubset(include_names)
    assert not set(selector.NODE_DISAGREEMENT_FEATURE_NAMES).intersection(omit_names)
    assert len(include_names) - len(omit_names) == 8
    assert include_nodes.shape == (2, len(include_names))
    assert omit_nodes.shape == (2, len(omit_names))
    assert include_nodes.shape[-1] - omit_nodes.shape[-1] == 8
    assert include_edges.shape == (2, 2, len(edge_names))
    assert omit_edges.shape == include_edges.shape
    np.testing.assert_allclose(omit_edges, include_edges)
    np.testing.assert_array_equal(omit_mask, include_mask)


def test_graph_layer_type_rejects_invalid_cli_value() -> None:
    with pytest.raises(SystemExit):
        selector.build_parser().parse_args(["--graph-layer-type", "invalid"])


def test_prediction_mode_rejects_invalid_cli_value() -> None:
    with pytest.raises(SystemExit):
        selector.build_parser().parse_args(["--prediction-mode", "invalid"])


@pytest.mark.parametrize("raw_weight", ["-0.1", "nan", "inf"])
def test_residual_loss_weight_rejects_invalid_values(raw_weight: str) -> None:
    args = selector.build_parser().parse_args(["--residual-loss-weight", raw_weight])

    with pytest.raises(SystemExit, match="residual-loss-weight"):
        selector.validate_args(args)


@pytest.mark.parametrize("validation_seed_min", ["67", "68"])
def test_development_validation_seed_min_must_be_below_development_max(
    validation_seed_min: str,
) -> None:
    args = selector.build_parser().parse_args(
        [
            "--development-seed-max-exclusive",
            "67",
            "--development-validation-seed-min",
            validation_seed_min,
        ]
    )

    with pytest.raises(SystemExit, match="development-validation-seed-min"):
        selector.validate_args(args)


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


def test_residual_refine_forward_returns_offsets_and_masks_invalid_candidates() -> None:
    model = selector.TrajectoryCandidateGraphSelector(
        node_feature_dim=5,
        edge_feature_dim=3,
        hidden_dim=8,
        dropout=0.0,
        graph_layers=1,
        prediction_mode="residual_refine",
    )

    output = model(
        node_features=torch.randn(2, 3, 5),
        edge_features=torch.randn(2, 3, 3, 3),
        candidate_mask=torch.tensor([[True, False, True], [False, True, True]]),
    )

    assert output["logits"].shape == torch.Size([2, 3])
    assert output["probabilities"].shape == torch.Size([2, 3])
    assert output["position_offsets"].shape == torch.Size([2, 3, 3])
    torch.testing.assert_close(output["position_offsets"][0, 1], torch.zeros(3))
    torch.testing.assert_close(output["position_offsets"][1, 0], torch.zeros(3))


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


def test_development_validation_seed_min_splits_development_samples(tmp_path: Path) -> None:
    train_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed5_split5_20260623"
    validation_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed23_split23_20260623"
    holdout_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    scenario = "process_noise_shift_test"
    for source_dir in (train_dir, validation_dir, holdout_dir):
        _write_tiny_prediction_dir(source_dir, scenario=scenario, ekf_error=1.0, ukf_error=2.0)
    samples = selector.load_samples(
        source_runs=[
            selector.SourceRun(path=train_dir, seed=5, split=5),
            selector.SourceRun(path=validation_dir, seed=23, split=23),
            selector.SourceRun(path=holdout_dir, seed=67, split=67),
        ],
        scenarios=[scenario],
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
    )

    default_split = selector.split_development_samples(
        samples,
        development_seed_max_exclusive=67,
    )
    assert default_split.validation_enabled is False
    assert len(default_split.development_samples) == 4
    assert len(default_split.train_samples) == 4
    assert default_split.validation_samples == []

    validation_split = selector.split_development_samples(
        samples,
        development_seed_max_exclusive=67,
        development_validation_seed_min=23,
    )
    selector.validate_development_sample_split(validation_split)
    assert validation_split.validation_enabled is True
    assert {sample.seed for sample in validation_split.train_samples} == {5}
    assert {sample.seed for sample in validation_split.validation_samples} == {23}
    assert len(validation_split.train_samples) == 2
    assert len(validation_split.validation_samples) == 2

    empty_train_split = selector.split_development_samples(
        samples,
        development_seed_max_exclusive=67,
        development_validation_seed_min=5,
    )
    with pytest.raises(SystemExit, match="training samples"):
        selector.validate_development_sample_split(empty_train_split)

    empty_validation_split = selector.split_development_samples(
        samples,
        development_seed_max_exclusive=67,
        development_validation_seed_min=66,
    )
    with pytest.raises(SystemExit, match="validation samples"):
        selector.validate_development_sample_split(empty_validation_split)


def test_train_selector_model_records_validation_checkpoint_metadata(tmp_path: Path) -> None:
    train_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed5_split5_20260623"
    validation_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed23_split23_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(train_dir, scenario=scenario, ekf_error=0.0, ukf_error=3.0)
    _write_tiny_prediction_dir(validation_dir, scenario=scenario, ekf_error=3.0, ukf_error=0.0)
    samples = selector.load_samples(
        source_runs=[
            selector.SourceRun(path=train_dir, seed=5, split=5),
            selector.SourceRun(path=validation_dir, seed=23, split=23),
        ],
        scenarios=[scenario],
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
    )
    development_split = selector.split_development_samples(
        samples,
        development_seed_max_exclusive=67,
        development_validation_seed_min=23,
    )
    selector.validate_development_sample_split(development_split)
    model_kwargs = {
        "node_feature_dim": int(development_split.train_samples[0].node_features.shape[-1]),
        "edge_feature_dim": int(development_split.train_samples[0].edge_features.shape[-1]),
        "hidden_dim": 8,
        "dropout": 0.0,
        "graph_layers": 0,
        "graph_layer_type": "mean",
        "prediction_mode": "residual_refine",
        "residual_offset_dim": selector.RESIDUAL_POSITION_DIM,
    }
    model = selector.TrajectoryCandidateGraphSelector(**model_kwargs)
    training = selector.train_selector_model(
        train_samples=development_split.train_samples,
        validation_samples=development_split.validation_samples,
        model=model,
        output_dir=tmp_path / "out",
        device=torch.device("cpu"),
        epochs=1,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        seed=123,
        model_kwargs=model_kwargs,
        candidate_methods=["EKF", "UKF"],
        feature_names=selector.build_feature_names([scenario], ["EKF", "UKF"]),
        edge_feature_names=selector.build_edge_feature_names(),
        config={
            "prediction_mode": "residual_refine",
            "residual_loss_weight": 0.5,
            "residual_offset_dim": selector.RESIDUAL_POSITION_DIM,
            "residual_offset_application": selector.RESIDUAL_OFFSET_APPLICATION,
        },
        prediction_mode="residual_refine",
        residual_loss_weight=0.5,
    )

    assert training["fit_sample_count"] == 2
    assert training["train_sample_count"] == 2
    assert training["validation_sample_count"] == 2
    assert training["checkpoint_selection_metric"] == "validation_loss"
    assert training["best_train_loss"] == pytest.approx(training["history"]["train_loss"][0])
    assert training["best_validation_loss"] == pytest.approx(training["history"]["validation_loss"][0])
    assert len(training["history"]["validation_loss"]) == 1
    assert len(training["history"]["validation_ce_loss"]) == 1
    assert len(training["history"]["validation_residual_mse"]) == 1

    checkpoint = torch.load(training["best_checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["history"]["validation_loss"] == training["history"]["validation_loss"]
    assert checkpoint["config"]["fit_sample_count"] == 2
    assert checkpoint["config"]["validation_sample_count"] == 2
    assert checkpoint["config"]["checkpoint_selection_metric"] == "validation_loss"
    assert checkpoint["config"]["best_validation_loss"] == pytest.approx(training["best_validation_loss"])


def test_train_selector_model_restores_in_memory_model_to_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed5_split5_20260623"
    validation_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed23_split23_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(train_dir, scenario=scenario, ekf_error=0.0, ukf_error=3.0)
    _write_tiny_prediction_dir(validation_dir, scenario=scenario, ekf_error=3.0, ukf_error=0.0)
    samples = selector.load_samples(
        source_runs=[
            selector.SourceRun(path=train_dir, seed=5, split=5),
            selector.SourceRun(path=validation_dir, seed=23, split=23),
        ],
        scenarios=[scenario],
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
    )
    development_split = selector.split_development_samples(
        samples,
        development_seed_max_exclusive=67,
        development_validation_seed_min=23,
    )
    selector.validate_development_sample_split(development_split)
    model_kwargs = {
        "node_feature_dim": int(development_split.train_samples[0].node_features.shape[-1]),
        "edge_feature_dim": int(development_split.train_samples[0].edge_features.shape[-1]),
        "hidden_dim": 8,
        "dropout": 0.0,
        "graph_layers": 0,
        "graph_layer_type": "mean",
        "prediction_mode": "selector",
        "residual_offset_dim": selector.RESIDUAL_POSITION_DIM,
    }
    model = selector.TrajectoryCandidateGraphSelector(**model_kwargs)
    validation_losses = iter([0.0, 1.0])

    def fake_evaluate_selector_loss(**_: object) -> dict[str, float]:
        loss = next(validation_losses)
        return {"loss": loss, "ce_loss": loss, "residual_mse": 0.0}

    monkeypatch.setattr(selector, "evaluate_selector_loss", fake_evaluate_selector_loss)
    training = selector.train_selector_model(
        train_samples=development_split.train_samples,
        validation_samples=development_split.validation_samples,
        model=model,
        output_dir=tmp_path / "out",
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
        learning_rate=0.05,
        weight_decay=0.0,
        seed=123,
        model_kwargs=model_kwargs,
        candidate_methods=["EKF", "UKF"],
        feature_names=selector.build_feature_names([scenario], ["EKF", "UKF"]),
        edge_feature_names=selector.build_edge_feature_names(),
        config={
            "prediction_mode": "selector",
            "residual_loss_weight": 1.0,
            "residual_offset_dim": selector.RESIDUAL_POSITION_DIM,
            "residual_offset_application": selector.RESIDUAL_OFFSET_APPLICATION,
        },
        prediction_mode="selector",
        residual_loss_weight=1.0,
    )

    best_checkpoint = torch.load(training["best_checkpoint"], map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(training["last_checkpoint"], map_location="cpu", weights_only=False)
    assert training["history"]["validation_loss"] == [0.0, 1.0]
    assert any(
        not torch.equal(best_checkpoint["model_state_dict"][name], last_checkpoint["model_state_dict"][name])
        for name in best_checkpoint["model_state_dict"]
    )
    for name, best_value in best_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(model.state_dict()[name], best_value)


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


def test_residual_refine_evaluation_scores_probability_weighted_offsets(tmp_path: Path) -> None:
    source_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(source_dir, scenario=scenario, ekf_error=2.0, ukf_error=2.0)
    samples = selector.load_samples(
        source_runs=[selector.SourceRun(path=source_dir, seed=67, split=67)],
        scenarios=[scenario],
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
    )
    probabilities = np.tile(np.asarray([[0.25, 0.75]], dtype=np.float32), (len(samples), 1))
    position_offsets = np.zeros((len(samples), 2, 3), dtype=np.float32)
    position_offsets[:, :, 0] = -2.0

    rows = selector.evaluate_samples_from_probabilities(
        probabilities=probabilities,
        position_offsets=position_offsets,
        samples=samples,
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
        development_seed_max_exclusive=67,
        holdout_seed_min=67,
        future_seed_min=109,
        prediction_mode="residual_refine",
    )

    assert {row["prediction_mode"] for row in rows} == {"residual_refine"}
    assert {row["selected_candidate_method"] for row in rows} == {"UKF"}
    assert {row["selected_prediction_source"] for row in rows} == {
        "probability_weighted_candidate_plus_offset"
    }
    assert {row["residual_offset_application"] for row in rows} == {
        selector.RESIDUAL_OFFSET_APPLICATION
    }
    for row in rows:
        assert row["selected_observed_step_rmse_m"] == pytest.approx(0.0)
        assert row["best_single_trajectory_observed_step_rmse_m"] == pytest.approx(2.0)


def test_predict_ensemble_outputs_returns_averaged_residual_offsets(tmp_path: Path) -> None:
    source_dir = tmp_path / "adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623"
    scenario = "process_noise_shift_test"
    _write_tiny_prediction_dir(source_dir, scenario=scenario, ekf_error=1.0, ukf_error=2.0)
    samples = selector.load_samples(
        source_runs=[selector.SourceRun(path=source_dir, seed=67, split=67)],
        scenarios=[scenario],
        candidate_methods=["EKF", "UKF"],
        baseline_candidate_methods=["EKF", "UKF"],
    )
    models = [
        selector.TrajectoryCandidateGraphSelector(
            node_feature_dim=samples[0].node_features.shape[-1],
            edge_feature_dim=samples[0].edge_features.shape[-1],
            hidden_dim=8,
            dropout=0.0,
            graph_layers=0,
            prediction_mode="residual_refine",
        )
        for _ in range(2)
    ]
    for model, x_bias in zip(models, (1.0, 3.0), strict=True):
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
            assert model.residual_head is not None
            model.residual_head[-1].bias.copy_(torch.tensor([x_bias, 0.0, 0.0]))

    outputs = selector.predict_ensemble_outputs(
        models=models,
        samples=samples,
        device=torch.device("cpu"),
        batch_size=4,
        include_position_offsets=True,
    )

    assert outputs["probabilities"].shape == (len(samples), 2)
    assert outputs["position_offsets"].shape == (len(samples), 2, 3)
    np.testing.assert_allclose(outputs["position_offsets"][:, :, 0], 2.0, atol=1.0e-6)
    np.testing.assert_allclose(outputs["position_offsets"][:, :, 1:], 0.0, atol=1.0e-6)


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
        "--node-disagreement-features",
        "omit",
        "--device",
        "cpu",
        "--allow-cpu-smoke",
    ]
    subprocess.run(cmd, cwd=Path.cwd(), check=True, capture_output=True, text=True, timeout=90)

    assert (output_dir / "summary.json").exists()
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "rows.csv").exists()
    assert (output_dir / "checkpoints" / "best_selector.pt").exists()
    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "Node disagreement features: omit" in summary_md
    assert "Retained candidate-disagreement node features were omitted" in summary_md
    summary_text = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in summary_text
    assert "Infinity" not in summary_text
    summary = json.loads(summary_text)
    assert summary["schema_version"] == selector.SCHEMA_VERSION
    assert selector.BOUNDARY_STATEMENT in summary["boundary_statement"]
    assert summary["train_sample_count"] == 2
    assert summary["development_validation_seed_min"] is None
    assert summary["development_validation_enabled"] is False
    assert summary["development_sample_count"] == 2
    assert summary["fit_sample_count"] == 2
    assert summary["validation_sample_count"] == 0
    assert summary["checkpoint_selection_metric"] == "train_loss"
    assert summary["training"]["validation_sample_count"] == 0
    assert summary["training"]["checkpoint_selection_metric"] == "train_loss"
    assert summary["aggregate_tiers"]["fresh_extra"]["rows"] == 2
    assert summary["aggregate_tiers"]["all_eval_non_development"]["rows"] == 4
    assert summary["graph_layers"] == 0
    assert summary["graph_layer_type"] == "mean"
    assert summary["prediction_mode"] == "selector"
    assert summary["node_disagreement_features"] == "omit"
    assert not set(selector.NODE_DISAGREEMENT_FEATURE_NAMES).intersection(summary["feature_names"])
    assert summary["residual_loss_weight"] == pytest.approx(1.0)
    assert summary["residual_offset_dim"] == selector.RESIDUAL_POSITION_DIM
    assert summary["residual_offset_application"] == selector.RESIDUAL_OFFSET_APPLICATION
    assert summary["message_passing_enabled"] is False
    assert summary["model_kwargs"]["graph_layers"] == 0
    assert summary["model_kwargs"]["graph_layer_type"] == "mean"
    assert summary["model_kwargs"]["prediction_mode"] == "selector"
    assert summary["model_kwargs"]["residual_offset_dim"] == selector.RESIDUAL_POSITION_DIM
    checkpoint = torch.load(output_dir / "checkpoints" / "best_selector.pt", map_location="cpu", weights_only=False)
    assert checkpoint["model_kwargs"]["graph_layers"] == 0
    assert checkpoint["model_kwargs"]["graph_layer_type"] == "mean"
    assert checkpoint["model_kwargs"]["prediction_mode"] == "selector"
    assert checkpoint["model_kwargs"]["residual_offset_dim"] == selector.RESIDUAL_POSITION_DIM
    assert checkpoint["graph_layers"] == 0
    assert checkpoint["graph_layer_type"] == "mean"
    assert checkpoint["prediction_mode"] == "selector"
    assert checkpoint["node_disagreement_features"] == "omit"
    assert checkpoint["residual_loss_weight"] == pytest.approx(1.0)
    assert checkpoint["residual_offset_dim"] == selector.RESIDUAL_POSITION_DIM
    assert checkpoint["residual_offset_application"] == selector.RESIDUAL_OFFSET_APPLICATION
    assert checkpoint["message_passing_enabled"] is False
    assert checkpoint["config"]["graph_layers"] == 0
    assert checkpoint["config"]["graph_layer_type"] == "mean"
    assert checkpoint["config"]["prediction_mode"] == "selector"
    assert checkpoint["config"]["node_disagreement_features"] == "omit"
    assert checkpoint["config"]["residual_loss_weight"] == pytest.approx(1.0)
    assert checkpoint["config"]["residual_offset_dim"] == selector.RESIDUAL_POSITION_DIM
    assert checkpoint["config"]["residual_offset_application"] == selector.RESIDUAL_OFFSET_APPLICATION
    assert checkpoint["config"]["message_passing_enabled"] is False
    assert checkpoint["config"]["validation_sample_count"] == 0
    assert checkpoint["config"]["checkpoint_selection_metric"] == "train_loss"


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
    assert summary["prediction_mode"] == "selector"
    assert summary["node_disagreement_features"] == "include"
    assert summary["residual_loss_weight"] == pytest.approx(1.0)
    assert summary["residual_offset_application"] == selector.RESIDUAL_OFFSET_APPLICATION
    assert summary["message_passing_enabled"] is True
    assert summary["evaluation_used_averaged_probabilities"] is True
    assert summary["ensemble"]["probability_aggregation"] == "arithmetic_mean"
    assert summary["ensemble"]["evaluation_used_averaged_probabilities"] is True
    assert [member["member_seed"] for member in summary["training"]["members"]] == [101, 103]
    assert len(summary["rows"]) == 4
