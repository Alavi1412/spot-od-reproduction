from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import scripts.analyze_adaptive_candidate_fusion_global_portfolio as gp


def _record(
    *,
    states: np.ndarray,
    components: dict[str, np.ndarray],
    candidate_methods: list[str] | None = None,
    scenario: str = "process_noise_shift_test",
    seed: int = 1,
    visibility: np.ndarray | None = None,
) -> gp.PortfolioRecord:
    if visibility is None:
        visibility = np.ones(states.shape[:2] + (1,), dtype=np.float64)
    return gp.PortfolioRecord(
        run_dir=Path(f"results/example_seed{seed}_split{seed}"),
        run_name=f"example_seed{seed}_split{seed}",
        seed=seed,
        split=seed,
        scenario=scenario,
        role="val",
        states=states,
        visibility=visibility,
        eval_mask=np.ones(states.shape[:2], dtype=bool),
        components=components,
        candidate_methods=candidate_methods or ["EKF", "RFIS"],
        trajectory_indices=list(range(states.shape[0])),
    )


def test_observed_step_mask_and_rmse_use_visible_finite_steps() -> None:
    states = np.zeros((1, 3, 6), dtype=np.float64)
    preds = np.zeros_like(states)
    preds[0, :, 0] = [3.0, 4.0, 5.0]
    visibility = np.array([[[1.0], [0.0], [1.0]]], dtype=np.float64)
    eval_mask = np.ones((1, 3), dtype=bool)

    np.testing.assert_array_equal(
        gp.observed_step_mask(visibility),
        np.array([[True, False, True]]),
    )

    observed_sse, observed_count = gp.metric_sse_count(
        states=states,
        predictions=preds,
        visibility=visibility,
        eval_mask=eval_mask,
        metric="observed_step_pos_rmse_m",
    )
    all_sse, all_count = gp.metric_sse_count(
        states=states,
        predictions=preds,
        visibility=visibility,
        eval_mask=eval_mask,
        metric="all_step_pos_rmse_m",
    )

    assert observed_count == 2
    assert observed_sse == 34.0
    assert all_count == 3
    assert all_sse == 50.0
    assert gp.masked_pos_rmse(
        states=states,
        predictions=preds,
        visibility=visibility,
        eval_mask=eval_mask,
        metric="observed_step_pos_rmse_m",
    ) == np.sqrt(17.0)


def test_policy_grid_and_application_include_learned_candidate_blend() -> None:
    policies = gp.policy_grid(["EKF", "RFIS"], blend_grid_step=0.5)
    assert len(policies) == 10
    blend = next(policy for policy in policies if policy["policy_id"] == "blend:learned:RFIS:0.5")

    components = {
        "learned": np.full((1, 2, 6), 10.0, dtype=np.float64),
        "learned_hard": np.full((1, 2, 6), -5.0, dtype=np.float64),
        "EKF": np.full((1, 2, 6), 2.0, dtype=np.float64),
        "RFIS": np.full((1, 2, 6), 6.0, dtype=np.float64),
    }

    np.testing.assert_allclose(gp.apply_policy(blend, components), np.full((1, 2, 6), 8.0))


def test_policy_grid_family_filtering_partitions_learned_and_nonlearned_policies() -> None:
    policies = gp.policy_grid(["EKF", "RFIS"], blend_grid_step=0.5)

    all_policies = gp.filter_policy_grid_by_family(policies, "all")
    learned_including = gp.filter_policy_grid_by_family(policies, "learned_including")
    nonlearned_only = gp.filter_policy_grid_by_family(policies, "nonlearned_only")

    assert all_policies == policies
    assert len(learned_including) == 7
    assert len(nonlearned_only) == 3
    assert len(learned_including) + len(nonlearned_only) == len(policies)
    assert all(gp.policy_includes_learned_component(policy) for policy in learned_including)
    assert not any(gp.policy_includes_learned_component(policy) for policy in nonlearned_only)
    assert {policy["policy_id"] for policy in nonlearned_only} == {
        "single:EKF",
        "single:RFIS",
        "blend:EKF:RFIS:0.5",
    }
    with pytest.raises(ValueError, match="unsupported policy family"):
        gp.filter_policy_grid_by_family(policies, "unknown")


def _constant_x_component(value: float) -> np.ndarray:
    component = np.zeros((1, 2, 6), dtype=np.float64)
    component[:, :, 0] = float(value)
    return component


def test_policy_family_scenario_diagnostics_select_distinct_synthetic_policies() -> None:
    scenario = "process_noise_shift_test"
    states = _constant_x_component(1.0)
    components = {
        "learned": _constant_x_component(1.0),
        "learned_hard": _constant_x_component(8.0),
        "EKF": _constant_x_component(0.0),
        "RFIS": _constant_x_component(2.0),
    }
    records = [
        _record(
            states=states,
            components=components,
            candidate_methods=["EKF", "RFIS"],
            scenario=scenario,
            seed=7,
        )
    ]

    family_policies = gp.select_policy_family_scenario_policies(
        records,
        [scenario],
        selection_metric="observed_step_pos_rmse_m",
        blend_grid_step=0.5,
    )

    assert family_policies["all"][scenario]["policy_id"] == "single:learned"
    assert family_policies["learned_including"][scenario]["policy_id"] == "single:learned"
    assert family_policies["nonlearned_only"][scenario]["policy_id"] == "blend:EKF:RFIS:0.5"
    assert family_policies["nonlearned_only"][scenario]["policy_family"] == "nonlearned_only"

    family_diagnostics = gp.evaluate_policy_family_diagnostics(
        records,
        family_policies,
        bootstrap_samples=64,
        bootstrap_seed=99,
    )

    assert family_diagnostics["nonlearned_only"]["summary"]["wins"] == 1
    assert family_diagnostics["nonlearned_only"]["summary"]["rows"] == 1
    assert family_diagnostics["nonlearned_only"]["statistics"]["seed_paired"]["seed_wins"] == 1

    summary = {
        "schema_version": gp.SCHEMA_VERSION,
        "boundary_language": gp.BOUNDARY_LANGUAGE,
        "inputs": {"run_dirs": ["results/example_seed7_split7"]},
        "validation": {
            "global_scenario_policies": family_policies["all"],
            "global_all_scenarios_policy": family_policies["all"][scenario],
            "policy_family_scenario_policies": family_policies,
        },
        "eval": {
            "global_scenario_policy_rows": family_diagnostics["all"]["scenario_policy_rows"],
            "global_scenario_policy_summary": family_diagnostics["all"]["summary"],
            "global_scenario_policy_by_scenario": family_diagnostics["all"]["by_scenario"],
            "global_scenario_policy_statistics": family_diagnostics["all"]["statistics"],
            "global_all_scenarios_policy_summary": family_diagnostics["all"]["summary"],
            "policy_family_diagnostics": family_diagnostics,
        },
    }
    markdown = gp.render_markdown(summary)

    assert "## Policy Family Diagnostics" in markdown
    assert "`nonlearned_only` is a validation-selected blend baseline" in markdown
    assert "`process_noise_shift_test`: `0.50*EKF + 0.50*RFIS`" in markdown


def test_parse_seed_split_keeps_directory_seed_separate_from_training_seed() -> None:
    config = {
        "seed": 42,
        "scenario_trajectory_splits": {"split_seed": 37},
    }

    seed, split = gp._parse_seed_split(
        Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed37_split37_20260624"),
        config,
    )

    assert seed == 37
    assert split == 37
    assert gp._training_seed_from_config(config) == 42


def test_parse_seed_split_falls_back_to_split_seed_when_directory_seed_missing() -> None:
    seed, split = gp._parse_seed_split(
        Path("results/adaptive_candidate_fusion_manual"),
        {"seed": 42, "scenario_trajectory_splits": {"split_seed": 37}},
    )

    assert seed == 37
    assert split == 37


def test_parser_exposes_deterministic_bootstrap_defaults() -> None:
    args = gp.build_parser().parse_args([])

    assert args.bootstrap_samples == 20_000
    assert args.bootstrap_seed == 12_345


def test_global_policy_selection_uses_pooled_validation_records() -> None:
    states = np.zeros((1, 3, 6), dtype=np.float64)
    states[:, :, 0] = 1.0
    learned = np.zeros((1, 3, 6), dtype=np.float64)
    learned_hard = np.zeros((1, 3, 6), dtype=np.float64)
    ekf = np.zeros((1, 3, 6), dtype=np.float64)
    rfis = np.zeros((1, 3, 6), dtype=np.float64)
    learned[:, :, 0] = 0.0
    learned_hard[:, :, 0] = 4.0
    ekf[:, :, 0] = 3.0
    rfis[:, :, 0] = 2.0
    components = {
        "learned": learned,
        "learned_hard": learned_hard,
        "EKF": ekf,
        "RFIS": rfis,
    }
    records = [
        _record(states=states, components=components),
        _record(states=states, components=components),
    ]

    policy = gp.select_global_policy(
        records,
        selection_metric="observed_step_pos_rmse_m",
        blend_grid_step=0.5,
        selection_source="test",
    )

    assert policy["policy_id"] == "blend:learned:RFIS:0.5"
    assert policy["components"] == ["learned", "RFIS"]
    assert policy["evaluated_validation_steps"] == 6
    assert policy["validation_rmse_m"] == 0.0

    row = gp.evaluate_policy_on_record(records[0], policy)
    assert row["best_input_candidate_method"] == "RFIS"
    assert row["portfolio_observed_step_pos_rmse_m"] == 0.0
    assert row["gain_vs_best_input_observed_step_percent"] == 100.0
    assert row["result_vs_best_input"] == "win"


def test_script_has_no_checkpoint_mutating_or_writer_function_calls() -> None:
    source = inspect.getsource(gp)

    assert "update_checkpoint_inference_metadata" not in source
    assert "evaluate_scenario" not in source


def _eval_row(seed: int, scenario: str, gain_percent: float) -> dict[str, object]:
    return {
        "seed": seed,
        "scenario": scenario,
        "gain_vs_best_input_observed_step_percent": gain_percent,
    }


def test_exact_sign_p_value_and_row_gain_statistics_for_eighteen_of_twenty_wins() -> None:
    expected_p = 211 / 1_048_576
    assert gp.exact_one_sided_sign_p_value(18, 2) == expected_p

    rows = [
        _eval_row(idx, "process_noise_shift_test", gain)
        for idx, gain in enumerate([1.0] * 18 + [0.0, -1.0])
    ]

    stats = gp.row_gain_statistics(rows, bootstrap_samples=128, bootstrap_seed=12345)

    assert stats["rows"] == 20
    assert stats["finite_gain_rows"] == 20
    assert stats["wins"] == 18
    assert stats["ties"] == 1
    assert stats["losses"] == 1
    assert stats["nonpositive"] == 2
    assert stats["exact_one_sided_sign_p_value"] == expected_p
    assert stats["bootstrap_samples"] == 128
    assert stats["bootstrap_seed"] == 12345
    assert len(stats["bootstrap_mean_gain_percent_ci95"]) == 2


def test_seed_paired_gain_statistics_groups_rows_and_preserves_negative_seed_mean() -> None:
    rows = [
        _eval_row(7, "process_noise_shift_test", 4.0),
        _eval_row(7, "maneuver_shift_test", 2.0),
        _eval_row(11, "process_noise_shift_test", 1.0),
        _eval_row(11, "maneuver_shift_test", -5.0),
        _eval_row(13, "process_noise_shift_test", 0.0),
        _eval_row(13, "maneuver_shift_test", 2.0),
    ]

    stats = gp.seed_paired_gain_statistics(rows, bootstrap_samples=128, bootstrap_seed=7)

    assert stats["seeds"] == 3
    assert stats["finite_gain_seeds"] == 3
    assert stats["seed_wins"] == 2
    assert stats["seed_ties"] == 0
    assert stats["seed_losses"] == 1
    assert stats["seed_nonpositive"] == 1
    assert stats["seed_mean_gain_percent"] == 2.0 / 3.0
    assert stats["seed_median_gain_percent"] == 1.0
    assert stats["seed_min_gain_percent"] == -2.0
    assert stats["seed_max_gain_percent"] == 3.0
    assert stats["exact_one_sided_sign_p_value"] == 0.5
    assert stats["seed_mean_gains"] == [
        {"seed": 7, "mean_gain_percent": 3.0},
        {"seed": 11, "mean_gain_percent": -2.0},
        {"seed": 13, "mean_gain_percent": 1.0},
    ]
    assert stats["seed_mean_gain_percent_values"] == [3.0, -2.0, 1.0]
