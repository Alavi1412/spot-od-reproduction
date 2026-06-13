"""Cheap checks for the larger observed-step endpoint replication artifact."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_JSON = (
    REPO_ROOT
    / "results"
    / "observed_step_prospective_replication"
    / "observed_step_prospective_replication.json"
)
RULE_JSON = (
    REPO_ROOT
    / "release"
    / "predeclarations"
    / "observed_step_prospective_replication_loop71.json"
)
MANIFEST_SCRIPT = REPO_ROOT / "scripts" / "build_supplementary_manifest.py"
PAPER_ASSET_SCRIPT = REPO_ROOT / "scripts" / "build_paper_assets.py"

EXPECTED_MANIFEST_REPLICATION_ARTIFACTS = [
    "results/observed_step_prospective_replication/observed_step_prospective_replication.json",
    "paper/tables/observed_step_prospective_replication.tex",
    "release/predeclarations/observed_step_prospective_replication_loop71.json",
]

_SPEC = importlib.util.spec_from_file_location(
    "build_paper_assets", REPO_ROOT / "scripts" / "build_paper_assets.py"
)
assets = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = assets
_SPEC.loader.exec_module(assets)

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "build_observed_step_prospective_replication",
    REPO_ROOT / "scripts" / "build_observed_step_prospective_replication.py",
)
replication_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
assert _SCRIPT_SPEC and _SCRIPT_SPEC.loader
sys.modules[_SCRIPT_SPEC.name] = replication_script
_SCRIPT_SPEC.loader.exec_module(replication_script)

SCENARIOS = ["test", "stress_test", "force_model_mismatch_test"]
METHODS = ["EKF", "UKF", "AUKF", "RGR-GF"]


def _sample_artifact(path: Path) -> Path:
    row_template = {
        "n_realizations": 32,
        "trajectories_per_realization": 24,
        "observed_step_pos_rmse_m": {
            "EKF": 410.0,
            "UKF": 430.0,
            "AUKF": 405.0,
            "RGR-GF": 420.0,
        },
        "all_step_pos_rmse_m": {
            "EKF": 11000.0,
            "UKF": 11100.0,
            "AUKF": 10950.0,
            "RGR-GF": 11020.0,
        },
        "best_method_primary": "AUKF",
        "best_classical_primary": "AUKF",
        "rgr_gf_minus_best_classical_primary_mean_m": 15.0,
        "rgr_gf_minus_best_classical_primary_ci_low_m": 5.0,
        "rgr_gf_minus_best_classical_primary_ci_high_m": 25.0,
        "learned_positive_under_frozen_rule": False,
    }
    payload = {
        "status": "completed",
        "schema_version": "observed_step_prospective_replication_v1",
        "frozen_rule": {
            "primary_metric": "observed_step_position_rmse_m",
            "reference_metric": "all_step_position_rmse_m",
            "num_realizations_per_scenario": 32,
            "trajectories_per_realization": 24,
            "frozen_before_evaluation": True,
            "not_external_preregistration": True,
        },
        "scenarios": [
            {
                **row_template,
                "name": name,
                "label": label,
            }
            for name, label in [
                ("test", "Nominal"),
                ("stress_test", "Measurement-noise stress"),
                ("force_model_mismatch_test", "Controlled force-model mismatch"),
            ]
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_rule_file_records_design_before_evaluation() -> None:
    rule = json.loads(RULE_JSON.read_text(encoding="utf-8"))

    assert rule["schema_version"] == "observed_step_prospective_replication_rule_v1"
    assert rule["frozen_before_evaluation"] is True
    assert rule["not_external_preregistration"] is True
    assert rule["base_seed"] == 880000
    assert rule["num_realizations_per_scenario"] == 32
    assert rule["trajectories_per_realization"] == 24
    assert rule["bootstrap_samples"] == 5000
    assert [s["name"] for s in rule["scenarios"]] == SCENARIOS
    assert rule["primary_metric"] == "observed_step_position_rmse_m"
    assert rule["reference_metric"] == "all_step_position_rmse_m"
    assert rule["fixed_released_checkpoint_inference_only"] is True
    assert rule["no_model_selection"] is True
    assert rule["no_tuning"] is True
    assert rule["no_retraining"] is True
    assert "not an external preregistration" in rule["interpretation_boundary"]
    assert "strictly below zero" in rule["decision_predicate"]["learned_positive"]


def test_builder_defaults_match_frozen_rule() -> None:
    rule = json.loads(RULE_JSON.read_text(encoding="utf-8"))

    assert replication_script.BASE_SEED == rule["base_seed"]
    assert (
        replication_script.DEFAULT_REALIZATIONS_PER_SCENARIO
        == rule["num_realizations_per_scenario"]
    )
    assert (
        replication_script.DEFAULT_TRAJECTORIES_PER_REALIZATION
        == rule["trajectories_per_realization"]
    )
    assert replication_script.DEFAULT_BOOTSTRAP_SAMPLES == rule["bootstrap_samples"]
    assert [s[0] for s in replication_script.SCENARIOS] == SCENARIOS


def test_parallel_filter_helper_preserves_serial_keys_and_shapes(monkeypatch) -> None:
    states = np.arange(3 * 4 * 6, dtype=np.float64).reshape(3, 4, 6)
    measurements = np.zeros((3, 4, 2, 4), dtype=np.float64)
    visibility = np.ones((3, 4, 2), dtype=np.float64)
    times = np.tile(np.arange(4, dtype=np.float64), (3, 1))
    x0 = np.arange(3 * 6, dtype=np.float64).reshape(3, 6)
    parallel_x0_batches = []

    def fake_run_filter_baselines(
        *,
        states,
        measurements,
        visibility,
        times,
        dataset_cfg,
        baseline_cfg,
        seed,
        x0_estimates,
    ):
        assert x0_estimates is not None
        if states.shape[0] == 1:
            parallel_x0_batches.append(x0_estimates.copy())
        base = states + x0_estimates[:, np.newaxis, :]
        return {
            "ekf": base + 10.0,
            "ukf": base + 20.0,
            "aukf": base + 30.0,
        }

    class InlineExecutor:
        def map(self, fn, tasks, chunksize=1):
            return [fn(task) for task in tasks]

    monkeypatch.setattr(
        replication_script,
        "run_filter_baselines",
        fake_run_filter_baselines,
    )

    serial = replication_script._run_filter_baselines_optional_parallel(
        states=states,
        measurements=measurements,
        visibility=visibility,
        times=times,
        dataset_cfg=object(),
        baseline_cfg=object(),
        seed=123,
        x0_estimates=x0,
        filter_workers=1,
    )
    parallel = replication_script._run_filter_baselines_optional_parallel(
        states=states,
        measurements=measurements,
        visibility=visibility,
        times=times,
        dataset_cfg=object(),
        baseline_cfg=object(),
        seed=123,
        x0_estimates=x0,
        filter_workers=2,
        executor=InlineExecutor(),
    )

    assert set(parallel) == {"ekf", "ukf", "aukf"}
    for key in ("ekf", "ukf", "aukf"):
        assert parallel[key].shape == states.shape
        np.testing.assert_allclose(parallel[key], serial[key])
    assert len(parallel_x0_batches) == states.shape[0]
    for idx, batch in enumerate(parallel_x0_batches):
        np.testing.assert_allclose(batch[0], x0[idx])


def test_parallel_filter_helper_requires_precomputed_x0() -> None:
    states = np.zeros((2, 3, 6), dtype=np.float64)
    measurements = np.zeros((2, 3, 1, 4), dtype=np.float64)
    visibility = np.ones((2, 3, 1), dtype=np.float64)
    times = np.tile(np.arange(3, dtype=np.float64), (2, 1))

    with pytest.raises(ValueError, match="x0_estimates"):
        replication_script._run_filter_baselines_optional_parallel(
            states=states,
            measurements=measurements,
            visibility=visibility,
            times=times,
            dataset_cfg=object(),
            baseline_cfg=object(),
            seed=123,
            x0_estimates=None,
            filter_workers=2,
        )


def test_result_schema_and_decision_invariants_if_materialized() -> None:
    if not RESULT_JSON.exists():
        pytest.skip("observed-step prospective replication artifact not materialized")

    data = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    rule = json.loads(RULE_JSON.read_text(encoding="utf-8"))
    frozen = data["frozen_rule"]

    assert data["schema_version"] == "observed_step_prospective_replication_v1"
    assert frozen["primary_metric"] == rule["primary_metric"]
    assert frozen["reference_metric"] == rule["reference_metric"]
    assert frozen["realization_base_seed"] == rule["base_seed"]
    assert frozen["num_realizations_per_scenario"] == rule["num_realizations_per_scenario"]
    assert frozen["trajectories_per_realization"] == rule["trajectories_per_realization"]
    assert frozen["frozen_before_evaluation"] is True
    assert frozen["not_external_preregistration"] is True
    assert frozen["no_selection_tuning_or_retraining"] is True
    assert frozen["inference_only"] is True
    assert [row["name"] for row in data["scenarios"]] == SCENARIOS

    for row in data["scenarios"]:
        assert row["n_realizations"] == rule["num_realizations_per_scenario"]
        assert row["trajectories_per_realization"] == rule["trajectories_per_realization"]
        assert set(row["observed_step_pos_rmse_m"]) == set(METHODS)
        assert set(row["all_step_pos_rmse_m"]) == set(METHODS)
        for method in METHODS:
            assert len(row["per_realization_observed_step_m"][method]) == row[
                "n_realizations"
            ]
            assert len(row["per_realization_reference_all_step_m"][method]) == row[
                "n_realizations"
            ]
        if row["learned_positive_under_frozen_rule"]:
            assert row["best_method_primary"] == "RGR-GF"
            assert row["rgr_gf_minus_best_classical_primary_ci_high_m"] < 0.0


def test_table_rendering_is_paper_safe(tmp_path: Path) -> None:
    artifact = _sample_artifact(tmp_path / "replication.json")
    tex = assets.build_observed_step_prospective_replication_table(artifact)

    assert "tab:observed_step_prospective_replication" in tex
    assert "Larger independent endpoint replication under the frozen observed-step rule" in tex
    assert "simulator-bound" in tex
    assert "not external preregistration" in tex
    assert "operational validation" in tex
    assert "$K{=}32$" in tex
    assert "Best classical" in tex
    assert "RGR-GF$-$best cl." in tex
    lower = tex.lower()
    for forbidden in [
        "script",
        "file path",
        "local path",
        "virtual env",
        "loop",
        "gpu",
        "cuda",
        "gpt",
        "claude",
        "gemini",
        "local " + 'environment',
    ]:
        assert forbidden not in lower


def test_asset_builder_dispatches_materialized_table() -> None:
    text = PAPER_ASSET_SCRIPT.read_text(encoding="utf-8")

    assert "Path(\"paper/tables/observed_step_prospective_replication.tex\")" in text
    assert "build_observed_step_prospective_replication_table()" in text


def test_manifest_source_indexes_materialized_replication_artifacts() -> None:
    text = MANIFEST_SCRIPT.read_text(encoding="utf-8")

    assert "larger_observed_step_endpoint_replication" in text
    assert "larger_simulator_bound_endpoint_replication" in text
    assert "not external" in text
    assert "operational validation" in text
    for rel in EXPECTED_MANIFEST_REPLICATION_ARTIFACTS:
        assert rel in text


def test_materialized_table_renders_if_artifact_exists() -> None:
    if not RESULT_JSON.exists():
        pytest.skip("observed-step prospective replication artifact not materialized")

    tex = assets.build_observed_step_prospective_replication_table(RESULT_JSON)
    assert "tab:observed_step_prospective_replication" in tex
    assert "observed-step rule" in tex
