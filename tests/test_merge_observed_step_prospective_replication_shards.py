from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "merge_observed_step_prospective_replication_shards.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "merge_observed_step_prospective_replication_shards", SCRIPT_PATH
)
merge_script = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = merge_script
_SPEC.loader.exec_module(merge_script)


FROZEN_RULE = {
    "rule_type": "larger independent endpoint replication under a frozen observed-step rule",
    "fixed_rule_path": "release/predeclarations/observed_step_prospective_replication_loop71.json",
    "not_external_preregistration": True,
    "frozen_before_evaluation": True,
    "primary_metric": "observed_step_position_rmse_m",
    "reference_metric": "all_step_position_rmse_m",
    "decision_predicate": "fixed predicate",
    "realization_base_seed": 880000,
    "seed_disjointness": "fixed seed statement",
    "scenarios": ["test", "stress_test", "force_model_mismatch_test"],
    "num_realizations_per_scenario": 32,
    "trajectories_per_realization": 24,
    "bootstrap_samples": 5000,
    "statistical_unit": "independent realization",
    "fixed_released_checkpoint": "rgr-gf.pt",
    "inference_only": True,
    "no_selection_tuning_or_retraining": True,
    "interpretation_boundary": "fixed before evaluation",
}

SCENARIO_NAMES = {
    0: ("test", "Nominal"),
    1: ("stress_test", "Measurement-noise stress"),
    2: ("force_model_mismatch_test", "Controlled force-model mismatch"),
}


def _scenario(index: int, best_method: str = "AUKF", learned: bool = False) -> dict:
    name, label = SCENARIO_NAMES.get(index, (f"scenario_{index}", f"Scenario {index}"))
    return {
        "name": name,
        "label": label,
        "regime": "synthetic",
        "scenario_index": index,
        "n_realizations": 32,
        "trajectories_per_realization": 24,
        "best_method_primary": best_method,
        "best_classical_primary": "AUKF",
        "learned_positive_under_frozen_rule": learned,
        "decision_predicate_satisfied": learned,
    }


def _write_shard(
    path: Path,
    index: int,
    *,
    best_method: str = "AUKF",
    learned: bool = False,
    frozen_rule: dict | None = None,
    source: str = "materialized scenario shard",
) -> Path:
    rule = copy.deepcopy(FROZEN_RULE if frozen_rule is None else frozen_rule)
    payload = {
        "status": "completed",
        "schema_version": "observed_step_prospective_replication_v1",
        "artifact_role": "larger_independent_endpoint_replication",
        "frozen_rule": rule,
        "statistical_unit": "independent realization",
        "source": source,
        "num_scenarios": 1,
        "scenarios": [
            _scenario(index, best_method=best_method, learned=learned),
        ],
        "summary": {
            "n_scenarios": 1,
            "num_realizations_per_scenario": rule["num_realizations_per_scenario"],
            "trajectories_per_realization": rule["trajectories_per_realization"],
            "scenarios_with_learned_positive_under_frozen_rule": int(learned),
            "scenarios_with_classical_best_on_primary": int(best_method != "RGR-GF"),
            "verdict": "shard-local verdict",
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_merge_sorts_scenarios_and_recomputes_summary(tmp_path: Path) -> None:
    s2 = _write_shard(tmp_path / "s2.json", 2, best_method="EKF")
    s0 = _write_shard(tmp_path / "s0.json", 0, best_method="AUKF")
    s1 = _write_shard(
        tmp_path / "s1.json",
        1,
        best_method="RGR-GF",
        learned=True,
    )
    out_path = tmp_path / "merged.json"

    merged = merge_script.merge_shards([s2, s0, s1], out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert written == merged
    assert [row["scenario_index"] for row in written["scenarios"]] == [0, 1, 2]
    assert written["num_scenarios"] == 3
    assert written["frozen_rule"] == FROZEN_RULE
    assert written["statistical_unit"] == "independent realization"
    assert written["source"] == "materialized scenario shard"
    assert written["summary"] == {
        "n_scenarios": 3,
        "num_realizations_per_scenario": 32,
        "trajectories_per_realization": 24,
        "scenarios_with_learned_positive_under_frozen_rule": 1,
        "scenarios_with_classical_best_on_primary": 2,
        "verdict": "learned positive observed under the frozen decision predicate",
    }


def test_rejects_duplicate_scenario_index_without_writing(tmp_path: Path) -> None:
    s0 = _write_shard(tmp_path / "s0.json", 0)
    s0_dup = _write_shard(tmp_path / "s0_dup.json", 0)
    out_path = tmp_path / "merged.json"
    out_path.write_text("do not replace", encoding="utf-8")

    with pytest.raises(merge_script.ValidationError, match="duplicate scenario_index 0"):
        merge_script.merge_shards([s0, s0_dup], out_path)

    assert out_path.read_text(encoding="utf-8") == "do not replace"


def test_rejects_mismatched_top_level_metadata(tmp_path: Path) -> None:
    s0 = _write_shard(tmp_path / "s0.json", 0)
    s1 = _write_shard(tmp_path / "s1.json", 1, source="different source")

    with pytest.raises(
        merge_script.ValidationError,
        match="top-level metadata key 'source' does not match",
    ):
        merge_script.build_merged_artifact([s0, s1], tmp_path / "merged.json")


def test_canonical_output_requires_complete_frozen_rule(tmp_path: Path) -> None:
    bad_rule = copy.deepcopy(FROZEN_RULE)
    bad_rule["bootstrap_samples"] = 100
    shards = [
        _write_shard(tmp_path / "s0.json", 0, frozen_rule=bad_rule),
        _write_shard(tmp_path / "s1.json", 1, frozen_rule=bad_rule),
        _write_shard(tmp_path / "s2.json", 2, frozen_rule=bad_rule),
    ]

    with pytest.raises(
        merge_script.ValidationError,
        match=r"canonical output requires frozen_rule\.bootstrap_samples == 5000",
    ):
        merge_script.build_merged_artifact(shards, merge_script.CANONICAL_OUT_PATH)


def test_canonical_output_requires_all_three_scenarios(tmp_path: Path) -> None:
    shards = [
        _write_shard(tmp_path / "s0.json", 0),
        _write_shard(tmp_path / "s1.json", 1),
    ]

    with pytest.raises(
        merge_script.ValidationError,
        match=r"canonical output requires exactly scenario_index values \[0, 1, 2\]",
    ):
        merge_script.build_merged_artifact(shards, merge_script.CANONICAL_OUT_PATH)
