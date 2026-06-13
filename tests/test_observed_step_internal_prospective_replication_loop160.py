"""Tests for the loop-160 internal prospective observed-step replication.

Cheap checks covering:
  - freeze record schema and self-hash canonicalization
  - seed disjointness from all prior bases
  - CSV export contract using a small synthetic artifact
  - table rendering claim boundaries (no forbidden terms, correct label)
  - build_paper_assets.py dispatch (function and output path present)
  - merge-shards: happy-path with differing frozen_rule.scenarios/rule_type,
    validation failures (missing scenario, bad learned_positive type, etc.)

No estimators are run. No model is loaded.
"""
from __future__ import annotations

import csv
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

_SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "build_observed_step_internal_prospective_replication_loop160.py"
)
_BPA_PATH = REPO_ROOT / "scripts" / "build_paper_assets.py"

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "build_observed_step_internal_prospective_replication_loop160",
    _SCRIPT_PATH,
)
assert _SCRIPT_SPEC and _SCRIPT_SPEC.loader
loop160_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
sys.modules[_SCRIPT_SPEC.name] = loop160_script
_SCRIPT_SPEC.loader.exec_module(loop160_script)

_BPA_SPEC = importlib.util.spec_from_file_location("build_paper_assets", _BPA_PATH)
assert _BPA_SPEC and _BPA_SPEC.loader
bpa = importlib.util.module_from_spec(_BPA_SPEC)
sys.modules[_BPA_SPEC.name] = bpa
_BPA_SPEC.loader.exec_module(bpa)

# ---------------------------------------------------------------------------
# Constants mirrored from the script (for isolation)
# ---------------------------------------------------------------------------

EXPECTED_BASE_SEED = 1160000
EXPECTED_NUM_REALIZATIONS = 32
EXPECTED_TRAJECTORIES = 24
EXPECTED_BOOTSTRAP_SAMPLES = 5000
EXPECTED_SCHEMA_VERSION = (
    "observed_step_internal_prospective_replication_loop160_rule_v1"
)
PRIOR_BASES = [
    # train/val cohort
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    # existing replication bases
    770000, 880000, 991117, 90000,
]
METHODS = ["EKF", "UKF", "AUKF", "RGR-GF"]
SCENARIOS = ["test", "stress_test", "force_model_mismatch_test"]

ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "observed_step_internal_prospective_replication_loop160"
    / "observed_step_internal_prospective_replication_loop160.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_artifact(
    path: Path,
    n_realizations: int = 4,
    n_traj: int = 6,
) -> Path:
    """Write a minimal synthetic artifact with the merge-schema format."""
    row_template: dict = {
        "scenario_index": 0,
        "n_realizations": n_realizations,
        "trajectories_per_realization": n_traj,
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
        "primary_observed_step_pos_rmse_m": {
            "EKF": 410.0,
            "UKF": 430.0,
            "AUKF": 405.0,
            "RGR-GF": 420.0,
        },
        "reference_all_step_pos_rmse_m": {
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
        "per_realization_observed_step_m": {
            m: [float(i + 400) for i in range(n_realizations)] for m in METHODS
        },
        "per_realization_reference_all_step_m": {
            m: [float(i + 10000) for i in range(n_realizations)] for m in METHODS
        },
    }
    scenarios_out = []
    for idx, (name, label) in enumerate(
        [
            ("test", "Nominal"),
            ("stress_test", "Measurement-noise stress"),
            ("force_model_mismatch_test", "Controlled force-model mismatch"),
        ]
    ):
        s = dict(row_template)
        s["scenario_index"] = idx
        s["name"] = name
        s["label"] = label
        scenarios_out.append(s)
    payload = {
        "status": "completed",
        "schema_version": "observed_step_prospective_replication_v1",
        "artifact_role": "additional_internal_prospective_replication_loop160",
        "frozen_rule": {
            "primary_metric": "observed_step_position_rmse_m",
            "reference_metric": "all_step_position_rmse_m",
            "num_realizations_per_scenario": n_realizations,
            "trajectories_per_realization": n_traj,
            "frozen_before_evaluation": True,
            "not_external_preregistration": True,
            "realization_base_seed": EXPECTED_BASE_SEED,
        },
        "num_scenarios": len(scenarios_out),
        "scenarios": scenarios_out,
        "summary": {
            "n_scenarios": len(scenarios_out),
            "scenarios_with_learned_positive_under_frozen_rule": 0,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests: script constants
# ---------------------------------------------------------------------------


def test_base_seed_value() -> None:
    """The canonical base seed must be exactly 1160000."""
    assert loop160_script.BASE_SEED == EXPECTED_BASE_SEED


def test_num_realizations_value() -> None:
    assert loop160_script.NUM_REALIZATIONS == EXPECTED_NUM_REALIZATIONS


def test_trajectories_value() -> None:
    assert loop160_script.TRAJECTORIES == EXPECTED_TRAJECTORIES


def test_bootstrap_samples_value() -> None:
    assert loop160_script.BOOTSTRAP_SAMPLES == EXPECTED_BOOTSTRAP_SAMPLES


def test_schema_version_value() -> None:
    assert loop160_script.SCHEMA_VERSION == EXPECTED_SCHEMA_VERSION


def test_scenario_names() -> None:
    names = [s["name"] for s in loop160_script.SCENARIOS]
    assert names == SCENARIOS


# ---------------------------------------------------------------------------
# Tests: seed disjointness
# ---------------------------------------------------------------------------


def test_base_seed_disjoint_from_all_prior_bases() -> None:
    """Base seed 1160000 must be far from all previously used seed bases."""
    seed = loop160_script.BASE_SEED
    # Not in the train/val cohort range.
    assert seed not in range(41, 56), "conflicts with train/val cohort"
    # Not one of the named prior bases.
    for prior in PRIOR_BASES:
        assert seed != prior, f"conflicts with prior base {prior}"
    # Not in the known shard seed range used in prior extension runs.
    assert seed not in range(770539, 770567), "conflicts with prior shard seeds"


def test_prior_seed_bases_dict_coverage() -> None:
    """The script's _PRIOR_SEED_BASES dict covers the expected prior seeds."""
    d = loop160_script._PRIOR_SEED_BASES
    assert "endpoint_fixation_k8_k16" in d
    assert d["endpoint_fixation_k8_k16"] == 770000
    assert d["k32_central_replication"] == 880000
    assert d["k96_stress_replication"] == 991117
    assert d["scenario_resampling"] == 90000


# ---------------------------------------------------------------------------
# Tests: freeze record schema and self-hash
# ---------------------------------------------------------------------------


def test_freeze_record_schema(tmp_path: Path) -> None:
    """Written freeze record must contain all required top-level fields."""
    out = tmp_path / "preregistration.json"
    loop160_script.write_freeze_record(out_path=out)
    assert out.exists(), "freeze record was not written"
    record = json.loads(out.read_text(encoding="utf-8"))

    required_fields = [
        "schema_version",
        "created_at_utc",
        "evidence_scope",
        "not_external_preregistration",
        "frozen_before_evaluation",
        "endpoint_hierarchy",
        "decision_rule",
        "seed_plan",
        "scenarios",
        "num_realizations_per_scenario",
        "trajectories_per_realization",
        "bootstrap_samples",
        "practical_floor_rule",
        "model_artifact_reference",
        "source_input_hashes",
        "interpretation_boundary",
        "canonical_rule_sha256",
    ]
    for field in required_fields:
        assert field in record, f"missing required field: {field!r}"


def test_freeze_record_values(tmp_path: Path) -> None:
    """Freeze record must encode the correct parameter values."""
    out = tmp_path / "preregistration.json"
    loop160_script.write_freeze_record(out_path=out)
    record = json.loads(out.read_text(encoding="utf-8"))

    assert record["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert record["not_external_preregistration"] is True
    assert record["frozen_before_evaluation"] is True
    assert record["num_realizations_per_scenario"] == EXPECTED_NUM_REALIZATIONS
    assert record["trajectories_per_realization"] == EXPECTED_TRAJECTORIES
    assert record["bootstrap_samples"] == EXPECTED_BOOTSTRAP_SAMPLES
    assert record["seed_plan"]["base_seed"] == EXPECTED_BASE_SEED

    ep = record["endpoint_hierarchy"]
    assert ep["primary_metric"] == "observed_step_position_rmse_m"
    assert ep["reference_metric"] == "all_step_position_rmse_m"

    dr = record["decision_rule"]
    assert "strictly below zero" in dr["decision_predicate"]
    assert "external preregistration" in record["interpretation_boundary"].lower()


def test_freeze_record_self_hash_valid(tmp_path: Path) -> None:
    """The canonical_rule_sha256 field must match a re-computation."""
    out = tmp_path / "preregistration.json"
    loop160_script.write_freeze_record(out_path=out)
    record = json.loads(out.read_text(encoding="utf-8"))
    stored = record.pop("canonical_rule_sha256")
    # Re-compute over record without the hash field.
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert stored == expected, "canonical_rule_sha256 does not match re-computation"


def test_validate_freeze_round_trip(tmp_path: Path) -> None:
    """validate_freeze_record must return True for a freshly written record."""
    out = tmp_path / "preregistration.json"
    loop160_script.write_freeze_record(out_path=out)
    ok = loop160_script.validate_freeze_record(path=out)
    assert ok is True


def test_validate_freeze_detects_tampering(tmp_path: Path) -> None:
    """validate_freeze_record must return False if the record is modified."""
    out = tmp_path / "preregistration.json"
    loop160_script.write_freeze_record(out_path=out)
    record = json.loads(out.read_text(encoding="utf-8"))
    record["num_realizations_per_scenario"] = 99  # tamper
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    ok = loop160_script.validate_freeze_record(path=out)
    assert ok is False


def test_source_input_hashes_keys(tmp_path: Path) -> None:
    """source_input_hashes must reference the three canonical input files."""
    out = tmp_path / "preregistration.json"
    loop160_script.write_freeze_record(out_path=out)
    record = json.loads(out.read_text(encoding="utf-8"))
    hashes = record["source_input_hashes"]
    assert "configs/experiment.yaml" in hashes
    assert "scripts/build_observed_step_prospective_replication.py" in hashes
    k32_key = (
        "release/predeclarations/observed_step_prospective_replication_loop71.json"
    )
    assert k32_key in hashes


# ---------------------------------------------------------------------------
# Tests: CSV export contract
# ---------------------------------------------------------------------------


def test_export_csv_scenario_summary(tmp_path: Path) -> None:
    """export-csv writes scenario_summary.csv with correct structure."""
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop160.json",
        n_realizations=4,
    )
    loop160_script.export_csv(artifact_path=artifact)
    summary_path = tmp_path / "scenario_summary.csv"
    assert summary_path.exists(), "scenario_summary.csv not written"
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    assert len(rows) == 3, f"expected 3 scenario rows, got {len(rows)}"
    for row in rows:
        assert row["scenario_name"] in SCENARIOS
        assert float(row["n_realizations"]) == 4
        assert row["best_classical_primary"] != ""
        assert row["learned_positive_under_frozen_rule"] != ""


def test_export_csv_per_realization(tmp_path: Path) -> None:
    """export-csv writes per_realization_observed_step.csv with correct row count."""
    n_real = 4
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop160.json",
        n_realizations=n_real,
    )
    loop160_script.export_csv(artifact_path=artifact)
    per_real_path = tmp_path / "per_realization_observed_step.csv"
    assert per_real_path.exists(), "per_realization_observed_step.csv not written"
    rows = list(csv.DictReader(per_real_path.open(encoding="utf-8")))
    # 3 scenarios * 4 realizations * 4 methods = 48 rows
    expected_rows = 3 * n_real * len(METHODS)
    assert len(rows) == expected_rows, (
        f"expected {expected_rows} rows, got {len(rows)}"
    )
    for row in rows:
        assert row["scenario_name"] in SCENARIOS
        assert row["method"] in METHODS
        assert int(row["realization_index"]) < n_real


def test_export_csv_requires_existing_artifact(tmp_path: Path) -> None:
    """export-csv must exit with error when artifact is missing."""
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(SystemExit) as exc_info:
        loop160_script.export_csv(artifact_path=missing)
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Tests: table rendering claim boundaries
# ---------------------------------------------------------------------------


def test_table_renders_with_synthetic_artifact(tmp_path: Path) -> None:
    """Table must render successfully from a synthetic artifact."""
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop160.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k32_table(
        artifact
    )
    assert "\\begin{table}" in tex
    assert "tab:observed_step_internal_prospective_replication" in tex


def test_table_paired_gap_two_decimal_formatting(tmp_path: Path) -> None:
    """Gap/CI column must use two-decimal formatting (e.g. 15.00 [5.00, 25.00]).

    The synthetic artifact uses mean=15.0, ci_low=5.0, ci_high=25.0.
    Two-decimal format renders as '15.00 [5.00, 25.00]'.
    One-decimal format would render as '15.0 [5.0, 25.0]' and must NOT appear.
    """
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop160.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k32_table(
        artifact
    )
    # Two-decimal values must be present.
    assert "15.00 [5.00, 25.00]" in tex, (
        "expected two-decimal gap/CI '15.00 [5.00, 25.00]' not found in table"
    )
    # One-decimal versions must NOT appear.
    assert "15.0 [5.0, 25.0]" not in tex, (
        "one-decimal gap/CI '15.0 [5.0, 25.0]' found; formatter not updated"
    )


def test_table_contains_required_claims(tmp_path: Path) -> None:
    """Table caption must contain required scope limitation statements."""
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop160.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k32_table(
        artifact
    )
    lower = tex.lower()
    assert "not external preregistration" in lower
    assert "internal" in lower
    assert "operational validation" in lower
    # Must mention independently seeded or similar framing.
    assert "independently seeded" in lower or "independently" in lower


def test_table_no_forbidden_terms(tmp_path: Path) -> None:
    """Table must not contain forbidden paper-facing terms."""
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop160.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k32_table(
        artifact
    )
    lower = tex.lower()
    for term in [
        "git" + "hub",
        "zen" + "odo",
        "pub" + "lic doi",
        "pub" + "lic repository",
        "ven" + "v",
        "virt" + "ual env",
        "clau" + "de",
        "cod" + "ex",
        "sub" + "agent",
        "oll" + "ama",
        "open" + "code",
        "tool" + "chain",
        "post" + "-review",
    ]:
        assert term not in lower, f"forbidden term {term!r} found in table"


def test_table_footnote_no_double_closing_brace(tmp_path: Path) -> None:
    """The footnote line must close the footnotesize group with exactly one brace.

    A double closing brace (}}) is a LaTeX syntax error because the f-string
    already opens the group with a single { (from the {{ escape) and so only
    one } is needed.  This test guards against the regression where the last
    string literal in the footnote concatenation accidentally carries }}.
    """
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop160.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k32_table(
        artifact
    )
    # Split into lines and find the footnote line.
    footnote_lines = [ln for ln in tex.splitlines() if "footnotesize" in ln]
    assert footnote_lines, "no footnotesize line found in generated table"
    for ln in footnote_lines:
        assert not ln.rstrip().endswith("}}"), (
            f"footnote line ends with double closing brace (LaTeX error): {ln!r}"
        )


def test_table_unavailable_stub(tmp_path: Path) -> None:
    """When artifact is missing the builder must return a comment stub."""
    missing = tmp_path / "does_not_exist.json"
    result = bpa.build_observed_step_internal_prospective_replication_k32_table(
        missing
    )
    assert result.startswith("%")


# ---------------------------------------------------------------------------
# Tests: decision-invariant check if artifact is materialized
# ---------------------------------------------------------------------------


def test_decision_invariant_if_materialized() -> None:
    """If the merged artifact exists, check that decision flags are consistent."""
    if not ARTIFACT_PATH.exists():
        pytest.skip("loop-160 artifact not yet materialized")
    data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    rows = data.get("scenarios", [])
    assert len(rows) == 3, "expected three scenario rows"
    for row in rows:
        lp = row.get("learned_positive_under_frozen_rule", None)
        assert isinstance(lp, bool)
        if lp:
            assert row["best_method_primary"] == "RGR-GF"
            assert row["rgr_gf_minus_best_classical_primary_ci_high_m"] < 0.0


# ---------------------------------------------------------------------------
# Tests: build_paper_assets.py dispatch
# ---------------------------------------------------------------------------


def test_bpa_dispatches_loop160_table() -> None:
    """build_paper_assets.py must dispatch the K32 table to the correct path."""
    text = _BPA_PATH.read_text(encoding="utf-8")
    assert (
        'Path("paper/tables/observed_step_internal_prospective_replication_k32.tex")'
        in text
    )
    assert "build_observed_step_internal_prospective_replication_k32_table()" in text


def test_bpa_function_defined() -> None:
    """build_paper_assets module must expose the K32 table builder."""
    assert hasattr(bpa, "build_observed_step_internal_prospective_replication_k32_table")
    assert callable(bpa.build_observed_step_internal_prospective_replication_k32_table)


# ---------------------------------------------------------------------------
# Tests: print-shard-commands output contract
# ---------------------------------------------------------------------------


def test_print_shard_commands_output(capsys: pytest.CaptureFixture) -> None:
    """print_shard_commands must emit all three scenario shard invocations."""
    loop160_script.print_shard_commands(args_device="auto")
    captured = capsys.readouterr().out
    # Must mention all three scenario indices.
    for idx in range(3):
        assert f"--only-scenario-index {idx}" in captured
    # Must reference base seed.
    assert str(EXPECTED_BASE_SEED) in captured
    # Must mention freeze record path pattern.
    assert "preregistration.json" in captured
    # Must reference the loop160-specific merge-shards subcommand, NOT the
    # generic merge utility.
    assert "merge-shards" in captured
    assert "build_observed_step_internal_prospective_replication_loop160" in captured
    # Must NOT reference the generic shard-merge script as the merge command.
    assert "merge_observed_step_prospective_replication_shards" not in captured
    # Must not contain forbidden terms.
    lower = captured.lower()
    for term in [
        "git" + "hub",
        "zen" + "odo",
        "ven" + "v",
        "virt" + "ual env",
        "clau" + "de",
        "cod" + "ex",
    ]:
        assert term not in lower, f"forbidden term {term!r} in shard-command output"


# ---------------------------------------------------------------------------
# Helpers for merge-shards tests
# ---------------------------------------------------------------------------

_INVARIANT_FROZEN_RULE: dict = {
    "not_external_preregistration": True,
    "frozen_before_evaluation": True,
    "primary_metric": "observed_step_position_rmse_m",
    "reference_metric": "all_step_position_rmse_m",
    "decision_predicate": (
        "For each scenario, a learned positive requires the fixed RGR-GF "
        "estimator to have the lowest mean observed-step position RMSE and the "
        "95% percentile bootstrap CI for the paired RGR-GF-minus-best-classical "
        "observed-step gap to be strictly below zero."
    ),
    "realization_base_seed": EXPECTED_BASE_SEED,
    "seed_disjointness": "test disjointness statement",
    "num_realizations_per_scenario": EXPECTED_NUM_REALIZATIONS,
    "trajectories_per_realization": EXPECTED_TRAJECTORIES,
    "bootstrap_samples": EXPECTED_BOOTSTRAP_SAMPLES,
    "statistical_unit": "independent realization",
    "inference_only": True,
    "no_selection_tuning_or_retraining": True,
    "interpretation_boundary": "test interpretation; not external preregistration",
    "fixed_released_estimator": "fixed previously trained RGR-GF estimator",
}


def _make_shard(
    path: Path,
    scenario_index: int,
    scenario_name: str,
    rule_type: str = "independent endpoint replication under a frozen observed-step rule",
    learned_positive: bool = False,
    best_method: str = "EKF",
    ci_high: float = 25.0,
    ci_low: float = 5.0,
    mean_gap: float = 15.0,
    **frozen_rule_overrides: object,
) -> Path:
    """Write a minimal synthetic loop-160 shard file for merge tests."""
    scenario_names_list = [scenario_name]
    frozen_rule: dict = {
        "rule_type": rule_type,
        "fixed_rule_path": (
            "results/observed_step_internal_prospective_replication_loop160/"
            "preregistration.json"
        ),
        **_INVARIANT_FROZEN_RULE,
        "scenarios": scenario_names_list,
        **frozen_rule_overrides,
    }
    scenario: dict = {
        "name": scenario_name,
        "label": scenario_name,
        "scenario_index": scenario_index,
        "n_realizations": EXPECTED_NUM_REALIZATIONS,
        "trajectories_per_realization": EXPECTED_TRAJECTORIES,
        "observed_step_pos_rmse_m": {
            "EKF": 400.0, "UKF": 420.0, "AUKF": 395.0, "RGR-GF": 410.0,
        },
        "primary_observed_step_pos_rmse_m": {
            "EKF": 400.0, "UKF": 420.0, "AUKF": 395.0, "RGR-GF": 410.0,
        },
        "all_step_pos_rmse_m": {
            "EKF": 11000.0, "UKF": 11100.0, "AUKF": 10950.0, "RGR-GF": 11020.0,
        },
        "reference_all_step_pos_rmse_m": {
            "EKF": 11000.0, "UKF": 11100.0, "AUKF": 10950.0, "RGR-GF": 11020.0,
        },
        "best_method_primary": best_method,
        "best_classical_primary": best_method if best_method != "RGR-GF" else "EKF",
        "rgr_gf_minus_best_classical_primary_mean_m": mean_gap,
        "rgr_gf_minus_best_classical_primary_ci_low_m": ci_low,
        "rgr_gf_minus_best_classical_primary_ci_high_m": ci_high,
        "learned_positive_under_frozen_rule": learned_positive,
        "decision_predicate_satisfied": learned_positive,
        "per_realization_observed_step_m": {
            m: [400.0] * EXPECTED_NUM_REALIZATIONS for m in METHODS
        },
        "per_realization_reference_all_step_m": {
            m: [11000.0] * EXPECTED_NUM_REALIZATIONS for m in METHODS
        },
    }
    payload: dict = {
        "status": "completed",
        "schema_version": "observed_step_prospective_replication_v1",
        "artifact_role": "additional_internal_prospective_replication_loop160",
        "frozen_rule": frozen_rule,
        "statistical_unit": "test",
        "source": "test",
        "num_scenarios": 1,
        "scenarios": [scenario],
        "summary": {
            "n_scenarios": 1,
            "scenarios_with_learned_positive_under_frozen_rule": 0,
            "verdict": "test",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _make_freeze_record(tmp_path: Path) -> Path:
    """Write a minimal synthetic preregistration.json for merge tests.

    The canonical_rule_sha256 self-hash is computed correctly so that
    validate_freeze_record() accepts the record.
    """
    record: dict = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "not_external_preregistration": True,
        "frozen_before_evaluation": True,
        "endpoint_hierarchy": {
            "primary_metric": "observed_step_position_rmse_m",
            "reference_metric": "all_step_position_rmse_m",
        },
        "decision_rule": {
            "decision_predicate": (
                "For each scenario, a learned positive requires the fixed "
                "RGR-GF estimator to have the lowest mean observed-step position "
                "RMSE and the 95% percentile bootstrap CI for the paired "
                "RGR-GF-minus-best-classical observed-step gap to be strictly "
                "below zero. The all-step position RMSE is reported only as a "
                "reference metric."
            ),
        },
        "seed_plan": {
            "base_seed": EXPECTED_BASE_SEED,
            "seed_disjointness": "test disjointness",
            "statistical_unit": "independent realization",
        },
        "scenarios": [
            {"name": "test", "label": "Nominal", "regime": "nominal"},
            {"name": "stress_test", "label": "Stress", "regime": "stress"},
            {"name": "force_model_mismatch_test", "label": "Mismatch", "regime": "mismatch"},
        ],
        "num_realizations_per_scenario": EXPECTED_NUM_REALIZATIONS,
        "trajectories_per_realization": EXPECTED_TRAJECTORIES,
        "bootstrap_samples": EXPECTED_BOOTSTRAP_SAMPLES,
        "model_artifact_reference": {
            "role": "fixed, previously trained RGR-GF estimator evaluated without any per-realization refitting",
        },
        "interpretation_boundary": "test interpretation boundary; not external preregistration",
    }
    # Compute correct self-hash (same algorithm as write_freeze_record).
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    self_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    record["canonical_rule_sha256"] = self_hash
    out = tmp_path / "preregistration.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Tests: merge-shards
# ---------------------------------------------------------------------------


def test_merge_shards_happy_path_differing_rule_type_and_scenarios(
    tmp_path: Path,
) -> None:
    """merge_shards_loop160 must succeed even when shards have different
    frozen_rule.rule_type and per-shard frozen_rule.scenarios lists."""
    s0 = _make_shard(
        tmp_path / "shard_s0.json",
        scenario_index=0,
        scenario_name="test",
        rule_type="independent endpoint replication under a frozen observed-step rule",
    )
    # shard_s1 carries a stress-focused rule_type -- the scenario that caused
    # the generic merge utility to reject all three loop160 shards.
    s1 = _make_shard(
        tmp_path / "shard_s1.json",
        scenario_index=1,
        scenario_name="stress_test",
        rule_type="stress-focused powered replication under a frozen observed-step rule",
    )
    s2 = _make_shard(
        tmp_path / "shard_s2.json",
        scenario_index=2,
        scenario_name="force_model_mismatch_test",
        rule_type="independent endpoint replication under a frozen observed-step rule",
    )
    freeze = _make_freeze_record(tmp_path)
    out = tmp_path / "merged.json"

    result = loop160_script.merge_shards_loop160(
        shard_paths=[s0, s1, s2],
        out_path=out,
        freeze_record_path=freeze,
    )

    assert out.exists(), "merged artifact was not written"
    assert result["status"] == "completed"
    assert result["num_scenarios"] == 3
    names = [s["name"] for s in result["scenarios"]]
    assert names == SCENARIOS, f"unexpected scenario order: {names}"
    # The merged frozen_rule must cover all three scenarios.
    assert result["frozen_rule"]["scenarios"] == SCENARIOS
    # The rule_type in the merged record is the canonical single-string value.
    assert result["frozen_rule"]["rule_type"] == (
        "independent endpoint replication under a frozen observed-step rule"
    )
    # Summary must have K and trajectories fields.
    assert result["summary"]["K"] == EXPECTED_NUM_REALIZATIONS
    assert result["summary"]["trajectories_per_realization"] == EXPECTED_TRAJECTORIES
    assert result["summary"]["scenarios_with_learned_positive_under_frozen_rule"] == 0
    assert result["summary"]["scenarios_with_classical_best_on_primary"] == 3


def test_merge_shards_output_contains_freeze_record_hash(tmp_path: Path) -> None:
    """Merged frozen_rule must include the freeze_record_canonical_sha256."""
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)
    out = tmp_path / "merged.json"

    result = loop160_script.merge_shards_loop160([s0, s1, s2], out, freeze)

    assert "freeze_record_canonical_sha256" in result["frozen_rule"]
    # The hash must match the value stored in the freeze record.
    stored = json.loads(freeze.read_text(encoding="utf-8")).get("canonical_rule_sha256", "")
    assert result["frozen_rule"]["freeze_record_canonical_sha256"] == stored


def test_merge_shards_rejects_missing_scenario(tmp_path: Path) -> None:
    """merge_shards_loop160 must raise MergeValidationError when a scenario
    is absent (providing only 2 shards instead of all 3)."""
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    freeze = _make_freeze_record(tmp_path)
    out = tmp_path / "merged.json"

    with pytest.raises(loop160_script.MergeValidationError, match=r"missing indices"):
        loop160_script.merge_shards_loop160([s0, s1], out, freeze)


def test_merge_shards_rejects_wrong_scenario_name(tmp_path: Path) -> None:
    """merge_shards_loop160 must fail if a scenario carries an unexpected name."""
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "wrong_name")  # bad name
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)

    with pytest.raises(loop160_script.MergeValidationError, match=r"scenario names"):
        loop160_script.merge_shards_loop160([s0, s1, s2], tmp_path / "out.json", freeze)


def test_merge_shards_rejects_duplicate_scenario_index(tmp_path: Path) -> None:
    """Duplicate scenario_index values across shards must be rejected."""
    s0a = _make_shard(tmp_path / "s0a.json", 0, "test")
    s0b = _make_shard(tmp_path / "s0b.json", 0, "stress_test")  # duplicate index 0
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)

    with pytest.raises(loop160_script.MergeValidationError, match=r"duplicate"):
        loop160_script.merge_shards_loop160([s0a, s0b, s2], tmp_path / "out.json", freeze)


def test_merge_shards_rejects_non_boolean_learned_positive(tmp_path: Path) -> None:
    """learned_positive_under_frozen_rule must be a Python bool; strings fail."""
    import json as _json

    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    # Inject a string "false" into the scenario field.
    payload = _json.loads(s0.read_text(encoding="utf-8"))
    payload["scenarios"][0]["learned_positive_under_frozen_rule"] = "false"
    s0.write_text(_json.dumps(payload, indent=2), encoding="utf-8")

    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)

    with pytest.raises(loop160_script.MergeValidationError, match=r"boolean"):
        loop160_script.merge_shards_loop160([s0, s1, s2], tmp_path / "out.json", freeze)


def test_merge_shards_rejects_learned_positive_without_rgr_gf_best(
    tmp_path: Path,
) -> None:
    """If learned_positive is True, best_method_primary must be 'RGR-GF'."""
    s0 = _make_shard(
        tmp_path / "s0.json", 0, "test",
        learned_positive=True,
        best_method="EKF",  # wrong -- should be RGR-GF
        ci_high=-5.0,
    )
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)

    with pytest.raises(loop160_script.MergeValidationError, match=r"best_method_primary"):
        loop160_script.merge_shards_loop160([s0, s1, s2], tmp_path / "out.json", freeze)


def test_merge_shards_rejects_learned_positive_ci_high_nonneg(
    tmp_path: Path,
) -> None:
    """If learned_positive is True, the CI high endpoint must be strictly < 0."""
    s0 = _make_shard(
        tmp_path / "s0.json", 0, "test",
        learned_positive=True,
        best_method="RGR-GF",
        ci_high=5.0,   # non-negative -- decision predicate not satisfied
        ci_low=-20.0,
        mean_gap=-10.0,
    )
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)

    with pytest.raises(loop160_script.MergeValidationError, match=r"ci_high"):
        loop160_script.merge_shards_loop160([s0, s1, s2], tmp_path / "out.json", freeze)


def test_merge_shards_rejects_wrong_invariant_field(tmp_path: Path) -> None:
    """A shard with a wrong invariant frozen_rule field (e.g. wrong base seed)
    must be rejected."""
    s0 = _make_shard(
        tmp_path / "s0.json", 0, "test",
        realization_base_seed=9999999,  # wrong seed
    )
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)

    with pytest.raises(loop160_script.MergeValidationError, match=r"realization_base_seed"):
        loop160_script.merge_shards_loop160([s0, s1, s2], tmp_path / "out.json", freeze)


def test_merge_shards_rejects_missing_freeze_record(tmp_path: Path) -> None:
    """merge_shards_loop160 must raise when the freeze record does not exist."""
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    missing_freeze = tmp_path / "no_preregistration.json"

    with pytest.raises(loop160_script.MergeValidationError, match=r"freeze record"):
        loop160_script.merge_shards_loop160([s0, s1, s2], tmp_path / "out.json", missing_freeze)


def test_merge_shards_rejects_tampered_freeze_record(tmp_path: Path) -> None:
    """merge_shards_loop160 must raise MergeValidationError when the freeze
    record self-hash does not match its contents (tampered after freeze)."""
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)

    # Tamper: change a payload field without recomputing the self-hash.
    record = json.loads(freeze.read_text(encoding="utf-8"))
    record["num_realizations_per_scenario"] = 99
    freeze.write_text(json.dumps(record, indent=2), encoding="utf-8")

    with pytest.raises(loop160_script.MergeValidationError, match=r"self-hash"):
        loop160_script.merge_shards_loop160(
            [s0, s1, s2], tmp_path / "out.json", freeze
        )


def test_merge_shards_scenario_order_independent_of_shard_input_order(
    tmp_path: Path,
) -> None:
    """Scenarios must be sorted by scenario_index regardless of shard input order."""
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    freeze = _make_freeze_record(tmp_path)

    # Provide shards in reverse order: s2, s0, s1.
    result = loop160_script.merge_shards_loop160(
        [s2, s0, s1], tmp_path / "out.json", freeze
    )
    names = [s["name"] for s in result["scenarios"]]
    assert names == SCENARIOS, f"expected sorted by scenario_index, got {names}"


def test_merge_shards_cli_subcommand(tmp_path: Path) -> None:
    """The merge-shards CLI subcommand must return 0 on success."""
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    s2 = _make_shard(tmp_path / "s2.json", 2, "force_model_mismatch_test")
    freeze = _make_freeze_record(tmp_path)
    out = tmp_path / "out.json"

    rc = loop160_script.main([
        "merge-shards",
        "--shard", str(s0),
        "--shard", str(s1),
        "--shard", str(s2),
        "--out-path", str(out),
        "--freeze-record", str(freeze),
    ])
    assert rc == 0
    assert out.exists()


def test_merge_shards_cli_returns_2_on_validation_failure(tmp_path: Path) -> None:
    """The merge-shards CLI must return exit code 2 on validation failure."""
    # Only two shards → missing scenario_index 2.
    s0 = _make_shard(tmp_path / "s0.json", 0, "test")
    s1 = _make_shard(tmp_path / "s1.json", 1, "stress_test")
    freeze = _make_freeze_record(tmp_path)
    out = tmp_path / "out.json"

    rc = loop160_script.main([
        "merge-shards",
        "--shard", str(s0),
        "--shard", str(s1),
        "--out-path", str(out),
        "--freeze-record", str(freeze),
    ])
    assert rc == 2


def test_merge_shards_if_artifact_materialized() -> None:
    """If the real merged artifact exists, run basic invariant checks on it."""
    if not ARTIFACT_PATH.exists():
        pytest.skip("loop-160 merged artifact not yet materialized")
    data = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert data["status"] == "completed"
    assert data["schema_version"] == "observed_step_prospective_replication_v1"
    assert data["artifact_role"] == "additional_internal_prospective_replication_loop160"
    assert data["frozen_rule"]["realization_base_seed"] == EXPECTED_BASE_SEED
    assert data["frozen_rule"]["scenarios"] == SCENARIOS
    assert data["summary"]["K"] == EXPECTED_NUM_REALIZATIONS
    assert data["num_scenarios"] == 3
    names = [s["name"] for s in data["scenarios"]]
    assert names == SCENARIOS
