"""Manuscript-integration tests for the loop-163 K=96 all-scenario table.

These cover the build_paper_assets.py generator and its wiring into the paper:
  - the builder function exists and is dispatched to the correct output path
  - the table renders from a synthetic artifact with the expected label
  - the label is distinct from the loop-160 K=32 replication label (no clash)
  - the caption carries the required scope-boundary language
  - two-decimal gap/CI formatting
  - no forbidden paper-facing terms; no double-closing-brace footnote
  - the supplement \\input wiring and the main-text S-Table reference are present
  - the real materialized table file (if present) is well-formed

No estimators are run and no model is loaded. The decision-rule, freeze-record,
CSV, and merge-shards behaviour of the underlying replication script are covered
by test_observed_step_internal_prospective_replication_loop163_k96.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_BPA_PATH = REPO_ROOT / "scripts" / "build_paper_assets.py"

_BPA_SPEC = importlib.util.spec_from_file_location("build_paper_assets", _BPA_PATH)
assert _BPA_SPEC and _BPA_SPEC.loader
bpa = importlib.util.module_from_spec(_BPA_SPEC)
sys.modules[_BPA_SPEC.name] = bpa
_BPA_SPEC.loader.exec_module(bpa)

METHODS = ["EKF", "UKF", "AUKF", "RGR-GF"]
SCENARIOS = ["test", "stress_test", "force_model_mismatch_test"]
TABLE_REL = (
    "paper/tables/observed_step_internal_prospective_replication_k96_allscenario.tex"
)
LABEL = "tab:observed_step_internal_prospective_replication_k96_allscenario"
LOOP160_LABEL = "tab:observed_step_internal_prospective_replication"
LOOP_SUFFIX_RE = re.compile(r"tab:[A-Za-z0-9_:-]*_loop\d+")


def _synthetic_artifact(path: Path, n_realizations: int = 96, n_traj: int = 24) -> Path:
    """Minimal synthetic merged artifact matching the loop-163 export format."""
    row_template: dict = {
        "scenario_index": 0,
        "n_realizations": n_realizations,
        "trajectories_per_realization": n_traj,
        "observed_step_pos_rmse_m": {
            "EKF": 410.0, "UKF": 430.0, "AUKF": 405.0, "RGR-GF": 420.0,
        },
        "primary_observed_step_pos_rmse_m": {
            "EKF": 410.0, "UKF": 430.0, "AUKF": 405.0, "RGR-GF": 420.0,
        },
        "best_method_primary": "AUKF",
        "best_classical_primary": "AUKF",
        "rgr_gf_minus_best_classical_primary_mean_m": 15.0,
        "rgr_gf_minus_best_classical_primary_ci_low_m": 5.0,
        "rgr_gf_minus_best_classical_primary_ci_high_m": 25.0,
        "learned_positive_under_frozen_rule": False,
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
        "artifact_role": "additional_internal_prospective_replication_loop163_k96",
        "frozen_rule": {
            "primary_metric": "observed_step_position_rmse_m",
            "reference_metric": "all_step_position_rmse_m",
            "num_realizations_per_scenario": n_realizations,
            "trajectories_per_realization": n_traj,
            "frozen_before_evaluation": True,
            "not_external_preregistration": True,
            "realization_base_seed": 1630000,
        },
        "num_scenarios": len(scenarios_out),
        "scenarios": scenarios_out,
        "summary": {
            "n_scenarios": len(scenarios_out),
            "K": n_realizations,
            "scenarios_with_learned_positive_under_frozen_rule": 0,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_builder_defined_and_callable() -> None:
    assert hasattr(
        bpa, "build_observed_step_internal_prospective_replication_k96_allscenario_table"
    )
    assert callable(
        bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table
    )


def test_bpa_dispatches_loop163_k96_table() -> None:
    """The build dispatch must write the K=96 table to its supplement path."""
    text = _BPA_PATH.read_text(encoding="utf-8")
    # The output path string literal (may be split across source lines).
    assert "observed_step_internal_prospective_replication_k96_allscenario.tex" in text
    assert (
        "build_observed_step_internal_prospective_replication_k96_allscenario_table()"
        in text
    )


def test_table_renders_with_synthetic_artifact(tmp_path: Path) -> None:
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop163_k96.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
        artifact
    )
    assert "\\begin{table}" in tex
    assert LABEL in tex
    # Three scenario rows must render.
    for label in ("Nominal", "Measurement-noise stress", "Controlled force-model mismatch"):
        assert label in tex


def test_label_distinct_from_loop160_table(tmp_path: Path) -> None:
    """The K=96 table must NOT reuse the loop-160 K=32 label (duplicate-label clash)."""
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop163_k96.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
        artifact
    )
    label_lines = [ln for ln in tex.splitlines() if "\\label{" in ln]
    assert label_lines, "no \\label line found"
    for ln in label_lines:
        assert LOOP160_LABEL not in ln or LABEL in ln, (
            "K=96 table reuses the loop-160 K=32 label"
        )
    # The bare loop-160 label must not appear as a standalone \label.
    assert f"\\label{{{LOOP160_LABEL}}}" not in tex


def test_generated_label_is_paper_facing_unsuffixed(tmp_path: Path) -> None:
    """The paper-facing \\label must not carry a _loop<N> suffix.

    verify_manuscript_revision.py's kalmannet_labels_are_paper_facing_unsuffixed
    check fails on any tab:..._loop\\d+ label in the expanded manuscript.
    """
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop163_k96.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
        artifact
    )
    label_lines = [ln for ln in tex.splitlines() if "\\label{" in ln]
    assert label_lines, "no \\label line found"
    for ln in label_lines:
        assert not LOOP_SUFFIX_RE.search(ln), (
            f"paper-facing label carries a _loop suffix: {ln!r}"
        )
    assert LABEL in tex
    assert not LOOP_SUFFIX_RE.search(f"\\ref{{{LABEL}}}")


def test_caption_carries_scope_boundary(tmp_path: Path) -> None:
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop163_k96.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
        artifact
    )
    lower = tex.lower()
    assert "not external preregistration" in lower
    assert "internal" in lower
    assert "operational validation" in lower
    assert "independently seeded" in lower
    # Must signal the all-scenario K=96 distinction from the stress-only check.
    assert "all-scenario" in lower or "all three scenarios" in lower
    assert "k{=}96" in lower or "$k=96$" in lower or "k=96" in lower


def test_gap_two_decimal_formatting(tmp_path: Path) -> None:
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop163_k96.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
        artifact
    )
    assert "15.00 [5.00, 25.00]" in tex
    assert "15.0 [5.0, 25.0]" not in tex


def test_no_forbidden_terms(tmp_path: Path) -> None:
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop163_k96.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
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
        "hon" + "est",
        "smoke" + "-test",
        "book" + "ends",
    ]:
        assert term not in lower, f"forbidden term {term!r} found in table"


def test_footnote_no_double_closing_brace(tmp_path: Path) -> None:
    artifact = _synthetic_artifact(
        tmp_path / "observed_step_internal_prospective_replication_loop163_k96.json"
    )
    tex = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
        artifact
    )
    footnote_lines = [ln for ln in tex.splitlines() if "footnotesize" in ln]
    assert footnote_lines, "no footnotesize line found in generated table"
    for ln in footnote_lines:
        assert not ln.rstrip().endswith("}}"), (
            f"footnote line ends with double closing brace (LaTeX error): {ln!r}"
        )


def test_unavailable_stub(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    result = bpa.build_observed_step_internal_prospective_replication_k96_allscenario_table(
        missing
    )
    assert result.startswith("%")


def test_supplement_input_wiring() -> None:
    supplement = (REPO_ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
    assert (
        "\\input{tables/observed_step_internal_prospective_replication_k96_allscenario.tex}"
        in supplement
    )


def test_main_text_references_stable_table() -> None:
    main_tex = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert f"\\ref{{{LABEL}}}" in main_tex


def test_materialized_table_well_formed() -> None:
    table_path = REPO_ROOT / TABLE_REL.replace("/", "\\")
    if not table_path.exists():
        # Fall back to forward-slash join for non-Windows checkouts.
        table_path = REPO_ROOT / TABLE_REL
    if not table_path.exists():
        import pytest

        pytest.skip("loop-163 K=96 table not materialized")
    tex = table_path.read_text(encoding="utf-8")
    assert LABEL in tex
    label_lines = [ln for ln in tex.splitlines() if "\\label{" in ln]
    for ln in label_lines:
        assert not LOOP_SUFFIX_RE.search(ln), (
            f"materialized paper-facing label carries a _loop suffix: {ln!r}"
        )
    assert "\\begin{table}" in tex and "\\end{table}" in tex
    assert "402.2" in tex  # nominal EKF observed-step RMSE from the real artifact
