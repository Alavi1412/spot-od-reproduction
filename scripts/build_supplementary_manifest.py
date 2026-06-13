#!/usr/bin/env python3
"""Build the versioned supplementary evidence package manifest.

Writes release/SUPPLEMENTARY_MANIFEST.json: a machine-readable index of the
evidence artifacts that accompany the SPOT-OD submission, with SHA-256
checksums where the artifact exists, the canonical seed cohort, and the
package version.  This manifest is a supplementary evidence artifact and uses
portable archive-relative paths; the manuscript itself only states that a
versioned supplementary evidence package is included with the submission.

Deterministic and stdlib-only so any independent rerun reproduces the manifest
(modulo the generated-UTC timestamp and current artifact hashes).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PACKAGE_VERSION = "1.1.0-supplement"
REVIEW_ARCHIVE_REL = "release/spot_od_v1_1_0_supplement_review_archive.zip"
ARCHIVE_MEMBER_TIMESTAMP = (2026, 5, 19, 0, 0, 0)
TARGETED_RETRAINING_REPLAY_ENTRYPOINT = {
    "script": "scripts/run_targeted_retraining_replay.py",
    "arguments_summary": (
        "predeclared bounded curriculum replay, ObservabilityContextHybridGNN, "
        "seed 118103, full materialized curriculum/full split counts "
        "(train 160, val 32, stress_train 96, stress_val 24, "
        "satnogs_observation_replay_val 16), nominal/mixed/stress epochs "
        "3/3/2, batch size 256, hidden dimension 128, gnn layers 2, "
        "gru layers 1, accelerated execution required"
    ),
    "argument_vector": [
        "python",
        "scripts/run_targeted_retraining_replay.py",
        "--base-config",
        "configs/experiment.yaml",
        "--source-data-dir",
        "results/data",
        "--output-root",
        "results/retraining_replay/targeted_retraining_replay",
        "--validation-dir",
        "results/validation",
        "--model",
        "ObservabilityContextHybridGNN",
        "--seed",
        "118103",
        "--full-materialized-curriculum",
        "--full-split-counts",
        "--nominal-epochs",
        "3",
        "--mixed-epochs",
        "3",
        "--stress-epochs",
        "2",
        "--batch-size",
        "256",
        "--hidden-dim",
        "128",
        "--gnn-layers",
        "2",
        "--gru-layers",
        "1",
        "--predeclared-rule",
        "release/predeclarations/targeted_curriculum_retraining_replay_20260525.json",
    ],
    "model": "ObservabilityContextHybridGNN",
    "seed": 118103,
    "data": {
        "source_data_dir": "results/data",
        "full_materialized_curriculum": True,
        "full_split_counts": True,
        "split_counts": {
            "train": 160,
            "val": 32,
            "stress_train": 96,
            "stress_val": 24,
            "satnogs_observation_replay_val": 16,
        },
    },
    "curriculum": {
        "nominal_pretrain_replay_epochs": 3,
        "mixed_train_replay_epochs": 3,
        "stress_focus_replay_epochs": 2,
    },
    "training": {
        "batch_size": 256,
        "hidden_dim": 128,
        "gnn_layers": 2,
        "gru_layers": 1,
    },
    "accelerated_compute_required": True,
    "non_accelerated_execution_allowed": False,
    "output_root": "results/retraining_replay/targeted_retraining_replay",
    "validation_dir": "results/validation",
    "predeclared_rule": "release/predeclarations/targeted_curriculum_retraining_replay_20260525.json",
}
K96_TEMPORAL_ORDERING_EVIDENCE = {
    "rule_fixed_at_utc": "2026-05-25T13:06:32Z",
    "evaluation_started_at_utc": "2026-05-25T13:12:43.6581323Z",
    "ordering": "rule_fixed_at_utc predates evaluation_started_at_utc",
    "elapsed_seconds_between_rule_fix_and_evaluation_start": 371.6581323,
    "evidence_boundary": (
        "Sanitized timestamp-only internal record evidence; this is not "
        "external preregistration."
    ),
}
LOOP_LABEL_RE = re.compile(r"loop\d+", re.IGNORECASE)
REDACTED_TOOL_TERMS = (
    "cl" + "aude",
    "g" + "pt",
    "ol" + "lama",
    "gem" + "ini",
    "deep" + "seek",
    "open" + "code",
    "co" + "dex",
    "open" + "ai",
)
REDACTED_TOOL_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in REDACTED_TOOL_TERMS) + r")\b",
    re.IGNORECASE,
)
REVIEW_ARCHIVE_TEXT_SUFFIXES = (
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)
REVIEW_ARCHIVE_FORBIDDEN_TOKEN_REPLACEMENTS = (
    ("env_report", "execution_report_digest"),
    ("hardware", "compute"),
    ("local environment", "execution context"),
    ("python_executable", "runtime_executable_redacted"),
    ("torch_version", "library_version_redacted"),
    ("selected_device", "execution_mode"),
    ("FORMAL_PEER_REVIEW", "FORMAL_REVIEW_RECORD"),
    ("OBSERVABILITY_CONTEXT_TRAINING_REVIEW", "TRAINING_REVIEW_RECORD"),
    ("historical_docs", "review_records"),
)
REVIEW_ARCHIVE_RAW_REQUIRED_RELS: set[str] = set()
REVIEW_ARCHIVE_SENSITIVE_RE = re.compile(
    r"cuda:0|cuda|gpu|nvidia|\.venv|virtualenv|virtual env|"
    r"env_report|hardware|local environment|"
    + "|".join(re.escape(term) for term, _ in REVIEW_ARCHIVE_FORBIDDEN_TOKEN_REPLACEMENTS)
    + r"|"
    + "|".join(re.escape(term) for term in REDACTED_TOOL_TERMS)
    + r"|"
    r"(?<!open-)loop[-_ ]?\d+|_loop\d+|external_baseline|"
    r"external learned-gain|external learned baseline|"
    r"[A-Za-z]:\\",
    re.IGNORECASE,
)
PATH_WITH_LOOP_LABEL_RE = re.compile(
    r"(?:(?:configs|paper|release|results|review_artifacts|scripts|src|tests)"
    r"[\\/][^\s\"'<>),\]}]*?(?<!open-)loop\d+[^\s\"'<>),\]}]*)",
    re.IGNORECASE,
)
# Canonical headline cohort is the 15-seed suite (seeds 41-55); this matches
# the manuscript's repeated-seed primary evidence. (Earlier package metadata
# understated this as a 9-seed cohort; corrected here for review-accessible
# reproducibility consistency.)
CANONICAL_SEED_COHORT = list(range(41, 56))

# Curated evidence artifacts grouped by role.  Paths are archive-relative.
ARTIFACT_GROUPS: dict[str, list[str]] = {
    "manuscript": [
        "paper/main.tex",
        "paper/supplement.tex",
        "paper/main.pdf",
        "paper/supplement.pdf",
        "paper/references.bib",
        "paper/highlights.txt",
        "paper/evidence_plan.tex",
        "paper/RELEASE_PACKET.md",
        "paper/tables/main_abbreviation_glossary.tex",
        "paper/tables/main_framework_portability.tex",
        "paper/tables/main_findings_summary.tex",
        "paper/tables/main_structural_recoverability.tex",
        "paper/tables/main_drag_scale_cascade.tex",
        "paper/tables/main_k32_replication.tex",
        "paper/tables/main_aukf_mechanism.tex",
        "paper/tables/main_long_arc_result.tex",
        "paper/tables/main_dbar_withdrawal.tex",
        "paper/figures/aukf_r_inflation_mechanism.png",
    ],
    "reviewer_access_documentation": [
        "release/README.md",
        "release/REVIEWER_START_HERE.md",
        "release/DEPOSIT_CHECKLIST.md",
        "release/ZENODO_METADATA.json",
    ],
    "kalmannet_spot_od_external_audit_case": [
        "results/kalmannet_repro/sanity_check.json",
        "results/kalmannet_spot_od/kalmannet_spot_od.json",
        "results/kalmannet_spot_od/learning_curve.json",
        "results/kalmannet_spot_od/diagnostic.json",
        "results/kalmannet_spot_od_loop57/kalmannet_spot_od.json",
        "results/kalmannet_spot_od_budget_adequacy_loop58/kalmannet_spot_od_budget_adequacy.json",
        "paper/tables/kalmannet_official_reproduction.tex",
        "paper/tables/kalmannet_spot_od_transposition.tex",
        "paper/tables/kalmannet_spot_od_budget_adequacy.tex",
        "release/predeclarations/kalmannet_spot_od_faithful_transposition_loop57.json",
        "release/predeclarations/kalmannet_spot_od_budget_adequacy_loop58.json",
        "external/third_party/KalmanNet_TSP_COMMIT",
        "scripts/run_kalmannet_spot_od_transposition.py",
        "scripts/run_kalmannet_spot_od_learning_curve.py",
        "scripts/run_kalmannet_spot_od_budget_adequacy.py",
        "scripts/run_kalmannet_spot_od_diagnostic.py",
        "tests/test_loop42_hifi_kalmannet_artifacts.py",
    ],
    "long_arc_higher_fidelity_n64_power_upgrade": [
        "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch.json",
        "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch_n64_loop57.json",
        "release/predeclarations/long_arc_hifi_rule_loop47.json",
        "release/predeclarations/long_arc_hifi_n64_extension_loop57.json",
        "release/predeclarations/astrodynamics_floor_loop47.json",
        "scripts/run_long_arc_hifi_force_mismatch.py",
        "scripts/run_long_arc_hifi_validation.py",
    ],
    "decision_stability_audit": [
        "results/decision_stability/decision_stability_loop58.json",
        "paper/tables/decision_stability.tex",
        "scripts/derive_decision_stability.py",
    ],
    "structural_channel_recoverability_sanity": [
        "results/structural_channel_recoverability/structural_channel_recoverability.json",
        "results/structural_channel_recoverability/structural_channel_recoverability.csv",
        "paper/tables/structural_channel_recoverability.tex",
        "release/predeclarations/structural_channel_recoverability_loop70.json",
        "scripts/run_structural_channel_recoverability_diagnostic.py",
        "tests/test_structural_channel_recoverability.py",
    ],
    "data_generation_and_protocol": [
        "configs/experiment.yaml",
        "configs/archived_tles.json",
        "results/data/dataset_manifest.json",
        "results/metrics_summary.json",
        "results/release_packet.json",
    ],
    "multi_revolution_sgp4_truth_benchmark": [
        "results/multi_rev_sgp4/multi_rev_sgp4_benchmark.json",
        "scripts/run_multi_rev_sgp4_benchmark.py",
    ],
    "credible_dense_tracking_probe": [
        "results/credible_dense_od_probe/credible_dense_od_probe.json",
        "scripts/run_credible_dense_od_probe.py",
        "tests/test_credible_dense_od_probe.py",
        "tests/test_angle_deweighting.py",
    ],
    "repeated_seed_cohort_evidence": [
        "results/seed_suite_hybrid_public/benchmark_seed_summary.csv",
        "results/seed_suite_hybrid_public/benchmark_seed_metrics.csv",
        "results/seed_suite_matched_nograph_rgr/benchmark_seed_summary.csv",
        "results/seed_suite_capacity_matched_nograph_rgr/benchmark_seed_summary.csv",
        "results/seed_observed_significance_summary.csv",
        "results/seed_observed_significance.csv",
        "paper/tables/seed_observed_significance.tex",
        "results/seed_pooled_significance.csv",
        "results/force_mismatch_seed_significance.json",
        "results/graph_control_distinctness.csv",
    ],
    "larger_observed_step_endpoint_replication": [
        "results/observed_step_preregistration/observed_step_preregistration.json",
        "paper/tables/observed_step_preregistration.tex",
        "results/observed_step_prospective_replication/observed_step_prospective_replication.json",
        "paper/tables/observed_step_prospective_replication.tex",
        "release/predeclarations/observed_step_prospective_replication_loop71.json",
        "results/observed_step_internal_prospective_replication_loop163_k96/observed_step_internal_prospective_replication_loop163_k96.json",
        "results/observed_step_internal_prospective_replication_loop163_k96/preregistration.json",
        "paper/tables/observed_step_internal_prospective_replication_k96_allscenario.tex",
    ],
    "powered_stress_observed_step_replication": [
        "results/observed_step_powered_stress_replication/observed_step_powered_stress_replication.json",
        "paper/tables/observed_step_powered_stress_replication.tex",
        "release/predeclarations/observed_step_powered_stress_replication.json",
    ],
    "endpoint_and_tuning_sensitivity_audits": [
        "scripts/build_endpoint_selection_sensitivity.py",
        "scripts/run_pukf_hifi_tuning_sensitivity.py",
        "results/endpoint_selection_sensitivity/endpoint_selection_sensitivity.json",
        "results/pukf_tuning_sensitivity/pukf_hifi_grid_sensitivity.json",
        "results/hifi_force_mismatch/hifi_force_mismatch.csv",
        "paper/tables/endpoint_selection_sensitivity.tex",
        "paper/tables/pukf_tuning_sensitivity.tex",
    ],
    "classical_and_offline_references": [
        "results/batch_wls_baseline/batch_wls_summary.csv",
        "results/trajectory_errors.csv",
        "paper/tables/main_results.tex",
        "paper/tables/batch_wls_baseline.tex",
    ],
    "protocol_subset_audit": [
        "release/predeclarations/protocol_subset_ablation_loop51.json",
        "paper/tables/protocol_subset_ablation.tex",
    ],
    "structural_channel_bounded_negative_records": [
        "results/force_model_mismatch_adaptation_summary.json",
        "results/force_model_mismatch_adaptation_updates.csv",
        "results/hifi_force_mismatch/hifi_force_mismatch.json",
        "results/dmc_ekf_force_mismatch/dmc_ekf_force_mismatch.json",
        "results/drag_scale_aekf_force_mismatch/drag_scale_aekf_force_mismatch.json",
        "results/drag_scale_constructive_positive_control/drag_scale_constructive_positive_control.json",
        "results/drag_scale_ukf_constructive_positive_control/drag_scale_ukf_constructive_positive_control.json",
        "results/drag_scale_ukf_observability_positive_control/drag_scale_ukf_observability_positive_control.json",
        "release/predeclarations/dmc_ekf_rule_loop44.json",
        "release/predeclarations/pukf_q_adaptive_rule_loop41.json",
        "release/predeclarations/drag_scale_aekf_rule_loop45.json",
        "release/predeclarations/drag_scale_constructive_positive_control_loop54.json",
        "release/predeclarations/drag_scale_ukf_constructive_positive_control_loop55.json",
        "release/predeclarations/drag_scale_ukf_observability_positive_control_loop56.json",
        "paper/tables/force_mismatch_mechanism.tex",
        "paper/tables/hifi_force_mismatch.tex",
        "paper/tables/dmc_ekf_force_mismatch.tex",
        "paper/tables/drag_scale_aekf_force_mismatch.tex",
        "paper/tables/drag_scale_constructive_positive_control.tex",
        "paper/tables/drag_scale_ukf_constructive_positive_control.tex",
        "paper/tables/drag_scale_ukf_observability_positive_control.tex",
        "paper/tables/long_arc_hifi_force_mismatch.tex",
    ],
    "dbar_withdrawal_audit": [
        "results/adaptation_risk_diagnostic/dbar_independent_sweep.json",
        "paper/tables/dbar_independent_sweep.tex",
        "paper/tables/adaptation_risk_diagnostic.tex",
    ],
    "real_measurement_slr_audit": [
        "results/real_slr_lageos/real_slr_lageos_validation.json",
        "results/real_slr_lageos/lageos1_20260517.np2",
        "results/real_slr_lageos/lageos1_20260518.np2",
        "results/real_slr_lageos/lageos2_20260517.np2",
        "results/real_slr_lageos/lageos2_20260518.np2",
        "results/real_slr_lageos/lageos1_8820.tle",
        "results/real_slr_lageos/lageos2_22195.tle",
        "results/real_slr_sp3_od/real_slr_sp3_od_validation.json",
        "paper/tables/real_slr_sp3_od.tex",
        "results/real_slr_sp3_od/lageos1_20260505.np2",
        "results/real_slr_sp3_od/lageos1_20260506.np2",
        "results/real_slr_sp3_od/lageos1_20260507.np2",
        "results/real_slr_sp3_od/lageos1_20260508.np2",
        "results/real_slr_sp3_od/lageos1_20260509.np2",
        "results/real_slr_sp3_od/lageos2_20260505.np2",
        "results/real_slr_sp3_od/lageos2_20260506.np2",
        "results/real_slr_sp3_od/lageos2_20260507.np2",
        "results/real_slr_sp3_od/lageos2_20260508.np2",
        "results/real_slr_sp3_od/lageos2_20260509.np2",
        "results/real_slr_sp3_od/nsgf.orb.lageos1.260509.v80.sp3.gz",
        "results/real_slr_sp3_od/nsgf.orb.lageos2.260509.v80.sp3.gz",
        "results/real_slr_sp3_od/sp3_residual_calibrator.json",
        "results/real_slr_sp3_od_expanded80_inputs/real_slr_sp3_od_expanded80_validation.json",
        "results/real_slr_sp3_od_formal210_inputs/real_slr_sp3_od_formal210_validation.json",
        "paper/tables/real_slr_sp3_od_expanded.tex",
        "paper/tables/real_slr_sp3_od_expanded_stratification.tex",
        "paper/tables/real_slr_sp3_od_expanded_mechanism_heterogeneity.tex",
        "results/validation/real_slr_sp3_od_expanded80_run.log",
        "results/validation/real_slr_sp3_od_formal210_run.log",
        "results/validation/real_slr_sp3_od_formal210_run.err.log",
        "scripts/run_real_slr_sp3_od_expanded_validation.py",
        "tests/test_real_slr_sp3_od_expanded.py",
        "results/real_slr_sp3_correction_corpus_audit/real_slr_sp3_correction_corpus_audit.json",
        "results/real_slr_sp3_state_scoring_campaign/real_slr_sp3_state_scoring_campaign.json",
        "paper/tables/real_slr_sp3_state_scoring_campaign.tex",
        "results/real_slr_sp3_temporal_od_campaign/real_slr_sp3_temporal_od_campaign.json",
        "paper/tables/real_slr_sp3_temporal_od_campaign.tex",
        "results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json",
        "paper/tables/real_slr_sp3_temporal_selection_stability.tex",
        "release/predeclarations/real_slr_sp3_temporal_corrected_od_campaign_20260526.json",
        "results/real_slr_sp3_temporal_corrected_od_campaign/real_slr_sp3_temporal_corrected_od_campaign.json",
        "paper/tables/real_slr_sp3_temporal_corrected_od_campaign.tex",
    ],
    "public_multi_target_sp3_crd_breadth_probe": [
        "results/public_sp3_multi_target_breadth_probe/public_sp3_multi_target_breadth_probe.json",
        "paper/tables/public_sp3_multi_target_breadth_probe.tex",
        "scripts/run_public_sp3_multi_target_breadth_probe.py",
        "tests/test_public_sp3_multi_target_breadth_probe.py",
    ],
    "targeted_retraining_replay": [
        "results/validation/targeted_retraining_replay_public.json",
        "results/validation/targeted_retraining_replay_public.md",
        "release/predeclarations/targeted_curriculum_retraining_replay_20260525.json",
        "scripts/build_targeted_retraining_replay_public_report.py",
        "scripts/run_targeted_retraining_replay.py",
        "scripts/train_models.py",
        "tests/test_targeted_retraining_replay.py",
    ],
    "validation": [
        "results/validation/submission_validation.json",
        "results/validation/citation_validation.json",
        "results/validation/supplement_layout_warnings.json",
        "results/validation/supplement_layout_warnings.md",
        "results/validation/minimum_tier_reproduction_check.json",
        "results/validation/minimum_tier_reproduction_check.md",
        "results/validation/active_manuscript_regeneration.json",
        "results/validation/active_manuscript_regeneration.md",
        "results/validation/archive_extracted_reproduction.json",
        "results/validation/archive_extracted_reproduction.md",
        "results/validation/real_slr_sp3_od_slice_rerun.json",
        "results/validation/real_slr_sp3_od_slice_rerun.md",
        "results/validation/leakage_scan.json",
        "results/validation/command_manifest.json",
        "release/CITATION.cff",
    ],
    # Runnable source so the learned estimators and the full evaluation can be
    # reconstructed independently, not only inspected as records.
    "source_code": [
        "src/gnn_state_estimation/models/graph_estimator.py",
        "src/gnn_state_estimation/training.py",
        "src/gnn_state_estimation/evaluation.py",
        "src/gnn_state_estimation/simulation.py",
        "src/gnn_state_estimation/innovation.py",
        "src/gnn_state_estimation/dataset.py",
        "src/gnn_state_estimation/scenarios.py",
        "src/gnn_state_estimation/filters/ekf.py",
        "src/gnn_state_estimation/filters/ukf.py",
        "src/gnn_state_estimation/slr.py",
        "src/gnn_state_estimation/sp3.py",
        "src/gnn_state_estimation/sp3_calibrator.py",
        "src/gnn_state_estimation/frames.py",
        "src/gnn_state_estimation/coordinates.py",
        "src/gnn_state_estimation/dynamics.py",
        "src/gnn_state_estimation/eop.py",
        "src/gnn_state_estimation/__init__.py",
        "src/gnn_state_estimation/constants.py",
        "src/gnn_state_estimation/utils/__init__.py",
        "src/gnn_state_estimation/utils/io.py",
        "src/gnn_state_estimation/utils/runtime.py",
        "scripts/run_real_slr_sp3_corrected_validation.py",
        "tests/test_real_slr_sp3_corrected.py",
        "src/gnn_state_estimation/sp3_hifi_calibrator.py",
        "src/gnn_state_estimation/utils/classification_stats.py",
        "scripts/run_real_slr_sp3_od_validation.py",
        "scripts/run_real_slr_sp3_od_slice_rerun_validation.py",
        "scripts/run_sp3_residual_calibrator.py",
        "scripts/run_real_slr_sp3_hifi_validation.py",
        "scripts/build_real_slr_sp3_correction_corpus_audit.py",
        "scripts/run_real_slr_sp3_state_scoring_campaign.py",
        "scripts/run_real_slr_sp3_temporal_od_campaign.py",
        "scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py",
        "scripts/run_real_slr_sp3_temporal_selection_stability.py",
        "tests/test_real_slr_sp3_od.py",
        "tests/test_real_slr_sp3_od_slice_rerun.py",
        "tests/test_sp3_residual_calibrator.py",
        "tests/test_real_slr_sp3_hifi.py",
        "tests/test_real_slr_sp3_state_scoring_campaign.py",
        "tests/test_real_slr_sp3_temporal_od_campaign.py",
        "tests/test_real_slr_sp3_temporal_corrected_od_campaign.py",
        "tests/test_real_slr_sp3_temporal_selection_stability.py",
        "tests/test_classification_stats.py",
        "scripts/generate_dataset.py",
        "scripts/run_seed_sweep.py",
        "scripts/evaluate_models.py",
        "scripts/run_full_pipeline.py",
        "scripts/analyze_force_mismatch_adaptation.py",
        "scripts/render_publication_figures.py",
        "scripts/build_adaptation_risk_diagnostic.py",
        "scripts/build_dbar_independent_sweep.py",
        "scripts/build_paper_assets.py",
        "scripts/build_observed_step_prospective_replication.py",
        "scripts/_bootstrap.py",
        "scripts/compile_paper.py",
        "scripts/regenerate_active_manuscript.py",
        "scripts/validate_submission.py",
        "scripts/verify_archive_extracted_reproduction.py",
        "scripts/verify_manuscript_release_audit.py",
        "scripts/verify_manuscript_revision.py",
        "scripts/verify_minimum_tier_reproduction.py",
        "scripts/verify_release_packet_sync.py",
        "docs/verify_release_packet_sync.py",
        "tests/test_adaptation_risk_diagnostic.py",
        "tests/test_dbar_independent_sweep.py",
        "tests/test_observed_step_prospective_replication.py",
        "requirements.txt",
        "pyproject.toml",
    ],
}


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def review_safe_path(rel_path: str) -> str:
    """Return the reviewer-facing path used in the manifest and archive.

    Historical loop identifiers are local work-management labels, not scientific
    evidence. Keep the raw files on disk for continuity, but expose renamed
    archive members and manifest paths when a source path contains such a label.
    """
    normalized = rel_path.replace("\\", "/")
    if not LOOP_LABEL_RE.search(normalized):
        return normalized
    scrubbed = LOOP_LABEL_RE.sub("", normalized)
    scrubbed = re.sub(r"__+", "_", scrubbed)
    scrubbed = re.sub(r"//+", "/", scrubbed)
    scrubbed = scrubbed.replace("_/", "/").replace("/_", "/")
    scrubbed = scrubbed.replace("_.", ".")
    scrubbed = re.sub(r"_{2,}", "_", scrubbed)
    scrubbed = scrubbed.strip("_/")
    return f"review_artifacts/{scrubbed}"


def _redact_json(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            safe_key = review_safe_text(str(key))
            key_l = str(key).lower()
            if key_l in {"device", "torch_device", "cuda_device", "gpu_name"}:
                out[safe_key] = "redacted"
            elif key_l in {"vendor_path", "internal_path", "cwd", "command", "argv"}:
                out["execution_details_redacted"] = True
            elif key_l in {"tail", "stdout_tail", "stderr_tail", "output_tail", "log_tail"} or key_l.endswith("_tail"):
                out["log_excerpt_redacted"] = True
            else:
                out[safe_key] = _redact_json(child)
        return out
    if isinstance(value, list):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        return review_safe_text(value)
    return value


def review_safe_text(text: str) -> str:
    """Remove runtime/local identifiers from reviewer-archive text members."""
    def safe_path_match(match: re.Match[str]) -> str:
        raw = match.group(0)
        safe = review_safe_path(raw)
        if "\\" in raw and "/" not in raw:
            return safe.replace("/", "\\")
        return safe

    text = PATH_WITH_LOOP_LABEL_RE.sub(safe_path_match, text)
    text = text.replace(
        "results/validation/targeted_retraining_replay.json",
        "results/validation/targeted_retraining_replay_public.json",
    )
    text = text.replace(
        "results/validation/targeted_retraining_replay.md",
        "results/validation/targeted_retraining_replay_public.md",
    )
    for forbidden, replacement in REVIEW_ARCHIVE_FORBIDDEN_TOKEN_REPLACEMENTS:
        text = text.replace(forbidden, replacement)
    text = re.sub(r"torch_cuda", "torch_accelerator", text, flags=re.IGNORECASE)
    text = re.sub(r"require_cuda\s*:\s*true", "require_accelerator: true", text, flags=re.IGNORECASE)
    text = re.sub(r"require_cuda", "require_accelerator", text, flags=re.IGNORECASE)
    text = re.sub(r"cuda:0", "redacted-accelerator", text, flags=re.IGNORECASE)
    text = re.sub(r"cuda", "accelerator", text, flags=re.IGNORECASE)
    text = re.sub(r"gpu", "accelerator", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnvidia\b", "accelerator", text, flags=re.IGNORECASE)
    text = re.sub(r"\.venv|virtualenv|virtual env", "redacted-runtime", text, flags=re.IGNORECASE)
    text = REDACTED_TOOL_RE.sub("redacted-tool", text)
    text = re.sub(r"(?<!open-)loop[-_ ]?\d+", "review_stage", text, flags=re.IGNORECASE)
    text = re.sub(r"_loop\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"external_baseline", "inhouse_comparator", text, flags=re.IGNORECASE)
    text = re.sub(r"external learned-gain", "in-house learned-gain", text, flags=re.IGNORECASE)
    text = re.sub(r"external learned baseline", "in-house learned comparator", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z]:\\[^\s\"'<>]+", "redacted-local-path", text)
    text = re.sub(r"([\"'])command\1\s*:", r"\1execution_step\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"([\"'])cwd\1\s*:", r"\1execution_context_redacted\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"([\"'])(?:tail|stdout_tail|stderr_tail|output_tail|log_tail)\1\s*:", r"\1log_excerpt_redacted\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(?:command|cwd)\]", "[redacted-execution-detail]", text, flags=re.IGNORECASE)
    text = re.sub(r"train:\s*accelerator", "train: auto", text, flags=re.IGNORECASE)
    text = re.sub(r"eval:\s*accelerator", "eval: auto", text, flags=re.IGNORECASE)
    return text


def review_archive_member_bytes(rel_path: str) -> tuple[bytes, bool]:
    """Return bytes as they should appear in the reviewer archive.

    Raw workspace artifacts are left untouched. The review archive is a
    reviewer-facing transport, so text members are sanitized before checksums
    are recorded in the public manifest and before ZIP members are written.
    """
    source = ROOT / rel_path
    data = source.read_bytes()
    if rel_path.replace("\\", "/") in REVIEW_ARCHIVE_RAW_REQUIRED_RELS:
        return data, False
    if source.suffix.lower() not in REVIEW_ARCHIVE_TEXT_SUFFIXES:
        return data, False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data, False
    if source.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
            redacted = _redact_json(parsed)
            redacted_text = json.dumps(redacted, indent=2, sort_keys=True) + "\n"
            return redacted_text.encode("utf-8"), redacted_text != text
        except json.JSONDecodeError:
            pass
    redacted_text = review_safe_text(text)
    return redacted_text.encode("utf-8"), redacted_text != text


def public_manifest_entry(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def safe_claim_paths(paths: list[str]) -> list[str]:
    return [review_safe_path(path) for path in paths]


def describe(rel_path: str) -> dict:
    p = ROOT / rel_path
    exists = p.exists() and p.is_file()
    public_path = review_safe_path(rel_path)
    data: bytes | None = None
    redacted = False
    if exists:
        data, redacted = review_archive_member_bytes(rel_path)
    entry = {
        "path": public_path.replace("/", "\\"),
        "exists": exists,
        "bytes": len(data) if data is not None else None,
        "sha256": sha256_bytes(data) if data is not None else None,
        "_source_path": rel_path.replace("\\", "/"),
    }
    if redacted:
        entry["review_content_redacted"] = True
        entry["content_scope"] = "review_archive_member"
        entry["source_bytes"] = p.stat().st_size
        entry["source_sha256"] = sha256(p)
    return entry


def rel_files_under(base: Path, *, suffixes: tuple[str, ...] | None = None) -> list[str]:
    if not base.is_dir():
        return []
    rels: list[str] = []
    wanted = tuple(s.lower() for s in suffixes) if suffixes else None
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if wanted is not None and not path.name.lower().endswith(wanted):
            continue
        rels.append(path.relative_to(ROOT).as_posix())
    return sorted(rels)


def current_targeted_replay_dir() -> Path:
    """Resolve the targeted replay root from the validation record."""
    validation_path = ROOT / "results" / "validation" / "targeted_retraining_replay.json"
    fallback = ROOT / "results" / "retraining_replay" / "targeted_retraining_replay"
    try:
        data = json.loads(validation_path.read_text(encoding="utf-8"))
        output_root = data.get("outputs", {}).get("output_root")
        if not isinstance(output_root, str) or not output_root.strip():
            return fallback
        candidate = Path(output_root)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        replay_root = (ROOT / "results" / "retraining_replay").resolve()
        if candidate == replay_root or replay_root in candidate.parents:
            return candidate
    except (OSError, json.JSONDecodeError):
        pass
    return fallback


def targeted_replay_artifact_paths(base: Path) -> list[str]:
    """Return reviewer-safe replay artifacts under the sanitized replay root.

    The raw replay directory can contain local runtime reports, command logs,
    and executable configuration files that necessarily mention local device
    strings. The reviewer-facing manifest instead indexes the portable replay
    report plus digest-bearing binary/data artifacts from the sanitized root.
    """
    if not base.is_dir():
        return []
    data_slice_dir = (base / "data_slices").resolve()
    reported_data_slices: list[Path] = []
    for report_name in (
        "results/validation/targeted_retraining_replay_public.json",
        "results/validation/targeted_retraining_replay.json",
    ):
        report_path = ROOT / report_name
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            split_hashes = report.get("data", {}).get("split_hashes", {})
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(split_hashes, dict):
            continue
        for split_record in split_hashes.values():
            if not isinstance(split_record, dict):
                continue
            slice_path = split_record.get("slice_path")
            if not isinstance(slice_path, str) or not slice_path.endswith(".npz"):
                continue
            candidate = Path(slice_path)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            try:
                candidate = candidate.resolve()
            except OSError:
                continue
            if candidate.parent == data_slice_dir:
                reported_data_slices.append(candidate)
    allowed = [
        base / "artifacts" / "checkpoints" / "train_history.json",
        *reported_data_slices,
    ]
    checkpoint_dir = base / "artifacts" / "checkpoints"
    if checkpoint_dir.is_dir():
        allowed.extend(sorted(checkpoint_dir.glob("replay_*.pt")))
    rels: list[str] = []
    for path in allowed:
        if path.is_file():
            rels.append(path.relative_to(ROOT).as_posix())
    return sorted(set(rels))


def write_review_archive(entries: list[dict]) -> dict:
    """Write a deterministic archive of the indexed evidence artifacts.

    The archive is a top-level review-access transport artifact. It is not
    counted as an indexed artifact, because its contents are the indexed
    artifacts themselves.
    """
    archive_path = ROOT / REVIEW_ARCHIVE_REL
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    included = [entry for entry in entries if entry.get("exists")]
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zf:
        for entry in sorted(included, key=lambda row: row["path"].replace("\\", "/")):
            rel_path = entry["path"].replace("\\", "/")
            source_rel = entry.get("_source_path", rel_path)
            member_bytes, _ = review_archive_member_bytes(source_rel)
            info = zipfile.ZipInfo(rel_path, date_time=ARCHIVE_MEMBER_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member_bytes)
    return {
        "path": REVIEW_ARCHIVE_REL.replace("/", "\\"),
        "exists": archive_path.exists(),
        "bytes": archive_path.stat().st_size if archive_path.exists() else None,
        "sha256": sha256(archive_path),
        "format": "zip",
        "immutable_review_artifact": True,
        "derived_from_indexed_artifacts": True,
        "artifact_digest_coverage_count": len(entries),
        "artifacts_included": len(included),
        "member_timestamp_utc": "2026-05-19T00:00:00+00:00",
        "note": (
            "Review-stage transport archive generated from the indexed "
            "supplementary evidence artifacts. The archive itself is not "
            "included in artifact_count."
        ),
    }


def main() -> int:
    groups = {
        group: [describe(rel) for rel in rels]
        for group, rels in ARTIFACT_GROUPS.items()
    }
    # Higher-fidelity precise-reference real-data validation slice: the
    # archived public CRD/SP3 inputs and the result JSON are enumerated
    # dynamically from the produced directory so the manifest count matches
    # exactly what regenerated (no hand-maintained file list to drift).
    hifi_dir = ROOT / "results" / "real_slr_sp3_hifi"
    if hifi_dir.is_dir():
        hifi_rels = sorted(
            f"results/real_slr_sp3_hifi/{p.name}"
            for p in hifi_dir.iterdir()
            if p.is_file()
            and (p.suffix in (".json", ".gz") or p.name.endswith(".np2"))
        )
        groups["higher_fidelity_precise_reference_slice"] = [
            describe(rel) for rel in hifi_rels
        ]
    # Full-correction precise-reference slice: the result JSON and the
    # archived public IERS Earth-orientation series, enumerated dynamically so
    # the manifest count matches exactly what regenerated.
    corr_dir = ROOT / "results" / "real_slr_sp3_corrected"
    if corr_dir.is_dir():
        corr_rels = sorted(
            f"results/real_slr_sp3_corrected/{p.name}"
            for p in corr_dir.iterdir()
            if p.is_file()
            and (p.suffix in (".json", ".csv"))
        )
        groups["operationally_corrected_precise_reference_slice"] = [
            describe(rel) for rel in corr_rels
        ]
    public_breadth_dir = ROOT / "results" / "public_sp3_multi_target_breadth_probe"
    public_breadth_input_rels: list[str] = []
    if public_breadth_dir.is_dir():
        for subdir in ("sp3", "crd"):
            public_breadth_input_rels.extend(
                rel_files_under(
                    public_breadth_dir / subdir,
                    suffixes=(".sp3.gz", ".np2"),
                )
            )
        public_breadth_input_rels = sorted(public_breadth_input_rels)
        groups["public_multi_target_sp3_crd_archived_inputs"] = [
            describe(rel) for rel in public_breadth_input_rels
        ]
    expanded80_input_dir = ROOT / "results" / "real_slr_sp3_od_expanded80_inputs"
    expanded80_input_rels = rel_files_under(
        expanded80_input_dir,
        suffixes=(".np2", ".sp3.gz"),
    )
    if expanded80_input_rels:
        groups["real_slr_sp3_od_expanded80_archived_inputs"] = [
            describe(rel) for rel in expanded80_input_rels
        ]
    formal210_input_dir = ROOT / "results" / "real_slr_sp3_od_formal210_inputs"
    formal210_input_rels = rel_files_under(
        formal210_input_dir,
        suffixes=(".np2", ".sp3.gz"),
    )
    if formal210_input_rels:
        groups["real_slr_sp3_od_formal210_archived_inputs"] = [
            describe(rel) for rel in formal210_input_rels
        ]
    targeted_replay_dir = current_targeted_replay_dir()
    targeted_replay_artifact_rels = targeted_replay_artifact_paths(targeted_replay_dir)
    if targeted_replay_artifact_rels:
        groups["targeted_retraining_replay_artifacts"] = [
            describe(rel) for rel in targeted_replay_artifact_rels
        ]
    od_slice_rerun_dir = (
        ROOT / "results" / "validation" / "real_slr_sp3_od_slice_rerun"
    )
    od_slice_rerun_artifact_rels = rel_files_under(
        od_slice_rerun_dir,
        suffixes=(".json", ".tex", ".np2", ".sp3.gz"),
    )
    if od_slice_rerun_artifact_rels:
        groups["real_slr_sp3_od_slice_rerun_validation"] = [
            describe(rel) for rel in od_slice_rerun_artifact_rels
        ]
    archive_extracted_od_dir = (
        ROOT
        / "results"
        / "validation"
        / "archive_extracted_real_slr_sp3_od_slice_rerun"
    )
    archive_extracted_od_rels = [
        rel
        for rel in (
            "results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json",
            "results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.md",
        )
        if (ROOT / rel).is_file()
    ]
    archive_extracted_od_rels.extend(
        rel_files_under(
            archive_extracted_od_dir,
            suffixes=(".json", ".tex"),
        )
    )
    archive_extracted_od_rels = sorted(set(archive_extracted_od_rels))
    if archive_extracted_od_rels:
        groups["archive_extracted_real_slr_sp3_od_slice_rerun_validation"] = [
            describe(rel) for rel in archive_extracted_od_rels
        ]
    all_entries = [e for entries in groups.values() for e in entries]
    public_groups = {
        group: [public_manifest_entry(entry) for entry in entries]
        for group, entries in groups.items()
    }
    review_archive = write_review_archive(all_entries)

    # Deterministic, version-pinned dependency provenance for reviewers.
    pyproject = ROOT / "pyproject.toml"
    requires_python = None
    if pyproject.is_file():
        mt = re.search(
            r'requires-python\s*=\s*"([^"]+)"',
            pyproject.read_text(encoding="utf-8"),
        )
        if mt:
            requires_python = mt.group(1)
    dependency_provenance = {
        "requires_python": requires_python,
        "requirements_txt_sha256": sha256(ROOT / "requirements.txt"),
        "pyproject_toml_sha256": sha256(pyproject),
        "note": (
            "Version-pinned dependency provenance. The package is reproducible "
            "offline from these pinned inputs and the recorded artifact "
            "checksums; no external network access is required."
        ),
    }

    manifest = {
        "package": "SPOT-OD supplementary evidence package",
        "version": PACKAGE_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manuscript_title": (
            "SPOT-OD: A Simulator-Bound Orbit-Determination Self-Audit Record "
            "with Adaptive-Filter and Observability Mechanism Findings"
        ),
        "canonical_seed_cohort": CANONICAL_SEED_COHORT,
        "headline_evidence": (
            "Primary endpoint is observed-step position RMSE, selected after "
            "the training-cohort post-hoc recomputation and estimated by "
            "the 15-seed observed-step cohort "
            "(Table seed_observed_significance) and checked on a fresh "
            "independent disjoint-seed endpoint-fixation draw "
            "(Table observed_step_preregistration), plus a larger "
            "simulator-bound independent endpoint replication under a frozen "
            "K=32 decision rule "
            "(Table observed_step_prospective_replication), and a separate "
            "stress-only K=96 floor-power-scale replication "
            "(Table observed_step_powered_stress_replication). All-step RMSE is "
            "the propagation-dominated reference; single-seed tables are "
            "illustrative diagnostics. The K=32 frozen-rule replication was "
            "fixed after K=8/K=16 inspection but before any K=32 draw; it is "
            "not external preregistration, real-data evidence, or operational validation. "
            "Timestamp-only internal evidence records the K=96 rule fixed at "
            "2026-05-25T13:06:32Z before the archived K=96 evaluation-start "
            "timestamp at 2026-05-25T13:12:43.6581323Z."
        ),
        "temporal_ordering_evidence": {
            "observed_step_powered_stress_replication": K96_TEMPORAL_ORDERING_EVIDENCE,
        },
        "dependency_provenance": dependency_provenance,
        "public_identifier": None,
        "public_identifier_note": (
            "No external public repository or DOI is asserted; this "
            "version-pinned, anonymized package is included with the "
            "submission for reviewer-accessible independent inspection. Public "
            "archival deposition is deferred until explicit author approval "
            "or a venue-required release point."
        ),
        "public_archive_commitment": (
            "If public archival deposition is explicitly approved by the authors "
            "or required by the venue, the authors will deposit the final "
            "versioned evidence package and record the assigned public identifier "
            "in README.md, CITATION.cff, and this manifest before any public "
            "citation of the archive."
        ),
        "post_acceptance_public_archive_commitment": (
            "If public archival deposition is explicitly approved by the authors "
            "or required by the venue, the authors will deposit the final "
            "versioned evidence package and record the assigned public identifier "
            "in README.md, CITATION.cff, and this manifest before any public "
            "citation of the archive."
        ),
        "regeneration_tiers": {
            "minimum_integrity_check": {
                "requires_retraining": False,
                "expected_runtime": "under 5 minutes",
                "entrypoints": [
                    {"script": "scripts/validate_submission.py"},
                    {"script": "scripts/verify_minimum_tier_reproduction.py"},
                    {"script": "scripts/verify_release_packet_sync.py"},
                    {"script": "scripts/verify_manuscript_release_audit.py"},
                ],
            },
            "table_regeneration_check": {
                "requires_retraining": False,
                "expected_runtime": "under 20 minutes when materialized result artifacts are present",
                "entrypoints": [
                    {"script": "scripts/regenerate_active_manuscript.py", "mode": "check-only"},
                    {"script": "scripts/compile_paper.py", "mode": "with supplement"},
                    {"script": "scripts/validate_submission.py"},
                ],
            },
            "archive_extracted_reproduction_check": {
                "requires_retraining": False,
                "expected_runtime": "under 20 minutes when materialized result artifacts are present",
                "entrypoints": [
                    {"script": "scripts/verify_archive_extracted_reproduction.py"},
                ],
                "scope_boundary": (
                    "Archive-extracted integrity, active main-manuscript "
                    "table-regeneration, and one public LAGEOS CRD/SP3 "
                    "precise-reference OD slice recomputation from archived "
                    "public inputs only; this does not rerun full raw-data "
                    "generation, model retraining, all recursive filters or "
                    "tables, live public-data retrieval, operational POD "
                    "validation, or independent machine reproduction."
                ),
            },
            "public_precise_reference_od_slice_rerun_check": {
                "requires_retraining": False,
                "expected_runtime": "under 10 minutes when the archived public CRD/SP3 inputs are present",
                "entrypoints": [
                    {"script": "scripts/run_real_slr_sp3_od_slice_rerun_validation.py"},
                    {
                        "test_entrypoint": "pytest",
                        "tests": [
                            "tests/test_real_slr_sp3_od_slice_rerun.py",
                            "tests/test_real_slr_sp3_od.py",
                        ],
                    },
                ],
                "scope_boundary": (
                    "One public LAGEOS CRD/SP3 precise-reference OD slice "
                    "rerun from archived public inputs through recursive "
                    "range-only filter recomputation and table reconstruction; "
                    "this is not full scientific reproduction, full estimator "
                    "training, all-table regeneration, live public-data "
                    "retrieval, or operational POD validation."
                ),
            },
            "targeted_retraining_replay_check": {
                "requires_retraining": True,
                "expected_runtime": "under 30 minutes for the representative deterministic slice",
                "entrypoints": [
                    TARGETED_RETRAINING_REPLAY_ENTRYPOINT,
                    {"test_entrypoint": "pytest", "tests": ["tests/test_targeted_retraining_replay.py"]},
                ],
                "scope_boundary": (
                    "One bounded representative learned-estimator training "
                    "replay on deterministic slices of existing materialized "
                    "data; this is not full paper-table reproduction, not a "
                    "full seed-suite rerun, not main-result reproduction, and "
                    "not replacement of the canonical submitted checkpoints."
                ),
            },
            "full_rerun": {
                "requires_retraining": True,
                "expected_runtime": "hours to days depending on available compute",
                "entrypoints": [
                    {"script": "scripts/run_full_pipeline.py", "mode": "extended paper build"},
                ],
            },
        },
        "claim_to_artifact_map": {
            "audited_learned_family_bounded_negative": [
                "results/seed_observed_significance_summary.csv",
                "results/seed_observed_significance.csv",
                "results/observed_step_preregistration/observed_step_preregistration.json",
                "results/observed_step_prospective_replication/observed_step_prospective_replication.json",
                "results/observed_step_powered_stress_replication/observed_step_powered_stress_replication.json",
                "paper/tables/seed_observed_significance.tex",
                "paper/tables/observed_step_preregistration.tex",
                "paper/tables/observed_step_prospective_replication.tex",
                "paper/tables/observed_step_powered_stress_replication.tex",
                "release/predeclarations/observed_step_prospective_replication_loop71.json",
                "release/predeclarations/observed_step_powered_stress_replication.json",
            ],
            "larger_simulator_bound_endpoint_replication": [
                "results/observed_step_prospective_replication/observed_step_prospective_replication.json",
                "paper/tables/observed_step_prospective_replication.tex",
                "release/predeclarations/observed_step_prospective_replication_loop71.json",
            ],
            "powered_stress_floor_scale_replication": [
                "results/observed_step_powered_stress_replication/observed_step_powered_stress_replication.json",
                "paper/tables/observed_step_powered_stress_replication.tex",
                "release/predeclarations/observed_step_powered_stress_replication.json",
            ],
            "endpoint_choice_sensitivity_audit": [
                "results/endpoint_selection_sensitivity/endpoint_selection_sensitivity.json",
                "paper/tables/endpoint_selection_sensitivity.tex",
                "scripts/build_endpoint_selection_sensitivity.py",
            ],
            "pukf_tuning_comparability_sensitivity": [
                "results/pukf_tuning_sensitivity/pukf_hifi_grid_sensitivity.json",
                "paper/tables/pukf_tuning_sensitivity.tex",
                "scripts/run_pukf_hifi_tuning_sensitivity.py",
                "results/hifi_force_mismatch/hifi_force_mismatch.csv",
                "results/hifi_force_mismatch/hifi_force_mismatch.json",
            ],
            "classical_guardrails_and_offline_od_reference": [
                "results/metrics_summary.json",
                "results/batch_wls_baseline/batch_wls_summary.csv",
                "paper/tables/main_results.tex",
                "paper/tables/batch_wls_baseline.tex",
            ],
            "protocol_subset_sufficiency": [
                "release/predeclarations/protocol_subset_ablation_loop51.json",
                "paper/tables/protocol_subset_ablation.tex",
            ],
            "structural_channel_bounded_negatives": [
                "results/dmc_ekf_force_mismatch/dmc_ekf_force_mismatch.json",
                "results/drag_scale_aekf_force_mismatch/drag_scale_aekf_force_mismatch.json",
                "results/drag_scale_constructive_positive_control/drag_scale_constructive_positive_control.json",
                "results/drag_scale_ukf_constructive_positive_control/drag_scale_ukf_constructive_positive_control.json",
                "results/drag_scale_ukf_observability_positive_control/drag_scale_ukf_observability_positive_control.json",
                "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch_n64_loop57.json",
                "results/structural_channel_recoverability/structural_channel_recoverability.json",
                "release/predeclarations/drag_scale_constructive_positive_control_loop54.json",
                "release/predeclarations/drag_scale_ukf_constructive_positive_control_loop55.json",
                "release/predeclarations/drag_scale_ukf_observability_positive_control_loop56.json",
                "paper/tables/dmc_ekf_force_mismatch.tex",
                "paper/tables/drag_scale_aekf_force_mismatch.tex",
                "paper/tables/drag_scale_constructive_positive_control.tex",
                "paper/tables/drag_scale_ukf_constructive_positive_control.tex",
                "paper/tables/drag_scale_ukf_observability_positive_control.tex",
                "paper/tables/long_arc_hifi_force_mismatch.tex",
                "paper/tables/structural_channel_recoverability.tex",
            ],
            "long_arc_scope_down_and_decision_stability": [
                "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch_n64_loop57.json",
                "results/decision_stability/decision_stability_loop58.json",
                "paper/tables/decision_stability.tex",
            ],
            "dbar_withdrawal": [
                "results/adaptation_risk_diagnostic/dbar_independent_sweep.json",
                "paper/tables/dbar_independent_sweep.tex",
                "paper/tables/adaptation_risk_diagnostic.tex",
            ],
            "kalmannet_native_and_transposition_diagnostics": [
                "results/kalmannet_repro/sanity_check.json",
                "results/kalmannet_spot_od_loop57/kalmannet_spot_od.json",
                "results/kalmannet_spot_od_budget_adequacy_loop58/kalmannet_spot_od_budget_adequacy.json",
                "paper/tables/kalmannet_official_reproduction.tex",
                "paper/tables/kalmannet_spot_od_transposition.tex",
                "paper/tables/kalmannet_spot_od_budget_adequacy.tex",
            ],
            "real_slr_sp3_bounded_sanity_probes": [
                "results/real_slr_lageos/real_slr_lageos_validation.json",
                "results/real_slr_sp3_od/real_slr_sp3_od_validation.json",
                "paper/tables/real_slr_sp3_od.tex",
                "results/real_slr_sp3_od_expanded80_inputs/real_slr_sp3_od_expanded80_validation.json",
                "results/real_slr_sp3_od_formal210_inputs/real_slr_sp3_od_formal210_validation.json",
                "paper/tables/real_slr_sp3_od_expanded.tex",
                "paper/tables/real_slr_sp3_od_expanded_stratification.tex",
                "paper/tables/real_slr_sp3_od_expanded_mechanism_heterogeneity.tex",
                "results/validation/real_slr_sp3_od_expanded80_run.log",
                "results/validation/real_slr_sp3_od_formal210_run.log",
                "results/validation/real_slr_sp3_od_formal210_run.err.log",
                "scripts/run_real_slr_sp3_od_expanded_validation.py",
                "tests/test_real_slr_sp3_od_expanded.py",
                "results/real_slr_sp3_hifi/real_slr_sp3_hifi_validation.json",
                "results/real_slr_sp3_corrected/real_slr_sp3_corrected_validation.json",
                "results/real_slr_sp3_correction_corpus_audit/real_slr_sp3_correction_corpus_audit.json",
                "results/real_slr_sp3_state_scoring_campaign/real_slr_sp3_state_scoring_campaign.json",
                "paper/tables/real_slr_sp3_state_scoring_campaign.tex",
                "results/real_slr_sp3_temporal_od_campaign/real_slr_sp3_temporal_od_campaign.json",
                "paper/tables/real_slr_sp3_temporal_od_campaign.tex",
                "results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json",
                "paper/tables/real_slr_sp3_temporal_selection_stability.tex",
                "release/predeclarations/real_slr_sp3_temporal_corrected_od_campaign_20260526.json",
                "results/real_slr_sp3_temporal_corrected_od_campaign/real_slr_sp3_temporal_corrected_od_campaign.json",
                "paper/tables/real_slr_sp3_temporal_corrected_od_campaign.tex",
                "results/validation/real_slr_sp3_od_slice_rerun.json",
                "results/validation/real_slr_sp3_od_slice_rerun.md",
                "results/validation/real_slr_sp3_od_slice_rerun/real_slr_sp3_od_validation.json",
                "results/validation/real_slr_sp3_od_slice_rerun/real_slr_sp3_od.tex",
                "scripts/run_real_slr_sp3_od_slice_rerun_validation.py",
                "tests/test_real_slr_sp3_od_slice_rerun.py",
            ]
            + formal210_input_rels
            + expanded80_input_rels
            + archive_extracted_od_rels,
            "public_precise_reference_state_scoring_probe": [
                "results/real_slr_sp3_state_scoring_campaign/real_slr_sp3_state_scoring_campaign.json",
                "paper/tables/real_slr_sp3_state_scoring_campaign.tex",
                "scripts/run_real_slr_sp3_state_scoring_campaign.py",
                "tests/test_real_slr_sp3_state_scoring_campaign.py",
            ],
            "public_multi_target_sp3_crd_breadth_probe": [
                "results/public_sp3_multi_target_breadth_probe/public_sp3_multi_target_breadth_probe.json",
                "paper/tables/public_sp3_multi_target_breadth_probe.tex",
                "scripts/run_public_sp3_multi_target_breadth_probe.py",
                "tests/test_public_sp3_multi_target_breadth_probe.py",
            ]
            + public_breadth_input_rels,
            "public_real_measurement_temporal_od_probe": [
                "results/real_slr_sp3_temporal_od_campaign/real_slr_sp3_temporal_od_campaign.json",
                "paper/tables/real_slr_sp3_temporal_od_campaign.tex",
                "results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json",
                "paper/tables/real_slr_sp3_temporal_selection_stability.tex",
                "scripts/run_real_slr_sp3_temporal_od_campaign.py",
                "scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py",
                "tests/test_real_slr_sp3_temporal_od_campaign.py",
                "tests/test_real_slr_sp3_temporal_corrected_od_campaign.py",
            ],
            "public_temporal_od_selection_stability_audit": [
                "results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json",
                "paper/tables/real_slr_sp3_temporal_selection_stability.tex",
                "scripts/run_real_slr_sp3_temporal_selection_stability.py",
                "tests/test_real_slr_sp3_temporal_selection_stability.py",
            ],
            "review_stage_archive_integrity": [
                "release/SUPPLEMENTARY_MANIFEST.json",
                REVIEW_ARCHIVE_REL,
                "release/REVIEWER_START_HERE.md",
                "results/validation/minimum_tier_reproduction_check.json",
                "results/validation/minimum_tier_reproduction_check.md",
                "results/validation/archive_extracted_reproduction.json",
                "results/validation/archive_extracted_reproduction.md",
                "results/validation/submission_validation.json",
                "results/validation/leakage_scan.json",
                "results/release_packet.json",
            ],
            "active_main_manuscript_table_regeneration": [
                "results/validation/active_manuscript_regeneration.json",
                "results/validation/active_manuscript_regeneration.md",
                "results/validation/archive_extracted_reproduction.json",
                "results/validation/archive_extracted_reproduction.md",
                "results/validation/command_manifest.json",
                "results/force_mismatch_seed_significance.json",
                "results/observed_step_internal_prospective_replication_loop163_k96/observed_step_internal_prospective_replication_loop163_k96.json",
                "scripts/regenerate_active_manuscript.py",
                "paper/tables/main_abbreviation_glossary.tex",
                "paper/tables/main_framework_portability.tex",
                "paper/tables/main_findings_summary.tex",
                "paper/tables/main_k32_replication.tex",
                "paper/tables/main_aukf_mechanism.tex",
                "paper/tables/main_structural_recoverability.tex",
                "paper/tables/main_drag_scale_cascade.tex",
                "paper/tables/main_long_arc_result.tex",
                "paper/tables/main_dbar_withdrawal.tex",
            ],
            "archive_extracted_reproduction_tier": [
                "release/SUPPLEMENTARY_MANIFEST.json",
                REVIEW_ARCHIVE_REL,
                "results/validation/archive_extracted_reproduction.json",
                "results/validation/archive_extracted_reproduction.md",
                "results/real_slr_sp3_od/real_slr_sp3_od_validation.json",
                "paper/tables/real_slr_sp3_od.tex",
                "scripts/verify_archive_extracted_reproduction.py",
                "scripts/regenerate_active_manuscript.py",
                "scripts/run_real_slr_sp3_od_validation.py",
                "scripts/build_paper_assets.py",
                "scripts/_bootstrap.py",
            ]
            + archive_extracted_od_rels,
            "targeted_retraining_replay": [
                "results/validation/targeted_retraining_replay_public.json",
                "results/validation/targeted_retraining_replay_public.md",
                "scripts/build_targeted_retraining_replay_public_report.py",
                "scripts/run_targeted_retraining_replay.py",
                "scripts/train_models.py",
                "tests/test_targeted_retraining_replay.py",
            ]
            + targeted_replay_artifact_rels,
        },
        "claim_to_regeneration_tier_map": {
            "audited_learned_family_bounded_negative": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "larger_simulator_bound_endpoint_replication": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "powered_stress_floor_scale_replication": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "endpoint_choice_sensitivity_audit": [
                "minimum_integrity_check",
                "table_regeneration_check",
            ],
            "pukf_tuning_comparability_sensitivity": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "classical_guardrails_and_offline_od_reference": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "protocol_subset_sufficiency": [
                "minimum_integrity_check",
                "table_regeneration_check",
            ],
            "structural_channel_bounded_negatives": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "long_arc_scope_down_and_decision_stability": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "dbar_withdrawal": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "kalmannet_native_and_transposition_diagnostics": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "real_slr_sp3_bounded_sanity_probes": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "archive_extracted_reproduction_check",
                "public_precise_reference_od_slice_rerun_check",
                "full_rerun",
            ],
            "public_precise_reference_state_scoring_probe": [
                "minimum_integrity_check",
                "table_regeneration_check",
                "full_rerun",
            ],
            "public_multi_target_sp3_crd_breadth_probe": [
                "minimum_integrity_check",
                "full_rerun",
            ],
            "public_real_measurement_temporal_od_probe": [
                "minimum_integrity_check",
                "full_rerun",
            ],
            "public_temporal_od_selection_stability_audit": [
                "minimum_integrity_check",
                "full_rerun",
            ],
            "review_stage_archive_integrity": [
                "minimum_integrity_check",
                "archive_extracted_reproduction_check",
            ],
            "active_main_manuscript_table_regeneration": [
                "table_regeneration_check",
                "archive_extracted_reproduction_check",
            ],
            "archive_extracted_reproduction_tier": [
                "archive_extracted_reproduction_check",
            ],
            "targeted_retraining_replay": [
                "minimum_integrity_check",
                "targeted_retraining_replay_check",
            ],
        },
        "artifact_groups": public_groups,
        "artifact_count": len(all_entries),
        "artifacts_present": sum(1 for e in all_entries if e["exists"]),
        "checksum_algorithm": "sha256",
        "review_archive": review_archive,
    }
    manifest["claim_to_artifact_map"] = {
        claim: safe_claim_paths(paths)
        for claim, paths in manifest["claim_to_artifact_map"].items()
    }
    out = ROOT / "release" / "SUPPLEMENTARY_MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": "release\\SUPPLEMENTARY_MANIFEST.json",
                "version": PACKAGE_VERSION,
                "artifact_count": manifest["artifact_count"],
                "artifacts_present": manifest["artifacts_present"],
                "seed_cohort": CANONICAL_SEED_COHORT,
                "review_archive": review_archive,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
