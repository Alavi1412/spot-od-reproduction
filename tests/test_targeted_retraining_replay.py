from pathlib import Path

from scripts.run_targeted_retraining_replay import REPORT_SCHEMA_VERSION, validate_report_schema


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_report_schema_accepts_minimal_valid_report():
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "targeted_learned_estimator_retraining_replay",
        "created_at_utc": "2026-05-22T00:00:00Z",
        "status": {"pass": True},
        "claim_boundary": "Targeted bounded replay only.",
        "predeclaration": {
            "path": "release/predeclarations/example.json",
            "sha256": "0" * 64,
        },
        "model": "KalmanNetGain",
        "seed": 86086,
        "device": {
            "accelerated_compute_required": True,
            "accelerated_compute_used": True,
        },
        "data": {},
        "training": {},
        "outputs": {},
        "criteria": {"training_step_returned_zero": True},
    }

    assert validate_report_schema(report) == []


def test_validate_report_schema_rejects_missing_and_non_boolean_criteria():
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "targeted_learned_estimator_retraining_replay",
        "status": {"pass": "yes"},
        "criteria": {"training_step_returned_zero": "true"},
    }

    errors = validate_report_schema(report)

    assert "status.pass must be boolean" in errors
    assert "all criteria values must be boolean" in errors
    assert any(error.startswith("missing top-level key:") for error in errors)


def test_manifest_source_indexes_targeted_retraining_replay():
    text = (REPO_ROOT / "scripts" / "build_supplementary_manifest.py").read_text(
        encoding="utf-8"
    )

    assert "targeted_retraining_replay" in text
    assert "targeted_retraining_replay_check" in text
    assert "results/validation/targeted_retraining_replay_public.json" in text
    assert "results/validation/targeted_retraining_replay_public.md" in text
    assert "release/predeclarations/targeted_curriculum_retraining_replay_20260525.json" in text
    assert "scripts/build_targeted_retraining_replay_public_report.py" in text
    assert "scripts/run_targeted_retraining_replay.py" in text
    assert "tests/test_targeted_retraining_replay.py" in text
    assert "targeted_replay_artifact_rels" in text
