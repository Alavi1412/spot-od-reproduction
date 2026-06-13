from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validation_tuned_enkf_comparator_smoke_schema_and_boundaries(tmp_path: Path) -> None:
    out_dir = tmp_path / "validation_tuned_enkf"
    tex_path = out_dir / "validation_tuned_enkf_comparator.tex"
    cmd = [
        sys.executable,
        "scripts/run_validation_tuned_enkf_comparator.py",
        "--trajectory-limit",
        "2",
        "--bootstrap-samples",
        "20",
        "--output-dir",
        str(out_dir),
        "--tex-output",
        str(tex_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, text=True, capture_output=True, timeout=240)

    json_path = out_dir / "validation_tuned_enkf_comparator.json"
    csv_path = out_dir / "validation_tuned_enkf_comparator.csv"
    assert json_path.exists()
    assert csv_path.exists()
    assert tex_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "validation_tuned_enkf_comparator_v1"
    assert payload["candidate_grid"] == [1.0, 1.03, 1.06, 1.1]
    assert payload["selected_inflation"] in payload["candidate_grid"]
    assert payload["selection_boundary"]["test_or_force_mismatch_data_used_for_selection"] is False
    assert payload["selection_boundary"]["baseline_priors_used_for_selection"] is False

    selection_inputs = payload["selection_boundary"]["selection_input_splits"]
    assert selection_inputs == ["val", "stress_val"]
    assert all(not split.endswith("_test") for split in selection_inputs)
    assert "force_model_mismatch_test" not in selection_inputs
    assert "test" not in selection_inputs

    for candidate in payload["validation_selection"]["candidate_scores"]:
        assert set(candidate["split_scores"]) == {"val", "stress_val"}
        assert candidate["inflation"] in payload["candidate_grid"]
        assert candidate["weighted_finite_trajectory_count"] > 0

    assert set(payload["test_split_results"]) == {
        "force_model_mismatch_test",
        "test",
        "stress_test",
    }
    primary = payload["test_split_results"]["force_model_mismatch_test"]
    assert "paired_enkf_vs_other" in primary
    assert "EKF" in primary["paired_enkf_vs_other"]
    assert primary["all_method_paired_n"] > 0

    tex = tex_path.read_text(encoding="utf-8")
    assert r"\label{tab:validation_tuned_enkf_comparator}" in tex
