from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from scripts.build_supplementary_manifest import ARTIFACT_GROUPS
import scripts.regenerate_active_manuscript as regen
from scripts.regenerate_active_manuscript import ROOT, active_artifacts
from scripts.verify_archive_extracted_reproduction import (
    OD_CANONICAL_DIR_REL,
    OD_RERUN_DIR_REL,
    PUBLIC_OD_RMSE_ABS_TOLERANCE_M,
    PUBLIC_OD_TABLE_TARGETED_ABS_TOLERANCE,
    REVIEW_ALIAS_RESTORE_SOURCES,
    compare_public_od_claim_summaries,
    compare_public_od_table_text,
    manifest_entries,
    prepare_archive_extracted_od_rerun_directory,
    python_script_from_command,
    restore_review_archive_aliases,
    review_archive_path,
    safe_public_input_name,
    safe_member_path,
    summarize_active_regeneration_failures,
)


def test_active_text_artifact_check_only_passes_normalized_line_endings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regen, "ROOT", tmp_path)
    target = tmp_path / "paper" / "tables" / "line_endings.tex"
    generated = tmp_path / "generated" / "line_endings.tex"
    target.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    target.write_bytes(b"alpha\r\nbeta\r\n")
    generated.write_bytes(b"alpha\nbeta\n")

    result = regen.compare_or_update_artifact(
        path="paper/tables/line_endings.tex",
        generated_temp=generated,
        source_evidence=[],
        check_only=True,
        command="test",
        builder="test_builder",
    )

    assert result["status"] == "pass"
    assert result["byte_match"] is False
    assert result["text_normalized_match"] is True
    assert result["comparison_mode"] == "normalized_text"
    assert result["before_sha256"] != result["generated_sha256"]


def test_active_text_artifact_check_only_rejects_actual_text_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regen, "ROOT", tmp_path)
    target = tmp_path / "paper" / "tables" / "changed.tex"
    generated = tmp_path / "generated" / "changed.tex"
    target.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    generated.write_text("alpha\ngamma\n", encoding="utf-8")

    result = regen.compare_or_update_artifact(
        path="paper/tables/changed.tex",
        generated_temp=generated,
        source_evidence=[],
        check_only=True,
        command="test",
        builder="test_builder",
    )

    assert result["status"] == "mismatch"
    assert result["text_normalized_match"] is False
    assert result["difference"]["first_difference"]["line"] == 2


def _write_rgb_png(
    path: Path,
    pixels: list[tuple[int, int, int]],
    *,
    compress_level: int = 6,
    text: str | None = None,
) -> None:
    from PIL import Image, PngImagePlugin

    image = Image.new("RGB", (2, 2))
    image.putdata(pixels)
    pnginfo = PngImagePlugin.PngInfo()
    if text is not None:
        pnginfo.add_text("test-note", text)
    image.save(path, format="PNG", compress_level=compress_level, pnginfo=pnginfo)


def _write_visual_test_png(path: Path, *, variant: str = "reference") -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 12, 186, 108), outline=(40, 40, 40), width=2)
    draw.line((24, 92, 176, 28), fill=(25, 110, 180), width=3)
    draw.line((24, 72, 176, 52), fill=(210, 90, 50), width=3)
    draw.rectangle((50, 40, 82, 90), fill=(245, 166, 35))
    draw.rectangle((104, 26, 136, 90), fill=(80, 160, 110))
    if variant == "content_drift":
        draw.rectangle((18, 16, 182, 104), fill=(10, 10, 10))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def test_active_png_check_only_accepts_encoding_drift_with_same_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regen, "ROOT", tmp_path)
    target = tmp_path / "paper" / "figures" / "same_pixels.png"
    generated = tmp_path / "generated" / "same_pixels.png"
    target.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    pixels = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
    ]
    _write_rgb_png(target, pixels, compress_level=0, text="reference")
    _write_rgb_png(generated, pixels, compress_level=9, text="regenerated")
    assert target.read_bytes() != generated.read_bytes()

    result = regen.compare_or_update_artifact(
        path="paper/figures/same_pixels.png",
        generated_temp=generated,
        source_evidence=[],
        check_only=True,
        command="test",
        builder="test_builder",
    )

    assert result["status"] == "pass"
    assert result["byte_match"] is False
    assert result["comparison_mode"] == regen.PNG_DECODED_COMPARISON_MODE
    assert result["image_content_match"] is True
    assert result["image_comparison"]["status"] == "pass"
    assert result["image_comparison"]["decision"] == "decoded_pixel_match"
    assert result["image_comparison"]["pixel_hash_match"] is True
    assert result["before_sha256"] != result["generated_sha256"]


def test_active_aukf_png_check_only_accepts_scoped_visual_render_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regen, "ROOT", tmp_path)
    target = tmp_path / "paper" / "figures" / "aukf_r_inflation_mechanism.png"
    generated = tmp_path / "generated" / "aukf_r_inflation_mechanism.png"
    _write_visual_test_png(target)
    target_image = target.read_bytes()

    from PIL import Image

    with Image.open(target) as image:
        resized = image.resize((199, 120), resample=Image.Resampling.BICUBIC)
        generated.parent.mkdir(parents=True, exist_ok=True)
        resized.save(generated, format="PNG")
    assert target_image != generated.read_bytes()

    result = regen.compare_or_update_artifact(
        path="paper/figures/aukf_r_inflation_mechanism.png",
        generated_temp=generated,
        source_evidence=[],
        check_only=True,
        command="test",
        builder="test_builder",
    )

    assert result["status"] == "pass"
    assert result["byte_match"] is False
    assert result["comparison_mode"] == regen.PNG_VISUAL_COMPARISON_MODE
    assert result["image_content_match"] is True
    assert result["image_comparison"]["decision"] == "visual_metric_match"
    assert result["image_comparison"]["visual_comparison_allowed"] is True
    assert result["image_comparison"]["visual"]["match"] is True


def test_active_png_check_only_rejects_decoded_pixel_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regen, "ROOT", tmp_path)
    target = tmp_path / "paper" / "figures" / "pixel_drift.png"
    generated = tmp_path / "generated" / "pixel_drift.png"
    target.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    _write_rgb_png(
        target,
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)],
    )
    _write_rgb_png(
        generated,
        [(255, 0, 0), (0, 255, 0), (0, 0, 0), (255, 255, 255)],
    )

    result = regen.compare_or_update_artifact(
        path="paper/figures/pixel_drift.png",
        generated_temp=generated,
        source_evidence=[],
        check_only=True,
        command="test",
        builder="test_builder",
    )

    assert result["status"] == "mismatch"
    assert result["exit_code"] == 1
    assert result["byte_match"] is False
    assert result["comparison_mode"] == regen.PNG_DECODED_COMPARISON_MODE
    assert result["image_content_match"] is False
    assert result["image_comparison"]["status"] == "fail"
    assert result["image_comparison"]["size_match"] is True
    assert result["image_comparison"]["pixel_hash_match"] is False
    assert result["image_comparison"]["visual_comparison_allowed"] is False


def test_active_aukf_png_check_only_rejects_visual_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regen, "ROOT", tmp_path)
    target = tmp_path / "paper" / "figures" / "aukf_r_inflation_mechanism.png"
    generated = tmp_path / "generated" / "aukf_r_inflation_mechanism.png"
    _write_visual_test_png(target)
    _write_visual_test_png(generated, variant="content_drift")

    result = regen.compare_or_update_artifact(
        path="paper/figures/aukf_r_inflation_mechanism.png",
        generated_temp=generated,
        source_evidence=[],
        check_only=True,
        command="test",
        builder="test_builder",
    )

    assert result["status"] == "mismatch"
    assert result["byte_match"] is False
    assert result["comparison_mode"] == regen.PNG_VISUAL_COMPARISON_MODE
    assert result["image_content_match"] is False
    assert result["image_comparison"]["decision"] == "visual_metric_mismatch"
    assert result["image_comparison"]["visual"]["match"] is False


def test_active_non_image_binary_check_only_remains_byte_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(regen, "ROOT", tmp_path)
    target = tmp_path / "results" / "payload.npz"
    generated = tmp_path / "generated" / "payload.npz"
    target.parent.mkdir(parents=True)
    generated.parent.mkdir(parents=True)
    target.write_bytes(b"binary-reference")
    generated.write_bytes(b"binary-regenerated")

    result = regen.compare_or_update_artifact(
        path="results/payload.npz",
        generated_temp=generated,
        source_evidence=[],
        check_only=True,
        command="test",
        builder="test_builder",
    )

    assert result["status"] == "mismatch"
    assert result["byte_match"] is False
    assert result["comparison_mode"] == "byte_sha256"
    assert "image_comparison" not in result


def test_active_figure_renderer_mapping_falls_back_when_report_omits_aukf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeLoader:
        def exec_module(self, module: types.SimpleNamespace) -> None:
            def render_publication_figures(
                *, results_dir: Path, paper_fig_dir: Path, report_path: Path
            ) -> dict:
                return {
                    "figures": {},
                    "render_errors": {
                        "aukf_r_inflation_mechanism": "not registered in bulk report"
                    },
                }

            def _render_aukf_r_inflation_mechanism(
                results_dir: Path, paper_fig_dir: Path
            ) -> dict:
                path = paper_fig_dir / "aukf_r_inflation_mechanism.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"generated figure bytes")
                return {"path": str(path), "bytes": path.stat().st_size}

            module.render_publication_figures = render_publication_figures
            module._render_aukf_r_inflation_mechanism = (
                _render_aukf_r_inflation_mechanism
            )

    fake_spec = types.SimpleNamespace(loader=FakeLoader())
    monkeypatch.setattr(
        regen.importlib.util,
        "spec_from_file_location",
        lambda name, script: fake_spec,
    )
    monkeypatch.setattr(
        regen.importlib.util,
        "module_from_spec",
        lambda spec: types.SimpleNamespace(),
    )

    records, blockers = regen.render_figures_to(
        tmp_path,
        ["paper/figures/aukf_r_inflation_mechanism.png"],
    )

    assert blockers == []
    assert records[0]["path"] == "paper/figures/aukf_r_inflation_mechanism.png"
    assert records[0]["builder"].endswith("::_render_aukf_r_inflation_mechanism")


def test_active_main_artifacts_are_indexed_for_review_archive() -> None:
    active = active_artifacts()
    active_paths = {
        row["path"].replace("\\", "/")
        for row in active["tables"] + active["figures"]
    }
    manifest_paths = {
        path.replace("\\", "/")
        for paths in ARTIFACT_GROUPS.values()
        for path in paths
    }

    assert "paper/tables/main_findings_summary.tex" in active_paths
    assert active_paths <= manifest_paths


def test_graph_anchor_pair_gate_active_registration_and_generators(tmp_path: Path) -> None:
    active = active_artifacts()
    active_paths = {
        row["path"].replace("\\", "/")
        for row in active["tables"] + active["figures"]
    }
    table_path = "paper/tables/graph_anchor_pair_gate_poc.tex"
    figure_path = "paper/figures/graph_anchor_pair_gate_seed_sweep_aggregate.png"

    assert table_path in active_paths
    assert figure_path in active_paths
    assert regen.TABLE_BUILDERS[table_path] == "build_graph_anchor_pair_gate_poc_table"
    assert (
        regen.FIGURE_RENDERERS[figure_path]
        == "_render_graph_anchor_pair_gate_seed_sweep_aggregate"
    )

    import scripts.build_paper_assets as assets
    import scripts.render_publication_figures as figures

    table = assets.build_graph_anchor_pair_gate_poc_table()
    assert "\\label{tab:graph_anchor_pair_gate_poc}" in table
    assert "GraphAnchorPairGate: 9/10 scenario-seed row wins" in table

    info = figures._render_graph_anchor_pair_gate_seed_sweep_aggregate(
        ROOT / "results",
        tmp_path,
    )
    output = Path(info["path"])
    assert output.name == "graph_anchor_pair_gate_seed_sweep_aggregate.png"
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert info["width_px"] > 0
    assert info["height_px"] > 0


def test_adaptive_candidate_fusion_table_active_registration() -> None:
    active = active_artifacts()
    active_paths = {
        row["path"].replace("\\", "/")
        for row in active["tables"] + active["figures"]
    }
    table_path = "paper/tables/adaptive_candidate_fusion_full_training_poc.tex"

    assert table_path in active_paths
    assert (
        regen.TABLE_BUILDERS[table_path]
        == "build_adaptive_candidate_fusion_full_training_poc_table"
    )
    assert (
        "results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/"
        "adaptive_candidate_fusion_fixed_soft_training_campaign_summary.json"
    ) in regen.ACTIVE_TABLE_SOURCES[table_path]
    assert (
        "results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/"
        "summary.json"
    ) in regen.ACTIVE_TABLE_SOURCES[table_path]

    source_paths = {
        "results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.json",
        "results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.md",
        "results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv",
        "results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json",
        "results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.md",
        "results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.csv",
    }
    manifest_paths = {
        path.replace("\\", "/")
        for paths in ARTIFACT_GROUPS.values()
        for path in paths
    }
    assert table_path in manifest_paths
    assert source_paths <= manifest_paths

    manifest = json.loads((ROOT / "release" / "SUPPLEMENTARY_MANIFEST.json").read_text(encoding="utf-8"))
    claim_paths = set(
        manifest["claim_to_artifact_map"]["active_main_manuscript_table_regeneration"]
    )
    assert table_path in claim_paths
    assert source_paths <= claim_paths

    import scripts.build_paper_assets as assets

    table = assets.build_adaptive_candidate_fusion_full_training_poc_table()
    assert "\\label{tab:adaptive_candidate_fusion_full_training_poc}" in table
    assert "Centered fixed-soft full retraining" in table
    assert "Observed-mask fixed-soft full retraining" in table
    assert "global_scenario_portfolio_15seed" in table
    assert "540 candidate policies per candidate set" in table
    assert "25/30" in table
    assert "13/15" in table
    assert "95\\% CI [+1.83,+5.63]\\%" in table
    assert "seed-paired 13/15 wins, 95\\% CI [+2.05,+5.50]\\%" in table
    assert "process 14/15 wins, mean +6.16\\%, CI [+4.03,+8.06]\\%" in table
    assert "maneuver 11/15 wins, mean +1.43\\%, CI [-1.46,+4.05]\\%" in table
    assert "9/15 seed-paired wins, seed-paired CI [-1.16,+2.63]\\%" in table
    assert "nonlearned-only" in table
    assert "do not adjust for validation-policy search" in table
    assert "summary.md" in table
    assert "not public v1.2.1 release evidence" in table


def test_current_active_regeneration_count_matches_current_active_set() -> None:
    report_path = ROOT / "results" / "validation" / "active_manuscript_regeneration.json"
    if not report_path.exists():
        pytest.skip("active manuscript regeneration report has not been generated")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    active = active_artifacts()
    expected_count = len(active["tables"]) + len(active["figures"])
    validation = report["validation_results"]

    assert validation["artifact_count"] == expected_count
    assert validation["pass_count"] == expected_count
    assert validation["mismatch_count"] == 0
    assert validation["documented_blocker_count"] == 0


def test_manifest_entries_normalize_paths_and_preserve_group() -> None:
    manifest = {
        "artifact_groups": {
            "validation": [
                {
                    "path": "results\\validation\\archive_extracted_reproduction.json",
                    "exists": True,
                    "bytes": 12,
                    "sha256": "abc",
                }
            ]
        }
    }

    rows = manifest_entries(manifest)

    assert rows == [
        {
            "path": "results\\validation\\archive_extracted_reproduction.json",
            "exists": True,
            "bytes": 12,
            "sha256": "abc",
            "group": "validation",
            "norm_path": "results/validation/archive_extracted_reproduction.json",
        }
    ]


@pytest.mark.parametrize(
    "command,expected",
    [
        ("python -I scripts\\verify_archive_extracted_reproduction.py", "scripts/verify_archive_extracted_reproduction.py"),
        ("python scripts\\build_supplementary_manifest.py", "scripts/build_supplementary_manifest.py"),
        ("python -m pytest tests", None),
    ],
)
def test_python_script_from_command(command: str, expected: str | None) -> None:
    assert python_script_from_command(command) == expected


def test_safe_member_path_rejects_traversal() -> None:
    assert safe_member_path("paper/tables/main_results.tex").as_posix() == "paper/tables/main_results.tex"

    with pytest.raises(ValueError):
        safe_member_path("../paper/main.tex")


def test_safe_public_input_name_rejects_paths() -> None:
    assert safe_public_input_name("lageos1_20260505.np2") == "lageos1_20260505.np2"

    with pytest.raises(ValueError):
        safe_public_input_name("../lageos1_20260505.np2")

    with pytest.raises(ValueError):
        safe_public_input_name("nested/lageos1_20260505.np2")


def test_review_archive_alias_restore_report_excludes_raw_paths(tmp_path) -> None:
    for raw_rel in REVIEW_ALIAS_RESTORE_SOURCES:
        safe_rel = review_archive_path(raw_rel)
        safe_path = tmp_path.joinpath(*safe_rel.split("/"))
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(f"payload for {safe_rel}\n", encoding="utf-8")

    result = restore_review_archive_aliases(tmp_path)
    serialized = json.dumps(result, sort_keys=True)

    assert result["status"] == "pass"
    expected_restore_count = sum(
        1 for raw_rel in REVIEW_ALIAS_RESTORE_SOURCES if review_archive_path(raw_rel) != raw_rel
    )
    assert result["restored_count"] == expected_restore_count
    assert result["missing_alias_count"] == 0
    assert "_loop" not in serialized
    assert "loop42" not in serialized.lower()
    assert "loop91" not in serialized.lower()
    assert all("archive_alias" in row for row in result["restored"])
    assert all("source_digest_id" in row for row in result["restored"])


def test_prepare_archive_extracted_od_rerun_directory_copies_only_from_extracted_tree(tmp_path) -> None:
    source_dir = tmp_path / OD_CANONICAL_DIR_REL.replace("/", "\\")
    source_dir.mkdir(parents=True)
    for name in ("lageos1_20260505.np2", "nsgf.orb.lageos1.260509.v80.sp3.gz"):
        (source_dir / name).write_bytes(f"input:{name}".encode("utf-8"))
    (source_dir / "sp3_residual_calibrator.json").write_text("{}", encoding="utf-8")
    stale_dir = tmp_path / OD_RERUN_DIR_REL.replace("/", "\\")
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")

    copied, support = prepare_archive_extracted_od_rerun_directory(
        extracted_root=tmp_path,
        canonical={
            "input_digests": [
                {"archived_input_id": "lageos1_20260505.np2"},
                {"archived_input_id": "nsgf.orb.lageos1.260509.v80.sp3.gz"},
            ]
        },
    )

    assert len(copied) == 2
    assert len(support) == 1
    assert not (stale_dir / "stale.txt").exists()
    assert (stale_dir / "lageos1_20260505.np2").read_bytes() == b"input:lageos1_20260505.np2"
    assert (stale_dir / "nsgf.orb.lageos1.260509.v80.sp3.gz").is_file()


def _minimal_public_od_payload(mean: float = 334.72) -> dict:
    return {
        "schema_version": "real_slr_sp3_od_v1",
        "generated_utc": "ignored",
        "status": "completed",
        "targets": ["LAGEOS-1", "LAGEOS-2"],
        "num_arcs": 10,
        "num_arcs_completed": 10,
        "sp3_analysis_center": "NSGF",
        "sp3_week_product": "260509",
        "fixed_station_subset": ["HERL", "MATM", "WETL", "YARL"],
        "pooled_held_out_position_rmse_m": {
            "EKF": {"n_arcs": 10, "mean_arc_rms_m": 367.98, "median_arc_rms_m": 248.67, "arcs_best_of": 1},
            "UKF (fixed-noise)": {"n_arcs": 10, "mean_arc_rms_m": mean, "median_arc_rms_m": 234.04, "arcs_best_of": 1},
            "AUKF (adaptive)": {"n_arcs": 10, "mean_arc_rms_m": 341.33, "median_arc_rms_m": 296.13, "arcs_best_of": 3},
            "SP3-IC propagation": {"n_arcs": 10, "mean_arc_rms_m": 402.92, "median_arc_rms_m": 260.68, "arcs_best_of": 5},
        },
        "dbar_external_validation": {
            "n_arcs_scored": 10,
            "n_correct": 6,
            "classification_accuracy": 0.6,
            "confusion": {"true_fire": 0, "true_no_fire": 6, "false_fire": 2, "false_no_fire": 2},
            "sensitivity": 0.0,
            "specificity": 0.75,
            "n_counterproductive_arcs": 2,
            "n_non_counterproductive_arcs": 8,
            "no_information_baseline": {
                "majority_class": "no_fire (negative)",
                "majority_class_accuracy": 0.8,
                "accuracy_minus_majority": -0.2,
                "beats_majority": False,
            },
        },
    }


def test_archive_extracted_public_od_claim_comparison_ignores_timestamps() -> None:
    first = _minimal_public_od_payload()
    second = _minimal_public_od_payload()
    second["generated_utc"] = "also ignored"

    result = compare_public_od_claim_summaries(first, second)

    assert result["status"] == "pass"
    assert result["mismatch_count"] == 0


def test_archive_extracted_public_od_claim_comparison_tolerates_small_numeric_drift() -> None:
    result = compare_public_od_claim_summaries(
        _minimal_public_od_payload(mean=334.72),
        _minimal_public_od_payload(mean=334.76),
    )

    assert result["status"] == "pass"
    assert result["mismatch_count"] == 0
    assert result["tolerated_numeric_difference_count"] == 1
    assert result["tolerated_numeric_differences"][0]["abs_tolerance"] == (
        PUBLIC_OD_RMSE_ABS_TOLERANCE_M
    )


def test_archive_extracted_public_od_claim_comparison_reports_field_mismatch() -> None:
    result = compare_public_od_claim_summaries(
        _minimal_public_od_payload(),
        _minimal_public_od_payload(mean=335.0),
    )

    assert result["status"] == "fail"
    assert result["mismatches"][0]["field"] == (
        "pooled_held_out_position_rmse_m.UKF (fixed-noise).mean_arc_rms_m"
    )


def test_archive_extracted_public_od_claim_comparison_rejects_count_type_change() -> None:
    changed = _minimal_public_od_payload()
    changed["pooled_held_out_position_rmse_m"]["UKF (fixed-noise)"]["n_arcs"] = 10.0

    result = compare_public_od_claim_summaries(_minimal_public_od_payload(), changed)

    assert result["status"] == "fail"
    mismatch = result["mismatches"][0]
    assert mismatch["field"] == "pooled_held_out_position_rmse_m.UKF (fixed-noise).n_arcs"
    assert mismatch["expected_type"] == "int"
    assert mismatch["actual_type"] == "float"
    assert mismatch["abs_tolerance"] is None


def test_archive_extracted_public_od_claim_comparison_rejects_bool_type_change() -> None:
    changed = _minimal_public_od_payload()
    changed["dbar_external_validation"]["no_information_baseline"]["beats_majority"] = 0

    result = compare_public_od_claim_summaries(_minimal_public_od_payload(), changed)

    assert result["status"] == "fail"
    mismatch = result["mismatches"][0]
    assert mismatch["field"] == "dbar_external_validation.no_information_baseline.beats_majority"
    assert mismatch["expected_type"] == "bool"
    assert mismatch["actual_type"] == "int"
    assert mismatch["abs_tolerance"] is None


def _public_od_table_excerpt() -> str:
    return (
        "    EKF & 367.98 & 248.67 & 1/10 \\\\\n"
        "    UKF (fixed-noise) & 334.72 & 234.04 & 1/10 \\\\\n"
        "    AUKF (adaptive) & 341.33 & 296.13 & 3/10 \\\\\n"
        "  \\\\[2pt] {\\footnotesize \\textbf{Compact recursive-filter readout.} "
        "The paired EKF-minus-AUKF gap (positive favors AUKF) is mean 26.65~m, "
        "median 16.77~m, with deterministic 20,000-resample bootstrap 95\\% CI "
        "$[-43.58,108.48]$~m, which spans zero. The fixed-noise UKF remains "
        "slightly best by pooled mean (334.72~m versus 341.33~m for AUKF; "
        "UKF-minus-AUKF mean -6.61~m, 95\\% CI $[-55.13,26.24]$~m), so this "
        "is an underpowered real-measurement discriminative readout.}\n"
    )


def test_archive_extracted_public_od_table_comparison_tolerates_known_linux_drift() -> None:
    submitted = _public_od_table_excerpt()
    generated = (
        "    EKF & 367.98 & 248.67 & 1/10 \\\\\n"
        "    UKF (fixed-noise) & 334.76 & 234.15 & 1/10 \\\\\n"
        "    AUKF (adaptive) & 341.20 & 296.00 & 3/10 \\\\\n"
        "  \\\\[2pt] {\\footnotesize \\textbf{Compact recursive-filter readout.} "
        "The paired EKF-minus-AUKF gap (positive favors AUKF) is mean 26.70~m, "
        "median 16.80~m, with deterministic 20,000-resample bootstrap 95\\% CI "
        "$[-43.50,108.40]$~m, which spans zero. The fixed-noise UKF remains "
        "slightly best by pooled mean (334.76~m versus 341.20~m for AUKF; "
        "UKF-minus-AUKF mean -6.50~m, 95\\% CI $[-55.00,26.30]$~m), so this "
        "is an underpowered real-measurement discriminative readout.}\n"
    )

    result = compare_public_od_table_text(generated, submitted)

    assert result["status"] == "pass"
    assert result["matches_submitted_table"] is True
    assert result["byte_identical_after_normalization"] is False
    assert result["tolerated_numeric_difference_count"] == 13
    assert result["max_observed_abs_delta"] <= PUBLIC_OD_TABLE_TARGETED_ABS_TOLERANCE


def test_archive_extracted_public_od_table_comparison_tolerates_targeted_table_row_drift() -> None:
    submitted = "UKF (fixed-noise) & 334.72 & 234.04 & 1/10 \\\\\n"
    generated = "UKF (fixed-noise) & 334.76 & 234.15 & 1/10 \\\\\n"

    result = compare_public_od_table_text(generated, submitted)

    assert result["status"] == "pass"
    assert result["matches_submitted_table"] is True
    assert result["byte_identical_after_normalization"] is False
    assert result["tolerated_numeric_difference_count"] == 2
    assert result["max_observed_abs_delta"] <= PUBLIC_OD_TABLE_TARGETED_ABS_TOLERANCE


def test_archive_extracted_public_od_table_comparison_rejects_wording_change() -> None:
    submitted = "AUKF is best on 6/10 arcs.\n"
    generated = "AUKF is worst on 6/10 arcs.\n"

    result = compare_public_od_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["unchanged_outside_tolerated_fields"] is False


def test_archive_extracted_public_od_table_comparison_rejects_integer_count_drift() -> None:
    submitted = "AUKF is best on 6/10 arcs.\n"
    generated = "AUKF is best on 6/11 arcs.\n"

    result = compare_public_od_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["unchanged_outside_tolerated_fields"] is False


def test_archive_extracted_public_od_table_comparison_rejects_reff_threshold_drift() -> None:
    submitted = "The predeclared DBAR rule ($R_{\\mathrm{eff}}>1.5$) is fixed.\n"
    generated = "The predeclared DBAR rule ($R_{\\mathrm{eff}}>2.0$) is fixed.\n"

    result = compare_public_od_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["unchanged_outside_tolerated_fields"] is False


def test_archive_extracted_public_od_table_comparison_rejects_specificity_drift() -> None:
    submitted = "Agreement is reported; specificity 0.75 on the 8 non-counterproductive arcs.\n"
    generated = "Agreement is reported; specificity 0.30 on the 8 non-counterproductive arcs.\n"

    result = compare_public_od_table_text(generated, submitted)

    assert result["status"] == "fail"
    assert result["unchanged_outside_tolerated_fields"] is False


def test_active_regeneration_failure_summary_is_bounded_and_path_safe() -> None:
    repo_table_abs = str(ROOT / "paper" / "tables" / "main_findings_summary.tex")
    repo_missing_abs = str(ROOT / "results" / "missing.json")
    nested = {
        "status_evidence": {
            "active_artifacts": [
                {
                    "path": "paper/tables/main_findings_summary.tex",
                    "status": "mismatch",
                    "builder": "scripts/build_paper_assets.py::build_main_findings_summary_table",
                    "generated_temp": "C:/Users/example/AppData/Local/Temp/active_manuscript_regen_x/paper/tables/main_findings_summary.tex",
                    "before_sha256": "before",
                    "generated_sha256": "generated",
                    "after_sha256": "after",
                    "byte_match": False,
                    "text_normalized_match": False,
                    "comparison_mode": "byte_sha256",
                    "difference": {
                        "first_difference": {
                            "line": 3,
                            "current": (
                                "old from C:/Users/example/AppData/Local/Temp/"
                                "active_manuscript_regen_x/paper/tables/main_findings_summary.tex"
                            ),
                            "generated": f"new from {repo_table_abs}",
                        },
                        "unified_diff_head": [
                            f"- {repo_table_abs}",
                            (
                                "+ C:/Users/example/AppData/Local/Temp/"
                                "active_manuscript_regen_x/results/generated.json"
                            ),
                        ]
                        + [f"line {i}" for i in range(48)],
                    },
                    "source_artifacts": [
                        {
                            "path": "results/source.json",
                            "exists": True,
                            "sha256": "source",
                            "bytes": 12,
                        }
                    ],
                },
                {
                    "path": "paper/tables/main_long_arc_result.tex",
                    "status": "blocked",
                    "builder": "scripts/build_paper_assets.py::build_main_long_arc_result_table",
                    "explicit_blocker": (
                        f"Missing source artifacts: {repo_missing_abs}; staged under "
                        "C:/Users/example/AppData/Local/Temp/archive_extract_x/results/missing.json"
                    ),
                    "renderer": "scripts/render_publication_figures.py::_render_aukf_r_inflation_mechanism",
                    "renderer_error": "missing source artifact",
                    "figure_render_errors": {
                        "aukf_r_inflation_mechanism": (
                            "C:/Users/example/AppData/Local/Temp/"
                            "archive_extract_x/results/missing.json"
                        )
                    },
                    "source_artifacts": [
                        {
                            "path": "results/missing.json",
                            "exists": False,
                            "sha256": None,
                            "bytes": None,
                        }
                    ],
                },
                {"path": "paper/tables/main_aukf_mechanism.tex", "status": "pass"},
            ]
        }
    }

    result = summarize_active_regeneration_failures(nested)
    serialized = json.dumps(result, sort_keys=True)

    assert result["artifact_count"] == 3
    assert result["failed_artifact_count"] == 2
    assert result["mismatch_artifact_count"] == 1
    assert result["blocked_artifact_count"] == 1
    assert result["artifacts"][0]["first_text_difference"]["line"] == 3
    assert result["artifacts"][0]["byte_match"] is False
    assert result["artifacts"][0]["text_normalized_match"] is False
    assert result["artifacts"][0]["comparison_mode"] == "byte_sha256"
    assert len(result["artifacts"][0]["unified_diff_head"]) == 40
    assert result["artifacts"][1]["source_blockers"][0]["path"] == "results/missing.json"
    assert result["artifacts"][1]["renderer"].endswith(
        "::_render_aukf_r_inflation_mechanism"
    )
    assert result["artifacts"][1]["renderer_error"] == "missing source artifact"
    assert "[redacted temp path]/results/missing.json" in json.dumps(
        result["artifacts"][1]["figure_render_errors"], sort_keys=True
    )
    assert "active_manuscript_regen_x" not in serialized
    assert "archive_extract_x" not in serialized
    assert "AppData" not in serialized
    assert "C:/Users/example" not in serialized
    assert "GNN State Estimation" not in serialized
    assert "[repo-root]/paper/tables/main_findings_summary.tex" in serialized
    assert "[redacted temp path]/paper/tables/main_findings_summary.tex" in serialized
    assert "[redacted temp path]/results/missing.json" in serialized


def test_active_regeneration_failure_summary_redacts_failed_artifact_temp_path_with_spaces() -> None:
    nested = {
        "status_evidence": {
            "active_artifacts": [
                {
                    "path": (
                        "C:/Users/Jane Doe/AppData/Local/Temp/"
                        "archive_extracted_repro_x/paper/tables/main_findings_summary.tex"
                    ),
                    "status": "blocked",
                    "builder": "scripts/build_paper_assets.py::build_main_findings_summary_table",
                }
            ]
        }
    }

    result = summarize_active_regeneration_failures(nested)
    serialized = json.dumps(result, sort_keys=True)

    assert result["artifacts"][0]["path"] == (
        "[redacted temp path]/paper/tables/main_findings_summary.tex"
    )
    assert "Jane Doe" not in serialized
    assert "AppData" not in serialized


def test_active_regeneration_failure_summary_redacts_source_artifact_temp_path_with_spaces() -> None:
    nested = {
        "status_evidence": {
            "active_artifacts": [
                {
                    "path": "paper/tables/main_findings_summary.tex",
                    "status": "blocked",
                    "builder": "scripts/build_paper_assets.py::build_main_findings_summary_table",
                    "source_artifacts": [
                        {
                            "path": (
                                "C:/Users/Jane Doe/AppData/Local/Temp/"
                                "archive extracted repro x/results/missing.json"
                            ),
                            "exists": False,
                            "sha256": None,
                            "bytes": None,
                        }
                    ],
                }
            ]
        }
    }

    result = summarize_active_regeneration_failures(nested)
    serialized = json.dumps(result, sort_keys=True)

    assert result["artifacts"][0]["source_blockers"][0]["path"] == (
        "[redacted temp path]/results/missing.json"
    )
    assert "Jane Doe" not in serialized
    assert "AppData" not in serialized
    assert "archive extracted repro x" not in serialized


def test_active_regeneration_failure_summary_preserves_safe_relative_source_path() -> None:
    nested = {
        "status_evidence": {
            "active_artifacts": [
                {
                    "path": "paper/tables/main_long_arc_result.tex",
                    "status": "blocked",
                    "builder": "scripts/build_paper_assets.py::build_main_long_arc_result_table",
                    "source_artifacts": [
                        {
                            "path": "results/missing.json",
                            "exists": False,
                            "sha256": None,
                            "bytes": None,
                        }
                    ],
                }
            ]
        }
    }

    result = summarize_active_regeneration_failures(nested)

    assert result["artifacts"][0]["source_blockers"][0]["path"] == "results/missing.json"


def test_active_regeneration_failure_summary_redacts_repo_absolute_path_with_spaces() -> None:
    repo_abs = str(ROOT / "paper" / "tables" / "main_findings_summary.tex")
    nested = {
        "status_evidence": {
            "active_artifacts": [
                {
                    "path": repo_abs,
                    "status": "blocked",
                    "builder": "scripts/build_paper_assets.py::build_main_findings_summary_table",
                }
            ]
        }
    }

    result = summarize_active_regeneration_failures(nested)
    serialized = json.dumps(result, sort_keys=True)

    assert result["artifacts"][0]["path"] == (
        "[repo-root]/paper/tables/main_findings_summary.tex"
    )
    assert str(ROOT) not in serialized
    assert str(ROOT).replace("\\", "/") not in serialized
    assert "GNN State Estimation" not in serialized
