from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import verify_v132_public_reproduction_package as verifier


LOCAL_RMSE = {
    "all_eval_non_development": 519.2821558058945,
    "fresh_extra": 706.7299439447123,
}
MEAN_RMSE = {
    "all_eval_non_development": 406.50913762826747,
    "fresh_extra": 409.06669847476724,
}
BEST_RETAINED_CI = {
    "all_eval_non_development": [11.45062980513122, 18.575159123708463],
    "fresh_extra": [5.394507848224321, 20.688404714449202],
}
LOCAL_GAIN = {
    "all_eval_non_development": 24.835317289797644,
    "fresh_extra": 45.32943778982349,
}
MEAN_GAIN = {
    "all_eval_non_development": 3.9832690946483287,
    "fresh_extra": 5.547619715108024,
}


def _tier(
    *,
    selector: float,
    best: float,
    gain: float,
    rows: int,
    wins: int,
    losses: int,
) -> dict[str, float | int]:
    return {
        "selector_observed_step_rmse_m": selector,
        "best_single_observed_step_rmse_m": best,
        "gain_vs_best_single_percent": gain,
        "rows": rows,
        "row_wins": wins,
        "row_losses": losses,
    }


def _graph_summary() -> dict[str, object]:
    return {
        "aggregate_tiers": {
            "all_eval_non_development": _tier(
                selector=390.31678478219936,
                best=459.5907664568724,
                gain=15.072970723221363,
                rows=230,
                wins=154,
                losses=76,
            ),
            "fresh_extra": _tier(
                selector=386.37323366223956,
                best=445.94330738322924,
                gain=13.358216781084483,
                rows=47,
                wins=29,
                losses=18,
            ),
        }
    }


def _reference_summary(values: dict[str, float]) -> dict[str, object]:
    return {
        "aggregate_tiers": {
            tier: {"selector_observed_step_rmse_m": rmse}
            for tier, rmse in values.items()
        }
    }


def _comparison_intervals() -> dict[str, object]:
    comparisons: dict[str, object] = {}
    graph = _graph_summary()
    for tier in ("all_eval_non_development", "fresh_extra"):
        graph_tier = graph["aggregate_tiers"][tier]  # type: ignore[index]
        comparisons[tier] = {
            "best_single_retained": {
                "reference_rmse_m": graph_tier["best_single_observed_step_rmse_m"],
                "gain_percent": graph_tier["gain_vs_best_single_percent"],
                "row_bootstrap_gain_percent_95ci": BEST_RETAINED_CI[tier],
            },
            "edge_only_local_residual_refine": {
                "reference_rmse_m": LOCAL_RMSE[tier],
                "gain_percent": LOCAL_GAIN[tier],
            },
            "edge_only_mean_residual_refine": {
                "reference_rmse_m": MEAN_RMSE[tier],
                "gain_percent": MEAN_GAIN[tier],
            },
        }
    return {"comparisons": comparisons}


def _metadata_text() -> str:
    return (
        f"{verifier.TAG} {verifier.PRIOR_VERSION_DOI} {verifier.CONCEPT_DOI} "
        "not public precise-reference validation not independent-machine reproduction "
        "not a full raw/training/all-filter rerun\n"
    )


def _base_main_files(repaired_payload: bytes) -> dict[str, str | bytes]:
    files: dict[str, str | bytes] = {}
    for member in verifier.REQUIRED_MAIN_MEMBERS:
        suffix = Path(member).suffix.lower()
        if member == verifier.REPAIRED_V131_ARCHIVE_MEMBER:
            files[member] = repaired_payload
        elif suffix in {".pdf", ".png", ".zip", ".pt"}:
            files[member] = b"fixture-binary"
        elif suffix == ".csv":
            files[member] = "field\nvalue\n"
        elif suffix == ".py":
            files[member] = "# fixture script\n"
        else:
            files[member] = _metadata_text()

    files[".zenodo.json"] = json.dumps(
        {
            "version": verifier.TAG,
            "related_identifiers": [
                {"scheme": "url", "relation": "isSupplementedBy", "identifier": verifier.REPO_URL},
                {"scheme": "url", "relation": "isSupplementedBy", "identifier": verifier.RELEASE_URL},
                {"scheme": "doi", "relation": "isNewVersionOf", "identifier": verifier.PRIOR_VERSION_DOI},
                {"scheme": "doi", "relation": "isVersionOf", "identifier": verifier.CONCEPT_DOI},
                {"scheme": "doi", "relation": "cites", "identifier": verifier.HISTORICAL_CITED_DOI},
            ],
            "notes": _metadata_text(),
        }
    )
    files["release/ZENODO_METADATA.json"] = json.dumps(
        {
            "metadata": {"version": verifier.TAG},
            "record": {
                "status": "pending_github_release_zenodo_import",
                "concept_doi": verifier.CONCEPT_DOI,
                "previous_version_doi": verifier.PRIOR_VERSION_DOI,
            },
            "github_release": {"tag": verifier.TAG, "url": verifier.RELEASE_URL},
        }
    )
    files["release/CITATION.cff"] = _metadata_text()
    files["release/README.md"] = f"Current release: `{verifier.TAG}`\n{_metadata_text()}"
    for member in verifier.V132_RELEASE_DOCS:
        files[member] = _metadata_text()

    files[verifier.v131.GRAPH_SUMMARY] = json.dumps(_graph_summary())
    files[verifier.v131.LOCAL_SUMMARY] = json.dumps(_reference_summary(LOCAL_RMSE))
    files[verifier.v131.MEAN_SUMMARY] = json.dumps(_reference_summary(MEAN_RMSE))
    files[verifier.v131.COMPARISON_INTERVALS] = json.dumps(_comparison_intervals())
    files[verifier.v131.TAIL_SUMMARY] = json.dumps(
        {
            "generated_from_saved_rows_only": True,
            "tiers": {
                "all_eval_non_development": {
                    "pooled_rmse_m": {"edge_only_attention": 390.31678478219936}
                }
            },
        }
    )
    files["scripts/_bootstrap.py"] = (
        "from __future__ import annotations\n"
        "import sys\n"
        "from pathlib import Path\n"
        "def ensure_src_on_path() -> None:\n"
        "    src_path = Path(__file__).resolve().parents[1] / 'src'\n"
        "    src_text = str(src_path)\n"
        "    if src_text not in sys.path:\n"
        "        sys.path.insert(0, src_text)\n"
    )
    files["scripts/__init__.py"] = '"""Fixture scripts package."""\n'
    files["src/gnn_state_estimation/__init__.py"] = '"""Fixture source package."""\n'
    files[verifier.v131.HELP_SMOKE_SCRIPT] = (
        "from __future__ import annotations\n"
        "try:\n"
        "    from _bootstrap import ensure_src_on_path\n"
        "except ModuleNotFoundError:\n"
        "    from scripts._bootstrap import ensure_src_on_path\n"
        "ensure_src_on_path()\n"
        "import argparse\n"
        "import gnn_state_estimation\n"
        "def build_parser() -> argparse.ArgumentParser:\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--source-glob', default='fixture')\n"
        "    return parser\n"
        "if __name__ == '__main__':\n"
        "    build_parser().parse_args()\n"
    )
    return files


def _base_training_files() -> dict[str, str | bytes]:
    files: dict[str, str | bytes] = {}
    for source_dir in verifier.EXPECTED_TRAINING_SOURCE_DIRS:
        for filename in verifier.TRAINING_ROOT_FILES:
            suffix = Path(filename).suffix.lower()
            payload: str | bytes = b"npz" if suffix == ".npz" else "{}\n"
            if suffix == ".csv":
                payload = "field\nvalue\n"
            files[f"{source_dir}/{filename}"] = payload
        for scenario in verifier.TRAINING_SCENARIOS:
            files[f"{source_dir}/{scenario}/adaptive_candidate_fusion_predictions.npz"] = b"npz"
            files[f"{source_dir}/{scenario}/adaptive_candidate_fusion_summary.json"] = "{}\n"

    files[verifier.TRAINING_MANIFEST_JSON] = json.dumps(
        {
            "release_tag": verifier.TAG,
            "release_url": verifier.RELEASE_URL,
            "previous_zenodo_version_doi": verifier.PRIOR_VERSION_DOI,
            "zenodo_concept_doi": verifier.CONCEPT_DOI,
            "checkpoints_omitted": True,
            "payload_file_count": verifier.EXPECTED_TRAINING_PAYLOAD_FILE_COUNT,
            "total_zip_file_count": verifier.EXPECTED_TRAINING_TOTAL_FILE_COUNT,
            "source_directory_count": len(verifier.EXPECTED_TRAINING_SOURCE_DIRS),
            "source_directories": list(verifier.EXPECTED_TRAINING_SOURCE_DIRS),
        }
    )
    files[verifier.TRAINING_MANIFEST_MD] = (
        "checkpoint-free retained candidate bundle; checkpoints are omitted; "
        "not full raw/training/all-filter reproduction\n"
    )
    return files


def _write_zip(tmp_path: Path, name: str, files: dict[str, str | bytes]) -> Path:
    archive = tmp_path / name
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member, payload in sorted(files.items()):
            zf.writestr(member, payload)
    return archive


def _valid_archives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    repaired_payload = b"repaired-v131-archive"
    monkeypatch.setattr(
        verifier,
        "REPAIRED_V131_ARCHIVE_SHA256",
        hashlib.sha256(repaired_payload).hexdigest(),
    )
    main_archive = _write_zip(tmp_path, "main.zip", _base_main_files(repaired_payload))
    training_archive = _write_zip(tmp_path, "training.zip", _base_training_files())
    return main_archive, training_archive


def test_minimal_valid_v132_fixture_passes_and_writes_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_archive, training_archive = _valid_archives(tmp_path, monkeypatch)

    result = verifier.build_result(str(main_archive), str(training_archive))

    assert result["status"] == "pass"
    assert result["main_package"]["checks"]["extracted_help_smoke"]["status"] == "pass"
    assert result["main_package"]["checks"]["metadata"]["status"] == "pass"
    assert result["training_inputs"]["checks"]["training_members"]["status"] == "pass"

    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    verifier.write_reports(result, json_out, md_out)
    assert json.loads(json_out.read_text(encoding="utf-8"))["status"] == "pass"
    assert "pending Zenodo GitHub import" in md_out.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "missing_member, expected_problem",
    [
        ("scripts/_bootstrap.py", "_bootstrap"),
        ("src/gnn_state_estimation/__init__.py", "gnn_state_estimation"),
    ],
)
def test_rejects_missing_runtime_import_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_member: str,
    expected_problem: str,
) -> None:
    repaired_payload = b"repaired-v131-archive"
    monkeypatch.setattr(
        verifier,
        "REPAIRED_V131_ARCHIVE_SHA256",
        hashlib.sha256(repaired_payload).hexdigest(),
    )
    main_files = _base_main_files(repaired_payload)
    main_files.pop(missing_member)
    main_archive = _write_zip(tmp_path, "main.zip", main_files)
    training_archive = _write_zip(tmp_path, "training.zip", _base_training_files())

    result = verifier.build_result(str(main_archive), str(training_archive))

    assert result["status"] == "fail"
    assert result["main_package"]["checks"]["required_members"]["status"] == "fail"
    smoke = result["main_package"]["checks"]["extracted_help_smoke"]
    assert smoke["status"] == "fail"
    assert expected_problem in smoke["stderr_tail"]


def test_rejects_invented_v132_doi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repaired_payload = b"repaired-v131-archive"
    monkeypatch.setattr(
        verifier,
        "REPAIRED_V131_ARCHIVE_SHA256",
        hashlib.sha256(repaired_payload).hexdigest(),
    )
    main_files = _base_main_files(repaired_payload)
    metadata = json.loads(str(main_files["release/ZENODO_METADATA.json"]))
    metadata["record"]["doi"] = "10.5281/zenodo.99999999"
    main_files["release/ZENODO_METADATA.json"] = json.dumps(metadata)
    main_archive = _write_zip(tmp_path, "main.zip", main_files)
    training_archive = _write_zip(tmp_path, "training.zip", _base_training_files())

    result = verifier.build_result(str(main_archive), str(training_archive))

    assert result["status"] == "fail"
    metadata_check = result["main_package"]["checks"]["metadata"]
    assert metadata_check["status"] == "fail"
    problems = {failure["problem"] for failure in metadata_check["failures"] if "problem" in failure}
    assert "v1.3.2 DOI must not be invented before Zenodo import" in problems


def test_rejects_missing_training_prediction_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_archive, _ = _valid_archives(tmp_path, monkeypatch)
    training_files = _base_training_files()
    first_dir = verifier.EXPECTED_TRAINING_SOURCE_DIRS[0]
    training_files.pop(f"{first_dir}/maneuver_shift_test/adaptive_candidate_fusion_predictions.npz")
    training_archive = _write_zip(tmp_path, "training.zip", training_files)

    result = verifier.build_result(str(main_archive), str(training_archive))

    assert result["status"] == "fail"
    training_check = result["training_inputs"]["checks"]["training_members"]
    assert training_check["status"] == "fail"
    assert any(
        failure.get("problem") == "missing_training_scenario_file"
        for failure in training_check["failures"]
    )


def test_rejects_training_checkpoint_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_archive, _ = _valid_archives(tmp_path, monkeypatch)
    training_files = _base_training_files()
    first_dir = verifier.EXPECTED_TRAINING_SOURCE_DIRS[0]
    training_files[f"{first_dir}/checkpoints/model.pt"] = b"checkpoint"
    training_archive = _write_zip(tmp_path, "training.zip", training_files)

    result = verifier.build_result(str(main_archive), str(training_archive))

    assert result["status"] == "fail"
    training_check = result["training_inputs"]["checks"]["training_members"]
    assert training_check["status"] == "fail"
    assert any(
        failure.get("problem") == "checkpoint_or_model_weight_member"
        for failure in training_check["failures"]
    )
