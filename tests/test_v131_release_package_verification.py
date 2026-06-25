from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts import verify_v131_release_package as verifier


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
    for tier in ("all_eval_non_development", "fresh_extra"):
        graph_tier = _graph_summary()["aggregate_tiers"][tier]  # type: ignore[index]
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


def _base_files() -> dict[str, str | bytes]:
    files: dict[str, str | bytes] = {}
    for member in verifier.REQUIRED_MEMBERS:
        suffix = Path(member).suffix.lower()
        if suffix in {".pdf", ".png"}:
            files[member] = b"fixture-binary"
        elif suffix == ".csv":
            files[member] = "field\nvalue\n"
        elif suffix == ".py":
            files[member] = "# fixture script\n"
        else:
            files[member] = (
                "v1.3.1 validation-selected residual-refine fixture "
                f"{verifier.VERSION_DOI} {verifier.CONCEPT_DOI} {verifier.TAG}\n"
            )

    files[".zenodo.json"] = json.dumps(
        {
            "version": verifier.TAG,
            "related_identifiers": [{"scheme": "doi", "relation": "isVersionOf", "identifier": verifier.CONCEPT_DOI}],
            "notes": f"Version DOI {verifier.VERSION_DOI}; concept DOI {verifier.CONCEPT_DOI}.",
        }
    )
    files["release/ZENODO_METADATA.json"] = json.dumps(
        {
            "metadata": {"version": verifier.TAG},
            "record": {"doi": verifier.VERSION_DOI, "concept_doi": verifier.CONCEPT_DOI},
            "github_release": {"tag": verifier.TAG},
        }
    )
    files[verifier.GRAPH_SUMMARY] = json.dumps(_graph_summary())
    files[verifier.LOCAL_SUMMARY] = json.dumps(_reference_summary(LOCAL_RMSE))
    files[verifier.MEAN_SUMMARY] = json.dumps(_reference_summary(MEAN_RMSE))
    files[verifier.COMPARISON_INTERVALS] = json.dumps(_comparison_intervals())
    files[verifier.TAIL_SUMMARY] = json.dumps(
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
    files[verifier.HELP_SMOKE_SCRIPT] = (
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


def _write_zip(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member, payload in files.items():
            zf.writestr(member, payload)
    return archive


def test_minimal_valid_release_fixture_passes_and_writes_reports(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path, _base_files())
    result = verifier.build_result(str(archive))

    assert result["status"] == "pass"
    assert result["checks"]["archive_members"]["status"] == "pass"
    assert result["checks"]["required_members"]["missing_count"] == 0
    assert result["checks"]["required_members"]["missing_prefixes"] == []
    assert result["checks"]["extracted_help_smoke"]["status"] == "pass"

    json_out = tmp_path / "report.json"
    md_out = tmp_path / "report.md"
    verifier.write_reports(result, json_out, md_out)
    assert json.loads(json_out.read_text(encoding="utf-8"))["status"] == "pass"
    markdown = md_out.read_text(encoding="utf-8")
    assert "not retraining" in markdown
    assert verifier.VERSION_DOI in markdown


@pytest.mark.parametrize(
    "bad_member, expected_problem",
    [
        ("../evil.txt", "unsafe_path_segment"),
        ("pkg/__pycache__/cached.py", "pycache_member"),
        ("pkg/cached.pyc", "pyc_member"),
        ("C:/evil.txt", "absolute_or_drive_member_path"),
    ],
)
def test_rejects_unsafe_and_bytecode_members(
    tmp_path: Path,
    bad_member: str,
    expected_problem: str,
) -> None:
    archive = _write_zip(tmp_path, {bad_member: "bad"})
    result = verifier.build_result(str(archive))

    assert result["status"] == "fail"
    assert result["checks"]["archive_members"]["status"] == "fail"
    problems = {
        problem
        for failure in result["checks"]["archive_members"]["failures"]
        for problem in failure["problems"]
    }
    assert expected_problem in problems
    assert result["checks"]["json_parse"]["status"] == "blocked"


def test_rejects_disallowed_current_text_terms(tmp_path: Path) -> None:
    files = _base_files()
    files["release/README_v1.3.1-validation-selected-residual-refine.md"] = "v1.3.1 DOI is pending\n"
    archive = _write_zip(tmp_path, files)

    result = verifier.build_result(str(archive))

    assert result["status"] == "fail"
    text_check = result["checks"]["text_hygiene"]
    assert text_check["status"] == "fail"
    assert text_check["failures"][0]["member"] == "release/README_v1.3.1-validation-selected-residual-refine.md"
    assert "v1.3.1 DOI is pending" in text_check["failures"][0]["disallowed_terms"]


def test_rejects_key_metric_mismatch(tmp_path: Path) -> None:
    files = _base_files()
    graph = json.loads(str(files[verifier.GRAPH_SUMMARY]))
    graph["aggregate_tiers"]["fresh_extra"]["row_wins"] = 28
    files[verifier.GRAPH_SUMMARY] = json.dumps(graph)
    archive = _write_zip(tmp_path, files)

    result = verifier.build_result(str(archive))

    assert result["status"] == "fail"
    metric_check = result["checks"]["metrics"]
    assert metric_check["status"] == "fail"
    failed_fields = {failure["field"] for failure in metric_check["failures"]}
    assert f"{verifier.GRAPH_SUMMARY}.aggregate_tiers.fresh_extra.row_wins" in failed_fields


@pytest.mark.parametrize(
    "missing_member, expected_problem",
    [
        ("scripts/_bootstrap.py", "_bootstrap"),
        ("src/gnn_state_estimation/__init__.py", "gnn_state_estimation"),
    ],
)
def test_extracted_help_smoke_rejects_missing_import_time_dependencies(
    tmp_path: Path,
    missing_member: str,
    expected_problem: str,
) -> None:
    files = _base_files()
    files.pop(missing_member)
    archive = _write_zip(tmp_path, files)

    result = verifier.build_result(str(archive))

    assert result["status"] == "fail"
    assert result["checks"]["required_members"]["status"] == "fail"
    smoke = result["checks"]["extracted_help_smoke"]
    assert smoke["status"] == "fail"
    assert smoke["returncode"] != 0
    assert expected_problem in smoke["stderr_tail"]
