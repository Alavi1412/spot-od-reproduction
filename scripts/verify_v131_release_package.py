#!/usr/bin/env python3
"""Verify the v1.3.1 validation-selected residual-refinement release package.

This is a bounded GitHub Actions verifier for the packaged release artifact. It
checks ZIP member safety, required packaged evidence, DOI/tag metadata, stale
current-claim text, and metric readback from JSON files inside the archive.

It intentionally does not extract the archive, retrain models, rerun raw data or
all filters, fetch public precise-reference inputs, or establish independent
third-party/operational validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARCHIVE_REL = "release/spot_od_v1_3_1_validation_selected_residual_refine.zip"
DEFAULT_JSON_OUT = "results/validation/v131_release_package_verification.json"
DEFAULT_MD_OUT = "results/validation/v131_release_package_verification.md"

TAG = "v1.3.1-validation-selected-residual-refine"
VERSION_DOI = "10.5281/zenodo.20844596"
CONCEPT_DOI = "10.5281/zenodo.20768672"

SCOPE_BOUNDARY = (
    "GitHub-hosted package integrity and metric readback only: this verifier "
    "opens the packaged v1.3.1 release ZIP, checks member safety, required "
    "files, current DOI/tag metadata, current text hygiene, selected saved "
    "metric JSON values, and a fast extracted-package --help smoke for the "
    "released graph-selector entry point. It is not retraining, not full raw/training/all-filter "
    "reproduction, not public precise-reference validation, not independent "
    "third-party reproduction, and not operational validation."
)

HELP_SMOKE_SCRIPT = "scripts/run_trajectory_candidate_graph_selector_poc.py"
HELP_SMOKE_ARGS = (HELP_SMOKE_SCRIPT, "--help")
HELP_SMOKE_TIMEOUT_SECONDS = 30

GRAPH_DIR = (
    "results/"
    "trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
LOCAL_DIR = (
    "results/"
    "trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
MEAN_DIR = (
    "results/"
    "trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
TAIL_DIR = "results/trajectory_candidate_edge_only_local_tail_diagnostic_val53_20260625"

GRAPH_SUMMARY = f"{GRAPH_DIR}/summary.json"
GRAPH_ROWS = f"{GRAPH_DIR}/rows.csv"
COMPARISON_INTERVALS = f"{GRAPH_DIR}/comparison_intervals.json"
COMPARISON_INTERVALS_MD = f"{GRAPH_DIR}/comparison_intervals.md"
LOCAL_SUMMARY = f"{LOCAL_DIR}/summary.json"
LOCAL_ROWS = f"{LOCAL_DIR}/rows.csv"
MEAN_SUMMARY = f"{MEAN_DIR}/summary.json"
MEAN_ROWS = f"{MEAN_DIR}/rows.csv"
TAIL_SUMMARY = f"{TAIL_DIR}/summary.json"
TAIL_ROWS = f"{TAIL_DIR}/rows.csv"

REQUIRED_MEMBERS: tuple[str, ...] = (
    ".zenodo.json",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "paper/main.tex",
    "paper/main.pdf",
    "paper/tables/main_findings_summary.tex",
    "paper/tables/main_revision_delta_and_public_repro.tex",
    "paper/tables/main_trajectory_graph_selector_ensemble_poc.tex",
    "paper/figures/trajectory_residual_refine_gain_distribution_val53.png",
    "release/README.md",
    "release/README_v1.3.1-validation-selected-residual-refine.md",
    "release/RELEASE_NOTES_v1.3.1-validation-selected-residual-refine.md",
    "release/MANIFEST_v1.3.1-validation-selected-residual-refine.md",
    "release/CITATION.cff",
    "release/LICENSE_CC_BY_4_0.txt",
    "release/ZENODO_METADATA.json",
    "scripts/__init__.py",
    "scripts/_bootstrap.py",
    "scripts/analyze_trajectory_candidate_graph_architecture_ensemble.py",
    "scripts/build_trajectory_residual_refine_comparison_intervals.py",
    "scripts/build_trajectory_residual_refine_tail_diagnostic.py",
    "scripts/build_trajectory_residual_refine_figure.py",
    HELP_SMOKE_SCRIPT,
    "src/gnn_state_estimation/__init__.py",
    "tests/test_build_trajectory_residual_refine_comparison_intervals.py",
    "tests/test_build_trajectory_residual_refine_tail_diagnostic.py",
    "tests/test_build_trajectory_residual_refine_figure.py",
    "tests/test_trajectory_candidate_graph_architecture_ensemble.py",
    GRAPH_SUMMARY,
    GRAPH_ROWS,
    COMPARISON_INTERVALS,
    COMPARISON_INTERVALS_MD,
    LOCAL_SUMMARY,
    LOCAL_ROWS,
    MEAN_SUMMARY,
    MEAN_ROWS,
    TAIL_SUMMARY,
    TAIL_ROWS,
)

REQUIRED_MEMBER_PREFIXES: tuple[str, ...] = (
    "src/gnn_state_estimation/",
)

TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_SCAN_EXCLUDES = {
    "release/RELEASE_NOTES_v1.3.0-edge-only-residual-refine.md",
}

DISALLOWED_CURRENT_TEXT_TERMS: tuple[str, ...] = (
    "18.682",
    "18.324",
    "33.504",
    "55.270",
    "6220529",
    "6,220,529",
    "9bf9dcf7",
    "6245298",
    "6,245,298",
    "15EA38D9786B72DE626ABD6EBE6E0470893F5C16B1FF9045406089CBB9C10974",
    "v1.3.1 DOI is pending",
    "Version DOI: pending Zenodo import",
    "version DOI is pending Zenodo import",
    "Its Zenodo version DOI and record are pending",
    "pending_github_release_zenodo_import",
    "pending_final_github_tag_target_at_release_creation",
    "pending_publication_or_import",
    "pending_or_not_recorded_for_v1.3.1",
)

FLOAT_ABS_TOLERANCE = 1e-9

EXPECTED_GRAPH_METRICS: tuple[tuple[str, str, float | int], ...] = (
    ("all_eval_non_development", "selector_observed_step_rmse_m", 390.31678478219936),
    ("all_eval_non_development", "best_single_observed_step_rmse_m", 459.5907664568724),
    ("all_eval_non_development", "gain_vs_best_single_percent", 15.072970723221363),
    ("all_eval_non_development", "rows", 230),
    ("all_eval_non_development", "row_wins", 154),
    ("all_eval_non_development", "row_losses", 76),
    ("fresh_extra", "selector_observed_step_rmse_m", 386.37323366223956),
    ("fresh_extra", "best_single_observed_step_rmse_m", 445.94330738322924),
    ("fresh_extra", "gain_vs_best_single_percent", 13.358216781084483),
    ("fresh_extra", "rows", 47),
    ("fresh_extra", "row_wins", 29),
    ("fresh_extra", "row_losses", 18),
)

EXPECTED_INTERVAL_METRICS: tuple[tuple[str, str, str, float | list[float]], ...] = (
    ("all_eval_non_development", "edge_only_local_residual_refine", "gain_percent", 24.835317289797644),
    ("fresh_extra", "edge_only_local_residual_refine", "gain_percent", 45.32943778982349),
    ("all_eval_non_development", "edge_only_mean_residual_refine", "gain_percent", 3.9832690946483287),
    ("fresh_extra", "edge_only_mean_residual_refine", "gain_percent", 5.547619715108024),
    (
        "all_eval_non_development",
        "best_single_retained",
        "row_bootstrap_gain_percent_95ci",
        [11.45062980513122, 18.575159123708463],
    ),
    (
        "fresh_extra",
        "best_single_retained",
        "row_bootstrap_gain_percent_95ci",
        [5.394507848224321, 20.688404714449202],
    ),
)

DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")


def norm(path: str | Path) -> str:
    return str(path).replace("\\", "/").strip()


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return norm(path.relative_to(ROOT))
    except ValueError:
        return norm(path)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def member_name_problems(raw_name: str) -> tuple[str, list[str]]:
    normalized = raw_name.replace("\\", "/")
    stripped = normalized.rstrip("/") if normalized.endswith("/") else normalized
    problems: list[str] = []

    if not stripped:
        problems.append("empty_member_name")
        return stripped, problems
    if normalized.startswith("/") or DRIVE_PREFIX_RE.match(normalized):
        problems.append("absolute_or_drive_member_path")

    parts = stripped.split("/")
    if any(part in ("", ".", "..") for part in parts):
        problems.append("unsafe_path_segment")
    if "__pycache__" in parts:
        problems.append("pycache_member")
    if parts[-1].lower().endswith(".pyc"):
        problems.append("pyc_member")

    return stripped, problems


def should_scan_text_member(member_name: str) -> bool:
    if member_name in TEXT_SCAN_EXCLUDES:
        return False
    return Path(member_name).suffix.lower() in TEXT_SUFFIXES


def read_member_text(zf: zipfile.ZipFile, member_name: str) -> str:
    return zf.read(member_name).decode("utf-8")


def read_member_json(zf: zipfile.ZipFile, member_name: str) -> Any:
    return json.loads(read_member_text(zf, member_name))


def status_from_failures(failures: list[Any]) -> str:
    return "pass" if not failures else "fail"


def check_archive_members(zf: zipfile.ZipFile) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    file_members: list[str] = []
    seen_files: set[str] = set()
    duplicate_files: list[str] = []

    crc_failure = zf.testzip()
    if crc_failure is not None:
        failures.append({"member": crc_failure, "problems": ["crc_failure"]})

    for info in zf.infolist():
        member_name, problems = member_name_problems(info.filename)
        if problems:
            failures.append({"member": info.filename, "normalized_member": member_name, "problems": problems})
        if info.is_dir():
            continue
        file_members.append(member_name)
        if member_name in seen_files:
            duplicate_files.append(member_name)
        seen_files.add(member_name)

    for member_name in sorted(set(duplicate_files)):
        failures.append({"member": member_name, "problems": ["duplicate_file_member"]})

    return {
        "status": status_from_failures(failures),
        "file_member_count": len(file_members),
        "unique_file_member_count": len(seen_files),
        "failures": failures,
        "members": sorted(seen_files),
    }


def check_required_members(member_set: set[str]) -> dict[str, Any]:
    missing = [member for member in REQUIRED_MEMBERS if member not in member_set]
    prefix_file_counts = {
        prefix: sum(1 for member in member_set if member.startswith(prefix) and not member.endswith("/"))
        for prefix in REQUIRED_MEMBER_PREFIXES
    }
    missing_prefixes = [
        prefix
        for prefix, count in prefix_file_counts.items()
        if count == 0
    ]
    return {
        "status": "pass" if not missing and not missing_prefixes else "fail",
        "required_count": len(REQUIRED_MEMBERS),
        "missing_count": len(missing),
        "missing": missing,
        "required_prefixes": list(REQUIRED_MEMBER_PREFIXES),
        "prefix_file_counts": prefix_file_counts,
        "missing_prefixes": missing_prefixes,
    }


def text_tail(value: str | bytes | None, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
    else:
        text = value
    if len(text) <= limit:
        return text
    return text[-limit:]


def check_extracted_help_smoke(zf: zipfile.ZipFile, member_set: set[str]) -> dict[str, Any]:
    if HELP_SMOKE_SCRIPT not in member_set:
        return {
            "status": "fail",
            "command": ["python", *HELP_SMOKE_ARGS],
            "failure": {
                "problem": "entrypoint_missing",
                "member": HELP_SMOKE_SCRIPT,
            },
        }

    with tempfile.TemporaryDirectory(prefix="spot_od_v131_release_help_") as tmp:
        extract_root = Path(tmp)
        zf.extractall(extract_root)
        entrypoint = extract_root / HELP_SMOKE_SCRIPT
        if not entrypoint.is_file():
            return {
                "status": "fail",
                "command": ["python", *HELP_SMOKE_ARGS],
                "failure": {
                    "problem": "entrypoint_not_extracted",
                    "member": HELP_SMOKE_SCRIPT,
                },
            }

        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                [sys.executable, *HELP_SMOKE_ARGS],
                cwd=extract_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=HELP_SMOKE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "fail",
                "command": ["python", *HELP_SMOKE_ARGS],
                "timeout_seconds": HELP_SMOKE_TIMEOUT_SECONDS,
                "failure": {
                    "problem": "timeout",
                    "stdout_tail": text_tail(exc.stdout),
                    "stderr_tail": text_tail(exc.stderr),
                },
            }

    stdout_tail = text_tail(completed.stdout)
    stderr_tail = text_tail(completed.stderr)
    combined_output = f"{completed.stdout}\n{completed.stderr}".lower()
    failure: dict[str, Any] | None = None
    if completed.returncode != 0:
        failure = {"problem": "nonzero_exit"}
    elif "usage:" not in combined_output:
        failure = {"problem": "help_usage_missing"}

    result: dict[str, Any] = {
        "status": "pass" if failure is None else "fail",
        "command": ["python", *HELP_SMOKE_ARGS],
        "python_executable": sys.executable,
        "timeout_seconds": HELP_SMOKE_TIMEOUT_SECONDS,
        "returncode": completed.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if failure is not None:
        result["failure"] = failure
    return result


def parse_required_json(zf: zipfile.ZipFile, member_set: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    json_members = {
        "zenodo_json": ".zenodo.json",
        "zenodo_metadata": "release/ZENODO_METADATA.json",
        "graph_summary": GRAPH_SUMMARY,
        "local_summary": LOCAL_SUMMARY,
        "mean_summary": MEAN_SUMMARY,
        "comparison_intervals": COMPARISON_INTERVALS,
        "tail_diagnostic_summary": TAIL_SUMMARY,
    }
    parsed: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for label, member_name in json_members.items():
        if member_name not in member_set:
            failures.append({"label": label, "member": member_name, "problem": "missing"})
            continue
        try:
            parsed[label] = read_member_json(zf, member_name)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append({"label": label, "member": member_name, "problem": repr(exc)})
    return parsed, {
        "status": status_from_failures(failures),
        "parsed_members": sorted(parsed),
        "failure_count": len(failures),
        "failures": failures,
    }


def require_equal(failures: list[dict[str, Any]], field: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        failures.append({"field": field, "expected": expected, "actual": actual})


def require_close(
    failures: list[dict[str, Any]],
    field: str,
    actual: Any,
    expected: float,
    *,
    abs_tol: float = FLOAT_ABS_TOLERANCE,
) -> None:
    if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=abs_tol):
        failures.append({"field": field, "expected": expected, "actual": actual, "abs_tolerance": abs_tol})


def require_close_list(
    failures: list[dict[str, Any]],
    field: str,
    actual: Any,
    expected: list[float],
    *,
    abs_tol: float = FLOAT_ABS_TOLERANCE,
) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        failures.append({"field": field, "expected": expected, "actual": actual, "abs_tolerance": abs_tol})
        return
    for idx, (actual_value, expected_value) in enumerate(zip(actual, expected, strict=True)):
        if not isinstance(actual_value, (int, float)) or not math.isclose(
            float(actual_value),
            expected_value,
            rel_tol=0.0,
            abs_tol=abs_tol,
        ):
            failures.append(
                {
                    "field": f"{field}[{idx}]",
                    "expected": expected_value,
                    "actual": actual_value,
                    "abs_tolerance": abs_tol,
                }
            )


def check_metadata(parsed: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    zenodo = parsed.get("zenodo_json", {})
    metadata_record = parsed.get("zenodo_metadata", {})
    metadata = metadata_record.get("metadata", {}) if isinstance(metadata_record, dict) else {}
    record = metadata_record.get("record", {}) if isinstance(metadata_record, dict) else {}
    github_release = metadata_record.get("github_release", {}) if isinstance(metadata_record, dict) else {}
    zenodo_related = zenodo.get("related_identifiers", []) if isinstance(zenodo, dict) else []
    zenodo_text = json.dumps(zenodo, sort_keys=True)
    metadata_text = json.dumps(metadata_record, sort_keys=True)

    require_equal(failures, ".zenodo.json.version", zenodo.get("version") if isinstance(zenodo, dict) else None, TAG)
    require_equal(failures, "release/ZENODO_METADATA.json.metadata.version", metadata.get("version"), TAG)
    require_equal(failures, "release/ZENODO_METADATA.json.github_release.tag", github_release.get("tag"), TAG)
    require_equal(failures, "release/ZENODO_METADATA.json.record.doi", record.get("doi"), VERSION_DOI)
    require_equal(failures, "release/ZENODO_METADATA.json.record.concept_doi", record.get("concept_doi"), CONCEPT_DOI)

    related_identifiers = [
        item.get("identifier")
        for item in zenodo_related
        if isinstance(item, dict)
    ]
    if CONCEPT_DOI not in related_identifiers:
        failures.append(
            {
                "field": ".zenodo.json.related_identifiers",
                "expected_contains": CONCEPT_DOI,
                "actual": related_identifiers,
            }
        )
    if VERSION_DOI not in zenodo_text:
        failures.append({"field": ".zenodo.json", "expected_contains": VERSION_DOI})
    if VERSION_DOI not in metadata_text:
        failures.append({"field": "release/ZENODO_METADATA.json", "expected_contains": VERSION_DOI})
    if CONCEPT_DOI not in metadata_text:
        failures.append({"field": "release/ZENODO_METADATA.json", "expected_contains": CONCEPT_DOI})

    return {
        "status": status_from_failures(failures),
        "version": TAG,
        "version_doi": VERSION_DOI,
        "concept_doi": CONCEPT_DOI,
        "failure_count": len(failures),
        "failures": failures,
    }


def check_text_hygiene(zf: zipfile.ZipFile, members: list[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    scanned: list[str] = []
    terms_lower = [(term, term.lower()) for term in DISALLOWED_CURRENT_TEXT_TERMS]

    for member_name in sorted(members):
        if not should_scan_text_member(member_name):
            continue
        try:
            text = read_member_text(zf, member_name)
        except UnicodeDecodeError as exc:
            failures.append({"member": member_name, "problem": f"utf8_decode_failed: {exc}"})
            continue
        scanned.append(member_name)
        lower_text = text.lower()
        hits = [term for term, lowered in terms_lower if lowered in lower_text]
        if hits:
            failures.append({"member": member_name, "disallowed_terms": hits})

    return {
        "status": status_from_failures(failures),
        "scanned_text_member_count": len(scanned),
        "scan_excludes": sorted(TEXT_SCAN_EXCLUDES),
        "failure_count": len(failures),
        "failures": failures,
    }


def tier_value(summary: dict[str, Any], tier: str, field: str) -> Any:
    return summary.get("aggregate_tiers", {}).get(tier, {}).get(field)


def interval_value(intervals: dict[str, Any], tier: str, comparison: str, field: str) -> Any:
    return intervals.get("comparisons", {}).get(tier, {}).get(comparison, {}).get(field)


def check_metrics(parsed: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    graph = parsed.get("graph_summary", {})
    local = parsed.get("local_summary", {})
    mean = parsed.get("mean_summary", {})
    intervals = parsed.get("comparison_intervals", {})
    tail = parsed.get("tail_diagnostic_summary", {})

    for tier, field, expected in EXPECTED_GRAPH_METRICS:
        actual = tier_value(graph, tier, field)
        metric_field = f"{GRAPH_SUMMARY}.aggregate_tiers.{tier}.{field}"
        if isinstance(expected, int):
            require_equal(failures, metric_field, actual, expected)
        else:
            require_close(failures, metric_field, actual, expected)

    for tier, comparison, field, expected in EXPECTED_INTERVAL_METRICS:
        actual = interval_value(intervals, tier, comparison, field)
        metric_field = f"{COMPARISON_INTERVALS}.comparisons.{tier}.{comparison}.{field}"
        if isinstance(expected, list):
            require_close_list(failures, metric_field, actual, expected)
        else:
            require_close(failures, metric_field, actual, expected)

    for tier in ("all_eval_non_development", "fresh_extra"):
        local_reference = interval_value(intervals, tier, "edge_only_local_residual_refine", "reference_rmse_m")
        mean_reference = interval_value(intervals, tier, "edge_only_mean_residual_refine", "reference_rmse_m")
        if isinstance(local_reference, (int, float)):
            require_close(
                failures,
                f"{LOCAL_SUMMARY}.aggregate_tiers.{tier}.selector_observed_step_rmse_m matches intervals",
                tier_value(local, tier, "selector_observed_step_rmse_m"),
                float(local_reference),
            )
        else:
            failures.append(
                {
                    "field": f"{COMPARISON_INTERVALS}.comparisons.{tier}.edge_only_local_residual_refine.reference_rmse_m",
                    "problem": "missing_or_non_numeric",
                    "actual": local_reference,
                }
            )
        if isinstance(mean_reference, (int, float)):
            require_close(
                failures,
                f"{MEAN_SUMMARY}.aggregate_tiers.{tier}.selector_observed_step_rmse_m matches intervals",
                tier_value(mean, tier, "selector_observed_step_rmse_m"),
                float(mean_reference),
            )
        else:
            failures.append(
                {
                    "field": f"{COMPARISON_INTERVALS}.comparisons.{tier}.edge_only_mean_residual_refine.reference_rmse_m",
                    "problem": "missing_or_non_numeric",
                    "actual": mean_reference,
                }
            )

    if tail.get("generated_from_saved_rows_only") is not True:
        failures.append(
            {
                "field": f"{TAIL_SUMMARY}.generated_from_saved_rows_only",
                "expected": True,
                "actual": tail.get("generated_from_saved_rows_only"),
            }
        )
    tail_all = tail.get("tiers", {}).get("all_eval_non_development", {}) if isinstance(tail, dict) else {}
    tail_pooled = tail_all.get("pooled_rmse_m", {}) if isinstance(tail_all, dict) else {}
    require_close(
        failures,
        f"{TAIL_SUMMARY}.tiers.all_eval_non_development.pooled_rmse_m.edge_only_attention",
        tail_pooled.get("edge_only_attention"),
        390.31678478219936,
    )

    metric_summary = {
        "graph_all_non_development": graph.get("aggregate_tiers", {}).get("all_eval_non_development", {}),
        "graph_fresh": graph.get("aggregate_tiers", {}).get("fresh_extra", {}),
        "comparison_all_non_development": intervals.get("comparisons", {}).get("all_eval_non_development", {}),
        "comparison_fresh": intervals.get("comparisons", {}).get("fresh_extra", {}),
    }
    return {
        "status": status_from_failures(failures),
        "abs_tolerance": FLOAT_ABS_TOLERANCE,
        "failure_count": len(failures),
        "failures": failures,
        "summary": metric_summary,
    }


def archive_record(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def blocked_check(reason: str) -> dict[str, Any]:
    return {"status": "blocked", "reason": reason}


def build_result(archive_rel: str = DEFAULT_ARCHIVE_REL) -> dict[str, Any]:
    archive_path = repo_path(archive_rel)
    archive = archive_record(archive_path)
    checks: dict[str, Any] = {}

    if not archive_path.is_file():
        checks["archive_members"] = {"status": "fail", "failures": [{"problem": "archive_missing"}]}
        checks["required_members"] = blocked_check("archive_missing")
        checks["json_parse"] = blocked_check("archive_missing")
        checks["metadata"] = blocked_check("archive_missing")
        checks["text_hygiene"] = blocked_check("archive_missing")
        checks["metrics"] = blocked_check("archive_missing")
        checks["extracted_help_smoke"] = blocked_check("archive_missing")
        return {
            "schema_version": "v131_release_package_verification.v1",
            "status": "fail",
            "scope": "v1.3.1_release_package_integrity_and_metric_readback",
            "scope_boundary": SCOPE_BOUNDARY,
            "archive": archive,
            "checks": checks,
        }

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            member_check = check_archive_members(zf)
            checks["archive_members"] = {k: v for k, v in member_check.items() if k != "members"}
            members = member_check["members"]
            member_set = set(members)
            checks["required_members"] = check_required_members(member_set)

            if member_check["status"] != "pass":
                checks["json_parse"] = blocked_check("unsafe_or_invalid_archive_members")
                checks["metadata"] = blocked_check("unsafe_or_invalid_archive_members")
                checks["text_hygiene"] = blocked_check("unsafe_or_invalid_archive_members")
                checks["metrics"] = blocked_check("unsafe_or_invalid_archive_members")
                checks["extracted_help_smoke"] = blocked_check("unsafe_or_invalid_archive_members")
            else:
                parsed, parse_check = parse_required_json(zf, member_set)
                checks["json_parse"] = parse_check
                checks["metadata"] = check_metadata(parsed) if parse_check["status"] == "pass" else blocked_check("json_parse_failed")
                checks["text_hygiene"] = check_text_hygiene(zf, members)
                checks["metrics"] = check_metrics(parsed) if parse_check["status"] == "pass" else blocked_check("json_parse_failed")
                checks["extracted_help_smoke"] = check_extracted_help_smoke(zf, member_set)
    except zipfile.BadZipFile as exc:
        checks["archive_members"] = {"status": "fail", "failures": [{"problem": "bad_zip", "details": repr(exc)}]}
        checks["required_members"] = blocked_check("bad_zip")
        checks["json_parse"] = blocked_check("bad_zip")
        checks["metadata"] = blocked_check("bad_zip")
        checks["text_hygiene"] = blocked_check("bad_zip")
        checks["metrics"] = blocked_check("bad_zip")
        checks["extracted_help_smoke"] = blocked_check("bad_zip")

    status = "pass" if all(check.get("status") == "pass" for check in checks.values()) else "fail"
    return {
        "schema_version": "v131_release_package_verification.v1",
        "status": status,
        "scope": "v1.3.1_release_package_integrity_and_metric_readback",
        "scope_boundary": SCOPE_BOUNDARY,
        "archive": archive,
        "package": {
            "version": TAG,
            "version_doi": VERSION_DOI,
            "concept_doi": CONCEPT_DOI,
        },
        "required_members": list(REQUIRED_MEMBERS),
        "checks": checks,
    }


def write_reports(result: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checks = result.get("checks", {})
    metrics = checks.get("metrics", {}).get("summary", {})
    graph_all = metrics.get("graph_all_non_development", {})
    graph_fresh = metrics.get("graph_fresh", {})
    comparison_all = metrics.get("comparison_all_non_development", {})
    comparison_fresh = metrics.get("comparison_fresh", {})
    smoke = checks.get("extracted_help_smoke", {})

    lines = [
        "# v1.3.1 Release Package Verification",
        "",
        f"Status: **{str(result.get('status')).upper()}**",
        "",
        "## Scope Boundary",
        str(result.get("scope_boundary", SCOPE_BOUNDARY)),
        "",
        "## Archive",
        f"- Path: `{result.get('archive', {}).get('path')}`",
        f"- Exists: `{result.get('archive', {}).get('exists')}`",
        f"- Bytes: `{result.get('archive', {}).get('bytes')}`",
        f"- SHA-256: `{result.get('archive', {}).get('sha256')}`",
        "",
        "## Metadata",
        f"- Version/tag: `{TAG}`",
        f"- Zenodo version DOI: `{VERSION_DOI}`",
        f"- Zenodo concept DOI: `{CONCEPT_DOI}`",
        "",
        "## Checks",
    ]
    for name in (
        "archive_members",
        "required_members",
        "json_parse",
        "metadata",
        "text_hygiene",
        "metrics",
        "extracted_help_smoke",
    ):
        check = checks.get(name, {})
        lines.append(f"- {name}: **{str(check.get('status')).upper()}**")
    lines += [
        "",
        "## Metric Readback",
        (
            "- All non-development graph selector: "
            f"`{graph_all.get('selector_observed_step_rmse_m')}` m vs best retained "
            f"`{graph_all.get('best_single_observed_step_rmse_m')}` m, gain "
            f"`{graph_all.get('gain_vs_best_single_percent')}`%, W/L "
            f"`{graph_all.get('row_wins')}/{graph_all.get('row_losses')}` over "
            f"`{graph_all.get('rows')}` rows."
        ),
        (
            "- Fresh graph selector: "
            f"`{graph_fresh.get('selector_observed_step_rmse_m')}` m vs best retained "
            f"`{graph_fresh.get('best_single_observed_step_rmse_m')}` m, gain "
            f"`{graph_fresh.get('gain_vs_best_single_percent')}`%, W/L "
            f"`{graph_fresh.get('row_wins')}/{graph_fresh.get('row_losses')}` over "
            f"`{graph_fresh.get('rows')}` rows."
        ),
        (
            "- Attention vs local gains: all non-development "
            f"`{comparison_all.get('edge_only_local_residual_refine', {}).get('gain_percent')}`%, fresh "
            f"`{comparison_fresh.get('edge_only_local_residual_refine', {}).get('gain_percent')}`%."
        ),
        (
            "- Attention vs mean gains: all non-development "
            f"`{comparison_all.get('edge_only_mean_residual_refine', {}).get('gain_percent')}`%, fresh "
            f"`{comparison_fresh.get('edge_only_mean_residual_refine', {}).get('gain_percent')}`%."
        ),
        "",
        "## Outputs",
        f"- JSON: `{rel(json_out)}`",
        f"- Markdown: `{rel(md_out)}`",
        "",
        "## Extracted Help Smoke",
        f"- Command: `python {' '.join(HELP_SMOKE_ARGS)}`",
        f"- Status: **{str(smoke.get('status')).upper()}**",
        f"- Return code: `{smoke.get('returncode')}`",
    ]

    failing_checks = {
        name: check
        for name, check in checks.items()
        if check.get("status") not in {"pass", None}
    }
    if failing_checks:
        lines += ["", "## Failures"]
        for name, check in failing_checks.items():
            failures = check.get("failures")
            failure = check.get("failure")
            reason = check.get("reason")
            if reason:
                lines.append(f"- {name}: {reason}")
            elif failures:
                lines.append(f"- {name}: {len(failures)} failure(s); see JSON report for details.")
            elif failure:
                lines.append(f"- {name}: {failure.get('problem')}; see JSON report for details.")

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the v1.3.1 release package from the ZIP contents.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE_REL)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", default=DEFAULT_MD_OUT)
    parser.add_argument("--check-only", action="store_true", help="Run checks without writing JSON/Markdown reports.")
    args = parser.parse_args()

    result = build_result(args.archive)
    json_out = repo_path(args.json_out)
    md_out = repo_path(args.md_out)
    if not args.check_only:
        write_reports(result, json_out, md_out)

    summary = {
        "status": result["status"],
        "archive": result["archive"],
        "json": rel(json_out),
        "markdown": rel(md_out),
        "check_only": args.check_only,
        "scope_boundary": SCOPE_BOUNDARY,
        "checks": {name: check.get("status") for name, check in result["checks"].items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
