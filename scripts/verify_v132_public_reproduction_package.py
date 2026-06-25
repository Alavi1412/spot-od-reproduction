#!/usr/bin/env python3
"""Verify the v1.3.2 public reproduction package and training inputs.

This verifier is a bounded GitHub Actions/local integrity check for the public
reproduction package repair. It checks the main release ZIP, the separate
checkpoint-free retained-candidate training-input ZIP, v1.3.2 metadata, the
unchanged v1.3.1 metric readback, and a fast extracted-package --help smoke.

It intentionally does not train models, rerun raw data generation, rerun all
filters, fetch public precise-reference inputs, or establish independent
third-party/operational validation.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

try:
    from scripts import verify_v131_release_package as v131
except (ImportError, ModuleNotFoundError):  # pragma: no cover - direct script execution
    import verify_v131_release_package as v131  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]

TAG = "v1.3.2-public-reproduction-package"
RELEASE_URL = "https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.2-public-reproduction-package"
REPO_URL = "https://github.com/Alavi1412/spot-od-reproduction"
PRIOR_VERSION_DOI = "10.5281/zenodo.20844596"
CONCEPT_DOI = "10.5281/zenodo.20768672"
HISTORICAL_CITED_DOI = "10.5281/zenodo.20840386"
HISTORICAL_V130_DOI = "10.5281/zenodo.20842573"
DEFAULT_ARCHIVE_REL = "release/spot_od_v1_3_2_public_reproduction_package.zip"
DEFAULT_TRAINING_ARCHIVE_REL = "release/spot_od_v1_3_2_public_reproduction_training_inputs.zip"
DEFAULT_JSON_OUT = "results/validation/v132_public_reproduction_package_verification.json"
DEFAULT_MD_OUT = "results/validation/v132_public_reproduction_package_verification.md"

SCOPE_BOUNDARY = (
    "GitHub-hosted public reproduction package integrity only: this verifier "
    "opens the packaged v1.3.2 main ZIP and training-input ZIP, checks member "
    "safety, required runtime import members, v1.3.2 metadata, absence of a "
    "claimed pre-import v1.3.2 DOI, unchanged selected metric readback, a fast "
    "extracted-package --help smoke for the graph-selector entry point, "
    "recursive private-path hygiene over text, nested ZIPs, and binary bytes, "
    "and checkpoint-free retained-candidate training-input directories. It is "
    "not checkpoint reload, not training, not full raw/training/all-filter "
    "reproduction, not public precise-reference validation, not independent "
    "third-party reproduction, and not operational validation."
)

MAIN_ASSET = "spot_od_v1_3_2_public_reproduction_package.zip"
TRAINING_ASSET = "spot_od_v1_3_2_public_reproduction_training_inputs.zip"
LEGACY_V131_ARCHIVE_MEMBER = "release/spot_od_v1_3_1_validation_selected_residual_refine.zip"
MAX_NESTED_ZIP_SCAN_DEPTH = 5

V132_RELEASE_DOCS: tuple[str, ...] = (
    "release/README_v1.3.2-public-reproduction-package.md",
    "release/RELEASE_NOTES_v1.3.2-public-reproduction-package.md",
    "release/MANIFEST_v1.3.2-public-reproduction-package.md",
)
V131_REPAIR_TRACEABILITY_MEMBERS: tuple[str, ...] = (
    "release/README_v1.3.1-validation-selected-residual-refine.md",
    "release/RELEASE_NOTES_v1.3.1-validation-selected-residual-refine.md",
    "release/MANIFEST_v1.3.1-validation-selected-residual-refine.md",
    "scripts/verify_v131_release_package.py",
    "tests/test_v131_release_package_verification.py",
)
V132_VERIFIER_MEMBERS: tuple[str, ...] = (
    "scripts/verify_v132_public_reproduction_package.py",
    "scripts/build_v132_public_reproduction_archives.py",
    "tests/test_v132_public_reproduction_package_verification.py",
    "tests/conftest.py",
)

REQUIRED_MAIN_MEMBERS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            *v131.REQUIRED_MEMBERS,
            *V132_RELEASE_DOCS,
            *V131_REPAIR_TRACEABILITY_MEMBERS,
            *V132_VERIFIER_MEMBERS,
            "release/TRAINING_INPUT_BUNDLE_MANIFEST_v1.3.2-public-reproduction-package.json",
            "release/TRAINING_INPUT_BUNDLE_v1.3.2-public-reproduction-package.md",
            "paper/tables/main_row_weighted_dls_poc.tex",
        ]
    )
)
REQUIRED_MAIN_PREFIXES: tuple[str, ...] = v131.REQUIRED_MEMBER_PREFIXES

CURRENT_METADATA_MEMBERS: tuple[str, ...] = (
    ".zenodo.json",
    "release/ZENODO_METADATA.json",
    "release/CITATION.cff",
    "release/README.md",
    *V132_RELEASE_DOCS,
)

TRAINING_MANIFEST_JSON = "release/TRAINING_INPUT_BUNDLE_MANIFEST_v1.3.2-public-reproduction-package.json"
TRAINING_MANIFEST_MD = "release/TRAINING_INPUT_BUNDLE_v1.3.2-public-reproduction-package.md"

EXPECTED_TRAINING_SOURCE_DIRS: tuple[str, ...] = (
    "results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625",
    "results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625",
    "results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625",
    "results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed167_split167_20260625",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed101_split101_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed103_split103_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed107_split107_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed109_split109_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed113_split113_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed127_split127_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed131_split131_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed137_split137_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed23_split23_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed29_split29_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed31_split31_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed37_split37_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed41_split41_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed43_split43_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed47_split47_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed53_split53_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed59_split59_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed61_split61_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed71_split71_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed73_split73_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed79_split79_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed83_split83_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed89_split89_20260623",
    "results/adaptive_candidate_fusion_observed_fixed_soft_seed97_split97_20260623",
)
TRAINING_SCENARIOS: tuple[str, ...] = ("maneuver_shift_test", "process_noise_shift_test")
TRAINING_ROOT_FILES: tuple[str, ...] = (
    "adaptive_candidate_fusion_summary.csv",
    "adaptive_candidate_fusion_summary.json",
    "env_report.json",
    "manifest.json",
    "run_config_summary.json",
    "train_history.json",
)
TRAINING_SCENARIO_FILES: tuple[str, ...] = (
    "adaptive_candidate_fusion_predictions.npz",
    "adaptive_candidate_fusion_summary.json",
)
EXPECTED_TRAINING_PAYLOAD_FILE_COUNT = 290
EXPECTED_TRAINING_TOTAL_FILE_COUNT = 292

DOI_RE = re.compile(r"10\.5281/zenodo\.\d+")
ALLOWED_METADATA_DOIS = {
    PRIOR_VERSION_DOI,
    CONCEPT_DOI,
    HISTORICAL_CITED_DOI,
    HISTORICAL_V130_DOI,
    "10.5281/zenodo.20825138",
    "10.5281/zenodo.20822968",
    "10.5281/zenodo.20811701",
}

_BS = "\\"
_PRIVATE_PATH_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("nas_backslash_unc", re.compile(r"(?i)" + re.escape(_BS) + r"+nas(?:[\\/]|$)")),
    ("nas_forward_unc", re.compile("(?i)" + "/" + "/" + "nas" + r"(?:/|$)")),
    ("windows_drive_absolute_path", re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]")),
    ("windows_user_profile", re.compile(r"(?i)(?<![A-Za-z0-9])C:[\\/]+Users[\\/]")),
    ("appdata_profile_path", re.compile(r"(?i)\b" + "App" + "Data" + r"\b")),
    ("unix_user_home", re.compile(r"(?i)(?<![A-Za-z0-9_])/(?:Users|home|mnt|Volumes)/")),
    ("local_training_worktree_name", re.compile("(?i)" + "spot_od_v131_public_training_" + "rerun")),
)
_PRIVATE_PATH_MARKER_BYTE_PATTERN_TEXTS: tuple[tuple[str, str], ...] = (
    ("nas_backslash_unc", r"(?i)" + re.escape(_BS) + r"+nas(?:[\\/]|$)"),
    ("nas_forward_unc", "(?i)" + "/" + "/" + "nas" + r"(?:/|$)"),
    ("windows_z_drive_absolute_path", r"(?i)(?<![A-Za-z0-9])Z:[\\/][A-Za-z0-9_. $-]"),
    ("windows_user_profile", r"(?i)(?<![A-Za-z0-9])C:[\\/]+Users[\\/]"),
    ("appdata_profile_path", r"(?i)\b" + "App" + "Data" + r"\b"),
    ("unix_user_home", r"(?i)(?<![A-Za-z0-9_])/(?:Users|home|mnt|Volumes)/"),
    ("local_training_worktree_name", "(?i)" + "spot_od_v131_public_training_" + "rerun"),
)
_PRIVATE_PATH_MARKER_BYTE_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = tuple(
    (label, re.compile(pattern_text.encode("ascii"), re.IGNORECASE))
    for label, pattern_text in _PRIVATE_PATH_MARKER_BYTE_PATTERN_TEXTS
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return v131.norm(path.relative_to(ROOT))
    except ValueError:
        return v131.norm(path)


def should_scan_public_text_member(member_name: str) -> bool:
    return Path(member_name).suffix.lower() in v131.TEXT_SUFFIXES


def private_path_marker_hits(text: str) -> list[str]:
    return [
        label
        for label, pattern in _PRIVATE_PATH_MARKER_PATTERNS
        if pattern.search(text)
    ]


def private_path_marker_byte_hits(payload: bytes) -> list[str]:
    return [
        label
        for label, pattern in _PRIVATE_PATH_MARKER_BYTE_PATTERNS
        if pattern.search(payload)
    ]


def is_checkpoint_or_model_member(member_name: str) -> bool:
    parts = member_name.split("/")
    return "checkpoints" in parts or Path(member_name).suffix.lower() in {".pt", ".pth", ".ckpt"}


def is_zip_payload(member_name: str, payload: bytes) -> bool:
    if Path(member_name).suffix.lower() == ".zip":
        return True
    return payload.startswith(b"PK\x03\x04")


def _scan_archive_payload(
    payload: bytes,
    *,
    archive_label: str,
    depth: int,
) -> dict[str, Any]:
    if depth >= MAX_NESTED_ZIP_SCAN_DEPTH:
        return {
            "text_members": [],
            "binary_members": [],
            "nested_archives": [],
            "failures": [
                {
                    "member": archive_label,
                    "problem": "nested_zip_depth_limit_exceeded",
                    "max_depth": MAX_NESTED_ZIP_SCAN_DEPTH,
                }
            ],
        }

    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as nested:
            return _scan_zip_for_private_path_markers(nested, archive_label=archive_label, depth=depth + 1)
    except zipfile.BadZipFile as exc:
        return {
            "text_members": [],
            "binary_members": [],
            "nested_archives": [],
            "failures": [
                {
                    "member": archive_label,
                    "problem": "bad_nested_zip",
                    "details": repr(exc),
                }
            ],
        }


def _scan_zip_for_private_path_markers(
    zf: zipfile.ZipFile,
    *,
    archive_label: str | None,
    depth: int,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    scanned_text: list[str] = []
    scanned_binary: list[str] = []
    nested_archives: list[str] = []

    crc_failure = zf.testzip()
    if crc_failure is not None:
        display_name = f"{archive_label}!{crc_failure}" if archive_label else crc_failure
        failures.append({"member": display_name, "problem": "crc_failure"})

    for info in sorted(zf.infolist(), key=lambda item: item.filename):
        member_name, name_problems = v131.member_name_problems(info.filename)
        if info.is_dir():
            continue
        display_name = f"{archive_label}!{member_name}" if archive_label else member_name

        if name_problems:
            failures.append(
                {
                    "member": display_name,
                    "problem": "unsafe_archive_member_name",
                    "problems": name_problems,
                }
            )

        name_hits = private_path_marker_hits(member_name)
        if name_hits:
            failures.append(
                {
                    "member": display_name,
                    "problem": "private_or_local_path_marker_in_member_name",
                    "markers": name_hits,
                }
            )

        try:
            payload = zf.read(info.filename)
        except (RuntimeError, zipfile.BadZipFile, KeyError) as exc:
            failures.append({"member": display_name, "problem": "member_read_failed", "details": repr(exc)})
            continue

        if should_scan_public_text_member(member_name):
            scanned_text.append(display_name)
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                failures.append({"member": display_name, "problem": f"utf8_decode_failed: {exc}"})
            else:
                hits = private_path_marker_hits(text)
                if hits:
                    failures.append(
                        {
                            "member": display_name,
                            "problem": "private_or_local_path_marker",
                            "markers": hits,
                        }
                    )
        else:
            scanned_binary.append(display_name)
            hits = private_path_marker_byte_hits(payload)
            if hits:
                failures.append(
                    {
                        "member": display_name,
                        "problem": "private_or_local_path_marker_in_binary_payload",
                        "markers": hits,
                    }
                )

        if is_zip_payload(member_name, payload):
            nested_archives.append(display_name)
            nested_result = _scan_archive_payload(payload, archive_label=display_name, depth=depth)
            scanned_text.extend(nested_result["text_members"])
            scanned_binary.extend(nested_result["binary_members"])
            nested_archives.extend(nested_result["nested_archives"])
            failures.extend(nested_result["failures"])

    return {
        "text_members": scanned_text,
        "binary_members": scanned_binary,
        "nested_archives": nested_archives,
        "failures": failures,
    }


def check_no_private_path_markers(zf: zipfile.ZipFile, members: list[str]) -> dict[str, Any]:
    del members
    scan = _scan_zip_for_private_path_markers(zf, archive_label=None, depth=0)
    marker_hit_counts: dict[str, int] = {}
    for failure in scan["failures"]:
        for hit in failure.get("markers", []):
            marker_hit_counts[hit] = marker_hit_counts.get(hit, 0) + 1

    return {
        "status": v131.status_from_failures(scan["failures"]),
        "scanned_text_member_count": len(scan["text_members"]),
        "scanned_binary_member_count": len(scan["binary_members"]),
        "nested_zip_member_count": len(scan["nested_archives"]),
        "marker_hit_counts": marker_hit_counts,
        "failure_count": len(scan["failures"]),
        "failures": scan["failures"],
    }


def check_required_members(member_set: set[str], required: tuple[str, ...], prefixes: tuple[str, ...]) -> dict[str, Any]:
    missing = [member for member in required if member not in member_set]
    prefix_file_counts = {
        prefix: sum(1 for member in member_set if member.startswith(prefix) and not member.endswith("/"))
        for prefix in prefixes
    }
    missing_prefixes = [prefix for prefix, count in prefix_file_counts.items() if count == 0]
    return {
        "status": "pass" if not missing and not missing_prefixes else "fail",
        "required_count": len(required),
        "missing_count": len(missing),
        "missing": missing,
        "required_prefixes": list(prefixes),
        "prefix_file_counts": prefix_file_counts,
        "missing_prefixes": missing_prefixes,
    }


def parse_main_json(zf: zipfile.ZipFile, member_set: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    json_members = {
        "zenodo_json": ".zenodo.json",
        "zenodo_metadata": "release/ZENODO_METADATA.json",
        "graph_summary": v131.GRAPH_SUMMARY,
        "local_summary": v131.LOCAL_SUMMARY,
        "mean_summary": v131.MEAN_SUMMARY,
        "comparison_intervals": v131.COMPARISON_INTERVALS,
        "tail_diagnostic_summary": v131.TAIL_SUMMARY,
    }
    parsed: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for label, member_name in json_members.items():
        if member_name not in member_set:
            failures.append({"label": label, "member": member_name, "problem": "missing"})
            continue
        try:
            parsed[label] = v131.read_member_json(zf, member_name)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append({"label": label, "member": member_name, "problem": repr(exc)})
    return parsed, {
        "status": v131.status_from_failures(failures),
        "parsed_members": sorted(parsed),
        "failure_count": len(failures),
        "failures": failures,
    }


def _related_identifiers_by_value(zenodo: dict[str, Any]) -> dict[str, set[str]]:
    related: dict[str, set[str]] = {}
    for item in zenodo.get("related_identifiers", []):
        if not isinstance(item, dict):
            continue
        identifier = item.get("identifier")
        relation = item.get("relation")
        if isinstance(identifier, str) and isinstance(relation, str):
            related.setdefault(identifier, set()).add(relation)
    return related


def check_main_metadata(parsed: dict[str, Any], zf: zipfile.ZipFile, member_set: set[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    zenodo = parsed.get("zenodo_json", {})
    metadata_record = parsed.get("zenodo_metadata", {})
    metadata = metadata_record.get("metadata", {}) if isinstance(metadata_record, dict) else {}
    record = metadata_record.get("record", {}) if isinstance(metadata_record, dict) else {}
    github_release = metadata_record.get("github_release", {}) if isinstance(metadata_record, dict) else {}

    v131.require_equal(failures, ".zenodo.json.version", zenodo.get("version") if isinstance(zenodo, dict) else None, TAG)
    v131.require_equal(failures, "release/ZENODO_METADATA.json.metadata.version", metadata.get("version"), TAG)
    v131.require_equal(failures, "release/ZENODO_METADATA.json.github_release.tag", github_release.get("tag"), TAG)
    v131.require_equal(failures, "release/ZENODO_METADATA.json.github_release.url", github_release.get("url"), RELEASE_URL)
    v131.require_equal(failures, "release/ZENODO_METADATA.json.record.status", record.get("status"), "pending_github_release_zenodo_import")
    v131.require_equal(failures, "release/ZENODO_METADATA.json.record.concept_doi", record.get("concept_doi"), CONCEPT_DOI)
    v131.require_equal(failures, "release/ZENODO_METADATA.json.record.previous_version_doi", record.get("previous_version_doi"), PRIOR_VERSION_DOI)

    if record.get("doi"):
        failures.append({"field": "release/ZENODO_METADATA.json.record.doi", "problem": "v1.3.2 DOI must not be invented before Zenodo import", "actual": record.get("doi")})

    related = _related_identifiers_by_value(zenodo if isinstance(zenodo, dict) else {})
    expected_related = {
        REPO_URL: "isSupplementedBy",
        RELEASE_URL: "isSupplementedBy",
        PRIOR_VERSION_DOI: "isNewVersionOf",
        CONCEPT_DOI: "isVersionOf",
        HISTORICAL_CITED_DOI: "cites",
    }
    for identifier, relation in expected_related.items():
        if relation not in related.get(identifier, set()):
            failures.append(
                {
                    "field": ".zenodo.json.related_identifiers",
                    "expected_identifier": identifier,
                    "expected_relation": relation,
                    "actual_relations": sorted(related.get(identifier, set())),
                }
            )

    for member_name in CURRENT_METADATA_MEMBERS:
        if member_name not in member_set:
            failures.append({"field": "current_metadata_members", "member": member_name, "problem": "missing"})
            continue
        try:
            text = v131.read_member_text(zf, member_name)
        except UnicodeDecodeError as exc:
            failures.append({"field": member_name, "problem": f"utf8_decode_failed: {exc}"})
            continue
        unknown_dois = sorted(set(DOI_RE.findall(text)) - ALLOWED_METADATA_DOIS)
        if unknown_dois:
            failures.append({"field": member_name, "problem": "unexpected_or_invented_doi", "unexpected_dois": unknown_dois})
        if "Current release: `v1.3.1" in text:
            failures.append({"field": member_name, "problem": "stale_current_release_text"})
        if TAG not in text:
            failures.append({"field": member_name, "problem": "missing_v132_tag"})

    return {
        "status": v131.status_from_failures(failures),
        "version": TAG,
        "previous_version_doi": PRIOR_VERSION_DOI,
        "concept_doi": CONCEPT_DOI,
        "historical_cited_doi": HISTORICAL_CITED_DOI,
        "failure_count": len(failures),
        "failures": failures,
    }


def check_main_payload_boundaries(member_set: set[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if LEGACY_V131_ARCHIVE_MEMBER in member_set:
        failures.append(
            {
                "member": LEGACY_V131_ARCHIVE_MEMBER,
                "problem": "legacy_v131_zip_must_not_be_embedded",
            }
        )
    for member_name in sorted(member_set):
        if is_checkpoint_or_model_member(member_name):
            failures.append({"member": member_name, "problem": "checkpoint_or_model_weight_member"})
    return {
        "status": v131.status_from_failures(failures),
        "legacy_v131_archive_member": LEGACY_V131_ARCHIVE_MEMBER,
        "checkpoint_policy": "main package is checkpoint-free; saved metric replay and training-input inspection only",
        "failure_count": len(failures),
        "failures": failures,
    }


def check_current_text_boundaries(zf: zipfile.ZipFile, member_set: set[str]) -> dict[str, Any]:
    required_phrases = (
        "not public precise-reference validation",
        "not independent-machine reproduction",
        "not a full raw/training/all-filter rerun",
    )
    failures: list[dict[str, Any]] = []
    for member_name in V132_RELEASE_DOCS:
        if member_name not in member_set:
            failures.append({"member": member_name, "problem": "missing"})
            continue
        text = v131.read_member_text(zf, member_name)
        lower_text = normalize_text(text)
        for phrase in required_phrases:
            if phrase not in lower_text:
                failures.append({"member": member_name, "problem": "missing_boundary_phrase", "phrase": phrase})
    return {
        "status": v131.status_from_failures(failures),
        "checked_members": list(V132_RELEASE_DOCS),
        "required_phrases": list(required_phrases),
        "failure_count": len(failures),
        "failures": failures,
    }


def check_main_archive(archive_rel: str) -> dict[str, Any]:
    archive_path = repo_path(archive_rel)
    archive = v131.archive_record(archive_path)
    checks: dict[str, Any] = {}

    if not archive_path.is_file():
        checks["archive_members"] = {"status": "fail", "failures": [{"problem": "archive_missing"}]}
        for name in ("required_members", "json_parse", "metadata", "text_boundaries", "private_path_hygiene", "metrics", "payload_boundaries", "extracted_help_smoke"):
            checks[name] = v131.blocked_check("archive_missing")
        return {"status": "fail", "archive": archive, "checks": checks}

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            member_check = v131.check_archive_members(zf)
            checks["archive_members"] = {k: val for k, val in member_check.items() if k != "members"}
            members = member_check["members"]
            member_set = set(members)
            checks["required_members"] = check_required_members(member_set, REQUIRED_MAIN_MEMBERS, REQUIRED_MAIN_PREFIXES)
            checks["payload_boundaries"] = check_main_payload_boundaries(member_set)

            if member_check["status"] != "pass":
                for name in ("json_parse", "metadata", "text_boundaries", "private_path_hygiene", "metrics", "extracted_help_smoke"):
                    checks[name] = v131.blocked_check("unsafe_or_invalid_archive_members")
            else:
                parsed, parse_check = parse_main_json(zf, member_set)
                checks["json_parse"] = parse_check
                if parse_check["status"] == "pass":
                    checks["metadata"] = check_main_metadata(parsed, zf, member_set)
                    checks["metrics"] = v131.check_metrics(parsed)
                else:
                    checks["metadata"] = v131.blocked_check("json_parse_failed")
                    checks["metrics"] = v131.blocked_check("json_parse_failed")
                checks["text_boundaries"] = check_current_text_boundaries(zf, member_set)
                checks["private_path_hygiene"] = check_no_private_path_markers(zf, members)
                checks["extracted_help_smoke"] = v131.check_extracted_help_smoke(zf, member_set)
    except zipfile.BadZipFile as exc:
        checks["archive_members"] = {"status": "fail", "failures": [{"problem": "bad_zip", "details": repr(exc)}]}
        for name in ("required_members", "json_parse", "metadata", "text_boundaries", "private_path_hygiene", "metrics", "payload_boundaries", "extracted_help_smoke"):
            checks[name] = v131.blocked_check("bad_zip")

    status = "pass" if all(check.get("status") == "pass" for check in checks.values()) else "fail"
    return {"status": status, "archive": archive, "checks": checks}


def check_training_archive_members(members: list[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    file_member_set = set(members)

    for member in sorted(file_member_set):
        parts = member.split("/")
        suffix = Path(member).suffix.lower()
        if "checkpoints" in parts or suffix in {".pt", ".pth", ".ckpt"}:
            failures.append({"member": member, "problem": "checkpoint_or_model_weight_member"})

    missing_manifests = [
        member
        for member in (TRAINING_MANIFEST_JSON, TRAINING_MANIFEST_MD)
        if member not in file_member_set
    ]
    for member in missing_manifests:
        failures.append({"member": member, "problem": "missing_training_manifest"})

    for source_dir in EXPECTED_TRAINING_SOURCE_DIRS:
        for filename in TRAINING_ROOT_FILES:
            member = f"{source_dir}/{filename}"
            if member not in file_member_set:
                failures.append({"member": member, "problem": "missing_training_root_file"})
        for scenario in TRAINING_SCENARIOS:
            for filename in TRAINING_SCENARIO_FILES:
                member = f"{source_dir}/{scenario}/{filename}"
                if member not in file_member_set:
                    failures.append({"member": member, "problem": "missing_training_scenario_file"})

    payload_files = [member for member in file_member_set if member.startswith("results/")]
    if len(payload_files) != EXPECTED_TRAINING_PAYLOAD_FILE_COUNT:
        failures.append(
            {
                "problem": "training_payload_file_count_mismatch",
                "expected": EXPECTED_TRAINING_PAYLOAD_FILE_COUNT,
                "actual": len(payload_files),
            }
        )
    if len(file_member_set) != EXPECTED_TRAINING_TOTAL_FILE_COUNT:
        failures.append(
            {
                "problem": "training_total_file_count_mismatch",
                "expected": EXPECTED_TRAINING_TOTAL_FILE_COUNT,
                "actual": len(file_member_set),
            }
        )

    return {
        "status": v131.status_from_failures(failures),
        "expected_source_directory_count": len(EXPECTED_TRAINING_SOURCE_DIRS),
        "expected_payload_file_count": EXPECTED_TRAINING_PAYLOAD_FILE_COUNT,
        "expected_total_file_count": EXPECTED_TRAINING_TOTAL_FILE_COUNT,
        "actual_payload_file_count": len(payload_files),
        "actual_total_file_count": len(file_member_set),
        "failure_count": len(failures),
        "failures": failures,
    }


def check_training_manifest(zf: zipfile.ZipFile, member_set: set[str]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if TRAINING_MANIFEST_JSON not in member_set:
        return {"status": "fail", "failures": [{"problem": "manifest_json_missing"}]}
    if TRAINING_MANIFEST_MD not in member_set:
        failures.append({"member": TRAINING_MANIFEST_MD, "problem": "manifest_md_missing"})

    try:
        manifest = v131.read_member_json(zf, TRAINING_MANIFEST_JSON)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "fail", "failures": [{"problem": "manifest_json_parse_failed", "details": repr(exc)}]}

    v131.require_equal(failures, "training_manifest.release_tag", manifest.get("release_tag"), TAG)
    v131.require_equal(failures, "training_manifest.release_url", manifest.get("release_url"), RELEASE_URL)
    v131.require_equal(failures, "training_manifest.previous_zenodo_version_doi", manifest.get("previous_zenodo_version_doi"), PRIOR_VERSION_DOI)
    v131.require_equal(failures, "training_manifest.zenodo_concept_doi", manifest.get("zenodo_concept_doi"), CONCEPT_DOI)
    v131.require_equal(failures, "training_manifest.checkpoints_omitted", manifest.get("checkpoints_omitted"), True)
    v131.require_equal(failures, "training_manifest.payload_file_count", manifest.get("payload_file_count"), EXPECTED_TRAINING_PAYLOAD_FILE_COUNT)
    v131.require_equal(failures, "training_manifest.total_zip_file_count", manifest.get("total_zip_file_count"), EXPECTED_TRAINING_TOTAL_FILE_COUNT)
    v131.require_equal(failures, "training_manifest.source_directory_count", manifest.get("source_directory_count"), len(EXPECTED_TRAINING_SOURCE_DIRS))

    source_dirs = manifest.get("source_directories")
    if source_dirs != list(EXPECTED_TRAINING_SOURCE_DIRS):
        failures.append(
            {
                "field": "training_manifest.source_directories",
                "expected": list(EXPECTED_TRAINING_SOURCE_DIRS),
                "actual": source_dirs,
            }
        )

    if TRAINING_MANIFEST_MD in member_set:
        md_text = normalize_text(v131.read_member_text(zf, TRAINING_MANIFEST_MD))
        for phrase in ("checkpoint-free", "checkpoints are omitted", "not full raw/training/all-filter reproduction"):
            if phrase not in md_text:
                failures.append({"member": TRAINING_MANIFEST_MD, "problem": "missing_manifest_scope_phrase", "phrase": phrase})

    return {
        "status": v131.status_from_failures(failures),
        "manifest": {
            "release_tag": manifest.get("release_tag"),
            "source_directory_count": manifest.get("source_directory_count"),
            "payload_file_count": manifest.get("payload_file_count"),
            "total_zip_file_count": manifest.get("total_zip_file_count"),
            "checkpoints_omitted": manifest.get("checkpoints_omitted"),
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def check_training_archive(training_archive_rel: str) -> dict[str, Any]:
    archive_path = repo_path(training_archive_rel)
    archive = v131.archive_record(archive_path)
    checks: dict[str, Any] = {}

    if not archive_path.is_file():
        checks["archive_members"] = {"status": "fail", "failures": [{"problem": "archive_missing"}]}
        checks["training_members"] = v131.blocked_check("archive_missing")
        checks["training_manifest"] = v131.blocked_check("archive_missing")
        checks["private_path_hygiene"] = v131.blocked_check("archive_missing")
        return {"status": "fail", "archive": archive, "checks": checks}

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            member_check = v131.check_archive_members(zf)
            checks["archive_members"] = {k: val for k, val in member_check.items() if k != "members"}
            members = member_check["members"]
            member_set = set(members)
            if member_check["status"] != "pass":
                checks["training_members"] = v131.blocked_check("unsafe_or_invalid_archive_members")
                checks["training_manifest"] = v131.blocked_check("unsafe_or_invalid_archive_members")
                checks["private_path_hygiene"] = v131.blocked_check("unsafe_or_invalid_archive_members")
            else:
                checks["training_members"] = check_training_archive_members(members)
                checks["training_manifest"] = check_training_manifest(zf, member_set)
                checks["private_path_hygiene"] = check_no_private_path_markers(zf, members)
    except zipfile.BadZipFile as exc:
        checks["archive_members"] = {"status": "fail", "failures": [{"problem": "bad_zip", "details": repr(exc)}]}
        checks["training_members"] = v131.blocked_check("bad_zip")
        checks["training_manifest"] = v131.blocked_check("bad_zip")
        checks["private_path_hygiene"] = v131.blocked_check("bad_zip")

    status = "pass" if all(check.get("status") == "pass" for check in checks.values()) else "fail"
    return {"status": status, "archive": archive, "checks": checks}


def build_result(archive_rel: str = DEFAULT_ARCHIVE_REL, training_archive_rel: str = DEFAULT_TRAINING_ARCHIVE_REL) -> dict[str, Any]:
    main_result = check_main_archive(archive_rel)
    training_result = check_training_archive(training_archive_rel)
    status = "pass" if main_result["status"] == "pass" and training_result["status"] == "pass" else "fail"
    return {
        "schema_version": "v132_public_reproduction_package_verification.v1",
        "status": status,
        "scope": "v1.3.2_public_reproduction_package_integrity",
        "scope_boundary": SCOPE_BOUNDARY,
        "package": {
            "version": TAG,
            "release_url": RELEASE_URL,
            "previous_version_doi": PRIOR_VERSION_DOI,
            "concept_doi": CONCEPT_DOI,
            "v132_version_doi_status": "pending_zenodo_import",
        },
        "main_package": main_result,
        "training_inputs": training_result,
    }


def write_reports(result: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    main = result.get("main_package", {})
    training = result.get("training_inputs", {})
    main_checks = main.get("checks", {})
    training_checks = training.get("checks", {})
    metrics = main_checks.get("metrics", {}).get("summary", {})
    graph_all = metrics.get("graph_all_non_development", {})
    graph_fresh = metrics.get("graph_fresh", {})

    lines = [
        "# v1.3.2 Public Reproduction Package Verification",
        "",
        f"Status: **{str(result.get('status')).upper()}**",
        "",
        "## Scope Boundary",
        str(result.get("scope_boundary", SCOPE_BOUNDARY)),
        "",
        "## Main Archive",
        f"- Path: `{main.get('archive', {}).get('path')}`",
        f"- Exists: `{main.get('archive', {}).get('exists')}`",
        f"- Bytes: `{main.get('archive', {}).get('bytes')}`",
        f"- SHA-256: `{main.get('archive', {}).get('sha256')}`",
        "",
        "## Training-Input Archive",
        f"- Path: `{training.get('archive', {}).get('path')}`",
        f"- Exists: `{training.get('archive', {}).get('exists')}`",
        f"- Bytes: `{training.get('archive', {}).get('bytes')}`",
        f"- SHA-256: `{training.get('archive', {}).get('sha256')}`",
        "",
        "## Metadata",
        f"- Version/tag: `{TAG}`",
        "- v1.3.2 Zenodo version DOI: pending Zenodo GitHub import; not claimed here.",
        f"- Previous Zenodo version DOI: `{PRIOR_VERSION_DOI}`",
        f"- Zenodo concept DOI: `{CONCEPT_DOI}`",
        "",
        "## Main Checks",
    ]
    for name in (
        "archive_members",
        "required_members",
        "json_parse",
        "metadata",
        "text_boundaries",
        "private_path_hygiene",
        "metrics",
        "payload_boundaries",
        "extracted_help_smoke",
    ):
        check = main_checks.get(name, {})
        lines.append(f"- {name}: **{str(check.get('status')).upper()}**")
    lines += ["", "## Training Checks"]
    for name in ("archive_members", "training_members", "training_manifest", "private_path_hygiene"):
        check = training_checks.get(name, {})
        lines.append(f"- {name}: **{str(check.get('status')).upper()}**")
    lines += [
        "",
        "## Metric Readback",
        (
            "- All non-development graph selector: "
            f"`{graph_all.get('selector_observed_step_rmse_m')}` m vs best retained "
            f"`{graph_all.get('best_single_observed_step_rmse_m')}` m, gain "
            f"`{graph_all.get('gain_vs_best_single_percent')}`%."
        ),
        (
            "- Fresh graph selector: "
            f"`{graph_fresh.get('selector_observed_step_rmse_m')}` m vs best retained "
            f"`{graph_fresh.get('best_single_observed_step_rmse_m')}` m, gain "
            f"`{graph_fresh.get('gain_vs_best_single_percent')}`%."
        ),
        "",
        "## Outputs",
        f"- JSON: `{rel(json_out)}`",
        f"- Markdown: `{rel(md_out)}`",
    ]

    failures: list[str] = []
    for group_name, checks in (("main", main_checks), ("training", training_checks)):
        for name, check in checks.items():
            if check.get("status") not in {"pass", None}:
                failures.append(f"- {group_name}.{name}: **{str(check.get('status')).upper()}**; see JSON report.")
    if failures:
        lines += ["", "## Failures", *failures]

    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify v1.3.2 public reproduction package ZIPs.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE_REL)
    parser.add_argument("--training-archive", default=DEFAULT_TRAINING_ARCHIVE_REL)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", default=DEFAULT_MD_OUT)
    parser.add_argument("--check-only", action="store_true", help="Run checks without writing JSON/Markdown reports.")
    args = parser.parse_args()

    result = build_result(args.archive, args.training_archive)
    json_out = repo_path(args.json_out)
    md_out = repo_path(args.md_out)
    if not args.check_only:
        write_reports(result, json_out, md_out)

    summary = {
        "status": result["status"],
        "main_archive": result["main_package"]["archive"],
        "training_archive": result["training_inputs"]["archive"],
        "json": rel(json_out),
        "markdown": rel(md_out),
        "check_only": args.check_only,
        "scope_boundary": SCOPE_BOUNDARY,
        "main_checks": {name: check.get("status") for name, check in result["main_package"]["checks"].items()},
        "training_checks": {name: check.get("status") for name, check in result["training_inputs"]["checks"].items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
