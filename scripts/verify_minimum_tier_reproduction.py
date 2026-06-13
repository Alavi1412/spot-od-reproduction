#!/usr/bin/env python3
"""Minimum-tier release reproduction check.

This verifier is intentionally narrow: it checks the submitted evidence
package inventory, checksums, review archive membership, and claim-map coverage
without regenerating tables, rerunning filters, or training learned estimators.
The report is kept deterministic so it can itself be indexed by the release
manifest without creating a timestamp/hash churn cycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = "release/SUPPLEMENTARY_MANIFEST.json"
DEFAULT_JSON_OUT = "results/validation/minimum_tier_reproduction_check.json"
DEFAULT_MD_OUT = "results/validation/minimum_tier_reproduction_check.md"
GENERATED_REPORT_RELS = {DEFAULT_JSON_OUT, DEFAULT_MD_OUT}


def norm(path: str | Path) -> str:
    return str(path).replace("\\", "/").strip()


def rel(path: Path) -> str:
    try:
        return norm(path.relative_to(ROOT))
    except ValueError:
        return norm(path)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_member_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    archive = manifest.get("review_archive", {})
    archive_path = ROOT / Path(norm(archive.get("path", "")))
    records: dict[str, dict[str, Any]] = {}
    if not archive_path.is_file():
        return records
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = norm(info.filename)
                data = zf.read(info)
                records[name] = {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
    except zipfile.BadZipFile:
        return {}
    return records


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = manifest.get("artifact_groups", {})
    if not isinstance(groups, dict):
        return rows
    for group, entries in groups.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            row = dict(entry)
            row["group"] = group
            row["norm_path"] = norm(str(entry.get("path", "")))
            rows.append(row)
    return rows


def resolved_artifact_record(
    path: str,
    archive_records: dict[str, dict[str, Any]],
    *,
    prefer_archive: bool = False,
) -> dict[str, Any]:
    archive_record = archive_records.get(path)
    if prefer_archive and archive_record is not None:
        return {
            "exists": True,
            "bytes": archive_record.get("bytes"),
            "sha256": archive_record.get("sha256"),
            "source": "review_archive_member",
        }
    file_path = ROOT / Path(path)
    if file_path.is_file():
        return {
            "exists": True,
            "bytes": file_path.stat().st_size,
            "sha256": sha256(file_path),
            "source": "workspace_file",
        }
    if archive_record is not None:
        return {
            "exists": True,
            "bytes": archive_record.get("bytes"),
            "sha256": archive_record.get("sha256"),
            "source": "review_archive_member",
        }
    return {"exists": False, "bytes": None, "sha256": None, "source": None}


def check_artifacts(entries: list[dict[str, Any]], archive_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    duplicate_paths: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        path = entry["norm_path"]
        if path in seen and path not in duplicate_paths:
            duplicate_paths.append(path)
        seen.add(path)
        resolved = resolved_artifact_record(
            path,
            archive_records,
            prefer_archive=bool(entry.get("review_content_redacted")),
        )
        exists = bool(resolved["exists"])
        if path in GENERATED_REPORT_RELS and not exists:
            continue
        actual_bytes = resolved["bytes"]
        actual_sha = resolved["sha256"]
        expected_exists = bool(entry.get("exists"))
        if path in GENERATED_REPORT_RELS and not expected_exists:
            continue
        expected_bytes = entry.get("bytes")
        expected_sha = entry.get("sha256")
        problems = []
        if not expected_exists:
            problems.append("manifest_entry_not_present")
        if not exists:
            problems.append("file_missing")
        if exists and expected_bytes != actual_bytes:
            problems.append("bytes_mismatch")
        if exists and expected_sha != actual_sha:
            problems.append("sha256_mismatch")
        if problems:
            failures.append(
                {
                    "path": path,
                    "group": entry.get("group"),
                    "problems": problems,
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                    "resolved_from": resolved["source"],
                }
            )
    return {
        "status": "pass" if not failures and not duplicate_paths else "fail",
        "failure_count": len(failures),
        "duplicate_path_count": len(duplicate_paths),
        "failures": failures,
        "duplicate_paths": sorted(duplicate_paths),
    }


def check_manifest_summary(
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    archive_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    actual_count = len(entries)
    actual_present = sum(
        1
        for entry in entries
        if resolved_artifact_record(
            entry["norm_path"],
            archive_records,
            prefer_archive=bool(entry.get("review_content_redacted")),
        )["exists"]
    )
    if manifest.get("artifact_count") != actual_count:
        failures.append("artifact_count_mismatch")
    if manifest.get("artifacts_present") != actual_present:
        failures.append("artifacts_present_mismatch")
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def check_archive(manifest: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    archive = manifest.get("review_archive", {})
    archive_path = ROOT / Path(norm(archive.get("path", "")))
    failures: list[str] = []
    if not archive_path.is_file():
        return {"status": "fail", "failures": ["archive_missing"]}
    if archive.get("sha256") != sha256(archive_path):
        failures.append("archive_sha256_mismatch")
    expected_members = sorted(entry["norm_path"] for entry in entries if entry.get("exists"))
    member_failures: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            actual_members = sorted(norm(name) for name in zf.namelist() if not name.endswith("/"))
            missing = sorted(set(expected_members) - set(actual_members))
            extra = sorted(set(actual_members) - set(expected_members))
            if missing:
                failures.append("archive_missing_indexed_members")
            if extra:
                failures.append("archive_extra_members")
            expected_by_path = {entry["norm_path"]: entry for entry in entries}
            for member in actual_members:
                entry = expected_by_path.get(member)
                if not entry:
                    continue
                member_sha = hashlib.sha256(zf.read(member)).hexdigest()
                if member_sha != entry.get("sha256"):
                    member_failures.append(
                        {
                            "path": member,
                            "expected_sha256": entry.get("sha256"),
                            "archive_member_sha256": member_sha,
                        }
                    )
            if member_failures:
                failures.append("archive_member_sha256_mismatch")
    except zipfile.BadZipFile:
        return {"status": "fail", "failures": ["archive_bad_zip"]}
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "member_sha256_failure_count": len(member_failures),
        "member_sha256_failures": member_failures,
    }


def check_claim_map(manifest: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    claim_map = manifest.get("claim_to_artifact_map", {})
    tiers = manifest.get("regeneration_tiers", {})
    claim_tiers = manifest.get("claim_to_regeneration_tier_map", {})
    indexed = {entry["norm_path"] for entry in entries}
    archive_path = norm(manifest.get("review_archive", {}).get("path", ""))
    allowed_top_level = {MANIFEST_REL, archive_path}
    failures: list[dict[str, Any]] = []
    if not isinstance(claim_map, dict) or not claim_map:
        failures.append({"claim": None, "problem": "claim_to_artifact_map_missing"})
    for claim, paths in sorted(claim_map.items()):
        if not isinstance(paths, list) or not paths:
            failures.append({"claim": claim, "problem": "claim_has_no_artifacts"})
            continue
        for path in paths:
            npath = norm(path)
            if npath not in indexed and npath not in allowed_top_level:
                failures.append(
                    {
                        "claim": claim,
                        "problem": "claim_artifact_not_indexed",
                        "path": npath,
                    }
                )
    if not isinstance(claim_tiers, dict) or set(claim_tiers) != set(claim_map):
        failures.append(
            {
                "claim": None,
                "problem": "claim_to_regeneration_tier_map_key_mismatch",
            }
        )
    else:
        tier_keys = set(tiers)
        for claim, tier_list in sorted(claim_tiers.items()):
            if not isinstance(tier_list, list) or not tier_list:
                failures.append({"claim": claim, "problem": "claim_has_no_regeneration_tiers"})
                continue
            unknown = sorted(str(tier) for tier in tier_list if str(tier) not in tier_keys)
            if unknown:
                failures.append(
                    {
                        "claim": claim,
                        "problem": "claim_has_unknown_regeneration_tiers",
                        "unknown_tiers": unknown,
                    }
                )
    return {
        "status": "pass" if not failures else "fail",
        "claim_count": len(claim_map) if isinstance(claim_map, dict) else 0,
        "failure_count": len(failures),
        "failures": failures,
    }


def write_reports(result: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Minimum-Tier Reproduction Check",
        "",
        f"Status: **{result['status'].upper()}**",
        "",
        "This check validates the release manifest inventory, recorded SHA-256",
        "digests, review archive membership, and claim-to-artifact map coverage.",
        "It does not rerun estimator training, table generation, or filter",
        "estimation.",
        "",
        "## Checks",
        f"- Manifest artifact count summary: **{result['checks']['manifest_summary']['status'].upper()}**",
        f"- Manifest artifact presence and checksums: **{result['checks']['artifacts']['status'].upper()}**",
        f"- Review archive digest and member checksums: **{result['checks']['review_archive']['status'].upper()}**",
        f"- Claim-map and regeneration-tier coverage: **{result['checks']['claim_map']['status'].upper()}**",
        "",
        "## Outputs",
        f"- JSON: `{rel(json_out)}`",
        f"- Markdown: `{rel(md_out)}`",
    ]
    md_out.write_text("\n".join(md) + "\n", encoding="utf-8")


def build_result() -> dict[str, Any]:
    manifest_path = ROOT / MANIFEST_REL
    manifest = read_json(manifest_path)
    entries = manifest_entries(manifest)
    archive_records = archive_member_records(manifest)
    manifest_summary_check = check_manifest_summary(manifest, entries, archive_records)
    artifact_check = check_artifacts(entries, archive_records)
    archive_check = check_archive(manifest, entries)
    claim_check = check_claim_map(manifest, entries)
    checks = {
        "manifest_summary": manifest_summary_check,
        "artifacts": artifact_check,
        "review_archive": archive_check,
        "claim_map": claim_check,
    }
    status = "pass" if all(check["status"] == "pass" for check in checks.values()) else "fail"
    return {
        "status": status,
        "scope": "minimum_integrity_check",
        "requires_retraining": False,
        "manifest": MANIFEST_REL,
        "package": manifest.get("package"),
        "version": manifest.get("version"),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify minimum-tier release integrity.")
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", default=DEFAULT_MD_OUT)
    parser.add_argument("--check-only", action="store_true", help="Do not write reports.")
    args = parser.parse_args()

    result = build_result()
    json_out = ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
    md_out = ROOT / args.md_out if not Path(args.md_out).is_absolute() else Path(args.md_out)
    if not args.check_only:
        write_reports(result, json_out, md_out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "json": rel(json_out),
                "markdown": rel(md_out),
                "manifest_summary_check": result["checks"]["manifest_summary"]["status"],
                "artifact_check": result["checks"]["artifacts"]["status"],
                "review_archive_check": result["checks"]["review_archive"]["status"],
                "claim_map_check": result["checks"]["claim_map"]["status"],
                "requires_retraining": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
