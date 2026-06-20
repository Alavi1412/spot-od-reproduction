from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_ARCHIVE = "release/spot_od_v1_1_0_supplement_review_archive.zip"
DEFAULT_MANIFEST = "release/SUPPLEMENTARY_MANIFEST.json"
DEFAULT_VERIFIER_JSON = "results/validation/github_actions_archive_extracted_reproduction.json"
DEFAULT_VERIFIER_MD = "results/validation/github_actions_archive_extracted_reproduction.md"
DEFAULT_JSON_OUT = "results/validation/github_actions_reproduction_attestation.json"
DEFAULT_MD_OUT = "results/validation/github_actions_reproduction_attestation.md"

EXPECTED_ARCHIVE_SHA256 = "9d6f34599b238749bfe1cc3e8bdda4d6a3034ee662f3e8b2f7c7cafc49831b3f"
EXPECTED_MANIFEST_SHA256 = "2d7a05dee73d83b436dcc88ebcd40f5d7caeaacbfc70ee5d170040474f99ff72"
EXPECTED_ARCHIVE_BYTES = 58908807
EXPECTED_ARCHIVE_MEMBERS = 1014

SCOPE_BOUNDARY = (
    "Archive-extracted reproduction only: archive integrity, manifest-indexed "
    "artifact checks, active manuscript artifact regeneration from archived "
    "results, and one archived-input public OD slice rerun. This does not claim "
    "full raw/training/all-filter reruns, live public-data retrieval, operational "
    "POD validation, or third-party independent validation."
)

GITHUB_ENV_KEYS = [
    "GITHUB_ACTIONS",
    "GITHUB_SERVER_URL",
    "GITHUB_WORKFLOW",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_NUMBER",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_JOB",
    "GITHUB_REPOSITORY",
    "GITHUB_REF",
    "GITHUB_REF_NAME",
    "GITHUB_REF_TYPE",
    "GITHUB_SHA",
    "GITHUB_ACTOR",
    "GITHUB_EVENT_NAME",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "RUNNER_NAME",
]


def repo_path(path_text: str) -> Path:
    return Path(path_text).resolve()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": rel(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }


def zip_member_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with zipfile.ZipFile(path) as archive:
        return len(archive.infolist())


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def github_context() -> dict[str, str]:
    context = {key: os.environ.get(key, "") for key in GITHUB_ENV_KEYS}
    server_url = context.get("GITHUB_SERVER_URL") or "https://github.com"
    repository = context.get("GITHUB_REPOSITORY", "")
    context["repository_url"] = f"{server_url}/{repository}" if repository else ""
    run_id = context.get("GITHUB_RUN_ID", "")
    context["workflow_run_url"] = f"{context['repository_url']}/actions/runs/{run_id}" if repository and run_id else ""
    return context


def hash_check(record: dict[str, Any], expected_sha256: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    checks = {
        "exists": bool(record.get("exists")),
        "sha256_matches": record.get("sha256") == expected_sha256 if expected_sha256 else None,
        "bytes_match": record.get("bytes") == expected_bytes if expected_bytes is not None else None,
    }
    required = [checks["exists"]]
    if expected_sha256:
        required.append(checks["sha256_matches"])
    if expected_bytes is not None:
        required.append(checks["bytes_match"])
    checks["status"] = "pass" if all(required) else "fail"
    return checks


def build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = repo_path(args.archive)
    manifest_path = repo_path(args.manifest)
    verifier_json_path = repo_path(args.verifier_json)
    verifier_md_path = repo_path(args.verifier_md)

    archive = file_record(archive_path)
    manifest = file_record(manifest_path)
    verifier_json = file_record(verifier_json_path)
    verifier_md = file_record(verifier_md_path)
    verifier_payload = read_json(verifier_json_path) or {}
    archive["member_count"] = zip_member_count(archive_path)
    github = github_context()

    checks = {
        "archive_hash": hash_check(archive, args.expected_archive_sha256, EXPECTED_ARCHIVE_BYTES),
        "manifest_hash": hash_check(manifest, args.expected_manifest_sha256),
        "archive_member_count": {
            "status": "pass" if archive.get("member_count") == EXPECTED_ARCHIVE_MEMBERS else "fail",
            "expected": EXPECTED_ARCHIVE_MEMBERS,
            "actual": archive.get("member_count"),
        },
        "verifier_json_exists": {"status": "pass" if verifier_json.get("exists") else "fail"},
        "verifier_md_exists": {"status": "pass" if verifier_md.get("exists") else "fail"},
        "verifier_status": {
            "status": "pass" if verifier_payload.get("status") == "pass" else "fail",
            "actual": verifier_payload.get("status"),
        },
    }
    status = "pass" if all(check.get("status") == "pass" for check in checks.values()) else "fail"

    return {
        "schema_version": "github_actions_reproduction_attestation_v1",
        "status": status,
        "scope": "archive_extracted_reproduction_check",
        "scope_boundary": SCOPE_BOUNDARY,
        "github_release_context": {
            "repository": github.get("GITHUB_REPOSITORY", ""),
            "repository_url": github.get("repository_url", ""),
            "workflow": github.get("GITHUB_WORKFLOW", ""),
            "workflow_run_url": github.get("workflow_run_url", ""),
            "event_name": github.get("GITHUB_EVENT_NAME", ""),
            "ref": github.get("GITHUB_REF", ""),
            "ref_name": github.get("GITHUB_REF_NAME", ""),
            "ref_type": github.get("GITHUB_REF_TYPE", ""),
            "sha": github.get("GITHUB_SHA", ""),
        },
        "expected": {
            "archive_sha256": args.expected_archive_sha256,
            "archive_bytes": EXPECTED_ARCHIVE_BYTES,
            "archive_members": EXPECTED_ARCHIVE_MEMBERS,
            "manifest_sha256": args.expected_manifest_sha256,
        },
        "files": {
            "archive": archive,
            "manifest": manifest,
            "verifier_json": verifier_json,
            "verifier_md": verifier_md,
        },
        "checks": checks,
        "verifier_summary": {
            "status": verifier_payload.get("status"),
            "claim_boundary": verifier_payload.get("claim_boundary"),
            "manifest": verifier_payload.get("manifest"),
        },
        "github": github,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "generated_utc": os.environ.get("CI_GENERATED_UTC", "not_recorded"),
    }


def write_markdown(attestation: dict[str, Any], path: Path) -> None:
    files = attestation["files"]
    checks = attestation["checks"]
    github = attestation["github"]
    lines = [
        "# GitHub Actions Reproduction Attestation",
        "",
        f"Status: **{attestation['status'].upper()}**",
        "",
        "## Scope Boundary",
        attestation["scope_boundary"],
        "",
        "## GitHub Context",
        f"- Repository: `{github.get('GITHUB_REPOSITORY', '')}`",
        f"- Repository URL: `{github.get('repository_url', '')}`",
        f"- Workflow: `{github.get('GITHUB_WORKFLOW', '')}`",
        f"- Run ID: `{github.get('GITHUB_RUN_ID', '')}`",
        f"- Run URL: `{github.get('workflow_run_url', '')}`",
        f"- Run attempt: `{github.get('GITHUB_RUN_ATTEMPT', '')}`",
        f"- Ref: `{github.get('GITHUB_REF', '')}`",
        f"- Ref type: `{github.get('GITHUB_REF_TYPE', '')}`",
        f"- SHA: `{github.get('GITHUB_SHA', '')}`",
        f"- Runner: `{github.get('RUNNER_OS', '')}/{github.get('RUNNER_ARCH', '')}`",
        "",
        "## File Hashes",
        f"- Archive: `{files['archive']['path']}`",
        f"  - SHA-256: `{files['archive']['sha256']}`",
        f"  - Bytes: `{files['archive']['bytes']}`",
        f"  - Members: `{files['archive']['member_count']}`",
        f"- Manifest: `{files['manifest']['path']}`",
        f"  - SHA-256: `{files['manifest']['sha256']}`",
        f"- Verifier JSON: `{files['verifier_json']['path']}`",
        f"  - SHA-256: `{files['verifier_json']['sha256']}`",
        f"- Verifier Markdown: `{files['verifier_md']['path']}`",
        f"  - SHA-256: `{files['verifier_md']['sha256']}`",
        "",
        "## Checks",
    ]
    for name in sorted(checks):
        lines.append(f"- {name}: **{checks[name]['status'].upper()}**")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write GitHub Actions archive-extracted reproduction attestation.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--verifier-json", default=DEFAULT_VERIFIER_JSON)
    parser.add_argument("--verifier-md", default=DEFAULT_VERIFIER_MD)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", default=DEFAULT_MD_OUT)
    parser.add_argument("--expected-archive-sha256", default=EXPECTED_ARCHIVE_SHA256)
    parser.add_argument("--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    args = parser.parse_args()

    attestation = build_attestation(args)
    json_out = repo_path(args.json_out)
    md_out = repo_path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(attestation, md_out)
    print(json.dumps({"status": attestation["status"], "json_out": rel(json_out), "md_out": rel(md_out)}, sort_keys=True))
    return 0 if attestation["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
