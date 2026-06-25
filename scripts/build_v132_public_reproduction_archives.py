#!/usr/bin/env python3
"""Build the v1.3.2 public reproduction release archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts import verify_v132_public_reproduction_package as verifier
except (ImportError, ModuleNotFoundError):  # pragma: no cover - direct script execution
    import verify_v132_public_reproduction_package as verifier  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]

MAIN_ARCHIVE = ROOT / verifier.DEFAULT_ARCHIVE_REL
TRAINING_ARCHIVE = ROOT / verifier.DEFAULT_TRAINING_ARCHIVE_REL

TRAINING_MANIFEST_JSON_PATH = ROOT / verifier.TRAINING_MANIFEST_JSON
TRAINING_MANIFEST_MD_PATH = ROOT / verifier.TRAINING_MANIFEST_MD
SOURCE_BUNDLE_ROOT_REL = "."
SOURCE_BUNDLE_MANIFEST_REL = "release/TRAINING_INPUT_BUNDLE_MANIFEST_v1.3.1-validation-selected-residual-refine.json"
SOURCE_BUNDLE_README_REL = "release/TRAINING_INPUT_BUNDLE_v1.3.1-validation-selected-residual-refine.md"

_BS = "\\"
_NAS_HOST = "nas"
_LOCAL_TRAINING_WORKTREE = "spot_od_v131_public_training_" + "rerun_20260625"
_APP_DATA = "App" + "Data"
_LOCAL_PATH_REDACTION = "<local-build-path-redacted>"

MAIN_EXTRA_MEMBERS: tuple[str, ...] = (
    ".github/workflows/archive-extracted-reproduction.yml",
    verifier.TRAINING_MANIFEST_JSON,
    verifier.TRAINING_MANIFEST_MD,
)


def norm(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_excluded_member(member: str, *, exclude_checkpoints: bool = False) -> bool:
    parts = member.split("/")
    suffix = Path(member).suffix.lower()
    if "__pycache__" in parts or suffix == ".pyc":
        return True
    if exclude_checkpoints and ("checkpoints" in parts or suffix in {".pt", ".pth", ".ckpt"}):
        return True
    return False


def validate_member_name(member: str) -> None:
    normalized, problems = verifier.v131.member_name_problems(member)
    if problems:
        raise ValueError(f"Unsafe archive member {member!r} normalized to {normalized!r}: {problems}")


def public_text_root_prefixes() -> tuple[str, ...]:
    return (
        _BS * 2 + _NAS_HOST + _BS + "Projects" + _BS + "Papers" + _BS + "GNN State Estimation" + _BS,
        "/" + "/" + _NAS_HOST + "/Projects/Papers/GNN State Estimation/",
        "Z:" + _BS + "Papers" + _BS + "GNN State Estimation" + _BS,
        "Z:" + "/" + "Papers/GNN State Estimation/",
    )


def source_bundle_root_prefixes() -> tuple[str, ...]:
    return (
        _BS * 2 + _NAS_HOST + _BS + "Projects" + _BS + "Papers" + _BS + _LOCAL_TRAINING_WORKTREE + _BS + "training_inputs" + _BS,
        "/" + "/" + _NAS_HOST + "/Projects/Papers/" + _LOCAL_TRAINING_WORKTREE + "/training_inputs/",
        "Z:" + _BS + "Papers" + _BS + _LOCAL_TRAINING_WORKTREE + _BS + "training_inputs" + _BS,
        "Z:" + "/" + "Papers/" + _LOCAL_TRAINING_WORKTREE + "/training_inputs/",
    )


def sanitize_public_text(text: str) -> str:
    sanitized = text
    for prefix in source_bundle_root_prefixes():
        sanitized = sanitized.replace(prefix, "")
    for prefix in public_text_root_prefixes():
        sanitized = sanitized.replace(prefix, "")

    nas_path_pattern = re.compile(
        r"(?i)(?:"
        + re.escape(_BS)
        + r"+"
        + _NAS_HOST
        + r"[\\/][^`\"'\r\n,;)]+|"
        + "/"
        + "/"
        + _NAS_HOST
        + r"/[^`\"'\r\n,;)]+)"
    )
    windows_abs_path_pattern = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^`\"'\r\n,;)]+")
    unix_home_path_pattern = re.compile(r"(?i)(?<![A-Za-z0-9_])/(?:Users|home|mnt|Volumes)/[^`\"'\r\n,;)]+")
    app_data_pattern = re.compile(r"(?i)\b" + re.escape(_APP_DATA) + r"\b")

    sanitized = nas_path_pattern.sub(_LOCAL_PATH_REDACTION, sanitized)
    sanitized = windows_abs_path_pattern.sub(_LOCAL_PATH_REDACTION, sanitized)
    sanitized = unix_home_path_pattern.sub(_LOCAL_PATH_REDACTION, sanitized)
    sanitized = app_data_pattern.sub("local-profile-cache", sanitized)
    sanitized = sanitized.replace(_LOCAL_TRAINING_WORKTREE, "source-training-bundle")
    return sanitized


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_public_text(value)
    return value


def sanitize_member_text(text: str, member: str) -> str:
    if Path(member).suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return sanitize_public_text(text)
        return json.dumps(sanitize_json_value(parsed), indent=2, sort_keys=True) + "\n"
    return sanitize_public_text(text)


def public_member_payload(path: Path, member: str) -> bytes | None:
    if not verifier.should_scan_public_text_member(member):
        return None
    text = path.read_text(encoding="utf-8")
    sanitized = sanitize_member_text(text, member)
    hits = verifier.private_path_marker_hits(sanitized)
    if hits:
        raise ValueError(f"Private/local path marker(s) remain in public archive member {member!r}: {hits}")
    return sanitized.encode("utf-8")


def public_member_size(path: Path, member: str) -> int:
    payload = public_member_payload(path, member)
    return path.stat().st_size if payload is None else len(payload)


def iter_files_under(root: Path, rel_dir: str, *, exclude_checkpoints: bool = False) -> Iterable[tuple[Path, str]]:
    base = root / rel_dir
    if not base.is_dir():
        raise FileNotFoundError(f"Required directory not found: {base}")
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        member = norm(path.relative_to(root))
        if is_excluded_member(member, exclude_checkpoints=exclude_checkpoints):
            continue
        validate_member_name(member)
        yield path, member


def write_zip(zip_path: Path, file_members: Iterable[tuple[Path, str]]) -> dict[str, object]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, member in file_members:
            validate_member_name(member)
            if member in seen:
                raise ValueError(f"Duplicate archive member: {member}")
            seen.add(member)
            payload = public_member_payload(path, member)
            if payload is None:
                zf.write(path, member)
            else:
                zf.writestr(member, payload)
    return {
        "path": norm(zip_path.relative_to(ROOT)),
        "bytes": zip_path.stat().st_size,
        "sha256": sha256_file(zip_path),
        "file_count": len(seen),
    }


def collect_training_payload(training_input_root: Path) -> list[tuple[Path, str]]:
    payload: list[tuple[Path, str]] = []
    for rel_dir in verifier.EXPECTED_TRAINING_SOURCE_DIRS:
        payload.extend(iter_files_under(training_input_root, rel_dir, exclude_checkpoints=True))
    payload.sort(key=lambda item: item[1])
    return payload


def write_training_manifests(training_input_root: Path, payload: list[tuple[Path, str]]) -> None:
    payload_bytes = sum(public_member_size(path, member) for path, member in payload)
    source_manifest = training_input_root / SOURCE_BUNDLE_MANIFEST_REL

    manifest = {
        "schema_version": "spot_od_v1_3_2_training_input_bundle.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_tag": verifier.TAG,
        "release_url": verifier.RELEASE_URL,
        "main_package_asset": verifier.MAIN_ASSET,
        "training_input_asset": verifier.TRAINING_ASSET,
        "previous_zenodo_version_doi": verifier.PRIOR_VERSION_DOI,
        "zenodo_concept_doi": verifier.CONCEPT_DOI,
        "source_bundle_root": SOURCE_BUNDLE_ROOT_REL,
        "source_bundle_manifest": SOURCE_BUNDLE_MANIFEST_REL,
        "source_bundle_manifest_sha256": sha256_file(source_manifest) if source_manifest.is_file() else None,
        "source_bundle_readme": SOURCE_BUNDLE_README_REL,
        "build_provenance": (
            "Built from an explicit local training-input root supplied via "
            "--training-input-root; local absolute build paths are intentionally "
            "omitted from this public manifest."
        ),
        "source_directory_count": len(verifier.EXPECTED_TRAINING_SOURCE_DIRS),
        "source_directories": list(verifier.EXPECTED_TRAINING_SOURCE_DIRS),
        "payload_file_count": len(payload),
        "payload_bytes": payload_bytes,
        "total_zip_file_count": len(payload) + 2,
        "checkpoints_omitted": True,
        "excluded_files_note": (
            "*.pt, *.pth, *.ckpt files and checkpoint directories are omitted because "
            "scripts/run_trajectory_candidate_graph_selector_poc.py consumes "
            "adaptive_candidate_fusion_predictions.npz from each source/scenario directory."
        ),
        "scope_boundary": (
            "Checkpoint-free upstream retained-candidate input bundle for rerunning "
            "the validation-selected edge-only attention residual-refinement GNN "
            "training command. It is not raw-data generation, not full "
            "raw/training/all-filter reproduction, not public precise-reference "
            "validation, not independent third-party reproduction, and not "
            "operational validation."
        ),
    }
    TRAINING_MANIFEST_JSON_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_dir_lines = "\n".join(f"- `{source_dir}`" for source_dir in verifier.EXPECTED_TRAINING_SOURCE_DIRS)
    readme = f"""# SPOT-OD v1.3.2 public reproduction training-input bundle

This bundle supplies the upstream retained-candidate input directories needed by
the v1.3.2 public reproduction package provenance command for the v1.3.1
validation-selected edge-only attention residual-refinement evidence.

Extract this ZIP at the repository root so the
`results/adaptive_candidate_fusion_observed_fixed_soft_*` paths exist beside
`scripts/run_trajectory_candidate_graph_selector_poc.py`.

Scope boundary: checkpoint-free retained-candidate input arrays and metadata
only. This is not raw-data generation, not full raw/training/all-filter
reproduction, not public precise-reference validation, not independent
third-party reproduction, and not operational validation.

Checkpoints are omitted. The GNN training loader consumes
`adaptive_candidate_fusion_predictions.npz` under each source/scenario
directory.

Payload file count: `{len(payload)}`
Total ZIP file count including these manifest files: `{len(payload) + 2}`
Source directory count: `{len(verifier.EXPECTED_TRAINING_SOURCE_DIRS)}`
Payload bytes before ZIP compression: `{payload_bytes}`

## Source directories

{source_dir_lines}
"""
    TRAINING_MANIFEST_MD_PATH.write_text(readme, encoding="utf-8")


def collect_main_members() -> list[tuple[Path, str]]:
    members: dict[str, Path] = {}
    for member in (*verifier.REQUIRED_MAIN_MEMBERS, *MAIN_EXTRA_MEMBERS):
        path = ROOT / member
        if not path.is_file():
            raise FileNotFoundError(f"Required main package file not found: {path}")
        if is_excluded_member(member):
            continue
        members[member] = path

    for rel_dir in (
        "src/gnn_state_estimation",
        verifier.v131.GRAPH_DIR,
        verifier.v131.LOCAL_DIR,
        verifier.v131.MEAN_DIR,
        verifier.v131.TAIL_DIR,
    ):
        for path, member in iter_files_under(ROOT, rel_dir):
            members[member] = path

    return [(path, member) for member, path in sorted(members.items())]


def build_archives(training_input_root: Path) -> dict[str, object]:
    training_input_root = training_input_root.resolve()
    payload = collect_training_payload(training_input_root)
    write_training_manifests(training_input_root, payload)

    training_members = payload + [
        (TRAINING_MANIFEST_JSON_PATH, verifier.TRAINING_MANIFEST_JSON),
        (TRAINING_MANIFEST_MD_PATH, verifier.TRAINING_MANIFEST_MD),
    ]
    training_record = write_zip(TRAINING_ARCHIVE, training_members)

    main_record = write_zip(MAIN_ARCHIVE, collect_main_members())
    return {
        "status": "built",
        "main_archive": main_record,
        "training_archive": training_record,
        "training_input_source": "runtime-provided local path omitted from public output",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v1.3.2 public reproduction ZIP archives.")
    parser.add_argument(
        "--training-input-root",
        required=True,
        help="Local path to an extracted v1.3.1 training-input source bundle; the path is not written to public metadata.",
    )
    args = parser.parse_args()

    result = build_archives(Path(args.training_input_root))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
