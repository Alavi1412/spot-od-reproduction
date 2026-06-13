#!/usr/bin/env python3
"""Archive-extracted reproduction-tier check.

This verifier checks the release archive as an extracted evidence package. It
verifies indexed artifact presence and SHA-256 digests after ZIP extraction,
checks claim-map and regeneration-tier references, and, when the required
script is present in the archive, runs the active main-manuscript table
regeneration check from the extracted tree using the current Python
interpreter in isolated mode.

The tier is intentionally bounded: it is not raw-data generation, model
retraining, recursive filter recomputation, live public-data retrieval, or an
independent end-to-end reproduction outside the supplied archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
import re


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = "release/SUPPLEMENTARY_MANIFEST.json"
DEFAULT_ARCHIVE_REL = "release/spot_od_v1_1_0_supplement_review_archive.zip"
DEFAULT_JSON_OUT = "results/validation/archive_extracted_reproduction.json"
DEFAULT_MD_OUT = "results/validation/archive_extracted_reproduction.md"
ACTIVE_REGEN_SCRIPT_REL = "scripts/regenerate_active_manuscript.py"
OD_VALIDATION_SCRIPT_REL = "scripts/run_real_slr_sp3_od_validation.py"
OD_CANONICAL_DIR_REL = "results/real_slr_sp3_od"
OD_CANONICAL_JSON_REL = "results/real_slr_sp3_od/real_slr_sp3_od_validation.json"
OD_CANONICAL_TABLE_REL = "paper/tables/real_slr_sp3_od.tex"
OD_RERUN_DIR_REL = "results/validation/archive_extracted_real_slr_sp3_od_slice_rerun"
OD_RERUN_JSON_REL = f"{OD_RERUN_DIR_REL}/real_slr_sp3_od_validation.json"
OD_RERUN_TABLE_REL = f"{OD_RERUN_DIR_REL}/real_slr_sp3_od.tex"
OD_RERUN_PUBLIC_SUMMARY_REL = f"{OD_RERUN_DIR_REL}/public_claim_summary.json"
OD_REPORT_JSON_REL = "results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json"
OD_REPORT_MD_REL = "results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.md"
OD_ESTIMATORS = (
    "EKF",
    "UKF (fixed-noise)",
    "AUKF (adaptive)",
    "SP3-IC propagation",
)
LOOP_LABEL_RE = re.compile(r"loop\d+", re.IGNORECASE)
REVIEW_ALIAS_RESTORE_SOURCES = (
    "results/kalmannet_spot_od_loop57/kalmannet_spot_od.json",
    "results/kalmannet_spot_od_budget_adequacy_loop58/kalmannet_spot_od_budget_adequacy.json",
    "release/predeclarations/kalmannet_spot_od_faithful_transposition_loop57.json",
    "release/predeclarations/kalmannet_spot_od_budget_adequacy_loop58.json",
    "tests/test_loop42_hifi_kalmannet_artifacts.py",
    "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch_n64_loop57.json",
    "release/predeclarations/long_arc_hifi_rule_loop47.json",
    "release/predeclarations/long_arc_hifi_n64_extension_loop57.json",
    "release/predeclarations/astrodynamics_floor_loop47.json",
    "results/decision_stability/decision_stability_loop58.json",
    "release/predeclarations/structural_channel_recoverability_loop70.json",
    "release/predeclarations/observed_step_prospective_replication_loop71.json",
    "release/predeclarations/protocol_subset_ablation_loop51.json",
    "release/predeclarations/dmc_ekf_rule_loop44.json",
    "release/predeclarations/pukf_q_adaptive_rule_loop41.json",
    "release/predeclarations/drag_scale_aekf_rule_loop45.json",
    "release/predeclarations/drag_scale_constructive_positive_control_loop54.json",
    "release/predeclarations/drag_scale_ukf_constructive_positive_control_loop55.json",
    "release/predeclarations/drag_scale_ukf_observability_positive_control_loop56.json",
    "results/validation/leakage_scan.json",
)

CLAIM_BOUNDARY = (
    "Archive-extracted integrity, active main-manuscript table-regeneration, "
    "and one public LAGEOS CRD/SP3 precise-reference OD slice recomputation "
    "from archived public inputs only; this does not rerun full raw-data "
    "generation, model retraining, all recursive filters or tables, live "
    "public-data retrieval, operational POD validation, or independent "
    "end-to-end reproduction outside the supplied archive."
)

PUBLIC_OD_SCOPE_BOUNDARY = (
    "Archive-extracted public OD slice recomputation only: one public LAGEOS "
    "CRD/SP3 precise-reference slice is rerun from archived public inputs "
    "contained in the extracted review archive through range-only "
    "EKF/UKF/AUKF/SP3-IC recomputation and table reconstruction. This is not "
    "full scientific reproduction, not full estimator training, not all "
    "filters/tables, not live public-data retrieval, and not operational POD "
    "validation."
)


def norm(path: str | Path) -> str:
    return str(path).replace("\\", "/").strip()


def review_archive_path(rel_path: str) -> str:
    normalized = norm(rel_path)
    if not LOOP_LABEL_RE.search(normalized):
        return normalized
    scrubbed = LOOP_LABEL_RE.sub("", normalized)
    scrubbed = re.sub(r"__+", "_", scrubbed)
    scrubbed = re.sub(r"//+", "/", scrubbed)
    scrubbed = scrubbed.replace("_/", "/").replace("/_", "/")
    scrubbed = scrubbed.replace("_.", ".")
    scrubbed = re.sub(r"_{2,}", "_", scrubbed)
    scrubbed = scrubbed.strip("_/")
    return f"review_artifacts/{scrubbed}"


def rel(path: Path) -> str:
    try:
        return norm(path.relative_to(ROOT))
    except ValueError:
        return norm(path)


def repo_path(path: str | Path) -> Path:
    p = Path(norm(path))
    return p if p.is_absolute() else ROOT / p


def posix_parts(path: str | Path) -> tuple[str, ...]:
    return PurePosixPath(norm(path)).parts


def extracted_path(extracted_root: Path, rel_path: str | Path) -> Path:
    return extracted_root / Path(*posix_parts(rel_path))


def ensure_child_path(parent: Path, child: Path) -> Path:
    parent_resolved = parent.resolve()
    child_resolved = child.resolve()
    child_resolved.relative_to(parent_resolved)
    return child_resolved


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def safe_member_path(member_name: str) -> PurePosixPath:
    normalized = norm(member_name)
    member = PurePosixPath(normalized)
    if member.is_absolute() or any(part in ("", ".", "..") for part in member.parts):
        raise ValueError(f"Unsafe archive member path: {member_name!r}")
    return member


def extract_archive(archive_path: Path, destination: Path) -> dict[str, Any]:
    members: list[str] = []
    failures: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                try:
                    member = safe_member_path(info.filename)
                    member_rel = member.as_posix()
                    target = destination / Path(*member.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(info))
                    members.append(member_rel)
                except Exception as exc:  # pragma: no cover - corrupt archive path
                    failures.append({"member": info.filename, "problem": repr(exc)})
    except zipfile.BadZipFile:
        return {"status": "fail", "members": [], "failures": [{"member": "", "problem": "bad_zip"}]}
    return {
        "status": "pass" if not failures else "fail",
        "members": sorted(members),
        "member_count": len(members),
        "failures": failures,
    }


def restore_review_archive_aliases(extracted_root: Path) -> dict[str, Any]:
    """Restore review archive aliases needed in the temp tree.

    The review ZIP member names omit historical loop labels. Some regeneration
    scripts still use those original filenames. Restoring them only inside
    the disposable extracted workspace preserves runnable checks without
    exposing loop-labeled member names in the manifest or review archive.
    """
    restored: list[dict[str, str]] = []
    missing_aliases: list[dict[str, str]] = []
    for raw_rel in REVIEW_ALIAS_RESTORE_SOURCES:
        alias_rel = review_archive_path(raw_rel)
        if alias_rel == raw_rel:
            continue
        alias_path = extracted_path(extracted_root, alias_rel)
        raw_path = extracted_path(extracted_root, raw_rel)
        if not alias_path.is_file():
            missing_aliases.append(
                {
                    "archive_alias": alias_rel,
                    "alias_digest_id": sha256_bytes(alias_rel.encode("utf-8")),
                }
            )
            continue
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(alias_path, raw_path)
        restored.append(
            {
                "archive_alias": alias_rel,
                "restored_from_archive_alias": alias_rel,
                "source_digest_id": sha256_file(alias_path) or "",
            }
        )
    return {
        "status": "pass" if not missing_aliases else "fail",
        "restored_count": len(restored),
        "missing_alias_count": len(missing_aliases),
        "restored": restored,
        "missing_aliases": missing_aliases,
    }


def load_manifest_for_archive(extracted_root: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    extracted_manifest = extracted_root / Path(*PurePosixPath(MANIFEST_REL).parts)
    if extracted_manifest.is_file():
        return read_json(extracted_manifest), {
            "path": norm(extracted_manifest.relative_to(extracted_root)),
            "source": "extracted_archive_member",
            "loaded_from_extracted_archive": True,
        }
    return read_json(manifest_path), {
        "path": rel(manifest_path),
        "source": "paired_release_manifest",
        "loaded_from_extracted_archive": False,
        "note": (
            "The review ZIP is digest-addressed by the paired release manifest; "
            "the manifest is therefore treated as an allowed release-level "
            "record rather than a self-referential ZIP member."
        ),
    }


def check_extracted_artifacts(extracted_root: Path, members: list[str], entries: list[dict[str, Any]]) -> dict[str, Any]:
    indexed_paths = [entry["norm_path"] for entry in entries]
    duplicate_paths = sorted({path for path in indexed_paths if indexed_paths.count(path) > 1})
    expected_present = sorted(entry["norm_path"] for entry in entries if entry.get("exists"))
    member_set = set(members)
    expected_set = set(expected_present)
    failures: list[dict[str, Any]] = []
    for entry in entries:
        path = entry["norm_path"]
        extracted_path = extracted_root / Path(*PurePosixPath(path).parts)
        exists = extracted_path.is_file()
        actual_bytes = extracted_path.stat().st_size if exists else None
        actual_sha = sha256_file(extracted_path) if exists else None
        problems: list[str] = []
        if not entry.get("exists"):
            problems.append("manifest_entry_not_present")
        if not exists:
            problems.append("extracted_file_missing")
        if exists and entry.get("bytes") != actual_bytes:
            problems.append("bytes_mismatch")
        if exists and entry.get("sha256") != actual_sha:
            problems.append("sha256_mismatch")
        if problems:
            failures.append(
                {
                    "path": path,
                    "group": entry.get("group"),
                    "problems": problems,
                    "expected_bytes": entry.get("bytes"),
                    "actual_bytes": actual_bytes,
                    "expected_sha256": entry.get("sha256"),
                    "actual_sha256": actual_sha,
                }
            )
    missing_members = sorted(expected_set - member_set)
    extra_members = sorted(member_set - expected_set)
    return {
        "status": "pass" if not failures and not duplicate_paths and not missing_members and not extra_members else "fail",
        "indexed_artifact_count": len(entries),
        "expected_present_count": len(expected_present),
        "extracted_member_count": len(members),
        "failure_count": len(failures),
        "duplicate_path_count": len(duplicate_paths),
        "missing_member_count": len(missing_members),
        "extra_member_count": len(extra_members),
        "failures": failures,
        "duplicate_paths": duplicate_paths,
        "missing_members": missing_members,
        "extra_members": extra_members,
    }


def check_claim_map(
    *,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    extracted_root: Path,
    archive_path: Path,
) -> dict[str, Any]:
    claim_map = manifest.get("claim_to_artifact_map", {})
    indexed = {entry["norm_path"] for entry in entries}
    archive_rel = norm(manifest.get("review_archive", {}).get("path", DEFAULT_ARCHIVE_REL))
    allowed_release_level = {MANIFEST_REL, archive_rel}
    failures: list[dict[str, Any]] = []
    allowed_records: list[dict[str, Any]] = []
    if not isinstance(claim_map, dict) or not claim_map:
        failures.append({"claim": None, "problem": "claim_to_artifact_map_missing"})
        claim_map = {}
    for claim, paths in sorted(claim_map.items()):
        if not isinstance(paths, list) or not paths:
            failures.append({"claim": claim, "problem": "claim_has_no_artifacts"})
            continue
        for path in paths:
            npath = norm(path)
            if npath in indexed:
                extracted_path = extracted_root / Path(*PurePosixPath(npath).parts)
                if not extracted_path.is_file():
                    failures.append(
                        {
                            "claim": claim,
                            "path": npath,
                            "problem": "indexed_claim_artifact_missing_from_extracted_archive",
                        }
                    )
            elif npath in allowed_release_level:
                release_path = repo_path(npath)
                if npath == archive_rel:
                    release_path = archive_path
                allowed_records.append({"claim": claim, "path": npath, "exists": release_path.is_file()})
                if not release_path.is_file():
                    failures.append({"claim": claim, "path": npath, "problem": "allowed_release_level_record_missing"})
            else:
                failures.append({"claim": claim, "path": npath, "problem": "claim_artifact_not_indexed_or_allowed"})
    return {
        "status": "pass" if not failures else "fail",
        "claim_count": len(claim_map),
        "failure_count": len(failures),
        "failures": failures,
        "allowed_release_level_records": allowed_records,
    }


def python_script_from_command(command: str) -> str | None:
    tokens = command.replace("\\", "/").split()
    for token in tokens:
        cleaned = token.strip("\"'")
        if cleaned.lower().endswith(".py"):
            return norm(cleaned)
    return None


def script_from_entrypoint(entrypoint: Any) -> str | None:
    if isinstance(entrypoint, dict):
        script = entrypoint.get("script")
        return norm(str(script)) if isinstance(script, str) and script else None
    return python_script_from_command(str(entrypoint))


def check_regeneration_tiers(
    *,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    extracted_root: Path,
) -> dict[str, Any]:
    tiers = manifest.get("regeneration_tiers", {})
    claim_map = manifest.get("claim_to_artifact_map", {})
    claim_tiers = manifest.get("claim_to_regeneration_tier_map", {})
    indexed = {entry["norm_path"] for entry in entries}
    failures: list[dict[str, Any]] = []
    entrypoint_records: list[dict[str, Any]] = []
    if not isinstance(tiers, dict) or not tiers:
        failures.append({"problem": "regeneration_tiers_missing"})
        tiers = {}
    for tier_name, tier in sorted(tiers.items()):
        if not isinstance(tier, dict):
            failures.append({"tier": tier_name, "problem": "tier_record_not_object"})
            continue
        if "requires_retraining" not in tier:
            failures.append({"tier": tier_name, "problem": "requires_retraining_missing"})
        entrypoints = tier.get("entrypoints")
        if not isinstance(entrypoints, list) or not entrypoints:
            entrypoints = tier.get("commands")
        if not isinstance(entrypoints, list) or not entrypoints:
            failures.append({"tier": tier_name, "problem": "entrypoints_missing"})
            continue
        for entrypoint in entrypoints:
            script = script_from_entrypoint(entrypoint)
            if script is None:
                continue
            extracted_exists = (extracted_root / Path(*PurePosixPath(script).parts)).is_file()
            entrypoint_records.append(
                {
                    "tier": tier_name,
                    "script": script,
                    "indexed": script in indexed,
                    "extracted_exists": extracted_exists,
                }
            )
            if tier_name == "archive_extracted_reproduction_check" and not extracted_exists:
                failures.append({"tier": tier_name, "script": script, "problem": "archive_tier_script_missing"})
    if not isinstance(claim_tiers, dict):
        failures.append({"problem": "claim_to_regeneration_tier_map_missing"})
    elif set(claim_tiers) != set(claim_map):
        failures.append({"problem": "claim_to_regeneration_tier_map_key_mismatch"})
    else:
        tier_names = set(tiers)
        for claim, tier_list in sorted(claim_tiers.items()):
            if not isinstance(tier_list, list) or not tier_list:
                failures.append({"claim": claim, "problem": "claim_has_no_regeneration_tiers"})
                continue
            unknown = sorted(str(tier) for tier in tier_list if str(tier) not in tier_names)
            if unknown:
                failures.append({"claim": claim, "problem": "claim_has_unknown_regeneration_tiers", "unknown_tiers": unknown})
    return {
        "status": "pass" if not failures else "fail",
        "tier_count": len(tiers),
        "claim_tier_entry_count": len(claim_tiers) if isinstance(claim_tiers, dict) else 0,
        "failure_count": len(failures),
        "failures": failures,
        "entrypoint_records": entrypoint_records,
    }


def run_active_table_regeneration(extracted_root: Path) -> dict[str, Any]:
    script = extracted_path(extracted_root, ACTIVE_REGEN_SCRIPT_REL)
    if not script.is_file():
        return {
            "status": "blocked",
            "attempted": False,
            "blocker": f"{ACTIVE_REGEN_SCRIPT_REL} is not present in the extracted archive.",
        }
    json_out = "results/validation/archive_extracted_active_manuscript_regeneration.json"
    md_out = "results/validation/archive_extracted_active_manuscript_regeneration.md"
    command = [
        sys.executable,
        "-I",
        ACTIVE_REGEN_SCRIPT_REL,
        "--check-only",
        "--json-out",
        json_out,
        "--md-out",
        md_out,
    ]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(
        command,
        cwd=extracted_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=1200,
        check=False,
    )
    nested_report_path = extracted_root / json_out
    nested: dict[str, Any] = {}
    if nested_report_path.is_file():
        try:
            nested = read_json(nested_report_path)
        except json.JSONDecodeError:
            nested = {}
    validation = nested.get("validation_results", {}) if isinstance(nested, dict) else {}
    status = "pass" if completed.returncode == 0 and nested.get("status") == "pass" else "fail"
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "status": status,
        "attempted": True,
        "step": "active_table_regeneration_from_extracted_tree",
        "execution_details_redacted": True,
        "exit_code": completed.returncode,
        "python_isolated_mode": True,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
        "stdout_line_count": len(stdout.splitlines()),
        "stderr_line_count": len(stderr.splitlines()),
        "nested_report": {
            "path": json_out,
            "exists": nested_report_path.is_file(),
            "status": nested.get("status"),
            "artifact_count": validation.get("artifact_count"),
            "pass_count": validation.get("pass_count"),
            "mismatch_count": validation.get("mismatch_count"),
            "documented_blocker_count": validation.get("documented_blocker_count"),
            "claim_boundary": nested.get("claim_boundary"),
        },
    }


def public_od_archived_input_names(canonical: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for row in canonical.get("input_digests", []):
        name = row.get("archived_input_id")
        if isinstance(name, str) and name:
            names.add(name)
    if not names:
        for arc in canonical.get("arcs", []):
            for kind in ("crd", "sp3"):
                block = arc.get(kind, {})
                name = block.get("archived_input_id") if isinstance(block, dict) else None
                if isinstance(name, str) and name:
                    names.add(name)
    return sorted(names)


def safe_public_input_name(name: str) -> str:
    member = PurePosixPath(norm(name))
    if len(member.parts) != 1 or member.name in ("", ".", ".."):
        raise ValueError(f"Unsafe public input name: {name!r}")
    return member.name


def prepare_archive_extracted_od_rerun_directory(
    *,
    extracted_root: Path,
    canonical: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_dir = extracted_path(extracted_root, OD_CANONICAL_DIR_REL)
    rerun_dir = extracted_path(extracted_root, OD_RERUN_DIR_REL)
    ensure_child_path(extracted_root, rerun_dir)
    if rerun_dir.exists():
        ensure_child_path(extracted_root, rerun_dir)
        shutil.rmtree(rerun_dir)
    rerun_dir.mkdir(parents=True, exist_ok=True)

    names = public_od_archived_input_names(canonical)
    if not names:
        raise ValueError("No archived public CRD/SP3 input names found in the submitted OD record.")

    copied_inputs: list[dict[str, Any]] = []
    for raw_name in names:
        name = safe_public_input_name(raw_name)
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing archived public input in extracted archive: {OD_CANONICAL_DIR_REL}/{name}")
        dest = rerun_dir / name
        shutil.copy2(source, dest)
        copied_inputs.append(
            {
                "kind": "public_crd_or_sp3_input",
                "source": norm(f"{OD_CANONICAL_DIR_REL}/{name}"),
                "destination": norm(f"{OD_RERUN_DIR_REL}/{name}"),
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        )

    table_support: list[dict[str, Any]] = []
    calibrator = source_dir / "sp3_residual_calibrator.json"
    if calibrator.is_file():
        dest = rerun_dir / calibrator.name
        shutil.copy2(calibrator, dest)
        table_support.append(
            {
                "kind": "submitted_table_support_record",
                "role": (
                    "Copied from the extracted archive so the extracted table "
                    "builder reconstructs the submitted real SLR/SP3 OD table "
                    "text; this support record is not rerun by this slice check."
                ),
                "source": norm(f"{OD_CANONICAL_DIR_REL}/{calibrator.name}"),
                "destination": norm(f"{OD_RERUN_DIR_REL}/{calibrator.name}"),
                "bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
            }
        )
    return copied_inputs, table_support


def run_archive_extracted_od_command(extracted_root: Path, timeout_s: int = 1200) -> dict[str, Any]:
    command = [
        sys.executable,
        OD_VALIDATION_SCRIPT_REL,
        "--out-dir",
        OD_RERUN_DIR_REL,
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        command,
        cwd=extracted_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "step": "archive_extracted_public_od_slice_rerun",
        "execution_details_redacted": True,
        "exit_code": completed.returncode,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
        "stdout_line_count": len(stdout.splitlines()),
        "stderr_line_count": len(stderr.splitlines()),
        "working_tree": "extracted_review_archive",
    }


def build_archive_extracted_od_table(extracted_root: Path, timeout_s: int = 300) -> dict[str, Any]:
    code = "\n".join(
        [
            "from pathlib import Path",
            "import sys",
            "root = Path('.').resolve()",
            "scripts_dir = root / 'scripts'",
            "sys.path.insert(0, str(root))",
            "sys.path.insert(0, str(scripts_dir))",
            "from build_paper_assets import build_real_slr_sp3_od_table",
            f"result_path = Path({OD_RERUN_JSON_REL!r})",
            f"out_path = Path({OD_RERUN_TABLE_REL!r})",
            "out_path.parent.mkdir(parents=True, exist_ok=True)",
            "out_path.write_text(build_real_slr_sp3_od_table(result_path=result_path), encoding='utf-8')",
        ]
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=extracted_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "step": "archive_extracted_table_rebuild",
        "execution_details_redacted": True,
        "exit_code": completed.returncode,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
        "stdout_line_count": len(stdout.splitlines()),
        "stderr_line_count": len(stderr.splitlines()),
        "working_tree": "extracted_review_archive",
    }


def normalize_table_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def extract_public_od_claim_summary(payload: dict[str, Any]) -> dict[str, Any]:
    pooled = payload.get("pooled_held_out_position_rmse_m", {})
    pooled_summary: dict[str, Any] = {}
    for name in OD_ESTIMATORS:
        row = pooled.get(name, {})
        pooled_summary[name] = {
            "n_arcs": row.get("n_arcs"),
            "mean_arc_rms_m": row.get("mean_arc_rms_m"),
            "median_arc_rms_m": row.get("median_arc_rms_m"),
            "arcs_best_of": row.get("arcs_best_of"),
        }

    dbar = payload.get("dbar_external_validation", {})
    ni = dbar.get("no_information_baseline", {})
    return {
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "targets": payload.get("targets"),
        "num_arcs": payload.get("num_arcs"),
        "num_arcs_completed": payload.get("num_arcs_completed"),
        "sp3_analysis_center": payload.get("sp3_analysis_center"),
        "sp3_week_product": payload.get("sp3_week_product"),
        "fixed_station_subset": payload.get("fixed_station_subset"),
        "pooled_held_out_position_rmse_m": pooled_summary,
        "dbar_external_validation": {
            "n_arcs_scored": dbar.get("n_arcs_scored"),
            "n_correct": dbar.get("n_correct"),
            "classification_accuracy": dbar.get("classification_accuracy"),
            "confusion": dbar.get("confusion"),
            "sensitivity": dbar.get("sensitivity"),
            "specificity": dbar.get("specificity"),
            "n_counterproductive_arcs": dbar.get("n_counterproductive_arcs"),
            "n_non_counterproductive_arcs": dbar.get("n_non_counterproductive_arcs"),
            "no_information_baseline": {
                "majority_class": ni.get("majority_class"),
                "majority_class_accuracy": ni.get("majority_class_accuracy"),
                "accuracy_minus_majority": ni.get("accuracy_minus_majority"),
                "beats_majority": ni.get("beats_majority"),
            },
        },
    }


def flatten_claim_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_claim_fields(value[key], child))
        return out
    if isinstance(value, list):
        out = {}
        for idx, item in enumerate(value):
            out.update(flatten_claim_fields(item, f"{prefix}[{idx}]"))
        return out
    return {prefix: value}


def compare_public_od_claim_summaries(
    submitted: dict[str, Any],
    rerun: dict[str, Any],
) -> dict[str, Any]:
    expected = extract_public_od_claim_summary(submitted)
    actual = extract_public_od_claim_summary(rerun)
    expected_flat = flatten_claim_fields(expected)
    actual_flat = flatten_claim_fields(actual)
    paths = sorted(set(expected_flat) | set(actual_flat))
    mismatches = [
        {
            "field": path,
            "expected": expected_flat.get(path),
            "actual": actual_flat.get(path),
        }
        for path in paths
        if expected_flat.get(path) != actual_flat.get(path)
    ]
    return {
        "status": "pass" if not mismatches else "fail",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }


def compare_public_od_table_text(generated: str, submitted: str) -> dict[str, Any]:
    gen = normalize_table_text(generated)
    sub = normalize_table_text(submitted)
    if gen == sub:
        digest = sha256_bytes(gen.encode("utf-8"))
        return {
            "status": "pass",
            "matches_submitted_table": True,
            "normalization": "line endings and final newline ignored",
            "generated_sha256": digest,
            "submitted_sha256": digest,
        }
    import difflib

    diff = list(
        difflib.unified_diff(
            sub.splitlines(),
            gen.splitlines(),
            fromfile="extracted_submitted",
            tofile="archive_extracted_rerun_generated",
            lineterm="",
        )
    )
    return {
        "status": "fail",
        "matches_submitted_table": False,
        "normalization": "line endings and final newline ignored",
        "generated_sha256": sha256_bytes(gen.encode("utf-8")),
        "submitted_sha256": sha256_bytes(sub.encode("utf-8")),
        "diff_head": diff[:120],
    }


def run_archive_extracted_public_od_slice_rerun(extracted_root: Path) -> dict[str, Any]:
    required = {
        "od_script": extracted_path(extracted_root, OD_VALIDATION_SCRIPT_REL),
        "submitted_json": extracted_path(extracted_root, OD_CANONICAL_JSON_REL),
        "submitted_table": extracted_path(extracted_root, OD_CANONICAL_TABLE_REL),
        "submitted_input_dir": extracted_path(extracted_root, OD_CANONICAL_DIR_REL),
        "table_builder": extracted_path(extracted_root, "scripts/build_paper_assets.py"),
    }
    missing = sorted(name for name, path in required.items() if not path.exists())
    if missing:
        return {
            "status": "blocked",
            "attempted": False,
            "blocker": "required_extracted_archive_members_missing",
            "missing_required_members": missing,
            "scope_boundary": PUBLIC_OD_SCOPE_BOUNDARY,
        }

    try:
        submitted = read_json(required["submitted_json"])
        copied_inputs, table_support = prepare_archive_extracted_od_rerun_directory(
            extracted_root=extracted_root,
            canonical=submitted,
        )
    except Exception as exc:
        return {
            "status": "fail",
            "attempted": False,
            "blocker": repr(exc),
            "scope_boundary": PUBLIC_OD_SCOPE_BOUNDARY,
        }

    rerun_execution = run_archive_extracted_od_command(extracted_root)
    rerun_json = extracted_path(extracted_root, OD_RERUN_JSON_REL)
    table_execution: dict[str, Any] = {"status": "blocked", "exit_code": None}
    comparisons: dict[str, Any] = {
        "public_claim_summary": {"status": "blocked", "mismatch_count": None, "mismatches": []},
        "table_text": {"status": "blocked", "matches_submitted_table": False},
    }
    generated_table = ""
    if rerun_execution["exit_code"] == 0 and rerun_json.is_file():
        table_execution = build_archive_extracted_od_table(extracted_root)
        rerun = read_json(rerun_json)
        comparisons["public_claim_summary"] = compare_public_od_claim_summaries(
            submitted,
            rerun,
        )
        rerun_table = extracted_path(extracted_root, OD_RERUN_TABLE_REL)
        if table_execution["exit_code"] == 0 and rerun_table.is_file():
            generated_table = rerun_table.read_text(encoding="utf-8")
            submitted_table = required["submitted_table"].read_text(encoding="utf-8")
            comparisons["table_text"] = compare_public_od_table_text(
                generated_table,
                submitted_table,
            )

    status = (
        "pass"
        if rerun_execution["exit_code"] == 0
        and table_execution.get("exit_code") == 0
        and comparisons["public_claim_summary"].get("status") == "pass"
        and comparisons["table_text"].get("status") == "pass"
        else "fail"
    )
    actual = comparisons["public_claim_summary"].get("actual", {})
    dbar = actual.get("dbar_external_validation", {}) if isinstance(actual, dict) else {}
    pooled = actual.get("pooled_held_out_position_rmse_m", {}) if isinstance(actual, dict) else {}
    return {
        "status": status,
        "attempted": True,
        "scope_boundary": PUBLIC_OD_SCOPE_BOUNDARY,
        "extracted_source_artifacts": {
            "submitted_json": {
                "path": OD_CANONICAL_JSON_REL,
                "sha256": sha256_file(required["submitted_json"]),
                "bytes": required["submitted_json"].stat().st_size,
            },
            "submitted_table": {
                "path": OD_CANONICAL_TABLE_REL,
                "sha256": sha256_file(required["submitted_table"]),
                "bytes": required["submitted_table"].stat().st_size,
            },
            "od_script": {
                "path": OD_VALIDATION_SCRIPT_REL,
                "sha256": sha256_file(required["od_script"]),
                "bytes": required["od_script"].stat().st_size,
            },
        },
        "staged_archived_public_inputs": copied_inputs,
        "staged_table_support_records": table_support,
        "rerun_execution": rerun_execution,
        "table_rebuild_execution": table_execution,
        "ephemeral_rerun_artifacts": {
            "result_json": {"path": OD_RERUN_JSON_REL, "exists": rerun_json.is_file()},
            "table_tex": {
                "path": OD_RERUN_TABLE_REL,
                "exists": extracted_path(extracted_root, OD_RERUN_TABLE_REL).is_file(),
            },
        },
        "comparisons": comparisons,
        "summary": {
            "completed_arcs": actual.get("num_arcs_completed") if isinstance(actual, dict) else None,
            "dbar_correct": dbar.get("n_correct"),
            "dbar_scored": dbar.get("n_arcs_scored"),
            "dbar_confusion": dbar.get("confusion"),
            "pooled_held_out_position_rmse_m": pooled,
            "table_text_matched": comparisons["table_text"].get("matches_submitted_table"),
        },
        "_transient_outputs": {
            "generated_table_text": generated_table,
            "public_claim_summary": actual,
        },
    }


def build_result(archive_rel: str = DEFAULT_ARCHIVE_REL) -> dict[str, Any]:
    archive_path = repo_path(archive_rel)
    manifest_path = repo_path(MANIFEST_REL)
    archive_record = {
        "path": rel(archive_path),
        "exists": archive_path.is_file(),
        "digest_note": (
            "Archive byte size and SHA-256 are intentionally not embedded in "
            "this indexed report, because the report is itself archived and "
            "would otherwise create a self-referential digest cycle. The "
            "authoritative archive digest is recorded in the paired release "
            "manifest."
        ),
    }
    if not archive_path.is_file():
        checks = {
            "extraction": {"status": "fail", "failures": [{"problem": "archive_missing"}]},
            "extracted_artifacts": {"status": "fail", "failures": [{"problem": "archive_missing"}]},
            "claim_map": {"status": "fail", "failures": [{"problem": "archive_missing"}]},
            "regeneration_tiers": {"status": "fail", "failures": [{"problem": "archive_missing"}]},
            "active_table_regeneration_from_extracted_tree": {"status": "blocked", "attempted": False, "blocker": "archive_missing"},
            "archive_extracted_public_od_slice_rerun": {"status": "blocked", "attempted": False, "blocker": "archive_missing"},
        }
        return {
            "status": "fail",
            "scope": "archive_extracted_reproduction_check",
            "claim_boundary": CLAIM_BOUNDARY,
            "archive": archive_record,
            "manifest": {"path": rel(manifest_path), "exists": manifest_path.is_file()},
            "checks": checks,
        }

    with tempfile.TemporaryDirectory(prefix="archive_extracted_repro_") as td:
        extracted_root = Path(td)
        extraction_check = extract_archive(archive_path, extracted_root)
        restore_check = restore_review_archive_aliases(extracted_root)
        manifest, manifest_source = load_manifest_for_archive(extracted_root, manifest_path)
        entries = manifest_entries(manifest)
        artifact_check = check_extracted_artifacts(extracted_root, extraction_check.get("members", []), entries)
        claim_check = check_claim_map(
            manifest=manifest,
            entries=entries,
            extracted_root=extracted_root,
            archive_path=archive_path,
        )
        tier_check = check_regeneration_tiers(manifest=manifest, entries=entries, extracted_root=extracted_root)
        table_regen = run_active_table_regeneration(extracted_root)
        public_od_rerun = run_archive_extracted_public_od_slice_rerun(extracted_root)
    checks = {
        "extraction": extraction_check,
        "review_archive_alias_restore": restore_check,
        "extracted_artifacts": artifact_check,
        "claim_map": claim_check,
        "regeneration_tiers": tier_check,
        "active_table_regeneration_from_extracted_tree": table_regen,
        "archive_extracted_public_od_slice_rerun": public_od_rerun,
    }
    status = "pass" if all(check.get("status") == "pass" for check in checks.values()) else "fail"
    return {
        "status": status,
        "scope": "archive_extracted_reproduction_check",
        "requires_retraining": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "archive": archive_record,
        "manifest": {
            "path": rel(manifest_path),
            "exists": manifest_path.is_file(),
            "source": manifest_source,
            "package": manifest.get("package"),
            "version": manifest.get("version"),
            "artifact_count": manifest.get("artifact_count"),
            "artifacts_present": manifest.get("artifacts_present"),
            "claim_count": len(manifest.get("claim_to_artifact_map", {}))
            if isinstance(manifest.get("claim_to_artifact_map", {}), dict)
            else 0,
            "regeneration_tier_count": len(manifest.get("regeneration_tiers", {}))
            if isinstance(manifest.get("regeneration_tiers", {}), dict)
            else 0,
        },
        "checks": checks,
    }


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path),
    }


def write_archive_extracted_od_reports(od_check: dict[str, Any]) -> dict[str, Any]:
    transient = od_check.pop("_transient_outputs", {}) if isinstance(od_check, dict) else {}
    out_dir = repo_path(OD_RERUN_DIR_REL)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = repo_path(OD_RERUN_TABLE_REL)
    summary_path = repo_path(OD_RERUN_PUBLIC_SUMMARY_REL)
    report_json_path = repo_path(OD_REPORT_JSON_REL)
    report_md_path = repo_path(OD_REPORT_MD_REL)

    generated_table = transient.get("generated_table_text", "")
    public_summary = transient.get("public_claim_summary", {})
    if generated_table:
        table_path.write_text(generated_table, encoding="utf-8")
    if public_summary:
        summary_path.write_text(
            json.dumps(public_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    persistent_artifacts = {
        "generated_table": artifact_record(table_path),
        "public_claim_summary": artifact_record(summary_path),
    }
    companion = {
        "schema_version": "archive_extracted_real_slr_sp3_od_slice_rerun_v1",
        "status": od_check.get("status"),
        "scope_boundary": od_check.get("scope_boundary", PUBLIC_OD_SCOPE_BOUNDARY),
        "extracted_source_artifacts": od_check.get("extracted_source_artifacts"),
        "staged_archived_public_inputs": od_check.get("staged_archived_public_inputs", []),
        "staged_table_support_records": od_check.get("staged_table_support_records", []),
        "rerun_execution": od_check.get("rerun_execution"),
        "table_rebuild_execution": od_check.get("table_rebuild_execution"),
        "ephemeral_rerun_artifacts": od_check.get("ephemeral_rerun_artifacts"),
        "comparisons": od_check.get("comparisons"),
        "summary": od_check.get("summary"),
        "persistent_artifacts": persistent_artifacts,
    }
    report_json_path.write_text(
        json.dumps(companion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = od_check.get("summary", {}) if isinstance(od_check, dict) else {}
    pooled = summary.get("pooled_held_out_position_rmse_m", {}) if isinstance(summary, dict) else {}
    dbar_conf = summary.get("dbar_confusion", {}) if isinstance(summary, dict) else {}
    claim = od_check.get("comparisons", {}).get("public_claim_summary", {})
    table = od_check.get("comparisons", {}).get("table_text", {})
    lines = [
        "# Archive-Extracted Real SLR/SP3 OD Slice Rerun",
        "",
        f"Status: **{str(od_check.get('status')).upper()}**",
        "",
        "## Scope Boundary",
        od_check.get("scope_boundary", PUBLIC_OD_SCOPE_BOUNDARY),
        "",
        "## Rerun",
        f"- Step: `{od_check.get('rerun_execution', {}).get('step')}`",
        f"- Exit code: `{od_check.get('rerun_execution', {}).get('exit_code')}`",
        f"- Table rebuild exit code: `{od_check.get('table_rebuild_execution', {}).get('exit_code')}`",
        "- Execution details: redacted from this reviewer-facing summary.",
        "",
        "## Comparisons",
        f"- Public-claim summary fields: **{str(claim.get('status')).upper()}** ({claim.get('mismatch_count')} mismatches).",
        f"- Generated table text matches extracted submitted table: **{str(table.get('status')).upper()}**.",
        "",
        "## Summary",
        f"- Completed arcs: `{summary.get('completed_arcs')}`.",
        f"- DBAR correct/scored: `{summary.get('dbar_correct')}/{summary.get('dbar_scored')}`.",
        f"- DBAR confusion: `{dbar_conf}`.",
        f"- Table text matched: `{summary.get('table_text_matched')}`.",
        "",
        "## Pooled Held-Out Position RMSE",
    ]
    for name in OD_ESTIMATORS:
        row = pooled.get(name, {}) if isinstance(pooled, dict) else {}
        lines.append(
            "- "
            f"`{name}` mean `{row.get('mean_arc_rms_m')}` m, "
            f"median `{row.get('median_arc_rms_m')}` m, "
            f"best `{row.get('arcs_best_of')}/{row.get('n_arcs')}`."
        )
    lines += [
        "",
        "## Outputs",
        f"- JSON: `{rel(report_json_path)}`",
        f"- Markdown: `{rel(report_md_path)}`",
        f"- Generated table: `{rel(table_path)}`",
        f"- Public-claim summary: `{rel(summary_path)}`",
    ]
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "companion_json": rel(report_json_path),
        "companion_markdown": rel(report_md_path),
        "generated_table": persistent_artifacts["generated_table"],
        "public_claim_summary": persistent_artifacts["public_claim_summary"],
    }


def write_reports(result: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    od_outputs = write_archive_extracted_od_reports(
        result["checks"].get("archive_extracted_public_od_slice_rerun", {})
    )
    result["checks"]["archive_extracted_public_od_slice_rerun"]["persistent_report_outputs"] = od_outputs
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checks = result["checks"]
    table = checks["active_table_regeneration_from_extracted_tree"]
    public_od = checks["archive_extracted_public_od_slice_rerun"]
    nested = table.get("nested_report", {}) if isinstance(table, dict) else {}
    manifest = result.get("manifest", {})
    source = manifest.get("source", {})
    public_od_summary = public_od.get("summary", {}) if isinstance(public_od, dict) else {}
    public_od_claim = public_od.get("comparisons", {}).get("public_claim_summary", {})
    public_od_table = public_od.get("comparisons", {}).get("table_text", {})
    md_lines = [
        "# Archive-Extracted Reproduction Check",
        "",
        f"Status: **{str(result['status']).upper()}**",
        "",
        "## Scope Boundary",
        result["claim_boundary"],
        "",
        "## Checks",
        f"- ZIP extraction: **{checks['extraction']['status'].upper()}**.",
        f"- Review archive alias restore for extracted rerun dependencies: **{checks['review_archive_alias_restore']['status'].upper()}**.",
        f"- Extracted manifest-indexed artifact presence and SHA-256 checks: **{checks['extracted_artifacts']['status'].upper()}**.",
        f"- Claim-to-artifact map resolution: **{checks['claim_map']['status'].upper()}**.",
        f"- Regeneration-tier key resolution: **{checks['regeneration_tiers']['status'].upper()}**.",
        f"- Active table regeneration from extracted tree: **{table['status'].upper()}**.",
        f"- Archive-extracted public OD slice rerun: **{public_od['status'].upper()}**.",
        "",
        "## Manifest Source",
        f"- Manifest source: `{source.get('source')}`.",
        f"- Loaded from extracted archive: `{source.get('loaded_from_extracted_archive')}`.",
    ]
    if source.get("note"):
        md_lines.append(f"- Note: {source['note']}")
    md_lines += [
        "",
        "## Counts",
        f"- Manifest-indexed artifacts checked after extraction: `{checks['extracted_artifacts'].get('indexed_artifact_count')}`.",
        f"- Extracted ZIP members: `{checks['extracted_artifacts'].get('extracted_member_count')}`.",
        f"- Claim-map entries: `{manifest.get('claim_count')}`.",
        f"- Regeneration tiers: `{manifest.get('regeneration_tier_count')}`.",
        "",
        "## Extracted Active Table Regeneration",
        f"- Attempted: `{table.get('attempted')}`.",
        f"- Exit code: `{table.get('exit_code')}`.",
        f"- Nested status: `{nested.get('status')}`.",
        f"- Active artifacts: `{nested.get('artifact_count')}`.",
        f"- Pass count: `{nested.get('pass_count')}`.",
        f"- Mismatch count: `{nested.get('mismatch_count')}`.",
        f"- Blocker count: `{nested.get('documented_blocker_count')}`.",
        "",
        "## Archive-Extracted Public OD Slice Rerun",
        f"- Attempted: `{public_od.get('attempted')}`.",
        f"- Step: `{public_od.get('rerun_execution', {}).get('step')}`.",
        f"- Exit code: `{public_od.get('rerun_execution', {}).get('exit_code')}`.",
        "- Execution details: redacted from this reviewer-facing summary.",
        f"- Completed arcs: `{public_od_summary.get('completed_arcs')}`.",
        f"- Public-claim summary fields: **{str(public_od_claim.get('status')).upper()}** ({public_od_claim.get('mismatch_count')} mismatches).",
        f"- DBAR correct/scored: `{public_od_summary.get('dbar_correct')}/{public_od_summary.get('dbar_scored')}`.",
        f"- Generated table text matches extracted submitted table: **{str(public_od_table.get('status')).upper()}**.",
        f"- Companion report: `{od_outputs.get('companion_json')}` and `{od_outputs.get('companion_markdown')}`.",
        "",
        "## Outputs",
        f"- JSON: `{rel(json_out)}`",
        f"- Markdown: `{rel(md_out)}`",
    ]
    md_out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify archive-extracted reproduction tier.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE_REL)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", default=DEFAULT_MD_OUT)
    parser.add_argument("--check-only", action="store_true", help="Run checks without writing reports.")
    args = parser.parse_args()

    result = build_result(args.archive)
    json_out = repo_path(args.json_out)
    md_out = repo_path(args.md_out)
    if not args.check_only:
        write_reports(result, json_out, md_out)
    summary = {
        "status": result["status"],
        "json": rel(json_out),
        "markdown": rel(md_out),
        "archive": result["archive"],
        "artifact_check": result["checks"]["extracted_artifacts"]["status"],
        "claim_map_check": result["checks"]["claim_map"]["status"],
        "regeneration_tier_check": result["checks"]["regeneration_tiers"]["status"],
        "active_table_regeneration_from_extracted_tree": result["checks"]["active_table_regeneration_from_extracted_tree"]["status"],
        "archive_extracted_public_od_slice_rerun": result["checks"]["archive_extracted_public_od_slice_rerun"]["status"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
