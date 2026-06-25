from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.build_supplementary_manifest import (
    ACF_AUDIT_SCOPE_BOUNDARY,
    GITHUB_ACTIONS_REPRODUCTION_ROUTE_BOUNDARY,
    GITHUB_ACTIONS_REPRODUCTION_ROUTE_CLAIM,
    GITHUB_ACTIONS_REPRODUCTION_ROUTE_PATHS,
    PACKAGE_VERSION,
    PUBLIC_GITHUB_RELEASE,
    PUBLIC_GITHUB_REPOSITORY,
    PUBLIC_RELEASE_COMMIT,
    PUBLIC_RELEASE_TAG,
    REVIEW_ARCHIVE_REL,
    RELEASE_TITLE,
    ROOT,
    SUPPLEMENTARY_MANIFEST_REL,
    V121_HISTORICAL_DOI,
    V122_HISTORICAL_DOI,
    V123_CONCEPT_DOI,
    V123_CONCEPT_DOI_URL,
    V123_DOI,
    V123_DOI_URL,
    V123_DOI_STATUS,
    V123_GITHUB_RELEASE_ASSET,
    V123_GITHUB_RELEASE_ASSET_KEY,
    V123_ZENODO_RECORD,
)
from scripts.verify_archive_extracted_reproduction import load_manifest_for_archive


WORKFLOW_PATH = ROOT / ".github" / "workflows" / "archive-extracted-reproduction.yml"
REMOVED_CI_PATH = ROOT / (".git" + "lab-ci.yml")
REQUEST_DOC = ROOT / "release" / "INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md"
MANIFEST_PATH = ROOT / "release" / "SUPPLEMENTARY_MANIFEST.json"
ARCHIVE_PATH = ROOT / "release" / "spot_od_v1_2_3_acf_holdout_audit_review_archive.zip"
EXPECTED_PATHS = [
    "release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md",
    ".github/workflows/archive-extracted-reproduction.yml",
]
VERIFIER_COMMAND = (
    "python scripts/verify_archive_extracted_reproduction.py --archive "
    "release/spot_od_v1_2_3_acf_holdout_audit_review_archive.zip --json-out "
    "results/validation/github_actions_archive_extracted_reproduction.json "
    "--md-out results/validation/github_actions_archive_extracted_reproduction.md"
)
V131_VERIFIER_COMMAND = (
    "python scripts/verify_v131_release_package.py --archive "
    "release/spot_od_v1_3_1_validation_selected_residual_refine.zip --json-out "
    "results/validation/v131_release_package_verification.json "
    "--md-out results/validation/v131_release_package_verification.md"
)
FOCUSED_V131_PYTEST_COMMAND = (
    "python -m pytest "
    "tests/test_v131_release_package_verification.py "
    "tests/test_build_trajectory_residual_refine_comparison_intervals.py "
    "tests/test_build_trajectory_residual_refine_tail_diagnostic.py "
    "tests/test_build_trajectory_residual_refine_figure.py "
    "tests/test_trajectory_candidate_graph_architecture_ensemble.py -q"
)
VERIFIER_OUTPUTS = [
    "results/validation/github_actions_archive_extracted_reproduction.json",
    "results/validation/github_actions_archive_extracted_reproduction.md",
]
V131_VERIFIER_OUTPUTS = [
    "results/validation/v131_release_package_verification.json",
    "results/validation/v131_release_package_verification.md",
]
FORBIDDEN_ROUTE_TERMS = (
    "Git" + "Lab",
    "git" + "lab",
    ".git" + "lab-ci",
    "g" + "lab",
    "papers" + "8721323",
)


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def _assert_no_removed_route_terms(text: str) -> None:
    lowered = text.lower()
    for term in FORBIDDEN_ROUTE_TERMS:
        assert term.lower() not in lowered


def test_manifest_uses_v123_acf_holdout_audit_public_release_metadata() -> None:
    manifest = _load_manifest()

    assert PACKAGE_VERSION == "1.2.3-acf-holdout-audit"
    assert manifest["version"] == PACKAGE_VERSION
    assert manifest["title"] == RELEASE_TITLE
    assert "SPOT-OD sparse-visibility orbit-determination self-audit" in (
        manifest["description"]
    )
    assert PUBLIC_GITHUB_RELEASE in manifest["description"]
    assert V123_DOI in manifest["description"]
    assert V123_ZENODO_RECORD in manifest["description"]
    assert "pending/expected" not in manifest["description"]
    assert V122_HISTORICAL_DOI in manifest["description"]
    assert V121_HISTORICAL_DOI in manifest["description"]
    assert ACF_AUDIT_SCOPE_BOUNDARY in manifest["description"]
    assert manifest["public_identifier"] == V123_DOI_URL
    assert "Zenodo DOI URL" in manifest["public_identifier_type"]
    assert "GitHub release URL" in manifest["public_identifier_type"]

    release = manifest["public_archive_release"]
    assert release["title"] == RELEASE_TITLE
    assert release["github_repository"] == PUBLIC_GITHUB_REPOSITORY
    assert release["github_release_target"] == PUBLIC_GITHUB_RELEASE
    assert release["release_tag"] == PUBLIC_RELEASE_TAG
    assert release["github_release_commit"] == PUBLIC_RELEASE_COMMIT
    assert release["doi_status"] == V123_DOI_STATUS
    assert release["zenodo_record"] == V123_ZENODO_RECORD
    assert release["doi"] == V123_DOI
    assert release["doi_url"] == V123_DOI_URL
    assert release["concept_doi"] == V123_CONCEPT_DOI
    assert release["concept_doi_url"] == V123_CONCEPT_DOI_URL
    assert "published; use DOI 10.5281/zenodo.20825138" in (
        release["publication_status_note"]
    )
    assert release["pending_fields"] == []
    assert release["github_release_asset"] == V123_GITHUB_RELEASE_ASSET
    assert release["prior_release"]["doi"] == V122_HISTORICAL_DOI
    assert release["prior_release"]["status"] == (
        "historical_acf_audit_before_public_boundary_repair"
    )
    assert release["prior_release"]["not_current_release_identifier"] is True
    assert release["historical_releases"]["v1.2.1_graph_anchor_gate_poc"]["doi"] == (
        V121_HISTORICAL_DOI
    )
    assert release["scope_boundary"] == ACF_AUDIT_SCOPE_BOUNDARY

    review_archive = manifest["review_archive"]
    archive_bytes = ARCHIVE_PATH.read_bytes()
    assert review_archive["path"].replace("\\", "/") == (
        "release/" + V123_GITHUB_RELEASE_ASSET_KEY
    )
    assert review_archive["bytes"] == len(archive_bytes)
    assert review_archive["sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert review_archive["github_release_asset"] == V123_GITHUB_RELEASE_ASSET_KEY
    assert review_archive["published_github_release_asset"] == V123_GITHUB_RELEASE_ASSET
    assert review_archive["matches_published_github_release_asset"] is False
    assert review_archive["bytes"] != V123_GITHUB_RELEASE_ASSET["size_bytes"]
    assert review_archive["sha256"] != V123_GITHUB_RELEASE_ASSET["sha256"]

    public_metadata = json.dumps(
        {
            key: manifest[key]
            for key in (
                "public_identifier",
                "public_identifier_note",
                "public_archive_commitment",
                "post_acceptance_public_archive_commitment",
                "public_archive_release",
            )
        },
        sort_keys=True,
    )
    assert PUBLIC_GITHUB_REPOSITORY in public_metadata
    assert PUBLIC_GITHUB_RELEASE in public_metadata
    assert PUBLIC_RELEASE_TAG in public_metadata
    assert V123_DOI_STATUS in public_metadata
    assert V123_ZENODO_RECORD in public_metadata
    assert V123_DOI in public_metadata
    assert V123_DOI_URL in public_metadata
    assert V123_CONCEPT_DOI in public_metadata
    assert V123_CONCEPT_DOI_URL in public_metadata
    assert V122_HISTORICAL_DOI in public_metadata
    assert V121_HISTORICAL_DOI in public_metadata
    assert "historical references only" in public_metadata
    assert "pending/expected" not in public_metadata
    assert "No external public " + "repository" not in public_metadata
    assert "deferred until explicit author " + "approval" not in public_metadata
    assert "1.1.0-" + "supplement" not in public_metadata

    boundary = release["scope_boundary"]
    for expected in (
        "Release-support evidence",
        "validation-selected compact-simulator PoC/audit-table evidence",
        "not operational precise-reference validation",
        "not independent-machine reproduction",
        "not third-party validation",
        "not a full scientific rerun",
        "not full raw/training/all-filter reproduction",
        "not universal learned-OD superiority",
    ):
        assert expected in boundary


def test_removed_ci_route_file_is_absent() -> None:
    assert not REMOVED_CI_PATH.exists()


def test_github_actions_archive_extracted_job_uses_minimal_dependency_route() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    compact = " ".join(text.split())

    assert "workflow_dispatch:" in text
    assert "push:" in text
    assert "pull_request:" in text
    assert "runs-on: ubuntu-latest" in text
    assert 'python-version: "3.11"' in text
    assert "MPLBACKEND: Agg" in text
    assert "archive-extracted-reproduction:" in text
    assert VERIFIER_COMMAND in compact
    assert V131_VERIFIER_COMMAND in compact
    assert FOCUSED_V131_PYTEST_COMMAND in compact
    assert "actions/upload-artifact@v4" in text
    assert "if: always()" in text
    assert "if-no-files-found: warn" in text

    for package in (
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "tqdm",
        "sgp4",
        "pytest",
    ):
        assert package in text
    assert "download.pytorch.org/whl/cpu" in text
    assert "cuda" not in text.lower()
    assert "optuna" not in text.lower()
    assert "verify_minimum_tier_reproduction.py" not in text

    for artifact in VERIFIER_OUTPUTS + V131_VERIFIER_OUTPUTS:
        assert artifact in text
    _assert_no_removed_route_terms(text)


def test_request_doc_keeps_github_actions_evidence_with_no_removed_route_terms() -> None:
    text = REQUEST_DOC.read_text(encoding="utf-8")
    compact = " ".join(text.split())

    assert "GitHub Actions verifier passed at" in text
    assert "maintainer-run platform evidence" in text
    assert "not third-party validation" in text
    assert "not a completed independent reproduction" in text
    assert "not be operational POD validation" in compact
    assert "not a fresh full scientific rerun" in compact
    _assert_no_removed_route_terms(text)


def test_github_actions_route_is_indexed_in_manifest_and_archive() -> None:
    assert GITHUB_ACTIONS_REPRODUCTION_ROUTE_PATHS == EXPECTED_PATHS

    manifest = _load_manifest()
    claim = GITHUB_ACTIONS_REPRODUCTION_ROUTE_CLAIM

    assert manifest["claim_to_artifact_map"][claim] == EXPECTED_PATHS
    artifact_paths = [
        _normalize(entry["path"])
        for entry in manifest["artifact_groups"][claim]
    ]
    assert artifact_paths == EXPECTED_PATHS
    assert all(entry["exists"] is True for entry in manifest["artifact_groups"][claim])
    assert all(entry["sha256"] for entry in manifest["artifact_groups"][claim])

    boundary = manifest["claim_boundary_map"][claim]
    assert boundary == GITHUB_ACTIONS_REPRODUCTION_ROUTE_BOUNDARY
    assert "GitHub Actions" in boundary
    assert "route/request only" in boundary
    assert "not a completed independent reproduction" in boundary
    assert "not third-party validation" in boundary
    assert "not operational POD" in boundary
    assert "not a full scientific rerun" in boundary
    _assert_no_removed_route_terms(boundary)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        names = set(archive.namelist())
    assert "release/SUPPLEMENTARY_MANIFEST.json" not in names
    assert ".github/workflows/archive-extracted-reproduction.yml" in names
    assert "release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md" in names
    assert REMOVED_CI_PATH.name not in names


def test_review_archive_omits_paired_manifest_release_record(tmp_path: Path) -> None:
    manifest = _load_manifest()
    manifest_rel = SUPPLEMENTARY_MANIFEST_REL
    archive_rel = REVIEW_ARCHIVE_REL

    indexed_release_record_groups = [
        group
        for group, entries in manifest["artifact_groups"].items()
        for entry in entries
        if _normalize(entry["path"]) == manifest_rel
    ]
    indexed_archive_record_groups = [
        group
        for group, entries in manifest["artifact_groups"].items()
        for entry in entries
        if _normalize(entry["path"]) == archive_rel
    ]
    assert indexed_release_record_groups == []
    assert indexed_archive_record_groups == []

    claim_refs = {
        claim
        for claim, paths in manifest["claim_to_artifact_map"].items()
        if manifest_rel in {_normalize(path) for path in paths}
    }
    assert "review_stage_archive_integrity" in claim_refs
    assert "archive_extracted_reproduction_tier" in claim_refs

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        names = set(archive.namelist())
    assert manifest_rel not in names
    assert archive_rel not in names
    assert ".github/workflows/archive-extracted-reproduction.yml" in names
    assert REMOVED_CI_PATH.name not in names

    paired_manifest, source = load_manifest_for_archive(tmp_path, MANIFEST_PATH)
    assert paired_manifest["version"] == manifest["version"]
    assert source["source"] == "paired_release_manifest"
    assert source["loaded_from_extracted_archive"] is False
    assert "release-level record" in source["note"]


def test_manifest_and_archive_members_have_no_removed_route_terms() -> None:
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    _assert_no_removed_route_terms(manifest_text)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        for name in EXPECTED_PATHS:
            _assert_no_removed_route_terms(archive.read(name).decode("utf-8"))
        assert REMOVED_CI_PATH.name not in archive.namelist()
