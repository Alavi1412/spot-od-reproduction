from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from scripts.build_supplementary_manifest import (
    INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_BOUNDARY,
    INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM,
    INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_PATHS,
    ROOT,
)


CI_PATH = ROOT / ".gitlab-ci.yml"
REQUEST_DOC = ROOT / "release" / "INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md"
MANIFEST_PATH = ROOT / "release" / "SUPPLEMENTARY_MANIFEST.json"
ARCHIVE_PATH = ROOT / "release" / "spot_od_v1_1_0_supplement_review_archive.zip"
EXPECTED_PATHS = [
    "release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md",
    ".gitlab-ci.yml",
]
VERIFIER_COMMAND = (
    "python scripts/verify_archive_extracted_reproduction.py --archive "
    "release/spot_od_v1_1_0_supplement_review_archive.zip --json-out "
    "results/validation/gitlab_ci_archive_extracted_reproduction.json "
    "--md-out results/validation/gitlab_ci_archive_extracted_reproduction.md"
)
CI_ARTIFACTS = [
    "results/validation/gitlab_ci_archive_extracted_reproduction.json",
    "results/validation/gitlab_ci_archive_extracted_reproduction.md",
    "results/validation/gitlab_ci_reproduction_attestation.json",
    "results/validation/gitlab_ci_reproduction_attestation.md",
]


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _normalize(path: str) -> str:
    return path.replace("\\", "/")


def test_gitlab_ci_archive_extracted_job_uses_minimal_dependency_route() -> None:
    text = CI_PATH.read_text(encoding="utf-8")

    assert "image: python:3.11-slim" in text
    assert 'MPLBACKEND: "Agg"' in text
    assert "PYTHONDONTWRITEBYTECODE" in text
    assert "PIP_CACHE_DIR" in text
    assert "archive_extracted_reproduction:" in text
    assert VERIFIER_COMMAND in " ".join(text.split())
    assert "expire_in: 90 days" in text

    for package in ("numpy", "scipy", "pandas", "matplotlib", "seaborn", "pyyaml", "tqdm", "sgp4", "pytest"):
        assert package in text
    assert "torch" not in text.lower()
    assert "optuna" not in text.lower()
    assert "verify_minimum_tier_reproduction.py" not in text

    for artifact in CI_ARTIFACTS:
        assert artifact in text


def test_gitlab_ci_request_doc_describes_route_without_claiming_it_has_run() -> None:
    text = REQUEST_DOC.read_text(encoding="utf-8")

    assert "Optional Private GitLab CI Route" in text
    assert ".gitlab-ci.yml" in text
    assert "route/request" in text
    assert "not completed reproduction until" in text
    assert "passed job URL" in text
    assert "artifacts" in text
    assert VERIFIER_COMMAND in " ".join(text.split())
    assert "It does not run the live-worktree minimum-tier verifier" in text
    assert "not DOI/public archive" in text
    assert "not operational POD" in text
    assert "not a full scientific rerun" in text

    forbidden_patterns = [
        r"GitLab CI job has already passed",
        r"GitLab CI passed",
        r"completed GitLab CI reproduction",
        r"completed independent-machine reproduction evidence",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_gitlab_ci_route_is_indexed_in_manifest_and_archive() -> None:
    assert (
        INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_PATHS == EXPECTED_PATHS
    )

    manifest = _load_manifest()
    claim = INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM

    assert manifest["claim_to_artifact_map"][claim] == EXPECTED_PATHS
    artifact_paths = [
        _normalize(entry["path"])
        for entry in manifest["artifact_groups"][claim]
    ]
    assert artifact_paths == EXPECTED_PATHS
    assert all(entry["exists"] is True for entry in manifest["artifact_groups"][claim])
    assert all(entry["sha256"] for entry in manifest["artifact_groups"][claim])

    boundary = manifest["claim_boundary_map"][claim]
    assert boundary == INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_BOUNDARY
    assert "optional private GitLab CI route" in boundary
    assert "not a passed CI attestation" in boundary
    assert "not DOI/public archive" in boundary
    assert "not operational POD" in boundary
    assert "artifacts must be attached or cited" in boundary

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        names = set(archive.namelist())
    assert ".gitlab-ci.yml" in names
    assert "release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md" in names
