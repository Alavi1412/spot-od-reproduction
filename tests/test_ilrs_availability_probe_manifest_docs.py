from __future__ import annotations

import json
import zipfile

from scripts.build_supplementary_manifest import (
    ARTIFACT_GROUPS,
    ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_CLAIM,
    ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_PATHS,
    INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM,
    INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_PATHS,
    ROOT,
)


MANIFEST_PATH = ROOT / "release" / "SUPPLEMENTARY_MANIFEST.json"
ARCHIVE_PATH = ROOT / "release" / "spot_od_v1_1_0_supplement_review_archive.zip"
ILRS_REPORT_JSON = "results/validation/ilrs_precise_reference_availability_20260617.json"
ILRS_REPORT_MD = "results/validation/ilrs_precise_reference_availability_20260617.md"
REPRO_REQUEST = "release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md"
DOC_PATHS = [
    ROOT / "release" / "README.md",
    ROOT / "release" / "REVIEWER_START_HERE.md",
]


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_ilrs_availability_probe_paths_are_indexed() -> None:
    assert ARTIFACT_GROUPS[ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_CLAIM] == (
        ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_PATHS
    )
    for rel_path in ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_PATHS:
        assert (ROOT / rel_path).is_file(), rel_path


def test_independent_machine_reproduction_request_is_indexed_as_request_only() -> None:
    assert ARTIFACT_GROUPS[INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM] == (
        INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_PATHS
    )
    request = (ROOT / REPRO_REQUEST).read_text(encoding="utf-8")
    assert "not a completed independent reproduction" in request
    assert "release/SUPPLEMENTARY_MANIFEST.json" in request
    assert "release/spot_od_v1_1_0_supplement_review_archive.zip" in request
    assert "scripts/verify_minimum_tier_reproduction.py" in request
    assert "scripts/verify_archive_extracted_reproduction.py" in request


def test_manifest_claims_bound_probe_and_reproduction_request() -> None:
    manifest = _manifest()

    assert manifest["claim_to_artifact_map"][
        ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_CLAIM
    ] == ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_PATHS
    assert manifest["claim_to_artifact_map"][
        INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM
    ] == INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_PATHS

    tier_map = manifest.get(
        "claim_tier_map",
        manifest["claim_to_regeneration_tier_map"],
    )
    assert tier_map[ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_CLAIM] == [
        "minimum_integrity_check"
    ]
    assert tier_map[INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM] == [
        "minimum_integrity_check"
    ]

    ilrs_boundary = manifest["claim_boundary_map"][
        ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_CLAIM
    ]
    assert "not scored validation" in ilrs_boundary
    assert "cached HTML placeholders" in ilrs_boundary
    assert "valid gzip/SP3 products" in ilrs_boundary

    request_boundary = manifest["claim_boundary_map"][
        INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM
    ]
    assert "request/template only" in request_boundary
    assert "not a completed independent reproduction" in request_boundary


def test_ilrs_report_classifies_cached_pending_placeholders_non_usable() -> None:
    report = json.loads((ROOT / ILRS_REPORT_JSON).read_text(encoding="utf-8"))
    assert report["probe_defaults"]["weeks"] == [
        "260606",
        "260613",
        "260620",
        "260627",
    ]
    readiness = report["availability_summary"]["schedule_readiness"]
    assert readiness["prospective_260613"]["required_weeks"] == [
        "260606",
        "260613",
    ]
    assert readiness["prospective_260613"]["status"] == (
        "pending_products_unavailable"
    )
    assert readiness["prospective_260613"]["ready_to_score"] is False
    cached = {
        (row["satellite"], row["week"]): row
        for row in report["local_cached_product_probes"]
        if row["exists"] and row["week"] in {"260613", "260620"}
    }

    assert set(cached) == {
        ("lageos1", "260613"),
        ("lageos2", "260613"),
        ("lageos1", "260620"),
        ("lageos2", "260620"),
    }
    for row in cached.values():
        assert row["byte_length"] == 825
        assert row["gzip_magic"] is False
        assert row["usable_sp3"] is False
        assert row["campaign_runner_alignment"]["campaign_input_kind"] == (
            "sp3_not_valid_gzip"
        )

    assert report["claim_boundary"]["availability_probe_is_scored_validation"] is False
    assert report["claim_boundary"]["independent_reproduction_completed_by_this_probe"] is False
    assert report["availability_summary"]["all_required_products_available"] is False

    report_md = (ROOT / ILRS_REPORT_MD).read_text(encoding="utf-8")
    assert "not scored validation" in report_md
    assert "HTML placeholders" in report_md


def test_docs_and_archive_include_new_request_and_probe_without_claim_upgrade() -> None:
    required = [
        ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_CLAIM,
        ILRS_REPORT_JSON,
        ILRS_REPORT_MD,
        INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_CLAIM,
        REPRO_REQUEST,
        "not scored validation",
        "not a completed independent reproduction",
    ]
    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        for phrase in required:
            assert phrase in text, f"{phrase!r} missing from {path}"

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        names = set(archive.namelist())

    for rel_path in (
        *ILRS_PRECISE_REFERENCE_AVAILABILITY_PROBE_PATHS,
        *INDEPENDENT_MACHINE_REPRODUCTION_REQUEST_PATHS,
    ):
        assert rel_path in names
