#!/usr/bin/env python
"""Probe official ILRS precise-reference SP3 product availability.

This script is intentionally conservative. A URL is marked as a usable SP3
product only when the fetched bytes are gzip data and the decompressed content
looks like an SP3 file. HTML placeholders, Earthdata Login pages, redirects,
directory shells, and non-SP3 gzip files are not counted as validation inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEEKS = ("260606", "260613", "260620", "260627")
DEFAULT_SATELLITES = ("lageos1", "lageos2")
DEFAULT_CENTERS = ("nsgf",)
DEFAULT_VERSIONS = ("v80",)
DEFAULT_POSITIVE_CONTROL_WEEKS = ("260509",)
DEFAULT_TIMEOUT_S = 45.0
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_LOCAL_CACHE_DIRS = ("results/real_slr_sp3_od_formal210_inputs",)
USER_AGENT = "SPOT-OD-ILRS-availability-probe/1.0"
SCHEMA_VERSION = "ilrs_precise_reference_availability_probe_v1"

EDC_ROOT = "https://edc.dgfi.tum.de/pub/slr/products/orbits"
CDDIS_ROOT = "https://cddis.nasa.gov/archive/slr/products/orbits"
OFFICIAL_SOURCE_URLS = {
    "edc_precise_orbits_description": "https://edc.dgfi.tum.de/en/products/precise-orbits/",
    "nasa_earthdata_slr_ac_orbit_product": "https://www.earthdata.nasa.gov/data/space-geodesy-techniques/slr/analysis-center-orbit-product",
}

SCHEDULE_REQUIREMENTS = {
    "prospective_260613": {
        "validation_week": "260606",
        "test_week": "260613",
        "required_weeks": ("260606", "260613"),
        "predeclaration_path": (
            "release/predeclarations/"
            "real_slr_sp3_temporal_corrected_od_prospective_260613.json"
        ),
        "output_json": (
            "results/real_slr_sp3_temporal_corrected_od_prospective_260613/"
            "real_slr_sp3_temporal_corrected_od_prospective_260613.json"
        ),
        "claim_note": (
            "No repository predeclaration was found for 260613 in this batch; "
            "do not treat any later-created rule as prospective unless an "
            "independent timestamped pre-scoring rule exists."
        ),
    },
    "prospective_260620": {
        "validation_week": "260613",
        "test_week": "260620",
        "required_weeks": ("260613", "260620"),
        "predeclaration_path": (
            "release/predeclarations/"
            "real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json"
        ),
        "output_json": (
            "results/real_slr_sp3_temporal_corrected_od_prospective_260620/"
            "real_slr_sp3_temporal_corrected_od_prospective_260620.json"
        ),
        "claim_note": (
            "Existing timestamped predeclaration covers the 2026-06-15.."
            "2026-06-19 test week; run scoring only after all validation/test "
            "SP3 products are valid gzip/SP3 files."
        ),
    },
    "prospective_260627": {
        "validation_week": "260620",
        "test_week": "260627",
        "required_weeks": ("260620", "260627"),
        "predeclaration_path": (
            "release/predeclarations/"
            "real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json"
        ),
        "output_json": (
            "results/real_slr_sp3_temporal_corrected_od_prospective_260627/"
            "real_slr_sp3_temporal_corrected_od_prospective_260627.json"
        ),
        "claim_note": (
            "Existing timestamped predeclaration covers the 2026-06-22.."
            "2026-06-26 test week; run scoring only after all validation/test "
            "SP3 products are valid gzip/SP3 files."
        ),
    },
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root_url: str
    requires_authentication_for_public_probe: bool = False

    def parent_listing_url(self, satellite: str) -> str:
        return f"{self.root_url}/{satellite}/"

    def week_listing_url(self, satellite: str, week: str) -> str:
        return f"{self.root_url}/{satellite}/{week}/"

    def product_url(
        self, satellite: str, week: str, center: str, version: str
    ) -> str:
        return (
            f"{self.root_url}/{satellite}/{week}/"
            f"{center}.orb.{satellite}.{week}.{version}.sp3.gz"
        )


SOURCES = (
    SourceSpec("EDC", EDC_ROOT),
    SourceSpec("CDDIS", CDDIS_ROOT, requires_authentication_for_public_probe=True),
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_html_like(raw: bytes, content_type: str | None = None) -> bool:
    head = raw[:1024].lstrip().lower()
    ctype = (content_type or "").lower()
    return (
        "text/html" in ctype
        or head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or b"<body" in head
        or b"</html" in head
        or b"<title" in head
    )


def is_login_like(raw: bytes, final_url: str = "") -> bool:
    text = raw[:8192].decode("utf-8", "replace").lower()
    url = final_url.lower()
    return (
        "earthdata login" in text
        or "urs.earthdata.nasa.gov" in text
        or "urs.earthdata.nasa.gov" in url
        or "oauth/authorize" in text
        or "login" in urllib.parse.urlparse(url).path.lower()
    )


def sp3_header_valid(text: str) -> bool:
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("#"):
        return False
    return any(
        line.startswith("##")
        or line.startswith("+")
        or line.startswith("*")
        or line == "EOF"
        for line in lines[1:]
    )


def classify_candidate_bytes(
    raw: bytes,
    *,
    content_type: str | None = None,
    final_url: str = "",
    truncated: bool = False,
) -> dict[str, Any]:
    """Classify fetched direct-product bytes without using network state."""
    gzip_magic = raw.startswith(b"\x1f\x8b")
    html_like = is_html_like(raw, content_type)
    login_like = is_login_like(raw, final_url)
    sha = _sha256(raw) if raw else None
    out: dict[str, Any] = {
        "byte_length": len(raw),
        "sha256": sha,
        "truncated": truncated,
        "gzip_magic": gzip_magic,
        "gzip_valid": False,
        "sp3_header_valid": False,
        "sp3_first_line": None,
        "html_like": html_like,
        "login_like": login_like,
        "usable_sp3": False,
    }
    if truncated:
        out["classification"] = "truncated_response_not_usable"
        out["interpretation"] = "Response exceeded max probe bytes; not counted as usable SP3."
        return out
    if html_like or login_like:
        out["classification"] = "html_or_login_not_sp3"
        out["interpretation"] = (
            "HTML/login/directory content is not a gzip/SP3 precise-orbit product."
        )
        return out
    if not gzip_magic:
        out["classification"] = "not_gzip_not_sp3"
        out["interpretation"] = "Response lacks gzip magic bytes and is not counted as SP3."
        return out
    try:
        decoded = gzip.decompress(raw)
        out["gzip_valid"] = True
    except OSError as exc:
        out["classification"] = "gzip_decompress_failed"
        out["interpretation"] = f"Gzip magic present but decompression failed: {exc!r}."
        return out
    text = decoded.decode("utf-8", "replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    out["sp3_first_line"] = first_line[:160]
    out["sp3_header_valid"] = sp3_header_valid(text)
    if not out["sp3_header_valid"]:
        out["classification"] = "gzip_not_sp3"
        out["interpretation"] = "Response is gzip data but decompressed content is not SP3."
        return out
    out["usable_sp3"] = True
    out["classification"] = "usable_gzip_sp3"
    out["interpretation"] = "Usable gzip-compressed SP3 product."
    return out


def campaign_input_kind(classification: dict[str, Any], *, file_name: str = "") -> str | None:
    """Map the probe classification to the campaign runner's input gate names."""
    if classification.get("usable_sp3"):
        return None
    if classification.get("classification") == "local_cache_missing":
        return None
    suffix_is_gz = file_name.endswith(".gz")
    if suffix_is_gz and not classification.get("gzip_magic"):
        return "sp3_not_valid_gzip"
    if classification.get("classification") == "gzip_decompress_failed":
        return "sp3_decompress_failed"
    if classification.get("gzip_valid") and not classification.get("sp3_header_valid"):
        return "sp3_not_sp3_format"
    if classification.get("html_like") or classification.get("login_like"):
        return "sp3_not_valid_gzip" if suffix_is_gz else "sp3_not_sp3_format"
    return "sp3_not_sp3_format"


def classify_local_cached_product(
    path: Path,
    *,
    satellite: str,
    week: str,
    center: str,
    version: str,
) -> dict[str, Any]:
    exists = path.is_file()
    raw = path.read_bytes() if exists else b""
    classification = (
        classify_candidate_bytes(raw)
        if exists
        else {
            "byte_length": 0,
            "sha256": None,
            "truncated": False,
            "gzip_magic": False,
            "gzip_valid": False,
            "sp3_header_valid": False,
            "sp3_first_line": None,
            "html_like": False,
            "login_like": False,
            "usable_sp3": False,
            "classification": "local_cache_missing",
            "interpretation": "No local cached product file was present.",
        }
    )
    kind = campaign_input_kind(classification, file_name=path.name)
    return {
        "probe_kind": "local_cached_product",
        "path": repo_rel(path),
        "exists": exists,
        "satellite": satellite,
        "week": week,
        "center": center,
        "version": version,
        **classification,
        "campaign_runner_alignment": {
            "matches_existing_sp3_input_gate": True,
            "campaign_input_kind": kind,
            "campaign_counts_as_available_input": classification.get("usable_sp3") is True,
            "note": (
                "The temporal corrected OD campaign treats a .sp3.gz path "
                "without gzip magic as input_unavailable/sp3_not_valid_gzip; "
                "this probe uses the same conservative availability boundary."
            ),
        },
    }


def probe_local_cached_products(
    cache_dirs: tuple[Path, ...],
    *,
    satellites: tuple[str, ...],
    weeks: tuple[str, ...],
    centers: tuple[str, ...],
    versions: tuple[str, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cache_dir in cache_dirs:
        base = cache_dir if cache_dir.is_absolute() else ROOT / cache_dir
        for satellite in satellites:
            for week in weeks:
                for center in centers:
                    for version in versions:
                        path = base / f"{center}.orb.{satellite}.{week}.{version}.sp3.gz"
                        records.append(
                            classify_local_cached_product(
                                path,
                                satellite=satellite,
                                week=week,
                                center=center,
                                version=version,
                            )
                        )
    return records


def extract_listing_names(raw: bytes) -> list[str]:
    text = raw.decode("utf-8", "replace")
    hrefs = re.findall(r"href=[\"']([^\"']+)[\"']", text, flags=re.IGNORECASE)
    names: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        path = urllib.parse.urlparse(href).path
        name = Path(path.rstrip("/")).name
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def classify_listing_bytes(
    raw: bytes,
    *,
    content_type: str | None = None,
    final_url: str = "",
    expected_week: str | None = None,
    expected_satellite: str | None = None,
) -> dict[str, Any]:
    names = extract_listing_names(raw)
    sp3_names = sorted(name for name in names if name.endswith(".sp3.gz"))
    week_names = sorted(name for name in names if re.fullmatch(r"\d{6}", name))
    login_like = is_login_like(raw, final_url)
    html_like = is_html_like(raw, content_type)
    expected_sp3_pattern = None
    matching_sp3_names: list[str] = []
    if expected_week and expected_satellite:
        expected_sp3_pattern = (
            rf"^[a-z0-9]+\.orb\.{re.escape(expected_satellite)}\."
            rf"{re.escape(expected_week)}\.v[0-9]+\.sp3\.gz$"
        )
        pattern = re.compile(expected_sp3_pattern)
        matching_sp3_names = [name for name in sp3_names if pattern.match(name)]
    if login_like:
        classification = "login_html_listing_not_public"
    elif html_like and (names or week_names or sp3_names):
        classification = "directory_listing"
    elif html_like:
        classification = "html_not_listing"
    else:
        classification = "non_html_listing_response"
    return {
        "byte_length": len(raw),
        "sha256": _sha256(raw) if raw else None,
        "html_like": html_like,
        "login_like": login_like,
        "classification": classification,
        "listing_name_count": len(names),
        "listed_names_sample": names[:30],
        "listed_weeks": week_names,
        "expected_week_listed": expected_week in week_names if expected_week else None,
        "listed_sp3_candidates": sp3_names,
        "expected_sp3_pattern": expected_sp3_pattern,
        "matching_sp3_candidates": matching_sp3_names,
        "matching_sp3_candidate_count": len(matching_sp3_names),
    }


def fetch_url(url: str, timeout_s: float, max_bytes: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read(max_bytes + 1)
            truncated = len(raw) > max_bytes
            if truncated:
                raw = raw[:max_bytes]
            return {
                "url": url,
                "final_url": response.geturl(),
                "http_status": getattr(response, "status", None),
                "content_type": response.headers.get("Content-Type"),
                "content_length_header": response.headers.get("Content-Length"),
                "bytes": raw,
                "truncated": truncated,
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]
        return {
            "url": url,
            "final_url": exc.geturl(),
            "http_status": exc.code,
            "content_type": exc.headers.get("Content-Type"),
            "content_length_header": exc.headers.get("Content-Length"),
            "bytes": raw,
            "truncated": truncated,
            "error": f"HTTPError: {exc.reason}",
        }
    except Exception as exc:
        return {
            "url": url,
            "final_url": url,
            "http_status": None,
            "content_type": None,
            "content_length_header": None,
            "bytes": b"",
            "truncated": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def public_fetch_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in record.items() if key != "bytes"}
    out["byte_length"] = len(record.get("bytes", b""))
    if record.get("bytes"):
        out["sha256"] = _sha256(record["bytes"])
    else:
        out["sha256"] = None
    return out


def probe_parent_listing(
    source: SourceSpec,
    satellite: str,
    *,
    weeks: tuple[str, ...],
    timeout_s: float,
    max_bytes: int,
) -> dict[str, Any]:
    fetched = fetch_url(source.parent_listing_url(satellite), timeout_s, max_bytes)
    listing = classify_listing_bytes(
        fetched["bytes"],
        content_type=fetched.get("content_type"),
        final_url=fetched.get("final_url") or "",
    )
    return {
        "source": source.name,
        "probe_kind": "parent_listing",
        "satellite": satellite,
        **public_fetch_record(fetched),
        **listing,
        "required_weeks_listed": {week: week in listing["listed_weeks"] for week in weeks},
    }


def probe_week_listing(
    source: SourceSpec,
    satellite: str,
    week: str,
    *,
    timeout_s: float,
    max_bytes: int,
) -> dict[str, Any]:
    fetched = fetch_url(source.week_listing_url(satellite, week), timeout_s, max_bytes)
    listing = classify_listing_bytes(
        fetched["bytes"],
        content_type=fetched.get("content_type"),
        final_url=fetched.get("final_url") or "",
        expected_week=week,
        expected_satellite=satellite,
    )
    return {
        "source": source.name,
        "probe_kind": "week_listing",
        "satellite": satellite,
        "week": week,
        **public_fetch_record(fetched),
        **listing,
    }


def probe_direct_product(
    source: SourceSpec,
    satellite: str,
    week: str,
    center: str,
    version: str,
    *,
    timeout_s: float,
    max_bytes: int,
) -> dict[str, Any]:
    url = source.product_url(satellite, week, center, version)
    fetched = fetch_url(url, timeout_s, max_bytes)
    classification = classify_candidate_bytes(
        fetched["bytes"],
        content_type=fetched.get("content_type"),
        final_url=fetched.get("final_url") or "",
        truncated=bool(fetched.get("truncated")),
    )
    return {
        "source": source.name,
        "probe_kind": "direct_product",
        "satellite": satellite,
        "week": week,
        "center": center,
        "version": version,
        **public_fetch_record(fetched),
        **classification,
    }


def build_campaign_commands() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for schedule, spec in SCHEDULE_REQUIREMENTS.items():
        predecl = spec["predeclaration_path"]
        output = spec["output_json"]
        predecl_exists = (ROOT / predecl).is_file()
        create_predecl = (
            "python scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py "
            f"--schedule {schedule} --predeclaration {predecl} "
            "--write-predeclaration-only"
        )
        score_cmd = (
            "python scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py "
            f"--schedule {schedule} --predeclaration {predecl} "
            f"--output-json {output} --no-table --refresh --resume"
        )
        rows.append(
            {
                "schedule": schedule,
                "validation_week": spec["validation_week"],
                "test_week": spec["test_week"],
                "required_weeks": list(spec["required_weeks"]),
                "predeclaration_path": predecl,
                "predeclaration_exists": predecl_exists,
                "predeclaration_creation_command_line": (
                    None if predecl_exists else create_predecl
                ),
                "scoring_command_line": score_cmd,
                "run_condition": (
                    "Run only after all required LAGEOS SP3 products are "
                    "classified usable_sp3=true in this probe or an equivalent "
                    "official-source probe."
                ),
                "claim_note": spec["claim_note"],
            }
        )
    return rows


def summarize_availability(
    direct_records: list[dict[str, Any]],
    *,
    satellites: tuple[str, ...],
    weeks: tuple[str, ...],
) -> dict[str, Any]:
    by_sat_week: dict[str, dict[str, Any]] = {}
    for satellite in satellites:
        for week in weeks:
            candidates = [
                row
                for row in direct_records
                if row["satellite"] == satellite and row["week"] == week
            ]
            usable = [row for row in candidates if row.get("usable_sp3")]
            key = f"{satellite}:{week}"
            by_sat_week[key] = {
                "satellite": satellite,
                "week": week,
                "candidate_count": len(candidates),
                "usable_candidate_count": len(usable),
                "usable_sp3": bool(usable),
                "usable_urls": [row["url"] for row in usable],
                "classifications": sorted(
                    {
                        str(row.get("classification"))
                        for row in candidates
                        if row.get("classification")
                    }
                ),
            }
    all_available = all(row["usable_sp3"] for row in by_sat_week.values())
    schedule_ready: dict[str, Any] = {}
    for schedule, spec in SCHEDULE_REQUIREMENTS.items():
        required_weeks = tuple(spec["required_weeks"])
        required_keys = [
            f"{satellite}:{week}"
            for satellite in satellites
            for week in required_weeks
        ]
        if not set(required_weeks).issubset(set(weeks)):
            status = "not_evaluated_required_week_not_in_probe_defaults"
            ready = False
        else:
            ready = all(by_sat_week[key]["usable_sp3"] for key in required_keys)
            status = "available_to_score" if ready else "pending_products_unavailable"
        schedule_ready[schedule] = {
            "required_weeks": list(required_weeks),
            "required_satellite_week_keys": required_keys,
            "ready_to_score": ready,
            "status": status,
        }
    return {
        "all_required_products_available": all_available,
        "by_satellite_week": by_sat_week,
        "schedule_readiness": schedule_ready,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    weeks = tuple(args.weeks)
    satellites = tuple(args.satellites)
    centers = tuple(args.centers)
    versions = tuple(args.versions)
    parent_listings: list[dict[str, Any]] = []
    week_listings: list[dict[str, Any]] = []
    direct_products: list[dict[str, Any]] = []
    positive_control_products: list[dict[str, Any]] = []
    local_cached_products = probe_local_cached_products(
        tuple(Path(item) for item in args.local_cache_dirs),
        satellites=satellites,
        weeks=weeks,
        centers=centers,
        versions=versions,
    )
    for source in SOURCES:
        if source.name.lower() not in args.sources:
            continue
        for satellite in satellites:
            parent_listings.append(
                probe_parent_listing(
                    source,
                    satellite,
                    weeks=weeks,
                    timeout_s=args.timeout_s,
                    max_bytes=args.max_bytes,
                )
            )
            for week in weeks:
                week_listings.append(
                    probe_week_listing(
                        source,
                        satellite,
                        week,
                        timeout_s=args.timeout_s,
                        max_bytes=args.max_bytes,
                    )
                )
                for center in centers:
                    for version in versions:
                        direct_products.append(
                            probe_direct_product(
                                source,
                                satellite,
                                week,
                                center,
                                version,
                                timeout_s=args.timeout_s,
                                max_bytes=args.max_bytes,
                            )
                        )
            if source.name == "EDC":
                for control_week in args.positive_control_weeks:
                    for center in centers:
                        for version in versions:
                            positive_control_products.append(
                                probe_direct_product(
                                    source,
                                    satellite,
                                    control_week,
                                    center,
                                    version,
                                    timeout_s=args.timeout_s,
                                    max_bytes=args.max_bytes,
                                )
                            )
    summary = summarize_availability(
        direct_products,
        satellites=satellites,
        weeks=weeks,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now().isoformat().replace("+00:00", "Z"),
        "purpose": (
            "Official-source ILRS precise-reference SP3 availability probe for "
            "pending LAGEOS prospective temporal OD campaigns."
        ),
        "official_source_urls": OFFICIAL_SOURCE_URLS,
        "probe_defaults": {
            "weeks": list(weeks),
            "satellites": list(satellites),
            "centers": list(centers),
            "versions": list(versions),
            "sources": list(args.sources),
        },
        "classification_rule": (
            "usable_sp3 is true only for gzip responses whose decompressed "
            "content has an SP3-like header. HTML placeholders, Earthdata "
            "Login pages, redirects, directory listings, and non-SP3 gzip "
            "content are not usable products."
        ),
        "claim_boundary": {
            "unavailable_products_are_scored_validation": False,
            "availability_probe_is_scored_validation": False,
            "independent_reproduction_completed_by_this_probe": False,
            "can_upgrade_manuscript_external_validation_claim": False,
            "interpretation": (
                "This report is an availability gate only. It is not scored "
                "validation unless required products become valid gzip/SP3 "
                "files and the frozen campaign command is actually run and "
                "reported."
            ),
        },
        "parent_listing_probes": parent_listings,
        "week_listing_probes": week_listings,
        "positive_control_direct_product_probes": positive_control_products,
        "direct_product_probes": direct_products,
        "local_cached_product_probes": local_cached_products,
        "availability_summary": summary,
        "campaign_commands": build_campaign_commands(),
    }


def write_reports(report: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = report["availability_summary"]
    lines = [
        "# ILRS Precise-Reference Availability Probe",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        "## Boundary",
        "",
        "This is an official-source availability gate, not scored validation. "
        "Unavailable products are not validation evidence, and this report does "
        "not establish independent-machine reproduction.",
        "",
        "## Overall Status",
        "",
        (
            f"- All required default satellite/week products available: "
            f"`{summary['all_required_products_available']}`"
        ),
        "- Usable product rule: gzip bytes plus SP3-like decompressed header.",
        "",
        "## Satellite/Week Matrix",
        "",
        "| Satellite | Week | Usable SP3 | Usable candidates | Classifications |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary["by_satellite_week"].values():
        lines.append(
            "| "
            f"{row['satellite']} | {row['week']} | "
            f"{row['usable_sp3']} | {row['usable_candidate_count']} | "
            f"{', '.join(row['classifications']) or 'none'} |"
        )
    lines.extend(["", "## Schedule Readiness", ""])
    for schedule, row in summary["schedule_readiness"].items():
        lines.append(
            f"- `{schedule}`: `{row['status']}`; required weeks "
            f"`{', '.join(row['required_weeks'])}`."
        )
    lines.extend(["", "## Direct Product Probes", ""])
    lines.extend(
        [
            "| Source | Satellite | Week | Center | Version | HTTP | Type | Bytes | Usable | Classification |",
            "|---|---|---:|---|---|---:|---|---:|---:|---|",
        ]
    )
    for row in report["direct_product_probes"]:
        lines.append(
            "| "
            f"{row['source']} | {row['satellite']} | {row['week']} | "
            f"{row['center']} | {row['version']} | {row.get('http_status')} | "
            f"{row.get('content_type') or ''} | {row.get('byte_length')} | "
            f"{row.get('usable_sp3')} | {row.get('classification')} |"
        )
    lines.extend(["", "## Historical EDC Positive Controls", ""])
    lines.append(
        "These known historical products test the direct EDC URL pattern. They "
        "do not make pending weeks available or scoreable."
    )
    lines.extend(
        [
            "",
            "| Satellite | Week | HTTP | Bytes | SHA-256 | Usable | Classification |",
            "|---|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in report["positive_control_direct_product_probes"]:
        lines.append(
            "| "
            f"{row['satellite']} | {row['week']} | {row.get('http_status')} | "
            f"{row.get('byte_length')} | `{row.get('sha256')}` | "
            f"{row.get('usable_sp3')} | {row.get('classification')} |"
        )
    lines.extend(["", "## Local Cached Product Checks", ""])
    lines.append(
        "Cached files with pending `.sp3.gz` names are not treated as products "
        "unless the gzip/SP3 gate passes. HTML placeholders under SP3 filenames "
        "remain unavailable/non-usable."
    )
    lines.extend(
        [
            "",
            "| Path | Exists | Bytes | SHA-256 | Usable | Campaign input kind | Classification |",
            "|---|---:|---:|---|---:|---|---|",
        ]
    )
    for row in report["local_cached_product_probes"]:
        alignment = row.get("campaign_runner_alignment", {})
        lines.append(
            "| "
            f"`{row['path']}` | {row['exists']} | {row.get('byte_length')} | "
            f"`{row.get('sha256')}` | {row.get('usable_sp3')} | "
            f"{alignment.get('campaign_input_kind')} | {row.get('classification')} |"
        )
    lines.extend(["", "## Campaign Commands", ""])
    lines.append(
        "Run these only after the schedule readiness row is `available_to_score` "
        "and all needed predeclaration boundaries are valid."
    )
    for row in report["campaign_commands"]:
        lines.extend(["", f"### `{row['schedule']}`", ""])
        if row.get("predeclaration_creation_command_line"):
            lines.extend(
                [
                    "Predeclaration command, only if this would still be a valid "
                    "pre-scoring rule:",
                    "",
                    "```powershell",
                    row["predeclaration_creation_command_line"],
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "Scoring command:",
                "",
                "```powershell",
                row["scoring_command_line"],
                "```",
                "",
                f"Note: {row['claim_note']}",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- JSON: `{repo_rel(json_out)}`",
            f"- Markdown: `{repo_rel(md_out)}`",
        ]
    )
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weeks", nargs="+", default=list(DEFAULT_WEEKS))
    parser.add_argument("--satellites", nargs="+", default=list(DEFAULT_SATELLITES))
    parser.add_argument("--centers", nargs="+", default=list(DEFAULT_CENTERS))
    parser.add_argument("--versions", nargs="+", default=list(DEFAULT_VERSIONS))
    parser.add_argument(
        "--positive-control-weeks",
        nargs="*",
        default=list(DEFAULT_POSITIVE_CONTROL_WEEKS),
        help=(
            "Historical EDC weeks to probe as source-health controls. These "
            "do not affect pending-week readiness."
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["edc", "cddis"],
        choices=["edc", "cddis"],
    )
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--local-cache-dirs",
        nargs="*",
        default=list(DEFAULT_LOCAL_CACHE_DIRS),
        help=(
            "Local directories whose cached pending .sp3.gz filenames should "
            "be classified with the same gzip/SP3 gate. Use an empty argument "
            "list to skip local cache checks."
        ),
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    return parser


def default_output_paths(now: dt.datetime | None = None) -> tuple[Path, Path]:
    now = now or utc_now()
    stamp = now.strftime("%Y%m%d")
    base = ROOT / "results" / "validation" / f"ilrs_precise_reference_availability_{stamp}"
    return base.with_suffix(".json"), base.with_suffix(".md")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    default_json, default_md = default_output_paths()
    json_out = args.json_out or default_json
    md_out = args.md_out or default_md
    json_out = json_out if json_out.is_absolute() else ROOT / json_out
    md_out = md_out if md_out.is_absolute() else ROOT / md_out
    args.sources = tuple(source.lower() for source in args.sources)

    report = build_report(args)
    write_reports(report, json_out, md_out)
    print(
        json.dumps(
            {
                "status": "available" if report["availability_summary"]["all_required_products_available"] else "pending_unavailable",
                "json": repo_rel(json_out),
                "markdown": repo_rel(md_out),
                "all_required_products_available": report["availability_summary"][
                    "all_required_products_available"
                ],
                "usable_direct_product_count": sum(
                    1 for row in report["direct_product_probes"] if row.get("usable_sp3")
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
