from __future__ import annotations

import gzip

from scripts.probe_ilrs_precise_reference_availability import (
    classify_local_cached_product,
    classify_candidate_bytes,
    classify_listing_bytes,
    sp3_header_valid,
)


def _gzip(raw: bytes) -> bytes:
    return gzip.compress(raw)


def test_classify_candidate_accepts_gzip_sp3_bytes() -> None:
    raw = _gzip(
        b"#cP2026 6 15  0  0  0.00000000      96 ORBIT IGS14 HLM  NGS\n"
        b"## 2423  86400.00000000   900.00000000 60000 0.0000000000000\n"
        b"+   1   L51\n"
        b"EOF\n"
    )

    result = classify_candidate_bytes(raw, content_type="application/gzip")

    assert result["usable_sp3"] is True
    assert result["classification"] == "usable_gzip_sp3"
    assert result["gzip_magic"] is True
    assert result["gzip_valid"] is True
    assert result["sp3_header_valid"] is True
    assert result["sha256"]


def test_classify_candidate_rejects_html_placeholder_even_with_200_type() -> None:
    raw = (
        b"<!DOCTYPE html><html><head><title>Not Found</title></head>"
        b"<body>missing</body></html>"
    )
    raw += b" " * (825 - len(raw))

    result = classify_candidate_bytes(raw, content_type="text/html; charset=utf-8")

    assert result["usable_sp3"] is False
    assert result["classification"] == "html_or_login_not_sp3"
    assert result["html_like"] is True
    assert result["gzip_magic"] is False
    assert result["byte_length"] == 825


def test_classify_candidate_rejects_earthdata_login_html() -> None:
    raw = b"<html><title>Earthdata Login</title><body>urs.earthdata.nasa.gov</body></html>"

    result = classify_candidate_bytes(
        raw,
        content_type="text/html",
        final_url="https://urs.earthdata.nasa.gov/oauth/authorize",
    )

    assert result["usable_sp3"] is False
    assert result["login_like"] is True
    assert result["classification"] == "html_or_login_not_sp3"


def test_classify_candidate_rejects_valid_gzip_non_sp3() -> None:
    raw = _gzip(b"not an sp3 file\nstill not an sp3 file\n")

    result = classify_candidate_bytes(raw, content_type="application/gzip")

    assert result["usable_sp3"] is False
    assert result["gzip_valid"] is True
    assert result["sp3_header_valid"] is False
    assert result["classification"] == "gzip_not_sp3"


def test_classify_listing_extracts_weeks_and_matching_sp3_candidates() -> None:
    html = b"""
    <html><body>
    <a href="260613/">260613/</a>
    <a href="260620/">260620/</a>
    <a href="nsgf.orb.lageos1.260620.v80.sp3.gz">sp3</a>
    <a href="esa.orb.lageos1.260620.v70.sp3.gz">sp3</a>
    </body></html>
    """

    result = classify_listing_bytes(
        html,
        content_type="text/html",
        expected_week="260620",
        expected_satellite="lageos1",
    )

    assert result["classification"] == "directory_listing"
    assert result["expected_week_listed"] is True
    assert result["listed_weeks"] == ["260613", "260620"]
    assert result["matching_sp3_candidate_count"] == 2
    assert "nsgf.orb.lageos1.260620.v80.sp3.gz" in result["matching_sp3_candidates"]


def test_classify_local_cached_sp3_gz_placeholder_aligns_with_campaign_gate(tmp_path) -> None:
    cached = tmp_path / "nsgf.orb.lageos1.260620.v80.sp3.gz"
    cached.write_bytes(b"<html><body>missing SP3 product</body></html>" + b" " * 780)

    result = classify_local_cached_product(
        cached,
        satellite="lageos1",
        week="260620",
        center="nsgf",
        version="v80",
    )

    assert result["exists"] is True
    assert result["usable_sp3"] is False
    assert result["gzip_magic"] is False
    assert result["classification"] == "html_or_login_not_sp3"
    assert result["campaign_runner_alignment"]["campaign_input_kind"] == (
        "sp3_not_valid_gzip"
    )
    assert (
        result["campaign_runner_alignment"]["campaign_counts_as_available_input"]
        is False
    )


def test_sp3_header_requires_hash_header_and_sp3_body_marker() -> None:
    assert sp3_header_valid("#cP2026 6 15\n## second line\nEOF\n") is True
    assert sp3_header_valid("not sp3\n## second line\nEOF\n") is False
    assert sp3_header_valid("#cP2026 6 15\njust text\n") is False
