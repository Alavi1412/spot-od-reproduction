#!/usr/bin/env python
"""Fetch cached public orbital-catalog and ground-station snapshots."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import collections
import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso


CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=JSON"
SATNOGS_URL = "https://network.satnogs.org/api/stations/?format=json"
SATNOGS_OBSERVATIONS_URL = "https://network.satnogs.org/api/observations/?format=json&status=good"
SATNOGS_OBSERVATION_BROWSE_URL = "https://network.satnogs.org/observations/?future=0&page={page}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-out", type=str, default="configs/public_celestrak_active_snapshot.json")
    parser.add_argument("--stations-out", type=str, default="configs/public_satnogs_stations_snapshot.json")
    parser.add_argument(
        "--observations-out",
        type=str,
        default="configs/public_satnogs_recent_good_observations.json",
    )
    parser.add_argument("--manifest-out", type=str, default="configs/public_tracking_manifest.json")
    parser.add_argument("--observation-pages", type=int, default=60)
    parser.add_argument("--observation-max-records", type=int, default=180)
    parser.add_argument("--observation-max-per-satellite", type=int, default=4)
    return parser


def powershell_download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"$ProgressPreference='SilentlyContinue'; "
        f"Invoke-WebRequest -UseBasicParsing -Uri '{url}' -OutFile '{output_path.resolve()}'"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )


def python_download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        output_path.write_bytes(response.read())


def fetch_text(url: str, *, max_attempts: int = 4, backoff_s: float = 1.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            attempt += 1
            if attempt >= max_attempts or exc.code not in {429, 500, 502, 503, 504}:
                raise
        except urllib.error.URLError:
            attempt += 1
            if attempt >= max_attempts:
                raise
        time.sleep(backoff_s * attempt)


def materialize_snapshot(url: str, output_path: Path, fallback_paths: list[Path]) -> None:
    try:
        powershell_download(url, output_path)
        return
    except subprocess.CalledProcessError:
        pass
    try:
        python_download(url, output_path)
        return
    except Exception:
        pass
    for fallback in fallback_paths:
        if fallback.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if fallback.resolve() != output_path.resolve():
                shutil.copyfile(fallback, output_path)
            return
    if output_path.exists():
        return
    raise RuntimeError(f"Failed to download {url} and no fallback snapshot exists for {output_path}.")


def load_station_lookup(path: Path) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    return {int(row["id"]): dict(row) for row in payload}


def extract_observation_listing_rows(html_text: str) -> list[tuple[int, str]]:
    pattern = re.compile(
        r'data-href="/observations/(?P<obs_id>\d+)/">.*?<span class="badge badge-(?P<badge>[a-z]+)">',
        re.IGNORECASE | re.DOTALL,
    )
    return [(int(match.group("obs_id")), str(match.group("badge")).lower()) for match in pattern.finditer(html_text)]


def _extract_first(pattern: str, text: str) -> tuple[str, ...] | None:
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return tuple(group.strip() for group in match.groups())


def parse_observation_detail_html(
    html_text: str,
    *,
    observation_id: int,
    status_badge: str,
    station_lookup: dict[int, dict],
) -> dict | None:
    sat_match = _extract_first(
        r'<span class="badge badge-secondary">Satellite</span>\s*</td>\s*<td>\s*<a[^>]*>\s*(\d+)\s*-\s*([^<]+?)\s*</a>',
        html_text,
    )
    station_match = _extract_first(
        r'<span class="badge badge-secondary">Station</span>\s*</td>\s*<td>\s*<a href="/stations/(\d+)/"[^>]*>\s*(?:\d+\s*-\s*)?([^<]+?)\s*</a>',
        html_text,
    )
    polar_tag = _extract_first(r'(<svg[^>]*id="polar"[^>]*>)', html_text)
    if not sat_match or not station_match or not polar_tag:
        return None
    polar = polar_tag[0]

    def attr(name: str) -> str | None:
        match = re.search(rf'{re.escape(name)}="([^"]+)"', polar, re.IGNORECASE)
        return match.group(1).strip() if match else None

    tle1 = attr("data-tle1")
    tle2 = attr("data-tle2")
    start = attr("data-timeframe-start")
    end = attr("data-timeframe-end")
    if not tle1 or not tle2 or not start or not end:
        return None

    station_id = int(station_match[0])
    station_meta = station_lookup.get(station_id, {})
    rating_status = _extract_first(
        r'<span class="badge badge-secondary">Status</span>.*?<span[^>]*>([^<]+)</span>\s*</span>',
        html_text,
    )
    waterfall_status = _extract_first(r'id="waterfall-status-badge"[^>]*>([^<]+)</div>', html_text)
    station_lat = station_meta.get("lat", attr("data-groundstation-lat"))
    station_lng = station_meta.get("lng", attr("data-groundstation-lon"))
    station_alt = station_meta.get("altitude", attr("data-groundstation-alt"))
    min_horizon = station_meta.get("min_horizon", 0)
    return {
        "id": observation_id,
        "norad_cat_id": int(sat_match[0]),
        "satellite_name": sat_match[1],
        "ground_station": station_id,
        "station_name": station_meta.get("name", station_match[1]),
        "status": status_badge.lower(),
        "vetted_status": (waterfall_status or rating_status or ("unknown",))[0].lower(),
        "waterfall_status": (waterfall_status or ("unknown",))[0].lower(),
        "start": start.replace("+00:00", "Z"),
        "end": end.replace("+00:00", "Z"),
        "tle1": tle1,
        "tle2": tle2,
        "source_url": f"https://network.satnogs.org/observations/{observation_id}/",
        "station_lat": float(station_lat),
        "station_lng": float(station_lng),
        "station_alt": float(station_alt),
        "min_horizon": float(min_horizon),
    }


def select_observation_records(
    records: list[dict],
    *,
    max_records: int,
    max_per_satellite: int,
) -> list[dict]:
    selected: list[dict] = []
    counts: collections.Counter[int] = collections.Counter()
    for record in records:
        norad_cat_id = int(record["norad_cat_id"])
        if counts[norad_cat_id] >= max_per_satellite:
            continue
        selected.append(record)
        counts[norad_cat_id] += 1
        if len(selected) >= max_records:
            break
    return selected


def scrape_recent_good_observations(
    *,
    station_snapshot_path: Path,
    max_pages: int,
    max_records: int,
    max_per_satellite: int,
) -> tuple[list[dict], dict]:
    station_lookup = load_station_lookup(station_snapshot_path)
    listing_rows: list[tuple[int, str]] = []
    for page in range(1, max_pages + 1):
        listing_html = fetch_text(SATNOGS_OBSERVATION_BROWSE_URL.format(page=page))
        listing_rows.extend(extract_observation_listing_rows(listing_html))
        time.sleep(0.05)
    deduped_good_rows: list[tuple[int, str]] = []
    seen_obs: set[int] = set()
    for observation_id, badge in listing_rows:
        if badge != "good" or observation_id in seen_obs:
            continue
        deduped_good_rows.append((observation_id, badge))
        seen_obs.add(observation_id)

    detailed_records: list[dict] = []
    for observation_id, badge in deduped_good_rows:
        detail_html = fetch_text(f"https://network.satnogs.org/observations/{observation_id}/")
        record = parse_observation_detail_html(
            detail_html,
            observation_id=observation_id,
            status_badge=badge,
            station_lookup=station_lookup,
        )
        if record is not None:
            detailed_records.append(record)
        time.sleep(0.05)
    selected_records = select_observation_records(
        detailed_records,
        max_records=max_records,
        max_per_satellite=max_per_satellite,
    )
    if not selected_records:
        raise RuntimeError("Failed to scrape any recent good SatNOGS observations from public HTML pages.")
    metadata = {
        "collection_method": "html_listing_detail_scrape",
        "pages_scanned": max_pages,
        "listing_row_count": len(listing_rows),
        "good_candidate_count": len(deduped_good_rows),
        "detail_record_count": len(detailed_records),
        "selected_count": len(selected_records),
        "selected_unique_satellites": len({int(row["norad_cat_id"]) for row in selected_records}),
        "selected_unique_stations": len({str(row["station_name"]) for row in selected_records}),
        "max_per_satellite": max_per_satellite,
    }
    return selected_records, metadata


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("observations", []))
    return int(len(payload))


def main() -> None:
    args = build_parser().parse_args()
    catalog_out = Path(args.catalog_out)
    stations_out = Path(args.stations_out)
    observations_out = Path(args.observations_out)
    manifest_out = Path(args.manifest_out)

    materialize_snapshot(
        CELESTRAK_URL,
        catalog_out,
        fallback_paths=[Path("tmp_celestrak_active.json"), catalog_out],
    )
    materialize_snapshot(
        SATNOGS_URL,
        stations_out,
        fallback_paths=[Path("tmp_satnogs_stations_page1.json"), stations_out],
    )
    observation_meta: dict | None = None
    try:
        observations, observation_meta = scrape_recent_good_observations(
            station_snapshot_path=stations_out,
            max_pages=max(1, int(args.observation_pages)),
            max_records=max(1, int(args.observation_max_records)),
            max_per_satellite=max(1, int(args.observation_max_per_satellite)),
        )
        dump_json({"observations": observations}, observations_out)
    except Exception as exc:
        materialize_snapshot(
            SATNOGS_OBSERVATIONS_URL,
            observations_out,
            fallback_paths=[observations_out],
        )
        observation_meta = {
            "collection_method": "api_or_cached_fallback",
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }

    manifest = {
        "fetched_at_utc": utc_now_iso(),
        "catalog": {
            "url": CELESTRAK_URL,
            "path": str(catalog_out),
            "sha256": sha256_file(catalog_out),
            "count": load_json_count(catalog_out),
        },
        "stations": {
            "url": SATNOGS_URL,
            "path": str(stations_out),
            "sha256": sha256_file(stations_out),
            "count": load_json_count(stations_out),
        },
        "observations": {
            "url": SATNOGS_OBSERVATIONS_URL,
            "path": str(observations_out),
            "sha256": sha256_file(observations_out),
            "count": load_json_count(observations_out),
            **(observation_meta or {}),
        },
    }
    dump_json(manifest, manifest_out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
