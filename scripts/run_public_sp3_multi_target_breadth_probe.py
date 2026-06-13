#!/usr/bin/env python
"""Public multi-target SP3/CRD breadth probe.

This additive probe broadens the public-data evidence base beyond the existing
LAGEOS-only slices where the EDC public archive supports it.  For each
predeclared weekly window, it discovers the public SP3 precise-orbit product
for each target, detects the satellite identifier from the SP3 file, and runs
a deterministic state-propagation readout from fixed interior SP3 epochs
against the same public precise reference.  For each predeclared day, it also
records CRD normal-point and station coverage.

The result is a bounded public precise-reference and coverage probe.  It is
not operational POD, not centimetre SLR validation, not a flight-readiness
claim, and not central external validation of the simulator conclusion.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - entrypoint dependent
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnn_state_estimation.sp3 import parse_sp3
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

try:  # noqa: E402
    from run_real_slr_sp3_hifi_validation import (
        ANALYSIS_CENTER,
        MAX_STEP_S,
        PD_EDGE_MARGIN_S,
        PD_GRID_S,
        PD_HORIZON_S,
        PD_N_STARTS,
        SPLIT_WEEKS,
        GcrsInterp,
        _pd_start_epochs,
        _propagate_grid,
    )
except ModuleNotFoundError:  # pragma: no cover - import context dependent
    from scripts.run_real_slr_sp3_hifi_validation import (
        ANALYSIS_CENTER,
        MAX_STEP_S,
        PD_EDGE_MARGIN_S,
        PD_GRID_S,
        PD_HORIZON_S,
        PD_N_STARTS,
        SPLIT_WEEKS,
        GcrsInterp,
        _pd_start_epochs,
        _propagate_grid,
    )


EDC_ROOT = "https://edc.dgfi.tum.de"
EDC_CRD_URL = (
    EDC_ROOT
    + "/pub/slr/data/npt_crd_v2/{sat_key}/{year}/{sat_key}_{date}.np2"
)
EDC_SP3_DIR_URL = EDC_ROOT + "/pub/slr/products/orbits/{sat_key}/{week}/"
DEFAULT_OUT_DIR = Path("results/public_sp3_multi_target_breadth_probe")
DEFAULT_TABLE = Path("paper/tables/public_sp3_multi_target_breadth_probe.tex")
SCHEMA_VERSION = "public_sp3_multi_target_breadth_probe_v1"
BOOTSTRAP_SEED = 20260522
BOOTSTRAP_N = 5000
USER_AGENT = "Mozilla/5.0"


@dataclass(frozen=True)
class TargetSpec:
    target: str
    sat_key: str
    is_lageos: bool = False


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("AJISAI", "ajisai"),
    TargetSpec("ETALON-1", "etalon1"),
    TargetSpec("ETALON-2", "etalon2"),
    TargetSpec("LAGEOS-1", "lageos1", True),
    TargetSpec("LAGEOS-2", "lageos2", True),
    TargetSpec("LARES", "lares"),
    TargetSpec("LARES-2", "lares2"),
    TargetSpec("LARETS", "larets"),
    TargetSpec("STARLETTE", "starlette"),
    TargetSpec("STELLA", "stella"),
)


class InputUnavailable(RuntimeError):
    """Raised when an archive input cannot be materialized as the expected file."""


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _urlopen_bytes(url: str, timeout_s: float = 90.0) -> bytes:
    with urllib.request.urlopen(_request(url), timeout=timeout_s) as resp:
        return resp.read()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_html(raw: bytes) -> bool:
    head = raw[:256].lstrip().lower()
    return head.startswith(b"<html") or b"<body" in head


def _read_sp3_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b" or path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


def _validate_sp3_bytes(raw: bytes) -> None:
    if _is_html(raw):
        raise InputUnavailable("archive response was HTML, not an SP3 file")
    try:
        text = gzip.decompress(raw).decode("utf-8", "replace")
    except OSError as exc:
        raise InputUnavailable("SP3 download was not valid gzip data") from exc
    if not text.startswith("#") or "\n*" not in text:
        raise InputUnavailable("downloaded gzip did not contain an SP3 product")


def _validate_crd_bytes(raw: bytes) -> None:
    if _is_html(raw):
        raise InputUnavailable("archive response was HTML, not a CRD file")
    text = raw.decode("utf-8", "replace")
    tags = {line.split()[0].lower() for line in text.splitlines() if line.split()}
    if "h1" not in tags and "11" not in tags:
        raise InputUnavailable("downloaded file did not look like CRD normal points")


def materialize_verified(
    url: str, path: Path, *, kind: str, refresh: bool = False
) -> str:
    """Download once, then reuse offline; verify that archive errors are not cached."""
    validator = _validate_sp3_bytes if kind == "sp3" else _validate_crd_bytes
    if path.exists() and not refresh:
        raw = path.read_bytes()
        validator(raw)
        return "archived_input"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _urlopen_bytes(url)
    validator(raw)
    path.write_bytes(raw)
    return "public_archive_download"


def discover_sp3_product(
    sat_key: str, week: str, out_dir: Path, *, refresh: bool = False
) -> dict:
    """Find the public SP3 product for a target/week, preferring NSGF."""
    archive_dir = out_dir / "sp3" / sat_key
    archived = sorted(
        archive_dir.glob(f"*.orb.{sat_key}.{week}.v*.sp3.gz"),
        key=lambda p: (not p.name.startswith("nsgf."), p.name),
    )
    if archived and not refresh:
        name = archived[0].name
        return {
            "status": "archived",
            "directory_url": EDC_SP3_DIR_URL.format(sat_key=sat_key, week=week),
            "filename": name,
            "url": EDC_SP3_DIR_URL.format(sat_key=sat_key, week=week) + name,
            "path": archived[0],
            "discovery_method": "archived_file_scan",
        }

    directory_url = EDC_SP3_DIR_URL.format(sat_key=sat_key, week=week)
    text = _urlopen_bytes(directory_url, timeout_s=45.0).decode(
        "utf-8", "replace"
    )
    hrefs = re.findall(r"href=['\"]([^'\"]+\.sp3\.gz)['\"]", text)
    names = [Path(urllib.parse.urlparse(h).path).name for h in hrefs]
    pattern = re.compile(
        rf"^[a-z0-9]+\.orb\.{re.escape(sat_key)}\.{re.escape(week)}"
        r"\.v[0-9]+\.sp3\.gz$"
    )
    candidates = sorted({name for name in names if pattern.match(name)})
    if not candidates:
        raise InputUnavailable(f"no SP3 product listed for {sat_key} {week}")
    nsgf = [name for name in candidates if name.startswith("nsgf.")]
    chosen = sorted(nsgf or candidates)[0]
    url = directory_url + chosen
    return {
        "status": "listed",
        "directory_url": directory_url,
        "filename": chosen,
        "url": url,
        "path": archive_dir / chosen,
        "discovery_method": "edc_directory_listing_prefer_nsgf",
        "listed_candidate_count": len(candidates),
        "listed_candidates": candidates,
    }


def detect_sp3_satellite_ids(text: str) -> list[str]:
    """Detect satellite IDs from SP3 header and P/V records."""
    found: list[str] = []
    seen: set[str] = set()
    token_re = re.compile(r"^[A-Z][A-Z0-9]{2}$")

    def add(token: str) -> None:
        tok = token.strip()
        if token_re.match(tok) and tok not in seen:
            found.append(tok)
            seen.add(tok)

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("++"):
            continue
        if line.startswith("+"):
            for tok in line.split()[1:]:
                add(tok)
        elif line.startswith(("P", "V")) and len(line) >= 4:
            add(line[1:4])
    return found


def detect_single_sp3_satellite_id(text: str) -> str:
    ids = detect_sp3_satellite_ids(text)
    if len(ids) != 1:
        raise ValueError(f"expected exactly one SP3 satellite ID, found {ids}")
    return ids[0]


def _iso(epoch_unix: float) -> str:
    return (
        dt.datetime.fromtimestamp(float(epoch_unix), tz=dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _round(x: float | None, ndigits: int = 2):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), ndigits)


def _bootstrap_ci(values: np.ndarray, *, seed: int = BOOTSTRAP_SEED) -> list:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_N, arr.size))
    means = arr[idx].mean(axis=1)
    return [
        round(float(np.percentile(means, 2.5)), 2),
        round(float(np.percentile(means, 97.5)), 2),
    ]


def _cluster_bootstrap_ci(clusters: list[np.ndarray], *, seed_offset: int) -> list:
    clean = []
    for values in clusters:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            clean.append(arr)
    if not clean:
        return [None, None]
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    idx = rng.integers(0, len(clean), size=(BOOTSTRAP_N, len(clean)))
    means = np.empty(BOOTSTRAP_N, dtype=np.float64)
    for i, sample in enumerate(idx):
        means[i] = np.concatenate([clean[j] for j in sample]).mean()
    return [
        round(float(np.percentile(means, 2.5)), 2),
        round(float(np.percentile(means, 97.5)), 2),
    ]


def paired_improvement_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired improvement of a over b (b-a; positive means a has lower RMSE)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"n": 0}
    diff = b[mask] - a[mask]
    ci = _bootstrap_ci(diff)
    return {
        "n": int(diff.size),
        "a_mean_rms_m": _round(a[mask].mean()),
        "b_mean_rms_m": _round(b[mask].mean()),
        "mean_improvement_m": _round(diff.mean()),
        "median_improvement_m": _round(np.median(diff)),
        "n_a_better": int((diff > 0.0).sum()),
        "fraction_a_better": round(float((diff > 0.0).mean()), 3),
        "bootstrap95_mean_improvement_m": ci,
        "start_epoch_bootstrap95_mean_improvement_m": ci,
        "bootstrap_unit": "fixed_start_epoch",
        "uncertainty_scope": (
            "finite-probe start-epoch readout; not target-population or "
            "operational uncertainty"
        ),
        "improvement_convention": (
            "b_minus_a; positive means candidate a has lower SP3-state RMSE"
        ),
    }


def _week_improvement_values(week: dict) -> np.ndarray:
    values = []
    for row in week.get("start_epoch_scores", []):
        compact = row.get("compact_rmse_m")
        hifi = row.get("hifi_rmse_m")
        if compact is None or hifi is None:
            continue
        if np.isfinite(float(compact)) and np.isfinite(float(hifi)):
            values.append(float(compact) - float(hifi))
    return np.asarray(values, dtype=np.float64)


def clustered_improvement_uncertainty(weeks: list[dict]) -> dict:
    """Clustered finite-probe sensitivity for hifi-over-compact improvement."""
    target_week_clusters: list[np.ndarray] = []
    target_clusters: dict[str, list[float]] = {}
    for week in weeks:
        if week.get("status") != "completed":
            continue
        values = _week_improvement_values(week)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        target_week_clusters.append(values)
        target_clusters.setdefault(str(week.get("target")), []).extend(
            float(v) for v in values
        )

    all_values = (
        np.concatenate(target_week_clusters)
        if target_week_clusters
        else np.asarray([], dtype=np.float64)
    )
    target_level = [
        np.asarray(values, dtype=np.float64) for values in target_clusters.values()
    ]
    return {
        "n_start_epochs": int(all_values.size),
        "n_target_week_clusters": int(len(target_week_clusters)),
        "target_week_mean_improvement_m": _round(all_values.mean())
        if all_values.size
        else None,
        "target_week_bootstrap95_mean_improvement_m": _cluster_bootstrap_ci(
            target_week_clusters, seed_offset=101
        ),
        "n_target_clusters": int(len(target_level)),
        "target_mean_improvement_m": _round(all_values.mean())
        if all_values.size
        else None,
        "target_bootstrap95_mean_improvement_m": _cluster_bootstrap_ci(
            target_level, seed_offset=202
        ),
        "improvement_convention": (
            "compact_minus_higher_fidelity; positive means higher-fidelity "
            "propagation has lower SP3-state RMSE"
        ),
        "uncertainty_scope": (
            "finite-probe clustered sensitivity over the scored target-weeks "
            "and targets; not target-population, target-epoch-population, or "
            "operational uncertainty"
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_N,
    }


def summarize_state_rows(rows: list[dict]) -> dict:
    completed = [
        r
        for r in rows
        if r.get("status") == "completed"
        and r.get("compact_rmse_m") is not None
        and r.get("hifi_rmse_m") is not None
    ]
    if not completed:
        return {"n_start_epochs": 0}
    c = np.array([r["compact_rmse_m"] for r in completed], dtype=np.float64)
    h = np.array([r["hifi_rmse_m"] for r in completed], dtype=np.float64)
    return {
        "n_start_epochs": int(len(completed)),
        "compact_mean_rms_m": _round(c.mean()),
        "compact_median_rms_m": _round(np.median(c)),
        "hifi_mean_rms_m": _round(h.mean()),
        "hifi_median_rms_m": _round(np.median(h)),
        "hifi_vs_compact": paired_improvement_summary(h, c),
    }


def summarize_state_weeks(weeks: list[dict]) -> dict:
    rows: list[dict] = []
    for week in weeks:
        rows.extend(week.get("start_epoch_scores", []))
    summary = summarize_state_rows(rows)
    if summary.get("n_start_epochs", 0) > 0:
        summary["cluster_uncertainty"] = clustered_improvement_uncertainty(weeks)
    return summary


def parse_crd_coverage(text: str) -> dict:
    """Count all CRD v2 normal-point records without filtering station IDs."""
    station_counts: dict[str, int] = {}
    station_meta: dict[str, dict] = {}
    pass_count = 0
    passes_with_normal_points = 0
    points_in_pass = 0
    cur_code: str | None = None
    cur_cdp: str | None = None

    def close_pass() -> None:
        nonlocal passes_with_normal_points, points_in_pass
        if points_in_pass > 0:
            passes_with_normal_points += 1
        points_in_pass = 0

    for raw in text.splitlines():
        parts = raw.split()
        if not parts:
            continue
        tag = parts[0].lower()
        if tag == "h2" and len(parts) >= 3:
            cur_code = parts[1]
            cur_cdp = parts[2]
        elif tag == "h4":
            close_pass()
            pass_count += 1
        elif tag in ("h8", "h9"):
            close_pass()
            cur_code = None
            cur_cdp = None
        elif tag == "11":
            station_key = f"{cur_code or 'unknown'}:{cur_cdp or 'unknown'}"
            station_counts[station_key] = station_counts.get(station_key, 0) + 1
            station_meta.setdefault(
                station_key,
                {"station_code": cur_code, "cdp_id": cur_cdp},
            )
            points_in_pass += 1
    close_pass()

    normal_points = int(sum(station_counts.values()))
    return {
        "normal_point_count": normal_points,
        "distinct_station_count": len(station_counts),
        "pass_count": int(pass_count),
        "passes_with_normal_points": int(passes_with_normal_points),
        "station_counts": [
            {
                **station_meta[key],
                "station_key": key,
                "normal_point_count": int(count),
            }
            for key, count in sorted(
                station_counts.items(), key=lambda kv: (-kv[1], kv[0])
            )
        ],
    }


def score_sp3_week(
    target: TargetSpec, week: str, split: str, out_dir: Path, *, refresh: bool
) -> dict:
    try:
        product = discover_sp3_product(target.sat_key, week, out_dir, refresh=refresh)
        input_method = materialize_verified(
            product["url"], product["path"], kind="sp3", refresh=refresh
        )
        text = _read_sp3_text(product["path"])
        sat_id = detect_single_sp3_satellite_id(text)
        eph = parse_sp3(text, sat_id)
        interp = GcrsInterp(eph, 9)
        starts = _pd_start_epochs(eph)
        rows = []
        for t0 in starts:
            x0 = interp.state_inertial(t0)
            compact = _propagate_grid(x0, t0, interp, hifi=False)
            hifi = _propagate_grid(x0, t0, interp, hifi=True)
            rows.append(
                {
                    "status": "completed",
                    "start_epoch_unix": float(t0),
                    "start_epoch_utc": _iso(t0),
                    "compact_rmse_m": _round(compact),
                    "hifi_rmse_m": _round(hifi),
                }
            )
        return {
            "target": target.target,
            "sat_key": target.sat_key,
            "is_lageos": target.is_lageos,
            "week": week,
            "split": split,
            "status": "completed" if rows else "insufficient_sp3_span",
            "sp3_sat_id": sat_id,
            "sp3_product": {
                "analysis_center_preferred": "NSGF",
                "analysis_center_recorded": eph.analysis_center or None,
                "analysis_center_context": ANALYSIS_CENTER,
                "directory_url": product["directory_url"],
                "url": product["url"],
                "filename": product["filename"],
                "discovery_method": product["discovery_method"],
                "input_source": input_method,
                "sha256": sha256_file(product["path"]),
                "bytes": product["path"].stat().st_size,
                "n_epochs": int(eph.epochs_unix.size),
                "start_utc": _iso(eph.start_unix),
                "end_utc": _iso(eph.end_unix),
                "coordinate_frame": eph.coordinate_frame,
                "time_system": eph.time_system,
            },
            "start_epoch_scores": rows,
            "summary": summarize_state_rows(rows),
        }
    except Exception as exc:
        return {
            "target": target.target,
            "sat_key": target.sat_key,
            "is_lageos": target.is_lageos,
            "week": week,
            "split": split,
            "status": "unavailable_or_unscored",
            "reason": str(exc),
        }


def summarize_crd_day(
    target: TargetSpec, date: str, week: str, split: str, out_dir: Path, *, refresh: bool
) -> dict:
    year = date[:4]
    url = EDC_CRD_URL.format(sat_key=target.sat_key, year=year, date=date)
    path = out_dir / "crd" / target.sat_key / f"{target.sat_key}_{date}.np2"
    try:
        input_method = materialize_verified(url, path, kind="crd", refresh=refresh)
        text = path.read_text(encoding="utf-8", errors="replace")
        coverage = parse_crd_coverage(text)
        return {
            "target": target.target,
            "sat_key": target.sat_key,
            "is_lageos": target.is_lageos,
            "date": date,
            "week": week,
            "split": split,
            "status": (
                "completed"
                if coverage["normal_point_count"] > 0
                else "no_normal_points"
            ),
            "crd": {
                "url": url,
                "filename": path.name,
                "input_source": input_method,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            },
            **coverage,
        }
    except Exception as exc:
        return {
            "target": target.target,
            "sat_key": target.sat_key,
            "is_lageos": target.is_lageos,
            "date": date,
            "week": week,
            "split": split,
            "status": "unavailable",
            "crd": {"url": url, "filename": path.name},
            "reason": str(exc),
            "normal_point_count": 0,
            "distinct_station_count": 0,
            "pass_count": 0,
            "passes_with_normal_points": 0,
            "station_counts": [],
        }


def coverage_summary(crd_days: list[dict]) -> dict:
    completed = [d for d in crd_days if d.get("status") == "completed"]
    station_keys = {
        s["station_key"]
        for d in completed
        for s in d.get("station_counts", [])
    }
    by_target = {}
    for target in TARGETS:
        rows = [d for d in crd_days if d["target"] == target.target]
        good = [d for d in rows if d.get("status") == "completed"]
        by_target[target.target] = {
            "attempted_days": len(rows),
            "days_with_normal_points": len(good),
            "normal_point_count": int(sum(d["normal_point_count"] for d in good)),
            "distinct_station_count": len(
                {
                    s["station_key"]
                    for d in good
                    for s in d.get("station_counts", [])
                }
            ),
        }
    return {
        "attempted_days": len(crd_days),
        "days_with_normal_points": len(completed),
        "normal_point_count": int(sum(d["normal_point_count"] for d in completed)),
        "distinct_station_count": len(station_keys),
        "by_target": by_target,
    }


def group_week_summaries(weeks: list[dict]) -> dict:
    completed = [w for w in weeks if w.get("status") == "completed"]
    by_split = {
        split: summarize_state_weeks([w for w in completed if w["split"] == split])
        for split in ("train", "val", "test")
    }
    by_target = {
        target.target: summarize_state_weeks(
            [w for w in completed if w["target"] == target.target]
        )
        for target in TARGETS
    }
    non_lageos = [w for w in completed if not w.get("is_lageos")]
    lageos = [w for w in completed if w.get("is_lageos")]
    return {
        "all": summarize_state_weeks(completed),
        "non_lageos": summarize_state_weeks(non_lageos),
        "lageos": summarize_state_weeks(lageos),
        "by_split": by_split,
        "by_target": by_target,
    }


def collect_input_digests(sp3_weeks: list[dict], crd_days: list[dict]) -> list[dict]:
    out = []
    for row in sp3_weeks:
        product = row.get("sp3_product")
        if isinstance(product, dict) and product.get("sha256"):
            out.append(
                {
                    "kind": "sp3",
                    "target": row["target"],
                    "week": row["week"],
                    "filename": product["filename"],
                    "sha256": product["sha256"],
                    "bytes": product["bytes"],
                    "url": product["url"],
                }
            )
    for row in crd_days:
        crd = row.get("crd")
        if isinstance(crd, dict) and crd.get("sha256"):
            out.append(
                {
                    "kind": "crd",
                    "target": row["target"],
                    "date": row["date"],
                    "filename": crd["filename"],
                    "sha256": crd["sha256"],
                    "bytes": crd["bytes"],
                    "url": crd["url"],
                }
            )
    return out


def build_result(out_dir: Path, *, refresh: bool = False) -> dict:
    sp3_weeks = []
    crd_days = []
    for week, (split, days) in SPLIT_WEEKS.items():
        for target in TARGETS:
            sp3_weeks.append(
                score_sp3_week(target, week, split, out_dir, refresh=refresh)
            )
            for date in days:
                crd_days.append(
                    summarize_crd_day(
                        target, date, week, split, out_dir, refresh=refresh
                    )
                )

    completed_sp3 = [w for w in sp3_weeks if w.get("status") == "completed"]
    completed_non_lageos = [w for w in completed_sp3 if not w.get("is_lageos")]
    completed_crd = [d for d in crd_days if d.get("status") == "completed"]
    scored_targets = sorted({w["target"] for w in completed_sp3})
    scored_non_lageos_targets = sorted(
        {w["target"] for w in completed_non_lageos}
    )
    summaries = group_week_summaries(sp3_weeks)
    crd_summary = coverage_summary(crd_days)
    expands_beyond_lageos = bool(completed_non_lageos)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now_iso(),
        "status": (
            "completed"
            if expands_beyond_lageos and summaries["all"]["n_start_epochs"] > 0
            else "feasibility_only"
        ),
        "targets_requested": [
            {"target": t.target, "sat_key": t.sat_key, "is_lageos": t.is_lageos}
            for t in TARGETS
        ],
        "split_weeks": {week: split for week, (split, _days) in SPLIT_WEEKS.items()},
        "predeclared_state_scoring": {
            "horizon_s": PD_HORIZON_S,
            "grid_s": PD_GRID_S,
            "n_start_epochs_per_target_week": PD_N_STARTS,
            "edge_margin_s": PD_EDGE_MARGIN_S,
            "max_step_s": MAX_STEP_S,
            "start_epoch_selection": (
                "fixed evenly spaced interior epochs from each SP3 product"
            ),
            "candidate_models": [
                "compact two-body+J2 propagation",
                "higher-fidelity deterministic propagation",
            ],
        },
        "selection_integrity": {
            "train_validation_test_labels_used_as_reporting_strata_only": True,
            "test_set_information_used_for_selection": False,
            "no_model_or_hyperparameter_selection_performed": True,
            "temporal_test_week": "260509",
            "validation_week": "260502",
        },
        "sp3_state_scoring": {
            "weeks": sp3_weeks,
            "summary": summaries,
        },
        "crd_coverage": {
            "days": crd_days,
            "summary": crd_summary,
        },
        "headline_readout": {
            "sp3_targets_scored": len(scored_targets),
            "sp3_targets_scored_list": scored_targets,
            "non_lageos_sp3_targets_scored": len(scored_non_lageos_targets),
            "non_lageos_sp3_targets_scored_list": scored_non_lageos_targets,
            "completed_target_weeks": len(completed_sp3),
            "completed_non_lageos_target_weeks": len(completed_non_lageos),
            "scored_start_epochs": summaries["all"]["n_start_epochs"],
            "non_lageos_scored_start_epochs": summaries["non_lageos"][
                "n_start_epochs"
            ],
            "all_compact_mean_rms_m": summaries["all"].get("compact_mean_rms_m"),
            "all_hifi_mean_rms_m": summaries["all"].get("hifi_mean_rms_m"),
            "non_lageos_compact_mean_rms_m": summaries["non_lageos"].get(
                "compact_mean_rms_m"
            ),
            "non_lageos_hifi_mean_rms_m": summaries["non_lageos"].get(
                "hifi_mean_rms_m"
            ),
            "test_compact_mean_rms_m": summaries["by_split"]["test"].get(
                "compact_mean_rms_m"
            ),
            "test_hifi_mean_rms_m": summaries["by_split"]["test"].get(
                "hifi_mean_rms_m"
            ),
            "crd_days_with_normal_points": len(completed_crd),
            "crd_normal_point_count": crd_summary["normal_point_count"],
            "crd_distinct_station_count": crd_summary["distinct_station_count"],
            "targets_beyond_lageos_successfully_included": expands_beyond_lageos,
        },
        "claim_boundary": {
            "defensible_status": (
                "bounded_public_multi_target_precise_reference_breadth_probe"
            ),
            "is_central_external_validation": False,
            "is_external_validation_of_simulator_conclusion": False,
            "is_operational_pod": False,
            "is_centimeter_slr_validation": False,
            "is_flight_readiness_validation": False,
            "claims_real_measurement_od": False,
            "crd_used_for_coverage_only": True,
            "expands_public_target_provenance_state_scoring_breadth_beyond_lageos": (
                expands_beyond_lageos
            ),
            "appropriate_use": (
                "Use as bounded public target/provenance and SP3-state "
                "scoring breadth evidence.  Do not use as an operational OD "
                "or centimetre SLR validation result."
            ),
            "why_no_multi_target_od_claim": [
                (
                    "The cross-target CRD files provide measurement coverage, "
                    "but this probe does not apply target-specific precise SLR "
                    "reductions, station-coordinate estimation, or centre-of-mass "
                    "correction models needed for defensible multi-target OD."
                ),
                (
                    "The SP3 arm starts from precise SP3 states and scores "
                    "state propagation against SP3, so it is a precise-reference "
                    "state-scoring breadth probe rather than real-measurement OD."
                ),
            ],
        },
        "input_digests": collect_input_digests(sp3_weeks, crd_days),
    }
    return result


def _fmt(x) -> str:
    if x is None:
        return "--"
    return f"{float(x):.2f}"


def _fmt_int(x) -> str:
    return str(int(x)) if x is not None else "--"


def _fmt_ci(ci) -> str:
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        return "--"
    if ci[0] is None or ci[1] is None:
        return "--"
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def write_table(result: dict, path: Path) -> None:
    h = result["headline_readout"]
    s = result["sp3_state_scoring"]["summary"]
    c = result["crd_coverage"]["summary"]
    rows = [
        (
            "All scored targets",
            h["sp3_targets_scored"],
            h["completed_target_weeks"],
            s["all"].get("n_start_epochs"),
            s["all"].get("compact_mean_rms_m"),
            s["all"].get("hifi_mean_rms_m"),
            s["all"].get("hifi_vs_compact", {}).get("mean_improvement_m"),
            s["all"].get("hifi_vs_compact", {}).get(
                "start_epoch_bootstrap95_mean_improvement_m"
            ),
            s["all"].get("cluster_uncertainty", {}).get(
                "target_week_bootstrap95_mean_improvement_m"
            ),
            s["all"].get("cluster_uncertainty", {}).get(
                "target_bootstrap95_mean_improvement_m"
            ),
            c["days_with_normal_points"],
            c["normal_point_count"],
        ),
        (
            "Non-LAGEOS targets",
            h["non_lageos_sp3_targets_scored"],
            h["completed_non_lageos_target_weeks"],
            s["non_lageos"].get("n_start_epochs"),
            s["non_lageos"].get("compact_mean_rms_m"),
            s["non_lageos"].get("hifi_mean_rms_m"),
            s["non_lageos"].get("hifi_vs_compact", {}).get("mean_improvement_m"),
            s["non_lageos"].get("hifi_vs_compact", {}).get(
                "start_epoch_bootstrap95_mean_improvement_m"
            ),
            s["non_lageos"].get("cluster_uncertainty", {}).get(
                "target_week_bootstrap95_mean_improvement_m"
            ),
            s["non_lageos"].get("cluster_uncertainty", {}).get(
                "target_bootstrap95_mean_improvement_m"
            ),
            sum(
                v["days_with_normal_points"]
                for k, v in c["by_target"].items()
                if not k.startswith("LAGEOS")
            ),
            sum(
                v["normal_point_count"]
                for k, v in c["by_target"].items()
                if not k.startswith("LAGEOS")
            ),
        ),
        (
            "Held-out test stratum",
            len(
                {
                    w["target"]
                    for w in result["sp3_state_scoring"]["weeks"]
                    if w.get("status") == "completed" and w.get("split") == "test"
                }
            ),
            len(
                [
                    w
                    for w in result["sp3_state_scoring"]["weeks"]
                    if w.get("status") == "completed" and w.get("split") == "test"
                ]
            ),
            s["by_split"]["test"].get("n_start_epochs"),
            s["by_split"]["test"].get("compact_mean_rms_m"),
            s["by_split"]["test"].get("hifi_mean_rms_m"),
            s["by_split"]["test"].get("hifi_vs_compact", {}).get(
                "mean_improvement_m"
            ),
            s["by_split"]["test"].get("hifi_vs_compact", {}).get(
                "start_epoch_bootstrap95_mean_improvement_m"
            ),
            s["by_split"]["test"].get("cluster_uncertainty", {}).get(
                "target_week_bootstrap95_mean_improvement_m"
            ),
            s["by_split"]["test"].get("cluster_uncertainty", {}).get(
                "target_bootstrap95_mean_improvement_m"
            ),
            len(
                [
                    d
                    for d in result["crd_coverage"]["days"]
                    if d.get("status") == "completed" and d.get("split") == "test"
                ]
            ),
            sum(
                d["normal_point_count"]
                for d in result["crd_coverage"]["days"]
                if d.get("status") == "completed" and d.get("split") == "test"
            ),
        ),
    ]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Public multi-target SP3/CRD breadth probe over the "
            r"predeclared April--May 2026 temporal windows. SP3 products are "
            r"used for deterministic precise-reference state-propagation "
            r"scoring from fixed interior epochs, and CRD normal-point files "
            r"are used only to summarize public measurement coverage. This "
            r"extends target and provenance breadth where public products are "
            r"available, but it is not operational POD or centimetre SLR "
            r"validation. The readout is state-scoring and coverage breadth "
            r"only, not multi-target OD validation.}"
        ),
        r"  \label{tab:public_sp3_multi_target_breadth_probe}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lcccccccccc}",
        r"    \toprule",
        (
            r"    Stratum & SP3 targets & Target-weeks & Starts & "
            r"Compact mean [m] & Higher-fidelity mean [m] & "
            r"HF improvement [m] & Start-epoch 95\% [m] & "
            r"Target-week cluster 95\% [m] & Target cluster 95\% [m] & "
            r"CRD normal points \\"
        ),
        r"    \midrule",
    ]
    for (
        label,
        targets,
        weeks,
        starts,
        compact,
        hifi,
        improvement,
        start_ci,
        target_week_ci,
        target_ci,
        _crd_days,
        crd_points,
    ) in rows:
        lines.append(
            f"    {label} & {_fmt_int(targets)} & {_fmt_int(weeks)} & "
            f"{_fmt_int(starts)} & {_fmt(compact)} & {_fmt(hifi)} & "
            f"{_fmt(improvement)} & {_fmt_ci(start_ci)} & "
            f"{_fmt_ci(target_week_ci)} & {_fmt_ci(target_ci)} & "
            f"{_fmt_int(crd_points)} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  }",
            (
                r"  \\[2pt] {\footnotesize HF improvement is compact minus "
                r"higher-fidelity RMSE, so positive values favour the "
                r"higher-fidelity propagation. Start-epoch intervals preserve "
                r"the original finite fixed-start readout; target-week "
                r"and target-cluster intervals resample scored target-week or "
                r"target clusters. The all-target and non-LAGEOS target-cluster "
                r"intervals include zero, while the held-out test stratum "
                r"remains positive under target-cluster sensitivity. All "
                r"intervals are finite-probe sensitivity summaries, not "
                r"target-population or operational uncertainty; CRD coverage "
                r"is not used to claim orbit determination.}"
            ),
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-table", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build_result(args.out_dir, refresh=args.refresh)
    output_json = args.out_dir / "public_sp3_multi_target_breadth_probe.json"
    dump_json(result, output_json)
    if not args.no_table:
        write_table(result, args.table)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_json": str(output_json),
                "table": None if args.no_table else str(args.table),
                "headline_readout": result["headline_readout"],
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
