#!/usr/bin/env python
"""Corpus-level public ILRS/SP3 correction and provenance audit.

This is a narrow companion to the existing real-SLR/SP3 probes.  It does not
run another orbit filter and does not introduce a new validation method.
Instead, it reuses the archived forty-arc public LAGEOS CRD/SP3 corpus from
``results/real_slr_sp3_hifi`` and the existing SLR/EOP correction utilities to
quantify:

* CRD/SP3 input coverage and checksums over the multi-week corpus;
* availability of CRD meteorology and transmit wavelength metadata;
* magnitudes of the already implemented tropospheric, centre-of-mass, and
  relativistic one-way range corrections;
* station/satellite frame displacement induced by adding IERS EOP terms.

The output is an evidence/provenance artifact only.  It is intended to make
the public-data boundary clearer without upgrading the paper to an operational
SLR/POD validation claim.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - entrypoint dependent
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import gzip
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from gnn_state_estimation.coordinates import rot_z
from gnn_state_estimation.eop import load_eop_series
from gnn_state_estimation.frames import itrf_to_gcrs, itrf_to_gcrs_eop
from gnn_state_estimation.slr import (
    LAGEOS_CENTRE_OF_MASS_OFFSET_M,
    SUPPORTED_STATIONS,
    SlrStation,
    gmst_rad,
    marini_murray_range_correction_m,
    nearest_met_record,
    parse_crd_v2_meteorology,
    parse_crd_v2_normal_points,
    parse_crd_v2_transmit_wavelength_nm,
    shapiro_delay_m,
)
from gnn_state_estimation.sp3 import Sp3Interpolator, parse_sp3
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

DEFAULT_INPUT_DIR = Path("results/real_slr_sp3_hifi")
DEFAULT_EOP_PATH = Path("results/real_slr_sp3_corrected/finals2000A.all.csv")
DEFAULT_OUTPUT_JSON = Path(
    "results/real_slr_sp3_correction_corpus_audit/"
    "real_slr_sp3_correction_corpus_audit.json"
)
DEFAULT_WAVELENGTH_NM = 532.0
ANALYSIS_CENTER = "NSGF (Space Geodesy Facility, ILRS analysis centre)"

SPLIT_WEEKS = {
    "260418": ("train", ("20260413", "20260414", "20260415",
                          "20260416", "20260417")),
    "260425": ("train", ("20260420", "20260421", "20260422",
                          "20260423", "20260424")),
    "260502": ("val", ("20260427", "20260428", "20260429",
                        "20260430", "20260501")),
    "260509": ("test", ("20260504", "20260505", "20260506",
                         "20260507", "20260508")),
}


@dataclass(frozen=True)
class Arc:
    target: str
    sat_key: str
    sp3_sat_id: str
    date: str
    sp3_week: str
    split: str


def corpus_arcs() -> list[Arc]:
    arcs: list[Arc] = []
    for week, (split, days) in SPLIT_WEEKS.items():
        for target, key, sid in (
            ("LAGEOS-1", "lageos1", "L51"),
            ("LAGEOS-2", "lageos2", "L52"),
        ):
            for date in days:
                arcs.append(Arc(target, key, sid, date, week, split))
    return arcs


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sp3_text(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw).decode("utf-8", "replace")
    return raw.decode("utf-8", "replace")


def station_up_unit(station: SlrStation) -> np.ndarray:
    lat = np.deg2rad(station.lat_deg)
    lon = np.deg2rad(station.lon_deg)
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=np.float64,
    )


def elevation_rad(station: SlrStation, sat_ecef_m: np.ndarray) -> float:
    los = np.asarray(sat_ecef_m, dtype=np.float64) - station.ecef_m()
    norm = float(np.linalg.norm(los))
    if norm <= 0.0:
        return np.pi / 2.0
    return float(
        np.arcsin(np.clip(np.dot(station_up_unit(station), los / norm), -1.0, 1.0))
    )


def finite_stats(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    return {
        "count": int(arr.size),
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "p95": round(float(np.percentile(arr, 95.0)), 6),
    }


def load_outcome_context() -> dict:
    paths = {
        "forty_arc_hifi": Path(
            "results/real_slr_sp3_hifi/real_slr_sp3_hifi_validation.json"
        ),
        "ten_arc_full_correction": Path(
            "results/real_slr_sp3_corrected/real_slr_sp3_corrected_validation.json"
        ),
        "ten_arc_gmst_od": Path(
            "results/real_slr_sp3_od/real_slr_sp3_od_validation.json"
        ),
        "four_arc_range_residual": Path(
            "results/real_slr_lageos/real_slr_lageos_validation.json"
        ),
    }
    out: dict[str, dict] = {}
    for name, path in paths.items():
        if not path.exists():
            out[name] = {"status": "missing", "path": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "forty_arc_hifi":
            out[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "num_arcs_completed": payload.get("num_arcs_completed"),
                "controlled_pure_dynamics_all": payload.get(
                    "controlled_pure_dynamics", {}
                ).get("all"),
                "controlled_pure_dynamics_test": payload.get(
                    "controlled_pure_dynamics", {}
                ).get("test"),
                "sparse_slr_test": payload.get(
                    "sparse_slr_operational_realism", {}
                ).get("test"),
                "learned_calibrator_verdict": payload.get(
                    "learned_calibrator", {}
                ).get("verdict"),
            }
        elif name == "ten_arc_full_correction":
            out[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "num_arcs_completed": payload.get("num_arcs_completed"),
                "full_mean_m": payload.get(
                    "head_to_head_vs_committed_gmst_only", {}
                ).get("corrected_full_mean_m"),
                "gmst_only_mean_m": payload.get(
                    "head_to_head_vs_committed_gmst_only", {}
                ).get("gmst_only_reproduction_mean_m"),
                "correction_sensitivity_audit": payload.get(
                    "correction_sensitivity_audit"
                ),
            }
        elif name == "ten_arc_gmst_od":
            out[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "num_arcs_completed": payload.get("num_arcs_completed"),
                "pooled_held_out_position_rmse_m": payload.get(
                    "pooled_held_out_position_rmse_m"
                ),
                "dbar_external_validation": payload.get(
                    "dbar_external_validation"
                ),
            }
        else:
            out[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "num_arcs_completed": payload.get("num_arcs_completed"),
                "num_held_out_total": payload.get("num_held_out_total"),
                "pooled_estimators": payload.get("pooled_estimators"),
                "learned_beats_classical_wls": payload.get(
                    "learned_beats_classical_wls"
                ),
            }
    return out


def audit_arc(arc, input_dir: Path, eop) -> tuple[dict, dict[str, list[float]]]:
    crd_path = input_dir / f"{arc.sat_key}_{arc.date}.np2"
    sp3_path = input_dir / f"nsgf.orb.{arc.sat_key}.{arc.sp3_week}.v80.sp3.gz"
    arc_id = f"{arc.target} {arc.date}"
    if not crd_path.exists() or not sp3_path.exists():
        return (
            {
                "arc_id": arc_id,
                "target": arc.target,
                "date": arc.date,
                "sp3_week": arc.sp3_week,
                "split": arc.split,
                "status": "missing_input",
                "missing": [
                    str(p)
                    for p in (crd_path, sp3_path)
                    if not p.exists()
                ],
            },
            {},
        )

    crd_text = crd_path.read_text(encoding="utf-8", errors="replace")
    points = parse_crd_v2_normal_points(crd_text)
    met_recs = parse_crd_v2_meteorology(crd_text)
    wavelength_nm = parse_crd_v2_transmit_wavelength_nm(crd_text)
    wavelength_um = (wavelength_nm or DEFAULT_WAVELENGTH_NM) / 1000.0

    eph = parse_sp3(read_sp3_text(sp3_path), arc.sp3_sat_id)
    interp = Sp3Interpolator(eph, order=9)
    points = [p for p in points if eph.covers(p.epoch_unix, margin_s=-30.0)]

    if len(points) == 0:
        return (
            {
                "arc_id": arc_id,
                "target": arc.target,
                "date": arc.date,
                "sp3_week": arc.sp3_week,
                "split": arc.split,
                "status": "no_covered_observations",
                "crd_sha256": sha256_file(crd_path),
                "sp3_sha256": sha256_file(sp3_path),
            },
            {},
        )

    values: dict[str, list[float]] = {
        "troposphere_m": [],
        "centre_of_mass_m": [],
        "relativity_m": [],
        "total_range_correction_m": [],
        "elevation_deg": [],
        "nearest_met_delta_s": [],
        "station_full_vs_no_eop_m": [],
        "station_full_vs_gmst_m": [],
        "satellite_full_vs_no_eop_m": [],
        "satellite_full_vs_gmst_m": [],
    }
    station_counts = Counter(p.station_code for p in points)
    met_missing = 0
    eop_outside = 0

    for p in points:
        station = SUPPORTED_STATIONS[p.cdp_id]
        sat_ecef = interp.position_ecef_m(p.epoch_unix)
        el = elevation_rad(station, sat_ecef)
        met = nearest_met_record(met_recs, p.epoch_unix) if met_recs else None
        if met is None:
            tropo = np.nan
            met_missing += 1
        else:
            tropo = marini_murray_range_correction_m(
                el,
                met.pressure_hpa,
                met.temperature_k,
                met.humidity_pct,
                np.deg2rad(station.lat_deg),
                station.alt_m,
                wavelength_um,
            )
            values["nearest_met_delta_s"].append(abs(met.epoch_unix - p.epoch_unix))
        rel = shapiro_delay_m(station.ecef_m(), sat_ecef)
        total = tropo + LAGEOS_CENTRE_OF_MASS_OFFSET_M + rel

        values["troposphere_m"].append(tropo)
        values["centre_of_mass_m"].append(LAGEOS_CENTRE_OF_MASS_OFFSET_M)
        values["relativity_m"].append(rel)
        values["total_range_correction_m"].append(total)
        values["elevation_deg"].append(np.rad2deg(el))

        if not eop.covers(p.epoch_unix):
            eop_outside += 1
        xp, yp = eop.polar_motion_rad(p.epoch_unix)
        dut1 = eop.ut1_minus_utc_s(p.epoch_unix)
        st_full = itrf_to_gcrs_eop(station.ecef_m(), p.epoch_unix, xp, yp, dut1)
        st_no_eop = itrf_to_gcrs(station.ecef_m(), p.epoch_unix)
        st_gmst = rot_z(gmst_rad(p.epoch_unix)).T @ station.ecef_m()
        sat_full = itrf_to_gcrs_eop(sat_ecef, p.epoch_unix, xp, yp, dut1)
        sat_no_eop = itrf_to_gcrs(sat_ecef, p.epoch_unix)
        sat_gmst = rot_z(gmst_rad(p.epoch_unix)).T @ sat_ecef
        values["station_full_vs_no_eop_m"].append(float(np.linalg.norm(st_full - st_no_eop)))
        values["station_full_vs_gmst_m"].append(float(np.linalg.norm(st_full - st_gmst)))
        values["satellite_full_vs_no_eop_m"].append(float(np.linalg.norm(sat_full - sat_no_eop)))
        values["satellite_full_vs_gmst_m"].append(float(np.linalg.norm(sat_full - sat_gmst)))

    arc_summary = {
        "arc_id": arc_id,
        "target": arc.target,
        "date": arc.date,
        "sp3_week": arc.sp3_week,
        "split": arc.split,
        "status": "completed",
        "num_observations": len(points),
        "arc_span_hours": round(
            float((points[-1].epoch_unix - points[0].epoch_unix) / 3600.0), 6
        ),
        "station_counts": dict(sorted(station_counts.items())),
        "distinct_stations": len(station_counts),
        "num_met_records": len(met_recs),
        "met_missing_observations": met_missing,
        "eop_outside_range_observations": eop_outside,
        "wavelength_nm": wavelength_nm or DEFAULT_WAVELENGTH_NM,
        "wavelength_source": "crd_c0_record" if wavelength_nm else "default_532_nm",
        "sp3_analysis_center": ANALYSIS_CENTER,
        "crd": {
            "archived_input_id": crd_path.name,
            "sha256": sha256_file(crd_path),
            "bytes": crd_path.stat().st_size,
        },
        "sp3": {
            "archived_input_id": sp3_path.name,
            "sha256": sha256_file(sp3_path),
            "bytes": sp3_path.stat().st_size,
            "n_epochs": int(eph.epochs_unix.size),
            "coordinate_frame": eph.coordinate_frame,
            "time_system": eph.time_system,
        },
        "correction_component_stats": {
            key: finite_stats(vals)
            for key, vals in values.items()
            if key
            in {
                "troposphere_m",
                "centre_of_mass_m",
                "relativity_m",
                "total_range_correction_m",
                "elevation_deg",
                "nearest_met_delta_s",
            }
        },
        "frame_displacement_stats": {
            key: finite_stats(vals)
            for key, vals in values.items()
            if key.endswith("_m")
            and key
            in {
                "station_full_vs_no_eop_m",
                "station_full_vs_gmst_m",
                "satellite_full_vs_no_eop_m",
                "satellite_full_vs_gmst_m",
            }
        },
    }
    return arc_summary, values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--eop-path", type=Path, default=DEFAULT_EOP_PATH)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.eop_path.exists():
        raise FileNotFoundError(
            f"EOP file not found: {args.eop_path}. Run the existing "
            "full-correction probe first, or pass --eop-path."
        )
    eop = load_eop_series(args.eop_path)

    arc_summaries = []
    pooled_values: dict[str, list[float]] = {}
    for arc in corpus_arcs():
        summary, values = audit_arc(arc, args.input_dir, eop)
        arc_summaries.append(summary)
        for key, vals in values.items():
            pooled_values.setdefault(key, []).extend(vals)

    completed = [a for a in arc_summaries if a.get("status") == "completed"]
    total_obs = int(sum(a.get("num_observations", 0) for a in completed))
    met_missing = int(sum(a.get("met_missing_observations", 0) for a in completed))
    eop_outside = int(
        sum(a.get("eop_outside_range_observations", 0) for a in completed)
    )
    input_digests = []
    for a in completed:
        for kind in ("crd", "sp3"):
            block = a.get(kind, {})
            input_digests.append(
                {
                    "arc_id": a["arc_id"],
                    "kind": kind,
                    "archived_input_id": block.get("archived_input_id"),
                    "sha256": block.get("sha256"),
                    "bytes": block.get("bytes"),
                }
            )

    result = {
        "schema_version": "real_slr_sp3_correction_corpus_audit_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed" if completed else "no_completed_arcs",
        "scope": (
            "Forty-arc public LAGEOS CRD/SP3 corpus-level correction and "
            "provenance audit. No filter estimates are recomputed."
        ),
        "input_dir": str(args.input_dir),
        "eop_series": {
            "path": str(args.eop_path),
            "sha256": sha256_file(args.eop_path),
            "bytes": args.eop_path.stat().st_size,
            "n_rows": int(eop.mjd.size),
            "source": eop.source,
        },
        "sp3_analysis_center": ANALYSIS_CENTER,
        "num_arcs": len(arc_summaries),
        "num_arcs_completed": len(completed),
        "targets": sorted({a["target"] for a in completed}),
        "sp3_weeks": sorted({a["sp3_week"] for a in completed}),
        "splits": dict(Counter(a["split"] for a in completed)),
        "num_observations_total": total_obs,
        "met_coverage": {
            "num_met_records_total": int(
                sum(a.get("num_met_records", 0) for a in completed)
            ),
            "met_missing_observations": met_missing,
            "fraction_observations_with_nearest_met": (
                round(float((total_obs - met_missing) / total_obs), 6)
                if total_obs
                else None
            ),
        },
        "eop_coverage": {
            "eop_outside_range_observations": eop_outside,
            "fraction_observations_inside_eop_range": (
                round(float((total_obs - eop_outside) / total_obs), 6)
                if total_obs
                else None
            ),
        },
        "pooled_correction_component_stats": {
            key: finite_stats(vals)
            for key, vals in pooled_values.items()
            if key
            in {
                "troposphere_m",
                "centre_of_mass_m",
                "relativity_m",
                "total_range_correction_m",
                "elevation_deg",
                "nearest_met_delta_s",
            }
        },
        "pooled_frame_displacement_stats": {
            key: finite_stats(vals)
            for key, vals in pooled_values.items()
            if key
            in {
                "station_full_vs_no_eop_m",
                "station_full_vs_gmst_m",
                "satellite_full_vs_no_eop_m",
                "satellite_full_vs_gmst_m",
            }
        },
        "arcs": arc_summaries,
        "input_digests": input_digests,
        "existing_outcome_context": load_outcome_context(),
        "claim_boundary": {
            "is_operational_validation": False,
            "is_central_learned_vs_classical_validation": False,
            "appropriate_use": (
                "Use as public-data provenance and measurement-reduction "
                "coverage evidence only; do not use as operational POD, "
                "centimetre SLR validation, or external validation of the "
                "central simulator-bound learned-vs-classical conclusion."
            ),
            "smallest_defensible_paper_use": (
                "A supplement or evidence-index row can state that the "
                "existing forty-arc public corpus has complete archived "
                "CRD/SP3/EOP coverage and quantified correction/frame "
                "magnitudes, while retaining the existing bounded-probe "
                "language."
            ),
        },
    }
    dump_json(result, args.output_json)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_json": str(args.output_json),
                "num_arcs_completed": result["num_arcs_completed"],
                "num_observations_total": result["num_observations_total"],
                "met_coverage": result["met_coverage"],
                "eop_coverage": result["eop_coverage"],
                "pooled_total_range_correction_m": result[
                    "pooled_correction_component_stats"
                ].get("total_range_correction_m"),
                "station_full_vs_gmst_m": result[
                    "pooled_frame_displacement_stats"
                ].get("station_full_vs_gmst_m"),
                "satellite_full_vs_gmst_m": result[
                    "pooled_frame_displacement_stats"
                ].get("satellite_full_vs_gmst_m"),
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
