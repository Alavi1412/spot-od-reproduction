#!/usr/bin/env python
"""Precise-reference real-data OD slice + externally-defined DBAR validation.

This script scores a self-contained range-only EKF / fixed-noise UKF /
innovation-adaptive UKF on *real* public ILRS satellite-laser-ranging (SLR)
normal points for the LAGEOS geodetic pair, with held-out state error scored
against an *independent* ILRS analysis-centre SP3-c precise orbit product (the
external reference).  It serves two reviewer-binding purposes:

1. **A precise-reference real-data OD slice (M1/M2).**  Unlike the SGP4-mean-
   element benchmark and the range-residual-only SLR probe, held-out error here
   is a state error against a genuine independent precise orbit product.  It is
   bounded fidelity --- approximate GMST-only Earth rotation (common-mode
   between the station geometry and the reference, so it largely cancels),
   compact two-body+J2 dynamics, no precise SLR reduction --- and is reported
   as such, not as centimetre operational OD.

2. **An externally defined DBAR validation (Fatal-Flaw-1).**  The DBAR
   indicator is computed from the real adaptive-vs-fixed filter innovation
   stream (its prediction), but the counterproductivity *outcome* it is scored
   against is defined **externally** from held-out state error versus the
   independent SP3 reference --- not from the DBAR statistic and not from a
   project-simulator self-referential label.  An arc is counterproductive iff
   the adaptive UKF held-out SP3 position RMSE exceeds the fixed-noise UKF's by
   more than the predeclared 5 % margin.

Public inputs are archived on first fetch and reused offline thereafter, so the
slice regenerates without the network; ``--refresh`` re-downloads them.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - entrypoint dependent
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnn_state_estimation.slr import (
    SUPPORTED_STATIONS,
    parse_crd_v2_normal_points,
    station_pseudo_inertial_m,
)
from gnn_state_estimation.sp3 import (
    RangeObs,
    Sp3Interpolator,
    held_out_position_rmse,
    mean_r_eff_scale,
    median_nis_r,
    parse_sp3,
    propagate_compact,
    run_range_aukf,
    run_range_ekf,
    run_range_ukf_fixed,
)
from gnn_state_estimation.utils.classification_stats import (
    binary_classification_report,
)
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

EDC_CRD_URL = (
    "https://edc.dgfi.tum.de/pub/slr/data/npt_crd_v2/{sat_key}/{year}/"
    "{sat_key}_{date}.np2"
)
# Independent ILRS analysis-centre SP3-c weekly precise orbit product (SGF).
EDC_SP3_URL = (
    "https://edc.dgfi.tum.de/pub/slr/products/orbits/{sat_key}/{week}/"
    "nsgf.orb.{sat_key}.{week}.v80.sp3.gz"
)
ANALYSIS_CENTER = "NSGF (Space Geodesy Facility, ILRS analysis centre)"

# Predeclared DBAR rule -- identical to the in-simulator DBAR; only the
# outcome label changes (here it is external precise-SP3 state error).
TAU_R = 1.5
TAU_RHO = 1.5
MATERIALITY_MARGIN = 0.05

# Predeclared filter settings (fixed a priori, not tuned to outcomes).
RANGE_STD_M = 20.0          # assumed nominal measurement-noise std
ACCEL_PSD = 1.0e-12         # white-noise-acceleration process spectral density
INIT_POS_STD_M = 50.0
INIT_VEL_STD_MPS = 0.5
TRAIN_FRAC = 0.6
MAX_STEP_S = 30.0


@dataclass(frozen=True)
class Arc:
    target: str
    sat_key: str
    sp3_sat_id: str
    date: str           # YYYYMMDD (CRD daily file / UTC day)
    sp3_week: str        # YYMMDD weekly SP3 product id covering the day


# SP3 weekly product 260509 covers 2026-05-03 00:00 -> 2026-05-10 00:04 UTC.
# Daily CRD files inside that window for both geodetic satellites.
_SP3_WEEK = "260509"
DEFAULT_ARCS: tuple[Arc, ...] = tuple(
    Arc(t, k, sid, f"202605{dd:02d}", _SP3_WEEK)
    for (t, k, sid) in (
        ("LAGEOS-1", "lageos1", "L51"),
        ("LAGEOS-2", "lageos2", "L52"),
    )
    for dd in (5, 6, 7, 8, 9)
)


def _python_download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as response:
        output_path.write_bytes(response.read())


def _powershell_download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "$ProgressPreference='SilentlyContinue'; "
        f"Invoke-WebRequest -UseBasicParsing -Uri '{url}' "
        f"-OutFile '{output_path.resolve()}'"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
    )


def materialize(url: str, output_path: Path, *, refresh: bool) -> str:
    """Reuse an archived public input if present; else fetch it publicly."""
    if not refresh and output_path.exists():
        return "archived_input"
    try:
        _python_download(url, output_path)
        return "public_archive_download"
    except (urllib.error.URLError, OSError):
        try:
            _powershell_download(url, output_path)
            return "public_archive_download"
        except subprocess.CalledProcessError as exc:
            if output_path.exists():
                return "archived_input_fallback"
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_sp3_text(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        import gzip

        return gzip.decompress(raw).decode("utf-8", "replace")
    return raw.decode("utf-8", "replace")


def run_arc(arc: Arc, out_dir: Path, args: argparse.Namespace) -> dict:
    year = arc.date[:4]
    crd_url = EDC_CRD_URL.format(
        sat_key=arc.sat_key, year=year, date=arc.date
    )
    sp3_url = EDC_SP3_URL.format(sat_key=arc.sat_key, week=arc.sp3_week)
    crd_path = out_dir / f"{arc.sat_key}_{arc.date}.np2"
    sp3_path = out_dir / f"nsgf.orb.{arc.sat_key}.{arc.sp3_week}.v80.sp3.gz"

    crd_method = materialize(crd_url, crd_path, refresh=args.refresh)
    sp3_method = materialize(sp3_url, sp3_path, refresh=args.refresh)

    points = parse_crd_v2_normal_points(
        crd_path.read_text(encoding="utf-8", errors="replace")
    )
    eph = parse_sp3(_read_sp3_text(sp3_path), arc.sp3_sat_id)
    interp = Sp3Interpolator(eph, order=9)

    # Keep only normal points that fall inside the SP3 coverage (with a small
    # interpolation margin) so every scored epoch has a precise reference.
    margin = 30.0
    points = [
        p
        for p in points
        if eph.covers(p.epoch_unix, margin_s=-margin)
    ]

    arc_id = f"{arc.target} {arc.date}"
    provenance = {
        "arc_id": arc_id,
        "target": arc.target,
        "date": arc.date,
        "sp3_week": arc.sp3_week,
        "sp3_analysis_center": ANALYSIS_CENTER,
        "sp3_satellite_id": arc.sp3_sat_id,
        "sp3_time_system": eph.time_system,
        "sp3_frame": eph.coordinate_frame,
        "crd": {
            "url": crd_url,
            "archived_input_id": crd_path.name,
            "input_source": crd_method,
            "sha256": sha256_file(crd_path),
            "bytes": crd_path.stat().st_size,
        },
        "sp3": {
            "url": sp3_url,
            "archived_input_id": sp3_path.name,
            "input_source": sp3_method,
            "sha256": sha256_file(sp3_path),
            "bytes": sp3_path.stat().st_size,
            "n_epochs": int(eph.epochs_unix.size),
        },
    }

    if len(points) < 10:
        return {
            **provenance,
            "status": "insufficient_observations",
            "num_observations": len(points),
        }

    epochs = np.array([p.epoch_unix for p in points], dtype=np.float64)
    ranges = np.array([p.range_m for p in points], dtype=np.float64)
    station_pi = np.vstack(
        [
            station_pseudo_inertial_m(
                SUPPORTED_STATIONS[p.cdp_id], p.epoch_unix
            )
            for p in points
        ]
    )
    obs_all = [
        RangeObs(epoch_unix=float(epochs[i]),
                 station_pi_m=station_pi[i],
                 range_m=float(ranges[i]))
        for i in range(len(points))
    ]

    n = len(obs_all)
    n_fit = max(6, int(np.floor(TRAIN_FRAC * n)))
    n_fit = min(n_fit, n - 3)
    fit_obs = obs_all[:n_fit]
    held_epochs = epochs[n_fit:]
    fit_last_epoch = fit_obs[-1].epoch_unix

    # Precise SP3 initial condition at the first fit epoch, in the shared
    # pseudo-inertial frame.
    t0 = fit_obs[0].epoch_unix
    x0 = interp.state_pseudo_inertial_m(t0)
    p0 = np.diag(
        np.array(
            [INIT_POS_STD_M**2] * 3 + [INIT_VEL_STD_MPS**2] * 3,
            dtype=np.float64,
        )
    )

    ekf = run_range_ekf(fit_obs, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    ukf = run_range_ukf_fixed(
        fit_obs, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S
    )
    aukf = run_range_aukf(
        fit_obs, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S
    )

    held = {
        "EKF": held_out_position_rmse(
            ekf["state"], fit_last_epoch, held_epochs, interp, MAX_STEP_S
        ),
        "UKF (fixed-noise)": held_out_position_rmse(
            ukf["state"], fit_last_epoch, held_epochs, interp, MAX_STEP_S
        ),
        "AUKF (adaptive)": held_out_position_rmse(
            aukf["state"], fit_last_epoch, held_epochs, interp, MAX_STEP_S
        ),
    }
    # Pure compact-model propagation from the precise SP3 IC (no measurement
    # updates): the external dynamics-model-bias floor for this arc.
    sp3_ic_state = propagate_compact(x0, fit_last_epoch - t0, MAX_STEP_S)
    held["SP3-IC propagation"] = held_out_position_rmse(
        sp3_ic_state, fit_last_epoch, held_epochs, interp, MAX_STEP_S
    )

    rmse = {k: v.get("rms_m", float("nan")) for k, v in held.items()}
    finite = {k: v for k, v in rmse.items() if np.isfinite(v)}
    best = min(finite, key=finite.get) if finite else None

    med_aukf = median_nis_r(aukf["records"])
    med_ukf = median_nis_r(ukf["records"])
    rho_nis = med_aukf / med_ukf if med_ukf else float("nan")
    r_eff = mean_r_eff_scale(aukf["records"])
    fired = bool(
        np.isfinite(r_eff)
        and np.isfinite(rho_nis)
        and (r_eff > TAU_R)
        and (rho_nis >= TAU_RHO)
    )

    ukf_rms = rmse["UKF (fixed-noise)"]
    aukf_rms = rmse["AUKF (adaptive)"]
    external_outcome_available = bool(
        np.isfinite(ukf_rms) and np.isfinite(aukf_rms) and ukf_rms > 0.0
    )
    aukf_vs_twin_pct = (
        100.0 * (aukf_rms - ukf_rms) / ukf_rms
        if external_outcome_available
        else float("nan")
    )
    adaptation_counterproductive_external = bool(
        external_outcome_available
        and aukf_rms > ukf_rms * (1.0 + MATERIALITY_MARGIN)
    )
    dbar_correct = bool(
        external_outcome_available
        and (fired == adaptation_counterproductive_external)
    )

    return {
        **provenance,
        "status": "completed",
        "num_observations": n,
        "num_fit": int(n_fit),
        "num_held_out": int(n - n_fit),
        "distinct_stations": len({p.station_code for p in points}),
        "station_counts": {
            s: int(sum(1 for p in points if p.station_code == s))
            for s in sorted({p.station_code for p in points})
        },
        "arc_span_hours": float((epochs[-1] - epochs[0]) / 3600.0),
        "held_out_position_rmse_m": {
            k: (round(v, 2) if np.isfinite(v) else None)
            for k, v in rmse.items()
        },
        "held_out_detail": held,
        "best_held_out_estimator": best,
        "dbar": {
            "median_nis_r_aukf": round(med_aukf, 4),
            "median_nis_r_ukf": round(med_ukf, 4),
            "rho_nis": round(rho_nis, 4) if np.isfinite(rho_nis) else None,
            "r_eff_scale": round(r_eff, 4) if np.isfinite(r_eff) else None,
            "dbar_fired": fired,
            "aukf_vs_fixed_twin_pct": (
                round(aukf_vs_twin_pct, 2)
                if np.isfinite(aukf_vs_twin_pct)
                else None
            ),
            "external_outcome_available": external_outcome_available,
            "adaptation_counterproductive_external": (
                adaptation_counterproductive_external
            ),
            "dbar_correct_external": dbar_correct,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=str, default="results/real_slr_sp3_od"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the public CRD and SP3 inputs from their archive "
        "URLs. By default existing local input files are reused without "
        "contacting the network so the slice is reproducible offline.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-derive only the external-DBAR classification summary "
        "(majority-class baseline, Wilson confidence intervals, power) from "
        "the per-arc records already stored in the artifact, without "
        "re-running the filters. The OD pooled RMSE table is preserved "
        "byte-for-byte; the per-arc external outcomes are deterministic.",
    )
    return parser


def _dbar_summary_from_arcs(completed: list[dict]) -> dict:
    """External-DBAR confusion + hardened classification report from arcs."""
    ext = [
        a["dbar"]
        for a in completed
        if a.get("dbar", {}).get("external_outcome_available")
    ]
    tp = sum(
        1
        for d in ext
        if d["dbar_fired"] and d["adaptation_counterproductive_external"]
    )
    tn = sum(
        1
        for d in ext
        if not d["dbar_fired"]
        and not d["adaptation_counterproductive_external"]
    )
    fp = sum(
        1
        for d in ext
        if d["dbar_fired"]
        and not d["adaptation_counterproductive_external"]
    )
    fn = sum(
        1
        for d in ext
        if not d["dbar_fired"]
        and d["adaptation_counterproductive_external"]
    )
    n_ext = len(ext)
    n_correct = tp + tn
    n_pos = tp + fn
    n_neg = tn + fp
    report = binary_classification_report(tp, tn, fp, fn)
    return {
        "n_arcs_scored": n_ext,
        "n_correct": n_correct,
        "external_label_definition": (
            "adaptation counterproductive iff the adaptive UKF held-out SP3 "
            "position RMSE exceeds the fixed-noise UKF held-out SP3 position "
            "RMSE by more than the predeclared 5% margin (external precise "
            "reference; not the DBAR statistic, not a simulator label)"
        ),
        "classification_accuracy": (
            round(n_correct / n_ext, 4) if n_ext else None
        ),
        "confusion": {
            "true_fire": tp,
            "true_no_fire": tn,
            "false_fire": fp,
            "false_no_fire": fn,
        },
        "sensitivity": round(tp / n_pos, 4) if n_pos else None,
        "specificity": round(tn / n_neg, 4) if n_neg else None,
        "n_counterproductive_arcs": n_pos,
        "n_non_counterproductive_arcs": n_neg,
        "classification_report": report,
        "no_information_baseline": report["no_information"],
        "incremental_accuracy_over_majority_class": report[
            "no_information"
        ]["accuracy_minus_majority"],
        "beats_trivial_majority_classifier": report["no_information"][
            "beats_majority"
        ],
    }


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reprocess:
        path = out_dir / "real_slr_sp3_od_validation.json"
        prev = json.loads(path.read_text(encoding="utf-8"))
        completed = [
            a for a in prev["arcs"] if a.get("status") == "completed"
        ]
        prev["dbar_external_validation"] = _dbar_summary_from_arcs(completed)
        prev["generated_utc"] = utc_now_iso()
        dump_json(prev, path)
        print(
            json.dumps(
                {
                    "status": "reprocessed",
                    "pooled_held_out_position_rmse_m": prev[
                        "pooled_held_out_position_rmse_m"
                    ],
                    "dbar_external_validation": prev[
                        "dbar_external_validation"
                    ],
                },
                indent=2,
            )
        )
        return 0

    caveats = (
        "Bounded precise-reference real-data OD slice. Held-out error is a "
        "state error against an independent ILRS analysis-centre SP3-c "
        "precise orbit product, but the OD itself is bounded fidelity: "
        "approximate GMST-only Earth rotation (no polar motion/nutation; "
        "UT1~UTC) applied identically to the station geometry and the SP3 "
        "reference so it is common-mode and largely cancels, compact "
        "two-body+J2 dynamics with no precise SLR reduction (relativistic, "
        "tropospheric, centre-of-mass, solid-Earth-tide), and a precise SP3 "
        "initial condition. Absolute magnitudes are a model-mismatch stress, "
        "not centimetre operational OD or flight readiness. The DBAR "
        "prediction is the real adaptive-vs-fixed filter innovation ratio; "
        "the counterproductivity outcome it is scored against is defined "
        "externally from held-out state error versus the independent SP3 "
        "reference, not from the DBAR statistic and not from a project "
        "simulator."
    )

    arcs = list(DEFAULT_ARCS)
    arc_blocks = [run_arc(a, out_dir, args) for a in arcs]
    completed = [a for a in arc_blocks if a.get("status") == "completed"]

    estimators = [
        "EKF",
        "UKF (fixed-noise)",
        "AUKF (adaptive)",
        "SP3-IC propagation",
    ]
    pooled = {}
    arc_best_count = {e: 0 for e in estimators}
    for a in completed:
        b = a.get("best_held_out_estimator")
        if b in arc_best_count:
            arc_best_count[b] += 1
    for e in estimators:
        vals = [
            a["held_out_detail"][e]["rms_m"]
            for a in completed
            if e in a["held_out_detail"]
            and np.isfinite(a["held_out_detail"][e].get("rms_m", np.nan))
        ]
        arr = np.asarray(vals, dtype=np.float64)
        pooled[e] = {
            "n_arcs": int(arr.size),
            "mean_arc_rms_m": (
                round(float(np.mean(arr)), 2) if arr.size else None
            ),
            "median_arc_rms_m": (
                round(float(np.median(arr)), 2) if arr.size else None
            ),
            "arcs_best_of": arc_best_count[e],
        }

    # External DBAR confusion + hardened classification report.
    dbar_summary = _dbar_summary_from_arcs(completed)

    input_digests = []
    for a in arc_blocks:
        for kind in ("crd", "sp3"):
            blk = a.get(kind)
            if isinstance(blk, dict) and "sha256" in blk:
                input_digests.append(
                    {
                        "arc_id": a.get("arc_id"),
                        "kind": kind,
                        "archived_input_id": blk.get("archived_input_id"),
                        "sha256": blk.get("sha256"),
                        "bytes": blk.get("bytes"),
                        "url": blk.get("url"),
                    }
                )

    serial_arcs = []
    for a in arc_blocks:
        a2 = {k: v for k, v in a.items() if k != "held_out_detail"}
        serial_arcs.append(a2)

    result = {
        "schema_version": "real_slr_sp3_od_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed" if completed else "insufficient_observations",
        "targets": sorted({a["target"] for a in completed}),
        "num_arcs": len(arcs),
        "num_arcs_completed": len(completed),
        "sp3_analysis_center": ANALYSIS_CENTER,
        "sp3_week_product": _SP3_WEEK,
        "fixed_station_subset": sorted(
            {s.code for s in SUPPORTED_STATIONS.values()}
        ),
        "predeclared": {
            "tau_r_eff": TAU_R,
            "tau_rho_nis": TAU_RHO,
            "materiality_margin": MATERIALITY_MARGIN,
            "range_std_m": RANGE_STD_M,
            "accel_psd": ACCEL_PSD,
            "train_frac": TRAIN_FRAC,
            "max_step_s": MAX_STEP_S,
        },
        "arcs": serial_arcs,
        "pooled_held_out_position_rmse_m": pooled,
        "dbar_external_validation": dbar_summary,
        "input_digests": input_digests,
        "caveats": caveats,
    }
    dump_json(result, out_dir / "real_slr_sp3_od_validation.json")
    print(
        json.dumps(
            {
                "status": result["status"],
                "num_arcs_completed": result["num_arcs_completed"],
                "pooled_held_out_position_rmse_m": pooled,
                "dbar_external_validation": dbar_summary,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
