#!/usr/bin/env python
"""Full-correction precise-reference real-data sanity probe.

This slice answers the loop-30 review's operational-real-data concern head-on.
It re-runs the bounded ten-arc real LAGEOS laser-range precise-reference probe,
but replaces the approximate GMST-only Earth rotation with a full IAU-76/80
reduction that **ingests the real public IERS Earth-orientation series** (polar
motion and UT1-UTC) and applies the standard precise-SLR-reduction
corrections the review listed as missing --- tropospheric refraction
(Marini--Murray, from the real CRD surface meteorology and transmit
wavelength), the satellite centre-of-mass offset, and the relativistic
(Shapiro) range delay --- and it adds a rigorous correction-sensitivity audit
that turns each correction off and reports the change in held-out state error
against the same independent ILRS analysis-centre SP3 precise orbit product.

It is additive: the committed ``real_slr_sp3_od`` slice, its JSON, table, and
locked phrases are untouched.  The dynamics remain the compact two-body+J2
model so this isolates the Earth-orientation / SLR-reduction effect rather than
introducing a force-model change, and it is reported as a bounded sanity /
provenance probe, not a centimetre operational or flight-grade OD claim.
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
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnn_state_estimation.coordinates import rot_z
from gnn_state_estimation.eop import EopSeries, load_eop_series
from gnn_state_estimation.frames import itrf_to_gcrs, itrf_to_gcrs_eop
from gnn_state_estimation.slr import (
    SUPPORTED_STATIONS,
    LAGEOS_CENTRE_OF_MASS_OFFSET_M,
    SlrStation,
    gmst_rad,
    marini_murray_range_correction_m,
    nearest_met_record,
    parse_crd_v2_meteorology,
    parse_crd_v2_normal_points,
    parse_crd_v2_transmit_wavelength_nm,
    shapiro_delay_m,
)
from gnn_state_estimation.sp3 import (
    RangeObs,
    Sp3Interpolator,
    parse_sp3,
    propagate_compact,
    run_range_aukf,
    run_range_ekf,
    run_range_ukf_fixed,
)
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

# The archived CRD/SP3 inputs are reused from the committed bounded slice so
# this probe shares byte-identical real measurements and adds no new heavy
# downloads; only the small public IERS EOP series is fetched here.
SRC_INPUT_DIR = Path("results/real_slr_sp3_od")
EOP_URL = "https://datacenter.iers.org/data/csv/finals2000A.all.csv"
EOP_NAME = "finals2000A.all.csv"
ANALYSIS_CENTER = "NSGF (Space Geodesy Facility, ILRS analysis centre)"

# Predeclared filter settings -- identical to the committed bounded slice so
# the only changes are the Earth-orientation frame and the SLR reduction.
RANGE_STD_M = 20.0
ACCEL_PSD = 1.0e-12
INIT_POS_STD_M = 50.0
INIT_VEL_STD_MPS = 0.5
TRAIN_FRAC = 0.6
MAX_STEP_S = 30.0
DEFAULT_WAVELENGTH_NM = 532.0

# Correction-sensitivity audit configurations.  ``full`` applies the complete
# full correction stack; each ablation removes exactly one ingredient;
# ``gmst_only`` reproduces the committed bounded construction (GMST-only frame,
# no SLR reduction) and is the head-to-head baseline / internal cross-check.
AUDIT_CONFIGS = (
    "full",
    "no_eop",
    "no_troposphere",
    "no_centre_of_mass",
    "no_relativity",
    "gmst_only",
)


@dataclass(frozen=True)
class Arc:
    target: str
    sat_key: str
    sp3_sat_id: str
    date: str
    sp3_week: str


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


def materialize_eop(out_dir: Path, *, refresh: bool) -> tuple[Path, str]:
    path = out_dir / EOP_NAME
    if not refresh and path.exists():
        return path, "archived_input"
    try:
        _python_download(EOP_URL, path)
        return path, "public_archive_download"
    except (urllib.error.URLError, OSError) as exc:
        if path.exists():
            return path, "archived_input_fallback"
        raise RuntimeError(f"Failed to download {EOP_URL}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_sp3_text(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        import gzip

        return gzip.decompress(raw).decode("utf-8", "replace")
    return raw.decode("utf-8", "replace")


def _station_up_unit(station: SlrStation) -> np.ndarray:
    """Geodetic local-up unit vector of the station in ECEF."""
    lat = np.deg2rad(station.lat_deg)
    lon = np.deg2rad(station.lon_deg)
    return np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        dtype=np.float64,
    )


def _elevation_rad(
    station: SlrStation, sat_ecef_m: np.ndarray
) -> float:
    up = _station_up_unit(station)
    los = np.asarray(sat_ecef_m, dtype=np.float64) - station.ecef_m()
    n = float(np.linalg.norm(los))
    if n <= 0.0:
        return np.pi / 2.0
    return float(np.arcsin(np.clip(np.dot(up, los / n), -1.0, 1.0)))


def _frame_mapper(config: str, eop: EopSeries):
    """Return an ITRF/ECEF -> inertial position map for an audit config."""
    if config == "gmst_only":
        def m(r_ecef: np.ndarray, epoch_unix: float) -> np.ndarray:
            return rot_z(gmst_rad(epoch_unix)).T @ np.asarray(
                r_ecef, dtype=np.float64
            )

        return m
    if config == "no_eop":
        def m(r_ecef: np.ndarray, epoch_unix: float) -> np.ndarray:
            return itrf_to_gcrs(r_ecef, epoch_unix)

        return m

    def m(r_ecef: np.ndarray, epoch_unix: float) -> np.ndarray:
        xp, yp = eop.polar_motion_rad(epoch_unix)
        dut1 = eop.ut1_minus_utc_s(epoch_unix)
        return itrf_to_gcrs_eop(r_ecef, epoch_unix, xp, yp, dut1)

    return m


def _range_correction_m(
    config: str,
    station: SlrStation,
    sat_ecef_m: np.ndarray,
    epoch_unix: float,
    met,
    wavelength_um: float,
) -> float:
    """Total deterministic correction (m) subtracted from the measured range.

    ``measured = geometric(to CoM) + tropo + Shapiro + CoM-offset``, so the
    geometric station--centre-of-mass vacuum range fed to the filter is the
    measured range minus this sum.  ``gmst_only`` applies no SLR reduction
    (reproducing the committed bounded construction).
    """
    if config == "gmst_only":
        return 0.0
    total = 0.0
    if config != "no_troposphere" and met is not None:
        el = _elevation_rad(station, sat_ecef_m)
        total += marini_murray_range_correction_m(
            el,
            met.pressure_hpa,
            met.temperature_k,
            met.humidity_pct,
            np.deg2rad(station.lat_deg),
            station.alt_m,
            wavelength_um,
        )
    if config != "no_centre_of_mass":
        total += LAGEOS_CENTRE_OF_MASS_OFFSET_M
    if config != "no_relativity":
        total += shapiro_delay_m(station.ecef_m(), np.asarray(sat_ecef_m))
    return float(total)


def _state_inertial(
    interp: Sp3Interpolator, mapper, epoch_unix: float, h_s: float = 1.0
) -> np.ndarray:
    p0 = mapper(interp.position_ecef_m(epoch_unix), epoch_unix)
    pp = mapper(interp.position_ecef_m(epoch_unix + h_s), epoch_unix + h_s)
    pm = mapper(interp.position_ecef_m(epoch_unix - h_s), epoch_unix - h_s)
    v = (pp - pm) / (2.0 * h_s)
    return np.hstack([p0, v]).astype(np.float64)


def _held_out_rmse(
    final_state: np.ndarray,
    fit_last_epoch: float,
    held_epochs: np.ndarray,
    interp: Sp3Interpolator,
    mapper,
) -> dict:
    errs: list[float] = []
    state = np.asarray(final_state, dtype=np.float64).copy()
    t_prev = float(fit_last_epoch)
    for te in np.asarray(held_epochs, dtype=np.float64):
        state = propagate_compact(state, te - t_prev, MAX_STEP_S)
        t_prev = float(te)
        ref = mapper(interp.position_ecef_m(te), te)
        errs.append(float(np.linalg.norm(state[:3] - ref)))
    arr = np.asarray(errs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "rms_m": float("nan")}
    return {
        "count": int(arr.size),
        "rms_m": float(np.sqrt(np.mean(arr**2))),
        "mean_m": float(np.mean(arr)),
        "p95_abs_m": float(np.percentile(arr, 95.0)),
    }


def _score_arc_config(
    config: str,
    points,
    met_recs,
    wavelength_um: float,
    interp: Sp3Interpolator,
    eop: EopSeries,
) -> dict:
    mapper = _frame_mapper(config, eop)
    epochs = np.array([p.epoch_unix for p in points], dtype=np.float64)
    obs = []
    for p in points:
        station = SUPPORTED_STATIONS[p.cdp_id]
        sat_ecef = interp.position_ecef_m(p.epoch_unix)
        met = nearest_met_record(met_recs, p.epoch_unix) if met_recs else None
        corr = _range_correction_m(
            config, station, sat_ecef, p.epoch_unix, met, wavelength_um
        )
        obs.append(
            RangeObs(
                epoch_unix=float(p.epoch_unix),
                station_pi_m=mapper(station.ecef_m(), p.epoch_unix),
                range_m=float(p.range_m) - corr,
            )
        )
    n = len(obs)
    n_fit = max(6, int(np.floor(TRAIN_FRAC * n)))
    n_fit = min(n_fit, n - 3)
    fit_obs = obs[:n_fit]
    held_epochs = epochs[n_fit:]
    fit_last = fit_obs[-1].epoch_unix
    t0 = fit_obs[0].epoch_unix
    x0 = _state_inertial(interp, mapper, t0)
    p0 = np.diag(
        np.array(
            [INIT_POS_STD_M**2] * 3 + [INIT_VEL_STD_MPS**2] * 3,
            dtype=np.float64,
        )
    )
    ekf = run_range_ekf(fit_obs, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    ukf = run_range_ukf_fixed(fit_obs, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    aukf = run_range_aukf(fit_obs, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    sp3_ic = propagate_compact(x0, fit_last - t0, MAX_STEP_S)
    held = {
        "EKF": _held_out_rmse(ekf["state"], fit_last, held_epochs, interp, mapper),
        "UKF (fixed-noise)": _held_out_rmse(
            ukf["state"], fit_last, held_epochs, interp, mapper
        ),
        "AUKF (adaptive)": _held_out_rmse(
            aukf["state"], fit_last, held_epochs, interp, mapper
        ),
        "SP3-IC propagation": _held_out_rmse(
            sp3_ic, fit_last, held_epochs, interp, mapper
        ),
    }
    # Post-fit innovation RMS (filtered range residual over the fit segment);
    # a robust fit-quality proxy that does not require re-propagation.
    innov = np.asarray(
        [r["innovation_m"] for r in ekf["records"]], dtype=np.float64
    )
    innov = innov[np.isfinite(innov)]
    fit_innov_rms = (
        float(np.sqrt(np.mean(innov**2))) if innov.size else float("nan")
    )
    return {
        "config": config,
        "num_observations": n,
        "num_fit": int(n_fit),
        "num_held_out": int(n - n_fit),
        "ekf_fit_innovation_rms_m": (
            round(fit_innov_rms, 3) if np.isfinite(fit_innov_rms) else None
        ),
        "held_out_position_rmse_m": {
            k: (round(v["rms_m"], 2) if np.isfinite(v.get("rms_m", np.nan)) else None)
            for k, v in held.items()
        },
    }


def run_arc(arc: Arc, eop: EopSeries) -> dict:
    crd_path = SRC_INPUT_DIR / f"{arc.sat_key}_{arc.date}.np2"
    sp3_path = SRC_INPUT_DIR / f"nsgf.orb.{arc.sat_key}.{arc.sp3_week}.v80.sp3.gz"
    if not crd_path.exists() or not sp3_path.exists():
        return {"arc_id": f"{arc.target} {arc.date}", "status": "missing_input"}
    crd_text = crd_path.read_text(encoding="utf-8", errors="replace")
    points = parse_crd_v2_normal_points(crd_text)
    met_recs = parse_crd_v2_meteorology(crd_text)
    wl_nm = parse_crd_v2_transmit_wavelength_nm(crd_text) or DEFAULT_WAVELENGTH_NM
    eph = parse_sp3(_read_sp3_text(sp3_path), arc.sp3_sat_id)
    interp = Sp3Interpolator(eph, order=9)
    points = [p for p in points if eph.covers(p.epoch_unix, margin_s=-30.0)]
    arc_id = f"{arc.target} {arc.date}"
    prov = {
        "arc_id": arc_id,
        "target": arc.target,
        "date": arc.date,
        "wavelength_nm": wl_nm,
        "num_met_records": len(met_recs),
        "crd_sha256": sha256_file(crd_path),
        "sp3_sha256": sha256_file(sp3_path),
    }
    if len(points) < 10:
        return {**prov, "status": "insufficient_observations",
                "num_observations": len(points)}
    by_config = {}
    fit_innov = {}
    for cfg in AUDIT_CONFIGS:
        sc = _score_arc_config(
            cfg, points, met_recs, wl_nm / 1000.0, interp, eop
        )
        by_config[cfg] = sc["held_out_position_rmse_m"]
        fit_innov[cfg] = sc["ekf_fit_innovation_rms_m"]
    return {
        **prov,
        "status": "completed",
        "num_observations": len(points),
        "arc_span_hours": float((points[-1].epoch_unix - points[0].epoch_unix) / 3600.0),
        "held_out_position_rmse_m_by_config": by_config,
        "ekf_fit_innovation_rms_m_by_config": fit_innov,
    }


def _pool(arcs: list[dict], cfg: str, est: str) -> dict:
    vals = []
    for a in arcs:
        v = a.get("held_out_position_rmse_m_by_config", {}).get(cfg, {}).get(est)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n_arcs": int(arr.size),
        "mean_arc_rms_m": round(float(np.mean(arr)), 2) if arr.size else None,
        "median_arc_rms_m": round(float(np.median(arr)), 2) if arr.size else None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="results/real_slr_sp3_corrected")
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the public IERS Earth-orientation series. By "
        "default an archived copy is reused without contacting the network.",
    )
    p.add_argument("--done-sentinel", type=str, default="")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eop_path, eop_method = materialize_eop(out_dir, refresh=args.refresh)
    eop = load_eop_series(eop_path)

    arc_blocks = [run_arc(a, eop) for a in DEFAULT_ARCS]
    completed = [a for a in arc_blocks if a.get("status") == "completed"]

    estimators = [
        "EKF",
        "UKF (fixed-noise)",
        "AUKF (adaptive)",
        "SP3-IC propagation",
    ]
    pooled = {
        cfg: {e: _pool(completed, cfg, e) for e in estimators}
        for cfg in AUDIT_CONFIGS
    }
    fit_innov_pooled = {}
    for cfg in AUDIT_CONFIGS:
        vals = [
            a.get("ekf_fit_innovation_rms_m_by_config", {}).get(cfg)
            for a in completed
        ]
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
        fit_innov_pooled[cfg] = (
            round(float(np.mean(v)), 3) if v.size else None
        )

    # Correction-sensitivity audit: change in pooled UKF held-out RMSE when a
    # single correction is removed (positive => removing it makes the probe
    # worse, i.e. the correction helps).
    ref_est = "UKF (fixed-noise)"
    full_mean = pooled["full"][ref_est]["mean_arc_rms_m"]
    sensitivity = {}
    for cfg in AUDIT_CONFIGS:
        if cfg == "full":
            continue
        m = pooled[cfg][ref_est]["mean_arc_rms_m"]
        sensitivity[cfg] = {
            "pooled_mean_rms_m": m,
            "delta_vs_full_m": (
                round(m - full_mean, 2)
                if (m is not None and full_mean is not None)
                else None
            ),
        }

    # Head-to-head vs the committed GMST-only bounded slice (loop-26).
    prior_path = Path(
        "results/real_slr_sp3_od/real_slr_sp3_od_validation.json"
    )
    prior = {}
    if prior_path.exists():
        pj = json.loads(prior_path.read_text(encoding="utf-8"))
        pr = pj.get("pooled_held_out_position_rmse_m", {})
        prior = {
            e: pr.get(e, {}).get("mean_arc_rms_m") for e in estimators
        }

    caveats = (
        "Full-correction precise-reference real-data sanity probe. "
        "The approximate GMST-only Earth rotation is replaced with a full "
        "IAU-76/80 reduction ingesting the real public IERS finals2000A.all "
        "Earth-orientation series (polar motion and UT1-UTC), and the standard "
        "precise-SLR-reduction corrections the prior bounded slice omitted are "
        "applied: tropospheric refraction (Marini--Murray, from the real CRD "
        "surface meteorology and transmit wavelength), the satellite "
        "centre-of-mass offset, and the relativistic (Shapiro) range delay. "
        "The dynamics remain the compact two-body+J2 model, so this isolates "
        "the Earth-orientation and SLR-reduction effect and is not a "
        "force-model change. Held-out error is a state error against the same "
        "independent ILRS analysis-centre SP3-c precise orbit product. This is "
        "a bounded sanity / provenance probe with explicit approximation "
        "caveats, not a centimetre operational or flight-grade orbit "
        "determination claim; the correction-sensitivity audit reports the "
        "magnitude of each individual correction so the residual gap is "
        "quantified rather than hidden."
    )

    result = {
        "schema_version": "real_slr_sp3_corrected_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed" if completed else "insufficient_observations",
        "targets": sorted({a["target"] for a in completed}),
        "num_arcs": len(DEFAULT_ARCS),
        "num_arcs_completed": len(completed),
        "sp3_analysis_center": ANALYSIS_CENTER,
        "sp3_week_product": _SP3_WEEK,
        "fixed_station_subset": sorted({s.code for s in SUPPORTED_STATIONS.values()}),
        "eop_series": {
            "url": EOP_URL,
            "archived_input_id": EOP_NAME,
            "input_source": eop_method,
            "sha256": sha256_file(eop_path),
            "bytes": eop_path.stat().st_size,
            "n_rows": int(eop.mjd.size),
        },
        "audit_configs": list(AUDIT_CONFIGS),
        "predeclared": {
            "range_std_m": RANGE_STD_M,
            "accel_psd": ACCEL_PSD,
            "train_frac": TRAIN_FRAC,
            "max_step_s": MAX_STEP_S,
            "lageos_centre_of_mass_offset_m": LAGEOS_CENTRE_OF_MASS_OFFSET_M,
            "troposphere_model": "Marini-Murray (1973)",
        },
        "pooled_held_out_position_rmse_m": pooled,
        "pooled_ekf_fit_innovation_rms_m": fit_innov_pooled,
        "correction_sensitivity_audit": sensitivity,
        "head_to_head_vs_committed_gmst_only": {
            "committed_real_slr_sp3_od_mean_m": prior,
            "corrected_full_mean_m": {
                e: pooled["full"][e]["mean_arc_rms_m"] for e in estimators
            },
            "gmst_only_reproduction_mean_m": {
                e: pooled["gmst_only"][e]["mean_arc_rms_m"]
                for e in estimators
            },
        },
        "arcs": arc_blocks,
        "caveats": caveats,
    }
    dump_json(result, out_dir / "real_slr_sp3_corrected_validation.json")
    print(
        json.dumps(
            {
                "status": result["status"],
                "num_arcs_completed": result["num_arcs_completed"],
                "corrected_full_mean_m": result[
                    "head_to_head_vs_committed_gmst_only"
                ]["corrected_full_mean_m"],
                "committed_gmst_only_mean_m": prior,
                "correction_sensitivity_audit": sensitivity,
            },
            indent=2,
        )
    )
    if args.done_sentinel:
        Path(args.done_sentinel).write_text("done\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
