#!/usr/bin/env python
"""Bounded real ILRS SLR range-residual audit for the LAGEOS geodetic pair.

This script runs a deterministic, multi-object, multi-day real-measurement
audit on public ILRS satellite-laser-ranging (SLR) normal points.  By default
it scores four estimators on four real arcs --- LAGEOS-1 and LAGEOS-2, each on
two consecutive UTC days --- using a single fixed four-station ILRS subset, so
the audit spans two distinct geodetic satellites and four independent passes
rather than a single object/day.

For every arc it parses the public daily Consolidated Laser Ranging Data
(CRD v2) normal-point file from the EDC archive and a CelesTrak TLE prior,
splits the arc deterministically in time into an earlier fit set and a later
held-out set, and scores: the SGP4 prior, a classical range-only robust
weighted-least-squares fit, a learned bounded residual correction of the prior,
and a hybrid WLS-plus-learned residual calibration.  Held-out range residuals
are pooled across all arcs per estimator.

The audit is reproducible offline by default: archived public input files are
reused without contacting the network, so the result regenerates offline.
Passing ``--refresh`` re-downloads the public inputs from their archive URLs.

It is a deliberately bounded real-measurement audit, not an operational orbit
determination: the station transform is an approximate GMST rotation, the
dynamics are the compact two-body+J2 model (no precise SLR corrections), the
TLE is used only as a prior, and the held-out score is a range residual
because no independent truth state is available.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
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
from scipy.optimize import least_squares
from sgp4.api import Satrec

from gnn_state_estimation.dynamics import rk4_step
from gnn_state_estimation.slr import (
    SUPPORTED_STATIONS,
    parse_crd_v2_normal_points,
    station_pseudo_inertial_m,
    summarize_residuals,
)
from gnn_state_estimation.slr_learned import (
    build_feature_matrix,
    learned_residual_correction,
    orbital_period_seconds,
)
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

EDC_CRD_URL = (
    "https://edc.dgfi.tum.de/pub/slr/data/npt_crd_v2/{sat_key}/{year}/"
    "{sat_key}_{date}.np2"
)
CELESTRAK_TLE_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=TLE"
)
# LAGEOS-1 and LAGEOS-2 are passive geodetic spheres at ~5.8-5.9 Mm altitude:
# atmospheric drag is negligible, so a near-zero ballistic coefficient is used
# with the compact two-body+J2 model.  This is documented in the manuscript.
LAGEOS_BALLISTIC_COEFF = 1.0e-9


@dataclass(frozen=True)
class Arc:
    """One real SLR arc: a target satellite observed over one UTC day."""

    target: str
    sat_key: str
    catnr: int
    date: str


# Deterministic default audit: two geodetic satellites over two consecutive
# UTC days = four independent real arcs, scored on one fixed ILRS station
# subset.  These public daily files are archived on first fetch and reused
# offline thereafter.
DEFAULT_ARCS: tuple[Arc, ...] = (
    Arc("LAGEOS-1", "lageos1", 8820, "20260517"),
    Arc("LAGEOS-1", "lageos1", 8820, "20260518"),
    Arc("LAGEOS-2", "lageos2", 22195, "20260517"),
    Arc("LAGEOS-2", "lageos2", 22195, "20260518"),
)

ESTIMATOR_KINDS = {
    "SGP4 prior": "classical",
    "Range-only WLS fit": "classical",
    "Learned residual (SGP4 prior)": "learned",
    "WLS + learned residual calibration": "hybrid",
}


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


def _python_download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        output_path.write_bytes(response.read())


def materialize(url: str, output_path: Path, *, refresh: bool) -> str:
    """Provide the public input file, preferring an existing archived copy.

    By default an already-present local input file is reused without
    contacting the network, so the audit is reproducible offline from the
    archived public inputs.  When ``refresh`` is set the public file is
    re-downloaded from its archive URL, reverting to the existing archived
    copy only if the network is unavailable.
    """
    if not refresh and output_path.exists():
        return "archived_input"
    try:
        _powershell_download(url, output_path)
        return "public_archive_download"
    except subprocess.CalledProcessError:
        try:
            _python_download(url, output_path)
            return "public_archive_download"
        except (urllib.error.URLError, OSError) as exc:
            if output_path.exists():
                return "archived_input_fallback"
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sgp4_state_eci_m(sat: Satrec, epoch_unix: float) -> np.ndarray:
    """SGP4 TEME state in metres / metres-per-second for a POSIX epoch."""
    jd = epoch_unix / 86400.0 + 2440587.5
    jd_int = float(np.floor(jd - 0.5)) + 0.5
    fr = jd - jd_int
    err, r_km, v_km = sat.sgp4(jd_int, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 propagation error code {err}")
    return np.array(
        [r_km[0] * 1e3, r_km[1] * 1e3, r_km[2] * 1e3,
         v_km[0] * 1e3, v_km[1] * 1e3, v_km[2] * 1e3],
        dtype=np.float64,
    )


def propagate_states(
    state0: np.ndarray, rel_times_s: np.ndarray, max_step_s: float
) -> np.ndarray:
    """Propagate ``state0`` (compact two-body+J2) to each relative time.

    ``rel_times_s`` is assumed sorted and non-negative (time since epoch t0).
    """
    states = np.zeros((rel_times_s.size, 6), dtype=np.float64)
    state = np.asarray(state0, dtype=np.float64).copy()
    t = 0.0
    for i, target in enumerate(rel_times_s):
        span = float(target) - t
        if span > 0.0:
            n_sub = max(1, int(np.ceil(span / max_step_s)))
            dt = span / n_sub
            for _ in range(n_sub):
                state = rk4_step(
                    state,
                    dt=dt,
                    ballistic_coeff_m2_per_kg=LAGEOS_BALLISTIC_COEFF,
                    t_s=t,
                )
                t += dt
        states[i] = state
    return states


def predicted_ranges(
    sat_states_eci: np.ndarray, station_eci: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(sat_states_eci[:, :3] - station_eci, axis=1)


def split_block(
    residuals: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray
) -> dict:
    return {
        "all": summarize_residuals(residuals),
        "train": summarize_residuals(residuals[train_idx]),
        "validation": summarize_residuals(residuals[val_idx]),
    }


def run_arc(arc: Arc, out_dir: Path, args: argparse.Namespace) -> dict:
    """Score the four estimators on one real SLR arc.

    Returns a per-arc result block plus the raw fit-arc and held-out residual
    arrays per estimator so the caller can pool held-out residuals across arcs.
    """
    year = arc.date[:4]
    crd_url = EDC_CRD_URL.format(sat_key=arc.sat_key, year=year, date=arc.date)
    tle_url = CELESTRAK_TLE_URL.format(catnr=arc.catnr)
    crd_path = out_dir / f"{arc.sat_key}_{arc.date}.np2"
    tle_path = out_dir / f"{arc.sat_key}_{arc.catnr}.tle"

    crd_method = materialize(crd_url, crd_path, refresh=args.refresh)
    tle_method = materialize(tle_url, tle_path, refresh=args.refresh)

    crd_text = crd_path.read_text(encoding="utf-8", errors="replace")
    points = parse_crd_v2_normal_points(crd_text)

    tle_lines = [
        ln.strip()
        for ln in tle_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if ln.strip()
    ]
    tle1 = next(ln for ln in tle_lines if ln.startswith("1 "))
    tle2 = next(ln for ln in tle_lines if ln.startswith("2 "))
    sat = Satrec.twoline2rv(tle1, tle2)

    station_counts: dict[str, int] = {}
    for p in points:
        station_counts[p.station_code] = station_counts.get(p.station_code, 0) + 1

    arc_id = f"{arc.target} {arc.date}"
    provenance = {
        "arc_id": arc_id,
        "target": arc.target,
        "norad_cat_id": arc.catnr,
        "date": arc.date,
        "crd": {
            "url": crd_url,
            "archived_input_id": crd_path.name,
            "input_source": crd_method,
            "sha256": sha256_file(crd_path),
            "bytes": crd_path.stat().st_size,
        },
        "tle": {
            "url": tle_url,
            "archived_input_id": tle_path.name,
            "input_source": tle_method,
            "sha256": sha256_file(tle_path),
            "bytes": tle_path.stat().st_size,
            "line1": tle1,
            "line2": tle2,
        },
    }

    min_required = 8
    if len(points) < min_required:
        return {
            **provenance,
            "status": "insufficient_observations",
            "fit_success": False,
            "num_observations": len(points),
            "station_counts": station_counts,
            "_residuals": {},
        }

    epochs = np.array([p.epoch_unix for p in points], dtype=np.float64)
    ranges = np.array([p.range_m for p in points], dtype=np.float64)
    t0 = float(epochs[0])
    rel_times = epochs - t0

    station_eci = np.vstack([
        station_pseudo_inertial_m(SUPPORTED_STATIONS[p.cdp_id], p.epoch_unix)
        for p in points
    ])

    # --- SGP4 prior: predict each range directly from the TLE ---
    sgp4_states = np.vstack([sgp4_state_eci_m(sat, e) for e in epochs])
    prior_pred = predicted_ranges(sgp4_states, station_eci)
    prior_residuals = prior_pred - ranges

    # --- Deterministic time split (first train_frac fit, remainder held out) ---
    n = len(points)
    n_train = max(6, int(np.floor(args.train_frac * n)))
    n_train = min(n_train, n - 1)
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n)
    time_span_s = float(epochs[-1] - epochs[0])

    state0_prior = sgp4_state_eci_m(sat, t0)

    def residual_fn(state0: np.ndarray) -> np.ndarray:
        states = propagate_states(state0, rel_times[train_idx], args.max_step_s)
        pred = predicted_ranges(states, station_eci[train_idx])
        return pred - ranges[train_idx]

    fit_success = True
    fit_message = ""
    try:
        sol = least_squares(
            residual_fn,
            state0_prior.copy(),
            method="trf",
            loss="soft_l1",
            f_scale=50.0,
            x_scale=np.array([1e6, 1e6, 1e6, 1e3, 1e3, 1e3]),
            max_nfev=args.max_nfev,
        )
        state0_fit = sol.x
        fit_message = str(sol.message)
        fit_success = bool(sol.success) or sol.status > 0
    except Exception as exc:  # pragma: no cover - numerical safety net
        state0_fit = state0_prior.copy()
        fit_success = False
        fit_message = f"{type(exc).__name__}: {exc}"

    fitted_states = propagate_states(state0_fit, rel_times, args.max_step_s)
    fitted_pred = predicted_ranges(fitted_states, station_eci)
    fitted_residuals = fitted_pred - ranges

    # --- Learned residual estimators on the same deterministic split ---
    period_s = orbital_period_seconds(tle2)
    station_codes = [p.station_code for p in points]

    prior_feats = build_feature_matrix(
        station_codes, epochs, prior_pred, t0, time_span_s, period_s
    )
    prior_corr, learned_backend = learned_residual_correction(
        prior_feats, prior_residuals, train_idx, seed=0
    )
    learned_prior_residuals = prior_residuals - prior_corr

    fitted_feats = build_feature_matrix(
        station_codes, epochs, fitted_pred, t0, time_span_s, period_s
    )
    fitted_corr, _ = learned_residual_correction(
        fitted_feats, fitted_residuals, train_idx, seed=0
    )
    learned_fitted_residuals = fitted_residuals - fitted_corr

    resid_by_estimator = {
        "SGP4 prior": prior_residuals,
        "Range-only WLS fit": fitted_residuals,
        "Learned residual (SGP4 prior)": learned_prior_residuals,
        "WLS + learned residual calibration": learned_fitted_residuals,
    }

    arc_block = {
        **provenance,
        "status": "completed",
        "fit_success": fit_success,
        "fit_message": fit_message,
        "num_observations": n,
        "num_train": int(n_train),
        "num_validation": int(n - n_train),
        "time_span_hours": time_span_s / 3600.0,
        "first_epoch_iso": points[0].epoch_iso,
        "last_epoch_iso": points[-1].epoch_iso,
        "station_counts": station_counts,
        "distinct_stations": len(station_counts),
        "learned_backend": learned_backend,
        "estimators": {
            name: split_block(res, train_idx, val_idx)
            for name, res in resid_by_estimator.items()
        },
        "_residuals": {
            name: {
                "train": np.asarray(res[train_idx], dtype=np.float64),
                "validation": np.asarray(res[val_idx], dtype=np.float64),
            }
            for name, res in resid_by_estimator.items()
        },
    }
    return arc_block


def _pool(arrays: list[np.ndarray]) -> np.ndarray:
    finite = [a for a in arrays if a is not None and a.size]
    if not finite:
        return np.array([], dtype=np.float64)
    return np.concatenate(finite)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default="results/real_slr_lageos")
    parser.add_argument("--max-step-s", type=float, default=30.0)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the public CRD and TLE inputs from their archive "
             "URLs. By default existing local input files are reused without "
             "contacting the network so the audit is reproducible offline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    caveats = (
        "Bounded real-measurement audit only. Approximate GMST-only station "
        "rotation (no polar motion/nutation; UT1~UTC), compact two-body+J2 "
        "dynamics with no precise SLR corrections (relativistic, tropospheric, "
        "centre-of-mass), TLE used only as a prior. A fixed four-station ILRS "
        "subset is used across every arc. Two geodetic satellites (LAGEOS-1, "
        "LAGEOS-2) are each scored on two consecutive UTC days; held-out range "
        "residuals are pooled across all arcs per estimator on deterministic "
        "earlier-fit / later-held-out time splits. The held-out score is a "
        "range residual (no truth state is available) and the learned "
        "correction is hard-bounded and deterministic. Not an operational OD "
        "validation and not a flight-readiness result."
    )

    arcs = list(DEFAULT_ARCS)
    arc_blocks: list[dict] = []
    for arc in arcs:
        arc_blocks.append(run_arc(arc, out_dir, args))

    completed = [a for a in arc_blocks if a.get("status") == "completed"]

    # --- Pool held-out (and fit-arc) residuals across arcs per estimator ---
    per_estimator = []
    pooled_blocks: dict[str, dict] = {}
    arc_best_count: dict[str, int] = {n: 0 for n in ESTIMATOR_KINDS}

    for arc in completed:
        finite = {
            name: float(np.sqrt(np.mean(r["validation"] ** 2)))
            for name, r in arc["_residuals"].items()
            if r["validation"].size
            and np.all(np.isfinite(r["validation"]))
        }
        if finite:
            best_name = min(finite, key=lambda k: finite[k])
            arc_best_count[best_name] += 1

    for name, kind in ESTIMATOR_KINDS.items():
        held = _pool([
            a["_residuals"].get(name, {}).get("validation")
            for a in completed
        ])
        fit = _pool([
            a["_residuals"].get(name, {}).get("train")
            for a in completed
        ])
        held_sum = summarize_residuals(held)
        fit_sum = summarize_residuals(fit)
        pooled_blocks[name] = {"fit": fit_sum, "held_out": held_sum}
        per_estimator.append({
            "name": name,
            "kind": kind,
            "pooled_fit_rms_m": fit_sum["rms_m"],
            "pooled_held_out_rms_m": held_sum["rms_m"],
            "pooled_held_out_mae_m": held_sum["mae_m"],
            "pooled_held_out_p95_abs_m": held_sum["p95_abs_m"],
            "pooled_held_out_count": held_sum["count"],
            "arcs_best_of": arc_best_count[name],
        })

    def _hr(name: str) -> float:
        return float(pooled_blocks.get(name, {}).get("held_out", {}).get(
            "rms_m", float("nan")))

    finite_est = [
        e for e in per_estimator
        if np.isfinite(e["pooled_held_out_rms_m"])
    ]
    best_estimator = (
        min(finite_est, key=lambda e: e["pooled_held_out_rms_m"])["name"]
        if finite_est else None
    )
    prior_rms = _hr("SGP4 prior")
    wls_rms = _hr("Range-only WLS fit")
    learned_prior_rms = _hr("Learned residual (SGP4 prior)")
    learned_wls_rms = _hr("WLS + learned residual calibration")
    learned_beats_prior = bool(
        np.isfinite(learned_prior_rms) and np.isfinite(prior_rms)
        and learned_prior_rms < prior_rms
    )
    learned_beats_wls = bool(
        np.isfinite(wls_rms)
        and min(learned_prior_rms, learned_wls_rms) < wls_rms
    )
    held_out_improved = bool(
        np.isfinite(prior_rms) and np.isfinite(wls_rms)
        and wls_rms < prior_rms
    )

    # Serializable per-arc blocks (drop the raw residual arrays).
    serial_arcs = []
    for a in arc_blocks:
        a2 = {k: v for k, v in a.items() if k != "_residuals"}
        serial_arcs.append(a2)

    input_digests = []
    for a in serial_arcs:
        for kind_key in ("crd", "tle"):
            blk = a.get(kind_key)
            if isinstance(blk, dict) and "sha256" in blk:
                input_digests.append({
                    "arc_id": a.get("arc_id"),
                    "kind": kind_key,
                    "archived_input_id": blk.get("archived_input_id"),
                    "sha256": blk.get("sha256"),
                    "bytes": blk.get("bytes"),
                    "url": blk.get("url"),
                })

    n_obs_total = sum(a.get("num_observations", 0) for a in completed)
    n_val_total = sum(a.get("num_validation", 0) for a in completed)
    objects = sorted({a["target"] for a in completed})
    days = sorted({a["date"] for a in completed})

    result = {
        "schema_version": "real_slr_multi_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed" if completed else "insufficient_observations",
        "targets": objects,
        "num_objects": len(objects),
        "num_arcs": len(arcs),
        "num_arcs_completed": len(completed),
        "days": days,
        "fixed_station_subset": sorted(
            {s.code for s in SUPPORTED_STATIONS.values()}
        ),
        "num_observations_total": n_obs_total,
        "num_held_out_total": n_val_total,
        "train_frac": args.train_frac,
        "max_step_s": args.max_step_s,
        "learned_correction_clip_k": 4.0,
        "arcs": serial_arcs,
        "pooled_estimators": pooled_blocks,
        "estimators_summary": per_estimator,
        "best_held_out_estimator": best_estimator,
        "held_out_improved_over_prior": held_out_improved,
        "learned_improved_over_prior": learned_beats_prior,
        "learned_beats_classical_wls": learned_beats_wls,
        "input_digests": input_digests,
        "caveats": caveats,
    }
    dump_json(result, out_dir / "real_slr_lageos_validation.json")
    print(json.dumps({
        "status": result["status"],
        "num_objects": result["num_objects"],
        "num_arcs_completed": result["num_arcs_completed"],
        "days": result["days"],
        "num_observations_total": n_obs_total,
        "num_held_out_total": n_val_total,
        "best_held_out_estimator": best_estimator,
        "learned_improved_over_prior": learned_beats_prior,
        "learned_beats_classical_wls": learned_beats_wls,
        "estimators_summary": per_estimator,
    }, indent=2))


if __name__ == "__main__":
    main()
