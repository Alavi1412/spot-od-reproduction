#!/usr/bin/env python
"""Strictly held-out SP3-supervised dynamics-residual calibrator evaluation.

Loop-27 Track-A attempt at a genuine positive, externally validated real-data
contribution.  A lightweight least-squares empirical-acceleration calibrator
(:mod:`gnn_state_estimation.sp3_calibrator`) is fitted on SP3-derived residual
accelerations from *training* arcs only and scored on strictly held-out arcs
under two protocols:

* **LOAO** -- leave-one-arc-out: the held-out arc's SP3 and observations are
  never used to fit the calibrator.
* **LOOO** -- leave-one-object-out: train on every LAGEOS-1 arc and score the
  LAGEOS-2 arcs, and vice versa (cross-object, the strongest no-leakage test).

For every held-out arc the corrected EKF / fixed-noise UKF / SP3-IC
propagation are scored predict-only against the *independent* ILRS
analysis-centre SP3-c precise orbit, identically to the uncalibrated slice.
The uncalibrated EKF / fixed-noise UKF / adaptive UKF / SP3-IC baselines are
recomputed here on the same arcs and splits for an apples-to-apples
comparison.  The verdict is reported honestly: the calibrator is claimed as a
bounded positive contribution only if it beats the best uncalibrated classical
reference on a no-leakage split; otherwise it is reported as an honest
negative with a full audit trail.

Public inputs are the same archived CRD v2 / SP3-c files used by the
uncalibrated slice (offline-reproducible; ``--refresh`` re-downloads).
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - entrypoint dependent
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
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
    parse_sp3,
    propagate_compact,
    run_range_aukf,
    run_range_ekf,
    run_range_ukf_fixed,
)
from gnn_state_estimation.sp3_calibrator import (
    ACCEL_SAMPLE_STEP_S,
    FOURIER_ORDER,
    RIDGE_LAMBDA,
    SECOND_DIFF_STEP_S,
    fit_calibrator,
    held_out_position_rmse_corrected,
    propagate_corrected,
    run_corrected_ekf,
    run_corrected_ukf_fixed,
    sp3_residual_samples,
)
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

# Reuse the exact arc set, public URLs, archival, and predeclared filter
# settings of the uncalibrated slice so the comparison is apples-to-apples.
from scripts.run_real_slr_sp3_od_validation import (  # noqa: E402
    ACCEL_PSD,
    ANALYSIS_CENTER,
    DEFAULT_ARCS,
    EDC_CRD_URL,
    EDC_SP3_URL,
    INIT_POS_STD_M,
    INIT_VEL_STD_MPS,
    MAX_STEP_S,
    RANGE_STD_M,
    TRAIN_FRAC,
    _read_sp3_text,
    materialize,
    sha256_file,
)


def load_arc(arc, out_dir: Path, refresh: bool) -> dict | None:
    """Load one arc's observations, SP3 interpolator, and fit/held split."""
    year = arc.date[:4]
    crd_url = EDC_CRD_URL.format(sat_key=arc.sat_key, year=year, date=arc.date)
    sp3_url = EDC_SP3_URL.format(sat_key=arc.sat_key, week=arc.sp3_week)
    crd_path = out_dir / f"{arc.sat_key}_{arc.date}.np2"
    sp3_path = out_dir / f"nsgf.orb.{arc.sat_key}.{arc.sp3_week}.v80.sp3.gz"
    materialize(crd_url, crd_path, refresh=refresh)
    materialize(sp3_url, sp3_path, refresh=refresh)

    points = parse_crd_v2_normal_points(
        crd_path.read_text(encoding="utf-8", errors="replace")
    )
    eph = parse_sp3(_read_sp3_text(sp3_path), arc.sp3_sat_id)
    interp = Sp3Interpolator(eph, order=9)
    margin = 30.0
    points = [p for p in points if eph.covers(p.epoch_unix, margin_s=-margin)]
    if len(points) < 10:
        return None

    epochs = np.array([p.epoch_unix for p in points], dtype=np.float64)
    ranges = np.array([p.range_m for p in points], dtype=np.float64)
    station_pi = np.vstack(
        [
            station_pseudo_inertial_m(SUPPORTED_STATIONS[p.cdp_id], p.epoch_unix)
            for p in points
        ]
    )
    obs_all = [
        RangeObs(
            epoch_unix=float(epochs[i]),
            station_pi_m=station_pi[i],
            range_m=float(ranges[i]),
        )
        for i in range(len(points))
    ]
    n = len(obs_all)
    n_fit = max(6, int(np.floor(TRAIN_FRAC * n)))
    n_fit = min(n_fit, n - 3)
    fit_obs = obs_all[:n_fit]
    held_epochs = epochs[n_fit:]
    t0 = fit_obs[0].epoch_unix
    fit_last_epoch = fit_obs[-1].epoch_unix
    x0 = interp.state_pseudo_inertial_m(t0)
    p0 = np.diag(
        np.array(
            [INIT_POS_STD_M**2] * 3 + [INIT_VEL_STD_MPS**2] * 3,
            dtype=np.float64,
        )
    )
    # SP3-derived empirical-acceleration training pool over the arc's
    # observation window (used only when this arc is a *training* arc).
    phi, da = sp3_residual_samples(
        interp, float(epochs[0]), float(epochs[-1])
    )
    return {
        "arc_id": f"{arc.target} {arc.date}",
        "target": arc.target,
        "date": arc.date,
        "obs_all": obs_all,
        "fit_obs": fit_obs,
        "held_epochs": held_epochs,
        "t0": t0,
        "fit_last_epoch": fit_last_epoch,
        "x0": x0,
        "p0": p0,
        "interp": interp,
        "n_obs": n,
        "n_fit": int(n_fit),
        "n_held": int(n - n_fit),
        "phi": phi,
        "da": da,
        "crd": {
            "url": crd_url,
            "archived_input_id": crd_path.name,
            "sha256": sha256_file(crd_path),
            "bytes": crd_path.stat().st_size,
        },
        "sp3": {
            "url": sp3_url,
            "archived_input_id": sp3_path.name,
            "sha256": sha256_file(sp3_path),
            "bytes": sp3_path.stat().st_size,
        },
    }


def _rms(state, a):
    return held_out_position_rmse(
        state, a["fit_last_epoch"], a["held_epochs"], a["interp"], MAX_STEP_S
    ).get("rms_m", float("nan"))


def _rms_corr(state, a, calib):
    return held_out_position_rmse_corrected(
        state,
        a["fit_last_epoch"],
        a["held_epochs"],
        a["interp"],
        calib,
        MAX_STEP_S,
    ).get("rms_m", float("nan"))


def uncalibrated_arc_rmse(a: dict) -> dict:
    """Uncalibrated EKF / fixed UKF / adaptive UKF / SP3-IC for one arc."""
    fit, x0, p0 = a["fit_obs"], a["x0"], a["p0"]
    ekf = run_range_ekf(fit, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    ukf = run_range_ukf_fixed(fit, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    aukf = run_range_aukf(fit, x0, p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    sp3_ic = propagate_compact(
        x0, a["fit_last_epoch"] - a["t0"], MAX_STEP_S
    )
    return {
        "EKF": _rms(ekf["state"], a),
        "UKF (fixed-noise)": _rms(ukf["state"], a),
        "AUKF (adaptive)": _rms(aukf["state"], a),
        "SP3-IC propagation": _rms(sp3_ic, a),
    }


def calibrated_arc_rmse(a: dict, calib) -> dict:
    """Calibrator-corrected EKF / fixed UKF / SP3-IC for one held-out arc."""
    fit, x0, p0 = a["fit_obs"], a["x0"], a["p0"]
    ekf = run_corrected_ekf(
        fit, x0, p0, RANGE_STD_M, ACCEL_PSD, calib, MAX_STEP_S
    )
    ukf = run_corrected_ukf_fixed(
        fit, x0, p0, RANGE_STD_M, ACCEL_PSD, calib, MAX_STEP_S
    )
    sp3_ic = propagate_corrected(
        x0, a["fit_last_epoch"] - a["t0"], calib, MAX_STEP_S
    )
    return {
        "Calibrated-EKF": _rms_corr(ekf["state"], a, calib),
        "Calibrated-UKF (fixed-noise)": _rms_corr(ukf["state"], a, calib),
        "Calibrated-SP3-IC propagation": _rms_corr(sp3_ic, a, calib),
    }


def pooled_mean(per_arc: list[dict], key: str) -> float:
    vals = [
        d[key]
        for d in per_arc
        if key in d and np.isfinite(d.get(key, np.nan))
    ]
    return round(float(np.mean(vals)), 2) if vals else float("nan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=str, default="results/real_slr_sp3_od")
    p.add_argument("--refresh", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arcs = [load_arc(a, out_dir, args.refresh) for a in DEFAULT_ARCS]
    arcs = [a for a in arcs if a is not None]
    if len(arcs) < 4:
        raise RuntimeError("insufficient arcs for held-out calibration")

    per_arc = []
    for i, a in enumerate(arcs):
        unc = uncalibrated_arc_rmse(a)

        # LOAO: train on every other arc (held-out arc fully excluded).
        loao_phi = np.vstack(
            [arcs[j]["phi"] for j in range(len(arcs)) if j != i]
        )
        loao_da = np.vstack(
            [arcs[j]["da"] for j in range(len(arcs)) if j != i]
        )
        loao_cal = fit_calibrator(loao_phi, loao_da)
        loao = calibrated_arc_rmse(a, loao_cal)

        # LOOO: train on the *other object's* arcs only (cross-object).
        other = [
            arcs[j]
            for j in range(len(arcs))
            if arcs[j]["target"] != a["target"]
        ]
        looo_phi = np.vstack([o["phi"] for o in other])
        looo_da = np.vstack([o["da"] for o in other])
        looo_cal = fit_calibrator(looo_phi, looo_da)
        looo = calibrated_arc_rmse(a, looo_cal)

        per_arc.append(
            {
                "arc_id": a["arc_id"],
                "target": a["target"],
                "date": a["date"],
                "num_observations": a["n_obs"],
                "num_fit": a["n_fit"],
                "num_held_out": a["n_held"],
                "uncalibrated_rmse_m": {
                    k: (round(v, 2) if np.isfinite(v) else None)
                    for k, v in unc.items()
                },
                "loao_calibrated_rmse_m": {
                    k: (round(v, 2) if np.isfinite(v) else None)
                    for k, v in loao.items()
                },
                "looo_calibrated_rmse_m": {
                    k: (round(v, 2) if np.isfinite(v) else None)
                    for k, v in looo.items()
                },
                "loao_n_train_samples": loao_cal.n_train_samples,
                "looo_n_train_samples": looo_cal.n_train_samples,
                **{f"_unc_{k}": v for k, v in unc.items()},
                **{f"_loao_{k}": v for k, v in loao.items()},
                **{f"_looo_{k}": v for k, v in looo.items()},
            }
        )

    unc_keys = [
        "EKF",
        "UKF (fixed-noise)",
        "AUKF (adaptive)",
        "SP3-IC propagation",
    ]
    cal_keys = [
        "Calibrated-EKF",
        "Calibrated-UKF (fixed-noise)",
        "Calibrated-SP3-IC propagation",
    ]
    pooled = {
        "uncalibrated": {
            k: pooled_mean(per_arc, f"_unc_{k}") for k in unc_keys
        },
        "loao_calibrated": {
            k: pooled_mean(per_arc, f"_loao_{k}") for k in cal_keys
        },
        "looo_calibrated": {
            k: pooled_mean(per_arc, f"_looo_{k}") for k in cal_keys
        },
    }

    # Honest verdict: best uncalibrated classical reference vs best calibrated.
    best_unc_ref = min(
        ("EKF", "UKF (fixed-noise)", "AUKF (adaptive)"),
        key=lambda k: pooled["uncalibrated"][k],
    )
    best_unc_val = pooled["uncalibrated"][best_unc_ref]

    def _verdict(proto: str) -> dict:
        cal = pooled[f"{proto}_calibrated"]
        best_cal_key = min(cal, key=lambda k: cal[k])
        best_cal_val = cal[best_cal_key]
        # Paired per-arc comparison: calibrated UKF vs uncalibrated UKF.
        deltas = [
            d[f"_unc_UKF (fixed-noise)"]
            - d[f"_{proto}_Calibrated-UKF (fixed-noise)"]
            for d in per_arc
            if np.isfinite(d.get(f"_unc_UKF (fixed-noise)", np.nan))
            and np.isfinite(
                d.get(f"_{proto}_Calibrated-UKF (fixed-noise)", np.nan)
            )
        ]
        n_better = int(sum(1 for x in deltas if x > 0))
        return {
            "best_calibrated_estimator": best_cal_key,
            "best_calibrated_pooled_mean_m": best_cal_val,
            "beats_best_uncalibrated_reference": bool(
                best_cal_val < best_unc_val
            ),
            "calibrated_ukf_vs_uncalibrated_ukf": {
                "n_arcs": len(deltas),
                "n_arcs_calibrated_better": n_better,
                "mean_rmse_reduction_m": (
                    round(float(np.mean(deltas)), 2) if deltas else None
                ),
                "median_rmse_reduction_m": (
                    round(float(np.median(deltas)), 2) if deltas else None
                ),
            },
        }

    verdict = {
        "best_uncalibrated_classical_reference": best_unc_ref,
        "best_uncalibrated_classical_reference_pooled_mean_m": best_unc_val,
        "loao": _verdict("loao"),
        "looo": _verdict("looo"),
        "claimed_as_positive_contribution": bool(
            _verdict("loao")["beats_best_uncalibrated_reference"]
            and _verdict("looo")["beats_best_uncalibrated_reference"]
        ),
    }

    serial_arcs = [
        {k: v for k, v in d.items() if not k.startswith("_")}
        for d in per_arc
    ]
    result = {
        "schema_version": "sp3_residual_calibrator_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed",
        "sp3_analysis_center": ANALYSIS_CENTER,
        "num_arcs": len(arcs),
        "predeclared_calibrator": {
            "model": (
                "empirical RSW acceleration, Fourier in argument of latitude"
            ),
            "fourier_order": FOURIER_ORDER,
            "n_coefficients": 3 * (2 * FOURIER_ORDER + 1),
            "ridge_lambda": RIDGE_LAMBDA,
            "accel_sample_step_s": ACCEL_SAMPLE_STEP_S,
            "second_diff_step_s": SECOND_DIFF_STEP_S,
            "fit_method": "ridge least squares on SP3-derived residual accel.",
        },
        "no_leakage_protocol": (
            "The calibrator is fitted only on SP3-derived residual "
            "accelerations from training arcs. Under LOAO the held-out arc's "
            "SP3 and observations are fully excluded from the fit; under LOOO "
            "the calibrator is trained on one LAGEOS object and scored on the "
            "other (cross-object). Held-out error is a predict-only state "
            "error versus the independent ILRS analysis-centre SP3-c precise "
            "orbit, identical to the uncalibrated slice. Calibrator "
            "hyper-parameters are predeclared from standard "
            "empirical-acceleration practice, not tuned to held-out scores."
        ),
        "pooled_held_out_position_rmse_m": pooled,
        "verdict": verdict,
        "arcs": serial_arcs,
        "caveats": (
            "Bounded precise-reference real-data OD slice (same fidelity as "
            "the uncalibrated slice: approximate GMST-only Earth rotation "
            "common-mode between station geometry and the SP3 reference, "
            "compact two-body+J2 base dynamics, no precise SLR reduction, "
            "precise SP3 initial condition). The learned calibrator is a "
            "least-squares empirical-acceleration correction fitted on "
            "SP3-derived residual accelerations from strictly disjoint "
            "training arcs and is reported honestly whether or not it beats "
            "the best uncalibrated classical reference; absolute magnitudes "
            "remain a hundreds-of-metres model-mismatch stress, not "
            "centimetre operational OD or flight readiness."
        ),
    }
    dump_json(result, out_dir / "sp3_residual_calibrator.json")
    print(
        json.dumps(
            {
                "pooled_held_out_position_rmse_m": pooled,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
