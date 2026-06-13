#!/usr/bin/env python
"""Temporal public CRD/SP3 real-measurement OD campaign.

This is the smallest rigorous version of the stronger public-data experiment:

* fit a learned empirical residual-acceleration calibrator from training-week
  SP3 states only;
* select its ridge strength, and then the overall learned/classical candidate,
  using validation-week real CRD normal-point OD scores only;
* freeze that choice and score the strictly later test week against the
  independent SP3 precise-orbit states.

The result is deliberately bounded.  It is a public real-measurement OD probe
with no test-set selection leakage, not operational POD, centimetre SLR
validation, or central external validation of the simulator conclusion.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gnn_state_estimation.slr import parse_crd_v2_normal_points
from gnn_state_estimation.sp3 import (
    RangeObs,
    _process_cov,
    _sigma_points,
    _ukf_weights,
    parse_sp3,
)
from gnn_state_estimation.sp3_hifi_calibrator import (
    RIDGE_GRID,
    HifiCalibrator,
    fit_ridge,
    held_out_rmse_corrected,
    propagate_hifi_corrected,
    residual_samples,
)
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

try:  # noqa: E402
    from run_real_slr_sp3_hifi_validation import (
        ACCEL_PSD,
        ANALYSIS_CENTER,
        INIT_POS_STD_M,
        INIT_VEL_STD_MPS,
        RANGE_STD_M,
        SLR_MAX_STEP_S,
        SPLIT_WEEKS,
        TRAIN_FRAC,
        EDC_CRD_URL,
        EDC_SP3_URL,
        GcrsInterp,
        _arcs,
        _read_sp3_text,
        _station_inertial,
        _week_objects,
        materialize,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover - import context dependent
    from scripts.run_real_slr_sp3_hifi_validation import (
        ACCEL_PSD,
        ANALYSIS_CENTER,
        INIT_POS_STD_M,
        INIT_VEL_STD_MPS,
        RANGE_STD_M,
        SLR_MAX_STEP_S,
        SPLIT_WEEKS,
        TRAIN_FRAC,
        EDC_CRD_URL,
        EDC_SP3_URL,
        GcrsInterp,
        _arcs,
        _read_sp3_text,
        _station_inertial,
        _week_objects,
        materialize,
        sha256_file,
    )


DEFAULT_HIFI_JSON = Path(
    "results/real_slr_sp3_hifi/real_slr_sp3_hifi_validation.json"
)
DEFAULT_INPUT_DIR = Path("results/real_slr_sp3_hifi")
DEFAULT_OUTPUT_JSON = (
    Path("results/real_slr_sp3_temporal_od_campaign")
    / "real_slr_sp3_temporal_od_campaign.json"
)
DEFAULT_TABLE = Path("paper/tables/real_slr_sp3_temporal_od_campaign.tex")

BOOTSTRAP_SEED = 20260522
BOOTSTRAP_N = 5000
CALIBRATOR_SAMPLE_STEP_S = 300.0
CALIBRATOR_EDGE_MARGIN_S = 8.0 * 60.0
CLASSICAL_LABELS = (
    "UKF (compact)",
    "UKF (higher-fidelity)",
    "AUKF (higher-fidelity)",
)
LEARNED_LABEL = "Learned calibrated higher-fidelity UKF"


@dataclass(frozen=True)
class OdArc:
    arc_id: str
    target: str
    date: str
    split: str
    fit_obs: list[RangeObs]
    held_epochs: np.ndarray
    t0: float
    fit_last_epoch: float
    x0: np.ndarray
    p0: np.ndarray
    interp: GcrsInterp
    num_observations: int
    num_fit: int
    num_held: int
    provenance: dict


def _round(x, ndigits: int = 2):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), ndigits)


def _lambda_key(lam: float) -> str:
    return f"{float(lam):.0e}"


def select_lowest(means: dict[str, float | None]) -> str:
    finite = {
        k: float(v)
        for k, v in means.items()
        if v is not None and np.isfinite(v)
    }
    if not finite:
        raise ValueError("no finite candidate mean")
    return min(finite, key=lambda k: (finite[k], k))


def _bootstrap_ci(values: np.ndarray) -> list:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [None, None]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_N, arr.size))
    means = arr[idx].mean(axis=1)
    return [
        round(float(np.percentile(means, 2.5)), 2),
        round(float(np.percentile(means, 97.5)), 2),
    ]


def paired_gap_summary(
    rows: list[dict], a_key: str, b_key: str, *, field: str
) -> dict:
    """Paired gap a-b; positive means candidate ``a`` has larger RMSE."""
    gaps = []
    for row in rows:
        vals = row.get(field, {})
        a = vals.get(a_key)
        b = vals.get(b_key)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            gaps.append(float(a) - float(b))
    arr = np.asarray(gaps, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean_gap_m": _round(arr.mean()),
        "median_gap_m": _round(np.median(arr)),
        "n_a_lower_rmse": int((arr < 0.0).sum()),
        "fraction_a_lower_rmse": round(float((arr < 0.0).mean()), 3),
        "bootstrap95_mean_gap_m": _bootstrap_ci(arr),
        "gap_convention": (
            "a_minus_b; positive means candidate a has larger held-out RMSE"
        ),
    }


def _mean_by_label(rows: list[dict], label: str) -> float | None:
    vals = [
        row["held_out_position_rmse_m"].get(label)
        for row in rows
        if row.get("held_out_position_rmse_m", {}).get(label) is not None
    ]
    vals = [float(v) for v in vals if np.isfinite(v)]
    if not vals:
        return None
    return round(float(np.mean(vals)), 2)


def _hifi_arc_lookup(hifi: dict) -> dict[str, dict]:
    return {
        row["arc_id"]: row
        for row in hifi.get("arcs", [])
        if row.get("status") == "completed"
    }


def _classical_means_from_hifi(hifi: dict, split: str) -> dict[str, float]:
    pool = hifi["sparse_slr_operational_realism"][split]
    return {label: pool[label]["mean_arc_rms_m"] for label in CLASSICAL_LABELS}


def load_od_arc(arc, out_dir: Path, *, refresh: bool) -> OdArc | None:
    year = arc.date[:4]
    crd_url = EDC_CRD_URL.format(
        sat_key=arc.sat_key, year=year, date=arc.date
    )
    sp3_url = EDC_SP3_URL.format(
        sat_key=arc.sat_key, week=arc.sp3_week
    )
    crd_path = out_dir / f"{arc.sat_key}_{arc.date}.np2"
    sp3_path = out_dir / f"nsgf.orb.{arc.sat_key}.{arc.sp3_week}.v80.sp3.gz"
    crd_method = materialize(crd_url, crd_path, refresh=refresh)
    sp3_method = materialize(sp3_url, sp3_path, refresh=refresh)

    points = parse_crd_v2_normal_points(
        crd_path.read_text(encoding="utf-8", errors="replace")
    )
    eph = parse_sp3(_read_sp3_text(sp3_path), arc.sp3_sat_id)
    interp = GcrsInterp(eph, order=9)
    margin = 4.0 * 60.0
    points = [p for p in points if eph.covers(p.epoch_unix, margin_s=-margin)]
    if len(points) < 10:
        return None

    epochs = np.array([p.epoch_unix for p in points], dtype=np.float64)
    obs = [
        RangeObs(
            float(epochs[i]),
            _station_inertial(points[i].cdp_id, epochs[i]),
            float(points[i].range_m),
        )
        for i in range(len(points))
    ]
    n = len(obs)
    n_fit = min(max(6, int(np.floor(TRAIN_FRAC * n))), n - 3)
    fit_obs = obs[:n_fit]
    held = epochs[n_fit:]
    t0 = fit_obs[0].epoch_unix
    t_last = fit_obs[-1].epoch_unix
    x0 = interp.state_inertial(t0)
    p0 = np.diag(
        np.array(
            [INIT_POS_STD_M**2] * 3 + [INIT_VEL_STD_MPS**2] * 3,
            dtype=np.float64,
        )
    )
    provenance = {
        "sp3_analysis_center": ANALYSIS_CENTER,
        "sp3_week": arc.sp3_week,
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
    return OdArc(
        arc_id=f"{arc.target} {arc.date}",
        target=arc.target,
        date=arc.date,
        split=arc.split,
        fit_obs=fit_obs,
        held_epochs=held,
        t0=t0,
        fit_last_epoch=t_last,
        x0=x0,
        p0=p0,
        interp=interp,
        num_observations=n,
        num_fit=int(n_fit),
        num_held=int(n - n_fit),
        provenance=provenance,
    )


def run_range_ukf_fixed_hifi_corrected(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    calib: HifiCalibrator,
    max_step_s: float = SLR_MAX_STEP_S,
) -> dict:
    """Fixed-noise range-only UKF with higher-fidelity learned correction."""
    n = 6
    x = np.asarray(x0, dtype=np.float64).copy()
    p = np.asarray(p0, dtype=np.float64).copy()
    alpha, beta, kappa = 1e-3, 2.0, 0.0
    wm, wc, lam = _ukf_weights(n, alpha, beta, kappa)
    r_nom = float(range_std_m) ** 2
    records: list[dict] = []
    t_prev = obs[0].epoch_unix
    for o in obs:
        dt = o.epoch_unix - t_prev
        if dt != 0.0:
            pts = _sigma_points(x, p, lam)
            prop = np.vstack(
                [
                    propagate_hifi_corrected(pt, dt, t_prev, calib, max_step_s)
                    for pt in pts
                ]
            )
            x = wm @ prop
            dx = prop - x
            p = (wc[:, None, None] * dx[:, :, None] * dx[:, None, :]).sum(0)
            p = p + _process_cov(dt, accel_psd)
            p = 0.5 * (p + p.T)
            t_prev = o.epoch_unix

        pts = _sigma_points(x, p, lam)
        z_pred = np.array(
            [np.linalg.norm(pt[:3] - o.station_pi_m) for pt in pts]
        )
        z_mean = float(wm @ z_pred)
        dz = z_pred - z_mean
        innov = o.range_m - z_mean
        pzz = float((wc * dz) @ dz) + r_nom
        pxz = (wc[:, None] * (pts - x) * dz[:, None]).sum(0)
        k = pxz / pzz
        x = x + k * innov
        p = p - np.outer(k, k) * pzz
        p = 0.5 * (p + p.T)
        records.append(
            {
                "epoch_unix": o.epoch_unix,
                "innovation_m": innov,
                "nis_r": innov * innov / r_nom,
                "r_eff_scale": 1.0,
            }
        )
    return {"state": x, "cov": p, "records": records}


def _learned_rmse(arc: OdArc, calib: HifiCalibrator) -> float:
    res = run_range_ukf_fixed_hifi_corrected(
        arc.fit_obs,
        arc.x0,
        arc.p0,
        RANGE_STD_M,
        ACCEL_PSD,
        calib,
        SLR_MAX_STEP_S,
    )
    return held_out_rmse_corrected(
        res["state"],
        arc.fit_last_epoch,
        arc.held_epochs,
        arc.interp.pos_inertial,
        calib,
        SLR_MAX_STEP_S,
    )


def fit_train_week_calibrators(out_dir: Path, *, refresh: bool) -> dict:
    args = SimpleNamespace(refresh=refresh)
    week_objects = _week_objects(out_dir, args)
    train = [w for w in week_objects if w["split"] == "train"]
    if not train:
        raise RuntimeError("no training weeks available")

    phi_blocks = []
    da_blocks = []
    t0_ref = None
    per_week_samples = []
    for w in train:
        t0_ref = w["eph"].start_unix if t0_ref is None else t0_ref
        phi, da = residual_samples(
            w["interp"].pos_inertial,
            w["interp"].state_inertial,
            w["eph"].start_unix + CALIBRATOR_EDGE_MARGIN_S,
            w["eph"].end_unix - CALIBRATOR_EDGE_MARGIN_S,
            t0_ref,
            step_s=CALIBRATOR_SAMPLE_STEP_S,
        )
        per_week_samples.append(
            {
                "target": w["target"],
                "week": w["week"],
                "split": w["split"],
                "n_samples": int(phi.shape[0]),
            }
        )
        if phi.shape[0]:
            phi_blocks.append(phi)
            da_blocks.append(da)
    if not phi_blocks:
        raise RuntimeError("no residual samples from training weeks")

    phi_train = np.vstack(phi_blocks)
    da_train = np.vstack(da_blocks)
    calibrators = {
        _lambda_key(lam): fit_ridge(phi_train, da_train, lam, t0_ref)
        for lam in RIDGE_GRID
    }
    return {
        "calibrators": calibrators,
        "training_summary": {
            "training_weeks": [
                week
                for week, (split, _days) in SPLIT_WEEKS.items()
                if split == "train"
            ],
            "n_train_samples": int(phi_train.shape[0]),
            "feature_count": int(phi_train.shape[1]),
            "residual_sample_step_s": CALIBRATOR_SAMPLE_STEP_S,
            "ridge_grid": [_lambda_key(lam) for lam in RIDGE_GRID],
            "per_week_object_samples": per_week_samples,
            "sp3_used_for_fit_splits": ["train"],
        },
    }


def score_learned_rows(
    arcs: list[OdArc],
    calibrators: dict[str, HifiCalibrator],
    *,
    lambda_keys: list[str],
) -> tuple[list[dict], dict[str, float | None]]:
    rows = []
    for arc in arcs:
        values = {}
        for key in lambda_keys:
            values[key] = _round(_learned_rmse(arc, calibrators[key]))
        rows.append(
            {
                "arc_id": arc.arc_id,
                "target": arc.target,
                "date": arc.date,
                "split": arc.split,
                "num_observations": arc.num_observations,
                "num_fit": arc.num_fit,
                "num_held_out": arc.num_held,
                "learned_ridge_rmse_m": values,
            }
        )
    means = {
        key: _round(
            np.mean(
                [
                    row["learned_ridge_rmse_m"][key]
                    for row in rows
                    if row["learned_ridge_rmse_m"].get(key) is not None
                ]
            )
        )
        for key in lambda_keys
    }
    return rows, means


def _compose_test_rows(
    hifi: dict,
    learned_rows: list[dict],
    selected_lam_key: str,
) -> list[dict]:
    hifi_by_arc = _hifi_arc_lookup(hifi)
    out = []
    for lr in learned_rows:
        arc_id = lr["arc_id"]
        hrow = hifi_by_arc[arc_id]
        vals = {
            label: hrow["held_out_position_rmse_m"].get(label)
            for label in CLASSICAL_LABELS
        }
        vals[LEARNED_LABEL] = lr["learned_ridge_rmse_m"][selected_lam_key]
        out.append(
            {
                "arc_id": arc_id,
                "target": lr["target"],
                "date": lr["date"],
                "split": lr["split"],
                "num_observations": lr["num_observations"],
                "num_fit": lr["num_fit"],
                "num_held_out": lr["num_held_out"],
                "held_out_position_rmse_m": vals,
            }
        )
    return out


def _fmt(x) -> str:
    if x is None:
        return "--"
    return f"{float(x):.2f}"


def write_table(result: dict, path: Path) -> None:
    s = result["selection"]
    t = result["test_readout"]
    selected_gap = t["selected_vs_test_best_paired_gap"]
    learned_gap = t["learned_vs_best_classical_paired_gap"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Temporal public real-measurement orbit-determination "
            r"probe on LAGEOS CRD normal points. The learned residual "
            r"correction is fitted only on training weeks, all ridge and "
            r"candidate choices use only the validation week, and the frozen "
            r"choice is scored on the later test week against the independent "
            r"SP3 precise-orbit product. The probe is bounded by the small "
            r"LAGEOS-only corpus and the non-operational reduction model.}"
        ),
        r"  \label{tab:real_slr_sp3_temporal_od_campaign}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        (
            r"    Comparison & Validation-selected candidate & "
            r"Validation mean [m] & Test mean [m] & Paired test gap [m] \\"
        ),
        r"    \midrule",
        (
            "    Frozen temporal selector & "
            f"{s['selected_candidate']} & "
            f"{_fmt(s['selected_validation_mean_rms_m'])} & "
            f"{_fmt(t['selected_test_mean_rms_m'])} & "
            f"{_fmt(selected_gap.get('mean_gap_m'))} "
            f"[{_fmt(selected_gap.get('bootstrap95_mean_gap_m', [None, None])[0])}, "
            f"{_fmt(selected_gap.get('bootstrap95_mean_gap_m', [None, None])[1])}] \\\\"
        ),
        (
            "    Learned correction vs best classical & "
            f"{LEARNED_LABEL} & "
            f"{_fmt(s['validation_mean_rms_m'][LEARNED_LABEL])} & "
            f"{_fmt(t['test_mean_rms_m'][LEARNED_LABEL])} & "
            f"{_fmt(learned_gap.get('mean_gap_m'))} "
            f"[{_fmt(learned_gap.get('bootstrap95_mean_gap_m', [None, None])[0])}, "
            f"{_fmt(learned_gap.get('bootstrap95_mean_gap_m', [None, None])[1])}] \\\\"
        ),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        (
            r"  \\[2pt] {\footnotesize Gaps are candidate minus reference; "
            r"positive values mean larger held-out position RMSE. The "
            r"reference for the first row is the best test candidate in the "
            r"validation-defined pool; the reference for the second row is the "
            r"best classical test candidate.}"
        ),
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_result(args) -> dict:
    hifi = json.loads(args.hifi_json.read_text(encoding="utf-8"))
    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    cal = fit_train_week_calibrators(input_dir, refresh=args.refresh)
    calibrators: dict[str, HifiCalibrator] = cal["calibrators"]
    all_lambda_keys = [_lambda_key(lam) for lam in RIDGE_GRID]

    arcs = [load_od_arc(a, input_dir, refresh=args.refresh) for a in _arcs()]
    arcs = [a for a in arcs if a is not None]
    val_arcs = [a for a in arcs if a.split == "val"]
    test_arcs = [a for a in arcs if a.split == "test"]
    if not val_arcs or not test_arcs:
        raise RuntimeError("validation and test arcs are both required")

    val_learned_rows, val_learned_means = score_learned_rows(
        val_arcs, calibrators, lambda_keys=all_lambda_keys
    )
    selected_lam_key = select_lowest(val_learned_means)

    test_learned_rows, test_learned_means = score_learned_rows(
        test_arcs, calibrators, lambda_keys=[selected_lam_key]
    )
    learned_val_mean = val_learned_means[selected_lam_key]
    learned_test_mean = test_learned_means[selected_lam_key]

    validation_means = _classical_means_from_hifi(hifi, "val")
    validation_means[LEARNED_LABEL] = learned_val_mean
    test_means = _classical_means_from_hifi(hifi, "test")
    test_means[LEARNED_LABEL] = learned_test_mean

    selected_candidate = select_lowest(validation_means)
    test_best_candidate = select_lowest(test_means)
    best_classical_test = select_lowest(
        {label: test_means[label] for label in CLASSICAL_LABELS}
    )
    best_classical_validation = select_lowest(
        {label: validation_means[label] for label in CLASSICAL_LABELS}
    )

    test_rows = _compose_test_rows(hifi, test_learned_rows, selected_lam_key)
    selected_gap = paired_gap_summary(
        test_rows,
        selected_candidate,
        test_best_candidate,
        field="held_out_position_rmse_m",
    )
    learned_gap = paired_gap_summary(
        test_rows,
        LEARNED_LABEL,
        best_classical_test,
        field="held_out_position_rmse_m",
    )
    selected_vs_best_classical = paired_gap_summary(
        test_rows,
        selected_candidate,
        best_classical_test,
        field="held_out_position_rmse_m",
    )

    result = {
        "schema_version": "real_slr_sp3_temporal_od_campaign_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed",
        "source_artifacts": [
            {
                "artifact_id": args.hifi_json.as_posix(),
                "sha256": sha256_file(args.hifi_json),
            }
        ],
        "public_corpus": {
            "targets": hifi.get("targets", []),
            "num_sparse_slr_arcs": hifi.get("num_arcs_completed"),
            "sp3_analysis_center": hifi.get("sp3_analysis_center"),
            "split_weeks": hifi.get("split_weeks"),
            "normal_point_source": "public ILRS CRD v2 normal points",
            "state_reference": "independent ILRS analysis-centre SP3 precise orbit",
        },
        "predeclared": {
            "range_std_m": RANGE_STD_M,
            "accel_psd": ACCEL_PSD,
            "train_frac": TRAIN_FRAC,
            "slr_max_step_s": SLR_MAX_STEP_S,
            "learned_candidate": LEARNED_LABEL,
            "classical_candidate_pool": list(CLASSICAL_LABELS),
            "classical_pool_boundary": (
                "Best available sparse-SLR higher-fidelity public-corpus "
                "classical pool: compact UKF, higher-fidelity UKF, and "
                "higher-fidelity AUKF. This artifact does not claim a newly "
                "exhaustive EKF/UKF/AUKF sweep."
            ),
            "learned_ridge_grid": all_lambda_keys,
            "bootstrap_resamples": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "learned_calibrator_fit": cal["training_summary"],
        "selection_integrity": {
            "calibrator_fit_uses_only_train_weeks": True,
            "learned_ridge_selected_on_validation_only": True,
            "candidate_selected_on_validation_only": True,
            "test_set_information_used_for_selection": False,
            "train_weeks": [
                week
                for week, (split, _days) in SPLIT_WEEKS.items()
                if split == "train"
            ],
            "validation_week": "260502",
            "test_week": "260509",
            "selected_learned_ridge_lambda": selected_lam_key,
        },
        "validation": {
            "n_arcs": len(val_arcs),
            "learned_ridge_mean_rms_m": val_learned_means,
            "learned_rows": val_learned_rows,
        },
        "selection": {
            "selection_rule": (
                "choose the lowest validation-week mean arc RMSE after "
                "selecting the learned ridge on validation only"
            ),
            "validation_mean_rms_m": validation_means,
            "selected_candidate": selected_candidate,
            "selected_candidate_family": (
                "learned" if selected_candidate == LEARNED_LABEL else "classical"
            ),
            "selected_validation_mean_rms_m": validation_means[
                selected_candidate
            ],
            "best_classical_validation_candidate": best_classical_validation,
            "best_classical_validation_mean_rms_m": validation_means[
                best_classical_validation
            ],
            "selected_learned_ridge_lambda": selected_lam_key,
        },
        "test_readout": {
            "n_arcs": len(test_arcs),
            "test_mean_rms_m": test_means,
            "selected_candidate": selected_candidate,
            "selected_test_mean_rms_m": test_means[selected_candidate],
            "test_best_candidate": test_best_candidate,
            "test_best_mean_rms_m": test_means[test_best_candidate],
            "best_classical_test_candidate": best_classical_test,
            "best_classical_test_mean_rms_m": test_means[best_classical_test],
            "selected_vs_test_best_paired_gap": selected_gap,
            "selected_vs_best_classical_paired_gap": selected_vs_best_classical,
            "learned_vs_best_classical_paired_gap": learned_gap,
            "arcs": test_rows,
        },
        "headline_readout": {
            "selected_candidate": selected_candidate,
            "selected_candidate_family": (
                "learned" if selected_candidate == LEARNED_LABEL else "classical"
            ),
            "selected_test_mean_rms_m": test_means[selected_candidate],
            "test_best_candidate": test_best_candidate,
            "test_best_mean_rms_m": test_means[test_best_candidate],
            "learned_test_mean_rms_m": test_means[LEARNED_LABEL],
            "best_classical_test_mean_rms_m": test_means[best_classical_test],
            "learned_minus_best_classical_mean_gap_m": learned_gap.get(
                "mean_gap_m"
            ),
            "learned_positive_against_best_classical": bool(
                learned_gap.get("mean_gap_m") is not None
                and learned_gap["mean_gap_m"] < 0.0
                and learned_gap["bootstrap95_mean_gap_m"][1] is not None
                and learned_gap["bootstrap95_mean_gap_m"][1] < 0.0
            ),
            "paper_strength_class": "bounded_public_real_measurement_od_probe",
        },
        "claim_boundary": {
            "defensible_status": "bounded_public_real_measurement_od_probe",
            "scientifically_defensible_40_arc_temporal_campaign_feasible": True,
            "central_external_validation_status": False,
            "can_be_used_as_central_external_validation": False,
            "can_be_used_as_bounded_public_real_measurement_od_probe": True,
            "is_operational_validation": False,
            "is_centimetre_slr_or_flight_validation": False,
            "is_powered_confirmatory_campaign": False,
            "does_not_relabel_provenance_as_validation": True,
            "why_not_central_external_validation": [
                (
                    "The corpus is small, LAGEOS-only, and uses one SP3 "
                    "analysis-centre precise-orbit product."
                ),
                (
                    "The classical reference set is the best available "
                    "sparse-SLR hifi pool in the existing public-corpus "
                    "artifact, not a fresh exhaustive classical sweep."
                ),
                (
                    "The OD reduction remains bounded-fidelity rather than an "
                    "operational precise-SLR reduction."
                ),
                (
                    "No independent prospective power calculation was used to "
                    "size the endpoint before observing the corpus."
                ),
                (
                    "The learned candidate is a lightweight empirical residual "
                    "correction, not a broad learned-OD system replication."
                ),
            ],
            "appropriate_use": (
                "Use as a no-test-leakage public real-measurement OD probe "
                "scored against independent SP3 states, not as central "
                "operational validation."
            ),
        },
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hifi-json", type=Path, default=DEFAULT_HIFI_JSON)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-table", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = build_result(args)
    dump_json(result, args.output_json)
    if not args.no_table:
        write_table(result, args.table)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_json": str(args.output_json),
                "selected_candidate": result["headline_readout"][
                    "selected_candidate"
                ],
                "selected_test_mean_rms_m": result["headline_readout"][
                    "selected_test_mean_rms_m"
                ],
                "test_best_candidate": result["headline_readout"][
                    "test_best_candidate"
                ],
                "learned_test_mean_rms_m": result["headline_readout"][
                    "learned_test_mean_rms_m"
                ],
                "best_classical_test_mean_rms_m": result["headline_readout"][
                    "best_classical_test_mean_rms_m"
                ],
                "paper_strength_class": result["headline_readout"][
                    "paper_strength_class"
                ],
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
