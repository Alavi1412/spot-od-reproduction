#!/usr/bin/env python
"""Expanded compact real SLR/SP3 OD validation over hifi-derived schedules.

This additive runner reuses the compact bounded-fidelity OD implementation in
``run_real_slr_sp3_od_validation.py`` and changes only the campaign schedule
and summary aggregation.  The default output is separate from the committed
10-arc compact slice, so existing compact-slice artifacts are left untouched.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - entrypoint dependent
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gnn_state_estimation.slr import SUPPORTED_STATIONS
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

try:  # noqa: E402 - import context depends on script vs package execution
    from run_real_slr_sp3_hifi_validation import SPLIT_WEEKS
    from run_real_slr_sp3_od_validation import (
        ACCEL_PSD,
        ANALYSIS_CENTER,
        INIT_POS_STD_M,
        INIT_VEL_STD_MPS,
        EDC_CRD_URL,
        EDC_SP3_URL,
        MATERIALITY_MARGIN,
        MAX_STEP_S,
        RANGE_STD_M,
        TAU_R,
        TAU_RHO,
        TRAIN_FRAC,
        Arc,
        _dbar_summary_from_arcs,
        run_arc,
    )
except ModuleNotFoundError:  # pragma: no cover - pytest/package import path
    from scripts.run_real_slr_sp3_hifi_validation import SPLIT_WEEKS
    from scripts.run_real_slr_sp3_od_validation import (
        ACCEL_PSD,
        ANALYSIS_CENTER,
        INIT_POS_STD_M,
        INIT_VEL_STD_MPS,
        EDC_CRD_URL,
        EDC_SP3_URL,
        MATERIALITY_MARGIN,
        MAX_STEP_S,
        RANGE_STD_M,
        TAU_R,
        TAU_RHO,
        TRAIN_FRAC,
        Arc,
        _dbar_summary_from_arcs,
        run_arc,
    )


DEFAULT_OUT_DIR = Path("results/real_slr_sp3_od_expanded")
OUTPUT_FILENAME = "real_slr_sp3_od_expanded_validation.json"
EXTENDED_OUTPUT_FILENAME = "real_slr_sp3_od_expanded80_validation.json"
FORMAL_POWER_OUTPUT_FILENAME = "real_slr_sp3_od_formal190_validation.json"
FORMAL210_OUTPUT_FILENAME = "real_slr_sp3_od_formal210_validation.json"
FORMAL400_OUTPUT_FILENAME = "real_slr_sp3_od_formal400_validation.json"
DEFAULT_OUTPUT_JSON = DEFAULT_OUT_DIR / OUTPUT_FILENAME
DEFAULT_SCHEDULE = "hifi40"
EXTENDED_SCHEDULE = "extended80"
FORMAL_POWER_SCHEDULE = "formal190"
FORMAL210_SCHEDULE = "formal210"
FORMAL400_SCHEDULE = "formal400"

ESTIMATORS = (
    "EKF",
    "UKF (fixed-noise)",
    "AUKF (adaptive)",
    "SP3-IC propagation",
)
PAIRED_COMPARISONS = (
    ("EKF", "AUKF (adaptive)"),
    ("UKF (fixed-noise)", "AUKF (adaptive)"),
    ("EKF", "UKF (fixed-noise)"),
)
TARGETS = (
    ("LAGEOS-1", "lageos1", "L51"),
    ("LAGEOS-2", "lageos2", "L52"),
)
PRECEDING_WEEKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "260321": (
        "preceding",
        ("20260316", "20260317", "20260318", "20260319", "20260320"),
    ),
    "260328": (
        "preceding",
        ("20260323", "20260324", "20260325", "20260326", "20260327"),
    ),
    "260404": (
        "preceding",
        ("20260330", "20260331", "20260401", "20260402", "20260403"),
    ),
    "260411": (
        "preceding",
        ("20260406", "20260407", "20260408", "20260409", "20260410"),
    ),
}
FORMAL_POWER_ADDED_WEEKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "260103": (
        "preceding",
        ("20251229", "20251230", "20251231", "20260101", "20260102"),
    ),
    "260110": (
        "preceding",
        ("20260105", "20260106", "20260107", "20260108", "20260109"),
    ),
    "260117": (
        "preceding",
        ("20260112", "20260113", "20260114", "20260115", "20260116"),
    ),
    "260124": (
        "preceding",
        ("20260119", "20260120", "20260121", "20260122", "20260123"),
    ),
    "260131": (
        "preceding",
        ("20260126", "20260127", "20260128", "20260129", "20260130"),
    ),
    "260207": (
        "preceding",
        ("20260202", "20260203", "20260204", "20260205", "20260206"),
    ),
    "260214": (
        "preceding",
        ("20260209", "20260210", "20260211", "20260212", "20260213"),
    ),
    "260221": (
        "preceding",
        ("20260216", "20260217", "20260218", "20260219", "20260220"),
    ),
    "260228": (
        "preceding",
        ("20260223", "20260224", "20260225", "20260226", "20260227"),
    ),
    "260307": (
        "preceding",
        ("20260302", "20260303", "20260304", "20260305", "20260306"),
    ),
    "260314": (
        "preceding",
        ("20260309", "20260310", "20260311", "20260312", "20260313"),
    ),
}
FORMAL210_EXTRA_WEEKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "251220": (
        "preceding",
        ("20251215", "20251216", "20251217", "20251218", "20251219"),
    ),
    "251227": (
        "preceding",
        ("20251222", "20251223", "20251224", "20251225", "20251226"),
    ),
}
FORMAL400_EXTRA_WEEKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "250809": (
        "preceding",
        ("20250804", "20250805", "20250806", "20250807", "20250808"),
    ),
    "250816": (
        "preceding",
        ("20250811", "20250812", "20250813", "20250814", "20250815"),
    ),
    "250823": (
        "preceding",
        ("20250818", "20250819", "20250820", "20250821", "20250822"),
    ),
    "250830": (
        "preceding",
        ("20250825", "20250826", "20250827", "20250828", "20250829"),
    ),
    "250906": (
        "preceding",
        ("20250901", "20250902", "20250903", "20250904", "20250905"),
    ),
    "250913": (
        "preceding",
        ("20250908", "20250909", "20250910", "20250911", "20250912"),
    ),
    "250920": (
        "preceding",
        ("20250915", "20250916", "20250917", "20250918", "20250919"),
    ),
    "250927": (
        "preceding",
        ("20250922", "20250923", "20250924", "20250925", "20250926"),
    ),
    "251004": (
        "preceding",
        ("20250929", "20250930", "20251001", "20251002", "20251003"),
    ),
    "251011": (
        "preceding",
        ("20251006", "20251007", "20251008", "20251009", "20251010"),
    ),
    "251018": (
        "preceding",
        ("20251013", "20251014", "20251015", "20251016", "20251017"),
    ),
    "251025": (
        "preceding",
        ("20251020", "20251021", "20251022", "20251023", "20251024"),
    ),
    "251101": (
        "preceding",
        ("20251027", "20251028", "20251029", "20251030", "20251031"),
    ),
    "251108": (
        "preceding",
        ("20251103", "20251104", "20251105", "20251106", "20251107"),
    ),
    "251115": (
        "preceding",
        ("20251110", "20251111", "20251112", "20251113", "20251114"),
    ),
    "251122": (
        "preceding",
        ("20251117", "20251118", "20251119", "20251120", "20251121"),
    ),
    "251129": (
        "preceding",
        ("20251124", "20251125", "20251126", "20251127", "20251128"),
    ),
    "251206": (
        "preceding",
        ("20251201", "20251202", "20251203", "20251204", "20251205"),
    ),
    "251213": (
        "preceding",
        ("20251208", "20251209", "20251210", "20251211", "20251212"),
    ),
}
SCHEDULE_OUTPUT_FILENAMES = {
    DEFAULT_SCHEDULE: OUTPUT_FILENAME,
    EXTENDED_SCHEDULE: EXTENDED_OUTPUT_FILENAME,
    FORMAL_POWER_SCHEDULE: FORMAL_POWER_OUTPUT_FILENAME,
    FORMAL210_SCHEDULE: FORMAL210_OUTPUT_FILENAME,
    FORMAL400_SCHEDULE: FORMAL400_OUTPUT_FILENAME,
}
SCHEDULE_DESCRIPTIONS = {
    DEFAULT_SCHEDULE: "hifi 40-arc temporal schedule",
    EXTENDED_SCHEDULE: (
        "extended 80-arc temporal schedule including four preceding weeks"
    ),
    FORMAL_POWER_SCHEDULE: (
        "formal-power-scale 190-arc extension adding 11 earlier weeks before "
        "the extended80 schedule"
    ),
    FORMAL210_SCHEDULE: (
        "formal-power-scale 210-arc extension adding 13 earlier weeks before "
        "the extended80 schedule"
    ),
    FORMAL400_SCHEDULE: (
        "formal-power-scale 400-arc extension adding 32 earlier weeks before "
        "the extended80 schedule"
    ),
}
BOOTSTRAP_SEED = 20260526
BOOTSTRAP_RESAMPLES = 5000


def schedule_weeks(
    schedule_name: str = DEFAULT_SCHEDULE,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    if schedule_name == DEFAULT_SCHEDULE:
        return dict(SPLIT_WEEKS)
    if schedule_name == EXTENDED_SCHEDULE:
        weeks: dict[str, tuple[str, tuple[str, ...]]] = dict(PRECEDING_WEEKS)
        weeks.update(SPLIT_WEEKS)
        return weeks
    if schedule_name == FORMAL_POWER_SCHEDULE:
        weeks = dict(FORMAL_POWER_ADDED_WEEKS)
        weeks.update(PRECEDING_WEEKS)
        weeks.update(SPLIT_WEEKS)
        return weeks
    if schedule_name == FORMAL210_SCHEDULE:
        weeks = dict(FORMAL210_EXTRA_WEEKS)
        weeks.update(FORMAL_POWER_ADDED_WEEKS)
        weeks.update(PRECEDING_WEEKS)
        weeks.update(SPLIT_WEEKS)
        return weeks
    if schedule_name == FORMAL400_SCHEDULE:
        weeks = dict(FORMAL400_EXTRA_WEEKS)
        weeks.update(FORMAL210_EXTRA_WEEKS)
        weeks.update(FORMAL_POWER_ADDED_WEEKS)
        weeks.update(PRECEDING_WEEKS)
        weeks.update(SPLIT_WEEKS)
        return weeks
    raise ValueError(f"unknown schedule: {schedule_name}")


def build_expanded_arcs(schedule_name: str = DEFAULT_SCHEDULE) -> list[Arc]:
    """Return the selected weekly schedule as compact arcs."""
    arcs: list[Arc] = []
    for week, (_split, dates) in schedule_weeks(schedule_name).items():
        for target, sat_key, sp3_sat_id in TARGETS:
            for date in dates:
                arcs.append(Arc(target, sat_key, sp3_sat_id, date, week))
    return arcs


def split_by_arc_id(schedule_name: str = DEFAULT_SCHEDULE) -> dict[str, str]:
    """Map ``LAGEOS-N YYYYMMDD`` arc ids to the schedule split label."""
    out: dict[str, str] = {}
    for _week, (split, dates) in schedule_weeks(schedule_name).items():
        for target, _sat_key, _sp3_sat_id in TARGETS:
            for date in dates:
                out[f"{target} {date}"] = split
    return out


def schedule_metadata(schedule_name: str = DEFAULT_SCHEDULE) -> dict:
    weeks = schedule_weeks(schedule_name)
    if schedule_name == DEFAULT_SCHEDULE:
        return {
            "source": "scripts/run_real_slr_sp3_hifi_validation.py:SPLIT_WEEKS",
            "num_weeks": len(weeks),
            "dates_per_week": {
                week: len(dates) for week, (_split, dates) in weeks.items()
            },
            "targets_per_date": len(TARGETS),
            "weeks": {
                week: {"split": split, "dates": list(dates)}
                for week, (split, dates) in weeks.items()
            },
        }
    if schedule_name == EXTENDED_SCHEDULE:
        added_weeks = list(PRECEDING_WEEKS)
        base_schedule_name = DEFAULT_SCHEDULE
        source = (
            "scripts/run_real_slr_sp3_hifi_validation.py:SPLIT_WEEKS "
            "plus PRECEDING_WEEKS in this runner"
        )
        extra_metadata = {}
    elif schedule_name == FORMAL_POWER_SCHEDULE:
        formal_power_added_weeks = list(FORMAL_POWER_ADDED_WEEKS)
        added_weeks = formal_power_added_weeks
        base_schedule_name = EXTENDED_SCHEDULE
        source = (
            "scripts/run_real_slr_sp3_hifi_validation.py:SPLIT_WEEKS "
            "plus PRECEDING_WEEKS and FORMAL_POWER_ADDED_WEEKS in this runner"
        )
        extra_metadata = {
            "formal_power_added_weeks": formal_power_added_weeks,
            "num_formal_power_added_weeks": len(formal_power_added_weeks),
            "extended80_weeks": list(PRECEDING_WEEKS) + list(SPLIT_WEEKS),
            "num_extended80_weeks": len(PRECEDING_WEEKS) + len(SPLIT_WEEKS),
        }
    elif schedule_name == FORMAL210_SCHEDULE:
        formal210_extra_weeks = list(FORMAL210_EXTRA_WEEKS)
        formal_power_added_weeks = list(FORMAL_POWER_ADDED_WEEKS)
        formal210_added_weeks = formal210_extra_weeks + formal_power_added_weeks
        added_weeks = formal210_added_weeks
        base_schedule_name = EXTENDED_SCHEDULE
        source = (
            "scripts/run_real_slr_sp3_hifi_validation.py:SPLIT_WEEKS "
            "plus PRECEDING_WEEKS, FORMAL_POWER_ADDED_WEEKS, and "
            "FORMAL210_EXTRA_WEEKS in this runner"
        )
        extra_metadata = {
            "formal210_extra_weeks": formal210_extra_weeks,
            "num_formal210_extra_weeks": len(formal210_extra_weeks),
            "formal210_added_weeks": formal210_added_weeks,
            "num_formal210_added_weeks": len(formal210_added_weeks),
            "formal_power_added_weeks": formal_power_added_weeks,
            "num_formal_power_added_weeks": len(formal_power_added_weeks),
            "extended80_weeks": list(PRECEDING_WEEKS) + list(SPLIT_WEEKS),
            "num_extended80_weeks": len(PRECEDING_WEEKS) + len(SPLIT_WEEKS),
        }
    elif schedule_name == FORMAL400_SCHEDULE:
        formal400_extra_weeks = list(FORMAL400_EXTRA_WEEKS)
        formal210_extra_weeks = list(FORMAL210_EXTRA_WEEKS)
        formal_power_added_weeks = list(FORMAL_POWER_ADDED_WEEKS)
        formal400_added_weeks = (
            formal400_extra_weeks + formal210_extra_weeks + formal_power_added_weeks
        )
        added_weeks = formal400_added_weeks
        base_schedule_name = EXTENDED_SCHEDULE
        source = (
            "scripts/run_real_slr_sp3_hifi_validation.py:SPLIT_WEEKS "
            "plus PRECEDING_WEEKS, FORMAL_POWER_ADDED_WEEKS, "
            "FORMAL210_EXTRA_WEEKS, and FORMAL400_EXTRA_WEEKS in this runner"
        )
        extra_metadata = {
            "formal400_extra_weeks": formal400_extra_weeks,
            "num_formal400_extra_weeks": len(formal400_extra_weeks),
            "formal400_added_weeks": formal400_added_weeks,
            "num_formal400_added_weeks": len(formal400_added_weeks),
            "formal210_extra_weeks": formal210_extra_weeks,
            "num_formal210_extra_weeks": len(formal210_extra_weeks),
            "formal_power_added_weeks": formal_power_added_weeks,
            "num_formal_power_added_weeks": len(formal_power_added_weeks),
            "extended80_weeks": list(PRECEDING_WEEKS) + list(SPLIT_WEEKS),
            "num_extended80_weeks": len(PRECEDING_WEEKS) + len(SPLIT_WEEKS),
        }
    else:
        raise ValueError(f"unknown schedule: {schedule_name}")
    return {
        "schedule_name": schedule_name,
        "description": SCHEDULE_DESCRIPTIONS[schedule_name],
        "source": source,
        "base_schedule_name": base_schedule_name,
        "added_weeks": added_weeks,
        "num_added_weeks": len(added_weeks),
        "num_weeks": len(weeks),
        "dates_per_week": {
            week: len(dates) for week, (_split, dates) in weeks.items()
        },
        "targets_per_date": len(TARGETS),
        "weeks": {
            week: {"split": split, "dates": list(dates)}
            for week, (split, dates) in weeks.items()
        },
        **extra_metadata,
    }


def _round_stat(value: float | None, ndigits: int = 2):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), ndigits)


def _arc_rmse(arc: dict, estimator: str) -> float | None:
    detail = arc.get("held_out_detail", {})
    if isinstance(detail, dict) and estimator in detail:
        value = detail[estimator].get("rms_m")
    else:
        value = arc.get("held_out_position_rmse_m", {}).get(estimator)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def pooled_held_out_rmse(completed: list[dict]) -> dict:
    """Per-estimator arc-RMSE summary over completed arcs."""
    arc_best_count = {name: 0 for name in ESTIMATORS}
    for arc in completed:
        best = arc.get("best_held_out_estimator")
        if best in arc_best_count:
            arc_best_count[best] += 1

    pooled: dict[str, dict] = {}
    for estimator in ESTIMATORS:
        values = [
            value
            for value in (_arc_rmse(arc, estimator) for arc in completed)
            if value is not None
        ]
        arr = np.asarray(values, dtype=np.float64)
        pooled[estimator] = {
            "completed_arc_count": len(completed),
            "n_arcs": int(arr.size),
            "mean_arc_rms_m": _round_stat(arr.mean() if arr.size else None),
            "std_arc_rms_m": _round_stat(
                arr.std(ddof=1) if arr.size > 1 else (0.0 if arr.size else None)
            ),
            "median_arc_rms_m": _round_stat(
                np.median(arr) if arr.size else None
            ),
            "arc_best_count": arc_best_count[estimator],
        }
    return pooled


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, resamples: int
) -> list[float | None]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(resamples), arr.size))
    means = arr[idx].mean(axis=1)
    return [
        round(float(np.percentile(means, 2.5)), 2),
        round(float(np.percentile(means, 97.5)), 2),
    ]


def paired_difference_summary(
    completed: list[dict],
    first: str,
    second: str,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    """Paired first-minus-second RMSE; positive means first is worse."""
    diffs: list[float] = []
    for arc in completed:
        a = _arc_rmse(arc, first)
        b = _arc_rmse(arc, second)
        if a is not None and b is not None:
            diffs.append(a - b)
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.size == 0:
        return {
            "first_method": first,
            "second_method": second,
            "n": 0,
            "seed": int(seed),
            "resamples": int(resamples),
            "sign_convention": (
                "first_minus_second; positive means the first method has "
                "larger held-out RMSE than the second method"
            ),
        }
    return {
        "first_method": first,
        "second_method": second,
        "n": int(arr.size),
        "mean_difference_m": _round_stat(arr.mean()),
        "median_difference_m": _round_stat(np.median(arr)),
        "bootstrap95_mean_difference_m": _bootstrap_mean_ci(
            arr, seed=seed, resamples=resamples
        ),
        "seed": int(seed),
        "resamples": int(resamples),
        "n_first_larger_rmse": int((arr > 0.0).sum()),
        "fraction_first_larger_rmse": round(float((arr > 0.0).mean()), 3),
        "sign_convention": (
            "first_minus_second; positive means the first method has larger "
            "held-out RMSE than the second method"
        ),
    }


def paired_difference_summaries(
    completed: list[dict],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    return {
        f"{first} minus {second}": paired_difference_summary(
            completed, first, second, seed=seed, resamples=resamples
        )
        for first, second in PAIRED_COMPARISONS
    }


def input_digest_records(arc_blocks: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for arc in arc_blocks:
        for kind in ("crd", "sp3"):
            block = arc.get(kind)
            if isinstance(block, dict) and "sha256" in block:
                rows.append(
                    {
                        "arc_id": arc.get("arc_id"),
                        "kind": kind,
                        "archived_input_id": block.get("archived_input_id"),
                        "sha256": block.get("sha256"),
                        "bytes": block.get("bytes"),
                        "url": block.get("url"),
                    }
                )
    return rows


def serializable_arcs(arc_blocks: list[dict]) -> list[dict]:
    """Drop verbose per-epoch held-out detail, matching the compact artifact."""
    return [
        {key: value for key, value in arc.items() if key != "held_out_detail"}
        for arc in arc_blocks
    ]


def caveat_metadata() -> dict:
    return {
        "public_bounded_reference_replay": True,
        "reference": (
            "held-out state error against an independent ILRS analysis-centre "
            "SP3-c precise orbit product"
        ),
        "compact_gmst_only_frame": True,
        "compact_frame_description": (
            "GMST-only pseudo-inertial frame shared by station geometry and "
            "SP3 reference, without polar motion, nutation, or UT1-UTC EOP"
        ),
        "full_operational_slr_correction_stack": False,
        "omitted_operational_corrections": [
            "tropospheric delay",
            "relativistic corrections",
            "centre-of-mass corrections",
            "solid-Earth tides",
            "full EOP/polar-motion stack",
        ],
        "operational_pod": False,
        "not_operational_pod": True,
        "interpretation": (
            "bounded public replay for relative compact-filter behavior; "
            "absolute magnitudes are model-mismatch stress numbers, not "
            "centimetre operational POD accuracy"
        ),
    }


def empty_dbar_summary() -> dict:
    return {
        "n_arcs_scored": 0,
        "n_correct": 0,
        "external_label_definition": (
            "adaptation counterproductive iff the adaptive UKF held-out SP3 "
            "position RMSE exceeds the fixed-noise UKF held-out SP3 position "
            "RMSE by more than the predeclared 5% margin (external precise "
            "reference; not the DBAR statistic, not a simulator label)"
        ),
        "classification_accuracy": None,
        "confusion": {
            "true_fire": 0,
            "true_no_fire": 0,
            "false_fire": 0,
            "false_no_fire": 0,
        },
        "sensitivity": None,
        "specificity": None,
        "n_counterproductive_arcs": 0,
        "n_non_counterproductive_arcs": 0,
        "classification_report": {
            "n": 0,
            "note": "not computed because no arcs completed",
        },
        "no_information_baseline": None,
        "incremental_accuracy_over_majority_class": None,
        "beats_trivial_majority_classifier": False,
    }


def build_result(
    arc_blocks: list[dict],
    *,
    schedule_name: str = DEFAULT_SCHEDULE,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict:
    weeks = schedule_weeks(schedule_name)
    completed = [arc for arc in arc_blocks if arc.get("status") == "completed"]
    n_total = len(arc_blocks)
    if len(completed) == n_total and n_total:
        status = "completed"
    elif completed:
        status = "partial_completed"
    else:
        status = "insufficient_observations"
    dbar_summary = (
        _dbar_summary_from_arcs(completed) if completed else empty_dbar_summary()
    )

    caveats = (
        "Public bounded-reference replay of the compact real SLR/SP3 OD "
        f"slice over the {SCHEDULE_DESCRIPTIONS[schedule_name]}. "
        "The compact OD uses a "
        "GMST-only frame and compact two-body+J2 dynamics, does not apply the "
        "full operational SLR correction stack, and is not operational POD."
    )

    result = {
        "schema_version": "real_slr_sp3_od_expanded_v1",
        "generated_utc": utc_now_iso(),
        "status": status,
        "targets": sorted(
            {
                arc.get("target")
                for arc in arc_blocks
                if isinstance(arc.get("target"), str)
            }
        ),
        "num_arcs": n_total,
        "num_arcs_completed": len(completed),
        "sp3_analysis_center": ANALYSIS_CENTER,
        "sp3_week_products": list(weeks.keys()),
        "split_weeks": {week: split for week, (split, _dates) in weeks.items()},
        "schedule": schedule_metadata(schedule_name),
        "fixed_station_subset": sorted(
            {station.code for station in SUPPORTED_STATIONS.values()}
        ),
        "predeclared": {
            "tau_r_eff": TAU_R,
            "tau_rho_nis": TAU_RHO,
            "materiality_margin": MATERIALITY_MARGIN,
            "range_std_m": RANGE_STD_M,
            "accel_psd": ACCEL_PSD,
            "init_pos_std_m": INIT_POS_STD_M,
            "init_vel_std_mps": INIT_VEL_STD_MPS,
            "train_frac": TRAIN_FRAC,
            "max_step_s": MAX_STEP_S,
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_resamples": int(bootstrap_resamples),
        },
        "arcs": serializable_arcs(arc_blocks),
        "pooled_held_out_position_rmse_m": pooled_held_out_rmse(completed),
        "paired_differences": paired_difference_summaries(
            completed, seed=bootstrap_seed, resamples=bootstrap_resamples
        ),
        "dbar_external_validation": dbar_summary,
        "input_digests": input_digest_records(arc_blocks),
        "caveat_metadata": caveat_metadata(),
        "caveats": caveats,
    }
    if schedule_name != DEFAULT_SCHEDULE:
        result["schedule_name"] = schedule_name
    return result


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--schedule",
        choices=(
            DEFAULT_SCHEDULE,
            EXTENDED_SCHEDULE,
            FORMAL_POWER_SCHEDULE,
            FORMAL210_SCHEDULE,
            FORMAL400_SCHEDULE,
        ),
        default=DEFAULT_SCHEDULE,
        help=(
            "Campaign schedule to run. The default hifi40 preserves the "
            "reviewed 40-arc artifact; extended80 adds the four preceding "
            "weeks; formal190 adds 11 earlier weeks before extended80; "
            "formal210 adds 13 earlier weeks before extended80; "
            "formal400 adds 32 earlier weeks before extended80. Non-default "
            "schedules use distinct default output filenames."
        ),
    )
    parser.add_argument(
        "--include-preceding-weeks",
        dest="schedule",
        action="store_const",
        const=EXTENDED_SCHEDULE,
        help="Alias for --schedule extended80.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Output JSON path. Defaults to a schedule-specific filename "
            "under --out-dir."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-download public CRD/SP3 inputs. By default existing local "
            "inputs in --out-dir are reused offline."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected schedule and exit without running OD.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help="Seed for paired-difference bootstrap intervals.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=_positive_int,
        default=BOOTSTRAP_RESAMPLES,
        help="Number of paired-difference bootstrap resamples.",
    )
    return parser


def output_json_path(args: argparse.Namespace) -> Path:
    if args.output_json:
        return Path(args.output_json)
    return Path(args.out_dir) / SCHEDULE_OUTPUT_FILENAMES[args.schedule]


def _input_urls_for_arc(arc: Arc) -> tuple[str, str]:
    year = arc.date[:4]
    crd_url = EDC_CRD_URL.format(
        sat_key=arc.sat_key, year=year, date=arc.date
    )
    sp3_url = EDC_SP3_URL.format(sat_key=arc.sat_key, week=arc.sp3_week)
    return crd_url, sp3_url


def failed_arc_block(arc: Arc, exc: Exception) -> dict:
    crd_url, sp3_url = _input_urls_for_arc(arc)
    message = str(exc)
    status = "input_unavailable" if "Failed to download" in message else "arc_failed"
    return {
        "arc_id": f"{arc.target} {arc.date}",
        "target": arc.target,
        "date": arc.date,
        "sp3_week": arc.sp3_week,
        "sp3_analysis_center": ANALYSIS_CENTER,
        "sp3_satellite_id": arc.sp3_sat_id,
        "crd": {
            "url": crd_url,
            "archived_input_id": f"{arc.sat_key}_{arc.date}.np2",
        },
        "sp3": {
            "url": sp3_url,
            "archived_input_id": (
                f"nsgf.orb.{arc.sat_key}.{arc.sp3_week}.v80.sp3.gz"
            ),
        },
        "status": status,
        "error_type": type(exc).__name__,
        "error_message": message,
    }


def main() -> int:
    args = build_parser().parse_args()
    arcs = build_expanded_arcs(args.schedule)
    split_lookup = split_by_arc_id(args.schedule)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "output_json": str(output_json_path(args)),
                    "num_arcs": len(arcs),
                    "schedule": schedule_metadata(args.schedule),
                    "arcs": [
                        {
                            **asdict(arc),
                            "arc_id": f"{arc.target} {arc.date}",
                            "split": split_lookup[f"{arc.target} {arc.date}"],
                        }
                        for arc in arcs
                    ],
                },
                indent=2,
            )
        )
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arc_blocks: list[dict] = []
    for index, arc in enumerate(arcs, start=1):
        arc_id = f"{arc.target} {arc.date}"
        try:
            block = run_arc(arc, out_dir, args)
        except Exception as exc:  # noqa: BLE001 - preserve campaign progress.
            block = failed_arc_block(arc, exc)
        block["split"] = split_lookup[arc_id]
        block["campaign_arc_index"] = index
        arc_blocks.append(block)

    result = build_result(
        arc_blocks,
        schedule_name=args.schedule,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    output_path = output_json_path(args)
    if result["num_arcs_completed"] == 0:
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output_json": str(output_path),
                    "output_written": False,
                    "reason": "all arcs failed or had insufficient observations",
                    "num_arcs": result["num_arcs"],
                    "num_arcs_completed": result["num_arcs_completed"],
                    "arc_status_counts": {
                        status: sum(
                            1
                            for arc in arc_blocks
                            if arc.get("status") == status
                        )
                        for status in sorted(
                            {str(arc.get("status")) for arc in arc_blocks}
                        )
                    },
                    "dbar_external_validation": result[
                        "dbar_external_validation"
                    ],
                },
                indent=2,
            )
        )
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(result, output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_json": str(output_path),
                "num_arcs": result["num_arcs"],
                "num_arcs_completed": result["num_arcs_completed"],
                "pooled_held_out_position_rmse_m": result[
                    "pooled_held_out_position_rmse_m"
                ],
                "paired_differences": result["paired_differences"],
                "dbar_external_validation": result["dbar_external_validation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
