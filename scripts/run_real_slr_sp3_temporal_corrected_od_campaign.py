#!/usr/bin/env python
"""Temporal full-correction public CRD/SP3 OD campaign.

This runner is the full-correction counterpart to
``run_real_slr_sp3_temporal_od_campaign.py``.  It applies the same
public-IERS/full-SLR-reduction stack as
``run_real_slr_sp3_corrected_validation.py`` (EOP, troposphere,
centre-of-mass, and relativistic range corrections), keeps a frozen temporal
selection rule, and adds a lightweight learned residual-acceleration
candidate trained only on predeclared training-week SP3 states.

The default schedule is intentionally labelled post-hoc robustness: the later
EDC week was not reachable when this experiment was authored, so the default
uses archived formal210-style weeks while excluding the already-scored
260509 temporal-test week from selection and scoring.
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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gnn_state_estimation.eop import EopSeries, load_eop_series
from gnn_state_estimation.frames import itrf_to_gcrs_eop
from gnn_state_estimation.slr import (
    SUPPORTED_STATIONS,
    nearest_met_record,
    parse_crd_v2_meteorology,
    parse_crd_v2_normal_points,
    parse_crd_v2_transmit_wavelength_nm,
)
from gnn_state_estimation.sp3 import (
    RangeObs,
    Sp3Interpolator,
    _process_cov,
    _sigma_points,
    _ukf_weights,
    parse_sp3,
    propagate_compact,
    run_range_aukf,
    run_range_ekf,
    run_range_ukf_fixed,
)
from gnn_state_estimation.sp3_hifi_calibrator import (
    RIDGE_GRID,
    HifiCalibrator,
    fit_ridge,
    propagate_hifi_corrected,
    residual_samples,
)
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

try:  # noqa: E402 - import context depends on script vs package execution
    from run_real_slr_sp3_corrected_validation import (
        ACCEL_PSD,
        ANALYSIS_CENTER,
        DEFAULT_WAVELENGTH_NM as CORRECTED_DEFAULT_WAVELENGTH_NM,
        INIT_POS_STD_M,
        INIT_VEL_STD_MPS,
        MAX_STEP_S,
        RANGE_STD_M,
        TRAIN_FRAC,
        _range_correction_m,
        _read_sp3_text,
        materialize_eop,
        sha256_file,
    )
    from run_real_slr_sp3_hifi_validation import EDC_CRD_URL, EDC_SP3_URL, materialize
    from run_real_slr_sp3_od_expanded_validation import (
        FORMAL210_SCHEDULE,
        schedule_weeks as expanded_schedule_weeks,
    )
except ModuleNotFoundError:  # pragma: no cover - pytest/package import path
    from scripts.run_real_slr_sp3_corrected_validation import (
        ACCEL_PSD,
        ANALYSIS_CENTER,
        DEFAULT_WAVELENGTH_NM as CORRECTED_DEFAULT_WAVELENGTH_NM,
        INIT_POS_STD_M,
        INIT_VEL_STD_MPS,
        MAX_STEP_S,
        RANGE_STD_M,
        TRAIN_FRAC,
        _range_correction_m,
        _read_sp3_text,
        materialize_eop,
        sha256_file,
    )
    from scripts.run_real_slr_sp3_hifi_validation import (
        EDC_CRD_URL,
        EDC_SP3_URL,
        materialize,
    )
    from scripts.run_real_slr_sp3_od_expanded_validation import (
        FORMAL210_SCHEDULE,
        schedule_weeks as expanded_schedule_weeks,
    )


DEFAULT_INPUT_DIR = Path("results/real_slr_sp3_od_formal210_inputs")
DEFAULT_EOP_DIR = Path("results/real_slr_sp3_corrected")
DEFAULT_OUT_DIR = Path("results/real_slr_sp3_temporal_corrected_od_campaign")
DEFAULT_PREDECLARATION = (
    Path("release/predeclarations")
    / "real_slr_sp3_temporal_corrected_od_campaign_20260526.json"
)
DEFAULT_OUTPUT_JSON = DEFAULT_OUT_DIR / "real_slr_sp3_temporal_corrected_od_campaign.json"
DEFAULT_TABLE = Path("paper/tables/real_slr_sp3_temporal_corrected_od_campaign.tex")

PREDECLARATION_SCHEMA_VERSION = "real_slr_sp3_temporal_corrected_od_predeclaration_v1"
CAMPAIGN_SCHEMA_VERSION = "real_slr_sp3_temporal_corrected_od_campaign_v1"
ARC_CACHE_SCHEMA_VERSION = "real_slr_sp3_temporal_corrected_od_arc_cache_v2"

SCHEDULE_FORMAL210_MINI_PRE260509 = "formal210_mini_pre260509_posthoc"
SCHEDULE_FORMAL210_RECENT_PRE260509 = "formal210_recent_pre260509_posthoc"
SCHEDULE_FORMAL210_PRE260509 = "formal210_pre260509_posthoc"
SCHEDULE_HIFI40_POSTHOC = "hifi40_posthoc"
SCHEDULE_PROSPECTIVE_260516 = "prospective_260516"
SCHEDULE_PROSPECTIVE_260523 = "prospective_260523"
SCHEDULE_PROSPECTIVE_260530 = "prospective_260530"
SCHEDULE_PROSPECTIVE_260606 = "prospective_260606"
SCHEDULE_PROSPECTIVE_260613 = "prospective_260613"
SCHEDULE_PROSPECTIVE_260620 = "prospective_260620"

PROSPECTIVE_260516_WEEK: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "260516": (
        None,
        ("20260511", "20260512", "20260513", "20260514", "20260515"),
    )
}

PROSPECTIVE_260523_WEEK: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "260523": (
        None,
        ("20260518", "20260519", "20260520", "20260521", "20260522"),
    )
}

PROSPECTIVE_260530_WEEK: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "260530": (
        None,
        ("20260525", "20260526", "20260527", "20260528", "20260529"),
    )
}

PROSPECTIVE_260606_WEEK: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "260606": (
        None,
        ("20260601", "20260602", "20260603", "20260604", "20260605"),
    )
}

PROSPECTIVE_260613_WEEK: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "260613": (
        None,
        ("20260608", "20260609", "20260610", "20260611", "20260612"),
    )
}

PROSPECTIVE_260620_WEEK: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "260620": (
        None,
        ("20260615", "20260616", "20260617", "20260618", "20260619"),
    )
}

PROSPECTIVE_SCHEDULE_SPECS = {
    SCHEDULE_PROSPECTIVE_260516: {
        "validation_week": "260509",
        "test_week": "260516",
        "added_week_maps": (PROSPECTIVE_260516_WEEK,),
        "used_weeks": ("260509", "260516"),
    },
    SCHEDULE_PROSPECTIVE_260523: {
        "validation_week": "260516",
        "test_week": "260523",
        "added_week_maps": (PROSPECTIVE_260516_WEEK, PROSPECTIVE_260523_WEEK),
        "used_weeks": ("260509", "260516", "260523"),
    },
    SCHEDULE_PROSPECTIVE_260530: {
        "validation_week": "260523",
        "test_week": "260530",
        "added_week_maps": (
            PROSPECTIVE_260516_WEEK,
            PROSPECTIVE_260523_WEEK,
            PROSPECTIVE_260530_WEEK,
        ),
        "used_weeks": ("260509", "260516", "260523", "260530"),
    },
    SCHEDULE_PROSPECTIVE_260606: {
        "validation_week": "260530",
        "test_week": "260606",
        "added_week_maps": (
            PROSPECTIVE_260516_WEEK,
            PROSPECTIVE_260523_WEEK,
            PROSPECTIVE_260530_WEEK,
            PROSPECTIVE_260606_WEEK,
        ),
        "used_weeks": ("260509", "260516", "260523", "260530", "260606"),
    },
    SCHEDULE_PROSPECTIVE_260613: {
        "validation_week": "260606",
        "test_week": "260613",
        "added_week_maps": (
            PROSPECTIVE_260516_WEEK,
            PROSPECTIVE_260523_WEEK,
            PROSPECTIVE_260530_WEEK,
            PROSPECTIVE_260606_WEEK,
            PROSPECTIVE_260613_WEEK,
        ),
        "used_weeks": ("260509", "260516", "260523", "260530", "260606", "260613"),
    },
    SCHEDULE_PROSPECTIVE_260620: {
        "validation_week": "260613",
        "test_week": "260620",
        "added_week_maps": (
            PROSPECTIVE_260516_WEEK,
            PROSPECTIVE_260523_WEEK,
            PROSPECTIVE_260530_WEEK,
            PROSPECTIVE_260606_WEEK,
            PROSPECTIVE_260613_WEEK,
            PROSPECTIVE_260620_WEEK,
        ),
        "used_weeks": (
            "260509",
            "260516",
            "260523",
            "260530",
            "260606",
            "260613",
            "260620",
        ),
    },
}

TARGETS = (
    ("LAGEOS-1", "lageos1", "L51"),
    ("LAGEOS-2", "lageos2", "L52"),
)

CLASSICAL_LABELS = (
    "EKF (full correction)",
    "UKF (full correction)",
    "AUKF (full correction)",
    "SP3-IC propagation (full correction)",
)
BEST_CLASSICAL_LABELS = (
    "EKF (full correction)",
    "UKF (full correction)",
    "AUKF (full correction)",
)
LEARNED_LABEL = "Learned residual UKF (full correction)"

BOOTSTRAP_SEED = 20260526
BOOTSTRAP_N = 5000
CALIBRATOR_SAMPLE_STEP_S = 300.0
CALIBRATOR_EDGE_MARGIN_S = 8.0 * 60.0
SP3_COVERAGE_MARGIN_S = 4.0 * 60.0


@dataclass(frozen=True)
class ArcSpec:
    target: str
    sat_key: str
    sp3_sat_id: str
    date: str
    sp3_week: str
    split: str


@dataclass(frozen=True)
class CorrectedOdArc:
    arc_id: str
    target: str
    date: str
    sp3_week: str
    split: str
    fit_obs: list[RangeObs]
    held_epochs: np.ndarray
    t0: float
    fit_last_epoch: float
    x0: np.ndarray
    p0: np.ndarray
    interp: "CorrectedGcrsInterp"
    num_observations: int
    num_fit: int
    num_held: int
    distinct_stations: int
    provenance: dict


@dataclass(frozen=True)
class ArcCacheContext:
    schedule_name: str
    cache_dir: Path
    predeclaration_sha256: str
    eop_record: dict
    training_input_digest_sha256: str
    resume: bool


class CorrectedGcrsInterp:
    """SP3 interpolator mapped to GCRS with real IERS EOP."""

    def __init__(self, eph, eop: EopSeries, order: int = 9) -> None:
        self.eph = eph
        self.eop = eop
        self.sp3 = Sp3Interpolator(eph, order=order)

    def _mapper(self, r_ecef: np.ndarray, epoch_unix: float) -> np.ndarray:
        xp, yp = self.eop.polar_motion_rad(epoch_unix)
        dut1 = self.eop.ut1_minus_utc_s(epoch_unix)
        return itrf_to_gcrs_eop(r_ecef, epoch_unix, xp, yp, dut1)

    def pos_inertial(self, epoch_unix: float) -> np.ndarray:
        return self._mapper(self.sp3.position_ecef_m(epoch_unix), epoch_unix)

    def station_inertial(self, cdp_id: int, epoch_unix: float) -> np.ndarray:
        return self._mapper(SUPPORTED_STATIONS[cdp_id].ecef_m(), epoch_unix)

    def state_inertial(self, epoch_unix: float, h_s: float = 2.0) -> np.ndarray:
        p0 = self.pos_inertial(epoch_unix)
        pp = self.pos_inertial(epoch_unix + h_s)
        pm = self.pos_inertial(epoch_unix - h_s)
        return np.hstack([p0, (pp - pm) / (2.0 * h_s)]).astype(np.float64)


def _round(value, ndigits: int = 2):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), ndigits)


def _fmt(value, ndigits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{ndigits}f}"


def _lambda_key(lam: float) -> str:
    return f"{float(lam):.0e}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_sha256(payload: dict | list) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validate_sp3_input(path: Path) -> tuple[str | None, str | None, str | None]:
    """Validate a materialized SP3 or SP3.gz path before calling parse_sp3.

    Returns ``(text, None, None)`` on success or ``(None, kind, reason)`` on
    failure.  ``kind`` is one of ``'sp3_not_valid_gzip'``,
    ``'sp3_decompress_failed'``, or ``'sp3_not_sp3_format'``.  This guards
    against cases where the materialized file is an HTTP error page or other
    non-SP3 content that would otherwise raise deep inside parse_sp3.
    """
    import gzip as _gzip

    raw = path.read_bytes()
    if path.suffix == ".gz":
        if raw[:2] != b"\x1f\x8b":
            kind = "sp3_not_valid_gzip"
            reason = (
                f"SP3 path has .gz suffix but does not begin with gzip magic "
                f"bytes (\\x1f\\x8b); first 32 bytes: {raw[:32]!r}.  The "
                f"materialized file ({path.stat().st_size} bytes) appears to "
                f"be an HTTP error page or other non-gzip content.  Treating "
                f"as public input unavailable."
            )
            return None, kind, reason
        try:
            text = _gzip.decompress(raw).decode("utf-8", "replace")
        except Exception as exc:
            kind = "sp3_decompress_failed"
            reason = (
                f"gzip decompression of {path.name} failed: {exc}.  "
                f"Treating as public input unavailable."
            )
            return None, kind, reason
    elif raw[:2] == b"\x1f\x8b":
        try:
            text = _gzip.decompress(raw).decode("utf-8", "replace")
        except Exception as exc:
            kind = "sp3_decompress_failed"
            reason = (
                f"gzip decompression of {path.name} failed: {exc}.  "
                f"Treating as public input unavailable."
            )
            return None, kind, reason
    else:
        text = raw.decode("utf-8", "replace")

    stripped = text.lstrip()
    if not stripped.startswith("#"):
        kind = "sp3_not_sp3_format"
        reason = (
            f"decompressed content of {path.name} does not begin with '#' as "
            f"expected for SP3 format; first 64 chars: {stripped[:64]!r}.  "
            f"Treating as public input unavailable."
        )
        return None, kind, reason

    return text, None, None


def _formal210_weeks_with_prospective_260516() -> dict[
    str, tuple[str | None, tuple[str, ...]]
]:
    weeks = expanded_schedule_weeks(FORMAL210_SCHEDULE)
    return {**weeks, **PROSPECTIVE_260516_WEEK}


def _formal210_weeks_with_prospective_260523() -> dict[
    str, tuple[str | None, tuple[str, ...]]
]:
    weeks = expanded_schedule_weeks(FORMAL210_SCHEDULE)
    return {**weeks, **PROSPECTIVE_260516_WEEK, **PROSPECTIVE_260523_WEEK}


def _formal210_weeks_with_added_public_weeks(
    added_week_maps: tuple[dict[str, tuple[str | None, tuple[str, ...]]], ...]
) -> dict[str, tuple[str | None, tuple[str, ...]]]:
    weeks = expanded_schedule_weeks(FORMAL210_SCHEDULE)
    for week_map in added_week_maps:
        weeks.update(week_map)
    return weeks


def _schedule_weeks(schedule_name: str) -> dict[str, tuple[str | None, tuple[str, ...]]]:
    if schedule_name in (
        SCHEDULE_FORMAL210_MINI_PRE260509,
        SCHEDULE_FORMAL210_RECENT_PRE260509,
        SCHEDULE_FORMAL210_PRE260509,
    ):
        return expanded_schedule_weeks(FORMAL210_SCHEDULE)
    if schedule_name == SCHEDULE_HIFI40_POSTHOC:
        all_weeks = expanded_schedule_weeks(FORMAL210_SCHEDULE)
        return {week: all_weeks[week] for week in ("260418", "260425", "260502", "260509")}
    if schedule_name in PROSPECTIVE_SCHEDULE_SPECS:
        return _formal210_weeks_with_added_public_weeks(
            PROSPECTIVE_SCHEDULE_SPECS[schedule_name]["added_week_maps"]
        )
    raise ValueError(f"unknown schedule: {schedule_name}")


def split_plan(schedule_name: str) -> dict[str, list[str]]:
    if schedule_name == SCHEDULE_FORMAL210_MINI_PRE260509:
        weeks = list(_schedule_weeks(schedule_name))
        used = ["260418", "260425", "260502"]
        return {
            "train": ["260418"],
            "validation": ["260425"],
            "test": ["260502"],
            "unused": [week for week in weeks if week not in used],
        }
    if schedule_name == SCHEDULE_FORMAL210_RECENT_PRE260509:
        weeks = list(_schedule_weeks(schedule_name))
        used = ["260321", "260328", "260404", "260411", "260418", "260425", "260502"]
        return {
            "train": ["260321", "260328", "260404", "260411", "260418"],
            "validation": ["260425"],
            "test": ["260502"],
            "unused": [week for week in weeks if week not in used],
        }
    if schedule_name == SCHEDULE_FORMAL210_PRE260509:
        weeks = list(_schedule_weeks(schedule_name))
        return {
            "train": weeks[:18],
            "validation": ["260425"],
            "test": ["260502"],
            "unused": ["260509"],
        }
    if schedule_name == SCHEDULE_HIFI40_POSTHOC:
        return {
            "train": ["260418", "260425"],
            "validation": ["260502"],
            "test": ["260509"],
            "unused": [],
        }
    if schedule_name in PROSPECTIVE_SCHEDULE_SPECS:
        weeks = list(_schedule_weeks(schedule_name))
        spec = PROSPECTIVE_SCHEDULE_SPECS[schedule_name]
        train = [
            week
            for week in expanded_schedule_weeks(FORMAL210_SCHEDULE)
            if week <= "260502"
        ]
        used = [*train, *spec["used_weeks"]]
        return {
            "train": train,
            "validation": [spec["validation_week"]],
            "test": [spec["test_week"]],
            "unused": [week for week in weeks if week not in used],
        }
    raise ValueError(f"unknown schedule: {schedule_name}")


def _dates_for_schedule(
    schedule_name: str, week: str, dates: tuple[str, ...]
) -> tuple[str, ...]:
    if schedule_name == SCHEDULE_FORMAL210_MINI_PRE260509:
        if week == "260425":
            return ("20260420",)
        if week == "260502":
            return ("20260427",)
    return dates


def build_arc_specs(schedule_name: str, split: str | None = None) -> list[ArcSpec]:
    weeks = _schedule_weeks(schedule_name)
    plan = split_plan(schedule_name)
    week_to_split = {
        week: label
        for label in ("train", "validation", "test", "unused")
        for week in plan[label]
    }
    arcs: list[ArcSpec] = []
    for week, (_old_split, dates) in weeks.items():
        label = week_to_split.get(week)
        if label is None:
            continue
        if split is not None and label != split:
            continue
        active_dates = _dates_for_schedule(schedule_name, week, dates)
        for target, sat_key, sp3_sat_id in TARGETS:
            for date in active_dates:
                arcs.append(ArcSpec(target, sat_key, sp3_sat_id, date, week, label))
    return arcs


def build_week_objects(schedule_name: str, split: str | None = None) -> list[dict]:
    weeks = _schedule_weeks(schedule_name)
    plan = split_plan(schedule_name)
    week_to_split = {
        week: label
        for label in ("train", "validation", "test", "unused")
        for week in plan[label]
    }
    rows: list[dict] = []
    for week in weeks:
        label = week_to_split.get(week)
        if label is None:
            continue
        if split is not None and label != split:
            continue
        for target, sat_key, sp3_sat_id in TARGETS:
            rows.append(
                {
                    "target": target,
                    "sat_key": sat_key,
                    "sp3_sat_id": sp3_sat_id,
                    "sp3_week": week,
                    "split": label,
                }
            )
    return rows


def _is_prospective_schedule(schedule_name: str) -> bool:
    return schedule_name in PROSPECTIVE_SCHEDULE_SPECS


def _source_schedule_role(
    schedule_name: str, week: str, original_formal210_split: str | None
) -> str:
    if _is_prospective_schedule(schedule_name) and original_formal210_split is None:
        return "prospective_added_public_week"
    if original_formal210_split is None:
        return "non_formal210_public_week"
    return "formal210_archived_public_week"


def _confirmatory_status(schedule_name: str) -> str:
    if _is_prospective_schedule(schedule_name):
        return "predeclared_prospective_public_temporal_holdout"
    if schedule_name.endswith("_posthoc"):
        return "post_hoc_robustness"
    return "predeclared_prospective"


def _arc_count_phrase(n_arcs: int) -> str:
    if n_arcs < 0:
        raise ValueError(f"n_arcs must be non-negative, got {n_arcs}")
    if n_arcs == 1:
        return "one-arc"
    if n_arcs == 2:
        return "two-arc"
    return f"{n_arcs}-arc"


def _arc_count_scope_token(n_arcs: int) -> str:
    if n_arcs < 0:
        raise ValueError(f"n_arcs must be non-negative, got {n_arcs}")
    if n_arcs == 1:
        return "one_arc"
    if n_arcs == 2:
        return "two_arc"
    return f"{n_arcs}_arc"


def learned_vs_recursive_classical_readout_context(
    schedule_name: str, n_test_arcs: int, planned_n_test_arcs: int | None = None
) -> dict[str, str]:
    if planned_n_test_arcs is None:
        planned_n_test_arcs = n_test_arcs
    if planned_n_test_arcs < n_test_arcs:
        raise ValueError(
            "planned_n_test_arcs must be greater than or equal to n_test_arcs"
        )
    arc_phrase = _arc_count_phrase(n_test_arcs)
    arc_scope = _arc_count_scope_token(n_test_arcs)
    if planned_n_test_arcs == n_test_arcs:
        prospective_scope = (
            f"prospective_public_week_{arc_scope}_readout_not_operational_pod"
        )
        prospective_phrase = (
            f"predeclared prospective public-week {arc_phrase} test readout"
        )
    else:
        planned_scope = _arc_count_scope_token(planned_n_test_arcs)
        prospective_scope = (
            "prospective_public_week_"
            f"{arc_scope}_completed_of_{planned_scope}_planned_"
            "readout_not_operational_pod"
        )
        prospective_phrase = (
            "predeclared prospective public-week test readout "
            f"({n_test_arcs} completed arcs of {planned_n_test_arcs} planned)"
        )
    if _is_prospective_schedule(schedule_name):
        return {
            "readout_scope": prospective_scope,
            "boundary": (
                "The learned residual is compared with the best recursive "
                f"classical candidate on this {prospective_phrase}; "
                "SP3-IC propagation "
                "remains part of the readout pool. This is not operational "
                "POD validation, central external validation, or "
                "simulator-result validation."
            ),
        }
    return {
        "readout_scope": (
            f"bounded_post_hoc_{arc_scope}_readout_not_validation_selection"
        ),
        "boundary": (
            "The learned residual has lower held-out SP3 state RMSE than "
            "the best recursive classical candidate on this bounded "
            f"post-hoc {arc_phrase} test readout only; SP3-IC propagation "
            "remains the test-floor candidate. This is not validation-"
            "selected evidence, operational POD validation, central "
            "external validation, or simulator-result validation."
        ),
    }


def _reason_for_boundary(schedule_name: str) -> str:
    if schedule_name in PROSPECTIVE_SCHEDULE_SPECS:
        spec = PROSPECTIVE_SCHEDULE_SPECS[schedule_name]
        test_week = spec["test_week"]
        validation_week = spec["validation_week"]
        return (
            f"Rule fixed before scoring newly available public EDC week {test_week}. "
            "Training uses historical formal210-style weeks through 260502, "
            f"validation uses {validation_week} only, and the frozen selected "
            f"rule is scored once on {test_week}. This is a prospective public "
            "temporal holdout, not operational POD."
        )
    if schedule_name in (
        SCHEDULE_FORMAL210_MINI_PRE260509,
        SCHEDULE_FORMAL210_RECENT_PRE260509,
        SCHEDULE_FORMAL210_PRE260509,
    ):
        return (
            "Later public EDC weeks were not reachable during this run; the "
            "default therefore uses already materialized formal210-style "
            "weeks and excludes the previously scored 260509 temporal-test "
            "week from scoring. It is robustness evidence only."
        )
    return "Uses the already inspected hifi40 weeks; robustness only."


def _predeclaration_claim_boundary(schedule_name: str) -> dict:
    if schedule_name in PROSPECTIVE_SCHEDULE_SPECS:
        test_week = PROSPECTIVE_SCHEDULE_SPECS[schedule_name]["test_week"]
        return {
            "prospective_public_temporal_holdout": True,
            "rule_fixed_before_scoring_new_public_week": True,
            "post_hoc_robustness_not_confirmatory": False,
            "can_be_used_as_operational_pod_validation": False,
            "can_be_used_as_central_external_validation": False,
            "can_be_used_as_learned_estimator_skill_validation": False,
            "appropriate_use": (
                "A predeclared prospective public-week temporal holdout under "
                f"LAGEOS CRD/SP3 data (test week {test_week}). It can reduce "
                "measurement-pipeline risk and test public-week transfer, but "
                "it is not operational POD validation or simulator-result "
                "validation."
            ),
        }
    return {
        "post_hoc_robustness_not_confirmatory": True,
        "can_be_used_as_operational_pod_validation": False,
        "can_be_used_as_central_external_validation": False,
        "can_be_used_as_learned_estimator_skill_validation": False,
        "appropriate_use": (
            "A temporally frozen, full-correction real-measurement stress "
            "test of transfer under public LAGEOS CRD/SP3 data. It can "
            "reduce measurement-pipeline risk but cannot validate "
            "operational POD or the simulator-bound learned-negative claim."
        ),
    }


def schedule_metadata(schedule_name: str) -> dict:
    weeks = _schedule_weeks(schedule_name)
    plan = split_plan(schedule_name)
    return {
        "schedule_name": schedule_name,
        "confirmatory_status": _confirmatory_status(schedule_name),
        "reason_for_boundary": _reason_for_boundary(schedule_name),
        "train_weeks": plan["train"],
        "validation_weeks": plan["validation"],
        "test_weeks": plan["test"],
        "unused_weeks": plan["unused"],
        "weeks": {
            week: {
                "dates": list(_dates_for_schedule(schedule_name, week, dates)),
                "source_week_dates": list(dates),
                "original_formal210_split": old_split,
                "source_schedule_role": _source_schedule_role(
                    schedule_name, week, old_split
                ),
                "temporal_corrected_split": next(
                    (
                        label
                        for label in ("train", "validation", "test", "unused")
                        if week in plan[label]
                    ),
                    None,
                ),
            }
            for week, (old_split, dates) in weeks.items()
            if any(week in plan[label] for label in ("train", "validation", "test", "unused"))
        },
        "targets": [target for target, _key, _sid in TARGETS],
        "train_week_objects": build_week_objects(schedule_name, "train"),
        "validation_arcs": [asdict(a) for a in build_arc_specs(schedule_name, "validation")],
        "test_arcs": [asdict(a) for a in build_arc_specs(schedule_name, "test")],
    }


def build_predeclaration(schedule_name: str) -> dict:
    return {
        "schema_version": PREDECLARATION_SCHEMA_VERSION,
        "created_utc": utc_now_iso(),
        "artifact_role": "predeclared_rule_schedule_before_scoring",
        "schedule": schedule_metadata(schedule_name),
        "candidate_set": {
            "classical": list(CLASSICAL_LABELS),
            "classical_skill_reference": list(BEST_CLASSICAL_LABELS),
            "learned": LEARNED_LABEL,
        },
        "selection_rule": {
            "primary_metric": (
                "mean arc-level held-out SP3 position RMSE in metres on the "
                "frozen test split"
            ),
            "learned_model_training": (
                "fit ridge residual-acceleration calibrators only on "
                "training-week SP3 states"
            ),
            "learned_hyperparameter_selection": (
                "select ridge lambda by lowest validation mean arc RMSE"
            ),
            "candidate_selection": (
                "choose the candidate with lowest validation mean arc RMSE, "
                "then score that frozen choice once on the test split"
            ),
            "test_set_information_used_for_selection": False,
        },
        "correction_stack": {
            "earth_orientation": "IERS finals2000A polar motion and UT1-UTC",
            "frame": "IAU-76/80 ITRF-to-GCRS reduction",
            "range_reductions": [
                "Marini-Murray troposphere from CRD meteorology",
                "LAGEOS centre-of-mass offset",
                "relativistic Shapiro delay",
            ],
        },
        "learned_ridge_grid": [_lambda_key(lam) for lam in RIDGE_GRID],
        "fixed_filter_settings": {
            "range_std_m": RANGE_STD_M,
            "accel_psd": ACCEL_PSD,
            "train_frac": TRAIN_FRAC,
            "max_step_s": MAX_STEP_S,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_N,
        },
        "claim_boundary": _predeclaration_claim_boundary(schedule_name),
    }


def write_predeclaration(schedule_name: str, path: Path) -> dict:
    payload = build_predeclaration(schedule_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def select_lowest(means: dict[str, float | None]) -> str:
    finite = {
        key: float(value)
        for key, value in means.items()
        if value is not None and np.isfinite(value)
    }
    if not finite:
        raise ValueError("no finite candidate mean")
    return min(finite, key=lambda key: (finite[key], key))


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


def paired_gap_summary(rows: list[dict], a_key: str, b_key: str) -> dict:
    gaps: list[float] = []
    for row in rows:
        vals = row.get("held_out_position_rmse_m", {})
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
        float(row["held_out_position_rmse_m"][label])
        for row in rows
        if row.get("held_out_position_rmse_m", {}).get(label) is not None
        and np.isfinite(row["held_out_position_rmse_m"][label])
    ]
    return _round(np.mean(vals)) if vals else None


def default_arc_cache_dir(output_json: Path) -> Path:
    return output_json.parent / f"{output_json.stem}_arc_cache"


def _cache_settings() -> dict:
    return {
        "range_std_m": RANGE_STD_M,
        "accel_psd": ACCEL_PSD,
        "train_frac": TRAIN_FRAC,
        "max_step_s": MAX_STEP_S,
        "calibrator_sample_step_s": CALIBRATOR_SAMPLE_STEP_S,
        "calibrator_edge_margin_s": CALIBRATOR_EDGE_MARGIN_S,
        "sp3_coverage_margin_s": SP3_COVERAGE_MARGIN_S,
        "learned_ridge_grid": [_lambda_key(lam) for lam in RIDGE_GRID],
    }


def _cache_input_record(record: dict) -> dict:
    keys = (
        "url",
        "archived_input_id",
        "input_source",
        "sha256",
        "bytes",
        "n_epochs",
        "n_rows",
    )
    return {key: record[key] for key in keys if key in record}


def _arc_input_provenance(arc: CorrectedOdArc, eop_record: dict) -> dict:
    return {
        "target": arc.target,
        "date": arc.date,
        "sp3_week": arc.sp3_week,
        "crd": _cache_input_record(arc.provenance["crd"]),
        "sp3": _cache_input_record(arc.provenance["sp3"]),
        "eop": _cache_input_record(eop_record),
    }


def arc_cache_key(
    *,
    schedule_name: str,
    split: str,
    arc_id: str,
    target: str,
    date: str,
    sp3_week: str,
    predeclaration_sha256: str,
    input_provenance: dict,
    training_input_digest_sha256: str,
    lambda_keys: list[str],
    selected_lam_key: str | None,
) -> dict:
    return {
        "schema_version": ARC_CACHE_SCHEMA_VERSION,
        "campaign_schema_version": CAMPAIGN_SCHEMA_VERSION,
        "schedule_name": schedule_name,
        "split": split,
        "arc_id": arc_id,
        "target": target,
        "date": date,
        "sp3_week": sp3_week,
        "predeclaration_sha256": predeclaration_sha256,
        "input_provenance": input_provenance,
        "training_input_digest_sha256": training_input_digest_sha256,
        "scored_learned_ridge_lambdas": list(lambda_keys),
        "selected_learned_ridge_lambda": selected_lam_key,
        "settings": _cache_settings(),
    }


def arc_cache_digest(key: dict) -> str:
    payload = json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _arc_cache_path(cache_dir: Path, key: dict) -> Path:
    digest = arc_cache_digest(key)
    split = str(key["split"])
    safe_arc = (
        str(key["arc_id"])
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    return cache_dir / split / f"{safe_arc}_{digest[:16]}.json"


def _load_cached_arc_row(path: Path, key: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != ARC_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("cache_key") != key:
        return None
    row = payload.get("row")
    if not isinstance(row, dict) or row.get("status") != "completed":
        return None
    return row


def _write_cached_arc_row(path: Path, key: dict, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARC_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "row": row,
    }
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _write_cache_selection(
    cache_dir: Path,
    *,
    schedule_name: str,
    predeclaration_sha256: str,
    selected_lam_key: str,
    validation_learned_means: dict[str, float | None],
) -> Path:
    payload = {
        "schema_version": ARC_CACHE_SCHEMA_VERSION,
        "record_type": "selected_learned_ridge_lambda",
        "schedule_name": schedule_name,
        "predeclaration_sha256": predeclaration_sha256,
        "selected_learned_ridge_lambda": selected_lam_key,
        "selection_source": "recomputed_from_validation_rows_after_cache_or_score",
        "validation_learned_mean_rms_m": validation_learned_means,
    }
    path = cache_dir / "selected_learned_ridge_lambda.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def _cache_record(
    *,
    cache_context: ArcCacheContext | None,
    selected_lam_key: str | None,
    selection_cache_path: Path | None,
) -> dict:
    if cache_context is None:
        return {
            "enabled": False,
            "resume": False,
            "keying": (
                "No per-arc cache was active. Run with --resume to enable "
                "sidecar arc-row caching."
            ),
        }
    return {
        "enabled": True,
        "resume": cache_context.resume,
        "cache_dir": cache_context.cache_dir.as_posix(),
        "cache_schema_version": ARC_CACHE_SCHEMA_VERSION,
        "predeclaration_sha256": cache_context.predeclaration_sha256,
        "training_input_digest_sha256": cache_context.training_input_digest_sha256,
        "selected_learned_ridge_lambda": selected_lam_key,
        "selected_lambda_cache_path": (
            selection_cache_path.as_posix() if selection_cache_path else None
        ),
        "keying": (
            "Arc rows are keyed by cache schema version, campaign schema "
            "version, schedule, split, arc id, target, date, SP3 week, "
            "predeclaration SHA-256, CRD/SP3/EOP input SHA-256 and byte "
            "counts, training SP3 input digest, scored ridge lambdas, "
            "selected ridge lambda for test rows, and fixed "
            "filter/calibrator settings. "
            "Validation rows score the full ridge grid and the selected "
            "lambda is recomputed from validation rows only."
        ),
    }


def _load_eop(eop_dir: Path, *, refresh: bool) -> tuple[EopSeries, dict]:
    eop_path, eop_method = materialize_eop(eop_dir, refresh=refresh)
    eop = load_eop_series(eop_path)
    return eop, {
        "archived_input_id": eop_path.name,
        "input_source": eop_method,
        "sha256": sha256_file(eop_path),
        "bytes": eop_path.stat().st_size,
        "n_rows": int(eop.mjd.size),
    }


def _sp3_path(input_dir: Path, sat_key: str, week: str) -> Path:
    return input_dir / f"nsgf.orb.{sat_key}.{week}.v80.sp3.gz"


def _crd_path(input_dir: Path, sat_key: str, date: str) -> Path:
    return input_dir / f"{sat_key}_{date}.np2"


def _materialize_sp3(
    input_dir: Path,
    sat_key: str,
    week: str,
    *,
    refresh: bool,
) -> tuple[Path, str, str]:
    url = EDC_SP3_URL.format(sat_key=sat_key, week=week)
    path = _sp3_path(input_dir, sat_key, week)
    method = materialize(url, path, refresh=refresh)
    return path, url, method


def _materialize_crd(
    input_dir: Path,
    sat_key: str,
    date: str,
    *,
    refresh: bool,
) -> tuple[Path, str, str]:
    year = date[:4]
    url = EDC_CRD_URL.format(sat_key=sat_key, year=year, date=date)
    path = _crd_path(input_dir, sat_key, date)
    method = materialize(url, path, refresh=refresh)
    return path, url, method


def load_week_object(
    row: dict,
    input_dir: Path,
    eop: EopSeries,
    *,
    refresh: bool,
) -> dict:
    sp3_path, sp3_url, sp3_method = _materialize_sp3(
        input_dir, row["sat_key"], row["sp3_week"], refresh=refresh
    )
    eph = parse_sp3(_read_sp3_text(sp3_path), row["sp3_sat_id"])
    return {
        **row,
        "eph": eph,
        "interp": CorrectedGcrsInterp(eph, eop, order=9),
        "sp3": {
            "url": sp3_url,
            "archived_input_id": sp3_path.name,
            "input_source": sp3_method,
            "sha256": sha256_file(sp3_path),
            "bytes": sp3_path.stat().st_size,
            "n_epochs": int(eph.epochs_unix.size),
        },
    }


def load_od_arc(
    spec: ArcSpec,
    input_dir: Path,
    eop: EopSeries,
    *,
    refresh: bool,
) -> CorrectedOdArc | dict:
    crd_path, crd_url, crd_method = _materialize_crd(
        input_dir, spec.sat_key, spec.date, refresh=refresh
    )
    sp3_path, sp3_url, sp3_method = _materialize_sp3(
        input_dir, spec.sat_key, spec.sp3_week, refresh=refresh
    )
    crd_text = crd_path.read_text(encoding="utf-8", errors="replace")
    points = parse_crd_v2_normal_points(crd_text)
    met_recs = parse_crd_v2_meteorology(crd_text)
    wl_nm = parse_crd_v2_transmit_wavelength_nm(crd_text)
    wavelength_um = (
        float(wl_nm if wl_nm is not None else CORRECTED_DEFAULT_WAVELENGTH_NM)
        / 1000.0
    )

    # Validate the SP3 input before parsing.  If the materialized file is
    # not valid gzip or not SP3 content (e.g. an HTTP error page), record an
    # input-unavailable exclusion for this arc rather than raising an
    # uncontrolled exception.
    _sp3_text, _sp3_kind, _sp3_reason = _validate_sp3_input(sp3_path)
    if _sp3_text is None:
        arc_id = f"{spec.target} {spec.date}"
        return {
            "status": "input_unavailable",
            "kind": _sp3_kind,
            "reason": _sp3_reason,
            "arc_id": arc_id,
            "target": spec.target,
            "date": spec.date,
            "sp3_week": spec.sp3_week,
            "split": spec.split,
            "num_observations": len(points),
            "sp3": {
                "url": sp3_url,
                "archived_input_id": sp3_path.name,
                "input_source": sp3_method,
                "sha256": sha256_file(sp3_path),
                "bytes": sp3_path.stat().st_size,
            },
            "crd": {
                "url": crd_url,
                "archived_input_id": crd_path.name,
                "input_source": crd_method,
                "sha256": sha256_file(crd_path),
                "bytes": crd_path.stat().st_size,
            },
        }

    # Parse SP3; catch unexpected parse failures and record as input_unavailable.
    try:
        eph = parse_sp3(_sp3_text, spec.sp3_sat_id)
    except Exception as _exc:
        arc_id = f"{spec.target} {spec.date}"
        return {
            "status": "input_unavailable",
            "kind": "sp3_parse_failed",
            "reason": f"parse_sp3 raised: {_exc}",
            "arc_id": arc_id,
            "target": spec.target,
            "date": spec.date,
            "sp3_week": spec.sp3_week,
            "split": spec.split,
            "num_observations": len(points),
            "sp3": {
                "url": sp3_url,
                "archived_input_id": sp3_path.name,
                "input_source": sp3_method,
                "sha256": sha256_file(sp3_path),
                "bytes": sp3_path.stat().st_size,
            },
            "crd": {
                "url": crd_url,
                "archived_input_id": crd_path.name,
                "input_source": crd_method,
                "sha256": sha256_file(crd_path),
                "bytes": crd_path.stat().st_size,
            },
        }

    interp = CorrectedGcrsInterp(eph, eop, order=9)
    points = [
        p for p in points if eph.covers(p.epoch_unix, margin_s=-SP3_COVERAGE_MARGIN_S)
    ]
    arc_id = f"{spec.target} {spec.date}"
    provenance = {
        "arc_id": arc_id,
        "target": spec.target,
        "date": spec.date,
        "sp3_week": spec.sp3_week,
        "split": spec.split,
        "sp3_analysis_center": ANALYSIS_CENTER,
        "wavelength_nm": wl_nm if wl_nm is not None else CORRECTED_DEFAULT_WAVELENGTH_NM,
        "num_met_records": len(met_recs),
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

    obs: list[RangeObs] = []
    for point in points:
        station = SUPPORTED_STATIONS[point.cdp_id]
        sat_ecef = interp.sp3.position_ecef_m(point.epoch_unix)
        met = nearest_met_record(met_recs, point.epoch_unix) if met_recs else None
        correction = _range_correction_m(
            "full",
            station,
            sat_ecef,
            point.epoch_unix,
            met,
            wavelength_um,
        )
        obs.append(
            RangeObs(
                epoch_unix=float(point.epoch_unix),
                station_pi_m=interp.station_inertial(point.cdp_id, point.epoch_unix),
                range_m=float(point.range_m) - correction,
            )
        )

    epochs = np.asarray([o.epoch_unix for o in obs], dtype=np.float64)
    n = len(obs)
    n_fit = min(max(6, int(np.floor(TRAIN_FRAC * n))), n - 3)
    fit_obs = obs[:n_fit]
    held_epochs = epochs[n_fit:]
    t0 = fit_obs[0].epoch_unix
    fit_last = fit_obs[-1].epoch_unix
    p0 = np.diag(
        np.asarray(
            [INIT_POS_STD_M**2] * 3 + [INIT_VEL_STD_MPS**2] * 3,
            dtype=np.float64,
        )
    )
    return CorrectedOdArc(
        arc_id=arc_id,
        target=spec.target,
        date=spec.date,
        sp3_week=spec.sp3_week,
        split=spec.split,
        fit_obs=fit_obs,
        held_epochs=held_epochs,
        t0=t0,
        fit_last_epoch=fit_last,
        x0=interp.state_inertial(t0),
        p0=p0,
        interp=interp,
        num_observations=n,
        num_fit=int(n_fit),
        num_held=int(n - n_fit),
        distinct_stations=len({p.station_code for p in points}),
        provenance=provenance,
    )


def _held_out_rmse_compact(state, fit_last_epoch, held_epochs, interp) -> float:
    errs: list[float] = []
    s = np.asarray(state, dtype=np.float64).copy()
    t_prev = float(fit_last_epoch)
    for te in np.asarray(held_epochs, dtype=np.float64):
        s = propagate_compact(s, te - t_prev, MAX_STEP_S)
        t_prev = float(te)
        errs.append(float(np.linalg.norm(s[:3] - interp.pos_inertial(te))))
    arr = np.asarray(errs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def _held_out_rmse_learned(
    state,
    fit_last_epoch,
    held_epochs,
    interp,
    calib: HifiCalibrator,
) -> float:
    errs: list[float] = []
    s = np.asarray(state, dtype=np.float64).copy()
    t_prev = float(fit_last_epoch)
    for te in np.asarray(held_epochs, dtype=np.float64):
        s = propagate_hifi_corrected(s, te - t_prev, t_prev, calib, MAX_STEP_S)
        t_prev = float(te)
        errs.append(float(np.linalg.norm(s[:3] - interp.pos_inertial(te))))
    arr = np.asarray(errs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")


def run_range_ukf_fixed_hifi_corrected(
    obs: list[RangeObs],
    x0: np.ndarray,
    p0: np.ndarray,
    range_std_m: float,
    accel_psd: float,
    calib: HifiCalibrator,
    max_step_s: float = MAX_STEP_S,
) -> dict:
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
        z_pred = np.asarray(
            [np.linalg.norm(pt[:3] - o.station_pi_m) for pt in pts],
            dtype=np.float64,
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


def fit_train_calibrators(
    schedule_name: str,
    input_dir: Path,
    eop: EopSeries,
    *,
    refresh: bool,
) -> dict:
    train_objects = [
        load_week_object(row, input_dir, eop, refresh=refresh)
        for row in build_week_objects(schedule_name, "train")
    ]
    training_input_provenance = [
        {
            "target": row["target"],
            "sat_key": row["sat_key"],
            "sp3_sat_id": row["sp3_sat_id"],
            "sp3_week": row["sp3_week"],
            "split": row["split"],
            "sp3": _cache_input_record(row["sp3"]),
        }
        for row in train_objects
    ]
    phi_blocks: list[np.ndarray] = []
    da_blocks: list[np.ndarray] = []
    t0_ref = None
    per_week_samples = []
    for row in train_objects:
        t0_ref = row["eph"].start_unix if t0_ref is None else t0_ref
        phi, da = residual_samples(
            row["interp"].pos_inertial,
            row["interp"].state_inertial,
            row["eph"].start_unix + CALIBRATOR_EDGE_MARGIN_S,
            row["eph"].end_unix - CALIBRATOR_EDGE_MARGIN_S,
            t0_ref,
            step_s=CALIBRATOR_SAMPLE_STEP_S,
        )
        per_week_samples.append(
            {
                "target": row["target"],
                "sp3_week": row["sp3_week"],
                "split": row["split"],
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
            "schedule_name": schedule_name,
            "training_weeks": split_plan(schedule_name)["train"],
            "n_train_samples": int(phi_train.shape[0]),
            "feature_count": int(phi_train.shape[1]),
            "residual_sample_step_s": CALIBRATOR_SAMPLE_STEP_S,
            "ridge_grid": [_lambda_key(lam) for lam in RIDGE_GRID],
            "per_week_object_samples": per_week_samples,
            "training_input_provenance": training_input_provenance,
            "training_input_digest_sha256": _json_sha256(
                training_input_provenance
            ),
            "sp3_used_for_fit_splits": ["train"],
        },
    }


def score_classical_arc(arc: CorrectedOdArc) -> dict[str, float | None]:
    ekf = run_range_ekf(arc.fit_obs, arc.x0, arc.p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    ukf = run_range_ukf_fixed(
        arc.fit_obs, arc.x0, arc.p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S
    )
    aukf = run_range_aukf(arc.fit_obs, arc.x0, arc.p0, RANGE_STD_M, ACCEL_PSD, MAX_STEP_S)
    sp3_ic = propagate_compact(arc.x0, arc.fit_last_epoch - arc.t0, MAX_STEP_S)
    return {
        "EKF (full correction)": _round(
            _held_out_rmse_compact(
                ekf["state"], arc.fit_last_epoch, arc.held_epochs, arc.interp
            )
        ),
        "UKF (full correction)": _round(
            _held_out_rmse_compact(
                ukf["state"], arc.fit_last_epoch, arc.held_epochs, arc.interp
            )
        ),
        "AUKF (full correction)": _round(
            _held_out_rmse_compact(
                aukf["state"], arc.fit_last_epoch, arc.held_epochs, arc.interp
            )
        ),
        "SP3-IC propagation (full correction)": _round(
            _held_out_rmse_compact(
                sp3_ic, arc.fit_last_epoch, arc.held_epochs, arc.interp
            )
        ),
    }


def score_learned_arc(arc: CorrectedOdArc, calib: HifiCalibrator) -> float | None:
    res = run_range_ukf_fixed_hifi_corrected(
        arc.fit_obs,
        arc.x0,
        arc.p0,
        RANGE_STD_M,
        ACCEL_PSD,
        calib,
        MAX_STEP_S,
    )
    return _round(
        _held_out_rmse_learned(
            res["state"], arc.fit_last_epoch, arc.held_epochs, arc.interp, calib
        )
    )


def _score_arc_row(
    arc: CorrectedOdArc,
    calibrators: dict[str, HifiCalibrator],
    *,
    lambda_keys: list[str],
) -> dict:
    classical = score_classical_arc(arc)
    learned_grid = {
        key: score_learned_arc(arc, calibrators[key]) for key in lambda_keys
    }
    return {
        "arc_id": arc.arc_id,
        "target": arc.target,
        "date": arc.date,
        "sp3_week": arc.sp3_week,
        "split": arc.split,
        "status": "completed",
        "num_observations": arc.num_observations,
        "num_fit": arc.num_fit,
        "num_held_out": arc.num_held,
        "distinct_stations": arc.distinct_stations,
        "held_out_position_rmse_m": classical,
        "learned_ridge_rmse_m": learned_grid,
        "provenance": arc.provenance,
    }


def score_split(
    arcs: list[CorrectedOdArc],
    calibrators: dict[str, HifiCalibrator],
    *,
    lambda_keys: list[str],
    cache_context: ArcCacheContext | None = None,
    selected_lam_key: str | None = None,
) -> tuple[list[dict], dict[str, float | None], dict[str, float | None]]:
    rows = []
    for arc in arcs:
        key = None
        cache_path = None
        if cache_context is not None:
            key = arc_cache_key(
                schedule_name=cache_context.schedule_name,
                split=arc.split,
                arc_id=arc.arc_id,
                target=arc.target,
                date=arc.date,
                sp3_week=arc.sp3_week,
                predeclaration_sha256=cache_context.predeclaration_sha256,
                input_provenance=_arc_input_provenance(
                    arc, cache_context.eop_record
                ),
                training_input_digest_sha256=(
                    cache_context.training_input_digest_sha256
                ),
                lambda_keys=lambda_keys,
                selected_lam_key=selected_lam_key if arc.split == "test" else None,
            )
            cache_path = _arc_cache_path(cache_context.cache_dir, key)
            if cache_context.resume:
                cached_row = _load_cached_arc_row(cache_path, key)
                if cached_row is not None:
                    print(f"cache hit for {arc.split} arc {arc.arc_id}", flush=True)
                    rows.append(cached_row)
                    continue

        print(f"scoring {arc.split} arc {arc.arc_id}", flush=True)
        row = _score_arc_row(arc, calibrators, lambda_keys=lambda_keys)
        if cache_path is not None and key is not None:
            _write_cached_arc_row(cache_path, key, row)
        rows.append(row)

    classical_means = {
        label: _round(
            np.mean(
                [
                    row["held_out_position_rmse_m"][label]
                    for row in rows
                    if row["held_out_position_rmse_m"].get(label) is not None
                ]
            )
        )
        for label in CLASSICAL_LABELS
    }
    learned_means = {
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
    return rows, classical_means, learned_means


def compose_rows_with_learned(rows: list[dict], selected_lam_key: str) -> list[dict]:
    out = []
    for row in rows:
        vals = dict(row["held_out_position_rmse_m"])
        vals[LEARNED_LABEL] = row["learned_ridge_rmse_m"].get(selected_lam_key)
        finite = {k: v for k, v in vals.items() if v is not None and np.isfinite(v)}
        out.append(
            {
                **{k: v for k, v in row.items() if k not in ("held_out_position_rmse_m", "learned_ridge_rmse_m")},
                "held_out_position_rmse_m": vals,
                "best_held_out_candidate": min(finite, key=finite.get) if finite else None,
            }
        )
    return out


def render_table(result: dict) -> str:
    selection = result["selection"]
    test = result["test_readout"]
    schedule = result["predeclared_schedule"]["schedule"]
    selected_gap = test["selected_vs_test_best_paired_gap"]
    learned_gap = test["learned_vs_best_classical_paired_gap"]
    val_weeks = ", ".join(schedule["validation_weeks"])
    test_weeks = ", ".join(schedule["test_weeks"])
    status = schedule["confirmatory_status"].replace("_", "-")
    if schedule["confirmatory_status"].startswith("predeclared_prospective"):
        boundary_sentence = (
            r"the rule was fixed before scoring the newly available public "
            f"test week and the result is labelled {status}. "
        )
    else:
        boundary_sentence = (
            r"because no later public week was reachable in this run and the "
            r"materialized weeks had already been inspected, the result is "
            f"labelled {status} rather than confirmatory. "
        )

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Full-correction temporal public real-measurement "
            r"orbit-determination pass on LAGEOS CRD normal points. The "
            r"schedule was fixed before scoring; "
            f"{boundary_sentence}The pass applies the IERS "
            r"Earth-orientation series, Marini--Murray troposphere, LAGEOS "
            r"centre-of-mass, and Shapiro corrections; the learned residual is "
            r"trained only on the train weeks, then ridge and candidate choices "
            f"use validation week(s) {val_weeks} and the frozen choice is "
            f"scored on test week(s) {test_weeks}. Residuals remain at the "
            r"hundreds-of-metres scale, so this is not operational POD.}"
        ),
        r"  \label{tab:real_slr_sp3_temporal_corrected_od_campaign}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        (
            r"    Comparison & Candidate/readout & "
            r"Validation mean [m] & Test mean [m] & Paired test gap [m] \\"
        ),
        r"    \midrule",
        (
            "    Frozen temporal selector & "
            f"{selection['selected_candidate']} & "
            f"{_fmt(selection['selected_validation_mean_rms_m'])} & "
            f"{_fmt(test['selected_test_mean_rms_m'])} & "
            f"{_fmt(selected_gap.get('mean_gap_m'))} "
            f"[{_fmt(selected_gap.get('bootstrap95_mean_gap_m', [None, None])[0])}, "
            f"{_fmt(selected_gap.get('bootstrap95_mean_gap_m', [None, None])[1])}] \\\\"
        ),
        (
            "    Learned residual vs best classical & "
            "Learned residual UKF (not validation-selected) & "
            f"{_fmt(selection['validation_mean_rms_m'][LEARNED_LABEL])} & "
            f"{_fmt(test['test_mean_rms_m'][LEARNED_LABEL])} & "
            f"{_fmt(learned_gap.get('mean_gap_m'))} "
            f"[{_fmt(learned_gap.get('bootstrap95_mean_gap_m', [None, None])[0])}, "
            f"{_fmt(learned_gap.get('bootstrap95_mean_gap_m', [None, None])[1])}] \\\\"
        ),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        (
            r"  \\[2pt] {\footnotesize Gaps are candidate minus reference; "
            r"positive values mean larger held-out position RMSE. The first "
            r"row references the test-best candidate in the validation-defined "
            r"pool; the second references the best non-SP3-IC classical "
            r"recursive-filter candidate.}"
        ),
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def write_table(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_table(result), encoding="utf-8")


def build_result(args) -> dict:
    predeclared = json.loads(args.predeclaration.read_text(encoding="utf-8"))
    if predeclared.get("schema_version") != PREDECLARATION_SCHEMA_VERSION:
        raise ValueError("predeclaration schema mismatch")
    if predeclared.get("schedule", {}).get("schedule_name") != args.schedule:
        raise ValueError("predeclaration schedule does not match requested schedule")

    predeclaration_sha256 = _sha256(args.predeclaration)
    cache_dir = args.arc_cache_dir
    if cache_dir is None and args.resume:
        cache_dir = default_arc_cache_dir(args.output_json)

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    eop, eop_record = _load_eop(Path(args.eop_dir), refresh=args.refresh_eop)
    cal = fit_train_calibrators(args.schedule, input_dir, eop, refresh=args.refresh)
    print(
        "fitted learned residual calibrators "
        f"from {cal['training_summary']['n_train_samples']} SP3 samples",
        flush=True,
    )
    calibrators: dict[str, HifiCalibrator] = cal["calibrators"]
    all_lambda_keys = [_lambda_key(lam) for lam in RIDGE_GRID]
    cache_context = (
        ArcCacheContext(
            schedule_name=args.schedule,
            cache_dir=Path(cache_dir),
            predeclaration_sha256=predeclaration_sha256,
            eop_record=eop_record,
            training_input_digest_sha256=cal["training_summary"][
                "training_input_digest_sha256"
            ],
            resume=args.resume,
        )
        if cache_dir is not None
        else None
    )

    def load_split(split: str) -> tuple[list[CorrectedOdArc], list[dict]]:
        loaded = [
            load_od_arc(spec, input_dir, eop, refresh=args.refresh)
            for spec in build_arc_specs(args.schedule, split)
        ]
        completed = [arc for arc in loaded if isinstance(arc, CorrectedOdArc)]
        failed = [arc for arc in loaded if isinstance(arc, dict)]
        return completed, failed

    val_arcs, val_failed = load_split("validation")
    test_arcs, test_failed = load_split("test")
    if not val_arcs:
        raise RuntimeError(
            "validation arcs are required for candidate selection; "
            "all validation arcs failed or were excluded"
        )

    val_rows_raw, val_classical_means, val_learned_means = score_split(
        val_arcs,
        calibrators,
        lambda_keys=all_lambda_keys,
        cache_context=cache_context,
    )
    selected_lam_key = select_lowest(val_learned_means)
    selection_cache_path = None
    if cache_context is not None:
        selection_cache_path = _write_cache_selection(
            cache_context.cache_dir,
            schedule_name=args.schedule,
            predeclaration_sha256=predeclaration_sha256,
            selected_lam_key=selected_lam_key,
            validation_learned_means=val_learned_means,
        )

    val_rows = compose_rows_with_learned(val_rows_raw, selected_lam_key)
    validation_means = dict(val_classical_means)
    validation_means[LEARNED_LABEL] = val_learned_means[selected_lam_key]
    selected_candidate = select_lowest(validation_means)
    best_classical_validation = select_lowest(
        {label: validation_means[label] for label in BEST_CLASSICAL_LABELS}
    )

    n_test_arcs = len(test_arcs)
    planned_n_test_arcs = n_test_arcs + len(test_failed)
    prospective = _is_prospective_schedule(args.schedule)
    paper_strength_class = (
        "predeclared_prospective_public_temporal_holdout"
        if prospective
        else "post_hoc_full_correction_public_real_measurement_stress"
    )

    if test_arcs:
        test_rows_raw, test_classical_means, test_learned_means = score_split(
            test_arcs,
            calibrators,
            lambda_keys=[selected_lam_key],
            cache_context=cache_context,
            selected_lam_key=selected_lam_key,
        )
        test_rows = compose_rows_with_learned(test_rows_raw, selected_lam_key)
        test_means: dict[str, float | None] = dict(test_classical_means)
        test_means[LEARNED_LABEL] = test_learned_means[selected_lam_key]
        test_best_candidate: str | None = select_lowest(test_means)
        best_classical_test: str | None = select_lowest(
            {label: test_means[label] for label in BEST_CLASSICAL_LABELS}
        )
        selected_gap = paired_gap_summary(
            test_rows, selected_candidate, test_best_candidate
        )
        selected_vs_best_classical = paired_gap_summary(
            test_rows, selected_candidate, best_classical_test
        )
        learned_gap = paired_gap_summary(test_rows, LEARNED_LABEL, best_classical_test)
        learned_lower_than_best_recursive_classical = bool(
            learned_gap.get("mean_gap_m") is not None
            and learned_gap["mean_gap_m"] < 0.0
            and learned_gap["bootstrap95_mean_gap_m"][1] is not None
            and learned_gap["bootstrap95_mean_gap_m"][1] < 0.0
        )
        overall_status = "completed"
    else:
        # All test arcs excluded due to input unavailability; no test metrics
        # can be honestly computed.  Record an honest partial result with null
        # test metrics so the artifact does not fabricate estimator numbers.
        print(
            f"WARNING: all {planned_n_test_arcs} planned test arcs were "
            "excluded due to input unavailability; writing honest partial "
            "result with null test metrics.",
            flush=True,
        )
        test_rows = []
        test_means = {label: None for label in CLASSICAL_LABELS}
        test_means[LEARNED_LABEL] = None
        test_best_candidate = None
        best_classical_test = None
        selected_gap = {"n": 0}
        selected_vs_best_classical = {"n": 0}
        learned_gap = {"n": 0}
        learned_lower_than_best_recursive_classical = False
        overall_status = "zero_test_arcs_input_unavailable"

    learned_readout_context = learned_vs_recursive_classical_readout_context(
        args.schedule,
        n_test_arcs,
        planned_n_test_arcs=planned_n_test_arcs,
    )

    result = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "generated_utc": utc_now_iso(),
        "status": overall_status,
        "predeclared_schedule": predeclared,
        "source_artifacts": [
            {
                "artifact_id": args.predeclaration.as_posix(),
                "sha256": predeclaration_sha256,
            }
        ],
        "resume_cache": _cache_record(
            cache_context=cache_context,
            selected_lam_key=selected_lam_key,
            selection_cache_path=selection_cache_path,
        ),
        "public_corpus": {
            "targets": [target for target, _key, _sid in TARGETS],
            "sp3_analysis_center": ANALYSIS_CENTER,
            "normal_point_source": "public ILRS CRD v2 normal points",
            "state_reference": "independent ILRS analysis-centre SP3 precise orbit",
            "schedule_name": args.schedule,
        },
        "eop_series": eop_record,
        "predeclared": {
            "range_std_m": RANGE_STD_M,
            "accel_psd": ACCEL_PSD,
            "train_frac": TRAIN_FRAC,
            "max_step_s": MAX_STEP_S,
            "learned_candidate": LEARNED_LABEL,
            "classical_candidate_pool": list(CLASSICAL_LABELS),
            "classical_skill_reference_pool": list(BEST_CLASSICAL_LABELS),
            "learned_ridge_grid": all_lambda_keys,
            "bootstrap_resamples": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "learned_calibrator_fit": cal["training_summary"],
        "selection_integrity": {
            "predeclaration_artifact_required_before_scoring": True,
            "predeclaration_sha256": predeclaration_sha256,
            "calibrator_fit_uses_only_train_weeks": True,
            "learned_ridge_selected_on_validation_only": True,
            "candidate_selected_on_validation_only": True,
            "test_set_information_used_for_selection": False,
            "train_weeks": split_plan(args.schedule)["train"],
            "validation_weeks": split_plan(args.schedule)["validation"],
            "test_weeks": split_plan(args.schedule)["test"],
            "unused_weeks": split_plan(args.schedule)["unused"],
            "selected_learned_ridge_lambda": selected_lam_key,
            "confirmatory_status": predeclared["schedule"]["confirmatory_status"],
            "prospective_rule_fixed_before_scoring_public_week": _is_prospective_schedule(
                args.schedule
            ),
            "not_operational_pod": True,
        },
        "validation": {
            "n_arcs": len(val_arcs),
            "n_failed_or_excluded_arcs": len(val_failed),
            "learned_ridge_mean_rms_m": val_learned_means,
            "classical_mean_rms_m": val_classical_means,
            "rows": val_rows,
            "failed_or_excluded_arcs": val_failed,
        },
        "selection": {
            "selection_rule": predeclared["selection_rule"]["candidate_selection"],
            "validation_mean_rms_m": validation_means,
            "selected_candidate": selected_candidate,
            "selected_candidate_family": (
                "learned" if selected_candidate == LEARNED_LABEL else "classical"
            ),
            "selected_validation_mean_rms_m": validation_means[selected_candidate],
            "best_classical_validation_candidate": best_classical_validation,
            "best_classical_validation_mean_rms_m": validation_means[
                best_classical_validation
            ],
            "selected_learned_ridge_lambda": selected_lam_key,
        },
        "test_readout": {
            "n_arcs": n_test_arcs,
            "n_planned_arcs": planned_n_test_arcs,
            "n_failed_or_excluded_arcs": len(test_failed),
            "test_mean_rms_m": test_means,
            "selected_candidate": selected_candidate,
            "selected_test_mean_rms_m": test_means.get(selected_candidate),
            "test_best_candidate": test_best_candidate,
            "test_best_mean_rms_m": (
                None if test_best_candidate is None
                else test_means[test_best_candidate]
            ),
            "best_classical_test_candidate": best_classical_test,
            "best_classical_test_mean_rms_m": (
                None if best_classical_test is None
                else test_means[best_classical_test]
            ),
            "selected_vs_test_best_paired_gap": selected_gap,
            "selected_vs_best_classical_paired_gap": selected_vs_best_classical,
            "learned_vs_best_classical_paired_gap": learned_gap,
            "arcs": test_rows,
            "failed_or_excluded_arcs": test_failed,
        },
        "headline_readout": {
            "selected_candidate": selected_candidate,
            "selected_candidate_family": (
                "learned" if selected_candidate == LEARNED_LABEL else "classical"
            ),
            "selected_test_mean_rms_m": test_means.get(selected_candidate),
            "test_best_candidate": test_best_candidate,
            "test_best_mean_rms_m": (
                None if test_best_candidate is None
                else test_means[test_best_candidate]
            ),
            "test_floor_candidate": test_best_candidate,
            "learned_test_mean_rms_m": test_means.get(LEARNED_LABEL),
            "best_classical_test_mean_rms_m": (
                None if best_classical_test is None
                else test_means[best_classical_test]
            ),
            "learned_minus_best_classical_mean_gap_m": learned_gap.get("mean_gap_m"),
            "completed_test_arcs": n_test_arcs,
            "planned_test_arcs": planned_n_test_arcs,
            "learned_residual_is_validation_selected_candidate": (
                selected_candidate == LEARNED_LABEL
            ),
            "learned_lower_than_best_recursive_classical": (
                learned_lower_than_best_recursive_classical
            ),
            **(
                {
                    "learned_lower_than_best_recursive_classical_on_two_posthoc_test_arcs": (
                        learned_lower_than_best_recursive_classical
                    ),
                }
                if not prospective and n_test_arcs == 2
                else {}
            ),
            "learned_lower_than_best_recursive_classical_on_bounded_posthoc_test_arcs": (
                learned_lower_than_best_recursive_classical and not prospective
            ),
            "learned_lower_than_best_recursive_classical_on_prospective_test_arcs": (
                learned_lower_than_best_recursive_classical and prospective
            ),
            "learned_vs_recursive_classical_readout_scope": (
                learned_readout_context["readout_scope"]
            ),
            "paper_strength_class": paper_strength_class,
        },
        "claim_boundary": {
            "defensible_status": paper_strength_class,
            "prospective_public_temporal_holdout": prospective,
            "rule_fixed_before_scoring_new_public_week": prospective,
            "post_hoc_robustness_not_confirmatory": not prospective,
            "central_external_validation_status": False,
            "can_be_used_as_central_external_validation": False,
            "can_be_used_as_bounded_public_real_measurement_od_probe": True,
            "is_operational_validation": False,
            "is_centimetre_slr_or_flight_validation": False,
            "is_simulator_result_validation": False,
            "does_not_relabel_provenance_as_validation": True,
            "learned_vs_recursive_classical_boundary": learned_readout_context[
                "boundary"
            ],
            "appropriate_use": (
                "Use as a predeclared prospective public temporal holdout "
                "and measurement-pipeline risk reduction, not as operational "
                "POD or simulator-result validation."
                if prospective
                else (
                    "Use as full-correction real-measurement stress evidence "
                    "and measurement-pipeline risk reduction, not as "
                    "operational POD or simulator-result validation."
                )
            ),
        },
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--schedule",
        choices=(
            SCHEDULE_FORMAL210_MINI_PRE260509,
            SCHEDULE_FORMAL210_RECENT_PRE260509,
            SCHEDULE_FORMAL210_PRE260509,
            SCHEDULE_HIFI40_POSTHOC,
            SCHEDULE_PROSPECTIVE_260516,
            SCHEDULE_PROSPECTIVE_260523,
            SCHEDULE_PROSPECTIVE_260530,
            SCHEDULE_PROSPECTIVE_260606,
            SCHEDULE_PROSPECTIVE_260613,
            SCHEDULE_PROSPECTIVE_260620,
        ),
        default=SCHEDULE_FORMAL210_MINI_PRE260509,
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--eop-dir", type=Path, default=DEFAULT_EOP_DIR)
    p.add_argument("--predeclaration", type=Path, default=DEFAULT_PREDECLARATION)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--write-predeclaration-only", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--refresh-eop", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse and write per-arc scored-row cache entries. The default "
            "cache directory is derived from --output-json."
        ),
    )
    p.add_argument(
        "--arc-cache-dir",
        type=Path,
        default=None,
        help="Optional sidecar directory for per-arc resume cache entries.",
    )
    p.add_argument("--no-table", action="store_true")
    return p


def _provided_cli_options(argv: list[str]) -> set[str]:
    return {
        token.split("=", 1)[0]
        for token in argv
        if token.startswith("--") and len(token) > 2
    }


def _is_default_path(path: Path, default: Path) -> bool:
    try:
        return path.expanduser().resolve() == default.expanduser().resolve()
    except OSError:
        return path == default


def _require_prospective_nondefault_path(
    args, provided_options: set[str], option: str, default: Path
) -> None:
    value = getattr(args, option.lstrip("-").replace("-", "_"))
    if option not in provided_options or _is_default_path(value, default):
        raise SystemExit(
            f"{args.schedule} requires explicit non-default {option} to avoid "
            "writing generic campaign artifacts"
        )


def validate_prospective_cli_path_safety(
    args, provided_options: set[str]
) -> None:
    if not _is_prospective_schedule(args.schedule):
        return

    _require_prospective_nondefault_path(
        args, provided_options, "--predeclaration", DEFAULT_PREDECLARATION
    )
    if args.write_predeclaration_only:
        return
    _require_prospective_nondefault_path(
        args, provided_options, "--output-json", DEFAULT_OUTPUT_JSON
    )
    has_explicit_nondefault_table = (
        "--table" in provided_options and not _is_default_path(args.table, DEFAULT_TABLE)
    )
    if not args.no_table and not has_explicit_nondefault_table:
        raise SystemExit(
            f"{args.schedule} requires --no-table or explicit non-default "
            "--table to avoid writing the generic campaign table"
        )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = build_parser().parse_args(argv)
    validate_prospective_cli_path_safety(args, _provided_cli_options(argv))
    if args.write_predeclaration_only:
        payload = write_predeclaration(args.schedule, args.predeclaration)
        print(
            json.dumps(
                {
                    "status": "predeclaration_written",
                    "path": str(args.predeclaration),
                    "schedule_name": payload["schedule"]["schedule_name"],
                    "confirmatory_status": payload["schedule"][
                        "confirmatory_status"
                    ],
                    "test_weeks": payload["schedule"]["test_weeks"],
                },
                indent=2,
            )
        )
        return 0
    if not args.predeclaration.exists():
        raise SystemExit(
            "predeclaration missing; run with --write-predeclaration-only before scoring"
        )
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
                "learned_minus_best_classical_mean_gap_m": result[
                    "headline_readout"
                ]["learned_minus_best_classical_mean_gap_m"],
                "paper_strength_class": result["headline_readout"][
                    "paper_strength_class"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
