#!/usr/bin/env python
"""Selection-stability audit for the temporal public CRD/SP3 OD campaign.

The loop-84 temporal OD campaign already freezes the learned ridge and final
candidate on the validation week before scoring the later test week.  This
additive audit asks whether that validation choice is stable under
validation-only resampling and records public-corpus breadth limits that keep
the result a bounded probe rather than operational validation.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from gnn_state_estimation.slr import SUPPORTED_STATIONS, parse_crd_v2_normal_points
from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso

try:  # noqa: E402
    from run_real_slr_sp3_hifi_validation import SPLIT_WEEKS, TRAIN_FRAC
    from run_real_slr_sp3_temporal_od_campaign import (
        CLASSICAL_LABELS,
        LEARNED_LABEL,
        paired_gap_summary,
        select_lowest,
    )
except ModuleNotFoundError:  # pragma: no cover - import context dependent
    from scripts.run_real_slr_sp3_hifi_validation import SPLIT_WEEKS, TRAIN_FRAC
    from scripts.run_real_slr_sp3_temporal_od_campaign import (
        CLASSICAL_LABELS,
        LEARNED_LABEL,
        paired_gap_summary,
        select_lowest,
    )


DEFAULT_TEMPORAL_JSON = (
    Path("results/real_slr_sp3_temporal_od_campaign")
    / "real_slr_sp3_temporal_od_campaign.json"
)
DEFAULT_HIFI_JSON = Path("results/real_slr_sp3_hifi/real_slr_sp3_hifi_validation.json")
DEFAULT_INPUT_DIR = Path("results/real_slr_sp3_hifi")
DEFAULT_OUTPUT_JSON = (
    Path("results/real_slr_sp3_temporal_selection_stability")
    / "real_slr_sp3_temporal_selection_stability.json"
)
DEFAULT_TABLE = Path("paper/tables/real_slr_sp3_temporal_selection_stability.tex")

BOOTSTRAP_SEED = 20260522
BOOTSTRAP_N = 5000
MIN_FIT_OBS_WITHOUT_STATION = 6
MIN_HELD_STATION_OBS = 3


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _round(x, ndigits: int = 2):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), ndigits)


def _fmt(x, ndigits: int = 2) -> str:
    if x is None:
        return "--"
    return f"{float(x):.{ndigits}f}"


def _pct(x) -> str:
    if x is None:
        return "--"
    return f"{100.0 * float(x):.1f}\\%"


def _classical_rows_by_arc(hifi: dict) -> dict[str, dict]:
    return {
        row["arc_id"]: row
        for row in hifi.get("arcs", [])
        if row.get("status") == "completed"
    }


def _validation_learned_rows(temporal: dict) -> list[dict]:
    return list(temporal.get("validation", {}).get("learned_rows", []))


def _learned_ridge_keys(rows: list[dict]) -> list[str]:
    keys = sorted(
        {
            key
            for row in rows
            for key in row.get("learned_ridge_rmse_m", {}).keys()
        }
    )
    if not keys:
        raise ValueError("validation learned rows contain no ridge keys")
    return keys


def _sample_mean(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def select_on_validation_indices(
    learned_rows: list[dict],
    hifi_by_arc: dict[str, dict],
    indices: np.ndarray,
) -> dict:
    """Repeat the temporal selector on a validation-only index sample."""
    ridge_keys = _learned_ridge_keys(learned_rows)
    ridge_means = {}
    for key in ridge_keys:
        vals = [
            learned_rows[int(i)]["learned_ridge_rmse_m"].get(key)
            for i in indices
        ]
        ridge_means[key] = _sample_mean(vals)
    selected_ridge = select_lowest(ridge_means)

    candidate_values = {label: [] for label in CLASSICAL_LABELS}
    candidate_values[LEARNED_LABEL] = []
    for i in indices:
        row = learned_rows[int(i)]
        arc_id = row["arc_id"]
        hrow = hifi_by_arc[arc_id]
        for label in CLASSICAL_LABELS:
            candidate_values[label].append(
                hrow["held_out_position_rmse_m"].get(label)
            )
        candidate_values[LEARNED_LABEL].append(
            row["learned_ridge_rmse_m"].get(selected_ridge)
        )

    candidate_means = {
        label: _sample_mean(vals) for label, vals in candidate_values.items()
    }
    selected_candidate = select_lowest(candidate_means)
    finite = sorted(
        (float(v), k)
        for k, v in candidate_means.items()
        if v is not None and np.isfinite(v)
    )
    margin = None
    if len(finite) >= 2:
        margin = finite[1][0] - finite[0][0]
    return {
        "selected_learned_ridge_lambda": selected_ridge,
        "selected_candidate": selected_candidate,
        "selected_candidate_family": (
            "learned" if selected_candidate == LEARNED_LABEL else "classical"
        ),
        "candidate_mean_rms_m": {
            k: _round(v) for k, v in candidate_means.items()
        },
        "winner_margin_to_runner_up_m": _round(margin),
    }


def _counter_fraction(counter: Counter, total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {
        key: round(float(count) / float(total), 4)
        for key, count in sorted(counter.items())
    }


def jackknife_selection(learned_rows: list[dict], hifi_by_arc: dict[str, dict]) -> dict:
    n = len(learned_rows)
    rows = []
    for leave_out in range(n):
        idx = np.asarray([i for i in range(n) if i != leave_out], dtype=int)
        sel = select_on_validation_indices(learned_rows, hifi_by_arc, idx)
        rows.append(
            {
                "left_out_arc_id": learned_rows[leave_out]["arc_id"],
                **sel,
            }
        )
    candidate_counts = Counter(r["selected_candidate"] for r in rows)
    ridge_counts = Counter(r["selected_learned_ridge_lambda"] for r in rows)
    return {
        "n_resamples": n,
        "candidate_selection_counts": dict(sorted(candidate_counts.items())),
        "candidate_selection_fraction": _counter_fraction(candidate_counts, n),
        "learned_ridge_selection_counts": dict(sorted(ridge_counts.items())),
        "learned_ridge_selection_fraction": _counter_fraction(ridge_counts, n),
        "rows": rows,
    }


def bootstrap_selection(
    learned_rows: list[dict],
    hifi_by_arc: dict[str, dict],
    *,
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    n = len(learned_rows)
    rng = np.random.default_rng(seed)
    candidate_counts: Counter = Counter()
    ridge_counts: Counter = Counter()
    margins = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sel = select_on_validation_indices(learned_rows, hifi_by_arc, idx)
        candidate_counts[sel["selected_candidate"]] += 1
        ridge_counts[sel["selected_learned_ridge_lambda"]] += 1
        margin = sel.get("winner_margin_to_runner_up_m")
        if margin is not None and np.isfinite(margin):
            margins.append(float(margin))
    arr = np.asarray(margins, dtype=np.float64)
    return {
        "n_resamples": int(n_resamples),
        "sample_size_arcs": int(n),
        "seed": int(seed),
        "candidate_selection_counts": dict(sorted(candidate_counts.items())),
        "candidate_selection_fraction": _counter_fraction(
            candidate_counts, n_resamples
        ),
        "learned_ridge_selection_counts": dict(sorted(ridge_counts.items())),
        "learned_ridge_selection_fraction": _counter_fraction(
            ridge_counts, n_resamples
        ),
        "winner_margin_to_runner_up_m": {
            "mean": _round(arr.mean()) if arr.size else None,
            "median": _round(np.median(arr)) if arr.size else None,
            "p05": _round(np.percentile(arr, 5.0)) if arr.size else None,
            "p95": _round(np.percentile(arr, 95.0)) if arr.size else None,
        },
    }


def compose_validation_rows(
    learned_rows: list[dict],
    hifi_by_arc: dict[str, dict],
    ridge_key: str,
) -> list[dict]:
    out = []
    for row in learned_rows:
        arc_id = row["arc_id"]
        vals = {
            label: hifi_by_arc[arc_id]["held_out_position_rmse_m"].get(label)
            for label in CLASSICAL_LABELS
        }
        vals[LEARNED_LABEL] = row["learned_ridge_rmse_m"].get(ridge_key)
        out.append(
            {
                "arc_id": arc_id,
                "target": row.get("target"),
                "date": row.get("date"),
                "split": row.get("split"),
                "held_out_position_rmse_m": vals,
            }
        )
    return out


def candidate_margin(means: dict[str, float | None]) -> dict:
    finite = sorted(
        (float(v), k) for k, v in means.items() if v is not None and np.isfinite(v)
    )
    if not finite:
        return {"winner": None}
    out = {"winner": finite[0][1], "winner_mean_rms_m": _round(finite[0][0])}
    if len(finite) > 1:
        out["runner_up"] = finite[1][1]
        out["runner_up_mean_rms_m"] = _round(finite[1][0])
        out["winner_margin_to_runner_up_m"] = _round(finite[1][0] - finite[0][0])
    return out


def station_coverage(input_dir: Path) -> dict:
    date_split = {
        d: split
        for _week, (split, days) in SPLIT_WEEKS.items()
        for d in days
    }
    split_blocks = {
        split: {
            str(cdp): {
                "station_code": station.code,
                "fit_observations": 0,
                "held_out_observations": 0,
                "arcs_with_held_out_observations": 0,
                "station_holdout_eligible_arcs": 0,
            }
            for cdp, station in sorted(SUPPORTED_STATIONS.items())
        }
        for split in ("train", "val", "test", "all")
    }
    parsed_files = 0
    for path in sorted(input_dir.glob("lageos*_*.np2")):
        date = path.stem.split("_")[-1]
        split = date_split.get(date)
        if split is None:
            continue
        points = parse_crd_v2_normal_points(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if len(points) < 10:
            continue
        parsed_files += 1
        n_fit = min(max(6, int(math.floor(TRAIN_FRAC * len(points)))), len(points) - 3)
        fit = points[:n_fit]
        held = points[n_fit:]
        for scope in (split, "all"):
            for cdp in SUPPORTED_STATIONS:
                key = str(cdp)
                fit_other = sum(p.cdp_id != cdp for p in fit)
                held_station = sum(p.cdp_id == cdp for p in held)
                block = split_blocks[scope][key]
                block["fit_observations"] += sum(p.cdp_id == cdp for p in fit)
                block["held_out_observations"] += held_station
                if held_station:
                    block["arcs_with_held_out_observations"] += 1
                if (
                    fit_other >= MIN_FIT_OBS_WITHOUT_STATION
                    and held_station >= MIN_HELD_STATION_OBS
                ):
                    block["station_holdout_eligible_arcs"] += 1
    return {
        "parsed_crd_files": parsed_files,
        "supported_station_count": len(SUPPORTED_STATIONS),
        "minimum_fit_observations_without_station": MIN_FIT_OBS_WITHOUT_STATION,
        "minimum_held_out_observations_at_station": MIN_HELD_STATION_OBS,
        "by_split": split_blocks,
    }


def public_breadth_feasibility(temporal: dict, coverage: dict) -> dict:
    targets = temporal.get("public_corpus", {}).get("targets", [])
    test_cov = coverage.get("by_split", {}).get("test", {})
    eligible_test = {
        v["station_code"]: v["station_holdout_eligible_arcs"]
        for v in test_cov.values()
    }
    stations_with_test_support = {
        k: v for k, v in eligible_test.items() if int(v) > 0
    }
    return {
        "held_out_object_beyond_lageos_feasible_from_archived_corpus": False,
        "archived_public_targets": targets,
        "larger_temporal_campaign_feasible_from_archived_corpus": False,
        "archived_temporal_weeks": temporal.get("public_corpus", {}).get(
            "split_weeks", {}
        ),
        "held_out_station_campaign_general_feasible_from_archived_corpus": False,
        "station_holdout_test_eligible_arcs_by_station": eligible_test,
        "stations_with_any_test_station_holdout_support": stations_with_test_support,
        "feasibility_summary": (
            "The archived public real-measurement corpus is limited to "
            "LAGEOS-1/LAGEOS-2 and four weekly windows. Station diversity is "
            "useful for an audit, but the held-out test week has station-"
            "holdout support for only a subset of the four supported stations; "
            "therefore a general held-out-station campaign would be too small "
            "for a stronger breadth claim without adding a newly predeclared "
            "public-data collection."
        ),
        "fallback_implemented": (
            "validation-only selector jackknife/bootstrap stability plus a "
            "post-scoring test-oracle leakage sentinel"
        ),
    }


def write_table(result: dict, path: Path) -> None:
    temporal = result["temporal_campaign_readout"]
    val = result["validation_margin"]
    boot = result["bootstrap_selection_stability"]
    jack = result["jackknife_selection_stability"]
    neg = result["negative_control_sentinels"]
    test = temporal["frozen_test_readout"]

    selected = temporal["validation_selected_candidate"]
    learned_freq = boot["candidate_selection_fraction"].get(LEARNED_LABEL, 0.0)
    jack_learned = jack["candidate_selection_counts"].get(LEARNED_LABEL, 0)
    jack_n = jack["n_resamples"]
    ci = test["learned_vs_best_classical_bootstrap95_mean_gap_m"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Selection-stability audit for the temporal public "
            r"real-measurement orbit-determination probe. Ridge and candidate "
            r"choices are repeated only on validation-week arcs; the later "
            r"test week remains a frozen readout. The audit is a bounded "
            r"LAGEOS public-data stress test, not operational SLR POD.}"
        ),
        r"  \label{tab:real_slr_sp3_temporal_selection_stability}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Audit & Validation-only readout & Frozen test readout & Boundary \\",
        r"    \midrule",
        (
            "    Nominal selector & "
            f"{selected}; margin {_fmt(val['winner_margin_to_runner_up_m'])} m & "
            f"{_fmt(test['learned_mean_rms_m'])} m learned vs "
            f"{_fmt(test['best_classical_mean_rms_m'])} m best classical & "
            r"small validation margin \\"
        ),
        (
            "    Validation bootstrap & "
            f"learned selected {_pct(learned_freq)} of resamples & "
            f"test gap {_fmt(test['learned_minus_best_classical_mean_gap_m'])} "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] m & "
            r"resamples validation arcs only \\"
        ),
        (
            "    Leave-one-arc-out & "
            f"learned selected {jack_learned}/{jack_n} folds & "
            f"{test['test_best_candidate']} is the test-best candidate & "
            r"selector is fragile \\"
        ),
        (
            "    Test-oracle sentinel & "
            f"validation selects {selected} & "
            f"test-only oracle would select {neg['test_oracle_candidate']} & "
            r"oracle not used for selection \\"
        ),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_result(args) -> dict:
    temporal = json.loads(args.temporal_json.read_text(encoding="utf-8"))
    hifi = json.loads(args.hifi_json.read_text(encoding="utf-8"))
    learned_rows = _validation_learned_rows(temporal)
    hifi_by_arc = _classical_rows_by_arc(hifi)
    n_val = len(learned_rows)
    if n_val == 0:
        raise RuntimeError("temporal artifact contains no validation rows")

    nominal_idx = np.arange(n_val, dtype=int)
    nominal = select_on_validation_indices(learned_rows, hifi_by_arc, nominal_idx)
    selected_ridge = nominal["selected_learned_ridge_lambda"]
    val_rows = compose_validation_rows(learned_rows, hifi_by_arc, selected_ridge)

    validation_means = temporal.get("selection", {}).get("validation_mean_rms_m", {})
    val_margin = candidate_margin(validation_means)
    best_classical_val = temporal.get("selection", {}).get(
        "best_classical_validation_candidate"
    )
    val_gap = paired_gap_summary(
        val_rows,
        LEARNED_LABEL,
        best_classical_val,
        field="held_out_position_rmse_m",
    )

    jack = jackknife_selection(learned_rows, hifi_by_arc)
    boot = bootstrap_selection(
        learned_rows,
        hifi_by_arc,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    coverage = station_coverage(args.input_dir)
    feasibility = public_breadth_feasibility(temporal, coverage)

    test_readout = temporal["test_readout"]
    learned_gap_test = test_readout["learned_vs_best_classical_paired_gap"]
    selected = temporal["selection"]["selected_candidate"]
    test_best = test_readout["test_best_candidate"]
    result = {
        "schema_version": "real_slr_sp3_temporal_selection_stability_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed",
        "source_artifacts": [
            {
                "artifact_id": args.temporal_json.as_posix(),
                "sha256": _sha256(args.temporal_json),
            },
            {
                "artifact_id": args.hifi_json.as_posix(),
                "sha256": _sha256(args.hifi_json),
            },
        ],
        "predeclared": {
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "bootstrap_seed": int(args.bootstrap_seed),
            "jackknife_unit": "validation arc",
            "selector_repeated_on_validation_only": True,
            "test_set_information_used_for_selection": False,
            "station_coverage_min_fit_observations_without_station": (
                MIN_FIT_OBS_WITHOUT_STATION
            ),
            "station_coverage_min_held_out_observations_at_station": (
                MIN_HELD_STATION_OBS
            ),
        },
        "temporal_campaign_readout": {
            "train_weeks": temporal["selection_integrity"]["train_weeks"],
            "validation_week": temporal["selection_integrity"]["validation_week"],
            "test_week": temporal["selection_integrity"]["test_week"],
            "validation_selected_learned_ridge_lambda": selected_ridge,
            "validation_selected_candidate": selected,
            "validation_selected_candidate_family": temporal["selection"][
                "selected_candidate_family"
            ],
            "validation_mean_rms_m": validation_means,
            "frozen_test_readout": {
                "n_arcs": test_readout["n_arcs"],
                "selected_candidate": selected,
                "selected_mean_rms_m": test_readout["selected_test_mean_rms_m"],
                "test_best_candidate": test_best,
                "test_best_mean_rms_m": test_readout["test_best_mean_rms_m"],
                "best_classical_candidate": test_readout[
                    "best_classical_test_candidate"
                ],
                "best_classical_mean_rms_m": test_readout[
                    "best_classical_test_mean_rms_m"
                ],
                "learned_mean_rms_m": test_readout["test_mean_rms_m"][
                    LEARNED_LABEL
                ],
                "learned_minus_best_classical_mean_gap_m": learned_gap_test[
                    "mean_gap_m"
                ],
                "learned_vs_best_classical_bootstrap95_mean_gap_m": (
                    learned_gap_test["bootstrap95_mean_gap_m"]
                ),
                "learned_arcs_better_than_best_classical": learned_gap_test[
                    "n_a_lower_rmse"
                ],
            },
        },
        "validation_margin": {
            **val_margin,
            "learned_vs_best_classical_paired_gap": val_gap,
            "interpretation": (
                "The nominal validation win by the learned candidate is small "
                "relative to arc-level variation; stability is therefore the "
                "relevant readout, not a positive learned-OD claim."
            ),
        },
        "jackknife_selection_stability": jack,
        "bootstrap_selection_stability": boot,
        "negative_control_sentinels": {
            "test_oracle_candidate": test_best,
            "validation_selector_differs_from_test_oracle": selected != test_best,
            "test_oracle_is_forbidden_for_model_selection": True,
            "interpretation": (
                "The candidate that would be selected by looking at the test "
                "week is reported only after frozen scoring as a leakage "
                "sentinel; it is not used to change the selected candidate."
            ),
        },
        "public_breadth_feasibility": feasibility,
        "station_coverage_audit": coverage,
        "claim_boundary": {
            "defensible_status": (
                "auditable_selection_stability_and_breadth_feasibility_audit"
            ),
            "can_be_used_as_central_external_validation": False,
            "can_be_used_as_operational_pod_validation": False,
            "can_be_used_as_centimetre_slr_validation": False,
            "can_be_used_as_public_temporal_probe_stability_audit": True,
            "does_not_expand_targets_beyond_lageos": True,
            "does_not_select_or_tune_on_test": True,
            "appropriate_use": (
                "Use to show that the temporal public-data selector is fragile "
                "under validation-only resampling and remains negative when "
                "frozen on the held-out test week."
            ),
        },
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temporal-json", type=Path, default=DEFAULT_TEMPORAL_JSON)
    p.add_argument("--hifi-json", type=Path, default=DEFAULT_HIFI_JSON)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    p.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_N)
    p.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
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
                "selected_candidate": result["temporal_campaign_readout"][
                    "validation_selected_candidate"
                ],
                "bootstrap_candidate_selection_fraction": result[
                    "bootstrap_selection_stability"
                ]["candidate_selection_fraction"],
                "jackknife_candidate_selection_counts": result[
                    "jackknife_selection_stability"
                ]["candidate_selection_counts"],
                "test_oracle_candidate": result["negative_control_sentinels"][
                    "test_oracle_candidate"
                ],
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
