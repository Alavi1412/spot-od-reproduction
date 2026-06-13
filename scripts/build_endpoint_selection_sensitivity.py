#!/usr/bin/env python
"""Build reviewer-requested endpoint-selection sensitivity artifacts.

This is a deterministic reanalysis of retained endpoint records. It does not
rerun estimators; it compares the same materialized K=8 and K=32 realization
records under the natural endpoint choices already stored in those records:
observed-step position RMSE and all-step position RMSE.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RECORDS = [
    {
        "record_id": "k8_endpoint_fixation_support",
        "label": "K=8 endpoint-fixation support",
        "path": Path("results/observed_step_preregistration/observed_step_preregistration.json"),
        "role": "supporting endpoint-fixation record; no timestamp asserted by the manuscript",
        "expected_n_realizations": 8,
        "scenario_summary_key": "primary_k8_predeclared",
        "observed_mean_key": "primary_observed_step_pos_rmse_m",
        "observed_realization_key": "per_realization_primary_m",
        "all_step_mean_key": "reference_all_step_pos_rmse_m",
        "all_step_realization_key": "per_realization_reference_all_step_m",
    },
    {
        "record_id": "k32_frozen_rule_replication",
        "label": "K=32 frozen-rule replication",
        "path": Path(
            "results/observed_step_prospective_replication/"
            "observed_step_prospective_replication.json"
        ),
        "role": "timestamped central frozen-rule replication under the established endpoint hierarchy",
        "expected_n_realizations": 32,
        "observed_mean_key": "primary_observed_step_pos_rmse_m",
        "observed_realization_key": "per_realization_observed_step_m",
        "all_step_mean_key": "reference_all_step_pos_rmse_m",
        "all_step_realization_key": "per_realization_reference_all_step_m",
    },
]

METRICS = [
    {
        "metric_id": "observed_step_position_rmse",
        "label": "Observed-step position RMSE",
        "mean_key_name": "observed_mean_key",
        "realization_key_name": "observed_realization_key",
        "interpretation": "primary endpoint in current hierarchy",
    },
    {
        "metric_id": "all_step_position_rmse",
        "label": "All-step position RMSE",
        "mean_key_name": "all_step_mean_key",
        "realization_key_name": "all_step_realization_key",
        "interpretation": "propagation-dominated reference",
    },
]

SCENARIO_LABELS = {
    "test": "Nominal",
    "stress_test": "Measurement-noise stress",
    "force_model_mismatch_test": "Controlled force-model mismatch",
}

CLASSICAL_METHODS = ("EKF", "UKF", "AUKF")
LEARNED_METHOD = "RGR-GF"


def _paired_bootstrap_ci(
    diffs: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    finite = np.asarray(diffs, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    n = finite.size
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = float(np.mean(finite[idx]))
    return (
        float(np.mean(finite)),
        float(np.quantile(draws, alpha / 2.0)),
        float(np.quantile(draws, 1.0 - alpha / 2.0)),
    )


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _best_classical(means: dict[str, Any]) -> str:
    available = [m for m in CLASSICAL_METHODS if m in means and math.isfinite(_finite_float(means[m]))]
    if not available:
        return ""
    return min(available, key=lambda m: _finite_float(means[m]))


def _prefix_realizations(per_real: dict[str, Any], n: int) -> dict[str, list[Any]]:
    return {method: list(values)[:n] for method, values in per_real.items()}


def _scenario_metric_inputs(
    *,
    record: dict[str, Any],
    scenario: dict[str, Any],
    metric: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    mean_key = record[metric["mean_key_name"]]
    realization_key = record[metric["realization_key_name"]]
    summary_key = record.get("scenario_summary_key")
    if summary_key:
        summary = scenario.get(str(summary_key), {})
        n_realizations = int(summary.get("n_realizations", 0))
        means = dict(summary.get(mean_key, {}))
        per_real = _prefix_realizations(dict(scenario.get(realization_key, {})), n_realizations)
        return means, per_real, n_realizations, dict(summary)
    return (
        dict(scenario.get(mean_key, {})),
        dict(scenario.get(realization_key, {})),
        int(scenario.get("n_realizations", 0)),
        dict(scenario),
    )


def _stored_primary_gap(source: dict[str, Any]) -> dict[str, Any] | None:
    required = (
        "rgr_gf_minus_best_classical_primary_mean_m",
        "rgr_gf_minus_best_classical_primary_ci_low_m",
        "rgr_gf_minus_best_classical_primary_ci_high_m",
    )
    if not all(key in source for key in required):
        return None
    mean = _finite_float(source.get("rgr_gf_minus_best_classical_primary_mean_m"))
    lo = _finite_float(source.get("rgr_gf_minus_best_classical_primary_ci_low_m"))
    hi = _finite_float(source.get("rgr_gf_minus_best_classical_primary_ci_high_m"))
    if not all(math.isfinite(value) for value in (mean, lo, hi)):
        return None
    learned_positive = None
    for key in (
        "learned_positive_under_predeclared_rule",
        "learned_positive_under_frozen_rule",
        "decision_predicate_satisfied",
    ):
        if key in source:
            learned_positive = bool(source[key])
            break
    return {
        "best_classical": str(
            source.get("best_classical_primary")
            or source.get("best_method_primary")
            or ""
        ),
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "learned_positive": learned_positive,
    }


def _assert_pairwise_inputs(
    *,
    record_id: str,
    scenario_name: str,
    metric_id: str,
    per_real: dict[str, Any],
    learned_method: str,
    comparator: str,
    n_realizations: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_realizations <= 0:
        raise AssertionError(
            f"{record_id}/{scenario_name}/{metric_id} has no realizations"
        )
    learned_values = np.asarray(per_real.get(learned_method, []), dtype=np.float64)
    comparator_values = np.asarray(per_real.get(comparator, []), dtype=np.float64)
    if learned_values.shape[0] != n_realizations:
        raise AssertionError(
            f"{record_id}/{scenario_name}/{metric_id} {learned_method} has "
            f"{learned_values.shape[0]} rows, expected {n_realizations}"
        )
    if comparator_values.shape[0] != n_realizations:
        raise AssertionError(
            f"{record_id}/{scenario_name}/{metric_id} {comparator} has "
            f"{comparator_values.shape[0]} rows, expected {n_realizations}"
        )
    return learned_values, comparator_values


def _rule_timestamp(rule: dict[str, Any], *, result_path: Path) -> str | None:
    for key in (
        "fixed_at_local_iso",
        "fixed_before_evaluation_local_iso",
        "predeclared_on_utc",
        "fixed_at",
    ):
        value = rule.get(key)
        if value:
            return str(value)

    fixed_rule_path = rule.get("fixed_rule_path")
    if not fixed_rule_path:
        return None

    fallback_path = Path(str(fixed_rule_path))
    if fallback_path.is_absolute():
        candidates = [fallback_path]
    else:
        candidates = [
            Path.cwd() / fallback_path,
            Path(__file__).resolve().parents[1] / fallback_path,
            result_path.parent / fallback_path,
        ]
    existing = next((candidate for candidate in candidates if candidate.exists()), None)
    if existing is None:
        return None

    try:
        fallback = json.loads(existing.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _rule_timestamp(fallback, result_path=result_path)


def _metric_row(
    *,
    record: dict[str, Any],
    scenario: dict[str, Any],
    metric: dict[str, str],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    means, per_real, n_realizations, stored_source = _scenario_metric_inputs(
        record=record,
        scenario=scenario,
        metric=metric,
    )
    best = _best_classical(means)
    if not best:
        raise AssertionError(
            f"{record['record_id']}/{scenario.get('name')}/{metric['metric_id']} "
            "has no finite classical comparator"
        )

    learned_mean = _finite_float(means.get(LEARNED_METHOD))
    best_mean = _finite_float(means.get(best))
    learned_values, best_values = _assert_pairwise_inputs(
        record_id=str(record["record_id"]),
        scenario_name=str(scenario.get("name")),
        metric_id=str(metric["metric_id"]),
        per_real=per_real,
        learned_method=LEARNED_METHOD,
        comparator=best,
        n_realizations=n_realizations,
    )
    stored_gap = (
        _stored_primary_gap(stored_source)
        if metric["metric_id"] == "observed_step_position_rmse"
        else None
    )
    if stored_gap is not None:
        stored_best = str(stored_gap.get("best_classical") or "")
        if stored_best and stored_best != best:
            raise AssertionError(
                f"{record['record_id']}/{scenario.get('name')} stored best "
                f"classical {stored_best} does not match retained means best {best}"
            )
        gap_mean = float(stored_gap["mean"])
        gap_lo = float(stored_gap["ci_low"])
        gap_hi = float(stored_gap["ci_high"])
        learned_positive = (
            bool(stored_gap["learned_positive"])
            if stored_gap.get("learned_positive") is not None
            else bool(
                math.isfinite(learned_mean)
                and math.isfinite(best_mean)
                and learned_mean < best_mean
                and math.isfinite(gap_hi)
                and gap_hi < 0.0
            )
        )
        ci_source = "stored_original_endpoint_record"
        row_bootstrap_seed = None
    else:
        diffs = learned_values - best_values
        gap_mean, gap_lo, gap_hi = _paired_bootstrap_ci(
            diffs,
            n_boot=bootstrap_samples,
            seed=bootstrap_seed,
        )
        learned_positive = bool(
            math.isfinite(learned_mean)
            and math.isfinite(best_mean)
            and learned_mean < best_mean
            and math.isfinite(gap_hi)
            and gap_hi < 0.0
        )
        ci_source = "sensitivity_recomputed_bootstrap"
        row_bootstrap_seed = int(bootstrap_seed)
    return {
        "record_id": record["record_id"],
        "record_label": record["label"],
        "record_role": record["role"],
        "scenario": scenario.get("name"),
        "scenario_label": SCENARIO_LABELS.get(str(scenario.get("name")), str(scenario.get("label", ""))),
        "metric_id": metric["metric_id"],
        "metric_label": metric["label"],
        "metric_interpretation": metric["interpretation"],
        "n_realizations": n_realizations,
        "trajectories_per_realization": int(scenario.get("trajectories_per_realization", 0)),
        "best_classical": best,
        "best_classical_mean_m": best_mean,
        "learned_method": LEARNED_METHOD,
        "learned_mean_m": learned_mean,
        "learned_minus_best_classical_mean_m": gap_mean,
        "learned_minus_best_classical_ci_low_m": gap_lo,
        "learned_minus_best_classical_ci_high_m": gap_hi,
        "confidence_interval_source": ci_source,
        "bootstrap_seed_used": row_bootstrap_seed,
        "learned_positive_under_metric": learned_positive,
        "qualitative_conclusion": (
            "learned positive under this metric"
            if learned_positive
            else "no learned positive under this metric"
        ),
    }


def build_endpoint_selection_sensitivity(
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260525,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    record_summaries: list[dict[str, Any]] = []
    expected_record_ids = {str(record["record_id"]) for record in DEFAULT_RECORDS}
    for rec_idx, record in enumerate(DEFAULT_RECORDS):
        path = Path(record["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing required endpoint record: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        rule = data.get("frozen_rule", data.get("pre_registration", {}))
        record_summaries.append(
            {
                "record_id": record["record_id"],
                "label": record["label"],
                "path": str(path),
                "role": record["role"],
                "status": data.get("status"),
                "expected_n_realizations": int(record["expected_n_realizations"]),
                "rule_timestamp": _rule_timestamp(rule, result_path=path),
            }
        )
        for scenario_idx, scenario in enumerate(data.get("scenarios", [])):
            n_actual = int(
                scenario.get(
                    "primary_k8_predeclared",
                    scenario,
                ).get("n_realizations", scenario.get("n_realizations", 0))
            )
            if n_actual != int(record["expected_n_realizations"]):
                raise AssertionError(
                    f"{record['record_id']}/{scenario.get('name')} has K={n_actual}; "
                    f"expected K={record['expected_n_realizations']}"
                )
            for metric_idx, metric in enumerate(METRICS):
                row_seed = bootstrap_seed + 1000 * rec_idx + 100 * scenario_idx + metric_idx
                rows.append(
                    _metric_row(
                        record=record,
                        scenario=scenario,
                        metric=metric,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=row_seed,
                    )
                )

    present_record_ids = {str(record["record_id"]) for record in record_summaries}
    if present_record_ids != expected_record_ids:
        raise AssertionError(
            f"Endpoint sensitivity expected records {sorted(expected_record_ids)}, "
            f"got {sorted(present_record_ids)}"
        )
    realization_counts = sorted({int(row["n_realizations"]) for row in rows})
    if realization_counts != [8, 32]:
        raise AssertionError(
            "Endpoint sensitivity must contain exactly K=8 and K=32 rows; "
            f"got {realization_counts}"
        )
    if any(int(row["n_realizations"]) == 16 for row in rows):
        raise AssertionError("K=16 strict-extension data must not enter endpoint sensitivity")

    all_no_positive = bool(rows) and not any(row["learned_positive_under_metric"] for row in rows)
    k32_rows = [row for row in rows if row["record_id"] == "k32_frozen_rule_replication"]
    k32_no_positive = bool(k32_rows) and not any(row["learned_positive_under_metric"] for row in k32_rows)
    return {
        "schema_version": "endpoint_selection_sensitivity_v1",
        "analysis_label": "reviewer-requested endpoint-selection sensitivity",
        "analysis_scope": (
            "Deterministic reanalysis of retained endpoint records under natural endpoint "
            "choices already materialized in the records. No estimator is rerun and no "
            "new endpoint is retroactively preregistered."
        ),
        "learned_method": LEARNED_METHOD,
        "classical_methods": list(CLASSICAL_METHODS),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "confidence_interval_policy": {
            "observed_step_position_rmse": (
                "Use the stored original endpoint-record mean and paired-bootstrap "
                "confidence interval when present."
            ),
            "all_step_position_rmse": (
                "No original all-step paired-gap interval is stored in the endpoint "
                "records; intervals are recomputed for this sensitivity audit and "
                "labeled as sensitivity recomputation."
            ),
        },
        "record_inclusion_assertions": {
            "expected_records": sorted(expected_record_ids),
            "present_records": sorted(present_record_ids),
            "realization_counts_present": realization_counts,
            "k16_rows_present": False,
        },
        "records": record_summaries,
        "rows": rows,
        "summary": {
            "num_rows": len(rows),
            "all_records_no_learned_positive": all_no_positive,
            "k32_no_learned_positive_under_observed_or_all_step": k32_no_positive,
            "qualitative_conclusion": (
                "The learned-versus-best-classical conclusion is unchanged under "
                "observed-step versus all-step endpoint choice in the retained K=8 and K=32 records."
                if all_no_positive
                else "At least one retained endpoint choice gives a learned positive; inspect rows."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default="results/endpoint_selection_sensitivity/endpoint_selection_sensitivity.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260525)
    args = parser.parse_args()

    artifact = build_endpoint_selection_sensitivity(
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
