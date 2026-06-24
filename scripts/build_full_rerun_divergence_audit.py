#!/usr/bin/env python3
"""Build a bounded divergence audit for the 2026-06-16 full rerun.

The audit reconciles divergence flags already present in the retained full
rerun metrics and scorecard artifacts. It does not rerun models, recompute
estimates, or redefine manuscript performance decisions.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RERUN_DIR = ROOT / "results" / "full_rerun_20260616"
DEFAULT_JSON_OUT = ROOT / "results" / "validation" / "full_rerun_divergence_audit_20260617.json"
DEFAULT_MD_OUT = ROOT / "results" / "validation" / "full_rerun_divergence_audit_20260617.md"
REPORT_SCHEMA_VERSION = "full_rerun_divergence_audit_v2"
POS_EXTREME_RMSE_THRESHOLD_M = 1.0e8
VEL_EXTREME_RMSE_THRESHOLD_MPS = 1.0e5
CLAIM_BOUNDARY = (
    "Diagnostic audit only; not a canonical table replacement, not operational "
    "validation, not independent reproduction, and not a rerun success upgrade."
)
FAILURE_CONDITIONED_BOUNDARY = (
    "Failure-conditioned summaries are diagnostic only. They are not replacement "
    "metrics, do not redefine performance, and do not rescue any method."
)
LEARNED_POSITIVE_BOUNDARY = (
    "No learned-positive claim should be inferred from raw tiny wins or "
    "failure-conditioned rows. The full-rerun scorecard already treats "
    "candidate divergence as failing the practical/headline decision logic."
)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def clean_number(value: Any) -> int | float | None:
    number = finite_float(value)
    if number is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return number


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def is_mask_flagged_trajectory(row: dict[str, float | None]) -> bool:
    """Mirror the evaluator-style extreme trajectory mask on a paired row."""
    pos = row["pos"]
    vel = row["vel"]
    return (
        pos is None
        or vel is None
        or pos > POS_EXTREME_RMSE_THRESHOLD_M
        or vel > VEL_EXTREME_RMSE_THRESHOLD_MPS
    )


def trajectory_values(rows: list[dict[str, float | None]], key: str) -> list[float]:
    return [value for row in rows if (value := row[key]) is not None]


def retained_mask_conditioned_values(rows: list[dict[str, float | None]], key: str) -> list[float]:
    return [
        value
        for row in rows
        if not is_mask_flagged_trajectory(row) and (value := row[key]) is not None
    ]


def mask_comparison_note(mask_count: int, metrics_count: int) -> str:
    if mask_count == metrics_count:
        return (
            "Evaluator-style paired trajectory mask count matches metrics "
            "num_diverged_trajectories. This remains diagnostic only and does "
            "not redefine the decision rule."
        )
    return (
        "Evaluator-style paired trajectory mask flagged "
        f"{mask_count} rows, while metrics num_diverged_trajectories reports "
        f"{metrics_count}. This audit reports both counts and does not force "
        "equality; the failure-conditioned summaries remain diagnostic only."
    )


def trajectory_distribution(rows: list[dict[str, float | None]], key: str) -> dict[str, Any]:
    values = trajectory_values(rows, key)
    retained = retained_mask_conditioned_values(rows, key)
    return {
        "count": len(values),
        "trajectory_row_count": len(rows),
        "nonfinite_count": len(rows) - len(values),
        "median": percentile(values, 50.0),
        "p90": percentile(values, 90.0),
        "p95": percentile(values, 95.0),
        "max": max(values) if values else None,
        "top_values_desc": sorted(values, reverse=True)[:5],
        "mean_excluding_mask_flagged_trajectories": mean(retained),
        "count_excluding_mask_flagged_trajectories": len(retained),
    }


def load_trajectory_errors(path: Path) -> dict[tuple[str, str], list[dict[str, float | None]]]:
    out: dict[tuple[str, str], list[dict[str, float | None]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"scenario", "method", "traj_pos_rmse_m", "traj_vel_rmse_mps"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            key = (str(row["scenario"]), str(row["method"]))
            bucket = out.setdefault(key, [])
            bucket.append(
                {
                    "pos": finite_float(row.get("traj_pos_rmse_m")),
                    "vel": finite_float(row.get("traj_vel_rmse_mps")),
                }
            )
    return out


def method_items(scenario_payload: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(scenario_payload, dict):
        return []
    rows: list[tuple[str, dict[str, Any]]] = []
    for method, payload in scenario_payload.items():
        if str(method).startswith("_") or not isinstance(payload, dict):
            continue
        rows.append((str(method), payload))
    return rows


def scorecard_candidate_flag(scorecard: dict[str, Any], scenario: str, method: str) -> bool | None:
    scenario_payload = scorecard.get(scenario)
    if not isinstance(scenario_payload, dict):
        return None
    method_payload = scenario_payload.get(method)
    if not isinstance(method_payload, dict):
        return None
    raw = method_payload.get("candidate_diverged")
    return bool(raw) if raw is not None else None


def build_divergence_cases(
    metrics: dict[str, Any],
    scorecard: dict[str, Any],
    trajectory_errors: dict[tuple[str, str], list[dict[str, float | None]]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for scenario, scenario_payload in metrics.items():
        if str(scenario).startswith("_"):
            continue
        for method, payload in method_items(scenario_payload):
            if payload.get("diverged") is not True:
                continue
            traj = trajectory_errors.get((scenario, method), [])
            num_diverged = int(payload.get("num_diverged_trajectories") or 0)
            num_mask_flagged = sum(1 for row in traj if is_mask_flagged_trajectory(row))
            mask_note = mask_comparison_note(num_mask_flagged, num_diverged)
            pos_dist = trajectory_distribution(traj, "pos")
            vel_dist = trajectory_distribution(traj, "vel")
            cases.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metrics_diverged": True,
                    "scorecard_candidate_diverged": scorecard_candidate_flag(scorecard, scenario, method),
                    "canonical_manuscript_table_membership": None,
                    "canonical_manuscript_table_membership_basis": (
                        "Not determined from audit inputs. This audit reads only the "
                        "full-rerun metrics, scorecard, and trajectory-error files; "
                        "it does not infer current manuscript table inclusion."
                    ),
                    "num_trajectories": len(traj),
                    "num_diverged_trajectories": num_diverged,
                    "num_mask_flagged_trajectories": num_mask_flagged,
                    "mask_vs_metrics_diverged_trajectory_count_note": mask_note,
                    "divergence_reason": payload.get("divergence_reason"),
                    "pos_rmse_m": clean_number(payload.get("pos_rmse_m")),
                    "vel_rmse_mps": clean_number(payload.get("vel_rmse_mps")),
                    "median_traj_pos_rmse_m": clean_number(payload.get("median_traj_pos_rmse_m")),
                    "max_traj_pos_rmse_m": clean_number(payload.get("max_traj_pos_rmse_m")),
                    "max_to_median_traj_pos_rmse_ratio": clean_number(
                        payload.get("max_to_median_traj_pos_rmse_ratio")
                    ),
                    "median_traj_vel_rmse_mps": clean_number(payload.get("median_traj_vel_rmse_mps")),
                    "max_traj_vel_rmse_mps": clean_number(payload.get("max_traj_vel_rmse_mps")),
                    "max_to_median_traj_vel_rmse_ratio": clean_number(
                        payload.get("max_to_median_traj_vel_rmse_ratio")
                    ),
                    "failure_conditioned_summary": {
                        "boundary": FAILURE_CONDITIONED_BOUNDARY,
                        "mask_definition": (
                            "A paired trajectory row is mask-flagged when position "
                            f"or velocity RMSE is nonfinite, position RMSE > "
                            f"{POS_EXTREME_RMSE_THRESHOLD_M:.0e} m, or velocity "
                            f"RMSE > {VEL_EXTREME_RMSE_THRESHOLD_MPS:.0e} m/s."
                        ),
                        "num_mask_flagged_trajectories": num_mask_flagged,
                        "num_diverged_trajectories_from_metrics": num_diverged,
                        "mask_vs_metrics_diverged_trajectory_count_note": mask_note,
                        "pos_rmse_m": {
                            "count": pos_dist["count"],
                            "trajectory_row_count": pos_dist["trajectory_row_count"],
                            "nonfinite_count": pos_dist["nonfinite_count"],
                            "median": pos_dist["median"],
                            "p90": pos_dist["p90"],
                            "p95": pos_dist["p95"],
                            "max": pos_dist["max"],
                            "top_values_desc": pos_dist["top_values_desc"],
                            "count_excluding_mask_flagged_trajectories": pos_dist[
                                "count_excluding_mask_flagged_trajectories"
                            ],
                            "mean_pos_rmse_excluding_mask_flagged_trajectories_m": (
                                pos_dist["mean_excluding_mask_flagged_trajectories"]
                            ),
                        },
                        "vel_rmse_mps": {
                            "count": vel_dist["count"],
                            "trajectory_row_count": vel_dist["trajectory_row_count"],
                            "nonfinite_count": vel_dist["nonfinite_count"],
                            "median": vel_dist["median"],
                            "p90": vel_dist["p90"],
                            "p95": vel_dist["p95"],
                            "max": vel_dist["max"],
                            "top_values_desc": vel_dist["top_values_desc"],
                            "count_excluding_mask_flagged_trajectories": vel_dist[
                                "count_excluding_mask_flagged_trajectories"
                            ],
                            "mean_vel_rmse_excluding_mask_flagged_trajectories_mps": (
                                vel_dist["mean_excluding_mask_flagged_trajectories"]
                            ),
                        },
                    },
                }
            )
    return cases


def scorecard_candidate_divergence_by_scenario(scorecard: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for scenario, scenario_payload in scorecard.items():
        if str(scenario).startswith("_") or not isinstance(scenario_payload, dict):
            continue
        flagged = [
            method
            for method, payload in method_items(scenario_payload)
            if payload.get("candidate_diverged") is True
        ]
        if flagged:
            out[str(scenario)] = flagged
    return out


def build_audit(
    *,
    metrics_path: Path,
    scorecard_path: Path,
    trajectory_errors_path: Path,
    generated_utc: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    scorecard = load_json(scorecard_path)
    trajectory_errors = load_trajectory_errors(trajectory_errors_path)
    cases = build_divergence_cases(metrics, scorecard, trajectory_errors)
    methods_by_scenario: dict[str, list[str]] = {}
    for case in cases:
        methods_by_scenario.setdefault(case["scenario"], []).append(case["method"])
    for methods in methods_by_scenario.values():
        methods.sort()
    scenarios = [scenario for scenario in metrics if not str(scenario).startswith("_")]
    scenarios_with_divergence = sorted(methods_by_scenario)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": generated_utc or utc_now_iso(),
        "artifact_type": "full_rerun_divergence_audit",
        "claim_boundary": CLAIM_BOUNDARY,
        "input_paths": {
            "metrics_summary": rel(metrics_path, root),
            "scorecard_summary": rel(scorecard_path, root),
            "trajectory_errors": rel(trajectory_errors_path, root),
        },
        "input_sha256": {
            "metrics_summary": sha256_file(metrics_path),
            "scorecard_summary": sha256_file(scorecard_path),
            "trajectory_errors": sha256_file(trajectory_errors_path),
        },
        "overall_counts": {
            "total_scenarios": len(scenarios),
            "num_scenarios_with_any_divergence": len(scenarios_with_divergence),
            "scenarios_with_any_divergence": scenarios_with_divergence,
            "methods_with_divergence_by_scenario": methods_by_scenario,
            "scorecard_candidate_divergence_by_scenario": scorecard_candidate_divergence_by_scenario(
                scorecard
            ),
            "canonical_manuscript_table_membership_basis": (
                "Conservative: not inferred from these audit inputs. The full rerun "
                "is diagnostic/internal evidence and this audit is not a canonical "
                "manuscript-table replacement."
            ),
        },
        "decision_boundary": {
            "full_rerun_scorecard_candidate_divergence_fails_practical_headline_logic": True,
            "no_learned_positive_from_raw_tiny_wins_or_failure_conditioned_rows": True,
            "failure_conditioned_summaries_are_diagnostic_only": True,
            "boundary_text": LEARNED_POSITIVE_BOUNDARY,
        },
        "divergence_cases": cases,
    }


def fmt(value: Any) -> str:
    number = finite_float(value)
    if number is None:
        return "NA"
    if number == 0:
        return "0"
    if abs(number) >= 1.0e6 or abs(number) < 1.0e-2:
        return f"{number:.3e}"
    return f"{number:.3f}"


def render_markdown(audit: dict[str, Any]) -> str:
    counts = audit["overall_counts"]
    lines = [
        "# Full Rerun Divergence Audit",
        "",
        f"Generated UTC: `{audit['generated_utc']}`",
        f"Schema: `{audit['schema_version']}`",
        "",
        "## Boundary",
        audit["claim_boundary"],
        "",
        FAILURE_CONDITIONED_BOUNDARY,
        "",
        LEARNED_POSITIVE_BOUNDARY,
        "",
        "## Inputs",
        "| Input | Path | SHA-256 |",
        "|---|---|---|",
    ]
    for key, path in audit["input_paths"].items():
        lines.append(f"| `{key}` | `{path}` | `{audit['input_sha256'][key]}` |")
    lines.extend(
        [
            "",
            "## Overall Counts",
            f"- Total scenarios in metrics input: `{counts['total_scenarios']}`",
            f"- Scenarios with any metrics divergence: `{counts['num_scenarios_with_any_divergence']}`",
            "- Divergence is concentrated in: "
            + ", ".join(f"`{name}`" for name in counts["scenarios_with_any_divergence"]),
            "- Canonical manuscript table membership: "
            + counts["canonical_manuscript_table_membership_basis"],
            "",
            "## Divergence Cases",
            "| Scenario | Method | Candidate-diverged in scorecard | n | metrics flagged n | mask flagged n | all-traj pos RMSE [m] | median traj pos [m] | max traj pos [m] | max/median pos | Reason |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in audit["divergence_cases"]:
        lines.append(
            "| `{scenario}` | `{method}` | `{scorecard}` | {n} | {metrics_flagged} | {mask_flagged} | {pos} | {median} | {max_pos} | {ratio} | {reason} |".format(
                scenario=case["scenario"],
                method=case["method"],
                scorecard=case["scorecard_candidate_diverged"],
                n=case["num_trajectories"],
                metrics_flagged=case["num_diverged_trajectories"],
                mask_flagged=case["num_mask_flagged_trajectories"],
                pos=fmt(case["pos_rmse_m"]),
                median=fmt(case["median_traj_pos_rmse_m"]),
                max_pos=fmt(case["max_traj_pos_rmse_m"]),
                ratio=fmt(case["max_to_median_traj_pos_rmse_ratio"]),
                reason=case["divergence_reason"],
            )
        )
    lines.extend(
        [
            "",
            "## Failure-Conditioned Diagnostic Summaries",
            "Percentiles, maxima, and top values remain all-trajectory diagnostic context. "
            "Only the diagnostic mean in this section excludes paired rows selected "
            "by the evaluator-style extreme mask.",
            "",
            "| Scenario | Method | mask flagged n | retained n | median [m] | p90 [m] | p95 [m] | max [m] | mean excl. mask flagged rows [m] | Count diagnostic |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for case in audit["divergence_cases"]:
        pos = case["failure_conditioned_summary"]["pos_rmse_m"]
        summary = case["failure_conditioned_summary"]
        lines.append(
            "| `{scenario}` | `{method}` | {mask_flagged} | {retained} | {median} | {p90} | {p95} | {max_pos} | {mean_excl} | {note} |".format(
                scenario=case["scenario"],
                method=case["method"],
                mask_flagged=summary["num_mask_flagged_trajectories"],
                retained=pos["count_excluding_mask_flagged_trajectories"],
                median=fmt(pos["median"]),
                p90=fmt(pos["p90"]),
                p95=fmt(pos["p95"]),
                max_pos=fmt(pos["max"]),
                mean_excl=fmt(pos["mean_pos_rmse_excluding_mask_flagged_trajectories_m"]),
                note=summary["mask_vs_metrics_diverged_trajectory_count_note"],
            )
        )
    lines.extend(
        [
            "",
            "Failure-conditioned rows are for inspection of tail concentration only. "
            "They use the paired evaluator-style extreme mask rather than a top-N "
            "trim. They are not replacement manuscript metrics, do not redefine "
            "performance, and do not alter the canonical practical-floor decisions.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(audit: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(audit), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_RERUN_DIR / "metrics_summary.json")
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_RERUN_DIR / "scorecard_summary.json")
    parser.add_argument("--trajectory-errors", type=Path, default=DEFAULT_RERUN_DIR / "trajectory_errors.csv")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    audit = build_audit(
        metrics_path=args.metrics,
        scorecard_path=args.scorecard,
        trajectory_errors_path=args.trajectory_errors,
    )
    write_outputs(audit, args.json_out, args.md_out)
    print(
        json.dumps(
            {
                "status": "pass",
                "json": rel(args.json_out),
                "markdown": rel(args.md_out),
                "divergence_case_count": len(audit["divergence_cases"]),
                "scenarios_with_divergence": audit["overall_counts"]["scenarios_with_any_divergence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
