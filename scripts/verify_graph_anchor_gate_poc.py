#!/usr/bin/env python3
"""Verify the GraphAnchorPairGate PoC release archive.

This verifier is intentionally stdlib-only. It extracts the compact
GraphAnchorPairGate package, checks archive payload hashes against the embedded
MANIFEST.json, reads the retained seed-sweep CSVs, and recomputes the bounded
PoC metrics without rerunning training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_REL = "release/spot_od_v1_2_0_graph_anchor_gate_poc.zip"
DEFAULT_JSON_OUT = "results/validation/graph_anchor_gate_poc_verification.json"
DEFAULT_MD_OUT = "results/validation/graph_anchor_gate_poc_verification.md"
MANIFEST_MEMBER = "MANIFEST.json"
SUMMARY_CSV = "results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_summary.csv"
BY_SCENARIO_CSV = "results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_by_scenario.csv"
BY_SEED_CSV = "results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_by_seed.csv"
PAIRED_GAINS_CSV = "results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_paired_seed_gains.csv"
UNCERTAINTY_CSV = "results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_uncertainty_summary.csv"
FLOAT_ABS_TOL = 1.0e-10
EXPECTED = {
    "row_count": 10,
    "row_wins": 9,
    "paired_seed_count": 5,
    "paired_seed_both_scenario_wins": 4,
    "process_noise_shift_test_mean_gain_percent": 7.95663495038935,
    "maneuver_shift_test_mean_gain_percent": 8.05274642630686,
    "seed_19_process_noise_shift_gain_percent": -2.0925251807980216,
}
BOUNDARY = (
    "Local compact-simulator GraphAnchorPairGate PoC on all-step "
    "center-window RMSE for held-out eval trajectories in the process-noise "
    "and maneuver shift scenarios. This is not the primary observed-step "
    "endpoint, not operational precise-reference validation, not independent "
    "third-party reproduction, and not a full raw-data/training rerun."
)


def norm(path: str | Path) -> str:
    return str(path).replace("\\", "/").strip("/")


def repo_path(path: str | Path) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_member_path(member_name: str) -> PurePosixPath:
    normalized = norm(member_name)
    member = PurePosixPath(normalized)
    if member.is_absolute() or any(part in ("", ".", "..") for part in member.parts):
        raise ValueError(f"unsafe archive member path: {member_name!r}")
    return member


def extracted_path(extracted_root: Path, rel_path: str) -> Path:
    member = safe_member_path(rel_path)
    target = (extracted_root / Path(*member.parts)).resolve()
    target.relative_to(extracted_root.resolve())
    return target


def extract_archive(archive_path: Path, destination: Path) -> dict[str, Any]:
    members: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                try:
                    member = safe_member_path(info.filename)
                    rel = member.as_posix()
                    target = extracted_path(destination, rel)
                    data = zf.read(info)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    members[rel] = {
                        "path": rel,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                except Exception as exc:
                    failures.append({"member": info.filename, "problem": repr(exc)})
    except zipfile.BadZipFile:
        failures.append({"member": "", "problem": "bad_zip"})
    return {
        "status": "pass" if not failures else "fail",
        "member_count": len(members),
        "members": members,
        "failures": failures,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except KeyError as exc:
        raise ValueError(f"CSV row missing required column {key!r}") from exc


def parse_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def assert_close(name: str, actual: float, expected: float, failures: list[dict[str, Any]]) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > FLOAT_ABS_TOL:
        failures.append(
            {
                "check": name,
                "expected": expected,
                "actual": actual,
                "abs_delta": abs(actual - expected) if math.isfinite(actual) else None,
                "tolerance": FLOAT_ABS_TOL,
            }
        )


def assert_equal(name: str, actual: Any, expected: Any, failures: list[dict[str, Any]]) -> None:
    if actual != expected:
        failures.append({"check": name, "expected": expected, "actual": actual})


def verify_manifest(extracted_root: Path, extraction: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    manifest_path = extracted_path(extracted_root, MANIFEST_MEMBER)
    if not manifest_path.is_file():
        return {
            "status": "fail",
            "manifest_member": MANIFEST_MEMBER,
            "failures": [{"check": "manifest_present", "problem": "missing"}],
        }
    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        return {
            "status": "fail",
            "manifest_member": MANIFEST_MEMBER,
            "failures": [{"check": "manifest_json", "problem": repr(exc)}],
        }

    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []
        failures.append({"check": "manifest_artifacts", "problem": "artifacts_not_list"})

    members = extraction.get("members", {})
    payload_members = {path: record for path, record in members.items() if path != MANIFEST_MEMBER}
    listed: dict[str, dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            failures.append({"check": "manifest_entry", "problem": "entry_not_object"})
            continue
        path = norm(str(entry.get("path", "")))
        if not path:
            failures.append({"check": "manifest_entry_path", "problem": "empty_path"})
            continue
        if path in listed:
            failures.append({"check": "manifest_entry_path", "path": path, "problem": "duplicate"})
            continue
        listed[path] = entry

    for path, entry in listed.items():
        actual = payload_members.get(path)
        if actual is None:
            failures.append({"check": "manifest_member_present", "path": path, "problem": "missing"})
            continue
        expected_bytes = entry.get("bytes")
        expected_sha = entry.get("sha256")
        if actual.get("bytes") != expected_bytes:
            failures.append(
                {
                    "check": "member_bytes",
                    "path": path,
                    "expected": expected_bytes,
                    "actual": actual.get("bytes"),
                }
            )
        if actual.get("sha256") != expected_sha:
            failures.append(
                {
                    "check": "member_sha256",
                    "path": path,
                    "expected": expected_sha,
                    "actual": actual.get("sha256"),
                }
            )

    missing_from_manifest = sorted(set(payload_members) - set(listed))
    if missing_from_manifest:
        failures.append(
            {
                "check": "manifest_covers_payload_members",
                "problem": "payload_members_not_listed",
                "paths": missing_from_manifest,
            }
        )

    return {
        "status": "pass" if not failures else "fail",
        "package": manifest.get("package"),
        "version": manifest.get("version"),
        "created_utc": manifest.get("created_utc"),
        "source_boundary": manifest.get("source_boundary"),
        "artifact_count": len(listed),
        "payload_member_count": len(payload_members),
        "manifest_member": {
            "path": MANIFEST_MEMBER,
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
            "self_indexed": False,
            "note": "The embedded MANIFEST.json is not self-indexed to avoid a self-referential digest cycle.",
        },
        "failures": failures,
    }


def compute_graph_metrics(extracted_root: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    paths = {
        "summary": extracted_path(extracted_root, SUMMARY_CSV),
        "by_scenario": extracted_path(extracted_root, BY_SCENARIO_CSV),
        "by_seed": extracted_path(extracted_root, BY_SEED_CSV),
        "paired_gains": extracted_path(extracted_root, PAIRED_GAINS_CSV),
        "uncertainty": extracted_path(extracted_root, UNCERTAINTY_CSV),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return {
            "status": "fail",
            "failures": [{"check": "required_csvs_present", "missing": missing}],
            "paths": {name: norm(path.relative_to(extracted_root)) for name, path in paths.items()},
        }

    try:
        summary = read_csv_rows(paths["summary"])
        by_scenario = read_csv_rows(paths["by_scenario"])
        by_seed = read_csv_rows(paths["by_seed"])
        paired_gains = read_csv_rows(paths["paired_gains"])
        uncertainty = read_csv_rows(paths["uncertainty"])
    except Exception as exc:
        return {
            "status": "fail",
            "failures": [{"check": "csv_read", "problem": repr(exc)}],
            "paths": {name: norm(path.relative_to(extracted_root)) for name, path in paths.items()},
        }

    try:
        row_count = len(summary)
        row_wins = sum(1 for row in summary if parse_bool(row["beats_best_candidate"]))
        assert_equal("row_count", row_count, EXPECTED["row_count"], failures)
        assert_equal("row_wins", row_wins, EXPECTED["row_wins"], failures)

        rows_by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
        rows_by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in summary:
            rows_by_scenario[str(row["scenario"])].append(row)
            rows_by_seed[parse_int(row, "seed")].append(row)

        paired_seed_both_wins = 0
        paired_seed_records: list[dict[str, Any]] = []
        for seed in sorted(rows_by_seed):
            seed_rows = rows_by_seed[seed]
            both_win = len(seed_rows) == 2 and all(parse_bool(row["beats_best_candidate"]) for row in seed_rows)
            paired_seed_both_wins += int(both_win)
            paired_seed_records.append(
                {
                    "seed": seed,
                    "rows": len(seed_rows),
                    "both_scenarios_win": both_win,
                    "mean_gain_percent": sum(parse_float(row, "gain_vs_best_candidate_all_step_percent") for row in seed_rows)
                    / len(seed_rows),
                }
            )
        assert_equal("paired_seed_count", len(rows_by_seed), EXPECTED["paired_seed_count"], failures)
        assert_equal(
            "paired_seed_both_scenario_wins",
            paired_seed_both_wins,
            EXPECTED["paired_seed_both_scenario_wins"],
            failures,
        )

        scenario_metrics: dict[str, dict[str, Any]] = {}
        for scenario, scenario_rows in sorted(rows_by_scenario.items()):
            gains = [parse_float(row, "gain_vs_best_candidate_all_step_percent") for row in scenario_rows]
            wins = sum(1 for row in scenario_rows if parse_bool(row["beats_best_candidate"]))
            scenario_metrics[scenario] = {
                "rows": len(scenario_rows),
                "wins": wins,
                "mean_gain_percent": sum(gains) / len(gains),
                "min_gain_percent": min(gains),
                "max_gain_percent": max(gains),
            }

        assert_close(
            "process_noise_shift_test_mean_gain_percent",
            scenario_metrics["process_noise_shift_test"]["mean_gain_percent"],
            EXPECTED["process_noise_shift_test_mean_gain_percent"],
            failures,
        )
        assert_close(
            "maneuver_shift_test_mean_gain_percent",
            scenario_metrics["maneuver_shift_test"]["mean_gain_percent"],
            EXPECTED["maneuver_shift_test_mean_gain_percent"],
            failures,
        )

        seed19_process = [
            row
            for row in summary
            if parse_int(row, "seed") == 19 and row["scenario"] == "process_noise_shift_test"
        ]
        assert_equal("seed_19_process_noise_shift_rows", len(seed19_process), 1, failures)
        if seed19_process:
            assert_close(
                "seed_19_process_noise_shift_gain_percent",
                parse_float(seed19_process[0], "gain_vs_best_candidate_all_step_percent"),
                EXPECTED["seed_19_process_noise_shift_gain_percent"],
                failures,
            )
            assert_equal("seed_19_process_noise_shift_win", parse_bool(seed19_process[0]["beats_best_candidate"]), False, failures)

        by_scenario_records = {row["scenario"]: row for row in by_scenario}
        for scenario, metrics in scenario_metrics.items():
            record = by_scenario_records.get(scenario)
            if record is None:
                failures.append({"check": "by_scenario_contains_scenario", "scenario": scenario})
                continue
            assert_equal(f"by_scenario_{scenario}_rows", parse_int(record, "rows"), metrics["rows"], failures)
            assert_equal(f"by_scenario_{scenario}_wins", parse_int(record, "wins"), metrics["wins"], failures)
            assert_close(
                f"by_scenario_{scenario}_mean_gain_percent",
                parse_float(record, "mean_gain_percent"),
                metrics["mean_gain_percent"],
                failures,
            )

        by_seed_records = {parse_int(row, "seed"): row for row in by_seed}
        paired_csv_records = {parse_int(row, "seed"): row for row in paired_gains}
        for record in paired_seed_records:
            seed = record["seed"]
            by_seed_record = by_seed_records.get(seed)
            paired_record = paired_csv_records.get(seed)
            if by_seed_record is None:
                failures.append({"check": "by_seed_contains_seed", "seed": seed})
            else:
                assert_equal(
                    f"by_seed_{seed}_both_scenarios_win",
                    parse_bool(by_seed_record["both_scenarios_win"]),
                    record["both_scenarios_win"],
                    failures,
                )
                assert_close(
                    f"by_seed_{seed}_mean_gain_percent",
                    parse_float(by_seed_record, "mean_gain_percent"),
                    record["mean_gain_percent"],
                    failures,
                )
            if paired_record is None:
                failures.append({"check": "paired_gains_contains_seed", "seed": seed})
            else:
                assert_equal(
                    f"paired_gains_{seed}_both_scenarios_win",
                    parse_bool(paired_record["both_scenarios_win"]),
                    record["both_scenarios_win"],
                    failures,
                )
                assert_close(
                    f"paired_gains_{seed}_mean_gain_percent",
                    parse_float(paired_record, "mean_gain_percent"),
                    record["mean_gain_percent"],
                    failures,
                )

        uncertainty_records = {row["metric"]: row for row in uncertainty}
        expected_uncertainty = {
            "scenario_seed_row_wins": (EXPECTED["row_wins"], EXPECTED["row_count"]),
            "paired_seed_both_scenario_wins": (
                EXPECTED["paired_seed_both_scenario_wins"],
                EXPECTED["paired_seed_count"],
            ),
        }
        for metric, (successes, trials) in expected_uncertainty.items():
            record = uncertainty_records.get(metric)
            if record is None:
                failures.append({"check": "uncertainty_contains_metric", "metric": metric})
                continue
            assert_equal(f"uncertainty_{metric}_successes", parse_int(record, "successes"), successes, failures)
            assert_equal(f"uncertainty_{metric}_trials", parse_int(record, "trials"), trials, failures)
    except Exception as exc:
        failures.append({"check": "metric_computation", "problem": repr(exc)})
        scenario_metrics = {}
        paired_seed_records = []
        row_count = None
        row_wins = None
        paired_seed_both_wins = None

    return {
        "status": "pass" if not failures else "fail",
        "boundary": BOUNDARY,
        "paths": {name: norm(path.relative_to(extracted_root)) for name, path in paths.items()},
        "summary": {
            "row_count": row_count,
            "row_wins": row_wins,
            "paired_seed_count": len(paired_seed_records),
            "paired_seed_both_scenario_wins": paired_seed_both_wins,
            "scenario_metrics": scenario_metrics,
            "paired_seed_records": paired_seed_records,
        },
        "expected": EXPECTED,
        "tolerance": FLOAT_ABS_TOL,
        "failures": failures,
    }


def build_result(archive_rel: str) -> dict[str, Any]:
    archive_path = repo_path(archive_rel)
    archive_record: dict[str, Any] = {
        "path": norm(archive_path.relative_to(ROOT)) if archive_path.is_relative_to(ROOT) else norm(archive_path),
        "exists": archive_path.is_file(),
        "bytes": archive_path.stat().st_size if archive_path.is_file() else None,
        "sha256": sha256_file(archive_path) if archive_path.is_file() else None,
    }
    if not archive_path.is_file():
        return {
            "status": "fail",
            "scope": "graph_anchor_gate_poc_archive_verification",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "boundary": BOUNDARY,
            "archive": archive_record,
            "checks": {
                "archive_extraction": {"status": "fail", "failures": [{"problem": "archive_missing"}]},
                "manifest_hashes": {"status": "blocked", "failures": [{"problem": "archive_missing"}]},
                "seed_sweep_metrics": {"status": "blocked", "failures": [{"problem": "archive_missing"}]},
            },
        }

    with tempfile.TemporaryDirectory(prefix="graph_anchor_gate_poc_") as td:
        extracted_root = Path(td)
        extraction = extract_archive(archive_path, extracted_root)
        manifest_check = verify_manifest(extracted_root, extraction) if extraction["status"] == "pass" else {
            "status": "blocked",
            "failures": [{"problem": "archive_extraction_failed"}],
        }
        metric_check = compute_graph_metrics(extracted_root) if manifest_check["status"] == "pass" else {
            "status": "blocked",
            "boundary": BOUNDARY,
            "failures": [{"problem": "manifest_hash_check_failed"}],
        }

    checks = {
        "archive_extraction": extraction,
        "manifest_hashes": manifest_check,
        "seed_sweep_metrics": metric_check,
    }
    status = "pass" if all(check.get("status") == "pass" for check in checks.values()) else "fail"
    return {
        "status": status,
        "scope": "graph_anchor_gate_poc_archive_verification",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": BOUNDARY,
        "requires_training": False,
        "archive": archive_record,
        "checks": checks,
    }


def write_reports(result: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checks = result["checks"]
    metrics = checks["seed_sweep_metrics"].get("summary", {})
    scenario_metrics = metrics.get("scenario_metrics", {}) if isinstance(metrics, dict) else {}
    process = scenario_metrics.get("process_noise_shift_test", {}) if isinstance(scenario_metrics, dict) else {}
    maneuver = scenario_metrics.get("maneuver_shift_test", {}) if isinstance(scenario_metrics, dict) else {}
    lines = [
        "# GraphAnchorPairGate PoC Archive Verification",
        "",
        f"Status: **{str(result['status']).upper()}**",
        "",
        "## Boundary",
        result["boundary"],
        "",
        "## Archive",
        f"- Path: `{result['archive']['path']}`",
        f"- Bytes: `{result['archive']['bytes']}`",
        f"- SHA-256: `{result['archive']['sha256']}`",
        "",
        "## Checks",
        f"- ZIP extraction: **{checks['archive_extraction']['status'].upper()}**.",
        f"- Embedded manifest hashes: **{checks['manifest_hashes']['status'].upper()}**.",
        f"- Seed-sweep metric recomputation: **{checks['seed_sweep_metrics']['status'].upper()}**.",
        "",
        "## Recomputed Metrics",
        f"- Scenario-seed rows: `{metrics.get('row_count')}`.",
        f"- Scenario-seed row wins: `{metrics.get('row_wins')}`.",
        f"- Paired seeds winning both scenarios: `{metrics.get('paired_seed_both_scenario_wins')}/{metrics.get('paired_seed_count')}`.",
        f"- Process-shift mean gain: `{process.get('mean_gain_percent')}`%.",
        f"- Maneuver-shift mean gain: `{maneuver.get('mean_gain_percent')}`%.",
        f"- Expected seed-19 process-shift failure: `{EXPECTED['seed_19_process_noise_shift_gain_percent']}`%.",
        "",
        "## Outputs",
        f"- JSON: `{norm(json_out.relative_to(ROOT))}`",
        f"- Markdown: `{norm(md_out.relative_to(ROOT))}`",
    ]
    failures: list[dict[str, Any]] = []
    for check in checks.values():
        failures.extend(check.get("failures", []))
    if failures:
        lines += ["", "## Failures"]
        for failure in failures[:40]:
            lines.append(f"- `{json.dumps(failure, sort_keys=True)}`")
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the GraphAnchorPairGate PoC release archive.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE_REL)
    parser.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", default=DEFAULT_MD_OUT)
    parser.add_argument("--check-only", action="store_true", help="Run checks without writing JSON/Markdown reports.")
    args = parser.parse_args()

    result = build_result(args.archive)
    json_out = repo_path(args.json_out)
    md_out = repo_path(args.md_out)
    if not args.check_only:
        write_reports(result, json_out, md_out)
    summary = {
        "status": result["status"],
        "archive": result["archive"],
        "json": norm(json_out.relative_to(ROOT)) if json_out.is_relative_to(ROOT) else norm(json_out),
        "markdown": norm(md_out.relative_to(ROOT)) if md_out.is_relative_to(ROOT) else norm(md_out),
        "archive_extraction": result["checks"]["archive_extraction"]["status"],
        "manifest_hashes": result["checks"]["manifest_hashes"]["status"],
        "seed_sweep_metrics": result["checks"]["seed_sweep_metrics"]["status"],
        "boundary": BOUNDARY,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
