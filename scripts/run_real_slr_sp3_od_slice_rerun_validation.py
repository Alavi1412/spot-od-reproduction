#!/usr/bin/env python3
"""Offline rerun check for the public LAGEOS CRD/SP3 OD slice.

The check stages the archived public CRD/SP3 inputs from the submitted
``results/real_slr_sp3_od`` record into a separate validation directory,
reruns ``scripts/run_real_slr_sp3_od_validation.py`` without ``--refresh``,
rebuilds the corresponding table from the rerun JSON, and compares the
deterministic claim surface with the submitted canonical record.

It is intentionally bounded: one public precise-reference slice is rerun
through filter recomputation and table reconstruction. This is not full
scientific reproduction, full estimator training, all-table regeneration, live
public-data retrieval, or operational POD validation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "results" / "real_slr_sp3_od"
CANONICAL_JSON = CANONICAL_DIR / "real_slr_sp3_od_validation.json"
CANONICAL_TABLE = ROOT / "paper" / "tables" / "real_slr_sp3_od.tex"
DEFAULT_REPORT_JSON = ROOT / "results" / "validation" / "real_slr_sp3_od_slice_rerun.json"
DEFAULT_REPORT_MD = ROOT / "results" / "validation" / "real_slr_sp3_od_slice_rerun.md"
DEFAULT_RERUN_DIR = ROOT / "results" / "validation" / "real_slr_sp3_od_slice_rerun"
OD_SCRIPT_REL = "scripts/run_real_slr_sp3_od_validation.py"
RERUN_RESULT_NAME = "real_slr_sp3_od_validation.json"
RERUN_TABLE_NAME = "real_slr_sp3_od.tex"

ESTIMATORS = (
    "EKF",
    "UKF (fixed-noise)",
    "AUKF (adaptive)",
    "SP3-IC propagation",
)

SCOPE_BOUNDARY = (
    "One public LAGEOS CRD/SP3 precise-reference OD slice rerun from archived "
    "public inputs through range-only EKF/UKF/AUKF/SP3-IC recomputation and "
    "table reconstruction. This is not full scientific reproduction, not full "
    "estimator training, not all filters/tables, not live public-data "
    "retrieval, and not operational POD validation."
)


def norm(path: str | Path) -> str:
    return str(path).replace("\\", "/").strip()


def rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return norm(p.relative_to(ROOT))
    except ValueError:
        return norm(p)


def repo_path(path: str | Path) -> Path:
    p = Path(norm(path))
    return p if p.is_absolute() else ROOT / p


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": rel(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_table_text(text: str) -> str:
    """Ignore line-ending and final-newline differences only."""
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def extract_public_claim_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic fields that support public claims.

    Timestamps, command provenance, and per-run logging are intentionally
    excluded. The table-text comparison separately covers all rendered claims.
    """
    pooled = payload.get("pooled_held_out_position_rmse_m", {})
    pooled_summary: dict[str, Any] = {}
    for name in ESTIMATORS:
        row = pooled.get(name, {})
        pooled_summary[name] = {
            "n_arcs": row.get("n_arcs"),
            "mean_arc_rms_m": row.get("mean_arc_rms_m"),
            "median_arc_rms_m": row.get("median_arc_rms_m"),
            "arcs_best_of": row.get("arcs_best_of"),
        }

    dbar = payload.get("dbar_external_validation", {})
    ni = dbar.get("no_information_baseline", {})
    return {
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "targets": payload.get("targets"),
        "num_arcs": payload.get("num_arcs"),
        "num_arcs_completed": payload.get("num_arcs_completed"),
        "sp3_analysis_center": payload.get("sp3_analysis_center"),
        "sp3_week_product": payload.get("sp3_week_product"),
        "fixed_station_subset": payload.get("fixed_station_subset"),
        "pooled_held_out_position_rmse_m": pooled_summary,
        "dbar_external_validation": {
            "n_arcs_scored": dbar.get("n_arcs_scored"),
            "n_correct": dbar.get("n_correct"),
            "classification_accuracy": dbar.get("classification_accuracy"),
            "confusion": dbar.get("confusion"),
            "sensitivity": dbar.get("sensitivity"),
            "specificity": dbar.get("specificity"),
            "n_counterproductive_arcs": dbar.get("n_counterproductive_arcs"),
            "n_non_counterproductive_arcs": dbar.get("n_non_counterproductive_arcs"),
            "no_information_baseline": {
                "majority_class": ni.get("majority_class"),
                "majority_class_accuracy": ni.get("majority_class_accuracy"),
                "accuracy_minus_majority": ni.get("accuracy_minus_majority"),
                "beats_majority": ni.get("beats_majority"),
            },
        },
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value[key], child))
        return out
    if isinstance(value, list):
        out = {}
        for idx, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{idx}]"))
        return out
    return {prefix: value}


def compare_claim_summaries(
    canonical: dict[str, Any],
    rerun: dict[str, Any],
) -> dict[str, Any]:
    expected = extract_public_claim_summary(canonical)
    actual = extract_public_claim_summary(rerun)
    expected_flat = _flatten(expected)
    actual_flat = _flatten(actual)
    paths = sorted(set(expected_flat) | set(actual_flat))
    mismatches = [
        {
            "field": path,
            "expected": expected_flat.get(path),
            "actual": actual_flat.get(path),
        }
        for path in paths
        if expected_flat.get(path) != actual_flat.get(path)
    ]
    return {
        "status": "pass" if not mismatches else "fail",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "expected": expected,
        "actual": actual,
    }


def compare_table_text(generated: str, submitted: str) -> dict[str, Any]:
    gen = normalize_table_text(generated)
    sub = normalize_table_text(submitted)
    if gen == sub:
        return {
            "status": "pass",
            "matches_submitted_table": True,
            "normalization": "line endings and final newline ignored",
            "generated_sha256": hashlib.sha256(gen.encode("utf-8")).hexdigest(),
            "submitted_sha256": hashlib.sha256(sub.encode("utf-8")).hexdigest(),
        }
    import difflib

    diff = list(
        difflib.unified_diff(
            sub.splitlines(),
            gen.splitlines(),
            fromfile="submitted",
            tofile="rerun_generated",
            lineterm="",
        )
    )
    return {
        "status": "fail",
        "matches_submitted_table": False,
        "normalization": "line endings and final newline ignored",
        "generated_sha256": hashlib.sha256(gen.encode("utf-8")).hexdigest(),
        "submitted_sha256": hashlib.sha256(sub.encode("utf-8")).hexdigest(),
        "diff_head": diff[:120],
    }


def archived_public_input_names(canonical: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for row in canonical.get("input_digests", []):
        name = row.get("archived_input_id")
        if isinstance(name, str) and name:
            names.add(name)
    if not names:
        for arc in canonical.get("arcs", []):
            for kind in ("crd", "sp3"):
                name = arc.get(kind, {}).get("archived_input_id")
                if isinstance(name, str) and name:
                    names.add(name)
    return sorted(names)


def prepare_rerun_directory(
    *,
    canonical: dict[str, Any],
    source_dir: Path,
    rerun_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rerun_dir = rerun_dir.resolve()
    validation_root = (ROOT / "results" / "validation").resolve()
    if validation_root not in rerun_dir.parents and rerun_dir != validation_root:
        raise ValueError(f"Rerun directory must stay under {rel(validation_root)}")
    if rerun_dir.exists():
        shutil.rmtree(rerun_dir)
    rerun_dir.mkdir(parents=True, exist_ok=True)

    copied_inputs: list[dict[str, Any]] = []
    for name in archived_public_input_names(canonical):
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing archived public input: {rel(source)}")
        dest = rerun_dir / name
        shutil.copy2(source, dest)
        copied_inputs.append(
            {
                "kind": "public_crd_or_sp3_input",
                "source": rel(source),
                "destination": rel(dest),
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
            }
        )

    table_support: list[dict[str, Any]] = []
    calibrator = source_dir / "sp3_residual_calibrator.json"
    if calibrator.is_file():
        dest = rerun_dir / calibrator.name
        shutil.copy2(calibrator, dest)
        table_support.append(
            {
                "kind": "submitted_table_support_record",
                "role": (
                    "Copied so build_real_slr_sp3_od_table(result_path=rerun_json) "
                    "reconstructs the full submitted table text; this support "
                    "record is not rerun by the OD slice wrapper."
                ),
                "source": rel(calibrator),
                "destination": rel(dest),
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
            }
        )
    return copied_inputs, table_support


def run_od_slice(rerun_dir: Path, timeout_s: int) -> dict[str, Any]:
    command = [
        sys.executable,
        OD_SCRIPT_REL,
        "--out-dir",
        rel(rerun_dir),
    ]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return {
        "step": "real_slr_sp3_od_slice_rerun",
        "execution_details_redacted": True,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
        "stdout_line_count": len(stdout.splitlines()),
        "stderr_line_count": len(stderr.splitlines()),
    }


def build_table_from_rerun(rerun_json: Path) -> str:
    scripts_dir = ROOT / "scripts"
    for item in (str(ROOT), str(scripts_dir)):
        if item not in sys.path:
            sys.path.insert(0, item)
    from scripts.build_paper_assets import build_real_slr_sp3_od_table

    return build_real_slr_sp3_od_table(result_path=rerun_json)


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    comparisons = report["comparisons"]
    claim = comparisons["public_claim_summary"]
    table = comparisons["table_text"]
    rerun = report["rerun_artifacts"]["result_json"]
    lines = [
        "# Real SLR/SP3 OD Slice Rerun Validation",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "## Scope Boundary",
        report["scope_boundary"],
        "",
        "## Rerun",
        f"- Step: `{report['rerun_execution']['step']}`",
        f"- Exit code: `{report['rerun_execution']['exit_code']}`",
        "- Execution details: redacted from this reviewer-facing summary.",
        f"- Rerun JSON: `{rerun['path']}`",
        f"- Rerun table: `{report['rerun_artifacts']['table_tex']['path']}`",
        "",
        "## Comparisons",
        f"- Public-claim summary fields: **{claim['status'].upper()}** ({claim['mismatch_count']} mismatches).",
        f"- Generated table text matches submitted table: **{table['status'].upper()}**.",
        "",
        "## Summary",
        f"- Completed arcs: `{claim['actual'].get('num_arcs_completed')}`.",
        f"- DBAR correct/scored: `{claim['actual']['dbar_external_validation'].get('n_correct')}/{claim['actual']['dbar_external_validation'].get('n_arcs_scored')}`.",
        f"- Table text matched: `{table['matches_submitted_table']}`.",
    ]
    if claim["mismatches"]:
        lines += ["", "## Mismatches"]
        for row in claim["mismatches"][:20]:
            lines.append(
                f"- `{row['field']}` expected `{row['expected']}` but got `{row['actual']}`."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_validation_report(args: argparse.Namespace) -> dict[str, Any]:
    canonical_json = repo_path(args.canonical_json)
    canonical_table = repo_path(args.canonical_table)
    source_dir = repo_path(args.source_dir)
    rerun_dir = repo_path(args.rerun_dir)

    canonical = read_json(canonical_json)
    copied_inputs, table_support = prepare_rerun_directory(
        canonical=canonical,
        source_dir=source_dir,
        rerun_dir=rerun_dir,
    )

    rerun_execution = run_od_slice(rerun_dir, timeout_s=args.timeout_s)
    rerun_json = rerun_dir / RERUN_RESULT_NAME
    rerun_table = rerun_dir / RERUN_TABLE_NAME
    comparisons: dict[str, Any] = {
        "public_claim_summary": {
            "status": "blocked",
            "mismatch_count": None,
            "mismatches": [],
        },
        "table_text": {
            "status": "blocked",
            "matches_submitted_table": False,
        },
    }
    if rerun_execution["exit_code"] == 0 and rerun_json.is_file():
        rerun = read_json(rerun_json)
        comparisons["public_claim_summary"] = compare_claim_summaries(
            canonical,
            rerun,
        )
        generated_table = build_table_from_rerun(rerun_json)
        rerun_table.write_text(generated_table, encoding="utf-8")
        submitted_table = canonical_table.read_text(encoding="utf-8")
        comparisons["table_text"] = compare_table_text(
            generated_table,
            submitted_table,
        )

    status = (
        "pass"
        if rerun_execution["exit_code"] == 0
        and comparisons["public_claim_summary"].get("status") == "pass"
        and comparisons["table_text"].get("status") == "pass"
        else "fail"
    )
    return {
        "schema_version": "real_slr_sp3_od_slice_rerun_validation_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "status": status,
        "scope_boundary": SCOPE_BOUNDARY,
        "canonical_artifacts": {
            "result_json": artifact(canonical_json),
            "table_tex": artifact(canonical_table),
            "source_input_dir": rel(source_dir),
        },
        "rerun_artifacts": {
            "result_json": artifact(rerun_json),
            "table_tex": artifact(rerun_table),
            "rerun_dir": rel(rerun_dir),
        },
        "staged_archived_public_inputs": copied_inputs,
        "staged_table_support_records": table_support,
        "rerun_execution": rerun_execution,
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=rel(CANONICAL_DIR))
    parser.add_argument("--canonical-json", default=rel(CANONICAL_JSON))
    parser.add_argument("--canonical-table", default=rel(CANONICAL_TABLE))
    parser.add_argument("--rerun-dir", default=rel(DEFAULT_RERUN_DIR))
    parser.add_argument("--json-out", default=rel(DEFAULT_REPORT_JSON))
    parser.add_argument("--md-out", default=rel(DEFAULT_REPORT_MD))
    parser.add_argument("--timeout-s", type=int, default=1200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_validation_report(args)
    json_out = repo_path(args.json_out)
    md_out = repo_path(args.md_out)
    write_json(json_out, report)
    write_markdown_report(report, md_out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": rel(json_out),
                "markdown": rel(md_out),
                "rerun_json": report["rerun_artifacts"]["result_json"]["path"],
                "table_text_match": report["comparisons"]["table_text"].get(
                    "matches_submitted_table"
                ),
                "summary_match": report["comparisons"][
                    "public_claim_summary"
                ].get("status"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
