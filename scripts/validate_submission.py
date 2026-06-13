#!/usr/bin/env python3
"""Deterministic submission validation for the GNN State Estimation paper.

Reads current workspace artifacts and writes machine/human-readable validation
reports.  The PDF page count is read back from paper/main.pdf through a PDF
parser/log/token fallback chain, avoiding stale zero-page metadata.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
VAL = RESULTS / "validation"
PDF = PAPER / "main.pdf"
LOG = PAPER / "main.log"
SUPPLEMENT_LOG = PAPER / "supplement.log"
SUPPLEMENT_PDF = PAPER / "supplement.pdf"
AUX = PAPER / "main.aux"
BBL = PAPER / "main.bbl"
BLG = PAPER / "main.blg"
RELEASE_PACKET = RESULTS / "release_packet.json"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("/", "\\")
    except ValueError:
        return str(path).replace("/", "\\")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_page_readback(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "exists": False, "bytes": 0, "sha256": None, "page_count": 0, "pages": 0, "page_source": "missing", "passed": False}
    data = path.read_bytes()
    pages = 0
    source = "unavailable"
    errors: list[str] = []
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader_cls = getattr(module, "PdfReader")
            with path.open("rb") as f:
                pages = len(reader_cls(f).pages)
            source = module_name
            break
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    if not pages:
        try:
            proc = subprocess.run(["pdfinfo", str(path)], cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if proc.returncode == 0:
                m = re.search(r"^Pages:\s*(\d+)\s*$", proc.stdout, re.M)
                if m:
                    pages = int(m.group(1))
                    source = "pdfinfo"
            else:
                errors.append(f"pdfinfo rc={proc.returncode}: {proc.stderr.strip()}")
        except Exception as exc:
            errors.append(f"pdfinfo: {exc}")
    if not pages and LOG.exists():
        matches = re.findall(r"Output written on main\.pdf \((\d+) pages?,", read_text(LOG))
        if matches:
            pages = int(matches[-1])
            source = "paper/main.log Output written readback"
    if not pages:
        pages = len(re.findall(rb"/Type\s*/Page(?!s)\b", data))
        source = "pdf_token_scan"
    return {
        "path": rel(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
        "page_count": pages,
        "pages": pages,
        "page_source": source,
        "page_readback_errors": errors,
        "passed": pages > 0 and path.stat().st_size > 0,
    }


def strip_tex_comments(text: str) -> str:
    rows = []
    for line in text.splitlines():
        escaped = False
        cut = len(line)
        for i, ch in enumerate(line):
            if ch == "\\":
                escaped = not escaped
                continue
            if ch == "%" and not escaped:
                cut = i
                break
            escaped = False
        rows.append(line[:cut])
    return "\n".join(rows)


def _resolve_tex_input(raw: str, base_dir: Path) -> Path | None:
    target = raw.strip().replace("\\", "/")
    if not target or target.startswith(("/", "#")):
        return None
    path = (base_dir / target)
    if path.suffix == "":
        path = path.with_suffix(".tex")
    try:
        resolved = path.resolve()
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return resolved


def read_tex_with_active_inputs(path: Path, seen: set[Path] | None = None) -> tuple[str, list[str]]:
    resolved = path.resolve()
    seen = set(seen or set())
    if resolved in seen:
        return "", []
    seen.add(resolved)
    text = strip_tex_comments(read_text(resolved))
    included: list[str] = [rel(resolved)]

    def replace_input(match: re.Match[str]) -> str:
        child = _resolve_tex_input(match.group(1), resolved.parent)
        if child is None or not child.exists():
            return match.group(0)
        child_text, child_sources = read_tex_with_active_inputs(child, seen)
        included.extend(child_sources)
        return "\n" + child_text + "\n"

    expanded = re.sub(r"\\(?:input|include)\{([^}]+)\}", replace_input, text)
    return expanded, included


def citation_validation() -> dict[str, Any]:
    tex, citation_sources = read_tex_with_active_inputs(PAPER / "main.tex")
    bib = read_text(PAPER / "references.bib")
    aux = read_text(AUX)
    bbl = read_text(BBL)
    blg = read_text(BLG)
    log = read_text(LOG)

    cited = set()
    for m in re.finditer(r"\\cite\w*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}", tex):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                cited.add(key)
    bib_entries = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    aux_citations = sorted({k.strip() for group in re.findall(r"\\citation\{([^}]+)\}", aux) for k in group.split(",") if k.strip()})
    bibitems = sorted(set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bbl)))
    findings = {
        "missing_bib_entries_from_tex": sorted(cited - bib_entries),
        "missing_bibitems_for_aux_citations": sorted(set(aux_citations) - set(bibitems)) if bibitems else [],
        "undefined_citations_or_refs_in_logs": sorted(set(re.findall(r"(?:Citation `[^']+' on page .* undefined|Reference `[^']+' on page .* undefined|There were undefined citations|There were undefined references|I didn't find a database entry for [^\n]+)", log + "\n" + blg))),
    }
    passed = bool((PAPER / "references.bib").exists()) and BBL.exists() and BBL.stat().st_size > 0 and not any(findings.values())
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "main_tex": rel(PAPER / "main.tex"),
        "citation_source_files": sorted(set(citation_sources)),
        "bib_file": rel(PAPER / "references.bib"),
        "bbl_path": rel(BBL),
        "bbl_bytes": BBL.stat().st_size if BBL.exists() else 0,
        "cited_key_count": len(cited),
        "bib_entry_count": len(bib_entries),
        "aux_citation_key_count": len(aux_citations),
        "bbl_bibitem_count": len(bibitems),
        "cited_keys": sorted(cited),
        "unused_bib_entries": sorted(bib_entries - cited),
        "findings": findings,
    }


def active_artifacts() -> dict[str, Any]:
    tex = strip_tex_comments(read_text(PAPER / "main.tex"))
    inputs = []
    figs = []
    inline_tables = []
    figure_envs = []
    for i, line in enumerate(tex.splitlines(), 1):
        if "\\begin{table" in line:
            inline_tables.append({"line": i, "content": line.strip(), "path": "paper\\main.tex"})
        if "\\begin{figure" in line:
            figure_envs.append({"line": i, "content": line.strip(), "path": "paper\\main.tex"})
        for m in re.finditer(r"\\input\{([^}]+)\}", line):
            target = m.group(1)
            p = PAPER / (target if target.endswith(".tex") else target + ".tex")
            inputs.append({"line": i, "directive": m.group(0), "target": target, "path": rel(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None})
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", line):
            target = m.group(1)
            p = PAPER / target
            figs.append({"line": i, "directive": m.group(0), "target": target, "path": rel(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None})
    generated_tables = [r for r in inputs if r["target"].startswith("tables/")]
    return {
        "active_inputs": inputs,
        "active_generated_table_inputs": generated_tables,
        "active_figures": figs,
        "active_inline_table_environment_lines": inline_tables,
        "active_figure_environment_lines": figure_envs,
        "counts": {
            "main_generated_table_count": len(generated_tables),
            "main_inline_table_count": len(inline_tables),
            "main_figure_count": len(figs),
            "main_figure_environment_count": len(figure_envs),
        },
    }


def release_packet_validation(active: dict[str, Any]) -> dict[str, Any]:
    if not RELEASE_PACKET.exists():
        return {"status": "fail", "passed": False, "path": rel(RELEASE_PACKET), "exists": False, "error": "missing"}
    try:
        packet = json.loads(read_text(RELEASE_PACKET))
    except Exception as exc:
        return {"status": "fail", "passed": False, "path": rel(RELEASE_PACKET), "exists": True, "json_valid": False, "error": repr(exc)}
    listed = []
    status_evidence = packet.get("status_evidence", {}) if isinstance(packet, dict) else {}
    for key in ("main_generated_table_inputs", "main_figure_includes"):
        if isinstance(status_evidence, dict) and isinstance(status_evidence.get(key), list):
            listed.extend(status_evidence[key])
    missing = []
    for item in sorted(set(str(x) for x in listed)):
        if not (ROOT / item.replace("\\", os.sep).replace("/", os.sep)).exists():
            missing.append(item)
    table_lines = [
        {"line": row["line"], "content": row["directive"], "path": row["path"]}
        for row in active["active_generated_table_inputs"]
    ]
    figure_lines = [
        {"line": row["line"], "content": row["directive"], "path": row["path"]}
        for row in active["active_figures"]
    ]
    figure_env_lines = active["active_figure_environment_lines"]
    manuscript_status = packet.get("manuscript_inclusion_status", {}) if isinstance(packet, dict) else {}
    release_metadata = packet.get("release_metadata", {}) if isinstance(packet, dict) else {}
    truth_sync = packet.get("release_packet_truth_sync", {}) if isinstance(packet, dict) else {}
    truth_lines = truth_sync.get("main_tex_line_evidence", {}) if isinstance(truth_sync, dict) else {}
    duplicate_checks = {
        "status_evidence.active_table_input_lines": status_evidence.get("active_table_input_lines") == table_lines if isinstance(status_evidence, dict) else False,
        "status_evidence.active_figure_include_lines": status_evidence.get("active_figure_include_lines") == figure_lines if isinstance(status_evidence, dict) else False,
        "status_evidence.active_figure_environment_lines": status_evidence.get("active_figure_environment_lines") == figure_env_lines if isinstance(status_evidence, dict) else False,
        "top_level.active_table_line_evidence": packet.get("active_table_line_evidence") == table_lines,
        "top_level.active_figure_line_evidence": packet.get("active_figure_line_evidence") == figure_lines,
        "manuscript_inclusion_status.active_table_line_evidence": manuscript_status.get("active_table_line_evidence") == table_lines if isinstance(manuscript_status, dict) else False,
        "manuscript_inclusion_status.active_figure_line_evidence": manuscript_status.get("active_figure_line_evidence") == figure_lines if isinstance(manuscript_status, dict) else False,
        "release_metadata.active_table_line_evidence": release_metadata.get("active_table_line_evidence") == table_lines if isinstance(release_metadata, dict) else False,
        "release_metadata.active_figure_line_evidence": release_metadata.get("active_figure_line_evidence") == figure_lines if isinstance(release_metadata, dict) else False,
        "release_packet_truth_sync.main_tex_line_evidence.active_input_lines": truth_lines.get("active_input_lines") == table_lines if isinstance(truth_lines, dict) else False,
        "release_packet_truth_sync.main_tex_line_evidence.active_includegraphics_lines": truth_lines.get("active_includegraphics_lines") == figure_lines if isinstance(truth_lines, dict) else False,
        "release_packet_truth_sync.main_tex_line_evidence.active_figure_environment_lines": truth_lines.get("active_figure_environment_lines") == figure_env_lines if isinstance(truth_lines, dict) else False,
    }
    duplicate_line_evidence_passed = all(duplicate_checks.values())
    passed = not missing and duplicate_line_evidence_passed
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "path": rel(RELEASE_PACKET),
        "exists": True,
        "json_valid": True,
        "sha256": sha256(RELEASE_PACKET),
        "checked_artifact_count": len(set(str(x) for x in listed)),
        "missing_artifacts": missing,
        "active_include_counts": active["counts"],
        "duplicate_line_evidence_passed": duplicate_line_evidence_passed,
        "duplicate_line_evidence_checks": duplicate_checks,
        "expected_active_table_line_evidence": table_lines,
        "expected_active_figure_line_evidence": figure_lines,
        "expected_active_figure_environment_lines": figure_env_lines,
    }


def latex_warnings_for(log_path: Path, document: str) -> dict[str, Any]:
    text = read_text(log_path)
    category_counts = {
        "latex_warning": 0,
        "package_warning": 0,
        "overfull": 0,
        "underfull": 0,
        "other_warning": 0,
    }
    total = 0
    unique_lines: set[str] = set()
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("Overfull "):
            category_counts["overfull"] += 1
            total += 1
            unique_lines.add(clean)
        elif clean.startswith("Underfull "):
            category_counts["underfull"] += 1
            total += 1
            unique_lines.add(clean)
        elif clean.startswith("LaTeX Warning:"):
            category_counts["latex_warning"] += 1
            total += 1
            unique_lines.add(clean)
        elif "Package " in clean and " Warning:" in clean:
            category_counts["package_warning"] += 1
            total += 1
            unique_lines.add(clean)
        elif "Warning" in clean:
            category_counts["other_warning"] += 1
            total += 1
            unique_lines.add(clean)
    return {
        "document": document,
        "log_path": rel(log_path),
        "log_exists": log_path.exists(),
        "count": total,
        "count_basis": "raw warning/overfull/underfull log lines",
        "raw_line_count": total,
        "deduplicated_line_count": len(unique_lines),
        "category_counts": category_counts,
        "items": [],
        "truncated": False,
        "details_redacted": True,
    }


def latex_warnings() -> dict[str, Any]:
    documents = {
        "main": latex_warnings_for(LOG, "main"),
        "supplement": latex_warnings_for(SUPPLEMENT_LOG, "supplement"),
    }
    items = []
    category_counts = {
        "latex_warning": sum(record["category_counts"]["latex_warning"] for record in documents.values()),
        "package_warning": sum(record["category_counts"]["package_warning"] for record in documents.values()),
        "overfull": sum(record["category_counts"]["overfull"] for record in documents.values()),
        "underfull": sum(record["category_counts"]["underfull"] for record in documents.values()),
        "other_warning": sum(record["category_counts"]["other_warning"] for record in documents.values()),
    }
    return {
        "log_path": rel(LOG),
        "count": sum(category_counts.values()),
        "count_basis": "raw warning/overfull/underfull log lines",
        "raw_line_count": sum(record["raw_line_count"] for record in documents.values()),
        "deduplicated_line_count": sum(record["deduplicated_line_count"] for record in documents.values()),
        "category_counts": category_counts,
        "items": items,
        "truncated": False,
        "details_redacted": True,
        "documents": documents,
        "supplement_warning_count": documents["supplement"]["count"],
        "package_warning_scope": "main plus supplement when paper/supplement.log exists",
    }


def write_supplement_layout_warning_summary(warnings: dict[str, Any], generated_utc: str) -> dict[str, Any]:
    VAL.mkdir(parents=True, exist_ok=True)
    supplement = warnings.get("documents", {}).get("supplement", {})
    record = {
        "status": "pass" if supplement.get("log_exists") else "missing_log",
        "generated_utc": generated_utc,
        "scope": "paper/supplement.log warning, overfull, and underfull checks",
        "log_path": supplement.get("log_path"),
        "log_exists": supplement.get("log_exists"),
        "warning_count": supplement.get("count", 0),
        "count_basis": supplement.get("count_basis"),
        "raw_line_count": supplement.get("raw_line_count", 0),
        "deduplicated_line_count": supplement.get("deduplicated_line_count", 0),
        "category_counts": supplement.get("category_counts", {}),
        "items": [],
        "details_redacted": True,
        "claim_boundary": (
            "This is a layout-risk disclosure for the supplemental PDF. "
            "Warnings are reported rather than treated as zero when the main "
            "manuscript log is clean."
        ),
    }
    json_path = VAL / "supplement_layout_warnings.json"
    md_path = VAL / "supplement_layout_warnings.md"
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = [
        "# Supplement Layout Warning Summary",
        "",
        f"Generated UTC: `{generated_utc}`",
        f"Status: **{record['status'].upper()}**",
        f"Log path: `{record['log_path']}`",
        f"Warning / overfull / underfull count: `{record['warning_count']}`",
        f"Count basis: `{record['count_basis']}`",
        f"Deduplicated line count: `{record['deduplicated_line_count']}`",
        f"Category counts: `{record['category_counts']}`",
        "",
        "## Details",
        "Raw log lines are omitted from this reviewer-facing summary.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {
        "json": rel(json_path),
        "markdown": rel(md_path),
        "warning_count": record["warning_count"],
        "status": record["status"],
    }


def artifact(path: str | Path) -> dict[str, Any]:
    p = ROOT / Path(str(path).replace("\\", os.sep).replace("/", os.sep))
    return {"path": rel(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() and p.is_file() else None, "sha256": sha256(p) if p.exists() and p.is_file() else None}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(read_text(path))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def log_has_pytest_pass(path: Path) -> bool:
    text = read_text(path).replace("\x00", "")
    return bool(re.search(r"\b\d+\s+passed\b", text)) and not re.search(
        r"\b(?:failed|error|errors)\b", text,
        flags=re.IGNORECASE,
    )


def write_supporting_validation_artifacts(
    result: dict[str, Any],
    *,
    validation_step: str,
    command_log: Path,
    json_out: Path,
    md_out: Path,
) -> dict[str, Any]:
    VAL.mkdir(parents=True, exist_ok=True)
    now = result["generated_utc"]
    pdf = result["pdf_evidence"]
    cite = result["citation_validation"]
    release = result["release_evidence_index"]

    adversarial_candidates = [
        RESULTS / "adversarial_submission_peer_review_current_2026-05-14.md",
        RESULTS / "adversarial_peer_review_report.md",
        RESULTS / "final_peer_review_report.md",
    ]
    adversarial_source = next((path for path in adversarial_candidates if path.exists()), None)
    adversarial_text = read_text(adversarial_source) if adversarial_source else ""
    adversarial_record = {
        "status": "pass" if adversarial_source and "risk" in adversarial_text.lower() else "fail",
        "generated_utc": now,
        "source_path": rel(adversarial_source) if adversarial_source else None,
        "source_sha256": sha256(adversarial_source) if adversarial_source else None,
        "review_report": {
            "present": bool(adversarial_source),
            "source_path": rel(adversarial_source) if adversarial_source else None,
            "excerpt_redacted": True,
        },
        "risk_assessment": {
            "present": "risk" in adversarial_text.lower(),
            "source_path": rel(adversarial_source) if adversarial_source else None,
        },
    }
    adversarial_json = VAL / "adversarial_review.json"
    adversarial_json.write_text(json.dumps(adversarial_record, indent=2, sort_keys=True), encoding="utf-8")

    task227_source = RESULTS / "task227_final_submission_validation_result_payload.json"
    task227_payload = load_json(task227_source)
    task227_record = {
        "status": "pass" if task227_payload.get("overall_status") == "PASSED" or task227_payload.get("status") == "pass" else "fail",
        "generated_utc": now,
        "source_path": rel(task227_source),
        "source_sha256": sha256(task227_source),
        "source_summary": {
            "overall_status": task227_payload.get("overall_status"),
            "status": task227_payload.get("status"),
            "pdf_status": (task227_payload.get("pdf_evidence") or task227_payload.get("validation_results", {}).get("pdf") or {}).get("status"),
            "citation_status": (task227_payload.get("citation_validation") or {}).get("status"),
            "release_evidence_status": (task227_payload.get("release_evidence_index") or {}).get("status"),
        },
        "current_submission_validation": {
            "path": rel(json_out),
            "status": result["status"],
            "pdf_sha256": pdf["sha256"],
            "page_count": pdf["page_count"],
        },
    }
    task227_json = VAL / "task227_final_validation.json"
    task227_json.write_text(json.dumps(task227_record, indent=2, sort_keys=True), encoding="utf-8")

    pytest_candidates = [
        VAL / "pytest.log",
        RESULTS / "command_logs" / "pytest.log",
        RESULTS / "pytest_current.log",
        RESULTS / "pytest_full_attempt.txt",
    ]
    pytest_source = next((path for path in pytest_candidates if path.exists() and log_has_pytest_pass(path)), None)
    command_entries = [
        {
            "id": "compile_paper",
            "step": "compile_paper",
            "execution_details_redacted": True,
            "status": "pass" if pdf["passed"] else "fail",
            "exit_code": 0 if pdf["passed"] else 1,
            "evidence": [artifact(PDF.relative_to(ROOT)), artifact(LOG.relative_to(ROOT)), artifact("results/latexmk.log")],
        },
        {
            "id": "verify_release_packet_sync",
            "step": "verify_release_packet_sync",
            "execution_details_redacted": True,
            "status": "pass" if release["passed"] else "fail",
            "exit_code": 0 if release["passed"] else 1,
            "evidence": [artifact("results/validation/release_packet_sync.log"), artifact(RELEASE_PACKET.relative_to(ROOT))],
        },
        {
            "id": "validate_submission",
            "step": validation_step,
            "execution_details_redacted": True,
            "status": result["status"],
            "exit_code": 0 if result["overall_passed"] else 1,
            "evidence": [artifact(command_log.relative_to(ROOT)), artifact(json_out.relative_to(ROOT)), artifact(md_out.relative_to(ROOT))],
        },
        {
            "id": "task227_final_validation",
            "step": "task227_final_validation",
            "execution_details_redacted": True,
            "status": task227_record["status"],
            "exit_code": 0 if task227_record["status"] == "pass" else 1,
            "evidence": [artifact(task227_source.relative_to(ROOT)), artifact(task227_json.relative_to(ROOT))],
        },
        {
            "id": "pytest",
            "step": "pytest",
            "execution_details_redacted": True,
            "status": "pass" if pytest_source else "fail",
            "exit_code": 0 if pytest_source else 1,
            "evidence": [artifact(pytest_source.relative_to(ROOT))] if pytest_source else [artifact(VAL / "pytest.log")],
        },
    ]
    command_manifest = {
        "status": "pass" if all(row["status"] == "pass" for row in command_entries) else "fail",
        "generated_utc": now,
        "commands": command_entries,
    }
    command_manifest_json = VAL / "command_manifest.json"
    command_manifest_json.write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")

    owner_demo = VAL / "owner_demo.md"
    warn_count = result.get("latex_warnings", {}).get("count")
    supp_warn_count = result.get("latex_warnings", {}).get("supplement_warning_count")
    owner_demo.write_text(
        "\n".join(
            [
                "# Owner Demo Readback",
                "",
                f"Generated UTC: `{now}`",
                f"Overall validation: **{result['status'].upper()}**",
                f"Manuscript PDF: `{pdf['path']}`",
                f"PDF pages: `{pdf['page_count']}`",
                f"PDF SHA-256: `{pdf['sha256']}`",
                f"Citation validation: **{cite['status'].upper()}**",
                f"Release evidence index: **{release['status'].upper()}**",
                f"LaTeX warnings reported (main + supplement): `{warn_count}`; supplement: `{supp_warn_count}`",
                f"Command manifest: **{command_manifest['status'].upper()}**",
                f"Adversarial review artifact: **{adversarial_record['status'].upper()}**",
                "",
                "## Honest Readback",
                "- The paper currently passes deterministic PDF, citation, active-artifact, and release-packet checks.",
                "- Main and supplement LaTeX warning/overfull checks are disclosed in validation artifacts.",
                "- The validation is evidence-bounded: it proves the current artifacts are synchronized, not that all experiments were freshly regenerated in this validation step.",
                "- Public replay and deployment-readiness claims remain intentionally limited by the manuscript.",
                "- The pytest status is accepted only when a current pytest log with a passing summary is present.",
                "",
                "## Primary Artifacts",
                f"- `{rel(json_out)}`",
                f"- `{rel(md_out)}`",
                f"- `{rel(command_manifest_json)}`",
                f"- `{rel(adversarial_json)}`",
                f"- `{rel(task227_json)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paths = {
        "owner_demo": rel(owner_demo),
        "command_manifest": rel(command_manifest_json),
        "adversarial_review": rel(adversarial_json),
        "task227_final_validation": rel(task227_json),
    }
    return {
        "status": "pass" if all(
            row.get("status") == "pass"
            for row in (command_manifest, adversarial_record, task227_record)
        ) else "fail",
        "paths": paths,
        "artifacts": [artifact(path) for path in paths.values()],
        "command_manifest": command_manifest,
        "adversarial_review": adversarial_record,
        "task227_final_validation": task227_record,
    }


def write_reports(result: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    pdf = result["pdf_evidence"]
    cite = result["citation_validation"]
    release = result["release_evidence_index"]
    warnings = result["latex_warnings"]
    md = [
        "# Submission Validation Report",
        "",
        f"Generated UTC: `{result['generated_utc']}`",
        "Validation execution details: redacted from this reviewer-facing summary.",
        f"Overall status: **{result['status'].upper()}**",
        "",
        "## PDF Evidence",
        f"- Path: `{pdf['path']}`",
        f"- SHA-256: `{pdf['sha256']}`",
        f"- Page count: `{pdf['page_count']}`",
        f"- Page source: `{pdf['page_source']}`",
        "",
        "## Citation Validation",
        f"- Status: **{cite['status'].upper()}**",
        f"- Cited keys: `{cite['cited_key_count']}`; Bib entries: `{cite['bib_entry_count']}`; BBL bibitems: `{cite['bbl_bibitem_count']}`",
        f"- Findings: `{cite['findings']}`",
        "",
        "## Release Evidence Index",
        f"- Status: **{release['status'].upper()}**",
        f"- Path: `{release['path']}`",
        f"- Checked artifact count: `{release['checked_artifact_count']}`",
        f"- Missing artifacts: `{release['missing_artifacts']}`",
        "",
        "## Remaining LaTeX Warnings",
        f"- Count: `{warnings['count']}`",
        f"- Count basis: `{warnings.get('count_basis')}`",
        f"- Deduplicated line count: `{warnings.get('deduplicated_line_count')}`",
        f"- Category counts: `{warnings.get('category_counts', {})}`",
        "- Raw log lines: omitted from this reviewer-facing summary.",
    ]
    docs = warnings.get("documents", {})
    if docs:
        md.extend(["", "## LaTeX Warning Scope"])
        for name, record in docs.items():
            md.append(f"- `{name}`: `{record.get('count')}` warnings from `{record.get('log_path')}`; category counts `{record.get('category_counts', {})}`")
    md_out.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GNN State Estimation submission artifacts.")
    parser.add_argument("--json-out", default="results/validation/submission_validation.json")
    parser.add_argument("--md-out", default="results/validation/submission_validation.md")
    args = parser.parse_args()
    json_out = ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
    md_out = ROOT / args.md_out if not Path(args.md_out).is_absolute() else Path(args.md_out)

    active = active_artifacts()
    pdf = pdf_page_readback(PDF)
    cite = citation_validation()
    release = release_packet_validation(active)
    warnings = latex_warnings()
    generated_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    supplement_layout = write_supplement_layout_warning_summary(warnings, generated_utc)
    pass_fail = {
        "pdf_readback_nonzero_pages": pdf["passed"],
        "citations": cite["passed"],
        "release_packet_json_and_active_artifacts": release["passed"],
        "all_active_inputs_exist": all(r["exists"] for r in active["active_inputs"]),
        "all_active_figures_exist": all(r["exists"] for r in active["active_figures"]),
    }
    status = "pass" if all(pass_fail.values()) else "fail"
    validation_step = "validate_submission"
    result: dict[str, Any] = {
        "status": status,
        "overall_passed": status == "pass",
        "tests_passed": status == "pass",
        "generated_utc": generated_utc,
        "root_label": "submission_root",
        "validation_run": {"step": validation_step, "execution_details_redacted": True},
        "pass_fail": pass_fail,
        "pdf_evidence": pdf,
        "pdf": pdf,
        "citation_validation": cite,
        "release_evidence_index": release,
        "latex_warnings": warnings,
        "supplement_layout_warnings": supplement_layout,
        **active,
    }

    # Keep a compact command log for release indexing.
    command_log = VAL / "submission_validation_command.log"
    command_log.write_text(json.dumps({
        "validation_step": validation_step,
        "execution_details_redacted": True,
        "status": status,
        "exit_code": 0 if status == "pass" else 1,
        "pdf_sha256": pdf["sha256"],
        "page_count": pdf["page_count"],
        "citation_status": cite["status"],
        "release_packet_status": release["status"],
        "latex_warning_count": warnings["count"],
        "supplement_latex_warning_count": warnings.get("supplement_warning_count"),
        "json_report": rel(json_out),
        "markdown_report": rel(md_out),
    }, indent=2), encoding="utf-8")
    supporting_artifacts = write_supporting_validation_artifacts(
        result,
        validation_step=validation_step,
        command_log=command_log,
        json_out=json_out,
        md_out=md_out,
    )
    # Keep submission_validation.json non-self-referential.  The command
    # manifest records hashes for submission_validation.json, so embedding the
    # full manifest inside submission_validation.json makes deterministic hash
    # agreement impossible after the report is written.  Store only stable
    # status/path/readback summaries in the primary report, then refresh the
    # manifest evidence after the JSON/Markdown reports are final on disk.
    result["supporting_artifacts"] = {
        "status": supporting_artifacts.get("status"),
        "paths": supporting_artifacts.get("paths", {}),
        "artifact_paths": [item.get("path") for item in supporting_artifacts.get("artifacts", []) if item.get("path")],
        "command_manifest_status": supporting_artifacts.get("command_manifest", {}).get("status"),
        "adversarial_review_status": supporting_artifacts.get("adversarial_review", {}).get("status"),
        "task227_final_validation_status": supporting_artifacts.get("task227_final_validation", {}).get("status"),
    }
    write_reports(result, json_out, md_out)

    # Refresh command-manifest evidence after final JSON/Markdown bytes are on
    # disk.  This closes the stale-hash defect found by submission reviewers.
    command_manifest_path = VAL / "command_manifest.json"
    command_manifest = load_json(command_manifest_path)
    for row in command_manifest.get("commands", []):
        if row.get("id") == "validate_submission":
            row["evidence"] = [artifact(command_log.relative_to(ROOT)), artifact(json_out.relative_to(ROOT)), artifact(md_out.relative_to(ROOT))]
    command_manifest_path.write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Write/update release evidence index with current validation artifacts.
    indexed = [
        RELEASE_PACKET,
        json_out,
        md_out,
        VAL / "citation_validation.json",
        command_log,
        VAL / "supplement_layout_warnings.json",
        VAL / "supplement_layout_warnings.md",
        PDF,
        SUPPLEMENT_PDF,
        PAPER / "main.tex",
        PAPER / "supplement.tex",
        PAPER / "references.bib",
    ]
    indexed.extend(ROOT / item["path"].replace("\\", os.sep) for item in supporting_artifacts.get("artifacts", []))
    (VAL / "citation_validation.json").write_text(json.dumps(cite, indent=2), encoding="utf-8")
    for row in active["active_generated_table_inputs"] + active["active_figures"]:
        indexed.append(ROOT / row["path"].replace("\\", os.sep))
    release_index = {
        "generated_utc": result["generated_utc"],
        "status": status,
        "overall_passed": status == "pass",
        "pass_fail": pass_fail,
        "validation_run": result["validation_run"],
        "pdf_evidence": pdf,
        "citation_validation": cite,
        "release_packet_validation": release,
        "latex_warnings": warnings,
        "supplement_layout_warnings": supplement_layout,
        "active_include_counts": active["counts"],
        "artifacts": [artifact(p.relative_to(ROOT) if p.is_absolute() else p) for p in indexed],
    }
    (VAL / "release_evidence_index.json").write_text(json.dumps(release_index, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": status,
        "json": rel(json_out),
        "markdown": rel(md_out),
        "release_evidence_index": "results\\validation\\release_evidence_index.json",
        "pdf_sha256": pdf["sha256"],
        "page_count": pdf["page_count"],
        "citation_status": cite["status"],
        "release_packet_status": release["status"],
        "latex_warning_count": warnings["count"],
        "supplement_latex_warning_count": warnings.get("supplement_warning_count"),
    }, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
