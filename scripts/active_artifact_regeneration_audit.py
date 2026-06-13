#!/usr/bin/env python3
"""Active manuscript artifact regeneration audit for SPOT-OD/GNN State Estimation.

This script is intentionally conservative: it reads the currently active, uncommented
paper/main.tex table inputs and figure includes, records their line references and
checksums, attempts to discover and run a local regeneration harness when one is
explicitly available, and then compares before/after artifact checksums and extracted
TeX table numeric tokens.

Outputs:
  results/active_regeneration_audit.json
  results/active_regeneration_audit.md
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path.cwd()
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"
RESULTS = ROOT / "results"
OUT_JSON = RESULTS / "active_regeneration_audit.json"
OUT_MD = RESULTS / "active_regeneration_audit.md"
RELEASE_PACKET = RESULTS / "release_packet.json"

TABLE_RE = re.compile(r"\\input\{([^}]+)\}")
FIG_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("/", "\\")
    except Exception:
        return str(path)


def file_record(path: Path, kind: str, line: Optional[int] = None, tex: Optional[str] = None) -> Dict[str, Any]:
    exists = path.exists()
    rec: Dict[str, Any] = {
        "kind": kind,
        "path": rel(path),
        "active_main_tex_line": line,
        "main_tex_statement": tex,
        "exists": exists,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "modified_utc": _dt.datetime.fromtimestamp(path.stat().st_mtime, _dt.timezone.utc).isoformat(timespec="seconds") if exists and path.is_file() else None,
    }
    if exists and path.suffix.lower() == ".tex":
        text = path.read_text(encoding="utf-8", errors="replace")
        rec["tex_numeric_token_count"] = len(NUM_RE.findall(text))
        rec["tex_numeric_tokens"] = NUM_RE.findall(text)
        rec["line_count"] = text.count("\n") + (1 if text else 0)
    return rec


def resolve_input(raw: str) -> Path:
    p = (PAPER / raw)
    if p.suffix:
        return p
    if p.exists():
        return p
    return p.with_suffix(".tex")


def resolve_figure(raw: str) -> Path:
    p = PAPER / raw
    if p.suffix:
        return p
    for ext in [".pdf", ".png", ".jpg", ".jpeg"]:
        q = p.with_suffix(ext)
        if q.exists():
            return q
    return p


def active_artifacts() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not MAIN.exists():
        return [], [], [{"severity": "error", "message": "paper/main.tex not found", "path": rel(MAIN)}]
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    other_inputs: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for i, line in enumerate(MAIN.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("%"):
            continue
        for m in TABLE_RE.finditer(line):
            raw = m.group(1)
            path = resolve_input(raw)
            stmt = line.strip()
            kind = "active_table_input" if "tables/" in raw.replace("\\", "/") else "active_tex_input"
            rec = file_record(path, kind, i, stmt)
            if kind == "active_table_input":
                tables.append(rec)
            else:
                other_inputs.append(rec)
        for m in FIG_RE.finditer(line):
            raw = m.group(1)
            figures.append(file_record(resolve_figure(raw), "active_figure_include", i, line.strip()))
    return tables, figures, other_inputs + warnings


def read_release_packet() -> Dict[str, Any]:
    if not RELEASE_PACKET.exists():
        return {"exists": False, "path": rel(RELEASE_PACKET)}
    try:
        data = json.loads(RELEASE_PACKET.read_text(encoding="utf-8"))
        return {"exists": True, "path": rel(RELEASE_PACKET), "sha256": sha256(RELEASE_PACKET), "data": data}
    except Exception as e:
        return {"exists": True, "path": rel(RELEASE_PACKET), "sha256": sha256(RELEASE_PACKET), "parse_error": repr(e)}


def iter_strings(obj: Any, key_path: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_strings(v, f"{key_path}.{k}" if key_path else str(k))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            yield from iter_strings(v, f"{key_path}[{idx}]")
    elif isinstance(obj, str):
        yield key_path, obj


def discover_harness(release: Dict[str, Any]) -> Dict[str, Any]:
    """Find an explicit local regeneration command.

    Priority:
    1. ACTIVE_REGEN_COMMAND environment variable.
    2. Known project script names if present.
    3. Command-like strings in release_packet with keys containing regenerate/harness/command.

    The audit only executes local python/make commands. Everything else is recorded as
    discovered but not executed.
    """
    env_cmd = os.environ.get("ACTIVE_REGEN_COMMAND")
    candidates: List[Dict[str, Any]] = []
    if env_cmd:
        candidates.append({"source": "ACTIVE_REGEN_COMMAND", "command": env_cmd, "confidence": "explicit_environment"})

    script_candidates = [
        "scripts/regenerate_active_artifacts.py",
        "scripts/regenerate_paper_artifacts.py",
        "scripts/regenerate_manuscript_artifacts.py",
        "scripts/regenerate_release_artifacts.py",
        "scripts/build_paper_artifacts.py",
        "scripts/make_paper_artifacts.py",
        "scripts/generate_paper_artifacts.py",
        "scripts/generate_tables_and_figures.py",
        "scripts/generate_tables.py",
        "scripts/generate_figures.py",
        "scripts/reproduce_active_artifacts.py",
        "scripts/audit_active_artifacts.py",
    ]
    self_path = Path(__file__).resolve()
    for s in script_candidates:
        p = ROOT / s
        if p.exists() and p.resolve() != self_path:
            candidates.append({"source": "known_script_name", "path": s, "command": f"{shlex.quote(sys.executable)} {shlex.quote(s)}", "confidence": "filename_match"})

    if release.get("exists") and isinstance(release.get("data"), (dict, list)):
        for kp, val in iter_strings(release["data"]):
            low = (kp + " " + val).lower()
            if any(t in low for t in ["regenerat", "harness", "generate", "reproduce"]) and any(t in low for t in ["python", "make", ".py"]):
                candidates.append({"source": "release_packet", "key_path": kp, "command": val, "confidence": "release_packet_command_like_string"})

    return {"candidates": candidates, "selected": candidates[0] if candidates else None}


def safe_to_run(command: str) -> Tuple[bool, str]:
    try:
        parts = shlex.split(command, posix=(os.name != "nt"))
    except Exception as e:
        return False, f"cannot parse command: {e!r}"
    if not parts:
        return False, "empty command"
    exe = Path(parts[0]).name.lower()
    joined = " ".join(parts).lower()
    if exe in {"python", "python.exe", "py", "py.exe"} or exe.startswith("python"):
        return True, "local python command"
    if Path(parts[0]).resolve() == Path(sys.executable).resolve():
        return True, "current python executable"
    if exe in {"make", "make.exe"} and any(t in joined for t in ["artifact", "figure", "table", "paper", "regen", "reproduce"]):
        return True, "make artifact/paper target"
    return False, "not in conservative allowlist"


def snapshot(paths: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in paths:
        p = ROOT / rec["path"]
        out[rec["path"]] = file_record(p, rec["kind"], rec.get("active_main_tex_line"), rec.get("main_tex_statement"))
    return out


def run_harness(selected: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not selected:
        return {
            "attempted": False,
            "status": "no_explicit_harness_found",
            "blocker": "No ACTIVE_REGEN_COMMAND and no recognized local regeneration script/command discovered in the current workspace.",
        }
    cmd = selected.get("command")
    ok, reason = safe_to_run(cmd)
    if not ok:
        return {"attempted": False, "status": "not_run_conservative_allowlist", "selected": selected, "blocker": reason}
    started = now_iso()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), shell=True, text=True, capture_output=True, timeout=900)
        return {
            "attempted": True,
            "status": "completed" if proc.returncode == 0 else "failed",
            "selected": selected,
            "command": cmd,
            "started_utc": started,
            "ended_utc": now_iso(),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as e:
        return {"attempted": True, "status": "timeout", "selected": selected, "command": cmd, "started_utc": started, "ended_utc": now_iso(), "blocker": repr(e)}
    except Exception as e:
        return {"attempted": True, "status": "error", "selected": selected, "command": cmd, "started_utc": started, "ended_utc": now_iso(), "blocker": repr(e)}


def compare(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(set(before) | set(after)):
        b = before.get(path, {})
        a = after.get(path, {})
        row: Dict[str, Any] = {
            "path": path,
            "kind": a.get("kind") or b.get("kind"),
            "active_main_tex_line": a.get("active_main_tex_line") or b.get("active_main_tex_line"),
            "before_exists": b.get("exists"),
            "after_exists": a.get("exists"),
            "before_sha256": b.get("sha256"),
            "after_sha256": a.get("sha256"),
            "checksum_match": b.get("sha256") == a.get("sha256"),
            "before_size_bytes": b.get("size_bytes"),
            "after_size_bytes": a.get("size_bytes"),
        }
        if b.get("tex_numeric_tokens") is not None or a.get("tex_numeric_tokens") is not None:
            row["tex_numeric_tokens_match"] = b.get("tex_numeric_tokens") == a.get("tex_numeric_tokens")
            row["before_tex_numeric_token_count"] = b.get("tex_numeric_token_count")
            row["after_tex_numeric_token_count"] = a.get("tex_numeric_token_count")
        rows.append(row)
    return rows


def build_pdf_evidence() -> Dict[str, Any]:
    # This audit does not rebuild the manuscript by default. It records whether an
    # existing PDF is present; a separate build command can be supplied externally.
    candidates = [PAPER / "main.pdf", ROOT / "main.pdf"]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return {"manuscript_rebuilt_by_this_audit": False, "existing_pdf": file_record(existing[0], "existing_pdf")}
    return {"manuscript_rebuilt_by_this_audit": False, "existing_pdf": None, "status": "no_pdf_rebuild_requested_or_attempted"}


def sources_and_citations(release: Dict[str, Any]) -> Dict[str, Any]:
    bib_files = sorted([p for p in PAPER.glob("*.bib")] + [p for p in ROOT.glob("*.bib")])
    bib_records = []
    for p in bib_files:
        text = p.read_text(encoding="utf-8", errors="replace")
        keys = re.findall(r"@\w+\s*\{\s*([^,]+)", text)
        bib_records.append({"path": rel(p), "sha256": sha256(p), "entry_count": len(keys), "sample_keys": keys[:20]})
    command_sources = [
        {"path": "scripts/active_artifact_regeneration_audit.py", "description": "audit command executed from project root"},
        {"path": "paper/main.tex", "description": "source of active table/figure line references", "sha256": sha256(MAIN)},
        {"path": "results/release_packet.json", "description": "release/evidence index when present", "sha256": release.get("sha256"), "exists": release.get("exists")},
    ]
    dataset_sources: List[Dict[str, Any]] = []
    if release.get("exists") and isinstance(release.get("data"), (dict, list)):
        for kp, val in iter_strings(release["data"]):
            low = kp.lower()
            if any(t in low for t in ["data", "dataset", "source", "citation", "reference"]):
                dataset_sources.append({"key_path": kp, "value": val[:500]})
    return {"bibliography_files": bib_records, "command_sources": command_sources, "dataset_or_release_sources_from_release_packet": dataset_sources[:100]}


def write_markdown(audit: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Active Table and Figure Regeneration Audit")
    lines.append("")
    lines.append(f"Generated UTC: `{audit['generated_utc']}`")
    lines.append(f"Project root: `{audit['project_root']}`")
    lines.append("")
    hv = audit["harness_run"]
    lines.append("## Harness status")
    lines.append(f"- Status: `{hv.get('status')}`")
    lines.append(f"- Attempted: `{hv.get('attempted')}`")
    if hv.get("command"):
        lines.append(f"- Command: `{hv.get('command')}`")
    if hv.get("blocker"):
        lines.append(f"- Blocker: {hv.get('blocker')}")
    if hv.get("returncode") is not None:
        lines.append(f"- Return code: `{hv.get('returncode')}`")
    lines.append("")
    lines.append("## Active artifact checksum comparison")
    lines.append("")
    lines.append("| Kind | main.tex line | Path | Exists | SHA-256 before | SHA-256 after | Match | Numeric tokens match |")
    lines.append("|---|---:|---|---:|---|---|---:|---:|")
    for r in audit["comparison"]:
        lines.append("| {kind} | {line} | `{path}` | {exists} | `{b}` | `{a}` | {m} | {nm} |".format(
            kind=r.get("kind"),
            line=r.get("active_main_tex_line") or "",
            path=r.get("path"),
            exists=r.get("after_exists"),
            b=(r.get("before_sha256") or "missing")[:16],
            a=(r.get("after_sha256") or "missing")[:16],
            m=r.get("checksum_match"),
            nm=r.get("tex_numeric_tokens_match", "n/a"),
        ))
    lines.append("")
    lines.append("## Non-regenerable or blocker evidence")
    for item in audit["status_evidence"].get("blockers", []):
        lines.append(f"- **{item.get('severity','info')}**: {item.get('message')}")
    if not audit["status_evidence"].get("blockers"):
        lines.append("- No blockers recorded by the audit script.")
    lines.append("")
    lines.append("## PDF evidence")
    lines.append("```json")
    lines.append(json.dumps(audit["pdf_evidence"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Sources / citations")
    lines.append("```json")
    lines.append(json.dumps(audit["citations"], indent=2)[:12000])
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    tables, figures, other = active_artifacts()
    tracked = tables + figures
    before = snapshot(tracked)
    release = read_release_packet()
    harness = discover_harness(release)
    harness_run = run_harness(harness.get("selected"))
    # Re-read active files after harness; active line refs can change, so preserve fresh refs.
    tables_after, figures_after, other_after = active_artifacts()
    tracked_after = tables_after + figures_after
    after = snapshot(tracked_after)
    comp = compare(before, after)

    blockers: List[Dict[str, Any]] = []
    if harness_run.get("status") != "completed":
        blockers.append({
            "severity": "warning" if harness_run.get("status") in {"no_explicit_harness_found", "not_run_conservative_allowlist"} else "error",
            "message": f"Regeneration harness did not complete: {harness_run.get('status')}. {harness_run.get('blocker','')}",
        })
    for rec in tracked_after:
        if not rec.get("exists"):
            blockers.append({"severity": "error", "message": "Active artifact referenced by paper/main.tex is missing", "path": rec.get("path"), "line": rec.get("active_main_tex_line")})
    non_regen = []
    if harness_run.get("status") != "completed":
        for rec in tracked_after:
            non_regen.append({"path": rec.get("path"), "reason": "No completed regeneration command, so current file checksum is audit evidence but not independent regeneration evidence."})

    validation_results = {
        "active_table_count": len(tables_after),
        "active_figure_count": len(figures_after),
        "active_other_input_count": len(other_after),
        "all_active_artifacts_exist": all(r.get("exists") for r in tracked_after),
        "harness_completed": harness_run.get("status") == "completed",
        "all_checksums_match_after_harness": all(r.get("checksum_match") for r in comp),
        "all_tex_table_numeric_tokens_match_after_harness": all(r.get("tex_numeric_tokens_match", True) for r in comp),
        "comparison_row_count": len(comp),
    }

    release_evidence_index = {
        "release_packet": {k: v for k, v in release.items() if k != "data"},
        "active_table_line_evidence": [{"path": r.get("path"), "line": r.get("active_main_tex_line"), "statement": r.get("main_tex_statement"), "sha256": r.get("sha256")} for r in tables_after],
        "active_figure_line_evidence": [{"path": r.get("path"), "line": r.get("active_main_tex_line"), "statement": r.get("main_tex_statement"), "sha256": r.get("sha256")} for r in figures_after],
        "audit_outputs": [{"path": rel(OUT_JSON)}, {"path": rel(OUT_MD)}],
    }

    audit: Dict[str, Any] = {
        "generated_utc": now_iso(),
        "project_root": str(ROOT),
        "environment": {"python": sys.version, "platform": platform.platform(), "cwd": str(ROOT)},
        "command": f"{Path(sys.executable).name} scripts/active_artifact_regeneration_audit.py",
        "validation_results": validation_results,
        "release_evidence_index": release_evidence_index,
        "pdf_evidence": build_pdf_evidence(),
        "citations": sources_and_citations(release),
        "sources": sources_and_citations(release),
        "status_evidence": {"blockers": blockers, "non_regenerable_artifacts": non_regen, "harness_discovery": harness, "harness_run_status": harness_run.get("status")},
        "active_tables_after": tables_after,
        "active_figures_after": figures_after,
        "active_other_inputs_after": other_after,
        "harness_discovery": harness,
        "harness_run": harness_run,
        "comparison": comp,
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    OUT_MD.write_text(write_markdown(audit), encoding="utf-8")
    print(json.dumps({
        "validation_results": validation_results,
        "audit_json": rel(OUT_JSON),
        "audit_markdown": rel(OUT_MD),
        "harness_status": harness_run.get("status"),
        "blocker_count": len(blockers),
    }, indent=2))
    return 0 if validation_results["all_active_artifacts_exist"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
