#!/usr/bin/env python3
"""Build task result payload from active regeneration audit."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path.cwd()
AUDIT = ROOT / "results" / "active_regeneration_audit.json"
OUT = ROOT / "results" / "active_regeneration_task_result.json"
MD = ROOT / "results" / "active_regeneration_audit.md"

if not AUDIT.exists():
    payload = {
        "status": "BLOCKED",
        "summary": "Active regeneration audit JSON was not produced.",
        "result_payload": {
            "validation_results": {"status": "blocked", "missing": str(AUDIT)},
            "release_evidence_index": {"status": "blocked", "missing": str(AUDIT)},
            "pdf_evidence": {"status": "blocked", "reason": "audit missing"},
            "citations": [],
            "sources": [],
            "status_evidence": {"blockers": [{"severity": "error", "message": "results/active_regeneration_audit.json missing after attempted audit run"}]},
        },
    }
else:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    validation = audit.get("validation_results", {})
    blockers = audit.get("status_evidence", {}).get("blockers", [])
    status = "completed_with_blockers" if blockers else "completed"
    if validation.get("harness_completed") is False:
        status = "completed_audit_harness_not_completed"
    if validation.get("all_active_artifacts_exist") is False:
        status = "blocked_missing_active_artifact"
    payload = {
        "status": status,
        "summary": "Active manuscript table/figure regeneration audit produced machine-readable and reviewer-readable artifacts.",
        "artifacts": {
            "machine_readable_audit": "results/active_regeneration_audit.json",
            "reviewer_readable_audit": "results/active_regeneration_audit.md",
            "task_result_payload": "results/active_regeneration_task_result.json",
        },
        "result_payload": {
            "validation_results": validation,
            "release_evidence_index": audit.get("release_evidence_index", {}),
            "pdf_evidence": audit.get("pdf_evidence", {}),
            "citations": audit.get("citations", {}),
            "sources": audit.get("sources", {}),
            "status_evidence": audit.get("status_evidence", {}),
            "audit_outputs": {
                "json": "results/active_regeneration_audit.json",
                "markdown": "results/active_regeneration_audit.md",
            },
            "notion_export": {
                "status": "not_published",
                "reason": "Project-specific Notion destination/tool response was not available in this run; local reviewer-readable Markdown artifact is provided for demo/export.",
            },
            "open_questions": [
                "If no explicit local regeneration harness was discovered, which script or command should be registered as ACTIVE_REGEN_COMMAND for independent regeneration?",
                "Should PDF rebuilding be part of this audit command, or remain a separate manuscript-build validation step?",
                "Which dataset provenance fields in release_packet.json should be promoted into the manuscript/supplement evidence index?",
            ],
        },
    }
OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"wrote": "results/active_regeneration_task_result.json", "status": payload["status"]}, indent=2))
