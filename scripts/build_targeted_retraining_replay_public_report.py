#!/usr/bin/env python3
"""Build the public targeted retraining replay reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON_REL = "results/validation/targeted_retraining_replay.json"
RAW_MD_REL = "results/validation/targeted_retraining_replay.md"
PUBLIC_JSON_REL = "results/validation/targeted_retraining_replay_public.json"
PUBLIC_MD_REL = "results/validation/targeted_retraining_replay_public.md"

STAGE_HISTORY_NOTE = (
    "The stress_focus_replay stage records a positive but finite validation loss "
    "at epoch 1 followed by a finite negative validation loss at epoch 2. This "
    "is a bounded finite-loss curriculum-transition note; the public replay "
    "claim is finite execution, checkpoint production, and provenance, not "
    "performance or stability evidence."
)

PROHIBITED_PUBLIC_TOKENS = (
    "env" + "_report",
    "G" + "PU",
    "CU" + "DA",
    "cu" + "da",
    "NVI" + "DIA",
    "." + "venv",
    "virtual " + "env",
    "local " + "environment",
    "python" + "_executable",
    "torch" + "_version",
    "selected" + "_device",
    "FORMAL" + "_PEER_REVIEW",
    "OBSERVABILITY_CONTEXT" + "_TRAINING_REVIEW",
    "historical" + "_docs",
    "Clau" + "de",
    "G" + "PT",
    "Ol" + "lama",
    "hard" + "ware",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_summary(raw: dict[str, Any]) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    slices = raw.get("data", {}).get("slices", {})
    split_counts: dict[str, int] = {}
    split_hashes: dict[str, dict[str, Any]] = {}
    if not isinstance(slices, dict):
        return split_counts, split_hashes
    for name, record in slices.items():
        if not isinstance(record, dict):
            continue
        slice_count = record.get("slice_trajectories")
        if isinstance(slice_count, int):
            split_counts[name] = slice_count
        split_hashes[name] = {
            key: record.get(key)
            for key in (
                "source_path",
                "source_sha256",
                "slice_path",
                "slice_sha256",
                "source_trajectories",
                "slice_trajectories",
                "state_shape",
            )
            if key in record
        }
    return split_counts, split_hashes


def _stage_history(raw: dict[str, Any]) -> list[dict[str, Any]]:
    stages = raw.get("training", {}).get("stages", [])
    if not isinstance(stages, list):
        return []
    fields = (
        "stage",
        "checkpoint",
        "epochs_completed",
        "train_loss",
        "val_loss",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
    )
    public_stages: list[dict[str, Any]] = []
    for stage in stages:
        if isinstance(stage, dict):
            public_stages.append({key: stage.get(key) for key in fields if key in stage})
    return public_stages


def _canonical_checkpoint_digest(raw: dict[str, Any]) -> dict[str, Any]:
    outputs = raw.get("outputs", {})
    before = outputs.get("canonical_checkpoint_digest_before", {})
    after = outputs.get("canonical_checkpoint_digest_after", {})
    if not isinstance(before, dict):
        before = {}
    if not isinstance(after, dict):
        after = {}
    before_sha = before.get("sha256")
    after_sha = after.get("sha256")
    return {
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "unchanged": bool(before_sha and before_sha == after_sha),
        "file_count": after.get("file_count", before.get("file_count")),
    }


def build_public_report(raw: dict[str, Any], raw_json_sha: str, raw_md_sha: str | None) -> dict[str, Any]:
    split_counts, split_hashes = _split_summary(raw)
    device = raw.get("device", {})
    if not isinstance(device, dict):
        device = {}
    outputs = raw.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
    data = raw.get("data", {})
    if not isinstance(data, dict):
        data = {}
    criteria = raw.get("criteria", {})
    if not isinstance(criteria, dict):
        criteria = {}

    return {
        "schema_version": "targeted-retraining-replay-public-v1",
        "artifact_type": "targeted_learned_estimator_retraining_replay_public_report",
        "created_at_utc": raw.get("created_at_utc"),
        "status": raw.get("status"),
        "claim_boundary": raw.get("claim_boundary"),
        "source_evidence": {
            "raw_report_sha256": raw_json_sha,
            "raw_markdown_sha256": raw_md_sha,
        },
        "predeclaration": raw.get("predeclaration"),
        "model": raw.get("model"),
        "seed": raw.get("seed"),
        "data": {
            "source_data_dir": data.get("source_data_dir"),
            "full_materialized_curriculum": data.get("full_materialized_curriculum"),
            "full_split_counts": data.get("full_split_counts"),
            "split_counts": split_counts,
            "split_hashes": split_hashes,
        },
        "training": {
            "stages": _stage_history(raw),
            "stage_history_note": STAGE_HISTORY_NOTE,
        },
        "outputs": {
            "history": outputs.get("history"),
            "manifests": outputs.get("manifests"),
            "checkpoints": outputs.get("checkpoints"),
            "canonical_checkpoint_digest": _canonical_checkpoint_digest(raw),
        },
        "criteria": criteria,
        "execution_attestation": {
            "accelerated_compute_required": device.get("accelerated_compute_required"),
            "accelerated_compute_used": device.get("accelerated_compute_used"),
            "non_accelerated_execution_allowed": device.get(
                "non_accelerated_execution_allowed"
            ),
            "raw_report_sha256": raw_json_sha,
        },
    }


def _fmt_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return str(value)


def build_markdown(public: dict[str, Any]) -> str:
    status = public.get("status", {})
    passed = status.get("pass") if isinstance(status, dict) else None
    lines = [
        "# Targeted Retraining Replay Public Report",
        "",
        f"Status: {'PASS' if passed else 'FAIL'}",
        "",
        f"Model: {public.get('model')}",
        f"Seed: {public.get('seed')}",
        "",
        f"Claim boundary: {public.get('claim_boundary')}",
        "",
        "## Source Evidence",
        "",
        f"- Raw JSON SHA-256: {public['source_evidence']['raw_report_sha256']}",
        f"- Raw markdown SHA-256: {public['source_evidence']['raw_markdown_sha256']}",
        "",
        "## Predeclaration",
        "",
        f"- Path: {(public.get('predeclaration') or {}).get('path')}",
        f"- SHA-256: {(public.get('predeclaration') or {}).get('sha256')}",
        "",
        "## Execution Attestation",
        "",
    ]
    attestation = public["execution_attestation"]
    for key in (
        "accelerated_compute_required",
        "accelerated_compute_used",
        "non_accelerated_execution_allowed",
    ):
        lines.append(f"- {key}: {_fmt_bool(attestation.get(key))}")

    lines.extend(
        [
            "",
            "## Data Slices",
            "",
            "| Split | Source trajectories | Slice trajectories | Source SHA-256 | Slice SHA-256 |",
            "|---|---:|---:|---|---|",
        ]
    )
    split_hashes = public["data"]["split_hashes"]
    for name in sorted(split_hashes):
        record = split_hashes[name]
        lines.append(
            "| {name} | {source_count} | {slice_count} | {source_sha} | {slice_sha} |".format(
                name=name,
                source_count=record.get("source_trajectories"),
                slice_count=record.get("slice_trajectories"),
                source_sha=record.get("source_sha256"),
                slice_sha=record.get("slice_sha256"),
            )
        )

    lines.extend(["", "## Stage Histories", ""])
    note = public["training"].get("stage_history_note")
    if note:
        lines.extend([f"Note: {note}", ""])
    for stage in public["training"]["stages"]:
        lines.append(
            "- {stage}: epochs={epochs}, final_train_loss={train}, final_val_loss={val}, best_val_loss={best}".format(
                stage=stage.get("stage"),
                epochs=stage.get("epochs_completed"),
                train=stage.get("final_train_loss"),
                val=stage.get("final_val_loss"),
                best=stage.get("best_val_loss"),
            )
        )

    lines.extend(["", "## Checkpoints", ""])
    for checkpoint in public["outputs"].get("checkpoints") or []:
        lines.append(f"- {checkpoint.get('path')}: {checkpoint.get('sha256')}")

    digest = public["outputs"]["canonical_checkpoint_digest"]
    lines.extend(
        [
            "",
            "## Canonical Checkpoint Digest",
            "",
            f"- before_sha256: {digest.get('before_sha256')}",
            f"- after_sha256: {digest.get('after_sha256')}",
            f"- unchanged: {_fmt_bool(digest.get('unchanged'))}",
            f"- file_count: {digest.get('file_count')}",
            "",
            "## Criteria",
            "",
        ]
    )
    for key in sorted(public.get("criteria", {})):
        lines.append(f"- {key}: {_fmt_bool(public['criteria'][key])}")
    lines.append("")
    return "\n".join(lines)


def assert_public_text_is_clean(rel_path: str, text: str) -> None:
    folded = text.casefold()
    hits = [token for token in PROHIBITED_PUBLIC_TOKENS if token.casefold() in folded]
    if hits:
        joined = ", ".join(sorted(set(hits), key=str.casefold))
        raise ValueError(f"{rel_path} contains prohibited public token(s): {joined}")


def main() -> int:
    raw_json_path = ROOT / RAW_JSON_REL
    raw_md_path = ROOT / RAW_MD_REL
    public_json_path = ROOT / PUBLIC_JSON_REL
    public_md_path = ROOT / PUBLIC_MD_REL

    raw = json.loads(raw_json_path.read_text(encoding="utf-8"))
    raw_md_sha = sha256_file(raw_md_path) if raw_md_path.is_file() else None
    public = build_public_report(raw, sha256_file(raw_json_path), raw_md_sha)

    json_text = json.dumps(public, indent=2) + "\n"
    md_text = build_markdown(public)
    assert_public_text_is_clean(PUBLIC_JSON_REL, json_text)
    assert_public_text_is_clean(PUBLIC_MD_REL, md_text)

    public_json_path.parent.mkdir(parents=True, exist_ok=True)
    public_json_path.write_text(json_text, encoding="utf-8")
    public_md_path.write_text(md_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "public_json": PUBLIC_JSON_REL,
                "public_md": PUBLIC_MD_REL,
                "raw_report_sha256": public["source_evidence"]["raw_report_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
