#!/usr/bin/env python3
"""Strict manuscript leakage scan for the current review artifacts.

The rendered manuscript text is scanned for paper-facing contamination terms.
Active TeX comments are scanned separately so invisible labels, citations, and
table filenames are not mistaken for rendered manuscript prose. Changed
validation tooling is checked only for prohibited local-model mentions because
validation/release artifacts may legitimately contain repository-relative
commands and artifact names.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation" / "leakage_scan.json"

TEXT_FILES = [
    ROOT / "results" / "validation" / "paper_main_current_leakscan.txt",
    ROOT / "results" / "validation" / "paper_supplement_current_leakscan.txt",
]
TEX_FILES = [
    ROOT / "paper" / "main.tex",
    ROOT / "paper" / "supplement.tex",
]
TOOLING_FILES = [
    ROOT / "scripts" / "verify_archive_extracted_reproduction.py",
    ROOT / "scripts" / "build_supplementary_manifest.py",
    ROOT / "tests" / "test_archive_extracted_reproduction.py",
]

TEXT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "local_path": [
        re.compile(r"(?:[A-Za-z]:\\|\\\\[^\s]+)"),
        re.compile(r"\b(?:venv|virtualenv|\.venv)\b", re.IGNORECASE),
    ],
    "hardware": [re.compile(r"\b(?:gpu|cuda)\b", re.IGNORECASE)],
    "ai_tool": [
        re.compile(
            r"\b(?:model-provider|local-model-provider)\b",
            re.IGNORECASE,
        )
    ],
    "review_process": [
        re.compile(r"\brevision loop\b", re.IGNORECASE),
        re.compile(r"(?<!open-)\bloop[- ]?\d+\b", re.IGNORECASE),
    ],
    "workspace_or_code": [
        re.compile(r"\bGNN State Estimation\b", re.IGNORECASE),
        re.compile(r"\b(?:scripts|src|results)[\\/]"),
        re.compile(r"\b(?:\.py|\.json|\.csv|\.npz)\b", re.IGNORECASE),
    ],
}

COMMENT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    family: patterns
    for family, patterns in TEXT_PATTERNS.items()
    if family != "workspace_or_code"
}
COMMENT_PATTERNS["workspace_or_code"] = [
    re.compile(r"\bGNN State Estimation\b", re.IGNORECASE)
]

LOCAL_MODEL_RE = re.compile(r"\b(?:local model|external local model)\b", re.IGNORECASE)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def hit_context(text: str, start: int, end: int) -> str:
    lo = max(0, start - 80)
    hi = min(len(text), end + 80)
    return text[lo:hi].replace("\n", " ").strip()


def scan_text(text: str, patterns: dict[str, list[re.Pattern[str]]]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for family, regexes in patterns.items():
        for regex in regexes:
            for match in regex.finditer(text):
                hits.append(
                    {
                        "family": family,
                        "pattern": regex.pattern,
                        "match": match.group(0),
                        "context": hit_context(text, match.start(), match.end()),
                    }
                )
    return hits


def tex_comments(path: Path) -> str:
    comments: list[str] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        escaped = False
        for idx, char in enumerate(line):
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == "%" and not escaped:
                comments.append(f"{lineno}: {line[idx + 1:]}")
                break
            escaped = False
    return "\n".join(comments)


def main() -> int:
    report: dict[str, object] = {
        "status": "pass",
        "scope": {
            "rendered_text": [rel(p) for p in TEXT_FILES],
            "active_tex_comments": [rel(p) for p in TEX_FILES],
            "tooling_files_checked_for_local_model_mentions": [rel(p) for p in TOOLING_FILES],
            "notes": [
                "Rendered manuscript text is scanned for local paths, implementation artifacts, AI/tool names, and revision-process wording.",
                "Active TeX comments are scanned separately from visible manuscript text.",
                "Changed validation tooling is scanned for prohibited local-model mentions only.",
            ],
        },
        "rendered_text_hits": {},
        "active_tex_comment_hits": {},
        "tooling_local_model_hits": {},
    }

    for path in TEXT_FILES:
        hits = scan_text(
            path.read_text(encoding="utf-8", errors="replace"),
            TEXT_PATTERNS,
        )
        report["rendered_text_hits"][rel(path)] = hits  # type: ignore[index]
        if hits:
            report["status"] = "fail"

    for path in TEX_FILES:
        hits = scan_text(tex_comments(path), COMMENT_PATTERNS)
        report["active_tex_comment_hits"][rel(path)] = hits  # type: ignore[index]
        if hits:
            report["status"] = "fail"

    for path in TOOLING_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [
            {
                "match": match.group(0),
                "context": hit_context(text, match.start(), match.end()),
            }
            for match in LOCAL_MODEL_RE.finditer(text)
        ]
        report["tooling_local_model_hits"][rel(path)] = hits  # type: ignore[index]
        if hits:
            report["status"] = "fail"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{str(report['status']).upper()} {rel(OUT)}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
