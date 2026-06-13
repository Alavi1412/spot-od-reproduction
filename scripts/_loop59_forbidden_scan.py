"""Forbidden-token scan for loop59 rendered PDFs (main + supplement)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = [
    "checkpoint",
    "virtual env",
    "GPU",
    "cuda",
    "D:\\",
    "C:\\",
    ".venv",
    "python scripts",
    "python script",
    "script/scripts",
    ".py",
    ".json",
    ".csv",
    ".tex",
    "Claude",
    "Codex",
    "repository",
    "command line",
    "codebase",
    "local path",
    "source code",
    "runnable code",
    "revision loop",
    "loop59",
    "loop58",
    "loop57",
]


def scan(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    hits: list[tuple[str, str]] = []
    for token in FORBIDDEN:
        for m in re.finditer(re.escape(token), text, flags=re.IGNORECASE):
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            ctx = text[start:end].replace("\n", " ").strip()
            hits.append((token, ctx))
    return hits


def main() -> int:
    rc = 0
    for f in ("paper/main_text.txt", "paper/supplement_text.txt"):
        p = ROOT / f
        if not p.exists():
            print(f"{f}: MISSING")
            rc = 1
            continue
        hits = scan(p)
        if hits:
            print(f"{f}: {len(hits)} forbidden hits")
            for token, ctx in hits[:50]:
                print(f"  [{token}] ...{ctx}...")
            rc = 1
        else:
            print(f"{f}: CLEAN")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
