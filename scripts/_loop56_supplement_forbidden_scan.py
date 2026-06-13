"""Inline forbidden-token scan over paper/supplement_text.txt.

Mirrors the combined token set used by the main-text scans plus a regex
covering recent revision-loop tags. Internal-process / implementation-
logistics tokens must not surface in the journal-facing supplement.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys


P = Path("paper/supplement_text.txt")
if not P.exists():
    raise SystemExit("paper/supplement_text.txt not found; regenerate via pdftotext.")
txt = P.read_text(encoding="utf-8").lower()

LITERAL_FORBIDDEN = [
    "gpu", "cuda", "claude", "codex",
    "venv", "virtualenvs",
    "python3", "pip install", "pytest",
    "scripts/", "src/", "results/",
    ".py", ".json", ".csv", ".npz",
    "released compact rgr-gf model", "released checkpoint", "release-only",
    "github.com", "zenodo", "figshare", "doi.org", "dx.doi", "osf.io",
    "finals2000a.all",
    "d:\\", "c:\\",
    "lockstep", "verifier", "release packet", "evidence-locked",
    "scope gap", "checkpoint", "repository", "codebase",
]
REGEX_FORBIDDEN = [
    r"loop\s?[45][0-9]",
]

hits: list[tuple[str, str]] = []
for tok in LITERAL_FORBIDDEN:
    idx = txt.find(tok)
    while idx >= 0:
        ctx = txt[max(0, idx - 60): idx + len(tok) + 60]
        hits.append((tok, ctx))
        idx = txt.find(tok, idx + len(tok))
for pattern in REGEX_FORBIDDEN:
    for m in re.finditer(pattern, txt):
        ctx = txt[max(0, m.start() - 60): m.end() + 60]
        hits.append((pattern, ctx))

if hits:
    print("FORBIDDEN HITS in paper/supplement_text.txt:")
    for tok, ctx in hits[:80]:
        print(f"  {tok!r}: ...{ctx}...")
    if len(hits) > 80:
        print(f"  ... ({len(hits) - 80} more)")
    sys.exit(1)
print("No forbidden tokens detected in paper/supplement_text.txt.")
sys.exit(0)
