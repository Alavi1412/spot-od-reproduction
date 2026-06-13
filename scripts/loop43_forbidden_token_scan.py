"""Extended forbidden-token scan of the rendered manuscript PDF text.

This scan is additive over the earlier scan and adds the further forbidden
tokens that the journal-facing manuscript must not surface: internal-process
vocabulary, implementation logistics, and version-control / dependency
artefacts that have no place in the paper text. The check operates on
``paper/main_text.txt`` (extracted via pdftotext from the compiled PDF).
"""
from __future__ import annotations

from pathlib import Path
import re
import sys


P = Path("paper/main_text.txt")
if not P.exists():
    raise SystemExit("paper/main_text.txt not found; regenerate via pdftotext.")

txt = P.read_text(encoding="utf-8").lower()

# Literal forbidden tokens (case-insensitive substring match).
LITERAL_FORBIDDEN = [
    "lockstep",
    "verifier",
    "release packet",
    "evidence-locked",
    "scope gap",
    "checkpoint",
    "gpu",
    "cuda",
    "claude",
    "codex",
    "repository",
    "codebase",
    ".py",
    ".json",
    ".csv",
]
# Regex forbidden patterns.
REGEX_FORBIDDEN = [
    r"loop\s?4[0-9]",
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
    print("FORBIDDEN HITS:")
    for tok, ctx in hits[:50]:
        print(f"  {tok!r}: ...{ctx}...")
    if len(hits) > 50:
        print(f"  ... ({len(hits) - 50} more)")
    sys.exit(1)

print("No forbidden tokens detected in paper/main_text.txt.")
sys.exit(0)
