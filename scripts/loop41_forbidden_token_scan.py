"""Loop 41 forbidden-token scan of the rendered manuscript PDF text."""
from pathlib import Path
import sys

p = Path("paper/main_text.txt")
if not p.exists():
    raise SystemExit("paper/main_text.txt not found; regenerate via pdftotext.")

txt = p.read_text(encoding="utf-8").lower()
forbidden = [
    "gpu", "cuda", "claude", "codex",
    "venv", "virtualenvs",
    "python3", "pip install", "pytest",
    "scripts/", "src/", "results/",
    ".py", ".json", ".csv", ".npz",
    "released compact rgr-gf model", "released checkpoint", "release-only",
    "github.com", "zenodo", "figshare", "doi.org", "dx.doi", "osf.io",
    "finals2000a.all",
    "d:\\", "c:\\",
]
hits = []
for tok in forbidden:
    if tok in txt:
        idx = txt.find(tok)
        ctx = txt[max(0, idx - 60):idx + len(tok) + 60]
        hits.append((tok, ctx))

if hits:
    print("FORBIDDEN HITS:")
    for tok, ctx in hits:
        print(f"  {tok!r}: ...{ctx}...")
    sys.exit(1)
print("No forbidden tokens detected in paper/main_text.txt.")
sys.exit(0)
