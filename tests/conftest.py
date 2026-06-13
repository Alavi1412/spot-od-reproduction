"""Test import bootstrap for repository-local modules.

The project uses a src-layout package plus repository-level reproducibility
scripts.  Tests add both roots explicitly so import behavior is stable without
requiring an editable install.  This does not skip or mock optional/heavy
scientific dependencies; missing pandas/torch/sgp4 imports still fail where the
covered code genuinely needs them.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
