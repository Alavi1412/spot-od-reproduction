"""Smoke-test the plotting environment used by manuscript regeneration.

This script intentionally uses the default project Python interpreter, matching the
interpreter used to run scripts/regenerate_active_manuscript.py in validation.
It verifies that matplotlib imports, selects a non-interactive backend, and can
write a small PNG artifact under results/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    figure_path = results_dir / "matplotlib_smoke.png"
    payload_path = results_dir / "matplotlib_smoke.json"

    fig, ax = plt.subplots(figsize=(3, 2), dpi=120)
    ax.plot([0, 1, 2], [0, 1, 0], marker="o")
    ax.set_title("matplotlib smoke")
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    payload = {
        "status": "passed",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "matplotlib_version": matplotlib.__version__,
        "matplotlib_backend": matplotlib.get_backend(),
        "figure_path": str(figure_path),
        "figure_exists": figure_path.exists(),
        "figure_size_bytes": figure_path.stat().st_size if figure_path.exists() else 0,
        "same_invocation_pattern_as_regenerator": "python scripts/<script>.py from project root",
        "regenerator_script": "scripts/regenerate_active_manuscript.py",
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
