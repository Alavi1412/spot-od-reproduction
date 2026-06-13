"""Validate the matplotlib dependency repair for active manuscript figures.

Runs the same project-root Python invocation style used for
scripts/regenerate_active_manuscript.py, records command return codes and bounded
output, and fails if any required validation command fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


COMMANDS = [
    {
        "name": "matplotlib_import_smoke",
        "argv": [
            sys.executable,
            "-c",
            "import sys, matplotlib; print('python_executable=' + sys.executable); print('matplotlib_version=' + matplotlib.__version__); print('backend=' + matplotlib.get_backend())",
        ],
        "required": True,
    },
    {
        "name": "matplotlib_render_smoke",
        "argv": [sys.executable, "scripts/smoke_matplotlib_environment.py"],
        "required": True,
    },
    {
        "name": "active_manuscript_regeneration",
        "argv": [sys.executable, "scripts/regenerate_active_manuscript.py"],
        "required": True,
    },
    {
        "name": "pytest_suite",
        "argv": [sys.executable, "-m", "pytest", "-q"],
        "required": True,
    },
]


def trim(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<trimmed>...\n" + text[-limit:]


def main() -> int:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "dependency_repair_validation.json"

    results = []
    overall_ok = True
    for spec in COMMANDS:
        start = time.time()
        proc = subprocess.run(
            spec["argv"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        elapsed = round(time.time() - start, 3)
        ok = proc.returncode == 0
        if spec.get("required", True) and not ok:
            overall_ok = False
        results.append(
            {
                "name": spec["name"],
                "command": " ".join(spec["argv"]),
                "returncode": proc.returncode,
                "ok": ok,
                "required": spec.get("required", True),
                "elapsed_seconds": elapsed,
                "stdout": trim(proc.stdout),
                "stderr": trim(proc.stderr),
            }
        )

    status = {
        "status": "passed" if overall_ok else "failed",
        "python_executable": sys.executable,
        "commands": results,
        "verified_dependency_repair_files": [
            "pyproject.toml",
            "requirements.txt",
            "scripts/smoke_matplotlib_environment.py",
            "scripts/validate_dependency_repair.py",
        ],
        "artifacts_checked": {
            "matplotlib_smoke_json": Path("results/matplotlib_smoke.json").exists(),
            "matplotlib_smoke_png": Path("results/matplotlib_smoke.png").exists(),
            "release_packet_json": Path("results/release_packet.json").exists(),
        },
    }
    out_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
