"""Validate the matplotlib dependency fix used by active manuscript generation.

The validator runs commands with the same Python executable used to launch this
script and writes machine-readable evidence for the release manager.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "dependency_fix_validation.json"
COMMANDS = [
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    [sys.executable, "scripts/check_matplotlib_environment.py"],
    [sys.executable, "scripts/regenerate_active_manuscript.py"],
]


def run_command(command: list[str]) -> dict[str, object]:
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    output = proc.stdout or ""
    return {
        "command": " ".join(command),
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.time() - start, 2),
        "output_tail": output[-8000:],
    }


def main() -> int:
    results = []
    for command in COMMANDS:
        result = run_command(command)
        results.append(result)
        if result["returncode"] != 0:
            break
    payload = {
        "cwd": str(PROJECT_ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "mplbackend_env": os.environ.get("MPLBACKEND"),
        "results": results,
        "all_passed": all(result["returncode"] == 0 for result in results),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
