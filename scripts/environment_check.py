#!/usr/bin/env python
"""Record the active Python/CUDA environment for reproducible runs."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
from pathlib import Path

from gnn_state_estimation.utils.io import load_yaml
from gnn_state_estimation.utils.runtime import resolve_device, write_env_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    requested_device = args.device or cfg.get("device", {}).get("train", "auto")
    device = resolve_device(requested_device)
    output = Path(args.output or cfg["output"]["env_report_path"])
    report = write_env_report(output, device=device)
    print(f"Selected device: {report['selected_device']}")
    if report["gpu"] is not None:
        print(f"GPU: {report['gpu']['name']}")
        print(f"VRAM (GiB): {report['gpu']['total_memory_bytes'] / (1024**3):.2f}")
    print(f"Saved environment report to {output}")


if __name__ == "__main__":
    main()
