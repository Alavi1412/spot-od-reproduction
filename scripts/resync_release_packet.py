#!/usr/bin/env python3
"""Resync results/release_packet.json + paper/RELEASE_PACKET.md only.

This regenerates the release packet's per-line evidence of ``\\input{tables/...}``
and ``\\includegraphics`` positions in paper/main.tex after manuscript edits,
WITHOUT regenerating the auto-built result tables.  Skipping table regeneration
avoids races with any long-running seed sweep that writes into results/ while
this patch is being prepared; the canonical table content is unchanged here.

It reuses ``build_release_packet`` from scripts/build_paper_assets.py so the
output is byte-identical to what the full asset build would have produced for
the release-packet portion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_assets import build_release_packet, load_json  # noqa: E402
from gnn_state_estimation.utils.io import load_yaml  # noqa: E402


def main() -> int:
    cfg = load_yaml(str(ROOT / "configs" / "experiment.yaml"))
    metrics = load_json(ROOT / cfg["output"]["metrics_path"])
    main_tex_path = ROOT / "paper" / "main.tex"
    release_packet, release_md = build_release_packet(cfg, metrics, main_tex_path)
    (ROOT / "paper" / "RELEASE_PACKET.md").write_text(release_md, encoding="utf-8")
    (ROOT / "results" / "release_packet.json").write_text(
        json.dumps(release_packet, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "resynced": [
                    "results\\release_packet.json",
                    "paper\\RELEASE_PACKET.md",
                ],
                "paper_title": release_packet.get("paper_title"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
