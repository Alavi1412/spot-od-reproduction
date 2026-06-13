#!/usr/bin/env python3
"""Compatibility wrapper for release-packet synchronization verification.

The canonical validator lives in docs/verify_release_packet_sync.py. Some
manager review prompts refer to scripts/verify_release_packet_sync.py, so this
wrapper keeps that verification command executable without duplicating logic.
"""
from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "docs" / "verify_release_packet_sync.py"), run_name="__main__")
