"""Smoke test for the active manuscript figure-generation environment.

Run from the project root with the same Python executable used for
scripts/regenerate_active_manuscript.py. It verifies that matplotlib imports,
uses a headless-safe backend, and can create/close a figure.
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt


def main() -> None:
    fig, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [0, 1])
    ax.set_title("matplotlib smoke")
    fig.canvas.draw()
    plt.close(fig)
    print(
        f"matplotlib import ok: version={matplotlib.__version__}, "
        f"backend={matplotlib.get_backend()}"
    )


if __name__ == "__main__":
    main()
