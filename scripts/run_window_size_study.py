#!/usr/bin/env python
"""Run a window-size sensitivity study with retraining and reevaluation."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from gnn_state_estimation.utils.io import dump_json, load_yaml


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--window-sizes", type=str, default="8,12,16")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--vary-seed-by-window",
        action="store_true",
        help="If set, use seed+index for each window. Default keeps one fixed seed across windows.",
    )
    p.add_argument("--output-dir", type=str, default="results/window_size_study")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--skip-existing", action="store_true")
    return p


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_variant_config(base_cfg: dict, out_dir: Path, window_size: int, epochs: int, seed: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["training"]["window_size"] = int(window_size)
    cfg["training"]["num_epochs"] = int(epochs)

    cfg["output"]["checkpoint_dir"] = str(out_dir / "checkpoints")
    cfg["output"]["metrics_path"] = str(out_dir / "metrics_summary.json")
    cfg["output"]["per_step_path"] = str(out_dir / "per_step_errors.csv")
    cfg["output"]["figure_dir"] = str(out_dir / "figures")
    return cfg


def extract_summary_row(window_size: int, metrics_path: Path) -> dict[str, float | int]:
    metrics = load_yaml(metrics_path) if metrics_path.suffix in {".yml", ".yaml"} else None
    if metrics is None:
        import json

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    test = metrics["test"]
    stress = metrics["stress_test"]
    row = {
        "window_size": int(window_size),
        "test_hybrid_pos_rmse_m": float(test["HybridGNN"]["pos_rmse_m"]),
        "test_ukf_pos_rmse_m": float(test["UKF"]["pos_rmse_m"]),
        "test_aukf_pos_rmse_m": float(test["AUKF"]["pos_rmse_m"]),
        "test_hybrid_vs_ukf_percent": float(test["HybridGNN"]["improvement_vs_ukf_pos_rmse_percent"]),
        "stress_hybrid_pos_rmse_m": float(stress["HybridGNN"]["pos_rmse_m"]),
        "stress_ukf_pos_rmse_m": float(stress["UKF"]["pos_rmse_m"]),
        "stress_aukf_pos_rmse_m": float(stress["AUKF"]["pos_rmse_m"]),
        "stress_hybrid_vs_ukf_percent": float(stress["HybridGNN"]["improvement_vs_ukf_pos_rmse_percent"]),
    }
    return row


def main() -> None:
    args = build_parser().parse_args()
    windows = parse_int_list(args.window_sizes)
    base_cfg = load_yaml(args.config)
    py = args.python

    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    for idx, w in enumerate(windows):
        run_dir = root_out / f"w{w}"
        run_dir.mkdir(parents=True, exist_ok=True)
        variant_seed = int(args.seed) + idx if args.vary_seed_by_window else int(args.seed)
        cfg = build_variant_config(
            base_cfg=base_cfg,
            out_dir=run_dir,
            window_size=w,
            epochs=args.epochs,
            seed=variant_seed,
        )
        cfg_path = run_dir / "config.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        metrics_path = Path(cfg["output"]["metrics_path"])
        if args.skip_existing and metrics_path.exists():
            print(f"Skipping existing window size {w}: {metrics_path}")
        else:
            run([py, "scripts/train_models.py", "--config", str(cfg_path)], env=env)
            run(
                [
                    py,
                    "scripts/evaluate_models.py",
                    "--config",
                    str(cfg_path),
                    "--baseline-cache-dir",
                    str(run_dir / "baseline_cache"),
                    "--predictions-path",
                    str(run_dir / "predictions.npz"),
                ],
                env=env,
            )

        rows.append(extract_summary_row(window_size=w, metrics_path=metrics_path))

    df = pd.DataFrame(rows).sort_values("window_size")
    summary_csv = root_out / "window_size_summary.csv"
    df.to_csv(summary_csv, index=False)
    dump_json({"windows": rows}, root_out / "window_size_summary.json")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
