#!/usr/bin/env python
"""Run a process-noise sensitivity sweep with retraining and reevaluation."""

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


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def format_noise_tag(value: float) -> str:
    token = f"{value:.6g}"
    token = token.replace("-", "m").replace(".", "p")
    return token


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--process-noise-levels", type=str, default="0.0,0.2,0.4")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vary-seed-by-level", action="store_true")
    p.add_argument("--output-dir", type=str, default="results/process_noise_sweep")
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--skip-existing", action="store_true")
    return p


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_variant_config(
    base_cfg: dict,
    run_dir: Path,
    process_noise_std: float,
    epochs: int,
    seed: int,
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = int(seed)
    cfg["simulation"]["dynamics"]["process_noise_std"] = float(process_noise_std)
    cfg["training"]["num_epochs"] = int(epochs)

    cfg["data"]["output_dir"] = str(run_dir / "data")
    cfg["output"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    cfg["output"]["metrics_path"] = str(run_dir / "metrics_summary.json")
    cfg["output"]["per_step_path"] = str(run_dir / "per_step_errors.csv")
    cfg["output"]["figure_dir"] = str(run_dir / "figures")
    return cfg


def extract_summary_row(process_noise_std: float, metrics_path: Path) -> dict[str, float]:
    metrics = load_yaml(metrics_path) if metrics_path.suffix in {".yml", ".yaml"} else None
    if metrics is None:
        import json

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

    test = metrics["test"]
    stress = metrics["stress_test"]
    return {
        "process_noise_std": float(process_noise_std),
        "test_hybrid_pos_rmse_m": float(test["HybridGNN"]["pos_rmse_m"]),
        "test_ukf_pos_rmse_m": float(test["UKF"]["pos_rmse_m"]),
        "test_aukf_pos_rmse_m": float(test["AUKF"]["pos_rmse_m"]),
        "test_hybrid_vs_ukf_percent": float(test["HybridGNN"]["improvement_vs_ukf_pos_rmse_percent"]),
        "test_hybrid_vs_best_classical_percent": float(
            test["HybridGNN"]["improvement_vs_best_classical_pos_rmse_percent"]
        ),
        "stress_hybrid_pos_rmse_m": float(stress["HybridGNN"]["pos_rmse_m"]),
        "stress_ukf_pos_rmse_m": float(stress["UKF"]["pos_rmse_m"]),
        "stress_aukf_pos_rmse_m": float(stress["AUKF"]["pos_rmse_m"]),
        "stress_hybrid_vs_ukf_percent": float(stress["HybridGNN"]["improvement_vs_ukf_pos_rmse_percent"]),
        "stress_hybrid_vs_best_classical_percent": float(
            stress["HybridGNN"]["improvement_vs_best_classical_pos_rmse_percent"]
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    levels = parse_float_list(args.process_noise_levels)
    base_cfg = load_yaml(args.config)
    py = args.python

    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float]] = []
    for idx, level in enumerate(levels):
        tag = format_noise_tag(level)
        run_dir = root_out / f"pn_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        variant_seed = int(args.seed) + idx if args.vary_seed_by_level else int(args.seed)
        cfg = build_variant_config(
            base_cfg=base_cfg,
            run_dir=run_dir,
            process_noise_std=level,
            epochs=args.epochs,
            seed=variant_seed,
        )
        cfg_path = run_dir / "config.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        metrics_path = Path(cfg["output"]["metrics_path"])
        if args.skip_existing and metrics_path.exists():
            print(f"Skipping existing level {level}: {metrics_path}")
        else:
            run([py, "scripts/generate_dataset.py", "--config", str(cfg_path)], env=env)
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

        rows.append(extract_summary_row(process_noise_std=level, metrics_path=metrics_path))

    df = pd.DataFrame(rows).sort_values("process_noise_std")
    summary_csv = root_out / "process_noise_summary.csv"
    df.to_csv(summary_csv, index=False)
    dump_json({"levels": rows}, root_out / "process_noise_summary.json")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
