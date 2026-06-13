#!/usr/bin/env python
"""Run complete experiment pipeline end-to-end."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    default_python = str(Path(".venv/Scripts/python.exe")) if Path(".venv/Scripts/python.exe").exists() else sys.executable
    p.add_argument("--python", type=str, default=default_python)
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--extended", action="store_true")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--classical-tune", action="store_true")
    p.add_argument("--seed-list", type=str, default="7,13,23,37,42")
    p.add_argument("--seed-suite-model", type=str, default="InnovationHybridGNN")
    p.add_argument("--robustness-range-scales", type=str, default="0.7,1.0,1.4,1.8")
    p.add_argument("--robustness-outlier-probs", type=str, default="0.00,0.01,0.03,0.05")
    p.add_argument("--station-outage-topk", type=int, default=4)
    p.add_argument("--station-outage-max-trajectories", type=int, default=24)
    p.add_argument("--compile-paper", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    py = args.python
    cfg = args.config
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src_path

    device_args = ["--device", args.device] if args.device else []
    run([py, "scripts/environment_check.py", "--config", cfg, *device_args], env=env)
    run([py, "scripts/generate_dataset.py", "--config", cfg], env=env)
    if args.classical_tune:
        run([py, "scripts/tune_classical_baselines.py", "--config", cfg, *device_args], env=env)
    if args.tune:
        run([py, "scripts/tune_hybrid.py", "--config", cfg, *device_args], env=env)
    run([py, "scripts/train_models.py", "--config", cfg, *device_args], env=env)
    run([py, "scripts/evaluate_models.py", "--config", cfg, *device_args], env=env)
    if args.extended:
        run(
            [
                py,
                "scripts/run_seed_sweep.py",
                "--config",
                cfg,
                "--seeds",
                args.seed_list,
            ],
            env=env,
        )
        run(
            [
                py,
                "scripts/run_benchmark_seed_sweep.py",
                "--config",
                cfg,
                "--model",
                args.seed_suite_model,
                "--seeds",
                args.seed_list,
                *device_args,
            ],
            env=env,
        )
        run([py, "scripts/run_ablation_study.py", "--config", cfg], env=env)
        run(
            [
                py,
                "scripts/run_robustness_sweep.py",
                "--config",
                cfg,
                "--range-scales",
                args.robustness_range_scales,
                "--outlier-probs",
                args.robustness_outlier_probs,
            ],
            env=env,
        )
        run(
            [
                py,
                "scripts/run_station_outage_sweep.py",
                "--config",
                cfg,
                "--scenarios",
                "test,stress_test",
                "--topk-single-outages",
                str(args.station_outage_topk),
                "--max-trajectories",
                str(args.station_outage_max_trajectories),
            ],
            env=env,
        )
    run([py, "scripts/build_paper_assets.py", "--config", cfg], env=env)
    if args.compile_paper:
        run([py, "scripts/compile_paper.py"], env=env)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
