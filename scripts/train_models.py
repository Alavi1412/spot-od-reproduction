#!/usr/bin/env python
"""Train configured estimators with curriculum stages and run manifests."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import time
from dataclasses import replace
from pathlib import Path

from gnn_state_estimation.dataset import concatenate_dataset_arrays, load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.runtime import (
    build_run_manifest,
    duration_metadata,
    resolve_device,
    utc_now_iso,
    write_env_report,
)
from gnn_state_estimation.utils.seeding import seed_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of model names to train (e.g. KalmanNetGain). "
            "When omitted, all enabled models in the config are trained."
        ),
    )
    return p


def parse_models_filter(raw: str | None) -> set[str] | None:
    """Parse the --models CLI value into a set of requested model names.

    Returns None when no filter is requested (train every enabled model).
    """
    if raw is None:
        return None
    names = {part.strip() for part in raw.split(",") if part.strip()}
    return names or None


def should_train_model(
    model_name: str,
    spec: dict[str, object],
    models_filter: set[str] | None,
) -> bool:
    """Decide whether a configured model should be trained this run.

    A model is trained only when it is enabled in the config and, if a
    --models filter is supplied, its name is in that filter. This lets us
    retrain only e.g. KalmanNetGain without touching other enabled models.
    """
    if not bool(spec.get("enabled", False)):
        return False
    if models_filter is not None and model_name not in models_filter:
        return False
    return True


def load_split_group(data_dir: Path, split_names: list[str]):
    return concatenate_dataset_arrays([load_dataset_npz(data_dir / f"{name}.npz") for name in split_names])


def load_split_group_or_groups(data_dir: Path, split_names: list[str]):
    arrays = [load_dataset_npz(data_dir / f"{name}.npz") for name in split_names]
    try:
        return concatenate_dataset_arrays(arrays)
    except ValueError as exc:
        if "station geometry" not in str(exc):
            raise
        return arrays


def main() -> None:
    run_started_at = utc_now_iso()
    run_perf_start = time.perf_counter()
    args = build_parser().parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    cfg_text = cfg_path.read_text(encoding="utf-8")
    seed = int(cfg["seed"])
    seed_all(seed)

    train_cfg = parse_train_config(cfg["training"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(train_cfg, device=str(cfg.get("device", {}).get("train", train_cfg.device)))
    device = resolve_device(train_cfg.device)

    data_dir = Path(cfg["data"]["output_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    manifest_dir = Path(cfg["output"]["manifest_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_env_report(cfg["output"]["env_report_path"], device=device)

    dataset_cfg = parse_dataset_config(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    models_cfg = cfg.get("models", {})
    models_filter = parse_models_filter(args.models)
    if models_filter is not None:
        unknown = sorted(models_filter - set(models_cfg))
        if unknown:
            raise SystemExit(f"--models requested unknown model(s): {', '.join(unknown)}")
        print(f"Restricting training to requested models: {', '.join(sorted(models_filter))}")
    curriculum = cfg.get("curriculum", {}).get("stages", [])
    if not curriculum:
        curriculum = [{"name": "default", "train_splits": ["train"], "val_splits": ["val"], "epochs": train_cfg.num_epochs}]

    history_payload: dict[str, object] = {}
    for model_name, spec in models_cfg.items():
        if not should_train_model(model_name, spec, models_filter):
            continue
        model_seed = seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(model_name)) % 10_000
        print(f"\n=== Training {model_name} on {device} ===")
        current_checkpoint = None
        stage_history = []
        final_checkpoint = None
        for stage_idx, stage in enumerate(curriculum):
            stage_name = str(stage["name"])
            stage_started_at = utc_now_iso()
            stage_perf_start = time.perf_counter()
            train_arrays = load_split_group(data_dir, list(stage["train_splits"]))
            val_arrays = load_split_group_or_groups(data_dir, list(stage["val_splits"]))
            stage_cfg = replace(train_cfg, num_epochs=int(stage.get("epochs", train_cfg.num_epochs)))
            checkpoint_name = (
                spec["checkpoint_name"]
                if stage_idx == len(curriculum) - 1
                else f"{Path(spec['checkpoint_name']).stem}_{stage.get('checkpoint_suffix', stage_name)}.pt"
            )
            _, hist, best_path = train_model(
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                cfg=stage_cfg,
                output_dir=ckpt_dir,
                seed=model_seed + stage_idx,
                use_ekf_prior=bool(spec.get("use_ekf_prior", False)),
                model_kwargs=dict(spec.get("model_kwargs", {})),
                dataset_cfg=dataset_cfg,
                baseline_cfg=baseline_cfg,
                checkpoint_name=checkpoint_name,
                initial_checkpoint=current_checkpoint,
                device=device,
            )
            final_checkpoint = best_path
            current_checkpoint = best_path
            stage_history.append(
                {
                    "stage": stage_name,
                    "train_splits": list(stage["train_splits"]),
                    "val_splits": list(stage["val_splits"]),
                    "checkpoint": str(best_path),
                    "history": hist,
                }
            )
            build_run_manifest(
                command=["train_models.py", "--config", str(cfg_path), "--device", str(device)],
                config_text=cfg_text,
                config_path=cfg_path,
                output_path=manifest_dir / f"{model_name}_{stage_name}.json",
                device=device,
                seed=model_seed + stage_idx,
                dataset_paths={name: data_dir / f"{name}.npz" for name in [*stage["train_splits"], *stage["val_splits"]]},
                checkpoint_path=best_path,
                extra={"model_name": model_name, "stage": stage_name},
                repo_root=cfg_path.parent.parent,
                timing=duration_metadata(stage_perf_start, started_at_utc=stage_started_at),
            )
        history_payload[model_name] = {
            "best_checkpoint": str(final_checkpoint) if final_checkpoint is not None else None,
            "stages": stage_history,
            "model_kwargs": spec.get("model_kwargs", {}),
        }

    dump_json(
        {
            "run_started_at_utc": run_started_at,
            "run_duration_sec": float(max(time.perf_counter() - run_perf_start, 0.0)),
            "models": history_payload,
        },
        ckpt_dir / "train_history.json",
    )
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
