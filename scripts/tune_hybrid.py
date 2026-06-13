#!/usr/bin/env python
"""Optuna search for the stronger hybrid configuration."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
from dataclasses import replace
from pathlib import Path

import optuna

from gnn_state_estimation.dataset import concatenate_dataset_arrays, load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.runtime import resolve_device


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    return p


def _split_group(data_dir: Path, split_names: list[str]):
    return concatenate_dataset_arrays([load_dataset_npz(data_dir / f"{name}.npz") for name in split_names])


def _split_group_or_groups(data_dir: Path, split_names: list[str]):
    arrays = [load_dataset_npz(data_dir / f"{name}.npz") for name in split_names]
    try:
        return concatenate_dataset_arrays(arrays)
    except ValueError as exc:
        if "station geometry" not in str(exc):
            raise
        return arrays


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    tuning_cfg = cfg.get("tuning", {})
    if not bool(tuning_cfg.get("enabled", False)):
        print("Tuning disabled in config.")
        return

    train_cfg = parse_train_config(cfg["training"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    device = resolve_device(train_cfg.device)
    data_dir = Path(cfg["data"]["output_dir"])
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    models_cfg = cfg.get("models", {})
    model_name = str(args.model or tuning_cfg.get("target_model", "InnovationHybridGNN"))
    if model_name not in models_cfg or not bool(models_cfg[model_name].get("enabled", False)):
        raise ValueError(f"Model {model_name!r} is not enabled in the config.")
    model_spec = models_cfg[model_name]

    stage_cfg = cfg.get("curriculum", {}).get("stages", [])
    mixed_stage = next((s for s in stage_cfg if s.get("name") == "mixed_train"), None)
    if mixed_stage is None:
        mixed_stage = {"train_splits": ["train", "stress_train"], "val_splits": ["val", "stress_val"]}
    train_splits = list(tuning_cfg.get("train_splits", mixed_stage["train_splits"]))
    val_splits = list(tuning_cfg.get("val_splits", mixed_stage["val_splits"]))
    train_arrays = _split_group(data_dir, train_splits)
    val_arrays = _split_group_or_groups(data_dir, val_splits)

    out_dir = Path("results/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = replace(train_cfg, num_epochs=min(train_cfg.num_epochs, 18), early_stopping_patience=5)
    search = tuning_cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = int(trial.suggest_categorical("hidden_dim", search["hidden_dim"]))
        dropout = float(trial.suggest_categorical("dropout", search["dropout"]))
        residual_scale = float(trial.suggest_categorical("residual_scale", search["residual_scale"]))
        fusion_temperature = float(trial.suggest_categorical("fusion_temperature", search["fusion_temperature"]))
        residual_activity_weight = float(
            trial.suggest_categorical("residual_activity_weight", search["residual_activity_weight"])
        )
        fusion_entropy_weight = float(
            trial.suggest_categorical("fusion_entropy_weight", search["fusion_entropy_weight"])
        )
        trial_cfg = replace(
            train_cfg,
            hidden_dim=hidden_dim,
            dropout=dropout,
            residual_activity_weight=residual_activity_weight,
            fusion_entropy_weight=fusion_entropy_weight,
        )
        model_kwargs = dict(model_spec.get("model_kwargs", {}))
        model_kwargs["residual_scale"] = residual_scale
        if bool(model_kwargs.get("use_prior_bank_fusion", False)):
            model_kwargs["fusion_temperature"] = fusion_temperature
        _, hist, best_path = train_model(
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            cfg=trial_cfg,
            output_dir=out_dir,
            seed=int(cfg["seed"]) + trial.number,
            use_ekf_prior=bool(model_spec.get("use_ekf_prior", False)),
            model_kwargs=model_kwargs,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            checkpoint_name=f"{model_name.lower()}_trial_{trial.number:03d}.pt",
            device=device,
        )
        best_val = min(hist["val_loss"])
        trial.set_user_attr("checkpoint", str(best_path))
        return float(best_val)

    study = optuna.create_study(
        study_name=str(tuning_cfg.get("study_name", "hybrid_search")) + f"_{model_name}",
        storage=str(tuning_cfg.get("storage", "sqlite:///results/tuning/hybrid_search.db")),
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=int(tuning_cfg.get("n_trials", 10)),
        timeout=int(tuning_cfg.get("timeout_sec", 3600)),
    )
    dump_json(
        {
            "best_value": float(study.best_value),
            "best_params": study.best_params,
            "best_trial_number": int(study.best_trial.number),
            "trial_count": len(study.trials),
            "model": model_name,
            "train_splits": train_splits,
            "val_splits": val_splits,
        },
        out_dir / f"{model_name.lower()}_best_trial.json",
    )
    print(f"{model_name} best trial {study.best_trial.number}: {study.best_value:.6f}")


if __name__ == "__main__":
    main()
