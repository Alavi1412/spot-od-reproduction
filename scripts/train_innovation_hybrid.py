#!/usr/bin/env python
"""Train only the innovation-conditioned hybrid estimator."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from pathlib import Path

import torch

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import load_yaml
from gnn_state_estimation.utils.seeding import seed_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--checkpoint-name", type=str, default="best_innovation_hybrid.pt")
    p.add_argument("--seed-offset", type=int, default=2)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["seed"]) + int(args.seed_offset)
    seed_all(seed)

    data_dir = Path(cfg["data"]["output_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_arrays = load_dataset_npz(data_dir / "train.npz")
    val_arrays = load_dataset_npz(data_dir / "val.npz")
    train_cfg = parse_train_config(cfg["training"])
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n=== Training innovation-conditioned hybrid estimator ===")
    _, history, best_path = train_model(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        cfg=train_cfg,
        output_dir=ckpt_dir,
        seed=seed,
        use_ekf_prior=True,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        checkpoint_name=args.checkpoint_name,
        model_kwargs={
            "use_graph": True,
            "residual_scale": 0.03,
            "use_gating": True,
            "bounded_residual": True,
            "use_innovation_features": True,
            "use_context_budget": True,
            "use_prior_bank_fusion": True,
            "prior_bank_size": 3,
            "predict_noise_scale": True,
            "fusion_temperature": 0.75,
        },
    )
    print(f"Saved: {best_path}")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")

    history_path = ckpt_dir / "train_history.json"
    payload: dict[str, object]
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload["innovation_hybrid"] = {"best_checkpoint": str(best_path), "history": history}
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
