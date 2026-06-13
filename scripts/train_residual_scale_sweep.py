#!/usr/bin/env python
"""Loop-38 R3: train residual-scale-sweep comparators to characterise whether
the canonical-vs-unconstrained learned-residual gap is binary (i.e.\ a sharp
collapse upon removing the residual budget and the prior-anchoring penalty) or
graded (a smooth degradation as the residual scale is raised while the
tethering structure is preserved).

The reviewer noted that the canonical residual (residual_scale 0.03 with a
bounded ``tanh`` saturation, a learned per-dimension gate, a learned context
budget, and a prior-anchoring penalty) and the loop-37 RGR-U comparator
(residual_scale 1.0 with all of those tethers removed) are very different
constructions. The 7 km gap could therefore reflect raising the scale
33$\times$ \emph{or} removing every tether; the loop-37 single-point
comparison cannot distinguish the two.

This script trains additional residual-scale-sweep comparators that share the
canonical bounded/anchored architecture and curriculum and \emph{only} change
the residual scale, so the scale effect is isolated from the
budget/gate/anchor effect:

* ``--variant tethered`` keeps the canonical ``bounded_residual=True``,
  ``use_gating=True``, ``use_context_budget=True``, and
  ``residual_reg_weight=0.05`` (the prior-anchoring penalty); only
  ``residual_scale`` is swept.
* ``--variant untethered`` matches the loop-37 RGR-U construction at the
  selected residual scale (no bound, no gate, no context budget, zero
  prior-anchoring/activity/entropy/visibility penalties).

Everything else (hidden width, GNN/GRU depth, dropout, optimiser, learning
rate, the three curriculum stages and their splits/epochs, the prior-bank and
innovation inputs) is byte-for-byte the canonical configuration. The output
checkpoint and a per-run training-history file are written under a
non-canonical, scale-tagged name so the shared canonical
``train_history.json`` is never clobbered.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from gnn_state_estimation.dataset import concatenate_dataset_arrays, load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import load_yaml
from gnn_state_estimation.utils.runtime import resolve_device
from gnn_state_estimation.utils.seeding import seed_all

ROOT = Path(__file__).resolve().parents[1]

# Variants: keeping or removing the tethering mechanisms while varying the
# residual scale only.
TETHERED_MODEL_KWARGS = {
    "use_graph": True,
    "use_gating": True,            # canonical: learned per-dim gate retained
    "bounded_residual": True,      # canonical: tanh saturation retained
    "use_innovation_features": True,
    "use_context_budget": True,    # canonical: learned budget retained
    "use_prior_bank_fusion": True,
    "prior_bank_size": 3,
    "predict_noise_scale": True,
    "fusion_temperature": 1.0,
}
TETHERED_TRAIN_OVERRIDES = {
    # Canonical prior-anchoring + auxiliary penalties retained.
}
UNTETHERED_MODEL_KWARGS = {
    "use_graph": True,
    "use_gating": False,           # vs tethered: no learned gate
    "bounded_residual": False,     # vs tethered: no tanh saturation
    "use_innovation_features": True,
    "use_context_budget": False,   # vs tethered: no learned budget
    "use_prior_bank_fusion": True,
    "prior_bank_size": 3,
    "predict_noise_scale": True,
    "fusion_temperature": 1.0,
}
UNTETHERED_TRAIN_OVERRIDES = {
    "residual_reg_weight": 0.0,
    "residual_activity_weight": 0.0,
    "residual_activity_floor": 0.0,
    "fusion_entropy_weight": 0.0,
    "fusion_entropy_floor": 0.0,
    "visibility_consistency_weight": 0.0,
}
# Disjoint, documented seed offset (not shared with any canonical model seed
# or the loop-37 unconstrained-residual seed offset 37000).
SEED_OFFSET = 38000


def load_split_group(data_dir: Path, split_names: list[str]):
    return concatenate_dataset_arrays(
        [load_dataset_npz(data_dir / f"{name}.npz") for name in split_names]
    )


def load_split_group_or_groups(data_dir: Path, split_names: list[str]):
    arrays = [load_dataset_npz(data_dir / f"{name}.npz") for name in split_names]
    try:
        return concatenate_dataset_arrays(arrays)
    except ValueError as exc:
        if "station geometry" not in str(exc):
            raise
        return arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--residual-scale",
        type=float,
        required=True,
        help="Residual scale s. Canonical RGR-GF uses 0.03; RGR-U uses 1.0.",
    )
    ap.add_argument(
        "--variant",
        choices=("tethered", "untethered"),
        required=True,
        help=(
            "tethered: canonical bounded/anchored/gated architecture, only the "
            "residual scale changes. untethered: residual budget, gate, "
            "context budget, and prior-anchoring/auxiliary penalties removed."
        ),
    )
    ap.add_argument(
        "--checkpoint-name",
        default=None,
        help="Override checkpoint stem; defaults to residual_scale_<variant>_s<scale>.pt",
    )
    ap.add_argument(
        "--seed-offset",
        type=int,
        default=None,
        help="Override the documented seed offset for this run (defaults to 38000 + scale-tag).",
    )
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    base_seed = int(cfg["seed"])
    seed_all(base_seed)

    train_cfg = parse_train_config(cfg["training"])
    overrides = (
        TETHERED_TRAIN_OVERRIDES if args.variant == "tethered" else UNTETHERED_TRAIN_OVERRIDES
    )
    train_cfg = replace(train_cfg, **overrides)
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(
            train_cfg, device=str(cfg.get("device", {}).get("train", train_cfg.device))
        )
    device = resolve_device(train_cfg.device)

    model_kwargs = dict(
        TETHERED_MODEL_KWARGS if args.variant == "tethered" else UNTETHERED_MODEL_KWARGS
    )
    model_kwargs["residual_scale"] = float(args.residual_scale)

    data_dir = Path(cfg["data"]["output_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = parse_dataset_config(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    curriculum = cfg.get("curriculum", {}).get("stages", [])
    if not curriculum:
        curriculum = [
            {
                "name": "default",
                "train_splits": ["train"],
                "val_splits": ["val"],
                "epochs": train_cfg.num_epochs,
            }
        ]

    scale_tag = f"s{int(round(args.residual_scale * 100)):03d}"
    if args.checkpoint_name is not None:
        checkpoint_stem = Path(args.checkpoint_name).stem
    else:
        checkpoint_stem = f"residual_scale_{args.variant}_{scale_tag}"

    seed_offset = (
        int(args.seed_offset)
        if args.seed_offset is not None
        else SEED_OFFSET + int(round(args.residual_scale * 100))
    )
    model_seed = base_seed + seed_offset

    print(f"CUDA available: {torch.cuda.is_available()} -> device={device}")
    print(
        f"Sweep variant={args.variant} scale={args.residual_scale} seed_offset={seed_offset}"
    )
    print(
        "Sweep overrides: "
        f"model_kwargs={json.dumps(model_kwargs)} "
        f"train_overrides={json.dumps(overrides)}"
    )

    current_checkpoint = None
    final_checkpoint = None
    stage_history = []
    t0 = time.perf_counter()
    for stage_idx, stage in enumerate(curriculum):
        stage_name = str(stage["name"])
        train_arrays = load_split_group(data_dir, list(stage["train_splits"]))
        val_arrays = load_split_group_or_groups(data_dir, list(stage["val_splits"]))
        stage_cfg = replace(
            train_cfg, num_epochs=int(stage.get("epochs", train_cfg.num_epochs))
        )
        if stage_idx == len(curriculum) - 1:
            checkpoint_name = f"{checkpoint_stem}.pt"
        else:
            suffix = stage.get("checkpoint_suffix", stage_name)
            checkpoint_name = f"{checkpoint_stem}_{suffix}.pt"
        print(f"\n=== Stage {stage_idx + 1}/{len(curriculum)}: {stage_name} ===")
        _, hist, best_path = train_model(
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            cfg=stage_cfg,
            output_dir=ckpt_dir,
            seed=model_seed + stage_idx,
            use_ekf_prior=True,
            model_kwargs=dict(model_kwargs),
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
                "best_val_loss": float(
                    min(v for v in hist["val_loss"] if v == v)
                    if any(v == v for v in hist["val_loss"])
                    else float("nan")
                ),
                "epochs": len(hist["train_loss"]),
            }
        )

    out_dir = ROOT / "results" / "residual_scale_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed",
        "schema_version": "residual_scale_sweep_training_v1",
        "purpose": (
            "loop-38 R3 sweep: characterise whether the canonical-vs-RGR-U gap "
            "is binary or smooth by training intermediate residual scales "
            "under matched curriculum/architecture"
        ),
        "variant": args.variant,
        "residual_scale": float(args.residual_scale),
        "model_kwargs": model_kwargs,
        "train_overrides": overrides,
        "seed_offset": seed_offset,
        "base_seed": base_seed,
        "model_seed": model_seed,
        "final_checkpoint": str(final_checkpoint),
        "duration_sec": float(max(time.perf_counter() - t0, 0.0)),
        "stages": stage_history,
    }
    history_name = f"train_history_{checkpoint_stem}.json"
    (out_dir / history_name).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved sweep checkpoint: {final_checkpoint}")
    print(f"Wrote {out_dir / history_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
