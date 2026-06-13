#!/usr/bin/env python
"""Loop-37 FF1: train a deliberately *less-constrained* learned residual
comparator (the "RGR-U" family) to test whether the study's central negative
is an artifact of the tight residual budget and the prior-anchoring penalty.

A reviewer noted that the canonical RGR-GF residual is hard-bounded
(``residual = 0.03 * tanh(raw)`` in scaled state units), additionally damped
by a learned context budget and a per-dimension gate, and trained with a
prior-anchoring penalty (and auxiliary activity/entropy/visibility penalties),
"so any competitive-with-classical behaviour is in substantial part inherited
from the classical prior bank by construction." If so, "learned does not beat
classical" would be partly a design consequence rather than a finding.

This script removes exactly those tethering mechanisms while holding the
architecture, capacity, inputs, training data, curriculum, and seed-derivation
identical to the canonical RGR-GF training. Concretely the comparator differs
from RGR-GF only by:

* ``residual_scale`` 0.03 -> 1.0 (a scaled-unit residual of 1.0 spans the full
  position scale, ~1e7 m, so the head can wholly override the classical prior),
* ``bounded_residual`` True -> False (no ``tanh`` saturation),
* ``use_context_budget`` True -> False (no learned [0,1] residual budget),
* ``use_gating`` True -> False (no learned [0,1] per-dimension gate),
* ``residual_reg_weight`` 0.05 -> 0.0 (no prior-anchoring penalty),
* the auxiliary ``residual_activity`` / ``fusion_entropy`` /
  ``visibility_consistency`` penalties -> 0.0.

Everything else (hidden width, GNN/GRU depth, dropout, optimiser, learning
rate, the three curriculum stages and their splits/epochs, the prior-bank and
innovation inputs) is byte-for-byte the canonical configuration. The resulting
estimator is therefore free to depart arbitrarily from its classical priors;
whatever it does is learned, not imposed by a bound or an anchor.

Predeclared before running: the comparator hyperparameters above are fixed in
this file; the model is trained once on the canonical curriculum and then
evaluated, in inference only, by ``build_unconstrained_residual_comparator.py``
on a fresh independent realization set. No selection, tuning, or retraining is
performed on the evaluation realizations, and no canonical artifact
(``train_history.json``, the configured checkpoints) is written or mutated.
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

# Predeclared comparator definition (fixed before any result is produced).
UNCONSTRAINED_MODEL_KWARGS = {
    "use_graph": True,
    "residual_scale": 1.0,            # vs 0.03 canonical: full-scale residual
    "use_gating": False,             # vs True canonical: no learned gate
    "bounded_residual": False,       # vs True canonical: no tanh saturation
    "use_innovation_features": True,
    "use_context_budget": False,     # vs True canonical: no learned budget
    "use_prior_bank_fusion": True,
    "prior_bank_size": 3,
    "predict_noise_scale": True,
    "fusion_temperature": 1.0,
}
# Tethering training penalties all set to zero (vs the canonical config).
UNCONSTRAINED_TRAIN_OVERRIDES = {
    "residual_reg_weight": 0.0,
    "residual_activity_weight": 0.0,
    "residual_activity_floor": 0.0,
    "fusion_entropy_weight": 0.0,
    "fusion_entropy_floor": 0.0,
    "visibility_consistency_weight": 0.0,
}
# Disjoint, documented seed offset (not shared with any canonical model seed).
SEED_OFFSET = 37000


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
    ap.add_argument("--checkpoint-name", default="unconstrained_residual.pt")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    base_seed = int(cfg["seed"])
    seed_all(base_seed)

    train_cfg = parse_train_config(cfg["training"])
    train_cfg = replace(train_cfg, **UNCONSTRAINED_TRAIN_OVERRIDES)
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(
            train_cfg, device=str(cfg.get("device", {}).get("train", train_cfg.device))
        )
    device = resolve_device(train_cfg.device)

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

    print(f"CUDA available: {torch.cuda.is_available()} -> device={device}")
    print(
        "Comparator overrides: "
        f"model_kwargs={json.dumps(UNCONSTRAINED_MODEL_KWARGS)} "
        f"train_overrides={json.dumps(UNCONSTRAINED_TRAIN_OVERRIDES)}"
    )

    model_seed = base_seed + SEED_OFFSET
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
            checkpoint_name = args.checkpoint_name
        else:
            checkpoint_name = (
                f"{Path(args.checkpoint_name).stem}_"
                f"{stage.get('checkpoint_suffix', stage_name)}.pt"
            )
        print(f"\n=== Stage {stage_idx + 1}/{len(curriculum)}: {stage_name} ===")
        _, hist, best_path = train_model(
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            cfg=stage_cfg,
            output_dir=ckpt_dir,
            seed=model_seed + stage_idx,
            use_ekf_prior=True,
            model_kwargs=dict(UNCONSTRAINED_MODEL_KWARGS),
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

    out_dir = ROOT / "results" / "unconstrained_residual_comparator"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed",
        "schema_version": "unconstrained_residual_training_v1",
        "purpose": (
            "less-constrained learned residual comparator answering the "
            "preordained-negative objection: identical architecture/curriculum "
            "to the canonical bounded residual, with the residual budget, the "
            "learned gate, the context budget, and the prior-anchoring/"
            "activity/entropy/visibility penalties removed"
        ),
        "model_kwargs": UNCONSTRAINED_MODEL_KWARGS,
        "train_overrides": UNCONSTRAINED_TRAIN_OVERRIDES,
        "seed_offset": SEED_OFFSET,
        "base_seed": base_seed,
        "model_seed": model_seed,
        "final_checkpoint": str(final_checkpoint),
        "duration_sec": float(max(time.perf_counter() - t0, 0.0)),
        "stages": stage_history,
    }
    (out_dir / "train_history_unconstrained.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved comparator checkpoint: {final_checkpoint}")
    print(f"Wrote {out_dir / 'train_history_unconstrained.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
