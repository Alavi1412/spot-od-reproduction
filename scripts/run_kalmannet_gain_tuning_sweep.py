#!/usr/bin/env python
"""Validation-tuned sweep for the KalmanNet-style learned-gain comparator.

The in-house learned-gain comparator (EKF prior retained, analytic Kalman
gain replaced by a learned gain over normalized innovations, adapted from
Revach et al.) is trained over a small *predeclared* grid of its two
sensitivity hyperparameters: the learned-gain scale and the bounded
state-correction clip. Every other ingredient (curriculum stages, optimiser,
seeds, data splits, architecture) is held fixed.

The selection rule is fixed before any run and uses *only* held-out
validation loss --- no test-set information enters selection:

  winner = argmin over the grid of the lowest finite final-stage
           (stress-focus) validation loss; ties broken by smaller gain
           scale, then smaller correction clip.

The selected configuration's curriculum checkpoints are promoted to the
canonical KalmanNetGain checkpoint names and its hyperparameters are written
back into the experiment configuration so the standard evaluation reproduces
the validation-tuned comparator. A per-run sweep summary is written for
provenance.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import math
import shutil
import time
from dataclasses import replace
from pathlib import Path

try:
    from train_models import load_split_group, load_split_group_or_groups
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts.train_models import load_split_group, load_split_group_or_groups

from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.training import parse_train_config, train_model
from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.runtime import (
    duration_metadata,
    resolve_device,
    utc_now_iso,
)

# Predeclared grid: gain scale across (0.5x, 1x, 2x) the prior default at the
# stable correction clip, plus one looser-clip variant. Frozen before any run.
PREDECLARED_GRID: list[dict[str, object]] = [
    {"id": "g5e4_c5e3", "kalmannet_gain_scale": 5.0e-4, "kalmannet_correction_clip": 5.0e-3},
    {"id": "g1e3_c5e3", "kalmannet_gain_scale": 1.0e-3, "kalmannet_correction_clip": 5.0e-3},
    {"id": "g2e3_c5e3", "kalmannet_gain_scale": 2.0e-3, "kalmannet_correction_clip": 5.0e-3},
    {"id": "g1e3_c1e2", "kalmannet_gain_scale": 1.0e-3, "kalmannet_correction_clip": 1.0e-2},
]

SELECTION_RULE = (
    "argmin over the predeclared grid of the lowest finite final-stage "
    "(stress-focus) held-out validation loss; ties broken by smaller "
    "gain scale, then smaller correction clip (no test-set information used)"
)


def final_stage_val_score(stage_histories: list[dict[str, object]]) -> float:
    """Predeclared per-configuration selection score.

    The score is the minimum finite validation loss observed during the
    final curriculum stage (the stress-focus stage that produces the
    promoted checkpoint). Non-finite values are ignored; if no finite
    value exists the configuration scores ``+inf`` (never selected).
    """
    if not stage_histories:
        return float("inf")
    final = stage_histories[-1]
    val = [float(v) for v in final.get("history", {}).get("val_loss", []) if math.isfinite(float(v))]
    return min(val) if val else float("inf")


def select_winner(results: list[dict[str, object]]) -> str:
    """Apply the predeclared selection rule and return the winning grid id.

    ``results`` entries must carry ``id``, ``kalmannet_gain_scale``,
    ``kalmannet_correction_clip`` and ``val_score``. Selection is a pure
    function of validation scores and the predeclared tie-break, so it is
    unit-testable without any training.
    """
    if not results:
        raise ValueError("no sweep results to select from")

    def key(entry: dict[str, object]) -> tuple[float, float, float]:
        return (
            float(entry["val_score"]),
            float(entry["kalmannet_gain_scale"]),
            float(entry["kalmannet_correction_clip"]),
        )

    return str(min(results, key=key)["id"])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/experiment.yaml")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--summary-path",
        type=str,
        default="results/kalmannet_gain_tuning_sweep.json",
    )
    return p


def main() -> None:
    run_started_at = utc_now_iso()
    run_perf_start = time.perf_counter()
    args = _build_parser().parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    train_cfg = parse_train_config(cfg["training"])
    if args.device is not None:
        train_cfg = replace(train_cfg, device=args.device)
    else:
        train_cfg = replace(
            train_cfg, device=str(cfg.get("device", {}).get("train", train_cfg.device))
        )
    device = resolve_device(train_cfg.device)

    data_dir = Path(cfg["data"]["output_dir"])
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    sweep_root = ckpt_dir / "kng_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)

    dataset_cfg = parse_dataset_config(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])

    spec = cfg["models"]["KalmanNetGain"]
    base_kwargs = dict(spec.get("model_kwargs", {}))
    base_seed = int(cfg["seed"])
    model_seed = base_seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate("KalmanNetGain")) % 10_000
    curriculum = cfg.get("curriculum", {}).get("stages", [])

    results: list[dict[str, object]] = []
    for point in PREDECLARED_GRID:
        cid = str(point["id"])
        kwargs = dict(base_kwargs)
        kwargs["kalmannet_gain_scale"] = float(point["kalmannet_gain_scale"])
        kwargs["kalmannet_correction_clip"] = float(point["kalmannet_correction_clip"])
        point_dir = sweep_root / cid
        point_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== KalmanNetGain sweep config {cid} on {device}: {kwargs} ===")

        current_checkpoint = None
        stage_histories: list[dict[str, object]] = []
        final_checkpoint = None
        for stage_idx, stage in enumerate(curriculum):
            stage_name = str(stage["name"])
            train_arrays = load_split_group(data_dir, list(stage["train_splits"]))
            val_arrays = load_split_group_or_groups(data_dir, list(stage["val_splits"]))
            stage_cfg = replace(
                train_cfg, num_epochs=int(stage.get("epochs", train_cfg.num_epochs))
            )
            checkpoint_name = (
                spec["checkpoint_name"]
                if stage_idx == len(curriculum) - 1
                else f"{Path(spec['checkpoint_name']).stem}_{stage.get('checkpoint_suffix', stage_name)}.pt"
            )
            _, hist, best_path = train_model(
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                cfg=stage_cfg,
                output_dir=point_dir,
                seed=model_seed + stage_idx,
                use_ekf_prior=bool(spec.get("use_ekf_prior", False)),
                model_kwargs=kwargs,
                dataset_cfg=dataset_cfg,
                baseline_cfg=baseline_cfg,
                checkpoint_name=checkpoint_name,
                initial_checkpoint=current_checkpoint,
                device=device,
            )
            final_checkpoint = best_path
            current_checkpoint = best_path
            stage_histories.append(
                {
                    "stage": stage_name,
                    "checkpoint_name": checkpoint_name,
                    "checkpoint": str(best_path),
                    "history": hist,
                }
            )
        results.append(
            {
                "id": cid,
                "kalmannet_gain_scale": float(point["kalmannet_gain_scale"]),
                "kalmannet_correction_clip": float(point["kalmannet_correction_clip"]),
                "val_score": final_stage_val_score(stage_histories),
                "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
                "stages": stage_histories,
            }
        )

    winner_id = select_winner(results)
    winner = next(r for r in results if r["id"] == winner_id)
    print(
        f"\nSelected KalmanNetGain config '{winner_id}' "
        f"(gain_scale={winner['kalmannet_gain_scale']}, "
        f"correction_clip={winner['kalmannet_correction_clip']}, "
        f"val_score={winner['val_score']:.6f})"
    )

    # Promote the winner's curriculum checkpoints to the canonical names.
    for stage in winner["stages"]:
        src = Path(str(stage["checkpoint"]))
        dst = ckpt_dir / str(stage["checkpoint_name"])
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)

    # Write the winning hyperparameters back into the experiment config so
    # the standard evaluation and provenance reflect the tuned selection.
    cfg_text = cfg_path.read_text(encoding="utf-8")
    new_text = cfg_text
    for kw, value in (
        ("kalmannet_gain_scale", winner["kalmannet_gain_scale"]),
        ("kalmannet_correction_clip", winner["kalmannet_correction_clip"]),
    ):
        rendered = f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        import re

        new_text, n = re.subn(
            rf"(\n      {kw}: )[0-9.eE+-]+",
            rf"\g<1>{rendered}",
            new_text,
        )
        if n != 1:
            raise SystemExit(f"expected exactly one '{kw}:' line to update, found {n}")
    if new_text != cfg_text:
        cfg_path.write_text(new_text, encoding="utf-8")

    summary = {
        "run_started_at_utc": run_started_at,
        "run_duration_sec": float(max(time.perf_counter() - run_perf_start, 0.0)),
        "device": str(device),
        "selection_rule": SELECTION_RULE,
        "predeclared_grid": PREDECLARED_GRID,
        "selected_id": winner_id,
        "selected_kalmannet_gain_scale": winner["kalmannet_gain_scale"],
        "selected_kalmannet_correction_clip": winner["kalmannet_correction_clip"],
        "selected_val_score": winner["val_score"],
        "results": [
            {
                "id": r["id"],
                "kalmannet_gain_scale": r["kalmannet_gain_scale"],
                "kalmannet_correction_clip": r["kalmannet_correction_clip"],
                "val_score": r["val_score"],
            }
            for r in results
        ],
        "timing": duration_metadata(run_perf_start, started_at_utc=run_started_at),
    }
    dump_json(summary, Path(args.summary_path))
    print(f"\nSweep summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
