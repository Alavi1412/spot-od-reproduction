#!/usr/bin/env python
"""Run and report a bounded learned-estimator retraining replay.

This intentionally uses deterministic slices of the already materialized
training data and writes all training outputs under results/retraining_replay.
It is a targeted replay artifact, not a full scientific rerun.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from gnn_state_estimation.utils.io import dump_json, load_yaml
from gnn_state_estimation.utils.runtime import sha256_file, sha256_text, utc_now_iso


REPORT_SCHEMA_VERSION = "targeted-retraining-replay-v2"
DEFAULT_MODEL = "ObservabilityContextHybridGNN"
CANONICAL_CHECKPOINT_DIR = Path("results/checkpoints")
PUBLIC_REPLAY_OUTPUT_ROOT = Path("results/retraining_replay/targeted_retraining_replay")
DEFAULT_PREDECLARATION = Path(
    "release/predeclarations/targeted_curriculum_retraining_replay_20260525.json"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="configs/experiment.yaml")
    p.add_argument("--source-data-dir", default="results/data")
    p.add_argument("--output-root", default=str(PUBLIC_REPLAY_OUTPUT_ROOT))
    p.add_argument("--validation-dir", default="results/validation")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train-count", type=int, default=8)
    p.add_argument("--val-count", type=int, default=4)
    p.add_argument("--stress-train-count", type=int, default=None)
    p.add_argument("--stress-val-count", type=int, default=None)
    p.add_argument("--satnogs-val-count", type=int, default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--nominal-epochs", type=int, default=3)
    p.add_argument("--mixed-epochs", type=int, default=3)
    p.add_argument("--stress-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=1)
    p.add_argument("--gru-layers", type=int, default=1)
    p.add_argument("--seed", type=int, default=86086)
    p.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--full-materialized-curriculum",
        action="store_true",
        help=(
            "Replay the main nominal/mixed/stress curriculum shape using "
            "materialized train, stress, and validation splits."
        ),
    )
    p.add_argument(
        "--full-split-counts",
        action="store_true",
        help="Use every trajectory from each selected source split.",
    )
    p.add_argument(
        "--predeclared-rule",
        default=str(DEFAULT_PREDECLARATION),
        help="JSON rule fixed before this replay is evaluated.",
    )
    return p


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def resolve_replay_device(requested: str) -> tuple[str, dict[str, Any]]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    if requested == "cpu":
        raise RuntimeError(
            "Learned-estimator retraining replay requires CUDA; CPU execution "
            "is intentionally disallowed for this evidence artifact."
        )
    if not cuda_available:
        raise RuntimeError(
            "Learned-estimator retraining replay requires CUDA; aborting "
            "because torch reports no CUDA-capable device."
        )
    selected = "cuda"
    return selected, {
        "accelerated_compute_required": True,
        "accelerated_compute_used": True,
        "non_accelerated_execution_allowed": False,
        "runtime_details_redacted": True,
        "hardware_details_redacted": True,
        "software_acceleration_version_redacted": True,
    }


def rel_report_path(path: Path | str) -> str:
    p = Path(path)
    try:
        if p.is_absolute():
            return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.name
    return p.as_posix()


def public_replay_path(path: Path | str, actual_root: Path) -> str:
    p = Path(path)
    try:
        suffix = p.resolve().relative_to(actual_root.resolve())
        return (PUBLIC_REPLAY_OUTPUT_ROOT / suffix).as_posix()
    except ValueError:
        return rel_report_path(p)


def public_stage_histories(stages: list[dict[str, Any]], actual_root: Path) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for stage in stages:
        row = dict(stage)
        if row.get("checkpoint"):
            row["checkpoint"] = public_replay_path(row["checkpoint"], actual_root)
        public.append(row)
    return public


def public_data_records(records: Mapping[str, Any], actual_root: Path) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for split, record in records.items():
        row = dict(record)
        if row.get("source_path"):
            row["source_path"] = rel_report_path(row["source_path"])
        if row.get("slice_path"):
            row["slice_path"] = public_replay_path(row["slice_path"], actual_root)
        public[str(split)] = row
    return public


def public_execution_record(config_path: Path, model: str, selected_device: str) -> dict[str, Any]:
    return {
        "step": "targeted_training_replay",
        "entrypoint": "scripts/train_models.py",
        "config_redacted": True,
        "model": model,
        "accelerated_compute_required": selected_device == "cuda",
        "execution_details_redacted": True,
    }


def slice_npz(source: Path, dest: Path, count: int) -> dict[str, Any]:
    """Write a deterministic first-N trajectory slice of an NPZ dataset."""
    import numpy as np

    if count <= 0:
        raise ValueError("count must be positive")
    data = np.load(source)
    if "states" not in data.files:
        raise ValueError(f"{source} does not contain a states array")
    n_traj = int(data["states"].shape[0])
    if count > n_traj:
        raise ValueError(f"Requested {count} trajectories from {source}, but only {n_traj} exist")

    arrays: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = data[key]
        if arr.ndim > 0 and int(arr.shape[0]) == n_traj:
            arrays[key] = arr[:count]
        else:
            arrays[key] = arr
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "slice_path": str(dest),
        "slice_sha256": sha256_file(dest),
        "source_trajectories": n_traj,
        "slice_trajectories": count,
        "array_keys": list(data.files),
        "state_shape": list(arrays["states"].shape),
    }


def build_replay_config(
    *,
    base_cfg: Mapping[str, Any],
    base_config_text: str,
    model_name: str,
    data_dir: Path,
    artifacts_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    gnn_layers: int,
    gru_layers: int,
    device: str,
    curriculum_stages: list[dict[str, Any]] | None = None,
    predeclared_rule_sha256: str | None = None,
) -> dict[str, Any]:
    if model_name not in base_cfg.get("models", {}):
        raise ValueError(f"Unknown model in base config: {model_name}")

    cfg = copy.deepcopy(dict(base_cfg))
    cfg["seed"] = int(seed)
    cfg["device"] = {"train": device, "eval": device, "require_cuda": device == "cuda"}
    cfg["data"] = {
        "train_size": None,
        "val_size": None,
        "test_size": None,
        "stress_test_size": None,
        "output_dir": str(data_dir),
    }
    cfg["training"] = copy.deepcopy(dict(base_cfg["training"]))
    cfg["training"].update(
        {
            "window_size": 12,
            "batch_size": int(batch_size),
            "num_epochs": int(epochs),
            "hidden_dim": int(hidden_dim),
            "gnn_layers": int(gnn_layers),
            "gru_layers": int(gru_layers),
            "early_stopping_patience": max(int(epochs), 1),
            "num_workers": 0,
            "device": device,
            "use_amp": device == "cuda",
        }
    )
    model_spec = copy.deepcopy(dict(base_cfg["models"][model_name]))
    model_spec["enabled"] = True
    model_spec["checkpoint_name"] = f"replay_{model_name.lower()}.pt"
    cfg["models"] = {model_name: model_spec}
    if curriculum_stages is None:
        curriculum_stages = [
            {
                "name": "deterministic_slice_retrain",
                "train_splits": ["train"],
                "val_splits": ["val"],
                "epochs": int(epochs),
            }
        ]
    cfg["curriculum"] = {"stages": curriculum_stages}
    cfg["output"] = {
        "checkpoint_dir": str(artifacts_dir / "checkpoints"),
        "metrics_path": str(artifacts_dir / "metrics_summary.json"),
        "per_step_path": str(artifacts_dir / "per_step_errors.csv"),
        "figure_dir": str(artifacts_dir / "figures"),
        "env_report_path": str(artifacts_dir / "runtime" / "env_report.json"),
        "manifest_dir": str(artifacts_dir / "manifests"),
        "scorecard_path": str(artifacts_dir / "scorecard_summary.json"),
    }
    cfg["targeted_retraining_replay"] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "base_config_sha256": sha256_text(base_config_text),
        "predeclared_rule_sha256": predeclared_rule_sha256,
        "claim_boundary": (
            "Targeted bounded replay only: one learned estimator on deterministic "
            "slices; not a full main-results reproduction."
        ),
    }
    return cfg


def write_replay_config(cfg: Mapping[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(cfg), sort_keys=False)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "sha256": sha256_text(text)}


def canonical_checkpoint_digest(root: Path = CANONICAL_CHECKPOINT_DIR) -> dict[str, Any]:
    if not root.exists():
        return {"path": str(root), "exists": False, "file_count": 0, "sha256": None}
    h = __import__("hashlib").sha256()
    file_count = 0
    for path in sorted(p for p in root.iterdir() if p.is_file()):
        rel = path.name
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("utf-8"))
        h.update(b"\0")
        file_count += 1
    return {"path": str(root), "exists": True, "file_count": file_count, "sha256": h.hexdigest()}


def source_trajectory_count(path: Path) -> int:
    import numpy as np

    data = np.load(path)
    if "states" not in data.files:
        raise ValueError(f"{path} does not contain a states array")
    return int(data["states"].shape[0])


def curriculum_stage_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.full_materialized_curriculum:
        return [
            {
                "name": "deterministic_slice_retrain",
                "train_splits": ["train"],
                "val_splits": ["val"],
                "epochs": int(args.epochs),
            }
        ]
    return [
        {
            "name": "nominal_pretrain_replay",
            "train_splits": ["train"],
            "val_splits": ["val"],
            "epochs": int(args.nominal_epochs),
        },
        {
            "name": "mixed_train_replay",
            "train_splits": ["train", "stress_train"],
            "val_splits": [
                "val",
                "stress_val",
                "satnogs_observation_replay_val",
            ],
            "epochs": int(args.mixed_epochs),
        },
        {
            "name": "stress_focus_replay",
            "train_splits": ["stress_train"],
            "val_splits": ["stress_val", "satnogs_observation_replay_val"],
            "epochs": int(args.stress_epochs),
        },
    ]


def split_count_plan(args: argparse.Namespace, source_data_dir: Path) -> dict[str, int]:
    requested = {"train": args.train_count, "val": args.val_count}
    if args.full_materialized_curriculum:
        requested.update(
            {
                "stress_train": args.stress_train_count,
                "stress_val": args.stress_val_count,
                "satnogs_observation_replay_val": args.satnogs_val_count,
            }
        )
    out: dict[str, int] = {}
    for split, count in requested.items():
        source = source_data_dir / f"{split}.npz"
        n_source = source_trajectory_count(source)
        out[split] = n_source if args.full_split_counts or count is None else int(count)
    return out


def load_predeclared_rule(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Predeclared replay rule not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {"path": rel_report_path(path), "sha256": sha256_file(path)}


def validate_against_predeclaration(
    *,
    rule: Mapping[str, Any],
    args: argparse.Namespace,
    stages: list[dict[str, Any]],
    split_counts: Mapping[str, int],
) -> None:
    errors: list[str] = []
    if rule.get("model") != args.model:
        errors.append("model mismatch")
    if int(rule.get("seed", -1)) != int(args.seed):
        errors.append("seed mismatch")
    if bool(rule.get("accelerated_compute_required")) is not True:
        errors.append("accelerated_compute_required must be true")
    rule_training = rule.get("training", {})
    if isinstance(rule_training, Mapping):
        expected = {
            "batch_size": int(args.batch_size),
            "hidden_dim": int(args.hidden_dim),
            "gnn_layers": int(args.gnn_layers),
            "gru_layers": int(args.gru_layers),
        }
        for key, value in expected.items():
            if int(rule_training.get(key, -1)) != value:
                errors.append(f"training.{key} mismatch")
    else:
        errors.append("training block missing")
    expected_stages = rule.get("curriculum_stages")
    if expected_stages != stages:
        errors.append("curriculum_stages mismatch")
    expected_splits = rule.get("data_splits", {})
    if not isinstance(expected_splits, Mapping):
        errors.append("data_splits block missing")
    else:
        for split, count in split_counts.items():
            row = expected_splits.get(split)
            if not isinstance(row, Mapping):
                errors.append(f"data_splits.{split} missing")
                continue
            if int(row.get("slice_trajectories", -1)) != int(count):
                errors.append(f"data_splits.{split}.slice_trajectories mismatch")
    if errors:
        raise RuntimeError(
            "Replay arguments do not match the predeclared rule: "
            + "; ".join(errors)
        )


def flatten_stage_histories(history: Mapping[str, Any], model_name: str) -> list[dict[str, Any]]:
    model = history.get("models", {}).get(model_name, {})
    stages = model.get("stages", []) if isinstance(model, Mapping) else []
    out = []
    for stage in stages:
        hist = stage.get("history", {}) if isinstance(stage, Mapping) else {}
        train_loss = list(hist.get("train_loss", [])) if isinstance(hist, Mapping) else []
        val_loss = list(hist.get("val_loss", [])) if isinstance(hist, Mapping) else []
        out.append(
            {
                "stage": stage.get("stage"),
                "checkpoint": stage.get("checkpoint"),
                "epochs_completed": max(len(train_loss), len(val_loss)),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "final_train_loss": train_loss[-1] if train_loss else None,
                "final_val_loss": val_loss[-1] if val_loss else None,
                "best_val_loss": min([float(v) for v in val_loss if finite_float(v)], default=None),
            }
        )
    return out


def build_criteria(
    *,
    returncode: int,
    stages: list[dict[str, Any]],
    checkpoint_paths: list[Path],
    config_record: Mapping[str, Any],
    output_root: Path,
    canonical_before: Mapping[str, Any],
    canonical_after: Mapping[str, Any],
    device_info: Mapping[str, Any],
    predeclaration_record: Mapping[str, Any],
) -> dict[str, bool]:
    val_losses = [loss for stage in stages for loss in stage.get("val_loss", [])]
    train_losses = [loss for stage in stages for loss in stage.get("train_loss", [])]
    isolated = str(output_root.as_posix()).startswith("results/retraining_replay/")
    return {
        "training_step_returned_zero": returncode == 0,
        "model_history_present": bool(stages),
        "train_loss_finite": bool(train_losses) and all(finite_float(v) for v in train_losses),
        "validation_loss_finite": bool(val_losses) and all(finite_float(v) for v in val_losses),
        "checkpoint_produced": bool(checkpoint_paths) and all(p.exists() for p in checkpoint_paths),
        "checkpoint_sha256_recorded": bool(checkpoint_paths) and all(p.exists() and bool(sha256_file(p)) for p in checkpoint_paths),
        "deterministic_config_captured": bool(config_record.get("path")) and bool(config_record.get("sha256")),
        "isolated_output_dir_under_results_retraining_replay": isolated,
        "canonical_checkpoint_digest_unchanged": canonical_before == canonical_after,
        "accelerated_compute_required_and_used": bool(
            device_info.get("accelerated_compute_required")
            and device_info.get("accelerated_compute_used")
        ),
        "predeclaration_digest_recorded": bool(
            predeclaration_record.get("path") and predeclaration_record.get("sha256")
        ),
    }


def validate_report_schema(report: Mapping[str, Any]) -> list[str]:
    schema_version = report.get("schema_version")
    required = {
        "schema_version",
        "artifact_type",
        "created_at_utc",
        "status",
        "claim_boundary",
        "model",
        "seed",
        "device",
        "data",
        "training",
        "outputs",
        "criteria",
    }
    if schema_version == REPORT_SCHEMA_VERSION:
        required.add("predeclaration")
    errors = [f"missing top-level key: {key}" for key in sorted(required - set(report))]
    if schema_version not in {REPORT_SCHEMA_VERSION, "targeted-retraining-replay-v1"}:
        errors.append("schema_version mismatch")
    if report.get("artifact_type") != "targeted_learned_estimator_retraining_replay":
        errors.append("artifact_type mismatch")
    criteria = report.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        errors.append("criteria must be a non-empty mapping")
    elif not all(isinstance(v, bool) for v in criteria.values()):
        errors.append("all criteria values must be boolean")
    status = report.get("status")
    if not isinstance(status, Mapping) or not isinstance(status.get("pass"), bool):
        errors.append("status.pass must be boolean")
    return errors


def write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    criteria = report["criteria"]
    outputs = report["outputs"]
    training = report["training"]
    stages = training.get("stages", [])
    stage_lines = []
    for stage in stages:
        stage_lines.append(
            f"- {stage['stage']}: epochs={stage['epochs_completed']}, "
            f"final_train_loss={stage['final_train_loss']}, final_val_loss={stage['final_val_loss']}"
        )
    if not stage_lines:
        stage_lines.append("- No completed stages were recorded.")

    criterion_lines = [f"- {name}: {value}" for name, value in criteria.items()]
    checkpoint_lines = [
        f"- {item['path']}: {item['sha256']}" for item in outputs.get("checkpoints", [])
    ] or ["- None"]
    text = "\n".join(
        [
            "# Targeted Retraining Replay",
            "",
            f"Status: {'PASS' if report['status']['pass'] else 'FAIL'}",
            "",
            f"Model: {report['model']}",
            f"Seed: {report['seed']}",
            f"Training exit code: {training.get('returncode')}",
            "",
            "Claim boundary: " + str(report["claim_boundary"]),
            "",
            "## Compute Requirement",
            "",
            (
                "The learned-estimator replay required accelerated execution; "
                "runtime, hardware, and software-version details are redacted."
            ),
            "",
            "## Stage Histories",
            "",
            *stage_lines,
            "",
            "## Checkpoints",
            "",
            *checkpoint_lines,
            "",
            "## Criteria",
            "",
            *criterion_lines,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    started_at = utc_now_iso()
    perf_start = time.perf_counter()
    base_config_path = Path(args.base_config)
    source_data_dir = Path(args.source_data_dir)
    output_root = Path(args.output_root)
    validation_dir = Path(args.validation_dir)
    data_dir = output_root / "data_slices"
    artifacts_dir = output_root / "artifacts"
    log_path = output_root / "train_models_stdout.log"
    config_path = output_root / "targeted_retraining_config.yaml"
    predeclared_rule_path = Path(args.predeclared_rule)

    output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(base_config_path)
    base_config_text = base_config_path.read_text(encoding="utf-8")
    selected_device, device_info = resolve_replay_device(args.device)
    predeclared_rule, predeclaration_record = load_predeclared_rule(
        predeclared_rule_path
    )
    stages_plan = curriculum_stage_plan(args)
    split_counts = split_count_plan(args, source_data_dir)
    validate_against_predeclaration(
        rule=predeclared_rule,
        args=args,
        stages=stages_plan,
        split_counts=split_counts,
    )

    canonical_before = canonical_checkpoint_digest()
    data_records = {
        split: slice_npz(
            source_data_dir / f"{split}.npz",
            data_dir / f"{split}.npz",
            count,
        )
        for split, count in split_counts.items()
    }
    replay_cfg = build_replay_config(
        base_cfg=base_cfg,
        base_config_text=base_config_text,
        model_name=args.model,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        gru_layers=args.gru_layers,
        device=selected_device,
        curriculum_stages=stages_plan,
        predeclared_rule_sha256=predeclaration_record["sha256"],
    )
    config_record = write_replay_config(replay_cfg, config_path)

    command = [
        sys.executable,
        "scripts/train_models.py",
        "--config",
        str(config_path),
        "--models",
        args.model,
        "--device",
        selected_device,
    ]
    train_started = utc_now_iso()
    train_perf = time.perf_counter()
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    train_duration = float(max(time.perf_counter() - train_perf, 0.0))
    canonical_after = canonical_checkpoint_digest()

    history_path = artifacts_dir / "checkpoints" / "train_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    stages = flatten_stage_histories(history, args.model)
    checkpoint_paths = [
        Path(stage["checkpoint"]) for stage in stages if stage.get("checkpoint") and Path(stage["checkpoint"]).exists()
    ]
    manifest_paths = sorted((artifacts_dir / "manifests").glob("*.json"))
    env_report_path = artifacts_dir / "runtime" / "env_report.json"
    criteria = build_criteria(
        returncode=proc.returncode,
        stages=stages,
        checkpoint_paths=checkpoint_paths,
        config_record=config_record,
        output_root=output_root,
        canonical_before=canonical_before,
        canonical_after=canonical_after,
        device_info=device_info,
        predeclaration_record=predeclaration_record,
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "targeted_learned_estimator_retraining_replay",
        "created_at_utc": utc_now_iso(),
        "status": {
            "pass": bool(criteria) and all(criteria.values()),
            "failed_criteria": [name for name, value in criteria.items() if not value],
        },
        "claim_boundary": (
            "This artifact demonstrates one bounded from-scratch learned-estimator "
            "training replay on deterministic materialized splits under the "
            "predeclared replay rule. It does not reproduce the full paper "
            "tables, seed suites, or main-result claims."
        ),
        "predeclaration": predeclaration_record,
        "model": args.model,
        "seed": int(args.seed),
        "device": device_info,
        "data": {
            "source_data_dir": rel_report_path(source_data_dir),
            "slices": public_data_records(data_records, output_root),
            "full_materialized_curriculum": bool(args.full_materialized_curriculum),
            "full_split_counts": bool(args.full_split_counts),
        },
        "training": {
            "source_script": "scripts/train_models.py",
            "execution": public_execution_record(config_path, args.model, selected_device),
            "evaluation_started_at_utc": train_started,
            "evaluation_completed_at_utc": utc_now_iso(),
            "duration_redacted": True,
            "returncode": int(proc.returncode),
            "log_sha256": sha256_file(log_path),
            "stages": public_stage_histories(stages, output_root),
        },
        "outputs": {
            "output_root": PUBLIC_REPLAY_OUTPUT_ROOT.as_posix(),
            "config": {
                "path_redacted": True,
                "redaction_reason": (
                    "Raw executable replay config records acceleration and "
                    "local runtime details; retained by digest only."
                ),
                "sha256": config_record.get("sha256"),
            },
            "history": {
                "path": public_replay_path(history_path, output_root),
                "sha256": sha256_file(history_path) if history_path.exists() else None,
            },
            "env_report": {
                "sha256": sha256_file(env_report_path) if env_report_path.exists() else None,
                "path_redacted": True,
                "runtime_details_redacted": True,
            },
            "manifests": [
                {"artifact": path.name, "sha256": sha256_file(path)} for path in manifest_paths
            ],
            "checkpoints": [
                {
                    "path": public_replay_path(path, output_root),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in checkpoint_paths
            ],
            "train_log": {
                "sha256": sha256_file(log_path),
                "path_redacted": True,
                "redaction_reason": (
                    "Raw training log includes local runtime details; "
                    "retained by digest only."
                ),
            },
            "canonical_checkpoint_digest_before": canonical_before,
            "canonical_checkpoint_digest_after": canonical_after,
        },
        "criteria": criteria,
        "total_driver_duration_redacted": True,
        "driver_started_at_utc": started_at,
    }
    schema_errors = validate_report_schema(report)
    if schema_errors:
        report["status"]["pass"] = False
        report["status"]["schema_errors"] = schema_errors

    validation_dir.mkdir(parents=True, exist_ok=True)
    json_path = validation_dir / "targeted_retraining_replay.json"
    md_path = validation_dir / "targeted_retraining_replay.md"
    dump_json(report, json_path)
    write_markdown_report(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("PASS" if report["status"]["pass"] else "FAIL")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
