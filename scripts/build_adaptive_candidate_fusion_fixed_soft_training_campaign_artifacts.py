#!/usr/bin/env python
"""Build full-training fixed-soft campaign artifacts for AdaptiveCandidateFusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "adaptive_candidate_fusion_fixed_soft_training_campaigns.v1"
OUTPUT_DIR = Path("results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623")
SUMMARY_CSV_NAME = "adaptive_candidate_fusion_summary.csv"
RUN_CONFIG_NAME = "run_config_summary.json"
TRAIN_HISTORY_NAME = "train_history.json"
RESULT_DIR_RE = re.compile(r"seed(?P<seed>\d+)_split(?P<split>\d+)")
SCENARIO_ORDER = {
    "maneuver_shift_test": 0,
    "process_noise_shift_test": 1,
}
SCENARIO_LABELS = {
    "maneuver_shift_test": "maneuver",
    "process_noise_shift_test": "process",
}
CANDIDATE_COLUMNS = {
    "EKF": {
        "observed": "ekf_observed_step_pos_rmse_m",
        "all_step": "ekf_all_step_pos_rmse_m",
    },
    "UKF": {
        "observed": "ukf_observed_step_pos_rmse_m",
        "all_step": "ukf_all_step_pos_rmse_m",
    },
    "AUKF": {
        "observed": "aukf_observed_step_pos_rmse_m",
        "all_step": "aukf_all_step_pos_rmse_m",
    },
    "BatchWLS": {
        "observed": "batchwls_observed_step_pos_rmse_m",
        "all_step": "batchwls_all_step_pos_rmse_m",
    },
    "RFIS": {
        "observed": "rfis_observed_step_pos_rmse_m",
        "all_step": "rfis_all_step_pos_rmse_m",
    },
    "VA_RFIS": {
        "observed": "va_rfis_observed_step_pos_rmse_m",
        "all_step": "va_rfis_all_step_pos_rmse_m",
    },
}
REQUIRED_COLUMNS = {
    "scenario",
    "requested_inference_mode",
    "selected_inference_mode",
    "inference_mode_selection_source",
    "adaptivecandidatefusion_observed_step_pos_rmse_m",
    "adaptivecandidatefusion_all_step_pos_rmse_m",
    *(columns["observed"] for columns in CANDIDATE_COLUMNS.values()),
    *(columns["all_step"] for columns in CANDIDATE_COLUMNS.values()),
}
BOUNDARY_LANGUAGE = (
    "Full-training AdaptiveCandidateFusion fixed-soft campaign packaging for local "
    "current-workspace compact-simulator evidence. These runs retrain from the "
    "materialized campaign inputs and are not skip-training replays, as checked by "
    "non-empty train/validation histories and checkpoint files. They are not part of "
    "the published public v1.2.1 package unless a later release includes them; not "
    "independent-machine reproduction; not operational precise-reference validation; "
    "not third-party validation; not a full raw/all-filter/public rerun; and not a "
    "universal learned-OD claim."
)


@dataclass(frozen=True)
class CampaignSpec:
    key: str
    label: str
    result_dirs: tuple[Path, ...]
    expected_training_step_mask: str
    expected_validation_selection_metric: str
    expected_seeds: tuple[int, ...]
    interpretation: str


CAMPAIGNS = (
    CampaignSpec(
        key="centered_fixed_soft_full_retraining",
        label="Centered training-step mask, fixed-soft full retraining",
        result_dirs=(
            Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed7_split7_20260623"),
            Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed11_split11_20260623"),
            Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed13_split13_20260623"),
            Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed17_split17_20260623"),
            Path("results/adaptive_candidate_fusion_centered_fixed_soft_seed19_split19_20260623"),
        ),
        expected_training_step_mask="centered",
        expected_validation_selection_metric="all_step_pos_rmse_m",
        expected_seeds=(7, 11, 13, 17, 19),
        interpretation=(
            "Positive compact-simulator PoC: centered training plus fixed-soft full retraining "
            "reproduces the 8/10 observed-step result."
        ),
    ),
    CampaignSpec(
        key="observed_mask_fixed_soft_full_retraining",
        label="Observed training-step mask, fixed-soft full retraining",
        result_dirs=(
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed23_split23_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed29_split29_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed31_split31_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed37_split37_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed41_split41_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed43_split43_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed47_split47_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed53_split53_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed59_split59_20260623"),
            Path("results/adaptive_candidate_fusion_observed_fixed_soft_seed61_split61_20260623"),
        ),
        expected_training_step_mask="observed",
        expected_validation_selection_metric="observed_step_pos_rmse_m",
        expected_seeds=(23, 29, 31, 37, 41, 43, 47, 53, 59, 61),
        interpretation=(
            "Bounded negative/failure mode: observed-mask full retraining is negative overall, "
            "with large maneuver failures despite fixed-soft inference."
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate AdaptiveCandidateFusion fixed-soft full-training campaign summaries."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser


def _display_path(path: Path, root: Path | None = None) -> str:
    root = Path.cwd() if root is None else Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_seed_split(result_dir: Path) -> tuple[int, int]:
    match = RESULT_DIR_RE.search(result_dir.name)
    if match is None:
        raise ValueError(f"{result_dir}: expected directory name containing seed<N>_split<N>")
    return int(match.group("seed")), int(match.group("split"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return data


def _finite_float(row: dict[str, str], column: str, *, source: Path) -> float:
    value = row.get(column, "")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {column} must be a finite number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{source}: {column} must be finite, got {value!r}")
    if number <= 0.0:
        raise ValueError(f"{source}: {column} must be positive, got {value!r}")
    return number


def _finite_history(values: Any, *, source: Path, key: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{source}: history.{key} must be a non-empty list")
    out: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{source}: history.{key} values must be finite numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: history.{key} values must be finite numbers") from exc
        if not math.isfinite(number):
            raise ValueError(f"{source}: history.{key} values must be finite")
        out.append(number)
    return out


def _gain_percent(reference: float, candidate: float) -> float:
    return (float(reference) - float(candidate)) / float(reference) * 100.0


def _result_from_gain(gain_percent: float) -> str:
    if math.isclose(gain_percent, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
        return "tie"
    return "win" if gain_percent > 0.0 else "loss"


def _row_sort_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    scenario = str(row["scenario"])
    return (
        str(row["campaign"]),
        int(row["seed"]),
        SCENARIO_ORDER.get(scenario, 99),
        scenario,
    )


def _validate_columns(fieldnames: list[str] | None, source: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"{source}: missing CSV header")
    missing = sorted(REQUIRED_COLUMNS.difference(fieldnames))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")


def _validate_fixed_soft_row(row: dict[str, str], source: Path) -> None:
    expected = {
        "requested_inference_mode": "soft",
        "selected_inference_mode": "soft",
        "inference_mode_selection_source": "cli_fixed",
    }
    for column, expected_value in expected.items():
        actual = str(row.get(column, "")).strip()
        if actual != expected_value:
            raise ValueError(
                f"{source}: expected fixed-soft row with {column}={expected_value!r}, got {actual!r}"
            )


def _validate_campaign_metadata(
    *,
    spec: CampaignSpec,
    result_dir: Path,
    seed: int,
    split: int,
) -> dict[str, Any]:
    config_path = result_dir / RUN_CONFIG_NAME
    history_path = result_dir / TRAIN_HISTORY_NAME
    config = _load_json(config_path)
    history_data = _load_json(history_path)

    expected_values = {
        "seed": seed,
        "training_step_mask": spec.expected_training_step_mask,
        "validation_selection_metric": spec.expected_validation_selection_metric,
        "requested_inference_mode": "soft",
        "selected_inference_mode": "soft",
    }
    for key, expected in expected_values.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(f"{config_path}: expected {key}={expected!r}, got {actual!r}")

    split_seed = config.get("scenario_trajectory_splits", {}).get("split_seed")
    if split_seed != split:
        raise ValueError(f"{config_path}: expected scenario_trajectory_splits.split_seed={split}, got {split_seed!r}")

    for key in ("training_step_mask", "validation_selection_metric", "requested_inference_mode", "selected_inference_mode"):
        actual = history_data.get(key)
        expected = expected_values[key]
        if actual != expected:
            raise ValueError(f"{history_path}: expected {key}={expected!r}, got {actual!r}")

    history = history_data.get("history")
    if not isinstance(history, dict):
        raise ValueError(f"{history_path}: missing history object")
    train_loss = _finite_history(history.get("train_loss"), source=history_path, key="train_loss")
    val_loss = _finite_history(history.get("val_loss"), source=history_path, key="val_loss")

    best_checkpoint_raw = history_data.get("best_checkpoint")
    if not isinstance(best_checkpoint_raw, str) or not best_checkpoint_raw:
        raise ValueError(f"{history_path}: best_checkpoint must be a non-empty string")
    best_checkpoint = Path(best_checkpoint_raw)
    if not best_checkpoint.exists():
        best_checkpoint = result_dir / "checkpoints" / Path(best_checkpoint_raw).name
    if not best_checkpoint.exists() or best_checkpoint.stat().st_size <= 0:
        raise ValueError(f"{history_path}: best checkpoint is missing or empty: {best_checkpoint_raw}")

    checkpoint_dir = result_dir / "checkpoints"
    last_checkpoint = checkpoint_dir / "last_adaptive_candidate_fusion.pt"
    if not last_checkpoint.exists() or last_checkpoint.stat().st_size <= 0:
        raise ValueError(f"{result_dir}: missing non-empty last checkpoint {last_checkpoint}")

    return {
        "run_config": _display_path(config_path),
        "train_history": _display_path(history_path),
        "best_checkpoint": _display_path(best_checkpoint),
        "last_checkpoint": _display_path(last_checkpoint),
        "epochs_recorded": len(train_loss),
        "validation_epochs_recorded": len(val_loss),
        "training_step_mask": spec.expected_training_step_mask,
        "validation_selection_metric": spec.expected_validation_selection_metric,
        "full_training_evidence": "non-empty train/val loss histories plus best/last checkpoint files",
    }


def _best_candidate(
    row: dict[str, str],
    *,
    metric: str,
    source: Path,
) -> tuple[str, float, dict[str, float]]:
    values = {
        method: _finite_float(row, columns[metric], source=source)
        for method, columns in CANDIDATE_COLUMNS.items()
    }
    best_method = min(values, key=values.__getitem__)
    return best_method, values[best_method], values


def read_campaign_dir(spec: CampaignSpec, result_dir: Path, *, root: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_dir = Path(result_dir)
    summary_path = result_dir / SUMMARY_CSV_NAME
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    seed, split = _parse_seed_split(result_dir)
    if seed not in spec.expected_seeds:
        raise ValueError(f"{result_dir}: seed {seed} is not expected for {spec.key}")
    if split != seed:
        raise ValueError(f"{result_dir}: expected split {seed}, got {split}")
    metadata = _validate_campaign_metadata(spec=spec, result_dir=result_dir, seed=seed, split=split)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames, summary_path)
        csv_rows = list(reader)
    if not csv_rows:
        raise ValueError(f"{summary_path}: expected at least one scenario row")

    rows: list[dict[str, Any]] = []
    for row in csv_rows:
        _validate_fixed_soft_row(row, summary_path)
        scenario = str(row["scenario"]).strip()
        if not scenario:
            raise ValueError(f"{summary_path}: scenario must be non-empty")
        best_obs_method, best_obs, observed_candidates = _best_candidate(row, metric="observed", source=summary_path)
        best_all_method, best_all, all_step_candidates = _best_candidate(row, metric="all_step", source=summary_path)
        observed_value = _finite_float(row, "adaptivecandidatefusion_observed_step_pos_rmse_m", source=summary_path)
        all_step_value = _finite_float(row, "adaptivecandidatefusion_all_step_pos_rmse_m", source=summary_path)
        observed_gain = _gain_percent(best_obs, observed_value)
        all_step_gain = _gain_percent(best_all, all_step_value)
        rows.append(
            {
                "campaign": spec.key,
                "campaign_label": spec.label,
                "training_step_mask": spec.expected_training_step_mask,
                "validation_selection_metric": spec.expected_validation_selection_metric,
                "seed": seed,
                "split": split,
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                "source_dir": _display_path(result_dir, root=root),
                "source_csv": _display_path(summary_path, root=root),
                "requested_inference_mode": "soft",
                "selected_inference_mode": "soft",
                "inference_mode_selection_source": "cli_fixed",
                "best_input_candidate_method": best_obs_method,
                "best_input_candidate_observed_step_pos_rmse_m": best_obs,
                "fixed_soft_observed_step_pos_rmse_m": observed_value,
                "fixed_soft_gain_vs_best_input_observed_step_percent": observed_gain,
                "fixed_soft_result": _result_from_gain(observed_gain),
                "best_input_candidate_all_step_method": best_all_method,
                "best_input_candidate_all_step_pos_rmse_m": best_all,
                "fixed_soft_all_step_pos_rmse_m": all_step_value,
                "fixed_soft_gain_vs_best_input_all_step_percent": all_step_gain,
                "fixed_soft_all_step_result": _result_from_gain(all_step_gain),
                "candidate_observed_step_pos_rmse_m": observed_candidates,
                "candidate_all_step_pos_rmse_m": all_step_candidates,
            }
        )
    return sorted(rows, key=_row_sort_key), metadata


def _summarize_metric(rows: list[dict[str, Any]], *, metric: str, gain_key: str, result_key: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty campaign")
    gains = [float(row[gain_key]) for row in rows]
    wins = sum(1 for row in rows if row[result_key] == "win")
    ties = sum(1 for row in rows if row[result_key] == "tie")
    losses = sum(1 for row in rows if row[result_key] == "loss")
    paired_seed_rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        seed_gains = [float(row[gain_key]) for row in seed_rows]
        paired_seed_rows.append(
            {
                "seed": seed,
                "rows": len(seed_rows),
                "scenario_wins": sum(1 for row in seed_rows if row[result_key] == "win"),
                "both_scenarios_win": (
                    len(seed_rows) >= len(SCENARIO_ORDER)
                    and all(str(row[result_key]) == "win" for row in seed_rows)
                ),
                "mean_gain_percent": sum(seed_gains) / len(seed_gains),
                "min_gain_percent": min(seed_gains),
                "max_gain_percent": max(seed_gains),
            }
        )

    scenarios: dict[str, dict[str, Any]] = {}
    for scenario in sorted({str(row["scenario"]) for row in rows}, key=lambda item: (SCENARIO_ORDER.get(item, 99), item)):
        scenario_rows = [row for row in rows if str(row["scenario"]) == scenario]
        scenario_gains = [float(row[gain_key]) for row in scenario_rows]
        scenarios[scenario] = {
            "rows": len(scenario_rows),
            "wins": sum(1 for row in scenario_rows if row[result_key] == "win"),
            "ties": sum(1 for row in scenario_rows if row[result_key] == "tie"),
            "losses": sum(1 for row in scenario_rows if row[result_key] == "loss"),
            "mean_gain_percent": sum(scenario_gains) / len(scenario_gains),
            "min_gain_percent": min(scenario_gains),
            "max_gain_percent": max(scenario_gains),
        }

    return {
        "metric": metric,
        "rows": len(rows),
        "row_wins": wins,
        "row_ties": ties,
        "row_losses": losses,
        "paired_seed_count": len(paired_seed_rows),
        "paired_seed_both_scenario_wins": sum(1 for row in paired_seed_rows if bool(row["both_scenarios_win"])),
        "mean_gain_percent": sum(gains) / len(gains),
        "min_gain_percent": min(gains),
        "max_gain_percent": max(gains),
        "scenario_wins": scenarios,
        "paired_seed_rows": paired_seed_rows,
    }


def build_campaign_artifacts(campaigns: tuple[CampaignSpec, ...] = CAMPAIGNS, *, root: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    campaign_summaries: dict[str, Any] = {}
    for spec in campaigns:
        campaign_rows: list[dict[str, Any]] = []
        run_metadata: dict[str, Any] = {}
        seen_seeds: set[int] = set()
        for result_dir in spec.result_dirs:
            loaded_rows, metadata = read_campaign_dir(spec, result_dir, root=root)
            campaign_rows.extend(loaded_rows)
            seed = int(loaded_rows[0]["seed"])
            seen_seeds.add(seed)
            run_metadata[str(seed)] = metadata
        expected_seeds = set(spec.expected_seeds)
        if seen_seeds != expected_seeds:
            raise ValueError(f"{spec.key}: expected seeds {sorted(expected_seeds)}, got {sorted(seen_seeds)}")
        rows.extend(campaign_rows)
        campaign_summaries[spec.key] = {
            "label": spec.label,
            "interpretation": spec.interpretation,
            "training_step_mask": spec.expected_training_step_mask,
            "validation_selection_metric": spec.expected_validation_selection_metric,
            "expected_seeds": list(spec.expected_seeds),
            "result_dirs": [_display_path(path, root=root) for path in spec.result_dirs],
            "run_metadata": run_metadata,
            "observed_step": _summarize_metric(
                campaign_rows,
                metric="observed_step_pos_rmse_m",
                gain_key="fixed_soft_gain_vs_best_input_observed_step_percent",
                result_key="fixed_soft_result",
            ),
            "all_step_caveat": _summarize_metric(
                campaign_rows,
                metric="all_step_pos_rmse_m",
                gain_key="fixed_soft_gain_vs_best_input_all_step_percent",
                result_key="fixed_soft_all_step_result",
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "boundary_language": BOUNDARY_LANGUAGE,
        "fixed_soft_validation": {
            "requested_inference_mode": "soft",
            "selected_inference_mode": "soft",
            "inference_mode_selection_source": "cli_fixed",
        },
        "campaigns": campaign_summaries,
        "rows": sorted(rows, key=_row_sort_key),
    }


CSV_COLUMNS = [
    "campaign",
    "training_step_mask",
    "validation_selection_metric",
    "seed",
    "split",
    "scenario",
    "scenario_label",
    "requested_inference_mode",
    "selected_inference_mode",
    "inference_mode_selection_source",
    "best_input_candidate_method",
    "best_input_candidate_observed_step_pos_rmse_m",
    "fixed_soft_observed_step_pos_rmse_m",
    "fixed_soft_gain_vs_best_input_observed_step_percent",
    "fixed_soft_result",
    "best_input_candidate_all_step_method",
    "best_input_candidate_all_step_pos_rmse_m",
    "fixed_soft_all_step_pos_rmse_m",
    "fixed_soft_gain_vs_best_input_all_step_percent",
    "fixed_soft_all_step_result",
    "source_dir",
    "source_csv",
]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{float(value):+.6f}"


def render_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# AdaptiveCandidateFusion Fixed-Soft Full-Training Campaigns",
        "",
        f"Schema: `{data['schema_version']}`",
        "",
        "## Boundary",
        "",
        str(data["boundary_language"]),
        "",
        "## Campaign Summary",
        "",
    ]
    for key, campaign in data["campaigns"].items():
        observed = campaign["observed_step"]
        all_step = campaign["all_step_caveat"]
        lines.extend(
            [
                f"### {campaign['label']}",
                "",
                campaign["interpretation"],
                "",
                (
                    f"Observed-step: {observed['row_wins']}/{observed['rows']} row wins, "
                    f"{observed['paired_seed_both_scenario_wins']}/{observed['paired_seed_count']} paired seeds, "
                    f"mean gain {_fmt(observed['mean_gain_percent'])}%, "
                    f"min {_fmt(observed['min_gain_percent'])}%, "
                    f"max {_fmt(observed['max_gain_percent'])}%."
                ),
                "",
                (
                    f"All-step caveat: {all_step['row_wins']}/{all_step['rows']} row wins, "
                    f"{all_step['paired_seed_both_scenario_wins']}/{all_step['paired_seed_count']} paired seeds, "
                    f"mean gain {_fmt(all_step['mean_gain_percent'])}%, "
                    f"min {_fmt(all_step['min_gain_percent'])}%, "
                    f"max {_fmt(all_step['max_gain_percent'])}%."
                ),
                "",
                (
                    f"Training mask `{campaign['training_step_mask']}`; validation selection metric "
                    f"`{campaign['validation_selection_metric']}`."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Rows",
            "",
            "| Campaign | Seed | Scenario | Best Observed Input | Observed RMSE | Observed Gain % | Best All-Step Input | All-Step RMSE | All-Step Gain % |",
            "| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in data["rows"]:
        lines.append(
            "| "
            f"{row['campaign']} | {row['seed']} | `{row['scenario']}` | "
            f"`{row['best_input_candidate_method']}` | "
            f"{float(row['fixed_soft_observed_step_pos_rmse_m']):.6f} | "
            f"{_fmt(row['fixed_soft_gain_vs_best_input_observed_step_percent'])} | "
            f"`{row['best_input_candidate_all_step_method']}` | "
            f"{float(row['fixed_soft_all_step_pos_rmse_m']):.6f} | "
            f"{_fmt(row['fixed_soft_gain_vs_best_input_all_step_percent'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(data), encoding="utf-8")


def write_artifacts(data: dict[str, Any], *, output_dir: Path) -> dict[str, str]:
    rows_csv = Path(output_dir) / "adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv"
    summary_json = Path(output_dir) / "adaptive_candidate_fusion_fixed_soft_training_campaign_summary.json"
    summary_md = Path(output_dir) / "adaptive_candidate_fusion_fixed_soft_training_campaign_summary.md"
    write_csv(data["rows"], rows_csv)
    write_json(data, summary_json)
    write_markdown(data, summary_md)
    return {
        "rows_csv": rows_csv.as_posix(),
        "summary_json": summary_json.as_posix(),
        "summary_md": summary_md.as_posix(),
    }


def main() -> None:
    args = build_parser().parse_args()
    data = build_campaign_artifacts()
    paths = write_artifacts(data, output_dir=args.output_dir)
    centered = data["campaigns"]["centered_fixed_soft_full_retraining"]["observed_step"]
    observed = data["campaigns"]["observed_mask_fixed_soft_full_retraining"]["observed_step"]
    print(
        "Centered fixed-soft full retraining observed-step: "
        f"{centered['row_wins']}/{centered['rows']} row wins, "
        f"{centered['paired_seed_both_scenario_wins']}/{centered['paired_seed_count']} paired seeds, "
        f"mean gain {centered['mean_gain_percent']:+.6f}%."
    )
    print(
        "Observed-mask fixed-soft full retraining observed-step: "
        f"{observed['row_wins']}/{observed['rows']} row wins, "
        f"{observed['paired_seed_both_scenario_wins']}/{observed['paired_seed_count']} paired seeds, "
        f"mean gain {observed['mean_gain_percent']:+.6f}%."
    )
    print(f"Wrote {paths['rows_csv']}")
    print(f"Wrote {paths['summary_json']}")
    print(f"Wrote {paths['summary_md']}")


if __name__ == "__main__":
    main()
