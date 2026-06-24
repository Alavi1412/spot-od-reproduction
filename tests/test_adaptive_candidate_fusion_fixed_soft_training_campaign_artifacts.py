from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.build_adaptive_candidate_fusion_fixed_soft_training_campaign_artifacts as campaigns


FIELDNAMES = [
    "scenario",
    "requested_inference_mode",
    "selected_inference_mode",
    "inference_mode_selection_source",
    "adaptivecandidatefusion_observed_step_pos_rmse_m",
    "adaptivecandidatefusion_all_step_pos_rmse_m",
    "ekf_observed_step_pos_rmse_m",
    "ekf_all_step_pos_rmse_m",
    "ukf_observed_step_pos_rmse_m",
    "ukf_all_step_pos_rmse_m",
    "aukf_observed_step_pos_rmse_m",
    "aukf_all_step_pos_rmse_m",
    "batchwls_observed_step_pos_rmse_m",
    "batchwls_all_step_pos_rmse_m",
    "rfis_observed_step_pos_rmse_m",
    "rfis_all_step_pos_rmse_m",
    "va_rfis_observed_step_pos_rmse_m",
    "va_rfis_all_step_pos_rmse_m",
]


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_result_dir(
    tmp_path: Path,
    *,
    prefix: str,
    seed: int,
    training_step_mask: str,
    validation_selection_metric: str,
    rows: list[dict[str, object]],
    config_overrides: dict[str, object] | None = None,
    history_overrides: dict[str, object] | None = None,
    write_checkpoints: bool = True,
) -> Path:
    result_dir = tmp_path / f"{prefix}_seed{seed}_split{seed}_20260623"
    result_dir.mkdir()
    with (result_dir / "adaptive_candidate_fusion_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            base = {
                "requested_inference_mode": "soft",
                "selected_inference_mode": "soft",
                "inference_mode_selection_source": "cli_fixed",
                "ekf_observed_step_pos_rmse_m": 10.0,
                "ekf_all_step_pos_rmse_m": 100.0,
                "ukf_observed_step_pos_rmse_m": 12.0,
                "ukf_all_step_pos_rmse_m": 120.0,
                "aukf_observed_step_pos_rmse_m": 13.0,
                "aukf_all_step_pos_rmse_m": 130.0,
                "batchwls_observed_step_pos_rmse_m": 14.0,
                "batchwls_all_step_pos_rmse_m": 140.0,
                "rfis_observed_step_pos_rmse_m": 15.0,
                "rfis_all_step_pos_rmse_m": 150.0,
                "va_rfis_observed_step_pos_rmse_m": 16.0,
                "va_rfis_all_step_pos_rmse_m": 160.0,
            }
            base.update(row)
            writer.writerow(base)

    checkpoints = result_dir / "checkpoints"
    checkpoints.mkdir()
    best_checkpoint = checkpoints / "best_adaptive_candidate_fusion.pt"
    last_checkpoint = checkpoints / "last_adaptive_candidate_fusion.pt"
    if write_checkpoints:
        best_checkpoint.write_bytes(b"best")
        last_checkpoint.write_bytes(b"last")

    config = {
        "seed": seed,
        "training_step_mask": training_step_mask,
        "validation_selection_metric": validation_selection_metric,
        "requested_inference_mode": "soft",
        "selected_inference_mode": "soft",
        "scenario_trajectory_splits": {"split_seed": seed},
    }
    if config_overrides:
        config.update(config_overrides)
    _write_json(result_dir / "run_config_summary.json", config)

    history = {
        "best_checkpoint": str(best_checkpoint),
        "history": {"train_loss": [1.0, 0.9], "val_loss": [1.1, 1.0]},
        "training_step_mask": training_step_mask,
        "validation_selection_metric": validation_selection_metric,
        "requested_inference_mode": "soft",
        "selected_inference_mode": "soft",
    }
    if history_overrides:
        history.update(history_overrides)
    _write_json(result_dir / "train_history.json", history)
    return result_dir


def _spec(
    *,
    key: str,
    result_dir: Path,
    seed: int,
    training_step_mask: str,
    validation_selection_metric: str,
) -> campaigns.CampaignSpec:
    return campaigns.CampaignSpec(
        key=key,
        label=key,
        result_dirs=(result_dir,),
        expected_training_step_mask=training_step_mask,
        expected_validation_selection_metric=validation_selection_metric,
        expected_seeds=(seed,),
        interpretation="test campaign",
    )


def test_build_campaign_artifacts_summarizes_centered_and_observed_campaigns(tmp_path) -> None:
    centered_dir = _write_result_dir(
        tmp_path,
        prefix="centered_campaign",
        seed=7,
        training_step_mask="centered",
        validation_selection_metric="all_step_pos_rmse_m",
        rows=[
            {
                "scenario": "maneuver_shift_test",
                "adaptivecandidatefusion_observed_step_pos_rmse_m": 9.0,
                "adaptivecandidatefusion_all_step_pos_rmse_m": 90.0,
            },
            {
                "scenario": "process_noise_shift_test",
                "adaptivecandidatefusion_observed_step_pos_rmse_m": 11.0,
                "adaptivecandidatefusion_all_step_pos_rmse_m": 110.0,
            },
        ],
    )
    observed_dir = _write_result_dir(
        tmp_path,
        prefix="observed_campaign",
        seed=23,
        training_step_mask="observed",
        validation_selection_metric="observed_step_pos_rmse_m",
        rows=[
            {
                "scenario": "maneuver_shift_test",
                "adaptivecandidatefusion_observed_step_pos_rmse_m": 8.0,
                "adaptivecandidatefusion_all_step_pos_rmse_m": 80.0,
            },
            {
                "scenario": "process_noise_shift_test",
                "adaptivecandidatefusion_observed_step_pos_rmse_m": 17.0,
                "adaptivecandidatefusion_all_step_pos_rmse_m": 170.0,
            },
        ],
    )
    specs = (
        _spec(
            key="centered",
            result_dir=centered_dir,
            seed=7,
            training_step_mask="centered",
            validation_selection_metric="all_step_pos_rmse_m",
        ),
        _spec(
            key="observed",
            result_dir=observed_dir,
            seed=23,
            training_step_mask="observed",
            validation_selection_metric="observed_step_pos_rmse_m",
        ),
    )

    result = campaigns.build_campaign_artifacts(specs, root=tmp_path)

    assert result["campaigns"]["centered"]["observed_step"]["row_wins"] == 1
    assert result["campaigns"]["centered"]["observed_step"]["row_losses"] == 1
    assert result["campaigns"]["observed"]["observed_step"]["row_wins"] == 1
    assert result["campaigns"]["observed"]["observed_step"]["row_losses"] == 1
    assert result["campaigns"]["centered"]["run_metadata"]["7"]["epochs_recorded"] == 2
    assert "non-empty train/val loss histories" in result["campaigns"]["observed"]["run_metadata"]["23"]["full_training_evidence"]


def test_build_campaign_artifacts_rejects_non_fixed_soft_row(tmp_path) -> None:
    result_dir = _write_result_dir(
        tmp_path,
        prefix="centered_campaign",
        seed=7,
        training_step_mask="centered",
        validation_selection_metric="all_step_pos_rmse_m",
        rows=[
            {
                "scenario": "maneuver_shift_test",
                "requested_inference_mode": "auto",
                "adaptivecandidatefusion_observed_step_pos_rmse_m": 9.0,
                "adaptivecandidatefusion_all_step_pos_rmse_m": 90.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="requested_inference_mode"):
        campaigns.build_campaign_artifacts(
            (
                _spec(
                    key="centered",
                    result_dir=result_dir,
                    seed=7,
                    training_step_mask="centered",
                    validation_selection_metric="all_step_pos_rmse_m",
                ),
            ),
            root=tmp_path,
        )


def test_build_campaign_artifacts_rejects_campaign_config_mismatch(tmp_path) -> None:
    result_dir = _write_result_dir(
        tmp_path,
        prefix="observed_campaign",
        seed=23,
        training_step_mask="centered",
        validation_selection_metric="observed_step_pos_rmse_m",
        rows=[
            {
                "scenario": "maneuver_shift_test",
                "adaptivecandidatefusion_observed_step_pos_rmse_m": 9.0,
                "adaptivecandidatefusion_all_step_pos_rmse_m": 90.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="training_step_mask"):
        campaigns.build_campaign_artifacts(
            (
                _spec(
                    key="observed",
                    result_dir=result_dir,
                    seed=23,
                    training_step_mask="observed",
                    validation_selection_metric="observed_step_pos_rmse_m",
                ),
            ),
            root=tmp_path,
        )


@pytest.mark.parametrize(
    ("history_overrides", "write_checkpoints", "match"),
    [
        ({"history": {"train_loss": [], "val_loss": [1.0]}}, True, "train_loss"),
        ({}, False, "best checkpoint"),
    ],
)
def test_build_campaign_artifacts_requires_training_history_and_checkpoints(
    tmp_path,
    history_overrides: dict[str, object],
    write_checkpoints: bool,
    match: str,
) -> None:
    result_dir = _write_result_dir(
        tmp_path,
        prefix="centered_campaign",
        seed=7,
        training_step_mask="centered",
        validation_selection_metric="all_step_pos_rmse_m",
        rows=[
            {
                "scenario": "maneuver_shift_test",
                "adaptivecandidatefusion_observed_step_pos_rmse_m": 9.0,
                "adaptivecandidatefusion_all_step_pos_rmse_m": 90.0,
            }
        ],
        history_overrides=history_overrides,
        write_checkpoints=write_checkpoints,
    )

    with pytest.raises(ValueError, match=match):
        campaigns.build_campaign_artifacts(
            (
                _spec(
                    key="centered",
                    result_dir=result_dir,
                    seed=7,
                    training_step_mask="centered",
                    validation_selection_metric="all_step_pos_rmse_m",
                ),
            ),
            root=tmp_path,
        )
