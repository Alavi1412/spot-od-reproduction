"""Offline tests for the expanded compact real SLR/SP3 OD campaign wrapper."""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

import scripts.run_real_slr_sp3_od_expanded_validation as runner
from scripts.run_real_slr_sp3_od_expanded_validation import (
    EXTENDED_OUTPUT_FILENAME,
    EXTENDED_SCHEDULE,
    FORMAL210_EXTRA_WEEKS,
    FORMAL210_OUTPUT_FILENAME,
    FORMAL210_SCHEDULE,
    FORMAL400_EXTRA_WEEKS,
    FORMAL400_OUTPUT_FILENAME,
    FORMAL400_SCHEDULE,
    FORMAL_POWER_ADDED_WEEKS,
    FORMAL_POWER_OUTPUT_FILENAME,
    FORMAL_POWER_SCHEDULE,
    OUTPUT_FILENAME,
    PRECEDING_WEEKS,
    SPLIT_WEEKS,
    build_expanded_arcs,
    build_parser,
    build_result,
    output_json_path,
    paired_difference_summary,
    pooled_held_out_rmse,
    schedule_metadata,
    split_by_arc_id,
)


def _completed_arc(
    arc_id: str,
    *,
    ekf: float,
    ukf: float,
    aukf: float,
    sp3_ic: float,
    best: str,
    dbar_fired: bool = False,
    counterproductive: bool = False,
) -> dict:
    rms = {
        "EKF": ekf,
        "UKF (fixed-noise)": ukf,
        "AUKF (adaptive)": aukf,
        "SP3-IC propagation": sp3_ic,
    }
    return {
        "arc_id": arc_id,
        "target": arc_id.split()[0],
        "status": "completed",
        "held_out_position_rmse_m": rms,
        "held_out_detail": {
            label: {"rms_m": value, "count": 3}
            for label, value in rms.items()
        },
        "best_held_out_estimator": best,
        "dbar": {
            "external_outcome_available": True,
            "dbar_fired": dbar_fired,
            "adaptation_counterproductive_external": counterproductive,
            "dbar_correct_external": dbar_fired == counterproductive,
        },
    }


def test_expanded_arcs_match_hifi_split_schedule() -> None:
    arcs = build_expanded_arcs()
    assert len(arcs) == 40
    assert Counter(a.target for a in arcs) == {
        "LAGEOS-1": 20,
        "LAGEOS-2": 20,
    }
    assert Counter(a.sp3_week for a in arcs) == {
        "260418": 10,
        "260425": 10,
        "260502": 10,
        "260509": 10,
    }

    lookup = split_by_arc_id()
    for week, (split, dates) in SPLIT_WEEKS.items():
        for date in dates:
            for target in ("LAGEOS-1", "LAGEOS-2"):
                key = f"{target} {date}"
                assert lookup[key] == split
                assert any(
                    a.target == target and a.date == date and a.sp3_week == week
                    for a in arcs
                )

    meta = schedule_metadata()
    assert meta["num_weeks"] == 4
    assert "schedule_name" not in meta
    assert "added_weeks" not in meta


def test_extended80_schedule_has_expected_week_date_and_target_counts() -> None:
    arcs = build_expanded_arcs(EXTENDED_SCHEDULE)
    assert len(arcs) == 80
    assert Counter(a.target for a in arcs) == {
        "LAGEOS-1": 40,
        "LAGEOS-2": 40,
    }
    assert Counter(a.sp3_week for a in arcs) == {
        "260321": 10,
        "260328": 10,
        "260404": 10,
        "260411": 10,
        "260418": 10,
        "260425": 10,
        "260502": 10,
        "260509": 10,
    }
    assert Counter(a.date for a in arcs) == {
        "20260316": 2,
        "20260317": 2,
        "20260318": 2,
        "20260319": 2,
        "20260320": 2,
        "20260323": 2,
        "20260324": 2,
        "20260325": 2,
        "20260326": 2,
        "20260327": 2,
        "20260330": 2,
        "20260331": 2,
        "20260401": 2,
        "20260402": 2,
        "20260403": 2,
        "20260406": 2,
        "20260407": 2,
        "20260408": 2,
        "20260409": 2,
        "20260410": 2,
        "20260413": 2,
        "20260414": 2,
        "20260415": 2,
        "20260416": 2,
        "20260417": 2,
        "20260420": 2,
        "20260421": 2,
        "20260422": 2,
        "20260423": 2,
        "20260424": 2,
        "20260427": 2,
        "20260428": 2,
        "20260429": 2,
        "20260430": 2,
        "20260501": 2,
        "20260504": 2,
        "20260505": 2,
        "20260506": 2,
        "20260507": 2,
        "20260508": 2,
    }

    meta = schedule_metadata(EXTENDED_SCHEDULE)
    assert meta["schedule_name"] == EXTENDED_SCHEDULE
    assert meta["num_weeks"] == 8
    assert meta["added_weeks"] == ["260321", "260328", "260404", "260411"]
    assert meta["weeks"]["260321"]["dates"] == [
        "20260316",
        "20260317",
        "20260318",
        "20260319",
        "20260320",
    ]


def test_formal190_schedule_adds_11_weeks_before_extended80() -> None:
    arcs = build_expanded_arcs(FORMAL_POWER_SCHEDULE)
    assert len(arcs) == 190
    assert Counter(a.target for a in arcs) == {
        "LAGEOS-1": 95,
        "LAGEOS-2": 95,
    }
    assert all(
        count == 10 for count in Counter(a.sp3_week for a in arcs).values()
    )

    expected_week_order = (
        list(FORMAL_POWER_ADDED_WEEKS)
        + list(PRECEDING_WEEKS)
        + list(SPLIT_WEEKS)
    )
    meta = schedule_metadata(FORMAL_POWER_SCHEDULE)
    assert list(meta["weeks"]) == expected_week_order
    assert meta["schedule_name"] == FORMAL_POWER_SCHEDULE
    assert meta["base_schedule_name"] == EXTENDED_SCHEDULE
    assert meta["num_weeks"] == 19
    assert meta["added_weeks"] == list(FORMAL_POWER_ADDED_WEEKS)
    assert meta["num_added_weeks"] == 11
    assert meta["formal_power_added_weeks"] == list(FORMAL_POWER_ADDED_WEEKS)
    assert meta["num_formal_power_added_weeks"] == 11
    assert meta["extended80_weeks"] == list(PRECEDING_WEEKS) + list(SPLIT_WEEKS)
    assert meta["num_extended80_weeks"] == 8
    assert meta["weeks"]["260103"]["dates"] == [
        "20251229",
        "20251230",
        "20251231",
        "20260101",
        "20260102",
    ]
    assert "adding 11 earlier weeks before the extended80 schedule" in meta[
        "description"
    ]


def test_formal210_schedule_extends_formal190_with_two_earlier_weeks() -> None:
    arcs = build_expanded_arcs(FORMAL210_SCHEDULE)
    assert len(arcs) == 210
    assert Counter(a.target for a in arcs) == {
        "LAGEOS-1": 105,
        "LAGEOS-2": 105,
    }
    assert all(
        count == 10 for count in Counter(a.sp3_week for a in arcs).values()
    )

    expected_week_order = (
        list(FORMAL210_EXTRA_WEEKS)
        + list(FORMAL_POWER_ADDED_WEEKS)
        + list(PRECEDING_WEEKS)
        + list(SPLIT_WEEKS)
    )
    meta = schedule_metadata(FORMAL210_SCHEDULE)
    assert list(meta["weeks"]) == expected_week_order
    assert meta["schedule_name"] == FORMAL210_SCHEDULE
    assert meta["base_schedule_name"] == EXTENDED_SCHEDULE
    assert meta["num_weeks"] == 21
    assert meta["added_weeks"] == list(FORMAL210_EXTRA_WEEKS) + list(
        FORMAL_POWER_ADDED_WEEKS
    )
    assert meta["num_added_weeks"] == 13
    assert meta["formal210_extra_weeks"] == list(FORMAL210_EXTRA_WEEKS)
    assert meta["num_formal210_extra_weeks"] == 2
    assert meta["formal210_added_weeks"] == list(FORMAL210_EXTRA_WEEKS) + list(
        FORMAL_POWER_ADDED_WEEKS
    )
    assert meta["num_formal210_added_weeks"] == 13
    assert meta["formal_power_added_weeks"] == list(FORMAL_POWER_ADDED_WEEKS)
    assert meta["num_formal_power_added_weeks"] == 11
    assert meta["extended80_weeks"] == list(PRECEDING_WEEKS) + list(SPLIT_WEEKS)
    assert meta["num_extended80_weeks"] == 8
    assert meta["weeks"]["251220"]["dates"] == [
        "20251215",
        "20251216",
        "20251217",
        "20251218",
        "20251219",
    ]
    assert meta["weeks"]["251227"]["dates"] == [
        "20251222",
        "20251223",
        "20251224",
        "20251225",
        "20251226",
    ]
    assert "adding 13 earlier weeks before the extended80 schedule" in meta[
        "description"
    ]


def test_formal400_schedule_extends_formal210_with_19_earlier_weeks() -> None:
    arcs = build_expanded_arcs(FORMAL400_SCHEDULE)
    assert len(arcs) == 400
    assert Counter(a.target for a in arcs) == {
        "LAGEOS-1": 200,
        "LAGEOS-2": 200,
    }
    assert all(
        count == 10 for count in Counter(a.sp3_week for a in arcs).values()
    )

    expected_week_order = (
        list(FORMAL400_EXTRA_WEEKS)
        + list(FORMAL210_EXTRA_WEEKS)
        + list(FORMAL_POWER_ADDED_WEEKS)
        + list(PRECEDING_WEEKS)
        + list(SPLIT_WEEKS)
    )
    meta = schedule_metadata(FORMAL400_SCHEDULE)
    assert list(meta["weeks"]) == expected_week_order
    assert meta["schedule_name"] == FORMAL400_SCHEDULE
    assert meta["base_schedule_name"] == EXTENDED_SCHEDULE
    assert meta["num_weeks"] == 40
    assert meta["added_weeks"] == (
        list(FORMAL400_EXTRA_WEEKS)
        + list(FORMAL210_EXTRA_WEEKS)
        + list(FORMAL_POWER_ADDED_WEEKS)
    )
    assert meta["num_added_weeks"] == 32
    assert meta["formal400_extra_weeks"] == list(FORMAL400_EXTRA_WEEKS)
    assert meta["num_formal400_extra_weeks"] == 19
    assert meta["formal400_added_weeks"] == (
        list(FORMAL400_EXTRA_WEEKS)
        + list(FORMAL210_EXTRA_WEEKS)
        + list(FORMAL_POWER_ADDED_WEEKS)
    )
    assert meta["num_formal400_added_weeks"] == 32
    assert meta["formal210_extra_weeks"] == list(FORMAL210_EXTRA_WEEKS)
    assert meta["num_formal210_extra_weeks"] == 2
    assert meta["formal_power_added_weeks"] == list(FORMAL_POWER_ADDED_WEEKS)
    assert meta["num_formal_power_added_weeks"] == 11
    assert meta["extended80_weeks"] == list(PRECEDING_WEEKS) + list(SPLIT_WEEKS)
    assert meta["num_extended80_weeks"] == 8
    assert meta["weeks"]["250809"]["dates"] == [
        "20250804",
        "20250805",
        "20250806",
        "20250807",
        "20250808",
    ]
    assert meta["weeks"]["251213"]["dates"] == [
        "20251208",
        "20251209",
        "20251210",
        "20251211",
        "20251212",
    ]
    assert "adding 32 earlier weeks before the extended80 schedule" in meta[
        "description"
    ]


def test_dry_run_schedule_arc_counts(monkeypatch, capsys) -> None:
    expected_counts = {
        "hifi40": 40,
        EXTENDED_SCHEDULE: 80,
        FORMAL_POWER_SCHEDULE: 190,
        FORMAL210_SCHEDULE: 210,
        FORMAL400_SCHEDULE: 400,
    }
    for schedule_name, expected_count in expected_counts.items():
        argv = ["run_real_slr_sp3_od_expanded_validation.py", "--dry-run"]
        if schedule_name != "hifi40":
            argv.extend(["--schedule", schedule_name])
        monkeypatch.setattr(sys, "argv", argv)

        code = runner.main()
        printed = json.loads(capsys.readouterr().out)

        assert code == 0
        assert printed["status"] == "dry_run"
        assert printed["num_arcs"] == expected_count
        assert len(printed["arcs"]) == expected_count
        assert printed["schedule"]["num_weeks"] == expected_count // 10


def test_schedule_default_outputs_do_not_overwrite_each_other(tmp_path) -> None:
    parser = build_parser()
    default_args = parser.parse_args([])
    extended_args = parser.parse_args(["--schedule", EXTENDED_SCHEDULE])
    formal_args = parser.parse_args(["--schedule", FORMAL_POWER_SCHEDULE])
    formal210_args = parser.parse_args(["--schedule", FORMAL210_SCHEDULE])
    formal400_args = parser.parse_args(["--schedule", FORMAL400_SCHEDULE])
    alias_args = parser.parse_args(["--include-preceding-weeks"])
    custom_path = tmp_path / "custom_extended.json"
    custom_args = parser.parse_args(
        ["--schedule", EXTENDED_SCHEDULE, "--output-json", str(custom_path)]
    )

    assert output_json_path(default_args).name == OUTPUT_FILENAME
    assert output_json_path(extended_args).name == EXTENDED_OUTPUT_FILENAME
    assert output_json_path(formal_args).name == FORMAL_POWER_OUTPUT_FILENAME
    assert output_json_path(formal210_args).name == FORMAL210_OUTPUT_FILENAME
    assert output_json_path(formal400_args).name == FORMAL400_OUTPUT_FILENAME
    assert output_json_path(alias_args).name == EXTENDED_OUTPUT_FILENAME
    assert output_json_path(default_args) != output_json_path(extended_args)
    assert output_json_path(default_args) != output_json_path(formal_args)
    assert output_json_path(default_args) != output_json_path(formal210_args)
    assert output_json_path(default_args) != output_json_path(formal400_args)
    assert output_json_path(extended_args) != output_json_path(formal_args)
    assert output_json_path(extended_args) != output_json_path(formal210_args)
    assert output_json_path(extended_args) != output_json_path(formal400_args)
    assert output_json_path(formal_args) != output_json_path(formal210_args)
    assert output_json_path(formal_args) != output_json_path(formal400_args)
    assert output_json_path(formal210_args) != output_json_path(formal400_args)
    assert output_json_path(custom_args) == custom_path


def test_schedule_parser_choices_include_all_campaigns() -> None:
    parser = build_parser()
    schedule_action = next(
        action for action in parser._actions if action.dest == "schedule"
    )
    assert schedule_action.choices == (
        "hifi40",
        EXTENDED_SCHEDULE,
        FORMAL_POWER_SCHEDULE,
        FORMAL210_SCHEDULE,
        FORMAL400_SCHEDULE,
    )


def test_pooled_rmse_includes_std_best_counts_and_completed_count() -> None:
    completed = [
        _completed_arc(
            "LAGEOS-1 20260413",
            ekf=10.0,
            ukf=8.0,
            aukf=7.0,
            sp3_ic=12.0,
            best="AUKF (adaptive)",
        ),
        _completed_arc(
            "LAGEOS-2 20260413",
            ekf=5.0,
            ukf=6.0,
            aukf=4.0,
            sp3_ic=10.0,
            best="AUKF (adaptive)",
        ),
    ]

    pooled = pooled_held_out_rmse(completed)
    ekf = pooled["EKF"]
    assert ekf["completed_arc_count"] == 2
    assert ekf["n_arcs"] == 2
    assert ekf["mean_arc_rms_m"] == 7.5
    assert ekf["median_arc_rms_m"] == 7.5
    assert ekf["std_arc_rms_m"] == round(float(np.std([10.0, 5.0], ddof=1)), 2)
    assert pooled["AUKF (adaptive)"]["arc_best_count"] == 2


def test_paired_difference_summary_sign_convention_and_bootstrap_metadata() -> None:
    completed = [
        _completed_arc(
            "LAGEOS-1 20260413",
            ekf=10.0,
            ukf=8.0,
            aukf=7.0,
            sp3_ic=12.0,
            best="AUKF (adaptive)",
        ),
        _completed_arc(
            "LAGEOS-2 20260413",
            ekf=5.0,
            ukf=6.0,
            aukf=4.0,
            sp3_ic=10.0,
            best="AUKF (adaptive)",
        ),
    ]

    out = paired_difference_summary(
        completed,
        "EKF",
        "AUKF (adaptive)",
        seed=123,
        resamples=200,
    )
    assert out["n"] == 2
    assert out["mean_difference_m"] == 2.0
    assert out["median_difference_m"] == 2.0
    assert out["n_first_larger_rmse"] == 2
    assert out["seed"] == 123
    assert out["resamples"] == 200
    assert len(out["bootstrap95_mean_difference_m"]) == 2
    assert "positive means the first method has larger held-out RMSE" in out[
        "sign_convention"
    ]


def test_build_result_schema_caveats_and_serialized_arcs() -> None:
    arcs = [
        _completed_arc(
            "LAGEOS-1 20260413",
            ekf=10.0,
            ukf=8.0,
            aukf=7.0,
            sp3_ic=12.0,
            best="AUKF (adaptive)",
        ),
        _completed_arc(
            "LAGEOS-2 20260413",
            ekf=5.0,
            ukf=6.0,
            aukf=4.0,
            sp3_ic=10.0,
            best="AUKF (adaptive)",
            dbar_fired=True,
            counterproductive=True,
        ),
    ]

    result = build_result(arcs, bootstrap_seed=123, bootstrap_resamples=200)
    assert result["schema_version"] == "real_slr_sp3_od_expanded_v1"
    assert result["status"] == "completed"
    assert result["num_arcs_completed"] == 2
    assert result["schedule"]["num_weeks"] == 4
    assert "held_out_detail" not in result["arcs"][0]
    assert "EKF minus AUKF (adaptive)" in result["paired_differences"]
    assert result["dbar_external_validation"]["n_arcs_scored"] == 2

    caveats = result["caveat_metadata"]
    assert caveats["public_bounded_reference_replay"] is True
    assert caveats["compact_gmst_only_frame"] is True
    assert caveats["full_operational_slr_correction_stack"] is False
    assert caveats["operational_pod"] is False


def test_main_all_failed_campaign_returns_nonzero_without_writing(
    monkeypatch, tmp_path, capsys
) -> None:
    out_path = tmp_path / "all_failed.json"

    def fake_build_expanded_arcs(schedule_name: str = EXTENDED_SCHEDULE):
        return [
            runner.Arc("LAGEOS-1", "lageos1", "L51", "20260316", "260321"),
            runner.Arc("LAGEOS-2", "lageos2", "L52", "20260316", "260321"),
        ]

    def fake_run_arc(arc, out_dir, args):
        return {
            "arc_id": f"{arc.target} {arc.date}",
            "target": arc.target,
            "date": arc.date,
            "sp3_week": arc.sp3_week,
            "status": "insufficient_observations",
            "num_observations": 0,
        }

    monkeypatch.setattr(runner, "build_expanded_arcs", fake_build_expanded_arcs)
    monkeypatch.setattr(runner, "run_arc", fake_run_arc)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_real_slr_sp3_od_expanded_validation.py",
            "--schedule",
            EXTENDED_SCHEDULE,
            "--out-dir",
            str(tmp_path),
            "--output-json",
            str(out_path),
        ],
    )

    code = runner.main()
    printed = json.loads(capsys.readouterr().out)

    assert code == 2
    assert out_path.exists() is False
    assert printed["status"] == "insufficient_observations"
    assert printed["output_written"] is False
    assert printed["num_arcs"] == 2
    assert printed["num_arcs_completed"] == 0
    assert printed["arc_status_counts"] == {"insufficient_observations": 2}
    assert printed["dbar_external_validation"]["n_arcs_scored"] == 0
    assert printed["dbar_external_validation"]["classification_accuracy"] is None


def test_expanded_table_sp3_ic_note_uses_json_best_count(tmp_path) -> None:
    from scripts.build_paper_assets import build_real_slr_sp3_od_expanded_table

    held_out = {
        "EKF": 10.0,
        "UKF (fixed-noise)": 9.0,
        "AUKF (adaptive)": 8.0,
        "SP3-IC propagation": 7.0,
    }
    payload = {
        "schema_version": "real_slr_sp3_od_expanded_v1",
        "status": "completed",
        "targets": ["LAGEOS-1", "LAGEOS-2"],
        "num_arcs_completed": 5,
        "sp3_week_products": ["260000"],
        "split_weeks": {"260000": "test"},
        "pooled_held_out_position_rmse_m": {
            "EKF": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 10.0,
                "median_arc_rms_m": 10.0,
                "arc_best_count": 1,
            },
            "UKF (fixed-noise)": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 9.0,
                "median_arc_rms_m": 9.0,
                "arc_best_count": 1,
            },
            "AUKF (adaptive)": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 8.0,
                "median_arc_rms_m": 8.0,
                "arc_best_count": 0,
            },
            "SP3-IC propagation": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 7.0,
                "median_arc_rms_m": 7.0,
                "arc_best_count": 3,
            },
        },
        "arcs": [
            {
                "status": "completed",
                "split": "test",
                "held_out_position_rmse_m": held_out,
            }
            for _ in range(5)
        ],
        "paired_differences": {
            "EKF minus AUKF (adaptive)": {
                "n": 5,
                "mean_difference_m": 2.0,
                "median_difference_m": 2.0,
                "bootstrap95_mean_difference_m": [1.0, 3.0],
                "n_first_larger_rmse": 5,
            },
            "UKF (fixed-noise) minus AUKF (adaptive)": {
                "n": 5,
                "mean_difference_m": 1.0,
                "median_difference_m": 1.0,
                "bootstrap95_mean_difference_m": [0.0, 2.0],
                "n_first_larger_rmse": 5,
            },
            "EKF minus UKF (fixed-noise)": {
                "n": 5,
                "mean_difference_m": 1.0,
                "median_difference_m": 1.0,
                "bootstrap95_mean_difference_m": [0.0, 2.0],
                "n_first_larger_rmse": 5,
            },
        },
        "dbar_external_validation": {
            "n_arcs_scored": 5,
            "n_correct": 4,
            "classification_accuracy": 0.8,
            "no_information_baseline": {
                "majority_class_accuracy": 0.6,
                "beats_majority": True,
            },
            "classification_report": {"accuracy_ci": [0.38, 0.96]},
            "confusion": {
                "true_fire": 1,
                "true_no_fire": 3,
                "false_fire": 1,
                "false_no_fire": 0,
            },
            "sensitivity": 1.0,
            "specificity": 0.75,
        },
    }
    path = tmp_path / "real_slr_sp3_od_expanded_validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = build_real_slr_sp3_od_expanded_table(path)

    assert "its 3/5 best-of count" in table
    assert "its 20/40 best-of count" not in table
    assert (
        "Test split means are descriptive small-sample summaries, "
        "not stable ordering evidence"
    ) in table
    assert "The intervals all span zero" not in table
    assert "The listed paired-difference intervals do not all span zero" in table


def test_expanded_table_accepts_partial_completed_with_pooled_summaries(tmp_path) -> None:
    from scripts.build_paper_assets import build_real_slr_sp3_od_expanded_table

    held_out = {
        "EKF": 10.0,
        "UKF (fixed-noise)": 9.0,
        "AUKF (adaptive)": 8.0,
        "SP3-IC propagation": 7.0,
    }
    payload = {
        "schema_version": "real_slr_sp3_od_expanded_v1",
        "status": "partial_completed",
        "targets": ["LAGEOS-1", "LAGEOS-2"],
        "num_arcs": 6,
        "num_arcs_completed": 5,
        "sp3_week_products": [
            "251220",
            "251227",
            "260103",
            "260110",
            "260117",
            "260124",
            "260131",
            "260207",
            "260214",
        ],
        "split_weeks": {
            "251220": "preceding",
            "251227": "preceding",
            "260103": "preceding",
            "260110": "preceding",
            "260117": "preceding",
            "260124": "preceding",
            "260131": "train",
            "260207": "val",
            "260214": "test",
        },
        "schedule": {"schedule_name": "formal210"},
        "pooled_held_out_position_rmse_m": {
            "EKF": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 10.0,
                "median_arc_rms_m": 10.0,
                "arc_best_count": 1,
            },
            "UKF (fixed-noise)": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 9.0,
                "median_arc_rms_m": 9.0,
                "arc_best_count": 1,
            },
            "AUKF (adaptive)": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 8.0,
                "median_arc_rms_m": 8.0,
                "arc_best_count": 0,
            },
            "SP3-IC propagation": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 7.0,
                "median_arc_rms_m": 7.0,
                "arc_best_count": 3,
            },
        },
        "arcs": [
            {
                "status": "completed",
                "split": "test",
                "held_out_position_rmse_m": held_out,
            }
            for _ in range(5)
        ]
        + [{"status": "insufficient_observations", "split": "test"}],
        "paired_differences": {
            "EKF minus AUKF (adaptive)": {
                "n": 5,
                "mean_difference_m": 2.0,
                "median_difference_m": 2.0,
                "bootstrap95_mean_difference_m": [1.0, 3.0],
                "n_first_larger_rmse": 5,
            },
            "UKF (fixed-noise) minus AUKF (adaptive)": {
                "n": 5,
                "mean_difference_m": 1.0,
                "median_difference_m": 1.0,
                "bootstrap95_mean_difference_m": [0.0, 2.0],
                "n_first_larger_rmse": 5,
            },
            "EKF minus UKF (fixed-noise)": {
                "n": 5,
                "mean_difference_m": 1.0,
                "median_difference_m": 1.0,
                "bootstrap95_mean_difference_m": [0.0, 2.0],
                "n_first_larger_rmse": 5,
            },
        },
        "dbar_external_validation": {
            "n_arcs_scored": 5,
            "n_correct": 4,
            "classification_accuracy": 0.8,
            "no_information_baseline": {
                "majority_class_accuracy": 0.6,
                "beats_majority": True,
            },
            "classification_report": {"accuracy_ci": [0.38, 0.96]},
            "confusion": {
                "true_fire": 1,
                "true_no_fire": 3,
                "false_fire": 1,
                "false_no_fire": 0,
            },
            "sensitivity": 1.0,
            "specificity": 0.75,
        },
    }
    path = tmp_path / "real_slr_sp3_od_formal210_validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = build_real_slr_sp3_od_expanded_table(path)

    assert "5 completed archived public ILRS" in table
    assert "from 6 attempted arcs" in table
    assert "9 weekly products (251220, 251227, ..., 260207, 260214)" in table
    assert "1/6 attempted arcs were excluded (1 insufficient observations)" in table
    assert "not hidden, imputed, or pooled" in table


def test_expanded_table_handles_mixed_exclusion_statuses(tmp_path) -> None:
    from scripts.build_paper_assets import build_real_slr_sp3_od_expanded_table

    held_out = {
        "EKF": 10.0,
        "UKF (fixed-noise)": 9.0,
        "AUKF (adaptive)": 8.0,
        "SP3-IC propagation": 7.0,
    }
    payload = {
        "schema_version": "real_slr_sp3_od_expanded_v1",
        "status": "partial_completed",
        "targets": ["LAGEOS-1", "LAGEOS-2"],
        "num_arcs": 10,
        "num_arcs_completed": 5,
        "sp3_week_products": ["260509"],
        "split_weeks": {"260509": "test"},
        "schedule": {"schedule_name": "formal400"},
        "pooled_held_out_position_rmse_m": {
            "EKF": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 10.0,
                "median_arc_rms_m": 10.0,
                "arc_best_count": 1,
            },
            "UKF (fixed-noise)": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 9.0,
                "median_arc_rms_m": 9.0,
                "arc_best_count": 1,
            },
            "AUKF (adaptive)": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 8.0,
                "median_arc_rms_m": 8.0,
                "arc_best_count": 0,
            },
            "SP3-IC propagation": {
                "completed_arc_count": 5,
                "n_arcs": 5,
                "mean_arc_rms_m": 7.0,
                "median_arc_rms_m": 7.0,
                "arc_best_count": 3,
            },
        },
        "arcs": [
            {
                "status": "completed",
                "split": "test",
                "held_out_position_rmse_m": held_out,
            }
            for _ in range(5)
        ]
        + [{"status": "insufficient_observations", "split": "test"} for _ in range(3)]
        + [{"status": "arc_failed", "split": "test"} for _ in range(2)],
        "paired_differences": {
            "EKF minus AUKF (adaptive)": {
                "n": 5,
                "mean_difference_m": 2.0,
                "median_difference_m": 2.0,
                "bootstrap95_mean_difference_m": [1.0, 3.0],
                "n_first_larger_rmse": 5,
            },
            "UKF (fixed-noise) minus AUKF (adaptive)": {
                "n": 5,
                "mean_difference_m": 1.0,
                "median_difference_m": 1.0,
                "bootstrap95_mean_difference_m": [0.0, 2.0],
                "n_first_larger_rmse": 5,
            },
            "EKF minus UKF (fixed-noise)": {
                "n": 5,
                "mean_difference_m": 1.0,
                "median_difference_m": 1.0,
                "bootstrap95_mean_difference_m": [0.0, 2.0],
                "n_first_larger_rmse": 5,
            },
        },
        "dbar_external_validation": {
            "n_arcs_scored": 5,
            "n_correct": 4,
            "classification_accuracy": 0.8,
            "no_information_baseline": {
                "majority_class_accuracy": 0.6,
                "beats_majority": True,
            },
            "classification_report": {"accuracy_ci": [0.38, 0.96]},
            "confusion": {
                "true_fire": 1,
                "true_no_fire": 3,
                "false_fire": 1,
                "false_no_fire": 0,
            },
            "sensitivity": 1.0,
            "specificity": 0.75,
        },
    }
    path = tmp_path / "real_slr_sp3_od_formal400_validation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = build_real_slr_sp3_od_expanded_table(path)

    assert "5 completed archived public ILRS" in table
    assert "from 10 attempted arcs" in table
    assert "5/10 attempted arcs were excluded (3 insufficient observations; 2 public product unavailable or non-parseable)" in table
    assert "not hidden, imputed, or pooled" in table


def test_expanded_table_defaults_to_formal400_artifact_when_available() -> None:
    from scripts.build_paper_assets import build_real_slr_sp3_od_expanded_table

    table = build_real_slr_sp3_od_expanded_table()

    assert "Formal-power-scale 400-attempt/373-completed" in table
    assert "supersedes the 210-arc replay" in table
    assert "EKF & 566.60 & 351.25" in table
    assert "UKF (fixed-noise) & 573.24 & 357.82" in table
    assert "AUKF (adaptive) & 601.38 & 365.34" in table
    assert "SP3-IC propagation & 779.45 & 542.79" in table
    assert "EKF--AUKF: mean -34.78" in table
    assert "fixed-noise UKF--AUKF: mean -28.14" in table
    assert "The 373 completed arcs clear the earlier approximate 185" in table
    assert "with paired intervals reported below" in table
    assert (
        "SP3-IC-best arcs have lower median observation and held-out counts "
        "than non-SP3-IC-best arcs (51 vs 59.5 observations; 21 vs 24 held-out points)"
    ) in table
    assert (
        "0--5: n=2, SP3-IC 2; 6--10: n=39, SP3-IC 22; "
        "EKF 10, UKF 6, AUKF 1; 11--20: n=110, SP3-IC 46; "
        "EKF 30, AUKF 24, UKF 10; 21+: n=222, SP3-IC 75; "
        "AUKF 53, EKF 53, UKF 41"
    ) in table
    assert "Because the pooled mean remains worst (SP3-IC 779.45~m vs EKF 566.60" in table
    assert "27/400 attempted arcs were excluded" in table
    assert "(17 insufficient observations; 10 public product unavailable or non-parseable)" in table
    assert "not hidden, imputed, or pooled" in table
    assert "test split $n=10$" in table
    assert (
        "Test split means are descriptive small-sample summaries, "
        "not stable ordering evidence"
    ) in table
    assert "DBAR is correct on 256/373 arcs" in table
    assert (
        "Sensitivity rests on 98 positive arcs; its interval remains wide, "
        "so no positive-class inferential claim is made"
    ) in table
    assert "underpowered for a positive-class claim" not in table


def test_formal400_stratification_table_is_completed_only_from_artifact() -> None:
    from scripts.build_paper_assets import (
        build_real_slr_sp3_od_expanded_stratification_table,
    )

    table = build_real_slr_sp3_od_expanded_stratification_table()

    assert "400-attempt/373-completed formal compact" in table
    assert "Preceding split" in table
    assert "Training split" in table
    assert "Validation split" in table
    assert "Test split" in table
    assert "LAGEOS-1" in table
    assert "LAGEOS-2" in table
    assert "Excluded count: 27/400 attempted arcs" in table
    assert "(17 insufficient observations; 10 public product unavailable or non-parseable)" in table
    assert "not hidden, imputed, or pooled" in table
    assert "does not identify a causal driver" in table


def test_formal400_mechanism_heterogeneity_table_from_artifact() -> None:
    from scripts.build_paper_assets import (
        build_real_slr_sp3_od_expanded_mechanism_heterogeneity_table,
    )

    table = build_real_slr_sp3_od_expanded_mechanism_heterogeneity_table()

    assert "373-completed-arc formal compact" in table
    assert "tab:real_slr_sp3_od_expanded_mechanism_heterogeneity" in table
    assert "Q1=37, median=57, Q3=84" in table
    # Check for key strata
    assert "Best: EKF" in table
    assert "Best: fixed-noise UKF" in table
    assert "Best: AUKF" in table
    assert "Best: SP3-IC propagation" in table
    assert "1 station" in table
    assert "2--3 stations" in table
    assert "4+ stations" in table
    # Check for quartile strata labels
    assert "Obs $\\leq$ 37" in table
    assert "Obs 37--57" in table
    assert "Obs 57--84" in table
    assert "Obs $>$ 84" in table
    # Check conservative caveat phrases
    assert "diagnostic strata only" in table
    assert "do not establish causal drivers" in table
    assert "stable filter superiority" in table
    assert "operational POD" in table
    assert "real-data estimator-skill validation" in table
    # Check column headers
    assert "Med.~obs" in table
    assert "Med.~held-out" in table
    assert "Med.~stn" in table
    assert "Med.~$R_{\\mathrm{eff}}$" in table
    assert "DBAR fire" in table
    assert "Med.~AUKF--UKF" in table
    assert "DBAR ext.~correct" in table


def test_mechanism_heterogeneity_table_column_spec_matches_header() -> None:
    from scripts.build_paper_assets import (
        build_real_slr_sp3_od_expanded_mechanism_heterogeneity_table,
    )

    table = build_real_slr_sp3_od_expanded_mechanism_heterogeneity_table()

    # The table has 9 columns: Stratum, n, Med obs, Med held-out, Med stn,
    # Med R_eff, DBAR fire %, Med AUKF--UKF %, DBAR ext correct %.
    # The tabular spec must have exactly 9 column alignment specifiers.
    assert "\\begin{tabular}{lcccccccc}" in table, (
        "Column spec mismatch: table has 9 columns "
        "(Stratum, n, Med obs, Med held-out, Med stn, Med R_eff, "
        "DBAR fire %, Med AUKF--UKF %, DBAR ext correct %) "
        "but tabular spec does not match"
    )
