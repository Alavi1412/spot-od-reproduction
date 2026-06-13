from __future__ import annotations

import json

from scripts.build_paper_assets import (
    build_force_mismatch_mechanism_table,
    build_main_findings_summary_table,
    build_main_aukf_mechanism_table,
    build_main_framework_portability_table,
    build_main_k32_replication_table,
    build_main_long_arc_result_table,
)


def _mechanism_summary() -> dict:
    return {
        "trajectories_processed": 48,
        "dynamics_provenance": {
            "ballistic_coeff_m2_per_kg": {"estimator": 0.018, "truth": 0.045},
            "process_noise_std": {"estimator": 0.0, "truth": 0.45},
            "drag_rho_ref": {"estimator": 4e-11, "truth": 7e-11},
            "srp_area_to_mass_m2_per_kg": {"estimator": 0.02, "truth": 0.06},
            "srp_cr": {"estimator": 1.35, "truth": 1.60},
        },
        "aukf_config": {"nis_soft_gate": 16.0},
        "aukf_adaptation_mechanism": {
            "n_visible_updates": 1238,
            "mean_pre_adapt_nis": 4.90,
            "median_pre_adapt_nis": 3.49,
            "p90_pre_adapt_nis": 9.16,
            "percent_updates_exceeding_soft_gate": 2.50,
            "mean_robust_scale": 1.039,
            "mean_r_scale_pre": 2.891,
            "mean_r_proposal_scale": 4.513,
            "mean_r_scale_post": 3.021,
            "mean_r_eff_scale": 3.315,
            "mean_state_update_pos_norm_m": 466.31,
        },
        "cross_filter_r_only_nis": {
            "EKF": {"median_r_only_nis": 1.60},
            "UKF": {"median_r_only_nis": 1.83},
            "AUKF": {"median_r_only_nis": 4.73, "p90_r_only_nis": 20.38},
        },
        "observed_step_pos_rmse": {
            "EKF": {"observed_step_pos_rmse_m": 448.79},
            "UKF": {"observed_step_pos_rmse_m": 469.41},
            "AUKF": {"observed_step_pos_rmse_m": 526.39},
        },
        "aukf_reconstruction": {"max_abs_pos_diff_vs_cached_aukf_m": 0.0},
    }


def test_aukf_mechanism_caption_names_population(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_mechanism_summary()))

    table = build_main_aukf_mechanism_table(summary)

    assert "full controlled force-mismatch mechanism population" in table
    assert "48 processed trajectories" in table
    assert "visible AUKF update records" in table


def test_supplementary_mechanism_caption_names_population(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(_mechanism_summary()))

    table = build_force_mismatch_mechanism_table(summary)

    assert "full controlled force-mismatch mechanism population" in table
    assert "48 processed trajectories" in table
    assert "visible AUKF update records" in table


def test_main_k32_caption_formalizes_best_classical_rule(tmp_path) -> None:
    payload = {
        "frozen_rule": {
            "num_realizations_per_scenario": 32,
            "trajectories_per_realization": 24,
        },
        "scenarios": [
            {
                "name": "test",
                "observed_step_pos_rmse_m": {"EKF": 402.0, "RGR-GF": 407.4},
                "best_classical_primary": "EKF",
                "rgr_gf_minus_best_classical_primary_mean_m": 5.4,
                "rgr_gf_minus_best_classical_primary_ci_low_m": -15.7,
                "rgr_gf_minus_best_classical_primary_ci_high_m": 31.3,
            },
            {
                "name": "stress_test",
                "observed_step_pos_rmse_m": {"AUKF": 915.8, "RGR-GF": 964.6},
                "best_classical_primary": "AUKF",
                "rgr_gf_minus_best_classical_primary_mean_m": 48.8,
                "rgr_gf_minus_best_classical_primary_ci_low_m": 17.7,
                "rgr_gf_minus_best_classical_primary_ci_high_m": 88.8,
            },
            {
                "name": "force_model_mismatch_test",
                "observed_step_pos_rmse_m": {"EKF": 474.3, "RGR-GF": 491.1},
                "best_classical_primary": "EKF",
                "rgr_gf_minus_best_classical_primary_mean_m": 16.8,
                "rgr_gf_minus_best_classical_primary_ci_low_m": 6.1,
                "rgr_gf_minus_best_classical_primary_ci_high_m": 27.8,
            },
        ],
    }
    path = tmp_path / "observed_step_prospective_replication.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = build_main_k32_replication_table(path)

    assert (
        "lowest held-out mean observed-step RMSE among tuned EKF, fixed-noise "
        "UKF, and AUKF under the frozen endpoint hierarchy"
    ) in table
    assert "realized labels are Nominal=EKF, Stress=AUKF, Mismatch=EKF" in table
    assert r"\emph{Nominal (traceability only)}\textsuperscript{\dag}" in table
    assert (
        "The daggered nominal row is traceability-only; its CI direction must "
        "not be interpreted"
    ) in table


def test_main_long_arc_note_names_visible_step_independence_bound(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    payload = {
        "n_trajectories": 64,
        "observed_step_rmse_mean_m": {
            "EKF": 577.05,
            "UKF": 345.05,
            "AUKF": 312.66,
            "PUKF": 405.98,
            "DMC_EKF": 577.05,
            "DSA_EKF": 597.09,
        },
        "decision": {
            "best_non_dsa_estimator": "AUKF",
            "practical_significance_floor_m_absolute": 105.96,
            "dsa_minus_best_non_dsa_mean_m": 284.43,
            "dsa_minus_best_non_dsa_ci_lo_m": 8.72,
            "dsa_minus_best_non_dsa_ci_hi_m": 697.94,
        },
    }
    path = tmp_path / "long_arc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = build_main_long_arc_result_table(path)

    assert "treats visible steps as independent" in table
    assert "tighten the floor under within-pass correlation" in table
    assert (
        "pass-correlated S-Table row F is the opposite-direction bound at "
        "335.1 m"
    ) in table
    assert "Decision unchanged because DSA-EKF fails direction" in table


def test_main_framework_portability_uses_confidential_review_package_boundary() -> None:
    table = build_main_framework_portability_table()

    assert "Supports confidential journal-submission inspection" in table
    assert "no DOI or public identifier is claimed at initial submission" in table
    assert "public citable archival deposition" not in table
    assert "deferred public-archive route" not in table
    assert "Zenodo" not in table


def test_main_findings_summary_table_uses_materialized_evidence(tmp_path) -> None:
    force_summary = tmp_path / "force_summary.json"
    force_summary.write_text(
        json.dumps(
            {
                "aukf_adaptation_mechanism": {"mean_r_eff_scale": 4.567},
                "cross_filter_r_only_nis": {
                    "EKF": {"median_r_only_nis": 2.11},
                    "UKF": {"median_r_only_nis": 3.22},
                    "AUKF": {"median_r_only_nis": 4.33},
                },
            }
        ),
        encoding="utf-8",
    )
    force_significance = tmp_path / "force_significance.json"
    force_significance.write_text(
        json.dumps(
            {
                "classical_paired_rows": [
                    {"comparison": "EKF vs AUKF", "mean_paired_gain_m": 12.34}
                ]
            }
        ),
        encoding="utf-8",
    )
    observed_k32 = tmp_path / "observed_k32.json"
    observed_k32.write_text(
        json.dumps(
            {
                "summary": {
                    "num_realizations_per_scenario": 17,
                    "scenarios_with_learned_positive_under_frozen_rule": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    stress_k96 = tmp_path / "stress_k96.json"
    stress_k96.write_text(
        json.dumps(
            {
                "summary": {
                    "num_realizations_per_scenario": 96,
                    "scenarios_with_learned_positive_under_frozen_rule": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    all_k96 = tmp_path / "all_k96.json"
    all_k96.write_text(
        json.dumps(
            {
                "summary": {
                    "K": 96,
                    "scenarios_with_learned_positive_under_frozen_rule": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    dbar = tmp_path / "dbar.json"
    dbar.write_text(
        json.dumps(
            {
                "summary": {
                    "classification_accuracy": 0.912,
                    "no_information_baseline": {"majority_class_accuracy": 0.456},
                }
            }
        ),
        encoding="utf-8",
    )
    active_regen = tmp_path / "active_regen.json"
    active_regen.write_text(
        json.dumps(
            {
                "validation_results": {
                    "artifact_count": 12,
                    "pass_count": 11,
                    "mismatch_count": 1,
                    "documented_blocker_count": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    archive_repro = tmp_path / "archive_repro.json"
    archive_repro.write_text(
        json.dumps(
            {
                "checks": {
                    "archive_extracted_public_od_slice_rerun": {
                        "summary": {
                            "completed_arcs": 7,
                            "table_text_matched": True,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    table = build_main_findings_summary_table(
        force_summary_path=force_summary,
        force_significance_path=force_significance,
        observed_k32_path=observed_k32,
        stress_k96_path=stress_k96,
        all_scenario_k96_path=all_k96,
        dbar_path=dbar,
        active_regeneration_report_path=active_regen,
        archive_reproduction_report_path=archive_repro,
    )

    assert "Mean effective-$R$ scale 4.57" in table
    assert "median $R$-only NIS 4.33 (AUKF) vs 2.11 (EKF) and 3.22 (UKF)" in table
    assert r"EKF$-$AUKF paired mean $-12.3$~m" in table
    assert "$K=17$" in table
    assert r"DBAR characterization 91.2\% vs 45.6\% no-information baseline" in table
    assert "records 11/12 passes, 1 mismatches, and 2 blockers" in table
    assert "reruns 7 public CRD/SP3 arcs with exact submitted-table-text recovery" in table
