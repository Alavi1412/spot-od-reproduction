from __future__ import annotations

import json
from pathlib import Path

from scripts.build_paper_assets import (
    build_adaptive_candidate_fusion_full_training_poc_table,
    build_force_mismatch_mechanism_table,
    build_main_drag_scale_cascade_table,
    build_main_findings_summary_table,
    build_main_aukf_mechanism_table,
    build_main_framework_portability_table,
    build_main_k32_replication_table,
    build_main_long_arc_result_table,
)


ROOT = Path(__file__).resolve().parents[1]


def _acf_metric(
    *,
    row_wins: int,
    rows: int,
    paired_wins: int,
    paired_count: int,
    mean: float,
    min_gain: float,
    max_gain: float,
) -> dict:
    return {
        "row_wins": row_wins,
        "rows": rows,
        "paired_seed_both_scenario_wins": paired_wins,
        "paired_seed_count": paired_count,
        "mean_gain_percent": mean,
        "min_gain_percent": min_gain,
        "max_gain_percent": max_gain,
    }


def _acf_global_summary() -> dict:
    return {
        "schema_version": "adaptive_candidate_fusion_global_portfolio.v1",
        "validation": {
            "policy_space": {
                "policy_count_per_candidate_set": 540,
            },
            "global_scenario_policies": {
                "process_noise_shift_test": {
                    "alpha": 0.65,
                    "components": ["learned", "RFIS"],
                    "selection_metric": "observed_step_pos_rmse_m",
                },
                "maneuver_shift_test": {
                    "alpha": 0.55,
                    "components": ["learned", "EKF"],
                    "selection_metric": "observed_step_pos_rmse_m",
                },
            },
        },
        "eval": {
            "global_scenario_policy_statistics": {
                "rows": {
                    "wins": 25,
                    "rows": 30,
                    "mean_gain_percent": 3.793410580996871,
                    "min_gain_percent": -10.11448514448152,
                    "max_gain_percent": 12.384536098768079,
                    "bootstrap_mean_gain_percent_ci95": [
                        1.8304662751308267,
                        5.630976806698702,
                    ],
                },
                "seed_paired": {
                    "seed_wins": 13,
                    "seeds": 15,
                    "bootstrap_seed_mean_gain_percent_ci95": [
                        2.050875926391989,
                        5.503273704982441,
                    ],
                },
                "by_scenario": {
                    "process_noise_shift_test": {
                        "wins": 14,
                        "rows": 15,
                        "mean_gain_percent": 6.155760540527878,
                        "bootstrap_mean_gain_percent_ci95": [
                            4.030627876665985,
                            8.059701125604633,
                        ],
                    },
                    "maneuver_shift_test": {
                        "wins": 11,
                        "rows": 15,
                        "mean_gain_percent": 1.4310606214658632,
                        "bootstrap_mean_gain_percent_ci95": [
                            -1.4627353881195744,
                            4.050083039292123,
                        ],
                    },
                },
            },
            "policy_family_diagnostics": {
                "nonlearned_only": {
                    "summary": {
                        "wins": 19,
                        "rows": 30,
                        "mean_gain_percent": 0.7140145823400381,
                    },
                    "statistics": {
                        "seed_paired": {
                            "seed_wins": 9,
                            "seeds": 15,
                            "bootstrap_seed_mean_gain_percent_ci95": [
                                -1.1610439763564315,
                                2.6310888765184925,
                            ],
                        },
                    },
                },
            },
        },
    }


def test_adaptive_candidate_fusion_table_carries_claim_boundary(tmp_path) -> None:
    summary = {
        "campaigns": {
            "centered_fixed_soft_full_retraining": {
                "observed_step": _acf_metric(
                    row_wins=1,
                    rows=4,
                    paired_wins=2,
                    paired_count=6,
                    mean=12.3456784,
                    min_gain=-98.7654321,
                    max_gain=45.0000012,
                ),
                "all_step_caveat": _acf_metric(
                    row_wins=3,
                    rows=4,
                    paired_wins=1,
                    paired_count=6,
                    mean=-7.1111114,
                    min_gain=-8.0,
                    max_gain=9.0,
                ),
            },
            "observed_mask_fixed_soft_full_retraining": {
                "observed_step": _acf_metric(
                    row_wins=5,
                    rows=9,
                    paired_wins=4,
                    paired_count=8,
                    mean=-1.2345674,
                    min_gain=-11.1111114,
                    max_gain=22.2222224,
                ),
                "all_step_caveat": _acf_metric(
                    row_wins=6,
                    rows=9,
                    paired_wins=2,
                    paired_count=8,
                    mean=3.3333334,
                    min_gain=-4.0,
                    max_gain=5.0,
                ),
            },
        },
    }
    summary_path = tmp_path / "acf_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    global_summary_path = tmp_path / "acf_global_summary.json"
    global_summary_path.write_text(json.dumps(_acf_global_summary()), encoding="utf-8")

    table = build_adaptive_candidate_fusion_full_training_poc_table(
        summary_path, global_summary_path
    )
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

    assert "\\input{tables/adaptive_candidate_fusion_full_training_poc.tex}" in main
    assert "\\label{tab:adaptive_candidate_fusion_full_training_poc}" in table
    assert "Centered fixed-soft full retraining" in table
    assert "1/4" in table
    assert "2/6" in table
    assert "+12.345678" in table
    assert "-98.765432 / +45.000001" in table
    assert "3/4 rows, 1/6 paired, mean -7.111111\\%" in table
    assert "Observed-mask fixed-soft full retraining" in table
    assert "5/9" in table
    assert "4/8" in table
    assert "-1.234567" in table
    assert "-11.111111 / +22.222222" in table
    assert "6/9 rows, 2/8 paired, mean +3.333333\\%" in table
    assert "global_scenario_portfolio_15seed" in table
    assert "540 candidate policies per candidate set" in table
    assert "25/30" in table
    assert "13/15" in table
    assert "+3.79" in table
    assert "-10.11 / +12.38" in table
    assert "all rows 25/30 wins, mean +3.79\\%, 95\\% CI [+1.83,+5.63]\\%" in table
    assert "seed-paired 13/15 wins, 95\\% CI [+2.05,+5.50]\\%" in table
    assert "process 14/15 wins, mean +6.16\\%, CI [+4.03,+8.06]\\%" in table
    assert "maneuver 11/15 wins, mean +1.43\\%, CI [-1.46,+4.05]\\%" in table
    assert "nonlearned-only validation-selected blend baseline" in table
    assert "19/30 wins, +0.71\\% mean" in table
    assert (
        "nonlearned-only 19/30 row wins, mean +0.71\\%, 9/15 seed-paired wins, "
        "seed-paired CI [-1.16,+2.63]\\%"
    ) in table
    assert "do not adjust for validation-policy search" in table
    assert "+2.594443" not in table
    assert "-6.191539" not in table
    assert "not public v1.2.1 release evidence" in table
    assert "not external validation" in table


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


def test_main_framework_portability_uses_public_archive_reproduction_boundary() -> None:
    table = build_main_framework_portability_table()

    assert "Compact claim--evidence map for this technical note" in table
    assert "Tier & Load-bearing status & Evidence" in table
    assert "1. Primary compact mechanism" in table
    assert "Load-bearing mechanism evidence" in table
    assert "Supported primary claim & The larger frozen-rule" not in table
    assert "AUKF effective-$R$ inflation under compact force-model mismatch" in table
    assert "Table~\\ref{tab:main_aukf_mechanism}" in table
    assert "Figure~\\ref{fig:aukf_r_inflation_mechanism}" in table
    assert "Table~\\ref{tab:main_drag_scale_cascade}" in table
    assert "2. Secondary learned-negative" in table
    assert "Internal frozen-rule evidence; not external preregistration" in table
    assert "$K=32$/$K=96$ internal replications" in table
    assert "3. Public CRD/SP3 probes" in table
    assert "Support/provenance only" in table
    assert "pending/unscored rules are separated from validation claims" in table
    assert "4. Reproduction and inspection" in table
    assert "Access/integrity tier satisfied; scientific rerun still bounded" in table
    assert "Public DOI/GitHub archive deposition" in table
    assert "manifests, digests, archive extraction" in table
    assert "active manuscript artifact regeneration" in table
    assert "one archived-input public OD slice rerun" in table
    assert "one bounded learned-estimator replay" in table
    assert "non-destructive full rerun" in table
    assert "divergence audit" in table
    assert "clean full scientific reproduction" in table
    assert "independent-machine reproduction" in table
    assert "full raw/training/all-filter public reproduction" in table
    assert "live public-data retrieval" in table
    assert "third-party independent validation" in table
    assert "replacement manuscript metrics" in table
    assert "5. Exclusions and future validation" in table
    assert "Operational POD, independent-machine reproduction" in table
    assert "full raw/training/all-filter reruns" in table
    assert "broader learned OD" in table
    assert "localized EnKF" in table
    assert "particle/Gaussian-mixture filters" in table
    assert "broader EnKF hyperparameter searches" in table
    assert ("DOI/public archive " + "release") not in table
    assert "public DOI/archive reproduction" not in table
    assert "public citable archival deposition" not in table
    assert "deferred public-archive route" not in table


def test_supplement_decision_record_hierarchy_is_learned_negative() -> None:
    import re

    supplement = (
        Path(__file__).resolve().parents[1] / "paper" / "supplement.tex"
    ).read_text(encoding="utf-8")

    assert "The current claim hierarchy is:" not in supplement
    assert not re.search(
        r"central[-\s]+(?:\$K=32\$[-\s]+)?anchor",
        supplement,
        flags=re.IGNORECASE,
    )
    assert (
        "The following list orders only the secondary learned-estimator negative "
        "decision records"
    ) in supplement
    assert (
        "the primary claim is the compact AUKF effective-$R$ inflation and "
        "drag-scale cascade mechanism"
    ) in supplement


def _drag_scale_row(best_key: str, n_traj: int, boot: int) -> dict:
    return {
        "n_trajectories": n_traj,
        "bootstrap_samples": boot,
        "observed_step_rmse_mean_m": {
            "DSA_EKF": 1648.5,
            "DSA_UKF": 316.4,
            best_key: 359.7,
        },
        "decision": {
            "best_non_dsa_estimator": best_key,
            "best_non_candidate_estimator": best_key,
            "dsa_minus_best_non_dsa_mean_m": 1288.9,
            "dsa_minus_best_non_dsa_ci_lo_m": 378.1,
            "dsa_minus_best_non_dsa_ci_hi_m": 2561.4,
            "dsa_ukf_minus_best_non_candidate_mean_m": 3.3,
            "dsa_ukf_minus_best_non_candidate_ci_lo_m": -7.5,
            "dsa_ukf_minus_best_non_candidate_ci_hi_m": 16.2,
        },
        "dsa_diagnostics": {
            "median_max_abs_beta_deviation": 0.0012,
            "median_final_beta": 1.0,
        },
        "dsa_ukf_diagnostics": {
            "median_max_abs_beta_deviation": 0.0009,
            "median_final_beta": 1.0,
        },
        "drag_scale_separation_diagnostic_m": {
            "median_mean_separation_m": 9595.9,
            "median_final_separation_m": 29294.1,
        },
    }


def test_main_drag_scale_cascade_footnote_disclosures_from_json(tmp_path) -> None:
    ekf = tmp_path / "ekf.json"
    ukf = tmp_path / "ukf.json"
    obs = tmp_path / "obs.json"
    # Use distinct, non-default values to prove the footnote is generated from
    # the JSON evidence rather than hard-coded.
    ekf.write_text(json.dumps(_drag_scale_row("UKF", 11, 1234)), encoding="utf-8")
    ukf.write_text(json.dumps(_drag_scale_row("AUKF", 22, 1234)), encoding="utf-8")
    obs.write_text(json.dumps(_drag_scale_row("EKF", 33, 1234)), encoding="utf-8")

    table = build_main_drag_scale_cascade_table(ekf, ukf, obs)

    assert "held-out trajectory populations" in table
    assert "$n=11$, $n=22$, $n=33$ respectively" in table
    assert (
        "statistical/decision unit is trajectory-level observed-step RMSE" in table
    )
    assert "paired trajectory-bootstrap 95\\% CIs using 1234 resamples" in table
    assert "frozen before held-out scoring" in table
    assert "selected from a predeclared geometry grid" in table
    assert "no validation grid point satisfied the positive predicate" in table
    assert "mechanism/stability diagnostic" in table
    assert "not a stable UKF-family ranking" in table


def test_main_drag_scale_cascade_fixture_footnote_uses_evidence_sample_sizes() -> None:
    table = build_main_drag_scale_cascade_table()
    if table.strip().startswith("%"):
        return  # fixture artifacts unavailable in this checkout
    assert "$n=64$, $n=64$, $n=32$ respectively" in table
    assert "5000 resamples" in table
    assert "trajectory-level observed-step RMSE" in table
    assert "predeclared geometry grid" in table
    assert "not a stable UKF-family ranking" in table


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
