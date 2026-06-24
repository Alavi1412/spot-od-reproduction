# Canonical Release Packet

This file identifies the evidence path that is canonical for the current paper state.

## Canonical Evidence
- Manuscript: `paper\main.tex`
- Supplement: `paper\supplement.tex`
- Config: `configs/experiment.yaml`
- Metrics: `results/metrics_summary.json`
- Scorecard: `results/scorecard_summary.json`
- Evaluation manifest: `results\manifests\evaluation.json`
- Runtime provenance: machine-specific execution details are retained only in raw run records and are not summarized here.
- Public tracking manifest: `configs/public_tracking_manifest.json`
- Formal review: `paper/FORMAL_PEER_REVIEW_2026-04-11.md`
- Classical tuning summary: `results/classical_tuning/classical_tuning_summary.json`
- Classical tuning ledger: `results/classical_tuning/classical_tuning_ledger.csv`
- Benchmark seed summary: `results/seed_suite_hybrid_public/benchmark_seed_summary.csv`
- Stable learned-comparator seed summaries: `results/seed_suite_hybrid_public/benchmark_seed_summary.csv`, `results/seed_suite_matched_nograph_rgr/benchmark_seed_summary.csv`, `results/seed_suite_capacity_matched_nograph_rgr/benchmark_seed_summary.csv`, `results/seed_suite_observability_context/benchmark_seed_summary.csv`, `results/seed_suite_nograph_public/benchmark_seed_summary.csv`, `results/seed_suite_kalmannet_public/benchmark_seed_summary.csv`
- Withdrawn/provenance-only seed summary: `results/seed_suite_innovation_public/benchmark_seed_summary.csv` (not independent repeated-seed corroboration because the distinctness audit shows seed-41/43 duplication with RGR-GF).
- Benchmark task registry: `results/benchmark_tasks/packet_registry.json`
- Benchmark task definition: `results/benchmark_tasks/task_definition.json`
- Stability prediction summary: `results/benchmark_tasks/stability_prediction_summary.csv`
- Method selection summary: `results/benchmark_tasks/method_selection_summary.csv`
- Observability summary: `results/observability/observability_summary.csv`
- Observability correlations: `results/observability/observability_correlations.csv`
- Guarded observability selector summary: `results/observability_guard/guarded_selector_summary.json`
- Guarded observability selector metrics: `results/observability_guard/guarded_selector_metrics.csv`
- Guarded observability selector checkpoint: `results/observability_guard/guarded_observability_selector.pt`
- K=96 temporal-order evidence: timestamp-only internal evidence records the rule fixed at 2026-05-25T13:06:32Z before the archived K=96 evaluation-start timestamp at 2026-05-25T13:12:43.6581323Z.

## Current Main-Manuscript Generated Table Inputs
Static evidence from `paper/main.tex` currently shows exactly these generated table inputs:
- `paper\tables\main_abbreviation_glossary.tex`
- `paper\tables\main_framework_portability.tex`
- `paper\tables\main_findings_summary.tex`
- `paper\tables\main_k32_replication.tex`
- `paper\tables\adaptive_candidate_fusion_full_training_poc.tex`
- `paper\tables\graph_anchor_pair_gate_poc.tex`
- `paper\tables\main_aukf_mechanism.tex`
- `paper\tables\main_structural_recoverability.tex`
- `paper\tables\main_drag_scale_cascade.tex`
- `paper\tables\main_long_arc_result.tex`
- `paper\tables\main_dbar_withdrawal.tex`

### AdaptiveCandidateFusion Table Source Artifacts
`paper\tables\adaptive_candidate_fusion_full_training_poc.tex` is generated
from both the fixed-soft full-training campaign summary and the 15-seed
validation-selected global scenario portfolio:
- `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.json`
- `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.md`
- `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv`
- `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json`
- `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.md`
- `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.csv`

These are current-workspace internal compact-simulator artifacts only. They are
not part of the published public v1.2.1 package unless a later release includes
them, not operational precise-reference validation, not independent-machine
reproduction, not a full raw/all-filter/public rerun, and not a universal
learned orbit-determination claim.

## Current Main-Manuscript Figure Includes
Static evidence from `paper/main.tex` currently shows these figure includes:
- `paper\figures\graph_anchor_pair_gate_seed_sweep_aggregate.png`
- `paper\figures\aukf_r_inflation_mechanism.png`

## Canonical Main-Manuscript Artifact Arrays
`results/release_packet.json` stores `canonical_artifacts.tables` and `canonical_artifacts.figures` as the current active main-manuscript table inputs and figure includes parsed from `paper/main.tex`. Release-only, historical, and diagnostic files are tracked separately under `manuscript_inclusion_status` and `auxiliary_artifacts`.

## Current Supplement Generated Table Inputs
Static evidence from `paper/supplement.tex` currently shows these generated table inputs:
- `paper\tables\crlb_floor_sensitivity.tex`
- `paper\tables\decision_stability.tex`
- `paper\tables\structural_channel_recoverability.tex`
- `paper\tables\validation_tuned_enkf_comparator.tex`
- `paper\tables\drag_scale_constructive_positive_control.tex`
- `paper\tables\drag_scale_ukf_constructive_positive_control.tex`
- `paper\tables\drag_scale_ukf_observability_positive_control.tex`
- `paper\tables\kalmannet_spot_od_transposition.tex`
- `paper\tables\kalmannet_spot_od_budget_adequacy.tex`
- `paper\tables\novelty_audit_systematic.tex`
- `paper\tables\closest_full_text_audit.tex`
- `paper\tables\protocol_subset_ablation.tex`
- `paper\tables\observed_step_prospective_replication.tex`
- `paper\tables\observed_step_powered_stress_replication.tex`
- `paper\tables\observed_step_internal_prospective_replication_k32.tex`
- `paper\tables\observed_step_internal_prospective_replication_k96_allscenario.tex`
- `paper\tables\observed_step_confidential_timestamp_k16.tex`
- `paper\tables\seed_observed_significance.tex`
- `paper\tables\observed_step_preregistration.tex`
- `paper\tables\endpoint_selection_sensitivity.tex`
- `paper\tables\force_mismatch_mechanism.tex`
- `paper\tables\hifi_force_mismatch.tex`
- `paper\tables\dmc_ekf_force_mismatch.tex`
- `paper\tables\drag_scale_aekf_force_mismatch.tex`
- `paper\tables\constrained_aukf_mechanism_control.tex`
- `paper\tables\long_arc_hifi_force_mismatch.tex`
- `paper\tables\measurement_informed_results.tex`
- `paper\tables\credible_dense_od_probe.tex`
- `paper\tables\dense_tracking_tail_audit.tex`
- `paper\tables\pukf_force_mismatch.tex`
- `paper\tables\pukf_tuning_sensitivity.tex`
- `paper\tables\hifi_force_mismatch_extended.tex`
- `paper\tables\kalmannet_official_reproduction.tex`
- `paper\tables\public_data_summary.tex`
- `paper\tables\benchmark_tasks.tex`
- `paper\tables\multiplicity_adjusted.tex`
- `paper\tables\main_results.tex`
- `paper\tables\force_model_mismatch.tex`
- `paper\tables\force_mismatch_significance.tex`
- `paper\tables\multi_rev_sgp4_benchmark.tex`
- `paper\tables\seed_aware_significance.tex`
- `paper\tables\floor_sensitivity_sweep.tex`
- `paper\tables\unconstrained_residual_comparator.tex`
- `paper\tables\residual_scale_sweep.tex`
- `paper\tables\dense_visibility_probe.tex`
- `paper\tables\adaptation_risk_diagnostic.tex`
- `paper\tables\dbar_independent_sweep.tex`
- `paper\tables\ablation.tex`
- `paper\tables\graph_matched_control.tex`
- `paper\tables\benchmark_suite.tex`
- `paper\tables\scenario_resampling.tex`
- `paper\tables\kalmannet_gain_inhouse_comparator.tex`
- `paper\tables\engineering_failure.tex`
- `paper\tables\seed_suite_public.tex`
- `paper\tables\seed_pooled_significance.tex`
- `paper\tables\ukf_gain_estimands.tex`
- `paper\tables\observability_diagnostics.tex`
- `paper\tables\observability_guard.tex`
- `paper\tables\robustness.tex`
- `paper\tables\satnogs_selection_sensitivity.tex`

## Formal400 Real SLR/SP3 Source Artifacts
These archived source artifacts back the formal400 bounded real SLR/SP3 sanity-probe tables and logs:
- `results/real_slr_sp3_od_formal400_inputs/real_slr_sp3_od_formal400_validation.json`
- `results/validation/real_slr_sp3_od_formal400_run.log`
- `results/validation/real_slr_sp3_od_formal400_run.err.log`

## Formal210 Real SLR/SP3 Source Artifacts (superseded)
These archived source artifacts backed the formal210 bounded real SLR/SP3 sanity-probe tables (now superseded by formal400):
- `results/real_slr_sp3_od_formal210_inputs/real_slr_sp3_od_formal210_validation.json`
- `results/validation/real_slr_sp3_od_formal210_run.log`
- `results/validation/real_slr_sp3_od_formal210_run.err.log`

## Pre-Update NIS Diagnostic Evidence Artifacts
These reviewer-auditable campaign outputs are retained as bounded diagnostic/evidence artifacts. They are not generated table inputs or figures and are not added to the canonical artifact arrays.
### High-Fidelity Pre-Update NIS Campaign
- `results/hifi_pre_update_nis_campaign/hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616.csv`
- `results/hifi_pre_update_nis_campaign/hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616.json`
- `results/hifi_pre_update_nis_campaign/hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616.md`
- `results/hifi_pre_update_nis_campaign/hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616.rows.jsonl`
- `results/validation/hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616_v2.err.log`
- `results/validation/hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616_v2.out.log`
### Compact Pre-Update NIS Campaign
- `results/aukf_nis_sampled_campaign/aukf_pre_update_nis_sampled_campaign_k8_n12_20260616.csv`
- `results/aukf_nis_sampled_campaign/aukf_pre_update_nis_sampled_campaign_k8_n12_20260616.json`
- `results/aukf_nis_sampled_campaign/aukf_pre_update_nis_sampled_campaign_k8_n12_20260616.md`
- `results/aukf_nis_sampled_campaign/aukf_pre_update_nis_sampled_campaign_k8_n12_20260616.rows.jsonl`
- `results/validation/aukf_pre_update_nis_sampled_campaign_k8_n12_20260616.out.log`

## Full Non-Destructive Rerun Evidence Artifacts
These reviewer-auditable outputs are diagnostic/reproducibility evidence from one non-destructive full raw-data generation, all-enabled learned-model training, and all-scenario classical+learned evaluation rerun under `results/full_rerun_20260616`. They are not generated manuscript table inputs, are not included in `canonical_artifacts.tables` or `canonical_artifacts.figures`, did not overwrite submitted canonical artifacts, and do not establish independent-machine or public-DOI/archive reproduction.
- Status: `pass`; completed: `2026-06-17T13:57:27.2931188+02:00`; evaluated scenarios: `17`.
- Metrics SHA-256: `ad14c74abb4b3cbcb74b4cafc7181d26ed188ff3aabc4c36008bd4fe07d69e6b`.
- Scorecard SHA-256: `dd3169a9e654c2d5e3e0966cc29864b7cdfee4d53829d17d9332e3e23b228fc0`.
- Divergence caveat: `dense_visibility_test` flags `UKF`, `AUKF`, `NoGraphResidual`, `LearnedNoiseAdaptive`, `HybridGNN`, `MatchedNoGraphRGR`, `CapacityMatchedNoGraphRGR`, `InnovationHybridGNN`, and `ObservabilityContextHybridGNN`; `satnogs_observation_replay_test` flags `UKF`, `LearnedNoiseAdaptive`, `HybridGNN`, `MatchedNoGraphRGR`, `InnovationHybridGNN`, and `ObservabilityContextHybridGNN`. Treat this as an inspectable rerun/stress artifact, not clean reproduction of every scientific table or stable operational validity.
- Divergence audit: `results/validation/full_rerun_divergence_audit_20260617.json` and `.md` reconcile the retained full-rerun divergence flags from `metrics_summary.json`, `scorecard_summary.json`, and `trajectory_errors.csv`. The audit is diagnostic only; failure-conditioned rows are not replacement metrics, do not redefine performance, and do not rescue any method or learned-positive interpretation.
- `results/full_rerun_20260616/full_rerun_status.json`
- `results/full_rerun_20260616/full_rerun_summary.json`
- `results/full_rerun_20260616/01_environment_check.status.json`
- `results/full_rerun_20260616/02_generate_dataset.status.json`
- `results/full_rerun_20260616/03_train_models_all.status.json`
- `results/full_rerun_20260616/04_evaluate_models_all.status.json`
- `results/full_rerun_20260616/05_summarize_full_rerun.status.json`
- `results/full_rerun_20260616/metrics_summary.json`
- `results/full_rerun_20260616/scorecard_summary.json`
- `results/full_rerun_20260616/per_step_errors.csv`
- `results/full_rerun_20260616/trajectory_errors.csv`
- `results/full_rerun_20260616/trajectory_improvement.csv`
- `results/full_rerun_20260616/uncertainty_calibration.csv`
- `results/full_rerun_20260616/predictions_test.npz`
- `results/full_rerun_20260616/data/dataset_manifest.json`
- `results/full_rerun_20260616/checkpoints/train_history.json`
- `results/full_rerun_20260616/manifests/evaluation.json`
- `results/full_rerun_20260616/figures/hybrid_vs_ukf_improvement_hist.png`
- `results/full_rerun_20260616/figures/per_step_rmse.png`
- `results/full_rerun_20260616/figures/position_error_boxplot.png`
- `results/full_rerun_20260616/figures/position_error_ecdf.png`
- `results/full_rerun_20260616/figures/uncertainty_calibration.png`
- `results/full_rerun_20260616/figures/visibility_bucket_rmse.png`
- `results/full_rerun_20260616/baseline_cache/credible_dense_od_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/dense_visibility_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/force_component_omission_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/force_model_mismatch_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/high_drag_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/high_inclination_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/low_inclination_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/maneuver_shift_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/process_noise_shift_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/public_catalog_replay_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/satnogs_observation_replay_stress_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/satnogs_observation_replay_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/satnogs_observation_replay_val_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/semi_real_replay_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/stress_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/sunsync_like_test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/test_baselines.npz`
- `results/full_rerun_20260616/baseline_cache/val_baselines.npz`
- `results/validation/full_rerun_divergence_audit_20260617.json`
- `results/validation/full_rerun_divergence_audit_20260617.md`
- `scripts/build_full_rerun_divergence_audit.py`
- `tests/test_full_rerun_divergence_audit.py`

## Planned Appendix / Supplement or Candidate Integration Tables
These release artifacts exist and may be useful for an appendix, supplement, or future manuscript expansion, but they are not direct generated-table inputs in the current `paper/main.tex`:
- None

## Release-Only Diagnostic Tables
These generated table artifacts are reviewer-auditable diagnostics or generated counterparts to inline/prose evidence, but they are not direct generated-table inputs in the current manuscript:
- `paper\tables\method_activity.tex`
- `paper\tables\seed_sweep.tex`
- `paper\tables\trajectory_improvement.tex`
- `paper\tables\visibility_buckets.tex`

## Planned Appendix / Supplement or Candidate Integration Figures
These figure artifacts exist in the release packet, but no figure is currently included by `paper/main.tex`:
- None

## Release-Only Diagnostic Figures
These figures exist as reviewer-auditable diagnostics, but they are not current main-manuscript figure includes:
- `paper\figures\hybrid_vs_ukf_improvement_hist.png`
- `paper\figures\observability_vs_ekf_error.png`
- `paper\figures\position_error_boxplot.png`
- `paper\figures\training_curves.png`
- `paper\figures\visibility_bucket_relative_gain.png`

## Historical / Auxiliary Artifacts
These files may still be useful diagnostically, but they are not part of the current manuscript evidence path unless regenerated, explicitly cited, and intentionally integrated.
### Historical / Auxiliary Tables
- `paper\tables\anchor_pair_gate_poc.tex`
- `paper\tables\batch_wls_baseline.tex`
- `paper\tables\calibration.tex`
- `paper\tables\correction_component_audit.tex`
- `paper\tables\coverage_runtime.tex`
- `paper\tables\dropout_sensitivity.tex`
- `paper\tables\method_selection.tex`
- `paper\tables\process_noise_sensitivity.tex`
- `paper\tables\propagation_baseline.tex`
- `paper\tables\public_sp3_multi_target_breadth_probe.tex`
- `paper\tables\pure_gnn_training_sanity.tex`
- `paper\tables\real_slr_lageos_validation.tex`
- `paper\tables\real_slr_sp3_corrected.tex`
- `paper\tables\real_slr_sp3_hifi.tex`
- `paper\tables\real_slr_sp3_od.tex`
- `paper\tables\real_slr_sp3_od_expanded.tex`
- `paper\tables\real_slr_sp3_od_expanded_mechanism_heterogeneity.tex`
- `paper\tables\real_slr_sp3_od_expanded_stratification.tex`
- `paper\tables\real_slr_sp3_state_scoring_campaign.tex`
- `paper\tables\real_slr_sp3_temporal_corrected_od_campaign.tex`
- `paper\tables\real_slr_sp3_temporal_corrected_od_campaign_summary.tex`
- `paper\tables\real_slr_sp3_temporal_od_campaign.tex`
- `paper\tables\real_slr_sp3_temporal_selection_stability.tex`
- `paper\tables\rfis_smoother_shift.tex`
- `paper\tables\satnogs_observed_step_diagnostic.tex`
- `paper\tables\satnogs_timefix_validation.tex`
- `paper\tables\seed_suite_distinctness.tex`
- `paper\tables\significance.tex`
- `paper\tables\stability_prediction.tex`
- `paper\tables\station_outage.tex`
- `paper\tables\window_sensitivity.tex`
### Historical / Auxiliary Figures
- `paper\figures\anchor_pair_gate_candidate_comparison.png`
- `paper\figures\dropout_sensitivity.png`
- `paper\figures\graph_anchor_pair_gate_candidate_comparison.png`
- `paper\figures\process_noise_sensitivity.png`
- `paper\figures\robustness_aukf_heatmap.png`
- `paper\figures\robustness_ekf_heatmap.png`
- `paper\figures\robustness_hybrid_heatmap.png`
- `paper\figures\robustness_profile.png`
- `paper\figures\station_outage_sensitivity.png`
- `paper\figures\window_size_sensitivity.png`
### Historical Review Notes
- `paper\PEER_REVIEW_DOSSIER.md`
- `paper\DEEP_REVIEW_2026-04-16.md`
- `paper\REVIEW_50_PASS_MATRIX.md`
- `paper\review_rounds.md`
- `paper\NOVELTY_REVIEW_ROUNDS_2026-04-29.md`
- `paper\OBSERVABILITY_CONTEXT_TRAINING_REVIEW_2026-04-29.md`
- `paper\GUARDED_OBSERVABILITY_SELECTOR_REVIEW_2026-04-29.md`
- `paper\CHANGELOG_REVIEW_REPAIR.md`
- `paper\ISSUE_TRACKER.md`

## Provenance
- Config SHA256: `dd941062a09cee590256228fecee54200c8642add324570ddf1eead28534543f`
- Source snapshot SHA256: `721d19444b826366533bc8ec07e8dff3bac3efc2cf4437e8b2af553865107f9c`
- VCS available: `True`
- Runtime execution details: omitted from this reviewer-facing summary; raw run records retain their original metadata.

## Claim Boundary

This release-truth synchronization changes metadata labels only. It does not change numerical results, performance interpretations, novelty claims, or fresh-rerun status.
