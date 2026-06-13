# Canonical Release Packet

This file identifies the evidence path that is canonical for the current paper state.

## Canonical Evidence
- Manuscript: `main.tex`
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

Note: the evaluation manifest records the most recent targeted rerun.
The canonical multi-scenario comparison remains `results/metrics_summary.json` and `results/scorecard_summary.json`.
Last targeted rerun scenarios: `test,stress_test,force_model_mismatch_test`.

## Current Main-Manuscript Generated Table Inputs
Static evidence from `paper/main.tex` currently shows exactly these generated table inputs:
- `paper\tables\main_abbreviation_glossary.tex`
- `paper\tables\main_framework_portability.tex`
- `paper\tables\main_findings_summary.tex`
- `paper\tables\main_k32_replication.tex`
- `paper\tables\main_aukf_mechanism.tex`
- `paper\tables\main_structural_recoverability.tex`
- `paper\tables\main_drag_scale_cascade.tex`
- `paper\tables\main_long_arc_result.tex`
- `paper\tables\main_dbar_withdrawal.tex`

## Current Main-Manuscript Figure Includes
Static evidence from `paper/main.tex` currently shows these figure includes:
- `paper\figures\aukf_r_inflation_mechanism.png`

## Canonical Main-Manuscript Artifact Arrays
`results/release_packet.json` stores `canonical_artifacts.tables` and `canonical_artifacts.figures` as the current active main-manuscript table inputs and figure includes parsed from `paper/main.tex`. Release-only, historical, and diagnostic files are tracked separately under `manuscript_inclusion_status` and `auxiliary_artifacts`.

## Current Supplement Generated Table Inputs
Static evidence from `paper/supplement.tex` currently shows these generated table inputs:
- `paper\tables\crlb_floor_sensitivity.tex`
- `paper\tables\decision_stability.tex`
- `paper\tables\structural_channel_recoverability.tex`
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
- `paper\figures\dropout_sensitivity.png`
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
- Source snapshot SHA256: `19aae77331a347a8a48b96439938f92c4a5e94b4fd8c90d1e060d323fbef187d`
- Historical evaluation-manifest VCS available: `False`
- Current workspace git context: this worktree is inside a git repository at HEAD `afe78f7a67c5192108b7ae03c3813cd2e61660e0` on branch `main`, with origin `git@gitlab.com:papers8721323/gnn-state-estimation.git`.
- Current git metadata is provenance context only, not a complete reproducible source snapshot: most evidence/package files in this worktree are currently untracked, so reproducibility still depends on the submitted evidence bundle and any later committed or deposited archive.
- Runtime execution details: omitted from this reviewer-facing summary; raw run records retain their original metadata.

## Claim Boundary

This release-truth synchronization changes metadata labels only. It does not change numerical results, performance interpretations, novelty claims, or fresh-rerun status.
