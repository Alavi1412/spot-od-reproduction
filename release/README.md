# SPOT-OD Supplementary Evidence Package

Version: `1.1.0-supplement` (version-pinned, anonymized, included with the
submission for reviewer-accessible independent inspection; no external public
repository or DOI is asserted).

This versioned package accompanies the simulator-bound SPOT-OD self-audit
record submission and supports inspection of the displayed evidence. It is provided so
that the displayed evidence can be inspected, table-regenerated from
materialized artifacts, and selectively replayed at the documented tiers.

## Reviewer Quickstart

1. Open `REVIEWER_START_HERE.md` for the headline claim map, manifest-entry
   keys, and regeneration tiers.
2. Open `SUPPLEMENTARY_MANIFEST.json` and confirm that the package version,
   artifact count, artifact-present count, and per-artifact SHA-256 inventory
   are present.
3. Confirm that the reviewer transport archive digest recorded under
   `review_archive.sha256` matches the archive supplied with the submission.
4. Minimum inspection tier: compare the manifest inventory and SHA-256 digests
   against the submitted archive contents. This tier requires no retraining.
5. Archive-extracted reproduction tier: extract the review archive, verify the
   manifest-indexed member digests, resolve claim and regeneration-tier records,
   rerun active main-manuscript table regeneration from the extracted tree, and
   rerun the bounded public LAGEOS CRD/SP3 OD slice from the extracted archive.
   The current reports are in
   `results/validation/archive_extracted_reproduction.json` and `.md`.
6. Table regeneration tier: regenerate active main-manuscript generated tables
   from the recorded outputs and compare byte hashes with the submitted table
   files. The current report is in
   `results/validation/active_manuscript_regeneration.json` and `.md`.
7. Public precise-reference OD slice rerun: inspect
   `results/validation/real_slr_sp3_od_slice_rerun.json` and `.md`. This
   reruns one LAGEOS CRD/SP3 slice from archived public inputs through
   recursive filter recomputation and table reconstruction; it is not a full
   scientific rerun or operational POD validation.
8. Targeted retraining replay: inspect
   `results/validation/targeted_retraining_replay_public.json` and `.md`. This is one
   bounded representative training replay on deterministic slices of existing
   materialized data; it is not a full paper-table or full seed-suite rerun.
9. Full rerun tier: regenerate data, train the learned estimators, evaluate all
   methods, rebuild the manuscript and supplement, and rerun the manuscript and
   release checks. This tier is intentionally slower and is not required for the
   minimum integrity check.
10. Full-rerun divergence audit: inspect
    `results/validation/full_rerun_divergence_audit_20260617.json` and `.md`.
    This generated audit reconciles the retained dense-visibility and SatNOGS
    replay divergence flags from the full-rerun metrics, scorecard, and
    trajectory-error files. It is diagnostic only, not a canonical table
    replacement or rerun-success upgrade.
11. Official ILRS precise-reference availability gate: inspect
    `results/validation/ilrs_precise_reference_availability_20260617.json` and
    `.md`. This probes EDC/CDDIS URLs and local cached pending `.sp3.gz`
    filenames with a gzip/SP3 content gate; unavailable products and cached
    HTML placeholders are not scored validation.
12. Independent-machine reproduction handoff: inspect
    `release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`. This is a request
    and report template for an external operator, not a completed independent
    reproduction and not a completed independent reproduction record.

## Reading Manifest And Attestation Counts

`SUPPLEMENTARY_MANIFEST.json` is authoritative for final submitted-package
artifact counts and `review_archive.sha256` / byte count. Read the current
digest and counts directly from that manifest, not from this guide.
Archive-extracted reports intentionally do not embed the enclosing archive
SHA-256 or byte size because each report is itself archived, which would create
a self-referential digest cycle. Validation and attestation files are
run-boundary records; if a count or digest differs from the final manifest, use
the final manifest for the submitted package and treat the report count or
digest as the state used by that verifier run. The same verifier `pass` status
is not changed by that bookkeeping distinction.

## Regeneration Tiers

Wall-clock expectations are tiered. The minimum integrity check is normally
under 5 minutes; archive-extracted and table
regeneration checks are normally under about 20 minutes when materialized
artifacts are present; the public precise-reference OD slice is normally under
10 minutes with archived CRD/SP3 inputs. The targeted retraining replay is a
bounded deterministic-slice replay and is expected to be under about 30
minutes on typical compute. The full rerun tier can require hours to
days depending on available compute and is not required for minimum reviewer
inspection.

### Minimum Integrity Check

Expected runtime: under 5 minutes. No retraining is
required.

Entry points are recorded in `SUPPLEMENTARY_MANIFEST.json` under
`regeneration_tiers.minimum_integrity_check.entrypoints`.

### Archive-Extracted Reproduction Check

Expected runtime: under 20 minutes when materialized
result artifacts are already present. No learned-estimator retraining is
required.

The entry point is recorded in `SUPPLEMENTARY_MANIFEST.json` under
`regeneration_tiers.archive_extracted_reproduction_check.entrypoints`.

Boundary: Archive-extracted integrity, active main-manuscript
table-regeneration, and one public LAGEOS CRD/SP3 precise-reference OD slice
recomputation from archived public inputs only; this does not rerun full
raw-data generation, model retraining, all recursive filters or tables, live
public-data retrieval, operational POD validation, or independent
machine reproduction.

### Public Precise-Reference OD Slice Rerun

Expected runtime: under 10 minutes when the archived public CRD/SP3 inputs are
present. No learned-estimator retraining is required.

Entry points and associated tests are recorded in `SUPPLEMENTARY_MANIFEST.json`
under `regeneration_tiers.public_precise_reference_od_slice_rerun_check.entrypoints`.

Boundary: one public LAGEOS CRD/SP3 precise-reference OD slice is rerun from
archived public inputs through recursive range-only filter recomputation and
table reconstruction. This is not full scientific reproduction, full estimator
training, all-table regeneration, live public-data retrieval, or operational
POD validation.

### Targeted Retraining Replay

Expected runtime: under 30 minutes for the representative deterministic slice.
This tier reruns one bounded learned-estimator training replay and records
finite objective histories, output digests, and unchanged canonical checkpoint
digests. Machine-specific execution details are retained only in raw run
records and are not summarized in this release guide.

The bounded replay entry point, argument summary, and associated test are
recorded in `SUPPLEMENTARY_MANIFEST.json` under
`regeneration_tiers.targeted_retraining_replay_check.entrypoints`.

Boundary: representative replay only; not full paper-table reproduction, not a
full seed-suite rerun, not main-result reproduction, and not replacement of the
canonical submitted checkpoints.

### Table-Regeneration Check

Expected runtime: under 20 minutes when materialized
result artifacts are already present. No learned-estimator retraining is
required.

Entry points are recorded in `SUPPLEMENTARY_MANIFEST.json` under
`regeneration_tiers.table_regeneration_check.entrypoints`.

### Full Rerun

Expected runtime: hours to days depending on available compute. The core
pipeline entry point and mode are recorded in `SUPPLEMENTARY_MANIFEST.json`
under `regeneration_tiers.full_rerun.entrypoints`.
The bounded divergence audit at
`results/validation/full_rerun_divergence_audit_20260617.json` and `.md`
is generated from retained full-rerun outputs and does not rerun models.

## Public Archive Commitment

No public DOI is asserted for this anonymous review-stage package at initial
submission. Public archival deposition is deferred until explicit author
approval or a venue-required release point; if a DOI is assigned later, it will
be recorded in this README, `CITATION.cff`, and `SUPPLEMENTARY_MANIFEST.json`
before any public citation of the archive.

## Contents

- `CITATION.cff` - local citation metadata for the package.
- `ZENODO_METADATA.json` - DOI-ready deposit metadata for the committed
  Zenodo upload; it intentionally contains no DOI or public URL at submission.
- `REVIEWER_START_HERE.md` - reviewer-facing map from headline claims to
  manifest entries and regeneration tiers.
- `INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md` - clean-machine operator
  instructions and report template; this is not a completed independent
  reproduction.
- `SUPPLEMENTARY_MANIFEST.json` - machine-readable index of the evidence
  artifacts with SHA-256 checksums, the canonical seed cohort, and the package
  version. Regenerated by `scripts/build_supplementary_manifest.py`.
- `results/validation/active_manuscript_regeneration.json` and `.md` -
  table-regeneration tier report for the active generated tables included by
  the main manuscript.
- `results/validation/archive_extracted_reproduction.json` and `.md` -
  archive-extracted integrity, active table-regeneration, and public OD slice
  recomputation tier report.
- `results/validation/codex_clean_archive_attestation_20260614.json` and
  `.md` - same-workspace separate-agent clean extracted archive attestation;
  not independent-machine validation, not third-party reproduction, not
  DOI/public archive, not full scientific reproduction, and not operational
  POD.
- `results/validation/containerized_minimum_tier_attestation_20260614.json`
  and `.md` - local Docker Desktop clean-container staged minimum-tier
  integrity attestation; same-host and staged from the current
  workspace/review archive; not independent-machine validation, not
  third-party reproduction, not DOI/public archive, not full scientific
  reproduction, not operational POD, not independent external reproduction,
  and not a full scientific rerun.
- `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json` and
  `.md` - companion report for the archive-extracted public OD slice rerun;
  persistent comparison outputs are under
  `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun/`.
- `results/validation/real_slr_sp3_od_slice_rerun.json` and `.md` - bounded
  public precise-reference OD slice-rerun report; persistent rerun outputs are
  under `results/validation/real_slr_sp3_od_slice_rerun/`.
- `results/real_slr_sp3_od_formal400_inputs/real_slr_sp3_od_formal400_validation.json`,
  `results/validation/real_slr_sp3_od_formal400_run.log`,
  `results/validation/real_slr_sp3_od_formal400_run.err.log`, and
  `paper/tables/real_slr_sp3_od_expanded.tex` - formal-power-scale
  400-attempt/373-completed compact public SLR/SP3 recursive-filter replay;
  `paper/tables/real_slr_sp3_od_expanded_stratification.tex` and
  `paper/tables/real_slr_sp3_od_expanded_mechanism_heterogeneity.tex` are
  supplementary completed-arc diagnostic tables for the same validation
  record. The 27 non-completed arcs are explicit status records (17
  insufficient observations; 10 public product unavailable or non-parseable),
  not hidden or imputed. This replay supersedes the earlier formal210 and
  expanded80 compact replay archives as the strongest compact real-data
  diagnostic, still negative/diagnostic only.
- `results/validation/targeted_retraining_replay_public.json` and `.md` - bounded
  representative learned-estimator retraining replay report.
- `results/validation/full_rerun_divergence_audit_20260617.json` and `.md` -
  bounded generated audit of the full-rerun divergence flags; diagnostic only,
  not a canonical table replacement, not operational validation, not
  independent reproduction, and not a rerun success upgrade.
- `results/validation/ilrs_precise_reference_availability_20260617.json` and
  `.md` - official EDC/CDDIS and local-cache availability probe for pending
  LAGEOS prospective SP3 products. It classifies direct HTTP responses and
  cached `.sp3.gz` filenames as usable only when gzip and SP3-header checks
  pass.
- `results/public_sp3_multi_target_breadth_probe/public_sp3_multi_target_breadth_probe.json`
  and `paper/tables/public_sp3_multi_target_breadth_probe.tex` - public
  multi-target SP3/CRD breadth probe records, including fixed-start and
  target-week clustered finite-probe sensitivity summaries.
- `spot_od_v1_1_0_supplement_review_archive.zip` - reviewer-access transport
  archive generated from the indexed artifacts. Its SHA-256 digest is
  recorded separately in `SUPPLEMENTARY_MANIFEST.json`; the archive itself is
  not counted as an indexed artifact.

## Claim-To-Artifact Map

This map is the reviewer-facing route from each manuscript headline to the
records needed to inspect it. `REVIEWER_START_HERE.md` adds the corresponding
manifest keys and regeneration tiers.

| Manuscript headline | Primary records |
|---|---|
| Audited learned family bounded negative on observed-step RMSE | Manifest entry `audited_learned_family_bounded_negative`; primary records include the observed-step seed summaries, endpoint-fixation support, larger endpoint replication, and their generated tables. |
| Larger simulator-bound endpoint replication under a frozen K=32 decision rule and established observed-step hierarchy | Manifest entry `larger_simulator_bound_endpoint_replication`; primary records include the larger endpoint replication JSON and generated table. |
| Stress-only floor-power-scale observed-step replication | Manifest entry `powered_stress_floor_scale_replication`; primary records include the stress-only K=96 replication JSON, its generated table, and the frozen rule record; timestamp-only internal evidence records the rule fixed at 2026-05-25T13:06:32Z before the archived K=96 evaluation-start timestamp at 2026-05-25T13:12:43.6581323Z. |
| All-scenario K=96 internal observed-step replication | Manifest entry `all_scenario_k96_internal_replication`; primary records include `results/observed_step_internal_prospective_replication_loop163_k96/observed_step_internal_prospective_replication_loop163_k96.json`, `results/observed_step_internal_prospective_replication_loop163_k96/preregistration.json`, and `paper/tables/observed_step_internal_prospective_replication_k96_allscenario.tex`; this is internal simulator-bound evidence, not external validation. |
| Endpoint-choice sensitivity under observed-step and all-step RMSE | Manifest entry `endpoint_choice_sensitivity_audit`; primary records include `results/endpoint_selection_sensitivity/endpoint_selection_sensitivity.json`, `paper/tables/endpoint_selection_sensitivity.tex`, and the deterministic builder script. |
| PUKF tuning-comparability sensitivity on the higher-fidelity force-mismatch slice | Manifest entry `pukf_tuning_comparability_sensitivity`; primary records include `results/pukf_tuning_sensitivity/pukf_hifi_grid_sensitivity.json`, `paper/tables/pukf_tuning_sensitivity.tex`, the retained higher-fidelity CSV/JSON, and the generator script. |
| Tuned classical guardrails and offline OD reference prevent over-crediting | Manifest entry `classical_guardrails_and_offline_od_reference`; primary records include `results/metrics_summary.json`, `results/batch_wls_baseline/batch_wls_summary.csv`, `paper/tables/main_results.tex`, `paper/tables/batch_wls_baseline.tex`, and the controlled force-mismatch WLS summaries `results/batch_wls_force_mismatch/batch_wls_summary.csv`, `results/batch_wls_force_mismatch/batch_wls_summary.json`, and `results/batch_wls_force_mismatch/force_model_mismatch_test/batch_wls_summary.json`. Boundary: WLS is strong on nominal/stress/public full-arc reference rows, but under controlled force mismatch it fails observed-step scoring while slightly improving all-step RMSE. |
| Protocol subset sufficiency: each ingredient blocks a misleading positive or unsupported upgrade | Manifest entry `protocol_subset_sufficiency`; primary records include the protocol subset rule artifact and generated table. |
| Noise-side and force-side structural-channel bounded negatives plus favorable-geometry recoverability sanity check | Manifest entry `structural_channel_bounded_negatives`; primary records include the DMC/drag-scale/higher-fidelity/recoverability JSON records and generated tables. |
| Long-arc AUKF/EKF scope-down and decision stability | Manifest entry `long_arc_scope_down_and_decision_stability`; primary records include the long-arc higher-fidelity result, decision-stability result, and generated table. |
| DBAR withdrawal after adequately powered characterization | `results/adaptation_risk_diagnostic/dbar_independent_sweep.json`, `paper/tables/dbar_independent_sweep.tex`, `paper/tables/adaptation_risk_diagnostic.tex` |
| KalmanNet native benchmark and bounded SPOT-OD transposition | Manifest entry `kalmannet_native_and_transposition_diagnostics`; primary records include the native sanity check, SPOT-OD transposition records, and generated tables. |
| Real SLR/SP3 slices as bounded provenance and precise-reference sanity probes | `results/real_slr_lageos/real_slr_lageos_validation.json`, `results/real_slr_sp3_od/real_slr_sp3_od_validation.json`, `results/real_slr_sp3_od_formal400_inputs/real_slr_sp3_od_formal400_validation.json`, `results/validation/real_slr_sp3_od_formal400_run.log`, `results/validation/real_slr_sp3_od_formal400_run.err.log`, `paper/tables/real_slr_sp3_od_expanded.tex`, `paper/tables/real_slr_sp3_od_expanded_stratification.tex`, `paper/tables/real_slr_sp3_od_expanded_mechanism_heterogeneity.tex`, earlier/superseded archival `results/real_slr_sp3_od_formal210_inputs/real_slr_sp3_od_formal210_validation.json`, earlier/superseded archival `results/real_slr_sp3_od_expanded80_inputs/real_slr_sp3_od_expanded80_validation.json`, `results/real_slr_sp3_hifi/real_slr_sp3_hifi_validation.json`, `results/real_slr_sp3_corrected/real_slr_sp3_corrected_validation.json`, `results/real_slr_sp3_correction_corpus_audit/real_slr_sp3_correction_corpus_audit.json`, `results/real_slr_sp3_state_scoring_campaign/real_slr_sp3_state_scoring_campaign.json`, `results/real_slr_sp3_temporal_od_campaign/real_slr_sp3_temporal_od_campaign.json`, `results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json`, `results/validation/real_slr_sp3_od_slice_rerun.json`, corresponding `paper/tables/real_slr_*.tex` tables |
| Public ILRS/SP3 state-scoring campaign | `results/real_slr_sp3_state_scoring_campaign/real_slr_sp3_state_scoring_campaign.json`, `paper/tables/real_slr_sp3_state_scoring_campaign.tex`, `scripts/run_real_slr_sp3_state_scoring_campaign.py`, `tests/test_real_slr_sp3_state_scoring_campaign.py` |
| Public multi-target SP3/CRD breadth probe | `results/public_sp3_multi_target_breadth_probe/public_sp3_multi_target_breadth_probe.json`, `paper/tables/public_sp3_multi_target_breadth_probe.tex`, `scripts/run_public_sp3_multi_target_breadth_probe.py`, `tests/test_public_sp3_multi_target_breadth_probe.py`, archived SP3/CRD input files under `results/public_sp3_multi_target_breadth_probe/sp3/` and `results/public_sp3_multi_target_breadth_probe/crd/`; includes fixed-start, target-week clustered, and target-cluster sensitivity fields |
| Temporal public real-measurement OD probe | `results/real_slr_sp3_temporal_od_campaign/real_slr_sp3_temporal_od_campaign.json`, `paper/tables/real_slr_sp3_temporal_od_campaign.tex`, `results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json`, `paper/tables/real_slr_sp3_temporal_selection_stability.tex`, `scripts/run_real_slr_sp3_temporal_od_campaign.py`, `tests/test_real_slr_sp3_temporal_od_campaign.py` |
| Temporal public real-measurement OD selection-stability audit | `results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json`, `paper/tables/real_slr_sp3_temporal_selection_stability.tex`, `scripts/run_real_slr_sp3_temporal_selection_stability.py`, `tests/test_real_slr_sp3_temporal_selection_stability.py` |
| Prospective 260523 public-week full-correction readout | `results/real_slr_sp3_temporal_corrected_od_prospective_260523/real_slr_sp3_temporal_corrected_od_prospective_260523.json`, `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260523_20260526.json`, `paper/tables/real_slr_sp3_temporal_corrected_od_campaign_summary.tex`, `scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py`, `tests/test_real_slr_sp3_temporal_corrected_od_campaign.py`; the 260523 mean is numerically lower for the learned residual but the confidence interval spans zero, so this is indeterminate provenance/transfer evidence, not public validation or a DOI/archive claim. |
| Pending prospective 260620 public-week full-correction readout | Manifest entry `pending_public_real_measurement_temporal_od_prospective_260620`; primary records are `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json`, `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.sha256.txt`, `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json.ots`, `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_availability_20260613.json`, `results/validation/public_precise_reference_status_20260617.json`, `results/validation/public_precise_reference_status_20260617.md`, `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_timestamp_attestation.json`, and `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_timestamp_attestation.md`; created 2026-06-12 for test dates 2026-06-15..2026-06-19 and pending/not scored as of 2026-06-17; the OTS material is a Bitcoin-block-header-attested OpenTimestamps proof with Merkle roots cross-checked against the Blockstream API, but local Bitcoin-node verification was not performed and it is not scored validation, not DOI/public archive, not independent reproduction, and not operational POD; the retained 2026-06-13 availability audit is superseded for current availability by the 2026-06-17 status, which records parent listings with no 260613/260620 directories and direct 260613/260620 NSGF URLs returning HTTP 200 text/html; charset=utf-8 responses of length 825, not valid .sp3.gz products; cannot be used yet as validation and must not be represented as scored validation. |
| Pending prospective 260627 public-week full-correction readiness record | Manifest entry `pending_public_real_measurement_temporal_od_prospective_260627`; primary records are `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json`, `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.sha256.txt`, `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json.ots`, `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260627_predeclaration_status_20260617.json`, and `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260627_predeclaration_status_20260617.md`; created 2026-06-17 before test dates 2026-06-22..2026-06-26 with validation week 260620 and test week 260627; pending/not scored, with no campaign run; the status record cites `results/validation/public_precise_reference_status_20260617.json` and `.md` as the current availability basis; the OTS proof is calendar-pending and not Bitcoin-block-header-attested, not scored validation, not DOI/public archive, not independent reproduction, and not operational POD; 260627 cannot be used as validation until validation/test public SP3 products become valid gzip/SP3 products and the frozen rule is scored once. |
| Official ILRS precise-reference availability gate | Manifest entry `official_ilrs_precise_reference_availability_probe`; primary records are `scripts/probe_ilrs_precise_reference_availability.py`, `tests/test_probe_ilrs_precise_reference_availability.py`, `results/validation/ilrs_precise_reference_availability_20260617.json`, and `results/validation/ilrs_precise_reference_availability_20260617.md`; this is an availability probe only, not scored validation. Cached `.sp3.gz` files that are 825-byte HTML placeholders remain unavailable/non-usable and must not be relabelled as valid SP3. |
| Reproducibility and review-stage archive integrity | `REVIEWER_START_HERE.md`, `SUPPLEMENTARY_MANIFEST.json`, `spot_od_v1_1_0_supplement_review_archive.zip`, `results/validation/minimum_tier_reproduction_check.json`, `results/validation/submission_validation.json`, `results/release_packet.json` |
| Independent-machine reproduction request/template | Manifest entry `independent_machine_reproduction_request`; primary record is `release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`; this gives an external operator exact clean-machine verification steps and a report template but is not a completed independent reproduction and is not an already completed independent reproduction record. |
| Active main-manuscript table regeneration | `results/validation/active_manuscript_regeneration.json`, `results/validation/active_manuscript_regeneration.md`, `results/validation/command_manifest.json`, active `paper/tables/*.tex` files included by `paper/main.tex` |
| Archive-extracted reproduction tier | `results/validation/archive_extracted_reproduction.json`, `results/validation/archive_extracted_reproduction.md`, `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json`, `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.md`, `scripts/verify_archive_extracted_reproduction.py`, `scripts/regenerate_active_manuscript.py`, `scripts/run_real_slr_sp3_od_validation.py`, `scripts/build_paper_assets.py`, `spot_od_v1_1_0_supplement_review_archive.zip`, `SUPPLEMENTARY_MANIFEST.json` |
| Codex-agent clean archive attestation | Manifest entry `codex_clean_archive_attestation`; primary records are `results/validation/codex_clean_archive_attestation_20260614.json` and `results/validation/codex_clean_archive_attestation_20260614.md`; boundary: same-workspace separate-agent clean extracted archive attestation only, not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, and not independent external reproduction. |
| Containerized minimum-tier attestation | Manifest entry `containerized_minimum_tier_attestation`; primary records are `results/validation/containerized_minimum_tier_attestation_20260614.json` and `results/validation/containerized_minimum_tier_attestation_20260614.md`; boundary: local Docker Desktop clean-container staged minimum-tier integrity attestation only, same-host and staged from the current workspace/review archive, not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, not independent external reproduction, and not a full scientific rerun. |
| Targeted retraining replay | Manifest entry `targeted_retraining_replay`; primary records include the validation report, replay script, regression test, and indexed replay artifacts. |

Current inspection shortcuts:

- Public precise-reference OD slice rerun: use manifest entry
  `claim_to_artifact_map.real_slr_sp3_bounded_sanity_probes`.
  Primary artifacts are `results/validation/real_slr_sp3_od_slice_rerun.json`,
  `results/validation/real_slr_sp3_od_slice_rerun.md`, and the persistent
  rerun output under `results/validation/real_slr_sp3_od_slice_rerun/`. Confirm
  status `pass`, 10 completed arcs, zero public-claim summary mismatches, and
  generated table text matching the submitted table.

- Formal400 compact public SLR/SP3 OD replay: use manifest entry
  `claim_to_artifact_map.real_slr_sp3_bounded_sanity_probes`.
  Primary artifacts are
  `results/real_slr_sp3_od_formal400_inputs/real_slr_sp3_od_formal400_validation.json`,
  `results/validation/real_slr_sp3_od_formal400_run.log`,
  `results/validation/real_slr_sp3_od_formal400_run.err.log`,
  `paper/tables/real_slr_sp3_od_expanded.tex`,
  `paper/tables/real_slr_sp3_od_expanded_stratification.tex`,
  `paper/tables/real_slr_sp3_od_expanded_mechanism_heterogeneity.tex`,
  `scripts/run_real_slr_sp3_od_expanded_validation.py`, and
  `tests/test_real_slr_sp3_od_expanded.py`. Archived inputs are under
  `results/real_slr_sp3_od_formal400_inputs/`. Confirm 400 attempted arcs,
  373 completed arcs, and 27 explicit non-completed records (17 insufficient
  observations; 10 unavailable/non-parseable public products) with no hidden
  or imputed arcs; pooled means EKF 566.60 m, fixed-noise UKF 573.24 m, AUKF
  601.38 m, SP3-IC propagation 779.45 m; DBAR 256/373 (0.686) below the
  0.737 no-fire majority baseline; EKF-minus-AUKF mean -34.78 m with CI
  [-71.84,-2.76] and fixed-noise-UKF-minus-AUKF mean -28.14 m with CI
  [-61.80,-2.75]; and the boundary that this is compact diagnostic replay
  rather than operational POD or a positive AUKF/DBAR result. Formal210 and
  expanded80 are earlier/superseded archival evidence, not the current primary
  route.

- Temporal public real-measurement OD probe: use manifest entry
  `claim_to_artifact_map.public_real_measurement_temporal_od_probe`.
  Primary artifacts are
  `results/real_slr_sp3_temporal_od_campaign/real_slr_sp3_temporal_od_campaign.json`
  and `paper/tables/real_slr_sp3_temporal_od_campaign.tex`, with the
  selection-stability result/table linked in the same manifest entry.
- Prospective 260523 public-week full-correction readout: use manifest entry
  `claim_to_artifact_map.public_real_measurement_temporal_od_probe`.
  Primary artifacts are
  `results/real_slr_sp3_temporal_corrected_od_prospective_260523/real_slr_sp3_temporal_corrected_od_prospective_260523.json`,
  `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260523_20260526.json`,
  and `paper/tables/real_slr_sp3_temporal_corrected_od_campaign_summary.tex`.
  Confirm the learned-minus-best-recursive-classical gap of -19.84 m has
  CI [-145.49, 87.75] and learned lower on 4/10 arcs; this is indeterminate
  public-week transfer evidence, not operational POD or simulator validation.
- Pending prospective 260620 public-week full-correction readout: use manifest
  entry
  `claim_to_artifact_map.pending_public_real_measurement_temporal_od_prospective_260620`.
  Primary artifacts are
  `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json`,
  `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.sha256.txt`,
  `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json.ots`,
  `results/validation/public_precise_reference_status_20260617.json`,
  `results/validation/public_precise_reference_status_20260617.md`,
  `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_timestamp_attestation.json`,
  `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_timestamp_attestation.md`,
  and
  `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_availability_20260613.json`.
  Confirm this was created 2026-06-12 for test dates
  2026-06-15..2026-06-19 and is pending/not scored as of 2026-06-17.
  The OTS material is a Bitcoin-block-header-attested OpenTimestamps proof
  whose Merkle roots were cross-checked against the Blockstream API, but local
  Bitcoin-node verification was not performed. It is not scored validation, not
  DOI/public archive, not independent reproduction, and not operational POD.
  The retained 2026-06-13 availability audit is superseded for current
  availability by the 2026-06-17 status, which records no 260613/260620
  parent-listing entries for either LAGEOS target and direct 260613/260620 NSGF
  URLs returning HTTP 200 text/html; charset=utf-8 responses of length 825, not
  valid .sp3.gz products. This cannot be used yet as validation and must not be
  represented as scored validation.
- Pending prospective 260627 public-week readiness record: use manifest entry
  `claim_to_artifact_map.pending_public_real_measurement_temporal_od_prospective_260627`.
  Primary artifacts are
  `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json`,
  `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.sha256.txt`,
  `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json.ots`,
  `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260627_predeclaration_status_20260617.json`,
  and
  `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260627_predeclaration_status_20260617.md`.
  Confirm this was created 2026-06-17 before test dates
  2026-06-22..2026-06-26, uses validation week 260620 and test week 260627,
  and remains pending/not scored with no campaign run. The OTS proof is
  calendar-pending and not Bitcoin-block-header-attested; it is not scored
  validation, not DOI/public archive, not independent reproduction, and not
  operational POD. This record cannot be used as validation until
  validation/test public SP3 products become valid gzip/SP3 products and the
  frozen rule is scored once.
- Temporal OD selection-stability audit: use manifest entry
  `claim_to_artifact_map.public_temporal_od_selection_stability_audit`.
  Primary artifacts are
  `results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json`
  and `paper/tables/real_slr_sp3_temporal_selection_stability.tex`, with the
  listed run script and test recorded in the same manifest entry.
- Active main-manuscript table regeneration: use manifest entry
  `claim_to_artifact_map.active_main_manuscript_table_regeneration`.
  Primary artifacts are `results/validation/active_manuscript_regeneration.json`,
  `results/validation/active_manuscript_regeneration.md`,
  `results/validation/command_manifest.json`, and the active generated
  `paper/tables/*.tex` files included by `paper/main.tex`. Confirm the current
  parse has 9 active generated table inputs, 0 inline tables, and 1 external
  figure include, including `paper/tables/main_findings_summary.tex`.
- Archive-extracted reproduction tier: use manifest entry
  `claim_to_artifact_map.archive_extracted_reproduction_tier`.
  Primary artifacts are `results/validation/archive_extracted_reproduction.json`,
  `results/validation/archive_extracted_reproduction.md`, and
  `scripts/verify_archive_extracted_reproduction.py`. Confirm the added
  `archive_extracted_public_od_slice_rerun` block reports status `pass`, 10
  completed arcs, DBAR 6/10, zero public-claim summary mismatches, and table
  text matching the extracted submitted table.
- Public multi-target SP3/CRD breadth probe: use manifest entry
  `claim_to_artifact_map.public_multi_target_sp3_crd_breadth_probe`.
  It covers 10 SP3-scored targets, 40/40 target-weeks, 240 fixed start epochs,
  and 198/200 CRD target-days. Higher-fidelity-minus-compact improvement is
  24.25 m overall with fixed-start 95% interval [5.29, 44.40], target-week
  clustered interval [-0.99, 51.81], and target-cluster interval [-2.94, 57.66].
  It is state-scoring/coverage breadth only, not multi-target OD validation.
- Targeted retraining replay: use manifest entry
  `claim_to_artifact_map.targeted_retraining_replay`. Confirm status `pass`,
  finite objective histories, checkpoint hashes, and unchanged canonical
  checkpoint digest.

The 40-arc ILRS/SP3 correction-corpus audit recomputes correction and
provenance quantities only. It does not recompute recursive filter estimates,
and it is not operational POD or external validation of the simulator
learned-versus-classical conclusion.

The temporal OD probe is indexed as a bounded public real-measurement check:
train weeks 260418 and 260425 fit the learned calibrator, week 260502 selects
the ridge (`1e+06`) and candidate, and week 260509 scores the frozen choice. It
is a negative held-out learned-calibration result (learned calibrated
higher-fidelity UKF 411.51 m versus compact UKF 368.64 m on test; +42.87 m,
95% CI [23.69, 63.38]; 0/10 test arcs better for the learned candidate), not central external
validation or operational POD.

The selection-stability audit records why the validation choice is not a
learned-OD positive: the learned validation margin is 0.43 m, the paired
learned-minus-classical validation gap is -0.43 m with 95% CI [-7.85, 8.03],
validation bootstraps select learned in 62.4% of resamples, and leave-one-arc
validation folds select learned in 6/10 folds. The archived corpus remains
LAGEOS-1/-2 only; test-week station-holdout support is YARL 7 arcs, HERL 3,
MATM 0, and WETL 0.

The public multi-target SP3/CRD breadth probe is indexed separately from the
LAGEOS-only temporal OD probe. It records 10 SP3-scored targets, 40/40
target-weeks, 240 fixed start epochs, 198/200 CRD target-days, 25,449 normal
points, and 29 stations. The bounded readout is state-scoring and public-data
coverage breadth only. The JSON records fixed-start, target-week clustered, and
target-cluster finite-probe intervals; these are not target-population or
operational uncertainty intervals.

## Evidence The Package Indexes

- Fixed data-generation settings and the benchmark protocol.
- The canonical 15-seed cohort: seeds **41-55**. The primary endpoint is
  observed-step position RMSE, checked on a fresh independent disjoint-seed
  endpoint-fixation draw, replicated in a larger simulator-bound frozen-rule
  draw with 32 realizations per scenario and 24 trajectories per realization,
  checked on a stress-only K=96 floor-power-scale replication, and estimated by
  the 15-seed observed-step cohort; all-step RMSE is the propagation-dominated
  reference and single-seed result tables are illustrative diagnostics only.
  Timestamp-only internal evidence records the K=96 rule fixed at
  2026-05-25T13:06:32Z before the archived K=96 evaluation-start timestamp at
  2026-05-25T13:12:43.6581323Z.
  The larger endpoint and powered stress replications are internal
  simulator-bound records, not external preregistration, real-data evidence, or
  operational validation.
- Trained-model records and evaluation outputs for the classical references
  (EKF, UKF, AUKF, offline robust batch WLS, offline robust fixed-interval
  smoother) and the learned residual variants.
- Public-catalog SGP4 replay results and the SatNOGS observation-replay
  failure diagnostic.
- Submission-validation and citation-validation records, with artifact
  checksums for integrity checking.
- Archive-extracted active table-regeneration and the bounded targeted
  retraining replay reports.

## Honest Scope

The package documents and checksums the evidence behind the manuscript's
bounded claims. It does not assert flight readiness, public availability, or
any result beyond the reported tables and figures.
