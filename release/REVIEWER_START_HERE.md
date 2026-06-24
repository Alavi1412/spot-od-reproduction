# Reviewer Start Here

This page is the shortest route from the paper's headline statements to the
versioned evidence package. It is release-facing documentation for inspection
of the supplied archive; it is not part of the manuscript text.

## Public Release Status

Release-facing metadata is prepared for the bounded reproduction-support
package:

- Short title: `SPOT-OD v1.2.3 ACF holdout audit release`
- Zenodo record: <https://zenodo.org/records/20825138>
- DOI: `10.5281/zenodo.20825138`
- DOI URL: <https://doi.org/10.5281/zenodo.20825138>
- Zenodo concept DOI: `10.5281/zenodo.20768672`
- Zenodo status: `published`
- Zenodo resource type: `Software`
- Version: `1.2.3-acf-holdout-audit`
- License: `CC-BY-4.0`
- GitHub release:
  <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.3-acf-holdout-audit>
- Release tag: `v1.2.3-acf-holdout-audit`
- Release commit: `39e879d8665e489266bbf75f69634cab0e797fe8`
- Zenodo archived file:
  `Alavi1412/spot-od-reproduction-v1.2.3-acf-holdout-audit.zip`
- Zenodo archived file bytes: `187,254,529`
- Zenodo archived file MD5: `7eb8b43a9af90a4783482a7a3a086f92`
- GitHub release asset: `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`
- GitHub release asset bytes: `59,140,917`
- GitHub release asset SHA-256:
  `11909866b2ae1a375cdebe1472305d6d1fbd0b9f97453084fc7da16b78dcf70f`
- GitHub Actions verifier branch run:
  <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253565>
  (`success`)
- GitHub Actions verifier tag run:
  <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253538>
  (`success`)

Prior public v1.2.2 release history:

- Zenodo record: <https://zenodo.org/records/20822968>
- DOI: `10.5281/zenodo.20822968`
- DOI URL: <https://doi.org/10.5281/zenodo.20822968>
- Zenodo status: `published`
- GitHub release:
  <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.2-acf-audit>
- Release tag: `v1.2.2-acf-audit`
- Release commit: `6fbc88745b6d96939736d59731089e99786c1f8c`
- GitHub release asset: `spot_od_v1_2_2_acf_audit_review_archive.zip`

Prior public v1.2.1 release history:

- Zenodo record: <https://zenodo.org/records/20811701>
- DOI: `10.5281/zenodo.20811701`
- DOI URL: <https://doi.org/10.5281/zenodo.20811701>
- Zenodo status: `published`
- Version: `1.2.1-graph-anchor-gate-poc`
- GitHub release:
  <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc>
- Release tag: `v1.2.1-graph-anchor-gate-poc`
- Release commit: `2dcd542dcb72f1622dfaf1cf8981a550862312bf`
- Zenodo archived file:
  `Alavi1412/spot-od-reproduction-v1.2.1-graph-anchor-gate-poc.zip`
- Zenodo archived file bytes: `94,265,950`
- Zenodo archived file MD5: `233d2fc7fce1bc57afdd66332a3a7dc1`
- GitHub release asset: `spot_od_v1_2_1_graph_anchor_gate_poc.zip`
- GitHub release asset bytes: `17,710,047`
- GitHub release asset SHA-256:
  `3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c`
- Release-triggered GitHub Actions verifier:
  <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357>
  (`success`; ran the archive-extracted reproduction workflow and graph
  verifier on GitHub-hosted runners)
- Local post-release public-clean-clone maintainer evidence:
  `results/validation/public_clean_clone_v121_reproduction_20260623.json` and
  `.md` (`2026-06-23`; clean public clone from
  `https://github.com/Alavi1412/spot-od-reproduction.git`, tag
  `v1.2.1-graph-anchor-gate-poc`, detached HEAD commit
  `2dcd542dcb72f1622dfaf1cf8981a550862312bf`; both public verifiers passed).
  This record was created after v1.2.1 publication and is available in the
  current workspace/submission packet once included; it is not part of the
  published v1.2.1 Zenodo/GitHub release unless a later release includes it.

Version 1.2.3 supersedes v1.2.2 only by repairing the public release boundary
for the ACF audit/table tier and packaging the development/holdout split
summaries in the new release. Version 1.2.2 remains historical ACF audit package
history, and version 1.2.1 remains historical GraphAnchorPairGate package
history. Scientific metrics are as recorded, not upgraded to operational
validation.

This release satisfies the access/integrity deposition tier for archive
extraction, manifest hashes, active manuscript artifact regeneration, and one
archived-input public OD slice rerun, plus public packaging of the
GraphAnchorPairGate PoC and the ACF audit/table tier including the
development/holdout split summaries. It is not full
raw/training/all-filter reproduction, live public-data retrieval, operational
POD validation, independent-machine confirmation, third-party independent
validation, or full scientific reproduction.

The public `1.2.1-graph-anchor-gate-poc` release includes a post-manuscript
GraphAnchorPairGate proof of concept at
`results/graph_anchor_pair_gate_seed_sweep_20260623/`. It is a
GNN-based station-time graph message-passing plus GRU gate over
`RFIS:VA_RFIS`, using no-truth anchor features, and reports all-step
center-window position RMSE on held-out eval trajectories within
`process_noise_shift_test` and `maneuver_shift_test`. Across five local paired
seeds it records 9/10 scenario-seed row wins and 4/5 paired seeds winning both
scenarios; process shift is 4/5 wins with mean gain 7.95663495038935%, and
maneuver shift is 5/5 wins with mean gain 8.05274642630686%. The failure row is
seed 19 process shift: -2.0925251807980216% (5116.480181866038 m versus
`VA_RFIS` 5011.610960552836 m). The all-row aggregate figure is
`graph_anchor_pair_gate_seed_sweep_aggregate.png`; the paired seed gains and
uncertainty readout are
`graph_anchor_pair_gate_seed_sweep_paired_seed_gains.csv`,
`graph_anchor_pair_gate_seed_sweep_uncertainty_summary.csv`, and
`graph_anchor_pair_gate_seed_sweep_statistical_summary.md`. The descriptive
uncertainty summary records row wins 9/10 with Wilson 95% CI [0.596, 0.982]
and exact one-sided sign/binomial p=0.0107, plus paired both-scenario wins 4/5
with CI [0.376, 0.964] and p=0.1875. It is public archive evidence for this
bounded PoC package only: not an operational precise-reference validation, not
independent-machine reproduction, not third-party reproduction, not a full
raw/training/all-filter rerun, and not a replacement for the frozen
observed-step endpoint hierarchy.

A local current-workspace post-release retained-output audit of the same
GraphAnchorPairGate sweep is generated by
`scripts/build_graph_anchor_pair_gate_observed_step_audit.py` and writes
`results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.csv`,
`.json`, and `.md`. It reads retained per-scenario JSON files only and does not
rerun training/evaluation. The audit records 0/10 observed-step row wins and
0/5 paired seeds winning both scenarios, so the all-step graph benefit does NOT
transfer to observed-step primary endpoint superiority. This local audit is not
part of the published v1.2.1 Zenodo/GitHub release unless a later release
includes it, and it is not operational precise-reference validation, not
independent-machine reproduction, not third-party validation, and not a
universal learned-method claim.

The v1.2.3 ACF audit tier includes the current-workspace
AdaptiveCandidateFusion fixed-soft full-training campaign package generated by
`scripts/build_adaptive_candidate_fusion_fixed_soft_training_campaign_artifacts.py`
and writes
`results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv`,
`.json`, and `.md`. It reads centered and observed-mask full-training campaign
summaries, validates fixed-soft rows (`requested_inference_mode=soft`,
`selected_inference_mode=soft`, `inference_mode_selection_source=cli_fixed`),
validates campaign metadata, and treats non-empty train/validation loss
histories plus best/last checkpoint files as evidence that these are full
training runs rather than skip-training replays. Centered training
(`training_step_mask=centered`, `validation_selection_metric=all_step_pos_rmse_m`)
reproduces the observed-step pocket: 8/10 row wins, 3/5 paired seeds winning
both shift scenarios, mean gain +2.5944434632433686%, min -11.879589616936398%,
and max +12.02043576038026%; all-step remains a caveat at 5/10 row wins, 0/5
paired seeds, and mean gain -10.853657642974031%. Observed-mask full retraining
(`training_step_mask=observed`, `validation_selection_metric=observed_step_pos_rmse_m`)
is a bounded negative/failure mode: 12/20 observed-step row wins, 3/10 paired
seeds, mean gain -6.1915386036276825%, min -164.7209996196364%, and max
+13.302773387058087%; all-step is also negative overall at 11/20 row wins,
1/10 paired seeds, and mean gain -11.99436443792321%. This current-workspace
compact-simulator package shows centered-training observed-step gain and an
observed-mask failure mode, not broad learned-superiority. The compact
manuscript table is `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`.
The same table also includes the 15-seed validation-selected global scenario
portfolio at
`results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json`,
`.md`, and `.csv`, generated by
`scripts/analyze_adaptive_candidate_fusion_global_portfolio.py`. Its validation
selector chooses `0.65*learned + 0.35*RFIS` for process and
`0.55*learned + 0.45*EKF` for maneuver, then applies those policies to held-out
compact-simulator eval rows with no test-row policy tuning. It records 25/30
observed-step row wins, 13/15 paired-seed wins, mean gain +3.793410580996871%,
min -10.11448514448152%, and max +12.384536098768079%; the nonlearned-only
validation-selected blend baseline is weaker at 19/30 wins and mean gain
+0.7140145823400381%. The development/holdout split summaries at
`results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/`
are also indexed in v1.2.3; they give weak/mixed internal holdout evidence
(7/10 row wins, mean +1.45%, row CI [-2.22,+4.96], seed-paired 3/5 wins).
These artifacts are indexed by the supplementary manifest and included in
v1.2.3 as the ACF audit/table tier. They remain
validation-selected compact-simulator PoC evidence, not independent-machine
reproduction, not operational precise-reference validation, not third-party
validation, not a full raw/training/all-filter reproduction, not confirmatory
learned superiority, and not a universal learned-OD claim.

The public-clean-clone record listed above strengthens inspectability and
public-clone reproducibility only. It is NOT independent third-party validation
and is NOT independent-machine confirmation if independent machine means a host
operated by an external party. It is a local post-release record, not part of
the published v1.2.1 Zenodo/GitHub release unless a later release includes it.

## One-Pass Minimum Check

For a no-retraining integrity check, inspect these release records:

1. `SUPPLEMENTARY_MANIFEST.json`
2. `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`
3. `results/validation/minimum_tier_reproduction_check.json`
4. `results/validation/minimum_tier_reproduction_check.md`

The minimum tier checks that the manifest entries are present, that recorded
SHA-256 values match the files, that the review archive contains the indexed
members with matching digests, and that every claim-map record is covered by
the manifest or by an allowed top-level release artifact. It does not retrain
models, regenerate estimates, or rebuild tables.

### Manifest And Attestation Counts

`SUPPLEMENTARY_MANIFEST.json` is authoritative for final submitted-package
artifact counts and `review_archive.sha256` / byte count. Read the current
digest and counts directly from that manifest, not from this guide.
For the current DOI-synced branch, that manifest records a regenerated local
archive of 59,142,123 bytes with SHA-256
`11451c2032243c972534f7de9eb40ba04c44ff69b6c45db179f2053f97ad9b7e` and
`review_archive.matches_published_github_release_asset: false`. The immutable
published v1.2.3 GitHub release asset is a separate 59,140,917-byte artifact
with SHA-256
`11909866b2ae1a375cdebe1472305d6d1fbd0b9f97453084fc7da16b78dcf70f`; use it
only with the exact release/tag manifest set.
Archive-extracted reports intentionally omit the enclosing archive SHA-256 and
byte size because they are themselves archived, which would create a
self-referential digest cycle. Validation and attestation files are
run-boundary records; if a count or digest differs from the final manifest, use
the final manifest for the submitted package and treat the report count or
digest as the state used by that verifier run. The same verifier `pass` status
is not changed by that bookkeeping distinction.

For the next tier, inspect `results/validation/active_manuscript_regeneration.json`
and `.md`. That check regenerates the active generated tables and figure
include parsed from `paper/main.tex` from materialized result artifacts in a
temporary directory and compares the regenerated bytes with the submitted
artifacts.

For an archive-extracted tier, inspect
`results/validation/archive_extracted_reproduction.json` and `.md`. That check
extracts `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`, verifies extracted
manifest-indexed artifact digests, resolves claim-map and regeneration-tier
records, runs the active table-regeneration check from the extracted tree, and
reruns the bounded public LAGEOS CRD/SP3 OD slice from the extracted archive.
It is not a full scientific rerun or operational POD validation.

For the local post-release public-clean-clone maintainer evidence, inspect
`results/validation/public_clean_clone_v121_reproduction_20260623.json` and
`.md`. That record captures the 2026-06-23 clean public clone, the GitHub tag
warning, detached HEAD commit
`2dcd542dcb72f1622dfaf1cf8981a550862312bf`, clean status before verifier
output files, Python path, and passing public verifier commands. It is not
independent third-party validation, not independent-machine confirmation by an
external operator, not operational precise-reference validation, and not a full
raw-data/training rerun. It is available in the current workspace/submission
packet once included; it is not part of the published v1.2.1 Zenodo/GitHub
release unless a later release includes it.

For a same-workspace separate-agent clean extracted archive attestation, inspect
`results/validation/codex_clean_archive_attestation_20260614.json` and `.md`.
That record summarizes verifier commands run by a Codex implementation worker
against this workspace and the review archive. It is not independent-machine
validation, not third-party reproduction, not DOI/public archive, not full
scientific reproduction, not operational POD, and not independent external
reproduction.

For a local Docker Desktop clean-container staged minimum-tier integrity
attestation, inspect
`results/validation/containerized_minimum_tier_attestation_20260614.json` and
`.md`. That record summarizes a same-host staged `python:3.11` container run of
`scripts/verify_minimum_tier_reproduction.py --check-only` against the staged
manifest, review archive, and verifier script. Boundary: local Docker Desktop clean-container staged minimum-tier integrity attestation only, same-host and staged from the current workspace/review archive, not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, not independent external reproduction, and not a full scientific rerun.

For the public precise-reference OD slice rerun, inspect
`results/validation/real_slr_sp3_od_slice_rerun.json` and `.md`. That check
stages the archived public CRD/SP3 inputs into a validation directory, reruns
the LAGEOS OD slice without live retrieval, and compares the deterministic
claim summary and rebuilt table with the submitted records. It is one
slice-level rerun, not full scientific reproduction or operational POD
validation.

For a bounded representative retraining replay, inspect
`results/validation/targeted_retraining_replay_public.json` and `.md`. That report
records one deterministic-slice training replay, finite objective histories,
output digests, and unchanged canonical checkpoint digests. It is not a full
paper-table reproduction or full seed-suite rerun. Reviewer note: the
stress-focus stage has a positive but finite first validation epoch followed by
a finite second validation epoch; the replay claim is finite execution,
checkpoint production, and provenance, not performance or stability evidence.

For the full-rerun divergence reconciliation, inspect
`results/validation/full_rerun_divergence_audit_20260617.json` and `.md`.
That generated audit reads the retained full-rerun metrics, scorecard, and
trajectory-error files to localize divergence in the dense-visibility and
SatNOGS replay slices. It is diagnostic only, not a canonical table
replacement, not operational validation, not independent reproduction, and not
a rerun success upgrade.

For the official ILRS precise-reference availability gate, inspect
`results/validation/ilrs_precise_reference_availability_20260617.json` and
`.md`. That probe checks EDC/CDDIS URLs and local cached pending `.sp3.gz`
filenames using a gzip/SP3 content gate. It is not scored validation; cached
HTML placeholders remain unavailable/non-usable.

For independent-machine handoff, inspect
`release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`. It gives a third-party
operator exact clean-machine verification steps and a signed report template.
It is a request/template, not a completed independent reproduction.

## Regeneration Tiers

| Tier | Manifest key | Use |
|---|---|---|
| Minimum integrity check | `regeneration_tiers.minimum_integrity_check` | Verify manifest presence, SHA-256 checksums, archive membership, and claim-map coverage. |
| Archive-extracted reproduction check | `regeneration_tiers.archive_extracted_reproduction_check` | Extract the review archive, verify indexed member digests and claim/tier references, rerun active main-manuscript table regeneration, and rerun one bounded public LAGEOS CRD/SP3 OD slice from the extracted tree. |
| Table-regeneration check | `regeneration_tiers.table_regeneration_check` | Rebuild active main-manuscript generated tables from materialized outputs and compare byte hashes with the submitted tables. |
| Public precise-reference OD slice rerun check | `regeneration_tiers.public_precise_reference_od_slice_rerun_check` | Rerun one archived public LAGEOS CRD/SP3 OD slice through filter recomputation and table reconstruction, then compare with the submitted summary/table. |
| Targeted retraining replay check | `regeneration_tiers.targeted_retraining_replay_check` | Rerun one bounded representative learned-estimator training replay on deterministic slices of existing materialized data. |
| Full rerun | `regeneration_tiers.full_rerun` | Regenerate data, train learned estimators, evaluate all methods, rebuild paper artifacts, and rerun checks. |

## Headline Claim Map

Use the `Manifest entry` column to jump directly to
`SUPPLEMENTARY_MANIFEST.json::claim_to_artifact_map.<key>`. The `Tiers` column
uses the keys above and records the lowest inspection tier plus the stronger
tiers that can be used for deeper regeneration.

| Headline statement | Manifest entry | Tiers |
|---|---|---|
| No originally audited residual/GNN/KalmanNet-style learned construction beats the per-scenario best classical reference on the primary observed-step endpoint. | `audited_learned_family_bounded_negative` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| Public GraphAnchorPairGate PoC package records 9/10 scenario-seed row wins and 4/5 paired seeds winning both shift scenarios on all-step center-window RMSE for held-out eval trajectories, with one seed-19 process-shift failure and descriptive small-sample uncertainty. | Public `1.2.1-graph-anchor-gate-poc` release: Zenodo DOI `10.5281/zenodo.20811701`; GitHub release <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc>; commit `2dcd542dcb72f1622dfaf1cf8981a550862312bf`; successful release run <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357>. | Public archive/package inspection only; outside the primary observed-step endpoint hierarchy and not operational precise-reference validation, independent-machine reproduction, third-party reproduction, full raw/training/all-filter rerun, or a universal claim |
| Local GraphAnchorPairGate retained-output observed-step audit records 0/10 observed-step row wins and 0/5 paired both-scenario wins, so the all-step graph benefit must not be treated as primary-endpoint superiority. | Local current-workspace records `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.csv`, `.json`, and `.md`; generated by `scripts/build_graph_anchor_pair_gate_observed_step_audit.py`; not a manifest entry unless a later packet/release indexes it. | Retained-output post-release/current-workspace audit only; does not rerun training/evaluation, not part of the published v1.2.1 release unless later included, not operational precise-reference validation, not independent-machine reproduction, not third-party validation, and not a universal learned-method claim |
| AdaptiveCandidateFusion audit/table tier records include a 15-seed validation-selected global scenario portfolio (25/30 row wins, 13/15 paired wins, mean +3.793410580996871%), fixed-soft diagnostics, and a weak/mixed development/holdout split (7/10 row wins, mean +1.45%, row CI [-2.22,+4.96], seed-paired 3/5 wins). | Included in v1.2.3 as manifest-indexed records: `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json`, `.md`, and `.csv`; `results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/summary.json`, `.md`, and `.csv`; `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv`, `.json`, and `.md`; compact manuscript table `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`; generated by `scripts/analyze_adaptive_candidate_fusion_global_portfolio.py` and `scripts/build_adaptive_candidate_fusion_fixed_soft_training_campaign_artifacts.py`, with generator/test support in `scripts/build_paper_assets.py` and focused ACF tests. | Validation-selected compact-simulator PoC evidence; global policies are validation-selected (`0.65*learned + 0.35*RFIS` process, `0.55*learned + 0.45*EKF` maneuver) with no test-row policy tuning, the nonlearned-only validation-selected blend baseline is weaker at 19/30 wins and +0.7140145823400381% mean, and the development/holdout split is weak/mixed internal evidence; not operational precise-reference validation, not independent-machine reproduction, not third-party validation, not a full raw/training/all-filter reproduction, not confirmatory learned superiority, and not a universal learned-OD claim |
| The local post-release maintainer public-clean-clone run confirms both public verifiers pass from a clean GitHub clone of tag `v1.2.1-graph-anchor-gate-poc` on the current host. | Local post-release records `results/validation/public_clean_clone_v121_reproduction_20260623.json` and `.md`; not a manifest entry unless a later packet/release indexes it. | Bounded public-clean-clone maintainer evidence only; available in the current workspace/submission packet once included, not part of the published v1.2.1 Zenodo/GitHub release unless a later release includes it; strengthens inspectability and public-clone reproducibility, but is not independent third-party validation, independent-machine confirmation by an external operator, operational precise-reference validation, or full raw-data/training rerun |
| The larger K=32 independent endpoint replication under the frozen K=32 decision rule and established observed-step hierarchy is the central simulator anchor. | `larger_simulator_bound_endpoint_replication` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| The stress-row negative is checked in a separate K=96 stress-only replication exceeding the $K\approx94$ floor-power design check; timestamp-only internal evidence records the K=96 rule fixed at 2026-05-25T13:06:32Z before the archived K=96 evaluation-start timestamp at 2026-05-25T13:12:43.6581323Z. | `powered_stress_floor_scale_replication` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| A later all-scenario K=96 internal replication keeps all three observed-step scenarios under the same frozen decision predicate and fresh seed; it is internal simulator-bound evidence, not external validation. | `all_scenario_k96_internal_replication` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| The endpoint-choice sensitivity audit reuses stored observed-step intervals and labels all-step intervals as sensitivity recomputations. | `endpoint_choice_sensitivity_audit` | `minimum_integrity_check`, `table_regeneration_check` |
| The PUKF tuning-comparability sensitivity remains negative after deterministic held-out row-order alignment checks. | `pukf_tuning_comparability_sensitivity` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| Tuned classical guardrails and the offline OD reference prevent over-crediting of weak learned positives; the force-mismatch WLS row is an explicit comparator boundary, not a learned-positive result. | `classical_guardrails_and_offline_od_reference` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| Each protocol ingredient blocks at least one misleading positive, unsupported upgrade, or interpretation error in this evidence set. | `protocol_subset_sufficiency` | `minimum_integrity_check`, `table_regeneration_check` |
| Noise-side and force-side structural-channel tests remain bounded negatives, with a favorable-geometry recoverability sanity check. | `structural_channel_bounded_negatives` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| The long-arc higher-fidelity slice scopes down the compact-model EKF/AUKF diagnostic and records decision stability. | `long_arc_scope_down_and_decision_stability` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| The DBAR heuristic is withdrawn as a claim after the adequately powered characterization. | `dbar_withdrawal` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| The KalmanNet official sanity benchmark and SPOT-OD transposition diagnostics are bounded evidence, not a broad KalmanNet refutation. | `kalmannet_native_and_transposition_diagnostics` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| Public ILRS/SP3 slices, including the formal400 compact replay that supersedes the earlier formal210 and 80-arc compact replay archives, are bounded measurement-pipeline and provenance probes, not operational POD or simulator-conclusion validation. | `real_slr_sp3_bounded_sanity_probes` | `minimum_integrity_check`, `table_regeneration_check`, `public_precise_reference_od_slice_rerun_check`, `full_rerun` |
| The public ILRS/SP3 state-scoring campaign is a bounded precise-reference probe with negative learned no-leakage checks. | `public_precise_reference_state_scoring_probe` | `minimum_integrity_check`, `table_regeneration_check`, `full_rerun` |
| The public multi-target SP3/CRD breadth probe extends state-scoring and coverage breadth only, not multi-target OD validation. | `public_multi_target_sp3_crd_breadth_probe` | `minimum_integrity_check`, `full_rerun` |
| The temporal public real-measurement OD probe is a bounded negative learned-calibration check with train/validation/test-week separation, not central external validation. | `public_real_measurement_temporal_od_probe` | `minimum_integrity_check`, `full_rerun` |
| The temporal public real-measurement OD validation selector is fragile under validation-only resampling and remains negative on the frozen test week. | `public_temporal_od_selection_stability_audit` | `minimum_integrity_check`, `full_rerun` |
| The review-stage archive is checksum-indexed, extractable, and inspectable without retraining. | `review_stage_archive_integrity` | `minimum_integrity_check`, `archive_extracted_reproduction_check` |
| Active generated main-manuscript tables are regenerated from materialized artifacts and byte-compared without retraining. | `active_main_manuscript_table_regeneration` | `table_regeneration_check`, `archive_extracted_reproduction_check` |
| The archive-extracted tier verifies extracted artifact digests, reruns active table regeneration, and reruns one bounded public OD slice from the extracted tree. | `archive_extracted_reproduction_tier` | `archive_extracted_reproduction_check` |
| A Codex-agent clean archive attestation records same-workspace verifier commands and archive-extracted outputs with an explicit non-independent boundary. | `codex_clean_archive_attestation` | `minimum_integrity_check`, `archive_extracted_reproduction_check` |
| A containerized minimum-tier attestation records a local Docker Desktop clean-container staged minimum-tier integrity attestation with an explicit same-host boundary. | `containerized_minimum_tier_attestation` | `minimum_integrity_check` |
| The official ILRS availability probe records whether pending LAGEOS SP3 products are valid gzip/SP3 files before any prospective scoring command is run. | `official_ilrs_precise_reference_availability_probe` | `minimum_integrity_check` |
| The independent-machine reproduction request gives an external operator exact verifier commands and a report template but records no completed independent reproduction. | `independent_machine_reproduction_request` | `minimum_integrity_check` |
| The targeted retraining replay is a bounded representative replay on deterministic slices and leaves canonical submitted checkpoints unchanged. | `targeted_retraining_replay` | `minimum_integrity_check`, `targeted_retraining_replay_check` |
| The full-rerun divergence audit reconciles retained dense-visibility and SatNOGS replay divergence flags without rerunning models or changing decisions. | `full_rerun_divergence_audit` | `minimum_integrity_check`, `full_rerun` |

## Targeted Claim Shortcuts

| Claim | Manifest entry | Primary artifacts | Quick check |
|---|---|---|---|
| Public GraphAnchorPairGate PoC package | Public `1.2.1-graph-anchor-gate-poc` release; Zenodo record <https://zenodo.org/records/20811701>; DOI `10.5281/zenodo.20811701`; Zenodo archived file `Alavi1412/spot-od-reproduction-v1.2.1-graph-anchor-gate-poc.zip` (94,265,950 bytes; MD5 `233d2fc7fce1bc57afdd66332a3a7dc1`); GitHub release asset `spot_od_v1_2_1_graph_anchor_gate_poc.zip` (17,710,047 bytes; SHA-256 `3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c`) | `results/graph_anchor_pair_gate_seed_sweep_20260623/EVIDENCE_NOTE.md`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_summary.csv`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_by_scenario.csv`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_paired_seed_gains.csv`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_uncertainty_summary.csv`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_statistical_summary.md`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_seed_sweep_aggregate.png`; `results/graph_anchor_pair_gate_seed_sweep_20260623/seed_7_split_7/graph_anchor_pair_gate_candidate_comparison.png`; `results/graph_anchor_pair_gate_rfis_va_gpu_holdout_shift_all_candidates_seed7/graph_anchor_pair_gate_summary.csv` | Confirm the method boundary: GNN-based station-time graph message passing plus GRU gate over `RFIS:VA_RFIS`, using no-truth anchor features; trained on train+validation roles; evaluated on held-out eval trajectories within `process_noise_shift_test` and `maneuver_shift_test`; metric is all-step center-window position RMSE. Confirm seed 7 is illustrative and the aggregate figure covers all 10 scenario-seed rows. Confirm aggregate robustness: 9/10 row wins with Wilson 95% CI [0.596, 0.982] and exact one-sided sign/binomial p=0.0107; 4/5 paired seeds winning both scenarios with CI [0.376, 0.964] and p=0.1875; process 4/5 wins with mean gain 7.95663495038935%; maneuver 5/5 wins with mean gain 8.05274642630686%; paired seed-level mean gains +14.47%, +6.08%, +1.73%, +12.97%, and +4.77%; and seed-19 process failure -2.0925251807980216% (5116.480181866038 m vs `VA_RFIS` 5011.610960552836 m). Boundary: public archive/package evidence for this bounded PoC only; not an operational precise-reference validation, not independent-machine reproduction, not third-party reproduction, not a full raw/training/all-filter rerun, and not a frozen observed-step endpoint replacement. |
| Local GraphAnchorPairGate observed-step retained-output audit | Local current-workspace records; not a manifest entry unless a later packet/release indexes them; not part of the published v1.2.1 Zenodo/GitHub release unless later included. | `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.csv`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.json`; `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.md`; `scripts/build_graph_anchor_pair_gate_observed_step_audit.py`; `tests/test_graph_anchor_pair_gate_observed_step_audit.py` | Confirm the audit reads retained per-scenario JSONs only, validates required methods and `observed_step_pos_rmse_m`, and records 0/10 observed-step row wins, 0/5 paired both-scenario wins, mean gain -5.568334199477706%, min -12.482167132357489%, and max -0.8712975503737664%. Boundary: retained-output audit only, does not rerun training/evaluation, confirms the GraphAnchorPairGate all-step benefit does NOT transfer to observed-step primary endpoint superiority, not operational precise-reference validation, not independent-machine reproduction, not third-party validation, and not a universal learned-method claim. |
| AdaptiveCandidateFusion fixed-soft full-training, global scenario portfolio, and development/holdout split | Included in v1.2.3 as manifest-indexed compact-simulator PoC records. | `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`; `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json`; `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.md`; `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.csv`; `results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/summary.json`; `results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/summary.md`; `results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/summary.csv`; `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv`; `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.json`; `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.md`; `scripts/build_paper_assets.py`; `scripts/analyze_adaptive_candidate_fusion_global_portfolio.py`; `scripts/build_adaptive_candidate_fusion_fixed_soft_training_campaign_artifacts.py`; `tests/test_adaptive_candidate_fusion_fixed_soft_training_campaign_artifacts.py` | Confirm `build_adaptive_candidate_fusion_full_training_poc_table()` reads the source summaries. Fixed-soft: centered observed-step 8/10 row wins, 3/5 paired, mean +2.5944434632433686%, min -11.879589616936398%, max +12.02043576038026%; observed-mask 12/20 row wins, 3/10 paired, mean -6.1915386036276825%, min -164.7209996196364%, max +13.302773387058087%; all-step remains a caveat/negative overall. Global portfolio: validation-selected process policy `0.65*learned + 0.35*RFIS`, maneuver policy `0.55*learned + 0.45*EKF`, 25/30 row wins, 13/15 paired-seed wins, mean +3.793410580996871%, min -10.11448514448152%, max +12.384536098768079%; nonlearned-only validation-selected blend baseline 19/30 wins and +0.7140145823400381% mean. Development/holdout split: selected `0.60*learned + 0.40*RFIS` for process and `0.60*learned + 0.40*EKF` for maneuver, then produced weak/mixed holdout results (7/10 row wins, mean +1.45%, row CI [-2.22,+4.96], seed-paired 3/5 wins). Boundary: validation-selected compact-simulator PoC only, not operational precise-reference validation, not independent-machine reproduction, not third-party validation, not a full raw/training/all-filter reproduction, not confirmatory learned superiority, and not a universal learned-OD claim. |
| Local post-release public-clean-clone maintainer evidence | Local records; not a manifest entry unless a later packet/release indexes them | `results/validation/public_clean_clone_v121_reproduction_20260623.json`; `results/validation/public_clean_clone_v121_reproduction_20260623.md` | Confirm date 2026-06-23; source clone `https://github.com/Alavi1412/spot-od-reproduction.git`; tag warning `refs/tags/v1.2.1-graph-anchor-gate-poc a9842e8d76b41f5227695ef6e3a5532668c7c0e8 is not a commit`; detached HEAD commit `2dcd542dcb72f1622dfaf1cf8981a550862312bf`; clean status before verifier outputs; Python path `C:\Users\alavi\AppData\Local\Microsoft\WindowsApps\python.exe`; graph verifier archive bytes 17,710,047 and SHA-256 `3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c`; and both verifier statuses pass. This record is available in the current workspace/submission packet once included and is not part of the published v1.2.1 Zenodo/GitHub release unless a later release includes it. Boundary: not independent third-party validation, not independent-machine confirmation by an external operator, not operational precise-reference validation, and not a full raw-data/training rerun. |
| Public precise-reference OD slice rerun | `claim_to_artifact_map.real_slr_sp3_bounded_sanity_probes` | `results/validation/real_slr_sp3_od_slice_rerun.json`; `results/validation/real_slr_sp3_od_slice_rerun.md`; `results/validation/real_slr_sp3_od_slice_rerun/real_slr_sp3_od_validation.json`; `results/validation/real_slr_sp3_od_slice_rerun/real_slr_sp3_od.tex`; `scripts/run_real_slr_sp3_od_slice_rerun_validation.py`; `tests/test_real_slr_sp3_od_slice_rerun.py` | Confirm status `pass`, 10 completed arcs, zero public-claim summary mismatches, DBAR uses the real adaptive-vs-fixed innovation statistic against an external SP3 counterproductivity label and reports 6/10 correct below the 80% no-fire baseline, compact-filter readout reports AUKF 6/10, EKF 3/10, fixed-noise UKF 1/10, EKF-minus-AUKF CI spans zero, simple CI-width scaling implies roughly 80 comparable arcs for same-effect zero exclusion and approximately 185 arcs for formal 80% power using the bootstrap-implied standard deviation, fixed-noise UKF remains best by pooled mean, and generated table text matches the submitted table. |
| Formal400 compact public SLR/SP3 OD replay | `claim_to_artifact_map.real_slr_sp3_bounded_sanity_probes` | `results/real_slr_sp3_od_formal400_inputs/real_slr_sp3_od_formal400_validation.json`; `results/validation/real_slr_sp3_od_formal400_run.log`; `results/validation/real_slr_sp3_od_formal400_run.err.log`; `paper/tables/real_slr_sp3_od_expanded.tex`; `paper/tables/real_slr_sp3_od_expanded_stratification.tex`; `paper/tables/real_slr_sp3_od_expanded_mechanism_heterogeneity.tex`; `scripts/run_real_slr_sp3_od_expanded_validation.py`; `tests/test_real_slr_sp3_od_expanded.py`; archived CRD/SP3 inputs under `results/real_slr_sp3_od_formal400_inputs/`; earlier/superseded formal210 archive under `results/real_slr_sp3_od_formal210_inputs/` | Confirm status `partial_completed`, 400 attempted arcs, 373 completed arcs, and 27 explicit non-completed records (17 insufficient observations, 10 unavailable/non-parseable public products) with no hidden/imputed arcs; pooled means EKF 566.60 m, fixed-noise UKF 573.24 m, AUKF 601.38 m, SP3-IC propagation 779.45 m; DBAR 256/373 = 0.686 below the 0.737 no-fire baseline; EKF-minus-AUKF mean -34.78 m with CI [-71.84,-2.76], and fixed-noise-UKF-minus-AUKF mean -28.14 m with CI [-61.80,-2.75]; and the boundary that this is compact diagnostic replay rather than operational POD or a positive AUKF/DBAR result. Confirm completed-only preceding/train/validation/test, LAGEOS-1/LAGEOS-2 strata, and mechanism heterogeneity strata are diagnostic only and do not establish causal attribution or stable filter superiority. Formal210 is earlier/superseded archival evidence, not the current primary route. |
| Temporal public real-measurement OD probe | `claim_to_artifact_map.public_real_measurement_temporal_od_probe` | `results/real_slr_sp3_temporal_od_campaign/real_slr_sp3_temporal_od_campaign.json`; `paper/tables/real_slr_sp3_temporal_od_campaign.tex`; `results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json`; `paper/tables/real_slr_sp3_temporal_selection_stability.tex`; `scripts/run_real_slr_sp3_temporal_od_campaign.py`; `tests/test_real_slr_sp3_temporal_od_campaign.py` | Confirm the train weeks 260418/260425, validation week 260502, selected ridge `1e+06`, test week 260509, and negative held-out learned result. |
| Prospective 260523 public-week full-correction readout | `claim_to_artifact_map.public_real_measurement_temporal_od_probe` | `results/real_slr_sp3_temporal_corrected_od_prospective_260523/real_slr_sp3_temporal_corrected_od_prospective_260523.json`; `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260523_20260526.json`; `paper/tables/real_slr_sp3_temporal_corrected_od_campaign_summary.tex`; `scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py`; `tests/test_real_slr_sp3_temporal_corrected_od_campaign.py` | Confirm 10/10 completed arcs, learned residual 365.45 m versus 385.29 m best recursive classical, learned-minus-best-recursive-classical gap -19.84 m with CI [-145.49, 87.75], learned lower on 4/10 arcs, and the indeterminate/provenance-only boundary. |
| Pending prospective 260620 public-week full-correction readout | `claim_to_artifact_map.pending_public_real_measurement_temporal_od_prospective_260620` | `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json`; `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.sha256.txt`; `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json.ots`; `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_availability_20260613.json`; `results/validation/public_precise_reference_status_20260617.json`; `results/validation/public_precise_reference_status_20260617.md`; `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_timestamp_attestation.json`; `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_timestamp_attestation.md` | Confirm this was created 2026-06-12 for test dates 2026-06-15..2026-06-19 and is pending/not scored as of 2026-06-17. The OTS material is a Bitcoin-block-header-attested OpenTimestamps proof with Merkle roots cross-checked against the Blockstream API, but local Bitcoin-node verification was not performed and it is not scored validation, not DOI/public archive, not independent reproduction, and not operational POD. The retained 2026-06-13 availability audit is superseded for current availability by the 2026-06-17 status, which records parent listings with no 260613/260620 directories and direct 260613/260620 NSGF URLs returning HTTP 200 text/html; charset=utf-8 responses of length 825, not valid .sp3.gz products. This cannot be used yet as validation and must not be represented as scored validation. |
| Pending prospective 260627 public-week readiness record | `claim_to_artifact_map.pending_public_real_measurement_temporal_od_prospective_260627` | `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json`; `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.sha256.txt`; `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json.ots`; `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260627_predeclaration_status_20260617.json`; `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260627_predeclaration_status_20260617.md` | Confirm this was created 2026-06-17 before test dates 2026-06-22..2026-06-26, uses validation week 260620 and test week 260627, and remains pending/not scored with no campaign run. The status record cites `results/validation/public_precise_reference_status_20260617.json` and `.md` as the current availability basis. The OTS proof is calendar-pending and not Bitcoin-block-header-attested; it is not scored validation, not DOI/public archive, not independent reproduction, and not operational POD. This record cannot be used as validation until validation/test public SP3 products become valid gzip/SP3 products and the frozen rule is scored once. |
| Official ILRS precise-reference availability gate | `claim_to_artifact_map.official_ilrs_precise_reference_availability_probe` | `scripts/probe_ilrs_precise_reference_availability.py`; `tests/test_probe_ilrs_precise_reference_availability.py`; `results/validation/ilrs_precise_reference_availability_20260617.json`; `results/validation/ilrs_precise_reference_availability_20260617.md` | Confirm each direct product probe records URL, HTTP status/content type/length, SHA-256, gzip/SP3 validity fields, and `usable_sp3`. Cached pending `.sp3.gz` placeholders under `results/real_slr_sp3_od_formal210_inputs/` must be classified unavailable/non-usable with the campaign-aligned `sp3_not_valid_gzip` boundary. Do not treat this availability report as scored validation. |
| Public multi-target SP3/CRD breadth probe | `claim_to_artifact_map.public_multi_target_sp3_crd_breadth_probe` | `results/public_sp3_multi_target_breadth_probe/public_sp3_multi_target_breadth_probe.json`; `paper/tables/public_sp3_multi_target_breadth_probe.tex`; `scripts/run_public_sp3_multi_target_breadth_probe.py`; `tests/test_public_sp3_multi_target_breadth_probe.py`; archived inputs under `results/public_sp3_multi_target_breadth_probe/sp3/` and `results/public_sp3_multi_target_breadth_probe/crd/` | Confirm 10 targets, 40/40 target-weeks, 240 fixed start epochs, 198/200 CRD target-days, 25,449 normal points, 29 stations, fixed-start and clustered sensitivity fields, and the state-scoring/coverage-only boundary. |
| Temporal OD selection-stability audit | `claim_to_artifact_map.public_temporal_od_selection_stability_audit` | `results/real_slr_sp3_temporal_selection_stability/real_slr_sp3_temporal_selection_stability.json`; `paper/tables/real_slr_sp3_temporal_selection_stability.tex`; `scripts/run_real_slr_sp3_temporal_selection_stability.py`; `tests/test_real_slr_sp3_temporal_selection_stability.py` | Confirm the 0.43 m validation margin, paired validation gap -0.43 m with 95% CI [-7.85, 8.03], learned selection in 62.4% of validation bootstraps and 6/10 leave-one-arc-out folds, and frozen negative test result. |
| Active main-manuscript table regeneration | `claim_to_artifact_map.active_main_manuscript_table_regeneration` | `results/validation/active_manuscript_regeneration.json`; `results/validation/active_manuscript_regeneration.md`; `results/validation/command_manifest.json`; active `paper/tables/*.tex` files and figure includes parsed from `paper/main.tex` | Confirm current-workspace parse of 11 active table inputs and 2 active figure includes, including `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`; confirm 13 pass, 0 mismatch, and 0 blocker among compared generated artifacts. |
| Archive-extracted reproduction tier | `claim_to_artifact_map.archive_extracted_reproduction_tier` | `results/validation/archive_extracted_reproduction.json`; `results/validation/archive_extracted_reproduction.md`; `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json`; `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.md`; `scripts/verify_archive_extracted_reproduction.py`; `scripts/regenerate_active_manuscript.py`; `scripts/run_real_slr_sp3_od_validation.py`; `scripts/build_paper_assets.py` | Confirm status `pass`, extracted artifact SHA-256 checks pass, claim/tier records resolve, and the archive-extracted OD rerun reports 10 completed arcs, DBAR 6/10, zero public-claim mismatches, and table-text match. Its nested active-regeneration count is a public v1.2.1 release-boundary record. Version 1.2.3 includes the ACF audit artifacts and development/holdout split summaries as the release-facing audit/table tier, but the older v1.2.1 archive-extracted report remains an older release-boundary record unless a v1.2.3 archive tier is regenerated. |
| Codex-agent clean archive attestation | `claim_to_artifact_map.codex_clean_archive_attestation` | `results/validation/codex_clean_archive_attestation_20260614.json`; `results/validation/codex_clean_archive_attestation_20260614.md` | Confirm the boundary: same-workspace separate-agent clean extracted archive attestation only, not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, and not independent external reproduction. |
| Containerized minimum-tier attestation | `claim_to_artifact_map.containerized_minimum_tier_attestation` | `results/validation/containerized_minimum_tier_attestation_20260614.json`; `results/validation/containerized_minimum_tier_attestation_20260614.md` | Confirm status `pass`, command template uses `<local-stage>`, and the boundary: local Docker Desktop clean-container staged minimum-tier integrity attestation only, same-host and staged from the current workspace/review archive, not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, not independent external reproduction, and not a full scientific rerun. |
| Independent-machine reproduction request/template | `claim_to_artifact_map.independent_machine_reproduction_request` | `release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md` | Confirm the document gives exact clean-machine steps using `release/SUPPLEMENTARY_MANIFEST.json`, `release/spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`, `scripts/verify_minimum_tier_reproduction.py`, and `scripts/verify_archive_extracted_reproduction.py`, plus a report template. It is not a completed independent reproduction. |
| Endpoint-choice sensitivity audit | `claim_to_artifact_map.endpoint_choice_sensitivity_audit` | `results/endpoint_selection_sensitivity/endpoint_selection_sensitivity.json`; `paper/tables/endpoint_selection_sensitivity.tex`; `scripts/build_endpoint_selection_sensitivity.py` | Confirm the K=8 and K=32 records are present, the strict-extension record is excluded, observed-step intervals are stored-record intervals, and all-step intervals are labelled sensitivity recomputations. |
| PUKF tuning-comparability sensitivity | `claim_to_artifact_map.pukf_tuning_comparability_sensitivity` | `results/pukf_tuning_sensitivity/pukf_hifi_grid_sensitivity.json`; `paper/tables/pukf_tuning_sensitivity.tex`; `scripts/run_pukf_hifi_tuning_sensitivity.py`; retained higher-fidelity CSV/JSON records | Confirm source CSV and reference JSON hashes, deterministic row-order alignment, nonzero finite paired denominators, and the negative held-out comparison with AUKF. |
| Classical guardrails and offline OD reference | `claim_to_artifact_map.classical_guardrails_and_offline_od_reference` | `results/metrics_summary.json`; `results/batch_wls_baseline/batch_wls_summary.csv`; `paper/tables/main_results.tex`; `paper/tables/batch_wls_baseline.tex`; `results/batch_wls_force_mismatch/batch_wls_summary.csv`; `results/batch_wls_force_mismatch/batch_wls_summary.json`; `results/batch_wls_force_mismatch/force_model_mismatch_test/batch_wls_summary.json` | Confirm the force-mismatch WLS summaries are indexed with nonempty SHA-256 values and that Table `batch_wls_baseline.tex` includes the force-mismatch row. Interpret WLS as strong on nominal/stress/public full-arc rows but failing observed-step under controlled force mismatch while slightly improving all-step. |
| Targeted retraining replay | `claim_to_artifact_map.targeted_retraining_replay` | `results/validation/targeted_retraining_replay_public.json`; `results/validation/targeted_retraining_replay_public.md`; `scripts/run_targeted_retraining_replay.py`; `tests/test_targeted_retraining_replay.py`; indexed replay artifacts | Confirm status `pass`, finite train/validation objective histories, checkpoint hash recording, and unchanged canonical checkpoint digest. |
| Full-rerun divergence audit | `claim_to_artifact_map.full_rerun_divergence_audit` | `results/validation/full_rerun_divergence_audit_20260617.json`; `results/validation/full_rerun_divergence_audit_20260617.md`; `scripts/build_full_rerun_divergence_audit.py`; `tests/test_full_rerun_divergence_audit.py`; retained full-rerun metrics, scorecard, and trajectory-error files | Confirm the schema version, input SHA-256 hashes, dense-visibility and SatNOGS replay concentration, scorecard candidate-divergence fail boundary, and the explicit statement that failure-conditioned rows are diagnostic only and do not replace canonical decision metrics. |

Archive-extracted tier boundary: Archive-extracted integrity, active
main-manuscript table regeneration, and one public LAGEOS CRD/SP3
precise-reference OD slice recomputation from archived public inputs only; this
does not rerun full raw-data generation, model retraining, all recursive
filters or tables, live public-data retrieval, operational POD validation, or
independent machine reproduction.

## Public ILRS/SP3 Correction-Corpus Boundary

The 40-arc correction-corpus audit recomputes correction and provenance
quantities over the public ILRS/SP3 corpus: arc completion, normal-point
coverage, meteorology and EOP coverage, one-way correction magnitudes, and
frame-reduction sensitivities. It does not recompute recursive filter
estimates and does not validate operational POD or the simulator
learned-versus-classical conclusion.

The temporal OD probe adds a later-week public real-measurement readout: the
learned calibrator is fit on weeks 260418 and 260425, the ridge and candidate
are selected on week 260502 (selected ridge `1e+06`), and the frozen choice is
scored on week 260509.
The learned calibrated higher-fidelity UKF is validation-selected but tests
worse than the compact UKF (411.51 m versus 368.64 m; +42.87 m paired gap,
95% CI [23.69, 63.38]; 0/10 test arcs better for the learned candidate). This is indexed as a
bounded probe only, not central external validation or operational POD.

The selection-stability audit for the same temporal OD probe is indexed under
`claim_to_artifact_map.public_temporal_od_selection_stability_audit`. It records
a 0.43 m learned validation margin, paired learned-minus-classical validation
gap -0.43 m with 95% CI [-7.85, 8.03], learned selection in 62.4% of validation
bootstraps and 6/10 leave-one-arc-out folds, and the same frozen negative test
readout. The archived corpus remains LAGEOS-1/-2 only; test-week
station-holdout support is YARL 7 arcs, HERL 3, MATM 0, and WETL 0.

The public multi-target SP3/CRD breadth probe is indexed under
`claim_to_artifact_map.public_multi_target_sp3_crd_breadth_probe`. It extends
SP3 state scoring to 10 targets and 40/40 target-weeks with 240 fixed start
epochs; the non-LAGEOS subset covers eight targets and 32/32 target-weeks. CRD
coverage is 198/200 target-days, 25,449 normal points, and 29 stations. This is
state-scoring and coverage breadth only, not multi-target OD validation. The
overall higher-fidelity-minus-compact readout is 24.25 m with fixed-start 95%
interval [5.29, 44.40], target-week clustered interval [-0.99, 51.81], and
target-cluster interval [-2.94, 57.66]; these are finite-probe sensitivity
summaries, not target-population or operational uncertainty intervals.

## Manual Inspection Route

For each headline statement:

1. Open `SUPPLEMENTARY_MANIFEST.json`.
2. Locate `claim_to_artifact_map.<manifest entry>`.
3. Confirm every listed record appears in `artifact_groups` or is one of the
   top-level release artifacts (`SUPPLEMENTARY_MANIFEST.json` or the review
   archive).
4. Compare the listed record's SHA-256 value against the file in the supplied
   archive.
5. Use the archive-extracted, table-regeneration, or full-rerun tiers only if a
   deeper check is needed beyond minimum archive integrity.
