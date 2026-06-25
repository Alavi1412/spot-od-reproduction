# SPOT-OD Supplementary Evidence Package

Current release target: `v1.3.0-edge-only-residual-refine`
(`SPOT-OD v1.3.0 edge-only residual-refinement ablation`), with the final
GitHub tag target pending release creation. The GitHub release URL is
<https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.0-edge-only-residual-refine>.
Zenodo DOI and record assignment for v1.3.0 are pending GitHub release creation
and Zenodo import; no v1.3.0 DOI or Zenodo record is asserted in this packet.
The Zenodo concept DOI remains `10.5281/zenodo.20768672`. The prior clean
Zenodo version DOI is `10.5281/zenodo.20840386`.

Historical note: older v1.2.3 ACF holdout audit entries in this directory
describe a previous package and are retained only as release history, not as
the current active package for v1.3.0.

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
   `results/validation/active_manuscript_regeneration.json` and `.md`; the
   current source parse of `paper/main.tex` has 12 table inputs plus 2 figure
   includes, including `paper/tables/main_row_weighted_dls_poc.tex`. The public
   v1.2.1 archive-extracted report remains an older release-boundary record.
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
12. Public GraphAnchorPairGate PoC package: inspect
    `results/graph_anchor_pair_gate_seed_sweep_20260623/EVIDENCE_NOTE.md`,
    `graph_anchor_pair_gate_seed_sweep_summary.csv`,
    `graph_anchor_pair_gate_seed_sweep_by_scenario.csv`,
    `graph_anchor_pair_gate_seed_sweep_paired_seed_gains.csv`,
    `graph_anchor_pair_gate_seed_sweep_uncertainty_summary.csv`,
    `graph_anchor_pair_gate_seed_sweep_statistical_summary.md`, and
    `graph_anchor_pair_gate_seed_sweep_aggregate.png`, plus the seed-7
    display record under
    `results/graph_anchor_pair_gate_rfis_va_gpu_holdout_shift_all_candidates_seed7/`.
    This is a GNN-based station-time graph message-passing plus GRU gate over
    `RFIS:VA_RFIS`, using no-truth anchor features. It uses all-step
    center-window position RMSE on held-out eval trajectories within
    `process_noise_shift_test` and `maneuver_shift_test`. Across five local
    paired seeds it records 9/10 scenario-seed row wins and 4/5 paired seeds
    winning both scenarios; process shift is 4/5 wins with mean gain
    7.95663495038935%, and maneuver shift is 5/5 wins with mean gain
    8.05274642630686%. The failure row is seed 19 process shift:
    -2.0925251807980216% (5116.480181866038 m versus `VA_RFIS`
    5011.610960552836 m). The paired seed-level mean gains are +14.47%,
    +6.08%, +1.73%, +12.97%, and +4.77%. The descriptive uncertainty summary
    records row wins 9/10 with Wilson 95% CI [0.596, 0.982] and exact
    one-sided sign/binomial p=0.0107, plus paired both-scenario wins 4/5 with
    CI [0.376, 0.964] and p=0.1875. The graph input improves over the earlier
    scalar display's 7/10 row wins and 2/5 paired-seed wins, but this remains
    exploratory compact-simulator evidence, not an observed-step endpoint replacement,
    not operational precise-reference validation, and not independent-machine
    reproduction. It is included in the public
    `1.2.1-graph-anchor-gate-poc` package at Zenodo DOI
    `10.5281/zenodo.20811701` and GitHub release
    <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc>.
    The release run
    <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357>
    concluded `success` and ran the archive-extracted reproduction workflow
    and graph verifier on GitHub-hosted runners.
    A current-workspace post-release retained-output observed-step audit is
    generated locally by
    `scripts/build_graph_anchor_pair_gate_observed_step_audit.py` and writes
    `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.csv`,
    `.json`, and `.md`. It reads the retained per-scenario JSON files only and
    does not rerun training/evaluation. The audit records 0/10 observed-step
    row wins and 0/5 paired seeds winning both scenarios, so the all-step graph
    benefit does NOT transfer to observed-step primary endpoint superiority.
    This historical local GraphAnchorPairGate audit is not part of the
    published v1.2.1 Zenodo/GitHub release, and it is not operational
    precise-reference validation, not independent-machine reproduction, not
    third-party validation, and not a universal learned-method claim.
13. AdaptiveCandidateFusion audit release tier: inspect the fixed-soft
    full-retraining campaigns and 15-seed global scenario portfolio:
    inspect
    `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv`,
    `.json`, and `.md`, generated by
    `scripts/build_adaptive_candidate_fusion_fixed_soft_training_campaign_artifacts.py`
    from centered and observed-mask full-training campaign summaries. The
    builder validates each row is fixed soft
    (`requested_inference_mode=soft`, `selected_inference_mode=soft`,
    `inference_mode_selection_source=cli_fixed`), validates campaign metadata,
    and treats non-empty train/validation loss histories plus best/last
    checkpoint files as evidence that these are full training runs rather than
    skip-training replays. The centered training-step-mask campaign uses
    `training_step_mask=centered` and `validation_selection_metric=all_step_pos_rmse_m`;
    it records 8/10 observed-step scenario-seed row wins,
    3/5 paired seeds winning both shift scenarios, mean gain +2.594443463243368%,
    min -11.879589616936398%, and max +12.02043576038026% versus the best input
    candidate, while all-step is only a caveat at 5/10 row wins, 0/5 paired
    seeds, and mean gain -10.853657642974031%. The observed-mask campaign uses
    `training_step_mask=observed` and
    `validation_selection_metric=observed_step_pos_rmse_m`; it is a bounded
    negative/failure mode with 12/20 observed-step row wins, 3/10 paired seeds,
    mean gain -6.1915386036276825%, min -164.7209996196364%, and max
    +13.302773387058087%. Its all-step readout is also negative overall:
    11/20 row wins, 1/10 paired seeds, and mean gain -11.99436443792321%.
    The 15-seed validation-selected global scenario portfolio is retained at
    `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json`,
    `.md`, and `.csv`, generated by
    `scripts/analyze_adaptive_candidate_fusion_global_portfolio.py`. It selects
    scenario policies from pooled validation splits (`0.65*learned + 0.35*RFIS`
    for process and `0.55*learned + 0.45*EKF` for maneuver) and applies them to
    held-out compact-simulator eval rows with no test-row policy tuning. It
    records 25/30 observed-step row wins, 13/15 paired-seed wins, mean gain
    +3.793410580996871%, min -10.11448514448152%, and max +12.384536098768079%
    versus the best input candidate. The nonlearned-only validation-selected
    blend baseline is weaker at 19/30 wins and mean gain +0.7140145823400381%.
    These current-workspace compact-simulator artifacts show a stronger
    learned-including global-portfolio signal, a centered-training pocket, and
    an observed-mask failure mode, not broad learned-superiority. The compact
    manuscript table is `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`,
    and the supplementary manifest indexes both source artifact sets. In
    v1.2.3, these artifacts and the development/holdout split summaries are
    included as the ACF audit/table tier. They
    remain validation-selected compact-simulator PoC evidence, not
    independent-machine reproduction, not operational precise-reference
    validation, not third-party validation, not a full raw/training/all-filter
    reproduction, and not a universal learned-OD claim.
14. Local post-release public-clean-clone maintainer evidence: inspect
    `results/validation/public_clean_clone_v121_reproduction_20260623.json`
    and `.md`. This records a maintainer-run clean public clone from
    `https://github.com/Alavi1412/spot-od-reproduction.git` at tag
    `v1.2.1-graph-anchor-gate-poc`, detached HEAD commit
    `2dcd542dcb72f1622dfaf1cf8981a550862312bf`, clean git status before
    verifier output files, and passing public graph/archive verifiers. This
    post-release local record is historical maintainer evidence available in
    the current workspace/submission packet once included; it is not part of
    the published v1.2.1 Zenodo/GitHub release. It strengthens inspectability
    and public-clone reproducibility only; it is not independent third-party
    validation or independent-machine confirmation by an external operator.
15. Independent-machine reproduction handoff: inspect
    `release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`. This is a request
    and report template for an external operator, not a completed independent
    reproduction and not a completed independent reproduction record.

Pairing note: the immutable published v1.2.3 GitHub release asset listed below
is 59,140,917 bytes with SHA-256
`11909866b2ae1a375cdebe1472305d6d1fbd0b9f97453084fc7da16b78dcf70f`. The
current DOI-synced branch manifest records a regenerated local review archive,
59,142,123 bytes with SHA-256
`11451c2032243c972534f7de9eb40ba04c44ff69b6c45db179f2053f97ad9b7e`, and
`review_archive.matches_published_github_release_asset: false`. Use the
published asset only with the exact release/tag manifest set; use the current
manifest only with the regenerated local archive.

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
machine reproduction. Its nested active-regeneration count is tied to the
public v1.2.1 extracted archive. Historically, version 1.2.3 included the ACF
audit artifacts and development/holdout split summaries as its audit/table tier,
but the older v1.2.1 archive-extracted report remains an older release-boundary
record unless a historical v1.2.3 archive tier is regenerated.

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

## Public Archival Release

Release-facing metadata for the Zenodo/GitHub archival release:

- Short title: `SPOT-OD v1.3.0 edge-only residual-refinement ablation`
- Zenodo record: pending GitHub release creation and Zenodo import
- DOI: pending GitHub release creation and Zenodo import
- DOI URL: pending GitHub release creation and Zenodo import
- Zenodo concept DOI: `10.5281/zenodo.20768672`
- Prior clean Zenodo version DOI: `10.5281/zenodo.20840386`
- Zenodo status: `pending_github_release_zenodo_import`
- Version: `v1.3.0-edge-only-residual-refine`
- Resource type: Software
- License: `CC-BY-4.0`
- GitHub repository: <https://github.com/Alavi1412/spot-od-reproduction>
- GitHub release: <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.0-edge-only-residual-refine>
- Release tag: `v1.3.0-edge-only-residual-refine`
- Release commit: pending final GitHub tag target at release creation
- Zenodo archived file: pending GitHub release creation and Zenodo import
- Zenodo archived file bytes: pending
- Zenodo archived file MD5: pending
- GitHub release asset: pending
- GitHub release asset bytes: pending
- GitHub release asset SHA-256: pending

No v1.3.0 DOI or Zenodo record is asserted before import.

Prior public release history is retained:

- v1.2.3 Zenodo record: <https://zenodo.org/records/20825138>
- v1.2.3 DOI: `10.5281/zenodo.20825138`
- v1.2.3 DOI URL: <https://doi.org/10.5281/zenodo.20825138>
- v1.2.3 version: `1.2.3-acf-holdout-audit`
- v1.2.3 GitHub release:
  <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.3-acf-holdout-audit>
- v1.2.3 release tag: `v1.2.3-acf-holdout-audit`
- v1.2.3 release commit: `39e879d8665e489266bbf75f69634cab0e797fe8`
- v1.2.3 Zenodo archived file:
  `Alavi1412/spot-od-reproduction-v1.2.3-acf-holdout-audit.zip`
- v1.2.3 Zenodo archived file bytes: `187,254,529`
- v1.2.3 Zenodo archived file MD5: `7eb8b43a9af90a4783482a7a3a086f92`
- v1.2.3 GitHub release asset: `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`
- v1.2.3 GitHub release asset bytes: `59,140,917`
- v1.2.3 GitHub release asset SHA-256:
  `11909866b2ae1a375cdebe1472305d6d1fbd0b9f97453084fc7da16b78dcf70f`

- v1.2.2 Zenodo record: <https://zenodo.org/records/20822968>
- v1.2.2 DOI: `10.5281/zenodo.20822968`
- v1.2.2 DOI URL: <https://doi.org/10.5281/zenodo.20822968>
- v1.2.2 Zenodo status: `published`
- v1.2.2 GitHub release:
  <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.2-acf-audit>
- v1.2.2 release tag: `v1.2.2-acf-audit`
- v1.2.2 release commit: `6fbc88745b6d96939736d59731089e99786c1f8c`
- v1.2.2 Zenodo archived file:
  `Alavi1412/spot-od-reproduction-v1.2.2-acf-audit.zip`
- v1.2.2 Zenodo archived file bytes: `72,607,548`
- v1.2.2 Zenodo archived file MD5: `533b8363954cb6531f17bf4d405a5092`
- v1.2.2 GitHub release asset: `spot_od_v1_2_2_acf_audit_review_archive.zip`
- v1.2.2 GitHub release asset bytes: `59,127,034`
- v1.2.2 GitHub release asset SHA-256:
  `e6b6139bb0fb5463f5091bdde05e14b82a8191d1419466cdd21c8aafa533b240`
- v1.2.1 Zenodo record: <https://zenodo.org/records/20811701>
- v1.2.1 DOI: `10.5281/zenodo.20811701`
- v1.2.1 DOI URL: <https://doi.org/10.5281/zenodo.20811701>
- v1.2.1 Zenodo status: `published`
- v1.2.1 GitHub release:
  <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc>
- v1.2.1 release tag: `v1.2.1-graph-anchor-gate-poc`
- v1.2.1 release commit: `2dcd542dcb72f1622dfaf1cf8981a550862312bf`
- v1.2.1 Zenodo archived file:
  `Alavi1412/spot-od-reproduction-v1.2.1-graph-anchor-gate-poc.zip`
- v1.2.1 Zenodo archived file bytes: `94,265,950`
- v1.2.1 Zenodo archived file MD5: `233d2fc7fce1bc57afdd66332a3a7dc1`
- v1.2.1 GitHub release asset: `spot_od_v1_2_1_graph_anchor_gate_poc.zip`
- v1.2.1 GitHub release asset bytes: `17,710,047`
- v1.2.1 GitHub release asset SHA-256:
  `3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c`
- v1.2.3 branch GitHub Actions verifier:
  <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253565>
  (`success`)
- v1.2.3 tag GitHub Actions verifier:
  <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253538>
  (`success`)
- Prior v1.2.1 release-triggered GitHub Actions verifier:
  <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357>
  (`success`; ran the archive-extracted reproduction workflow and graph
  verifier on GitHub-hosted runners)

Local post-release public-clean-clone maintainer evidence is recorded in
`results/validation/public_clean_clone_v121_reproduction_20260623.json` and
`.md`. On 2026-06-23, the main session cloned
`https://github.com/Alavi1412/spot-od-reproduction.git` from the public tag
`v1.2.1-graph-anchor-gate-poc`, observed the public tag warning
`refs/tags/v1.2.1-graph-anchor-gate-poc a9842e8d76b41f5227695ef6e3a5532668c7c0e8 is not a commit`,
detached HEAD at commit `2dcd542dcb72f1622dfaf1cf8981a550862312bf`, confirmed
an empty/clean git status before verifier output files, and ran both public
verifiers with exit code 0. The graph verifier checked the GitHub release asset
`spot_od_v1_2_1_graph_anchor_gate_poc.zip` with 17,710,047 bytes and SHA-256
`3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c`; the
archive-extracted verifier passed active table regeneration, archived public
OD-slice rerun, artifact, claim-map, and regeneration-tier checks. This is NOT
independent third-party validation and is NOT independent-machine confirmation
if that means an external operator; it strengthens inspectability and
public-clone reproducibility only. It was created after v1.2.1 publication and
is available in the current workspace/submission packet once included; it is
not part of the published v1.2.1 Zenodo/GitHub release unless a later release
includes it.

Historical v1.2.3 superseded v1.2.2 only by repairing the public release
boundary for the ACF audit/table tier and packaging the development/holdout
split summaries in that older release. Version 1.2.2 remains historical ACF
audit package history, and version 1.2.1 remains historical GraphAnchorPairGate
package history. Scientific metrics are as recorded, not upgraded to
operational validation.

The historical v1.2.3 release is bounded reproduction-support evidence for
archive extraction, manifest hashes, active manuscript artifact regeneration,
one archived-input public OD slice rerun, public packaging of the
GraphAnchorPairGate PoC, and public packaging of the ACF audit/table tier
including the development/holdout split summaries. It is not the current
v1.3.0 package and is not full raw/training/all-filter reproduction, live
public-data retrieval, operational POD validation, independent-machine
confirmation, third-party independent validation, full scientific reproduction,
or confirmatory learned superiority.

The public `1.2.1-graph-anchor-gate-poc` package includes the
post-manuscript GraphAnchorPairGate proof-of-concept artifact under
`results/graph_anchor_pair_gate_seed_sweep_20260623/`. Its aggregate plot is
`graph_anchor_pair_gate_seed_sweep_aggregate.png`; its paired seed gains and
uncertainty readout are `graph_anchor_pair_gate_seed_sweep_paired_seed_gains.csv`,
`graph_anchor_pair_gate_seed_sweep_uncertainty_summary.csv`, and
`graph_anchor_pair_gate_seed_sweep_statistical_summary.md`. Treat it as bounded
compact-simulator evidence only: GNN-based station-time graph message passing
plus a GRU gate over `RFIS:VA_RFIS`, no-truth anchor features, all-step
center-window RMSE, held-out eval trajectories within the two shift scenarios,
five paired seeds, one seed-19 process-shift failure, and no replacement of the
frozen observed-step endpoint hierarchy.

A historical local post-release retained-output observed-step audit of the
same GraphAnchorPairGate sweep is generated in the current workspace at
`results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.csv`,
`.json`, and `.md` by
`scripts/build_graph_anchor_pair_gate_observed_step_audit.py`. It records 0/10
observed-step row wins and 0/5 paired both-scenario wins, confirming that the
all-step graph benefit does NOT transfer to observed-step primary endpoint
superiority. This audit is local/current-workspace historical GraphAnchorPairGate
evidence only; it is not part of the published v1.2.1 Zenodo/GitHub release,
not operational precise-reference validation, not independent-machine
reproduction, not third-party validation, and not a universal learned-method
claim.

## Contents

- `CITATION.cff` - local citation metadata for the package.
- `ZENODO_METADATA.json` - Zenodo deposit metadata for the current
  `v1.3.0-edge-only-residual-refine` release target. The v1.3.0 DOI/record is
  pending GitHub release creation and Zenodo import; the concept DOI and prior
  clean DOI are retained as historical/prior references.
- `REVIEWER_START_HERE.md` - reviewer-facing map from headline claims to
  manifest entries and regeneration tiers.
- `RELEASE_NOTES_v1.3.0-edge-only-residual-refine.md` - concise release note
  for the current v1.3.0 edge-only residual-refinement ablation release target.
- `RELEASE_NOTES_v1.2.3-acf-holdout-audit.md` - concise release note for the
  historical v1.2.3 ACF holdout audit release and minted Zenodo fields.
- `RELEASE_NOTES_v1.2.2-acf-audit.md` - historical release note for the v1.2.2
  ACF audit release.
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
- `results/graph_anchor_pair_gate_seed_sweep_20260623/EVIDENCE_NOTE.md`,
  `graph_anchor_pair_gate_seed_sweep_summary.csv`,
  `graph_anchor_pair_gate_seed_sweep_by_scenario.csv`,
  `graph_anchor_pair_gate_seed_sweep_paired_seed_gains.csv`,
  `graph_anchor_pair_gate_seed_sweep_uncertainty_summary.csv`,
  `graph_anchor_pair_gate_seed_sweep_statistical_summary.md`,
  `graph_anchor_pair_gate_seed_sweep_aggregate.png`,
  `seed_7_split_7/graph_anchor_pair_gate_candidate_comparison.png`, and
  `results/graph_anchor_pair_gate_rfis_va_gpu_holdout_shift_all_candidates_seed7/graph_anchor_pair_gate_summary.csv`
  are the post-manuscript GraphAnchorPairGate PoC record included in the public
  `1.2.1-graph-anchor-gate-poc` release. This artifact is not an observed-step
  endpoint replacement, not independent-machine reproduction, and not a public
  precise-reference result.
- `results/validation/public_clean_clone_v121_reproduction_20260623.json`
  and `.md` - local post-release maintainer-run public-clean-clone evidence
  for the public `v1.2.1-graph-anchor-gate-poc` tag and commit
  `2dcd542dcb72f1622dfaf1cf8981a550862312bf`; both public verifiers passed, but
  this is not independent third-party validation, not independent-machine
  confirmation by an external operator, not operational precise-reference
  validation, and not a full raw-data/training rerun. It is available in the
  current workspace/submission packet once included; it is not part of the
  published v1.2.1 Zenodo/GitHub release.
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
- `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip` - reviewer-access transport
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
| Public GraphAnchorPairGate PoC package | Public `1.2.1-graph-anchor-gate-poc` provenance: Zenodo record <https://zenodo.org/records/20811701>, DOI `10.5281/zenodo.20811701`, GitHub release <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc>, commit `2dcd542dcb72f1622dfaf1cf8981a550862312bf`, and successful release run <https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357>. Primary records are `results/graph_anchor_pair_gate_seed_sweep_20260623/EVIDENCE_NOTE.md`, `graph_anchor_pair_gate_seed_sweep_summary.csv`, `graph_anchor_pair_gate_seed_sweep_by_scenario.csv`, `graph_anchor_pair_gate_seed_sweep_paired_seed_gains.csv`, `graph_anchor_pair_gate_seed_sweep_uncertainty_summary.csv`, `graph_anchor_pair_gate_seed_sweep_statistical_summary.md`, `graph_anchor_pair_gate_seed_sweep_aggregate.png`, `seed_7_split_7/graph_anchor_pair_gate_candidate_comparison.png`, and `results/graph_anchor_pair_gate_rfis_va_gpu_holdout_shift_all_candidates_seed7/graph_anchor_pair_gate_summary.csv`. The artifact records a GNN-based station-time graph message-passing plus GRU gate over `RFIS:VA_RFIS` using no-truth anchor features; all-step center-window RMSE on held-out eval trajectories within the two shift scenarios; 9/10 scenario-seed row wins with Wilson 95% CI [0.596, 0.982]; 4/5 paired seeds winning both scenarios with CI [0.376, 0.964]; and the seed-19 process-shift failure at -2.0925251807980216%. It remains outside the primary observed-step endpoint hierarchy, not operational precise-reference validation, not independent-machine reproduction, not third-party reproduction, not a full raw/training/all-filter rerun, and not a universal claim. |
| Local GraphAnchorPairGate retained-output observed-step audit | Historical local current-workspace records; not a manifest entry and not part of the published v1.2.1 Zenodo/GitHub release. Primary records are `results/graph_anchor_pair_gate_seed_sweep_20260623/graph_anchor_pair_gate_observed_step_audit.csv`, `.json`, and `.md`, generated by `scripts/build_graph_anchor_pair_gate_observed_step_audit.py` from retained per-scenario JSONs. The audit records 0/10 observed-step row wins and 0/5 paired both-scenario wins; mean gain is -5.568334199477706%, min -12.482167132357489%, max -0.8712975503737664%. Boundary: retained-output audit only, does not rerun training/evaluation, confirms the GraphAnchorPairGate all-step benefit does NOT transfer to observed-step primary endpoint superiority, not operational precise-reference validation, not independent-machine reproduction, not third-party validation, and not a universal learned-method claim. |
| AdaptiveCandidateFusion audit/table tier | Included in v1.2.3 as manifest-indexed compact-simulator PoC records. Fixed-soft records are `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv`, `.json`, and `.md`, generated by `scripts/build_adaptive_candidate_fusion_fixed_soft_training_campaign_artifacts.py`; the 15-seed validation-selected global portfolio records are `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json`, `.md`, and `.csv`, generated by `scripts/analyze_adaptive_candidate_fusion_global_portfolio.py`; the development/holdout split summaries are `results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/summary.json`, `.md`, and `.csv`; the compact manuscript table is `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`; generator/test support includes `scripts/build_paper_assets.py` and the focused ACF tests. The fixed-soft builder validates rows, campaign metadata, training histories, and checkpoints. Centered training reproduces the observed-step pocket: 8/10 row wins, 3/5 paired wins, mean +2.5944434632433686%; observed-mask full retraining is a bounded negative/failure mode: 12/20 observed-step row wins, 3/10 paired, mean -6.1915386036276825%. The global scenario portfolio selects `0.65*learned + 0.35*RFIS` for process and `0.55*learned + 0.45*EKF` for maneuver from validation splits and records 25/30 observed-step row wins, 13/15 paired-seed wins, mean +3.793410580996871%, min -10.11448514448152%, and max +12.384536098768079%; the nonlearned-only validation-selected blend baseline is weaker at 19/30 wins and +0.7140145823400381% mean. The development/holdout split gives weak/mixed internal evidence only: 7/10 row wins, mean +1.45%, row CI [-2.22,+4.96], and 3/5 seed-paired wins. Boundary: validation-selected compact-simulator PoC only, not operational precise-reference validation, not independent-machine reproduction, not third-party validation, not a full raw/training/all-filter reproduction, not confirmatory learned superiority, and not a universal learned-OD claim. |
| Local post-release public-clean-clone maintainer evidence | Primary records are `results/validation/public_clean_clone_v121_reproduction_20260623.json` and `.md`. This records a clean public clone from GitHub tag `v1.2.1-graph-anchor-gate-poc`, detached HEAD commit `2dcd542dcb72f1622dfaf1cf8981a550862312bf`, clean status before verifier outputs, Python path `C:\Users\alavi\AppData\Local\Microsoft\WindowsApps\python.exe`, and exit-code-0 runs of the public GraphAnchorPairGate PoC verifier and archive-extracted verifier. It is historical maintainer evidence created after v1.2.1 publication and available in the current workspace/submission packet once included; it is not part of the published v1.2.1 Zenodo/GitHub release. Boundary: public-clean-clone maintainer reproduction on the current host only; not independent third-party validation, not independent-machine confirmation by an external operator, not operational precise-reference validation, and not a full raw-data/training rerun. |
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
| Reproducibility and review-stage archive integrity | `REVIEWER_START_HERE.md`, `SUPPLEMENTARY_MANIFEST.json`, `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`, `results/validation/minimum_tier_reproduction_check.json`, `results/validation/submission_validation.json`, `results/release_packet.json` |
| Independent-machine reproduction request/template | Manifest entry `independent_machine_reproduction_request`; primary record is `release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`; this gives an external operator exact clean-machine verification steps and a report template but is not a completed independent reproduction and is not an already completed independent reproduction record. |
| Active main-manuscript table regeneration | `results/validation/active_manuscript_regeneration.json`, `results/validation/active_manuscript_regeneration.md`, `results/validation/command_manifest.json`, active `paper/tables/*.tex` files included by `paper/main.tex`; current workspace parses 11 table inputs plus 2 figure includes, 13/13 passing |
| Archive-extracted reproduction tier | `results/validation/archive_extracted_reproduction.json`, `results/validation/archive_extracted_reproduction.md`, `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json`, `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.md`, `scripts/verify_archive_extracted_reproduction.py`, `scripts/regenerate_active_manuscript.py`, `scripts/run_real_slr_sp3_od_validation.py`, `scripts/build_paper_assets.py`, `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`, `SUPPLEMENTARY_MANIFEST.json` |
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
  source parse has 12 active table inputs and 2 external figure includes,
  including `paper/tables/main_findings_summary.tex`,
  `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`, and
  `paper/tables/main_row_weighted_dls_poc.tex`.
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
