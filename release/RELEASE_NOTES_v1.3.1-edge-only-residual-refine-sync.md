# SPOT-OD v1.3.1 Edge-Only Residual-Refinement Release Sync

This is a forward synchronization/correction release for the public
`v1.3.0-edge-only-residual-refine` GitHub/Zenodo release. It does not rewrite
v1.3.0 and does not promote a new residual-refinement branch.

## What changed

- Preserves the v1.3.0 train-loss-selected edge-only attention
  residual-refinement science claim and metrics.
- Updates the release-facing metadata, manifest, reviewer guide, deposit
  checklist, README, and independent-machine handoff to the new
  `v1.3.1-edge-only-residual-refine-sync` target.
- Cites the already published v1.3.0 DOI `10.5281/zenodo.20842573` as the
  prior clean version under concept DOI `10.5281/zenodo.20768672`.
- Carries the synchronized packet pieces already present in the cbc388c
  lineage: the rebuilt manuscript PDFs, synchronized release metadata/manifests,
  reviewer/independent-machine docs, and the previously missing
  `paper/tables/main_row_weighted_dls_poc.tex` table.
- Keeps alternative residual-refinement artifacts out of the release target.

## Preserved main result

The edge-only attention graph residual-refinement ensemble uses member seeds
2111, 2117, and 2129 with `prediction_mode=residual_refine`,
`residual_loss_weight=1e-5`, two attention graph layers, and
`--node-disagreement-features omit`. Omitting node-disagreement aggregates
reduces node features from 30 to 22 while preserving 10 pairwise edge features
for graph message passing.

Selection status: this edge-only run used 119 fit/training samples, 0 validation
samples, and train-loss checkpoint selection. It remains a post-freeze
exploratory proof of concept, not confirmatory model selection.

Observed-step position error:

| Slice | Edge-only attention RMSE | Reference RMSE | Gain |
| --- | ---: | ---: | ---: |
| All non-development vs best retained candidate | 373.728 m | 459.591 m | 18.682% |
| Fresh seeds 151/157/163/167 vs best retained candidate | 364.229 m | 445.943 m | 18.324% |
| All non-development vs matched local/no-message control | 373.728 m | 562.030 m | 33.504% |
| Fresh seeds 151/157/163/167 vs matched local/no-message control | 364.229 m | 814.283 m | 55.270% |

The edge-only mean graph remains closer but weaker/mixed: 386.224 m on all
non-development rows and 380.065 m on fresh rows. The graph-path isolation claim
is limited to edge-only attention versus the matched no-message local control.

## Key release artifacts

- `release/ZENODO_METADATA.json`
- `.zenodo.json`
- `release/SUPPLEMENTARY_MANIFEST.json`
- `release/README.md`
- `release/REVIEWER_START_HERE.md`
- `release/DEPOSIT_CHECKLIST.md`
- `release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`
- `paper/main.pdf`
- `paper/supplement.pdf`
- `paper/tables/main_trajectory_graph_selector_ensemble_poc.tex`
- `paper/tables/main_row_weighted_dls_poc.tex`
- `paper/figures/trajectory_residual_refine_gain_distribution.png`

## DOI status

Zenodo is connected to the GitHub repository. The v1.3.1 version DOI and record
are pending GitHub release import by Zenodo; no v1.3.1 DOI is asserted in this
packet. The prior clean version is v1.3.0, DOI `10.5281/zenodo.20842573`,
record <https://zenodo.org/records/20842573>. The concept DOI remains
`10.5281/zenodo.20768672`.

## Scope boundary

This is retained-candidate compact-simulator evidence only. It is not
operational precise orbit determination, not public precise-reference
validation, not independent-machine reproduction, not a full
raw/training/all-filter rerun, not standalone learned recursive filtering, not
broad learned orbit-determination validation, and not operational learned orbit
determination. The edge-only selection remains train-loss based with no
validation split, so it is exploratory rather than confirmatory.
