# SPOT-OD supplementary evidence package - public deposit ledger

This file is internal release-ledger bookkeeping; it is not paper-facing and is
not referenced from the manuscript. It records release-facing metadata for the
current v1.3.1 edge-only residual-refinement sync release target, the retained
v1.2.x public archive history, and the remaining scope boundaries.

## Status

- Public archival deposition: v1.3.1 GitHub release target prepared; Zenodo
  DOI/record pending GitHub release creation and Zenodo import.
- Short title: SPOT-OD v1.3.1 edge-only residual-refinement release sync
- Zenodo record: pending GitHub release creation and Zenodo import
- DOI: pending GitHub release creation and Zenodo import
- DOI URL: pending GitHub release creation and Zenodo import
- Zenodo concept DOI: 10.5281/zenodo.20768672
- Prior clean Zenodo version DOI: 10.5281/zenodo.20842573
- Zenodo status: pending_github_release_zenodo_import
- Version: v1.3.1-edge-only-residual-refine-sync
- Resource type: Software
- License: CC-BY-4.0
- GitHub repository: https://github.com/Alavi1412/spot-od-reproduction
- GitHub release: https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.1-edge-only-residual-refine-sync
- Release tag: v1.3.1-edge-only-residual-refine-sync
- Release commit: pending final GitHub tag target at release creation
- Supersession note: v1.3.1 is a forward sync/correction release that preserves
  the v1.3.0 train-loss edge-only retained-candidate residual-refinement
  result. It does not assert a v1.3.1 Zenodo DOI/record
  before GitHub release creation and Zenodo import. Scientific metrics are as
  recorded, not upgraded to operational validation.
- Historical v1.3.0 DOI: 10.5281/zenodo.20842573
- Historical v1.3.0 record: https://zenodo.org/records/20842573
- Historical v1.2.8 DOI: 10.5281/zenodo.20840386
- Historical v1.2.8 record: https://zenodo.org/records/20840386
- Historical v1.2.3 DOI: 10.5281/zenodo.20825138
- Historical v1.2.3 record: https://zenodo.org/records/20825138

## Public Archive Integrity

- v1.3.1 Zenodo file: pending GitHub release creation and Zenodo import
- v1.3.1 Zenodo archived file bytes: pending
- v1.3.1 Zenodo archived file MD5: pending
- v1.3.1 GitHub release asset: pending
- v1.3.1 GitHub release asset bytes: pending
- v1.3.1 GitHub release asset SHA-256: pending

Historical v1.2.3 public archive integrity:

- Zenodo file:
  `Alavi1412/spot-od-reproduction-v1.2.3-acf-holdout-audit.zip`
- Zenodo archived file bytes: 187,254,529
- Zenodo archived file MD5: 7eb8b43a9af90a4783482a7a3a086f92
- GitHub release asset:
  `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`
- GitHub release asset bytes: 59,140,917
- GitHub release asset SHA-256:
  11909866b2ae1a375cdebe1472305d6d1fbd0b9f97453084fc7da16b78dcf70f

Historical v1.2.2 public archive integrity:

- Zenodo file:
  `Alavi1412/spot-od-reproduction-v1.2.2-acf-audit.zip`
- Zenodo archived file bytes: 72,607,548
- Zenodo archived file MD5: 533b8363954cb6531f17bf4d405a5092
- GitHub release asset: `spot_od_v1_2_2_acf_audit_review_archive.zip`
- GitHub release asset bytes: 59,127,034
- GitHub release asset SHA-256:
  e6b6139bb0fb5463f5091bdde05e14b82a8191d1419466cdd21c8aafa533b240

Prior v1.2.1 public archive integrity:

- Zenodo file:
  `Alavi1412/spot-od-reproduction-v1.2.1-graph-anchor-gate-poc.zip`
- Zenodo archived file bytes: 94,265,950
- Zenodo archived file MD5: 233d2fc7fce1bc57afdd66332a3a7dc1
- GitHub release asset: `spot_od_v1_2_1_graph_anchor_gate_poc.zip`
- GitHub release asset bytes: 17,710,047
- GitHub release asset SHA-256:
  3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c

## Release-Triggered Verification

- v1.3.1 branch verifier: pending or not recorded
- v1.3.1 tag verifier: pending or not recorded

- Historical v1.2.3 branch verifier:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253565
- Historical v1.2.3 tag verifier:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253538

- Historical v1.2.2 branch verifier:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28075721074
- Historical v1.2.2 tag verifier:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28075722522

- Prior v1.2.1 GitHub Actions verifier:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357
- Status: success
- Verifier scope: ran the archive-extracted reproduction workflow and graph
  verifier on GitHub-hosted runners.

The release-triggered verifier is maintainer-run platform evidence for the
archive-extracted reproduction-support tier. It is not third-party independent
validation.

## Required Contents In The Public Release

1. Manuscript source (`paper/main.tex` + `paper/supplement.tex`), bibliography
   (`paper/references.bib`), generated tables and figures.
2. Version-pinned dependency provenance: `requirements.txt` and
   `pyproject.toml` with SHA-256 digests recorded in
   `release/SUPPLEMENTARY_MANIFEST.json`.
3. Predeclared rule artifacts: timestamped rule files under
   `release/predeclarations/`.
4. 15-seed cohort (seeds 41-55) and trained-model records.
5. Per-artifact SHA-256 digests for produced outputs, recorded in
   `release/SUPPLEMENTARY_MANIFEST.json`.
6. Archived public-input identifiers: CelesTrak GP/TLE snapshots, SatNOGS
   network/observation/API metadata, ILRS CRD normal-point filenames and
   source URLs, ILRS NSGF SP3-c precise orbit products, and the IERS
   Earth-orientation series.
7. Vendored KalmanNet source release at the recorded upstream commit hash.
8. Reviewer access guide (`release/REVIEWER_START_HERE.md`) and minimum-tier
   reproduction check reports under `results/validation/`.
9. Independent-machine reproduction request/template
   (`release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`) so an external
   operator can produce a signed clean-machine report without changing any
   manuscript claim before the report exists.
10. Official ILRS precise-reference availability probe output under
    `results/validation/`, documenting whether pending LAGEOS prospective SP3
    products are valid gzip/SP3 files before any scoring run.
11. Zenodo metadata (`release/ZENODO_METADATA.json`) updated to the v1.3.1
    release target with DOI/record pending GitHub release creation and Zenodo
    import; citation metadata must not assert a v1.3.1 DOI before import.
12. Edge-only residual-refinement artifacts and release note:
    `release/RELEASE_NOTES_v1.3.1-edge-only-residual-refine-sync.md`,
    `paper/tables/main_row_weighted_dls_poc.tex`, the residual-refinement
    comparison scripts, and the manifest-indexed result directories listed in
    the v1.3.1 release notes.
13. Historical AdaptiveCandidateFusion audit artifacts:
    `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`,
    `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/`,
    `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/`,
    `results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/`,
    and generator/test support in `scripts/build_paper_assets.py` and focused
    ACF tests.

## Scope Boundary

The v1.3.1 GitHub release target is bounded reproduction-support evidence for
the edge-only retained-candidate residual-refinement ablation and supporting
comparison artifacts only. It does not yet have a minted v1.3.1 Zenodo DOI or
record, and it does not claim full raw/training/all-filter reproduction, live
public-data retrieval, operational POD validation, independent-machine
confirmation, third-party independent validation, full scientific reproduction,
or universal learned-OD superiority. The v1.2.3 ACF audit/table tier remains
historical package history, not the current active package.

## Forbidden In Paper-Facing Text

Per the paper-facing release constraints, this checklist exists only in
`release/` and is never quoted, paraphrased, or referenced from any
paper-facing source (`paper/main.tex`, `paper/supplement.tex`,
`paper/evidence_plan.tex`). The Data Availability section in the manuscript
remains the canonical paper-facing statement.
