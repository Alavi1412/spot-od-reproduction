# SPOT-OD supplementary evidence package - public deposit ledger

This file is internal release-ledger bookkeeping; it is not paper-facing and is
not referenced from the manuscript. It records release-facing metadata for the
v1.2.3 ACF holdout audit release boundary, the retained v1.2.2/v1.2.1 public
archive history, and the remaining scope boundaries.

## Status

- Public archival deposition: v1.2.3 GitHub release and Zenodo software record
  published.
- Short title: SPOT-OD v1.2.3 ACF holdout audit release
- Zenodo record: https://zenodo.org/records/20825138
- DOI: 10.5281/zenodo.20825138
- DOI URL: https://doi.org/10.5281/zenodo.20825138
- Zenodo concept DOI: 10.5281/zenodo.20768672
- Zenodo status: published
- Version: 1.2.3-acf-holdout-audit
- Resource type: Software
- License: CC-BY-4.0
- GitHub repository: https://github.com/Alavi1412/spot-od-reproduction
- GitHub release: https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.3-acf-holdout-audit
- Release tag: v1.2.3-acf-holdout-audit
- Release commit: 39e879d8665e489266bbf75f69634cab0e797fe8
- Supersession note: v1.2.3 repairs the public release boundary for the ACF
  audit/table tier by packaging the development/holdout split summaries in the
  new release. Scientific metrics are as recorded, not upgraded to operational
  validation.
- Historical v1.2.2 DOI: 10.5281/zenodo.20822968
- Historical v1.2.2 record: https://zenodo.org/records/20822968
- Prior v1.2.1 DOI: 10.5281/zenodo.20811701
- Prior v1.2.1 DOI URL: https://doi.org/10.5281/zenodo.20811701
- Prior v1.2.1 GitHub release: https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc

## Public Archive Integrity

- v1.2.3 Zenodo file:
  `Alavi1412/spot-od-reproduction-v1.2.3-acf-holdout-audit.zip`
- v1.2.3 Zenodo archived file bytes: 187,254,529
- v1.2.3 Zenodo archived file MD5: 7eb8b43a9af90a4783482a7a3a086f92
- v1.2.3 GitHub release asset:
  `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`
- v1.2.3 GitHub release asset bytes: 59,140,917
- v1.2.3 GitHub release asset SHA-256:
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

- Successful v1.2.3 branch verifier:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253565
- Successful v1.2.3 tag verifier:
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
11. Zenodo metadata (`release/ZENODO_METADATA.json`) and citation metadata
    (`release/CITATION.cff`) updated to the minted v1.2.3 Zenodo DOI and
    record.
12. AdaptiveCandidateFusion audit artifacts:
    `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`,
    `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/`,
    `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/`,
    `results/adaptive_candidate_fusion_global_scenario_portfolio_dev10_holdout5_20260624/`,
    and generator/test support in `scripts/build_paper_assets.py` and focused
    ACF tests.

## Scope Boundary

The v1.2.3 GitHub release is bounded reproduction-support evidence for
archive extraction, manifest hashes, active manuscript artifact regeneration,
one archived-input public OD slice rerun, public packaging of the
GraphAnchorPairGate PoC, and public packaging of the ACF audit/table tier
including the development/holdout split summaries only. It does not claim
full raw/training/all-filter reproduction, live public-data retrieval,
operational POD validation, independent-machine confirmation, third-party
independent validation, full scientific reproduction, or universal learned-OD
superiority.

## Forbidden In Paper-Facing Text

Per the paper-facing release constraints, this checklist exists only in
`release/` and is never quoted, paraphrased, or referenced from any
paper-facing source (`paper/main.tex`, `paper/supplement.tex`,
`paper/evidence_plan.tex`). The Data Availability section in the manuscript
remains the canonical paper-facing statement.
