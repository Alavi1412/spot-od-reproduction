# SPOT-OD supplementary evidence package - public deposit ledger

This file is internal release-ledger bookkeeping; it is not paper-facing and is
not referenced from the manuscript. It records release-facing metadata for the
pending v1.2.2 ACF audit release, the retained v1.2.1 public archive history,
and the remaining scope boundaries.

## Status

- Public archival deposition: pending publication for v1.2.2.
- Short title: SPOT-OD v1.2.2 ACF audit release
- Zenodo record: pending
- DOI: pending; fill after Zenodo mints the v1.2.2 DOI
- DOI URL: pending
- Zenodo status: pending publication
- Version: 1.2.2-acf-audit
- Resource type: Dataset
- License: CC-BY-4.0
- GitHub repository: https://github.com/Alavi1412/spot-od-reproduction
- GitHub release: https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.2-acf-audit
- Release tag: v1.2.2-acf-audit
- Release commit: pending until the release tag is created
- Supersession note: v1.2.2 supersedes v1.2.1 only by adding the ACF
  audit/table tier and manuscript claim-boundary wording. Scientific metrics
  are as recorded, not upgraded to operational validation.
- Prior v1.2.1 DOI: 10.5281/zenodo.20811701
- Prior v1.2.1 DOI URL: https://doi.org/10.5281/zenodo.20811701
- Prior v1.2.1 GitHub release: https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc

## Public Archive Integrity

- v1.2.2 Zenodo file: pending
- v1.2.2 Zenodo archived file bytes: pending
- v1.2.2 Zenodo archived file MD5: pending
- v1.2.2 GitHub release asset: pending
- v1.2.2 GitHub release asset bytes: pending
- v1.2.2 GitHub release asset SHA-256: pending

Do not fill the v1.2.2 DOI, archived-file MD5, asset byte count, or asset
SHA-256 until Zenodo/GitHub have minted or published those concrete records.

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

No v1.2.2 release-triggered verifier run exists until the GitHub release is
created. The following run is retained prior-version v1.2.1 evidence.

- GitHub Actions verifier:
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
    (`release/CITATION.cff`) updated to v1.2.2, with the DOI left pending until
    Zenodo mints it.
12. AdaptiveCandidateFusion audit artifacts:
    `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`,
    `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/`,
    `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/`,
    and generator/test support in `scripts/build_paper_assets.py` and focused
    ACF tests.

## Scope Boundary

The v1.2.2 DOI/GitHub release is bounded reproduction-support evidence for
archive extraction, manifest hashes, active manuscript artifact regeneration,
one archived-input public OD slice rerun, and public packaging of the
GraphAnchorPairGate PoC plus the ACF audit/table tier only. It does not claim
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
