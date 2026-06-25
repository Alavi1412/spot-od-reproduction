# SPOT-OD v1.3.2 public reproduction package

This package is a public reproduction packaging repair for the v1.3.1
validation-selected edge-only retained-candidate attention graph
residual-refinement proof of concept.

It does not change the scientific metric readback from v1.3.1. The selected
attention residual-refinement evidence remains:

- All non-development rows: 390.317 m versus 459.591 m for the best retained
  candidate, a 15.073% gain.
- Fresh source-generation seeds 151/157/163/167: 386.373 m versus 445.943 m
  for the best retained candidate, a 13.358% gain.
- Versus matched edge-only local/no-message control: 24.835%
  all-non-development and 45.329% fresh advantages.
- Versus edge-only mean graph: 3.983% all-non-development and 5.548% fresh
  advantages.

## Release identifiers

- GitHub release:
  <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.2-public-reproduction-package>
- Release tag: `v1.3.2-public-reproduction-package`
- Main GitHub release asset: `spot_od_v1_3_2_public_reproduction_package.zip`
- Training-input GitHub release asset:
  `spot_od_v1_3_2_public_reproduction_training_inputs.zip`
- Zenodo v1.3.2 version DOI: assigned after GitHub release import; not claimed
  in this package.
- Previous Zenodo version DOI: `10.5281/zenodo.20844596`
- Zenodo concept DOI: `10.5281/zenodo.20768672`
- Repaired v1.3.1 archive SHA-256:
  `4d575f7f8d3326823dc50f71f5f542dab1f924780082f8b6f00195cbf22619a4`

Archive byte sizes and SHA-256 digests are reported after construction rather
than embedded inside this packaged README.

## What is included in the main package

- v1.3.2 GitHub/Zenodo metadata, release notes, package README, manifest,
  citation metadata, and license notice.
- v1.3.1 repaired release documentation and the repaired v1.3.1 release archive
  for traceability.
- Manuscript source/PDF, validation-selected table, and validation-selected
  figure needed to inspect the selected proof of concept.
- Final rows, summaries, comparison intervals, local-control diagnostic rows,
  and selected run checkpoints for the attention, mean-graph, and local/no-message
  residual-refinement runs.
- Scripts and tests for the retained-candidate graph selector, comparison
  intervals, tail diagnostic, figure regeneration, and package verification.
- Runtime import support for extracted-package execution: `src/`,
  `scripts/_bootstrap.py`, `scripts/__init__.py`, `pyproject.toml`,
  `requirements.txt`, and the repository `README.md`.

## Training-input package

The separate training-input ZIP supplies the upstream retained-candidate input
directories needed by the provenance command in the v1.3.1/v1.3.2 release
notes. Extract it at the repository root so the
`results/adaptive_candidate_fusion_observed_fixed_soft_*` paths exist beside
`scripts/run_trajectory_candidate_graph_selector_poc.py`.

The training-input ZIP is checkpoint-free. It contains saved retained-candidate
prediction arrays and metadata consumed by the GNN training loader. It does not
include upstream ACF checkpoints.

## Fast verification

From the repository root:

```powershell
python scripts\verify_v132_public_reproduction_package.py --archive release\spot_od_v1_3_2_public_reproduction_package.zip --training-archive release\spot_od_v1_3_2_public_reproduction_training_inputs.zip
```

The verifier checks ZIP member safety, required runtime import members, metadata
coherence, unchanged metric readback from packaged JSON files, the extracted
`--help` smoke for `scripts/run_trajectory_candidate_graph_selector_poc.py`, and
training-input directory/manifests.

## Boundary

This is retained-candidate compact-simulator evidence only. It is not a new
scientific-metrics release, not public precise-reference validation, not
independent-machine reproduction, not a full raw/training/all-filter rerun, not
standalone learned recursive filtering, not broad learned orbit-determination
validation, not operational precise orbit determination, and not operational
learned orbit determination.
