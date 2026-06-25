# SPOT-OD v1.3.1 validation-selected release package

This package is an inspection and downstream-replay archive for the validation-selected edge-only retained-candidate attention graph residual-refinement proof of concept.

## Release identifiers

- GitHub release: <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.1-validation-selected-residual-refine>
- Release tag: `v1.3.1-validation-selected-residual-refine`
- Tag commit: `c4882f1b367426c0966e906b9332f64d44d2279f`
- GitHub release asset: `spot_od_v1_3_1_validation_selected_residual_refine.zip`
- GitHub release asset integrity: corrective replacement asset byte size and SHA-256 are reported externally after archive construction, not embedded in this packaged README.
- Zenodo record: <https://zenodo.org/records/20844596>
- Zenodo version DOI: `10.5281/zenodo.20844596`
- Zenodo concept DOI: `10.5281/zenodo.20768672`
- Zenodo archived source file: `Alavi1412/spot-od-reproduction-v1.3.1-validation-selected-residual-refine.zip`
- Zenodo archived source file bytes: `212,947,668`
- Zenodo archived source file MD5: `863e5077d4d29a827c6fcfd1181dce34`

## What is included

- Manuscript source/PDF, validation-selected table, and validation-selected figure.
- Final rows, summaries, comparison intervals, and checkpoints for attention, mean-graph, and local/no-message residual-refinement runs.
- Local-control tail diagnostic rows and summary.
- Scripts for the retained-candidate graph selector, comparison intervals, tail diagnostic, and figure regeneration.
- A focused test for the graph architecture ensemble evidence.
- Zenodo/GitHub metadata, release notes, citation metadata, package manifest, and license notice.

## What is not included

The archive is not a full raw/training/all-filter rerun package. The training provenance command in the release notes depends on upstream retained-candidate input directories named `results/adaptive_candidate_fusion_observed_fixed_soft_*`; those upstream candidate-input directories are not bundled in this zip. The zip supports inspection of the selected PoC outputs and downstream replay of interval/figure artifacts from the included result directories.

This is not independent-machine reproduction, not public precise-reference validation, not standalone learned recursive filtering, not operational precise orbit determination, not broad learned orbit-determination validation, and not operational learned orbit determination.
