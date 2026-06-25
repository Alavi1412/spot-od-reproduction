# SPOT-OD v1.3.1 validation-selected release package

This package is an inspection and downstream-replay archive for the validation-selected edge-only retained-candidate attention graph residual-refinement proof of concept.

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