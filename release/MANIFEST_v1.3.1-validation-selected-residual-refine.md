# SPOT-OD v1.3.1 validation-selected package manifest

Release tag: `v1.3.1-validation-selected-residual-refine`
Concept DOI: `10.5281/zenodo.20768672`
Version DOI: pending Zenodo import after GitHub release creation

## Core files

- `.zenodo.json`
- `release/ZENODO_METADATA.json`
- `release/RELEASE_NOTES_v1.3.1-validation-selected-residual-refine.md`
- `release/README_v1.3.1-validation-selected-residual-refine.md`
- `release/MANIFEST_v1.3.1-validation-selected-residual-refine.md`
- `release/CITATION.cff`
- `release/LICENSE_CC_BY_4_0.txt`
- `paper/main.tex`
- `paper/main.pdf`
- `paper/tables/main_trajectory_graph_selector_ensemble_poc.tex`
- `paper/tables/main_row_weighted_dls_poc.tex`
- `paper/figures/trajectory_residual_refine_gain_distribution_val53.png`

## Scripts and tests

- `scripts/run_trajectory_candidate_graph_selector_poc.py`
- `scripts/analyze_trajectory_candidate_graph_architecture_ensemble.py`
- `scripts/build_trajectory_residual_refine_comparison_intervals.py`
- `scripts/build_trajectory_residual_refine_tail_diagnostic.py`
- `scripts/build_trajectory_residual_refine_figure.py`
- `tests/test_trajectory_candidate_graph_architecture_ensemble.py`
- `tests/test_build_trajectory_residual_refine_comparison_intervals.py`
- `tests/test_build_trajectory_residual_refine_tail_diagnostic.py`
- `tests/test_build_trajectory_residual_refine_figure.py`

## Result directories

- `results/trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_edge_only_local_tail_diagnostic_val53_20260625/`

## Boundary

The included outputs support the bounded retained-candidate compact-simulator claim reported in the manuscript. They do not include the upstream retained-candidate candidate-input directories needed to rerun training from scratch.