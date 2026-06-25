# SPOT-OD v1.3.1 validation-selected package manifest

Release tag: `v1.3.1-validation-selected-residual-refine`
Release URL: <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.1-validation-selected-residual-refine>
Tag commit: `c4882f1b367426c0966e906b9332f64d44d2279f`
GitHub release asset: `spot_od_v1_3_1_validation_selected_residual_refine.zip`
GitHub release asset integrity: corrective replacement asset byte size and SHA-256 are reported externally after archive construction, not embedded in this packaged manifest.
Zenodo record: <https://zenodo.org/records/20844596>
Version DOI: `10.5281/zenodo.20844596`
Concept DOI: `10.5281/zenodo.20768672`
Zenodo archived source file: `Alavi1412/spot-od-reproduction-v1.3.1-validation-selected-residual-refine.zip`
Zenodo archived source file bytes: `212,947,668`
Zenodo archived source file MD5: `863e5077d4d29a827c6fcfd1181dce34`

## Core files

- `.zenodo.json`
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `release/ZENODO_METADATA.json`
- `release/RELEASE_NOTES_v1.3.1-validation-selected-residual-refine.md`
- `release/README.md`
- `release/README_v1.3.1-validation-selected-residual-refine.md`
- `release/MANIFEST_v1.3.1-validation-selected-residual-refine.md`
- `release/CITATION.cff`
- `release/LICENSE_CC_BY_4_0.txt`
- `paper/main.tex`
- `paper/main.pdf`
- `paper/tables/main_findings_summary.tex`
- `paper/tables/main_revision_delta_and_public_repro.tex`
- `paper/tables/main_trajectory_graph_selector_ensemble_poc.tex`
- `paper/tables/main_row_weighted_dls_poc.tex`
- `paper/figures/trajectory_residual_refine_gain_distribution_val53.png`

## Scripts and tests

- `scripts/__init__.py`
- `scripts/_bootstrap.py`
- `scripts/run_trajectory_candidate_graph_selector_poc.py`
- `scripts/analyze_trajectory_candidate_graph_architecture_ensemble.py`
- `scripts/build_trajectory_residual_refine_comparison_intervals.py`
- `scripts/build_trajectory_residual_refine_tail_diagnostic.py`
- `scripts/build_trajectory_residual_refine_figure.py`
- `tests/test_trajectory_candidate_graph_architecture_ensemble.py`
- `tests/test_build_trajectory_residual_refine_comparison_intervals.py`
- `tests/test_build_trajectory_residual_refine_tail_diagnostic.py`
- `tests/test_build_trajectory_residual_refine_figure.py`

## Runtime source package

- `src/gnn_state_estimation/`

The source package and bootstrap helper are included so released scripts can resolve imports when run directly from the extracted archive root. The verifier checks this with `python scripts/run_trajectory_candidate_graph_selector_poc.py --help`; this is an import/help smoke, not model training.

## Result directories

- `results/trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_edge_only_local_tail_diagnostic_val53_20260625/`

## Boundary

The included outputs support the bounded retained-candidate compact-simulator claim reported in the manuscript. They do not include the upstream retained-candidate candidate-input directories needed to rerun training from scratch.
