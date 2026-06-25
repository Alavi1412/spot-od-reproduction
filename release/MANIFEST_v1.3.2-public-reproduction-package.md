# SPOT-OD v1.3.2 public reproduction package manifest

Release tag: `v1.3.2-public-reproduction-package`
Release URL:
<https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.2-public-reproduction-package>
Main GitHub release asset: `spot_od_v1_3_2_public_reproduction_package.zip`
Training-input GitHub release asset:
`spot_od_v1_3_2_public_reproduction_training_inputs.zip`
Zenodo v1.3.2 version DOI: assigned after GitHub release import; not claimed in
this package.
Previous version DOI: `10.5281/zenodo.20844596`
Concept DOI: `10.5281/zenodo.20768672`
Repaired v1.3.1 archive SHA-256:
`4d575f7f8d3326823dc50f71f5f542dab1f924780082f8b6f00195cbf22619a4`

Archive byte sizes and SHA-256 digests are reported after construction rather
than embedded inside this packaged manifest.

## Core files

- `.zenodo.json`
- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `release/ZENODO_METADATA.json`
- `release/RELEASE_NOTES_v1.3.2-public-reproduction-package.md`
- `release/README.md`
- `release/README_v1.3.2-public-reproduction-package.md`
- `release/MANIFEST_v1.3.2-public-reproduction-package.md`
- `release/CITATION.cff`
- `release/LICENSE_CC_BY_4_0.txt`
- `release/spot_od_v1_3_1_validation_selected_residual_refine.zip`
- `paper/main.tex`
- `paper/main.pdf`
- `paper/tables/main_findings_summary.tex`
- `paper/tables/main_revision_delta_and_public_repro.tex`
- `paper/tables/main_trajectory_graph_selector_ensemble_poc.tex`
- `paper/tables/main_row_weighted_dls_poc.tex`
- `paper/figures/trajectory_residual_refine_gain_distribution_val53.png`

## v1.3.1 repair traceability

- `release/README_v1.3.1-validation-selected-residual-refine.md`
- `release/RELEASE_NOTES_v1.3.1-validation-selected-residual-refine.md`
- `release/MANIFEST_v1.3.1-validation-selected-residual-refine.md`
- `scripts/verify_v131_release_package.py`
- `tests/test_v131_release_package_verification.py`

## Scripts and tests

- `scripts/__init__.py`
- `scripts/_bootstrap.py`
- `scripts/run_trajectory_candidate_graph_selector_poc.py`
- `scripts/analyze_trajectory_candidate_graph_architecture_ensemble.py`
- `scripts/build_trajectory_residual_refine_comparison_intervals.py`
- `scripts/build_trajectory_residual_refine_tail_diagnostic.py`
- `scripts/build_trajectory_residual_refine_figure.py`
- `scripts/verify_v132_public_reproduction_package.py`
- `tests/conftest.py`
- `tests/test_trajectory_candidate_graph_architecture_ensemble.py`
- `tests/test_build_trajectory_residual_refine_comparison_intervals.py`
- `tests/test_build_trajectory_residual_refine_tail_diagnostic.py`
- `tests/test_build_trajectory_residual_refine_figure.py`
- `tests/test_v132_public_reproduction_package_verification.py`

## Runtime source package

- `src/gnn_state_estimation/`

The source package and bootstrap helper are included so released scripts can
resolve imports when run directly from the extracted archive root. The verifier
checks this with `python scripts/run_trajectory_candidate_graph_selector_poc.py
--help`; this is an import/help smoke, not model training.

## Result directories

- `results/trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_edge_only_local_tail_diagnostic_val53_20260625/`

## Training-input ZIP

The separate training-input ZIP contains the checkpoint-free upstream retained
candidate source directories:

- `results/adaptive_candidate_fusion_observed_fixed_soft_seed*_split*_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625`
- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625`
- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625`
- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed167_split167_20260625`

It includes v1.3.2 training-input manifest files under `release/` and omits
upstream checkpoints.

## Boundary

The included outputs support the bounded retained-candidate compact-simulator
claim reported in the manuscript. They do not constitute new scientific metrics.
This package is not public precise-reference validation, not
independent-machine reproduction, and not a full raw/training/all-filter rerun.
