# SPOT-OD v1.3.0 Edge-Only Residual-Refinement Ablation

This release adds the edge-only retained-candidate residual-refinement ablation and
the supporting comparison artifacts for the current strongest graph neural network
proof of concept.

## Main Result

The edge-only attention graph residual-refinement ensemble uses member seeds
2111, 2117, and 2129 with `prediction_mode=residual_refine`,
`residual_loss_weight=1e-5`, two attention graph layers, and
`--node-disagreement-features omit`. Omitting node-disagreement aggregates
reduces node features from 30 to 22 while preserving 10 pairwise edge features
for graph message passing.

Selection status: this edge-only run used 119 fit/training samples, 0 validation
samples, and train-loss checkpoint selection. It is therefore a post-freeze
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
is therefore limited to edge-only attention versus the matched no-message local
control.

The saved-row local tail diagnostic aligns the attention, local/no-message, and
mean graph rows by `(source_name, scenario, trajectory_row)`. It shows the weak
local aggregate is driven by saved-row tail failures, while attention-vs-mean
remains weak/mixed.

## Added Evidence Artifacts

- `results/trajectory_candidate_graph_attention_nodeomit_residual_refine_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_mean_nodeomit_residual_refine_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_local_nodeomit_residual_refine_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- `results/trajectory_candidate_graph_vs_local_architecture_confidence_val53_newfresh139149151157163167_20260625/`
- `results/trajectory_candidate_edge_only_local_tail_diagnostic_20260625/`
- `paper/figures/trajectory_residual_refine_gain_distribution.png`

## Added Reproduction Scripts

- `scripts/build_trajectory_residual_refine_comparison_intervals.py`
- `scripts/build_trajectory_residual_refine_figure.py`
- `scripts/build_trajectory_residual_refine_tail_diagnostic.py`
- `scripts/compare_trajectory_candidate_architecture_summaries.py`

## Verification

Focused verification commands run in the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_trajectory_residual_refine_comparison_intervals.py tests\test_compare_trajectory_candidate_architecture_summaries.py tests\test_trajectory_candidate_graph_selector_poc.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_build_trajectory_residual_refine_tail_diagnostic.py -q
.\.venv\Scripts\python.exe scripts\build_trajectory_residual_refine_comparison_intervals.py
.\.venv\Scripts\python.exe scripts\build_trajectory_residual_refine_tail_diagnostic.py
.\.venv\Scripts\python.exe scripts\compare_trajectory_candidate_architecture_summaries.py --name graph_vs_local_architecture_confidence_val53 --left-summary results\trajectory_candidate_graph_architecture_confidence_allgraphs_val53_newfresh139149151157163167_20260625\summary.json --right-summary results\trajectory_candidate_local_architecture_confidence_alllocals_val53_newfresh139149151157163167_20260625\summary.json --output-dir results\trajectory_candidate_graph_vs_local_architecture_confidence_val53_newfresh139149151157163167_20260625 --bootstrap-samples 5000 --bootstrap-seed 20260625
.\.venv\Scripts\python.exe scripts\build_trajectory_residual_refine_figure.py
.\.venv\Scripts\python.exe scripts\compile_paper.py
```

The original focused suite passed with 44 tests, and the added tail-diagnostic
test passes separately with 2 tests. The paper build completed and produced
`paper/main.pdf`; the standalone main build still reports pre-existing
supplement-side unresolved references.

## Scope Boundary

This is retained-candidate compact-simulator evidence only. It is not
operational precise orbit determination, not public precise-reference validation,
not independent-machine reproduction, not a full raw/training/all-filter rerun,
not standalone learned recursive filtering, not broad learned orbit-determination
validation, and not operational learned orbit determination. The edge-only
selection remains train-loss based with no validation split, so it is exploratory
rather than confirmatory.

Zenodo DOI assignment for this version is pending creation of the corresponding
GitHub release.
