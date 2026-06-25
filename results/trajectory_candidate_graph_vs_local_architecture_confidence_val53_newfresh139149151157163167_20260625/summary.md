# Architecture Summary Comparison

Name: graph_vs_local_architecture_confidence_val53

Boundary: Saved-row compact-simulator comparison only; not independent-machine reproduction, not operational precise-reference validation, and not full raw/training/all-filter rerun.

Selection already happened upstream. This script does not select using truth; it only compares saved selected-output fields on aligned rows.

This artifact is not independent-machine reproduction, not operational precise-reference validation, and not a full raw/training/all-filter rerun.

Left summary: graph_architecture_confidence_allgraphs_val53 (results\trajectory_candidate_graph_architecture_confidence_allgraphs_val53_newfresh139149151157163167_20260625\summary.json)
Right summary: local_architecture_confidence_alllocals_val53 (results\trajectory_candidate_local_architecture_confidence_alllocals_val53_newfresh139149151157163167_20260625\summary.json)
Row key fields: source_name, scenario, trajectory_row
Bootstrap samples: 5000 seed: 20260625

| Tier | Rows | Observed steps | Left pooled RMSE m | Right pooled RMSE m | Left gain % | 95% CI | Wins/Ties/Losses | Mean row gain % | Median row gain % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 406.364 | 408.699 | 0.571386 | [-0.131566, 1.41986] | 6/109/4 | 0.146769 | 0 |
| holdout_seed_ge_67 | 183 | 5405 | 393.739 | 402.371 | 2.14529 | [0.898207, 3.54429] | 17/158/8 | 1.32589 | 0 |
| future_seed_ge_109 | 60 | 1800 | 410.395 | 426.069 | 3.67873 | [1.24758, 7.01316] | 9/49/2 | 2.74742 | 0 |
| fresh_extra | 70 | 2087 | 427.264 | 431.885 | 1.07006 | [-0.419739, 3.0012] | 5/57/8 | -1.17297 | 0 |
| all_eval_non_development | 253 | 7492 | 403.358 | 410.806 | 1.81299 | [0.839985, 2.90617] | 22/215/16 | 0.634507 | 0 |
