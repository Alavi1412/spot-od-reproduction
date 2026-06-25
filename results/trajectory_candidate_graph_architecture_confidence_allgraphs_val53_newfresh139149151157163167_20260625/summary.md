# Trajectory Candidate Graph Architecture Confidence Ensemble

Name: graph_architecture_confidence_allgraphs_val53

Boundary: Retained-output compact-simulator trajectory candidate graph selector evidence only; not independent-machine, not operational, not full-rerun evidence.

Selection rule: choose the graph architecture member with the largest `selected_probability` for each aligned row.

This selection uses only selected probabilities from graph model outputs. It does not use evaluation truth, candidate RMSE, best-single RMSE, labels, or reference/local-control outputs to choose a member. Truth/error fields are used only after selection for scoring.

Graph summaries: 10
Local/no-message members allowed: no
Graph members required: yes
Local/no-message member count: 0
Reference summary: trajectory_candidate_local_selector_ensemble9_seed2001_2005_2009_2011_2017_2023_2029_2039_2053_val53_newfresh139149151157163167_20260625

| Tier | Rows | Observed steps | Architecture ensemble RMSE m | Best single RMSE m | Gain vs best single % | Reference RMSE m | Gain vs reference % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 406.364 | 471.014 | 13.7257 | 408.699 | 0.571386 |
| holdout_seed_ge_67 | 183 | 5405 | 393.739 | 463.124 | 14.982 | 402.371 | 2.14529 |
| future_seed_ge_109 | 60 | 1800 | 410.395 | 477.693 | 14.0882 | 426.069 | 3.67873 |
| fresh_extra | 70 | 2087 | 427.264 | 481.79 | 11.3174 | 431.885 | 1.07006 |
| all_eval_non_development | 253 | 7492 | 403.358 | 468.399 | 13.8858 | 410.806 | 1.81299 |

The best-single denominator is inherited from the aligned selector summaries. Reference metrics, when present, are comparison-only and are not inputs to architecture selection.
