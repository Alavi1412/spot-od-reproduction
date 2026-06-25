# Trajectory Candidate Graph Architecture Confidence Ensemble

Name: local_architecture_confidence_alllocals_val53

Boundary: Retained-output compact-simulator trajectory candidate graph selector evidence only; not independent-machine, not operational, not full-rerun evidence.

Selection rule: choose the member with the largest `selected_probability` for each aligned row.

This selection uses only selected probabilities from member summaries. It does not use evaluation truth, candidate RMSE, best-single RMSE, labels, or reference/local-control outputs to choose a member. Truth/error fields are used only after selection for scoring.

Graph summaries: 9
Local/no-message members allowed: yes
Graph members required: no
Local/no-message member count: 9
Reference summary: trajectory_candidate_graph_architecture_confidence_allgraphs_val53_newfresh139149151157163167_20260625

| Tier | Rows | Observed steps | Architecture ensemble RMSE m | Best single RMSE m | Gain vs best single % | Reference RMSE m | Gain vs reference % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 408.699 | 471.014 | 13.2299 | 406.364 | -0.57467 |
| holdout_seed_ge_67 | 183 | 5405 | 402.371 | 463.124 | 13.1181 | 393.739 | -2.19232 |
| future_seed_ge_109 | 60 | 1800 | 426.069 | 477.693 | 10.807 | 410.395 | -3.81922 |
| fresh_extra | 70 | 2087 | 431.885 | 481.79 | 10.3582 | 427.264 | -1.08164 |
| all_eval_non_development | 253 | 7492 | 410.806 | 468.399 | 12.2957 | 403.358 | -1.84647 |

The best-single denominator is inherited from the aligned selector summaries. Reference metrics, when present, are comparison-only and are not inputs to architecture selection.
