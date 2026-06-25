# Trajectory Candidate Graph Selector PoC

Boundary: Retained-output compact-simulator trajectory candidate graph selector evidence only; not independent-machine, not operational, not full-rerun evidence.

Selector candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS
Best-single baseline candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS
Graph layer type: attention
Graph layers: 2
Prediction mode: residual_refine
Node disagreement features: omit
Residual loss weight: 1e-05
Residual offset application: per_candidate_constant_position_offset
Development samples: 119
Fit/training samples: 82
Validation samples: 37
Checkpoint selection metric: validation_loss

Retained candidate-disagreement node features were omitted; pairwise edge distance/overlap features remain in the graph inputs.

Residual refinement uses retained truth-free candidate, visibility, eval-mask, scenario, and configured node features, then applies learned probability-weighted constant 3D candidate offsets. Eval truth is used for scoring rows and baselines, not for decisions.

| Tier | Rows | Observed steps | Refined RMSE m | Best single RMSE m | Gain % |
| --- | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 394.133 | 471.014 | 16.3223 |
| holdout_seed_ge_67 | 183 | 5405 | 391.351 | 463.124 | 15.4977 |
| future_seed_ge_109 | 60 | 1800 | 405.443 | 477.693 | 15.1247 |
| fresh_extra | 47 | 1426 | 386.373 | 445.943 | 13.3582 |
| all_eval_non_development | 230 | 6831 | 390.317 | 459.591 | 15.073 |

The best-single denominator is selected per retained source run and scenario from the baseline candidate methods by observed-step RMSE.
This artifact is not independent-machine reproduction, not operational precise-reference validation, and not a full raw/training/all-filter rerun.
