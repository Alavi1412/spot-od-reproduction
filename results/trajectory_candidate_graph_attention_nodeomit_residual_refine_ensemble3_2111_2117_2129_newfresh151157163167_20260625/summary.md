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
Fit/training samples: 119
Validation samples: 0
Checkpoint selection metric: train_loss

Retained candidate-disagreement node features were omitted; pairwise edge distance/overlap features remain in the graph inputs.

Residual refinement uses retained truth-free candidate, visibility, eval-mask, scenario, and configured node features, then applies learned probability-weighted constant 3D candidate offsets. Eval truth is used for scoring rows and baselines, not for decisions.

| Tier | Rows | Observed steps | Refined RMSE m | Best single RMSE m | Gain % |
| --- | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 366.627 | 471.014 | 22.1622 |
| holdout_seed_ge_67 | 183 | 5405 | 376.194 | 463.124 | 18.7704 |
| future_seed_ge_109 | 60 | 1800 | 389.506 | 477.693 | 18.461 |
| fresh_extra | 47 | 1426 | 364.229 | 445.943 | 18.324 |
| all_eval_non_development | 230 | 6831 | 373.728 | 459.591 | 18.6825 |

The best-single denominator is selected per retained source run and scenario from the baseline candidate methods by observed-step RMSE.
This artifact is not independent-machine reproduction, not operational precise-reference validation, and not a full raw/training/all-filter rerun.
