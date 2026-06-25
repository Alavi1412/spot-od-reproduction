# Trajectory Candidate Graph Selector PoC

Boundary: Retained-output compact-simulator trajectory candidate graph selector evidence only; not independent-machine, not operational, not full-rerun evidence.

Selector candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS
Best-single baseline candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS
Graph layer type: mean
Graph layers: 2
Prediction mode: residual_refine
Residual loss weight: 1e-05
Residual offset application: per_candidate_constant_position_offset

Residual refinement uses retained truth-free candidate, visibility, eval-mask, scenario, and candidate-disagreement features, then applies learned probability-weighted constant 3D candidate offsets. Eval truth is used for scoring rows and baselines, not for decisions.

| Tier | Rows | Observed steps | Refined RMSE m | Best single RMSE m | Gain % |
| --- | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 368.273 | 471.014 | 21.8126 |
| holdout_seed_ge_67 | 183 | 5405 | 467.215 | 463.124 | -0.883369 |
| future_seed_ge_109 | 60 | 1800 | 605.514 | 477.693 | -26.7579 |
| fresh_extra | 47 | 1426 | 962.57 | 445.943 | -115.85 |
| all_eval_non_development | 230 | 6831 | 605.095 | 459.591 | -31.6596 |

The best-single denominator is selected per retained source run and scenario from the baseline candidate methods by observed-step RMSE.
This artifact is not independent-machine reproduction, not operational precise-reference validation, and not a full raw/training/all-filter rerun.
