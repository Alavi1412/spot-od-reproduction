# Trajectory Candidate Graph Selector PoC

Boundary: Retained-output compact-simulator trajectory candidate graph selector evidence only; not independent-machine, not operational, not full-rerun evidence.

Selector candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS
Best-single baseline candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS
Graph layer type: attention
Graph layers: 2

Selection uses retained truth-free candidate, visibility, eval-mask, scenario, and candidate-disagreement features only. Eval truth is used for scoring rows and baselines, not for selector decisions.

| Tier | Rows | Observed steps | Selector RMSE m | Best single RMSE m | Gain % |
| --- | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 403.498 | 471.014 | 14.3342 |
| holdout_seed_ge_67 | 183 | 5405 | 399.141 | 463.124 | 13.8156 |
| future_seed_ge_109 | 60 | 1800 | 423.451 | 477.693 | 11.355 |
| fresh_extra | 47 | 1426 | 387.159 | 445.943 | 13.1819 |
| all_eval_non_development | 230 | 6831 | 396.67 | 459.591 | 13.6907 |

The best-single denominator is selected per retained source run and scenario from the baseline candidate methods by observed-step RMSE.
This artifact is not independent-machine reproduction, not operational precise-reference validation, and not a full raw/training/all-filter rerun.
