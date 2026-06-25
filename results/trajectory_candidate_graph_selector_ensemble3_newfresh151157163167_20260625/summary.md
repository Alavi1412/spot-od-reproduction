# Trajectory Candidate Graph Selector PoC

Boundary: Retained-output compact-simulator trajectory candidate graph selector evidence only; not independent-machine, not operational, not full-rerun evidence.

Selector candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS
Best-single baseline candidate methods: EKF, UKF, AUKF, BatchWLS, RFIS, VA_RFIS

Selection uses retained truth-free candidate, visibility, eval-mask, scenario, and candidate-disagreement features only. Eval truth is used for scoring rows and baselines, not for selector decisions.

| Tier | Rows | Observed steps | Selector RMSE m | Best single RMSE m | Gain % |
| --- | ---: | ---: | ---: | ---: | ---: |
| development_seed_lt_67 | 119 | 3550 | 403.498 | 471.014 | 14.3342 |
| holdout_seed_ge_67 | 183 | 5405 | 403.803 | 463.124 | 12.8088 |
| future_seed_ge_109 | 60 | 1800 | 421.257 | 477.693 | 11.8143 |
| fresh_extra | 47 | 1426 | 403.648 | 445.943 | 9.48445 |
| all_eval_non_development | 230 | 6831 | 403.771 | 459.591 | 12.1455 |

The best-single denominator is selected per retained source run and scenario from the baseline candidate methods by observed-step RMSE.
This artifact is not independent-machine reproduction, not operational precise-reference validation, and not a full raw/training/all-filter rerun.
