# AdaptiveCandidateFusion Global Scenario Portfolio

Schema: `adaptive_candidate_fusion_global_portfolio.v1`

## Boundary

Internal compact-simulator validation-selected AdaptiveCandidateFusion portfolio evidence. Policies are selected from each run's stored validation split and applied to that run's held-out compact-simulator eval rows. This is not operational precise-reference validation, independent-machine reproduction, third-party validation, or a claim of universal learned orbit-determination performance.

## Selected Scenario Policies

| Scenario | Policy | Validation RMSE m | Validation Steps | Eval Wins/Rows | Mean Gain % | Min Gain % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `process_noise_shift_test` | `0.65*learned + 0.35*RFIS` | 372.179529 | 2988 | 14/15 | +6.16 | -3.69 |
| `maneuver_shift_test` | `0.55*learned + 0.45*EKF` | 554.194949 | 2293 | 11/15 | +1.43 | -10.11 |

## Policy Family Diagnostics

`nonlearned_only` is a validation-selected blend baseline: the same pooled validation RMSE selector is applied after excluding `learned` and `learned_hard` components.

| Family | Scenario Policies | Wins/Rows | Mean Gain % | Seed-Paired Wins/Seeds | Seed-Paired Mean Gain % | Seed-Paired Mean Gain 95% CI % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `all` | `process_noise_shift_test`: `0.65*learned + 0.35*RFIS`; `maneuver_shift_test`: `0.55*learned + 0.45*EKF` | 25/30 | +3.79 | 13/15 | +3.79 | [+2.05, +5.50] |
| `learned_including` | `process_noise_shift_test`: `0.65*learned + 0.35*RFIS`; `maneuver_shift_test`: `0.55*learned + 0.45*EKF` | 25/30 | +3.79 | 13/15 | +3.79 | [+2.05, +5.50] |
| `nonlearned_only` validation-selected blend baseline | `process_noise_shift_test`: `0.05*BatchWLS + 0.95*RFIS`; `maneuver_shift_test`: `0.90*EKF + 0.10*BatchWLS` | 19/30 | +0.71 | 9/15 | +0.71 | [-1.16, +2.63] |

## Eval Summary

Global scenario policies: 25/30 wins vs the best input candidate per row on observed-step position RMSE; mean gain +3.79%, median +4.08%, min -10.11%.

## Statistical Diagnostics

Exact p-values are one-sided sign/binomial tests for positive gain versus nonpositive gain; CIs are deterministic percentile bootstrap intervals for mean gain.

| Scope | n | Wins/Ties/Losses | Mean Gain % | Median Gain % | Min/Max Gain % | Sign p | Mean Gain 95% CI % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All rows | 30 | 25/0/5 | +3.79 | +4.08 | -10.11/+12.38 | 0.000162457 | [+1.83, +5.63] |
| `maneuver_shift_test` | 15 | 11/0/4 | +1.43 | +1.98 | -10.11/+9.86 | 0.0592346 | [-1.46, +4.05] |
| `process_noise_shift_test` | 15 | 14/0/1 | +6.16 | +7.15 | -3.69/+12.38 | 0.000488281 | [+4.03, +8.06] |
| Seed-paired means | 15 | 13/0/2 | +3.79 | +3.75 | -3.29/+11.12 | 0.00369263 | [+2.05, +5.50] |

## One Global Policy Diagnostic

`0.70*learned + 0.30*RFIS` selected across all scenarios: 22/30 wins, mean gain +2.21%, min -27.04%.

## Rows

| Seed | Scenario | Policy | Best Input | Portfolio RMSE m | Best Input RMSE m | Gain % | Result |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 7 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `EKF` | 311.363944 | 324.094657 | +3.93 | win |
| 7 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 515.491360 | 534.256057 | +3.51 | win |
| 11 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 358.116915 | 373.967374 | +4.24 | win |
| 11 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `RFIS` | 773.712128 | 845.490942 | +8.49 | win |
| 13 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `EKF` | 322.425796 | 328.253777 | +1.78 | win |
| 13 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `UKF` | 729.756262 | 774.000879 | +5.72 | win |
| 17 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 370.301649 | 405.098367 | +8.59 | win |
| 17 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 425.497792 | 420.833598 | -1.11 | loss |
| 19 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `EKF` | 370.583626 | 391.840144 | +5.42 | win |
| 19 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `RFIS` | 505.923106 | 508.444851 | +0.50 | win |
| 23 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 316.945881 | 341.343567 | +7.15 | win |
| 23 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 551.626436 | 591.187812 | +6.69 | win |
| 29 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 398.956613 | 445.165945 | +10.38 | win |
| 29 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 403.851012 | 414.133720 | +2.48 | win |
| 31 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `EKF` | 342.376131 | 390.771350 | +12.38 | win |
| 31 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `UKF` | 651.239669 | 722.436973 | +9.86 | win |
| 37 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 287.235108 | 309.653217 | +7.24 | win |
| 37 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 712.337066 | 646.905868 | -10.11 | loss |
| 41 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `UKF` | 375.178479 | 402.903716 | +6.88 | win |
| 41 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 645.544150 | 667.927114 | +3.35 | win |
| 43 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 317.102027 | 321.801267 | +1.46 | win |
| 43 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 470.063347 | 476.850044 | +1.42 | win |
| 47 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 388.896378 | 375.044095 | -3.69 | loss |
| 47 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 451.392950 | 438.734117 | -2.89 | loss |
| 53 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `EKF` | 278.172069 | 308.877985 | +9.94 | win |
| 53 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 443.511557 | 407.519262 | -8.83 | loss |
| 59 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 379.576211 | 412.771908 | +8.04 | win |
| 59 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 518.759073 | 529.263067 | +1.98 | win |
| 61 | `process_noise_shift_test` | `blend:learned:RFIS:0.65` | `RFIS` | 351.564304 | 384.628726 | +8.60 | win |
| 61 | `maneuver_shift_test` | `blend:learned:EKF:0.55` | `EKF` | 552.028550 | 554.261706 | +0.40 | win |

## Sources

- `results/adaptive_candidate_fusion_centered_fixed_soft_seed7_split7_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed11_split11_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed13_split13_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed17_split17_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed19_split19_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed23_split23_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed29_split29_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed31_split31_20260624`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed37_split37_20260624`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed41_split41_20260624`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed43_split43_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed47_split47_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed53_split53_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed59_split59_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed61_split61_20260623`
