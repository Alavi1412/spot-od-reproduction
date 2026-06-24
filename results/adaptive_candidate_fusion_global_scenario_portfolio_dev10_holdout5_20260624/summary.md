# AdaptiveCandidateFusion Global Scenario Portfolio

Schema: `adaptive_candidate_fusion_global_portfolio.v1`

## Boundary

Internal compact-simulator development/holdout AdaptiveCandidateFusion portfolio evidence. Policies are selected only from validation recomputations for selection_run_dirs and are applied to saved compact-simulator eval rows from eval_run_dirs. This run-directory split is an internal development check; it is not independent-machine reproduction, operational precise-reference validation, third-party validation, or a claim of universal learned orbit-determination performance.

## Selection/Eval Split

Policies are selected from development `selection_run_dirs` validation records and evaluated on holdout `eval_run_dirs` saved compact-simulator eval rows. This is internal compact-simulator development/holdout evidence, not independent-machine or precise-reference validation.

Selection run directories:

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

Eval run directories:

- `results/adaptive_candidate_fusion_centered_fixed_soft_seed43_split43_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed47_split47_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed53_split53_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed59_split59_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed61_split61_20260623`

## Selected Scenario Policies

| Scenario | Policy | Validation RMSE m | Validation Steps | Eval Wins/Rows | Mean Gain % | Min Gain % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `process_noise_shift_test` | `0.60*learned + 0.40*RFIS` | 367.444516 | 1920 | 4/5 | +4.95 | -2.86 |
| `maneuver_shift_test` | `0.60*learned + 0.40*EKF` | 562.509076 | 1577 | 3/5 | -2.06 | -10.10 |

## Policy Family Diagnostics

`nonlearned_only` is a validation-selected blend baseline: the same pooled validation RMSE selector is applied after excluding `learned` and `learned_hard` components.

| Family | Scenario Policies | Wins/Rows | Mean Gain % | Seed-Paired Wins/Seeds | Seed-Paired Mean Gain % | Seed-Paired Mean Gain 95% CI % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `all` | `process_noise_shift_test`: `0.60*learned + 0.40*RFIS`; `maneuver_shift_test`: `0.60*learned + 0.40*EKF` | 7/10 | +1.45 | 3/5 | +1.45 | [-1.18, +4.02] |
| `learned_including` | `process_noise_shift_test`: `0.60*learned + 0.40*RFIS`; `maneuver_shift_test`: `0.60*learned + 0.40*EKF` | 7/10 | +1.45 | 3/5 | +1.45 | [-1.18, +4.02] |
| `nonlearned_only` validation-selected blend baseline | `process_noise_shift_test`: `0.05*BatchWLS + 0.95*RFIS`; `maneuver_shift_test`: `0.90*EKF + 0.10*BatchWLS` | 6/10 | +0.11 | 3/5 | +0.11 | [-2.31, +2.68] |

## Eval Summary

Global scenario policies: 7/10 wins vs the best input candidate per row on observed-step position RMSE; mean gain +1.45%, median +1.54%, min -10.10%.

## Statistical Diagnostics

Exact p-values are one-sided sign/binomial tests for positive gain versus nonpositive gain; CIs are deterministic percentile bootstrap intervals for mean gain.

| Scope | n | Wins/Ties/Losses | Mean Gain % | Median Gain % | Min/Max Gain % | Sign p | Mean Gain 95% CI % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All rows | 10 | 7/0/3 | +1.45 | +1.54 | -10.10/+9.57 | 0.171875 | [-2.22, +4.96] |
| `maneuver_shift_test` | 5 | 3/0/2 | -2.06 | +0.31 | -10.10/+2.01 | 0.5 | [-6.41, +1.37] |
| `process_noise_shift_test` | 5 | 4/0/1 | +4.95 | +7.85 | -2.86/+9.57 | 0.1875 | [+0.32, +8.75] |
| Seed-paired means | 5 | 3/0/2 | +1.45 | +1.54 | -3.32/+4.93 | 0.5 | [-1.18, +4.02] |

## One Global Policy Diagnostic

`0.75*learned + 0.25*RFIS` selected across all scenarios: 5/10 wins, mean gain -1.59%, min -26.19%.

## Rows

| Seed | Scenario | Policy | Best Input | Portfolio RMSE m | Best Input RMSE m | Gain % | Result |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 43 | `process_noise_shift_test` | `blend:learned:RFIS:0.6` | `RFIS` | 315.959590 | 321.801267 | +1.82 | win |
| 43 | `maneuver_shift_test` | `blend:learned:EKF:0.6` | `EKF` | 470.825704 | 476.850044 | +1.26 | win |
| 47 | `process_noise_shift_test` | `blend:learned:RFIS:0.6` | `RFIS` | 385.786693 | 375.044095 | -2.86 | loss |
| 47 | `maneuver_shift_test` | `blend:learned:EKF:0.6` | `EKF` | 455.292220 | 438.734117 | -3.77 | loss |
| 53 | `process_noise_shift_test` | `blend:learned:RFIS:0.6` | `EKF` | 279.326255 | 308.877985 | +9.57 | win |
| 53 | `maneuver_shift_test` | `blend:learned:EKF:0.6` | `EKF` | 448.661310 | 407.519262 | -10.10 | loss |
| 59 | `process_noise_shift_test` | `blend:learned:RFIS:0.6` | `RFIS` | 380.369745 | 412.771908 | +7.85 | win |
| 59 | `maneuver_shift_test` | `blend:learned:EKF:0.6` | `EKF` | 518.614969 | 529.263067 | +2.01 | win |
| 61 | `process_noise_shift_test` | `blend:learned:RFIS:0.6` | `RFIS` | 352.418414 | 384.628726 | +8.37 | win |
| 61 | `maneuver_shift_test` | `blend:learned:EKF:0.6` | `EKF` | 552.567013 | 554.261706 | +0.31 | win |

## Sources

Selection run directories:

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

Eval run directories:

- `results/adaptive_candidate_fusion_centered_fixed_soft_seed43_split43_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed47_split47_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed53_split53_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed59_split59_20260623`
- `results/adaptive_candidate_fusion_centered_fixed_soft_seed61_split61_20260623`
