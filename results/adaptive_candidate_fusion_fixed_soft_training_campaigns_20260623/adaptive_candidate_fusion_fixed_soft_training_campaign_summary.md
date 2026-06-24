# AdaptiveCandidateFusion Fixed-Soft Full-Training Campaigns

Schema: `adaptive_candidate_fusion_fixed_soft_training_campaigns.v1`

## Boundary

Full-training AdaptiveCandidateFusion fixed-soft campaign packaging for local current-workspace compact-simulator evidence. These runs retrain from the materialized campaign inputs and are not skip-training replays, as checked by non-empty train/validation histories and checkpoint files. They are not part of the published public v1.2.1 package unless a later release includes them; not independent-machine reproduction; not operational precise-reference validation; not third-party validation; not a full raw/all-filter/public rerun; and not a universal learned-OD claim.

## Campaign Summary

### Centered training-step mask, fixed-soft full retraining

Positive compact-simulator PoC: centered training plus fixed-soft full retraining reproduces the 8/10 observed-step result.

Observed-step: 8/10 row wins, 3/5 paired seeds, mean gain +2.594443%, min -11.879590%, max +12.020436%.

All-step caveat: 5/10 row wins, 0/5 paired seeds, mean gain -10.853658%, min -124.850085%, max +20.502028%.

Training mask `centered`; validation selection metric `all_step_pos_rmse_m`.

### Observed training-step mask, fixed-soft full retraining

Bounded negative/failure mode: observed-mask full retraining is negative overall, with large maneuver failures despite fixed-soft inference.

Observed-step: 12/20 row wins, 3/10 paired seeds, mean gain -6.191539%, min -164.721000%, max +13.302773%.

All-step caveat: 11/20 row wins, 1/10 paired seeds, mean gain -11.994364%, min -119.719458%, max +21.364779%.

Training mask `observed`; validation selection metric `observed_step_pos_rmse_m`.

## Rows

| Campaign | Seed | Scenario | Best Observed Input | Observed RMSE | Observed Gain % | Best All-Step Input | All-Step RMSE | All-Step Gain % |
| --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |
| centered_fixed_soft_full_retraining | 7 | `maneuver_shift_test` | `EKF` | 527.011710 | +1.355969 | `RFIS` | 6809.484692 | +20.502028 |
| centered_fixed_soft_full_retraining | 7 | `process_noise_shift_test` | `EKF` | 316.097063 | +2.467672 | `BatchWLS` | 4835.376647 | -124.850085 |
| centered_fixed_soft_full_retraining | 11 | `maneuver_shift_test` | `RFIS` | 743.859246 | +12.020436 | `VA_RFIS` | 11722.321060 | -5.838744 |
| centered_fixed_soft_full_retraining | 11 | `process_noise_shift_test` | `RFIS` | 372.271363 | +0.453518 | `EKF` | 6948.632445 | +1.164778 |
| centered_fixed_soft_full_retraining | 13 | `maneuver_shift_test` | `UKF` | 713.471148 | +7.820370 | `VA_RFIS` | 20801.127696 | -0.260606 |
| centered_fixed_soft_full_retraining | 13 | `process_noise_shift_test` | `EKF` | 339.480253 | -3.420060 | `EKF` | 3473.412361 | +19.729374 |
| centered_fixed_soft_full_retraining | 17 | `maneuver_shift_test` | `EKF` | 470.826903 | -11.879590 | `UKF` | 10965.154122 | +4.684325 |
| centered_fixed_soft_full_retraining | 17 | `process_noise_shift_test` | `RFIS` | 362.998107 | +10.392602 | `VA_RFIS` | 7926.903961 | -27.807706 |
| centered_fixed_soft_full_retraining | 19 | `maneuver_shift_test` | `RFIS` | 505.410354 | +0.596819 | `EKF` | 5989.032477 | +12.975056 |
| centered_fixed_soft_full_retraining | 19 | `process_noise_shift_test` | `EKF` | 367.794097 | +6.136698 | `VA_RFIS` | 6457.201604 | -8.834997 |
| observed_mask_fixed_soft_full_retraining | 23 | `maneuver_shift_test` | `EKF` | 537.579287 | +9.067935 | `BatchWLS` | 10598.349833 | +2.370479 |
| observed_mask_fixed_soft_full_retraining | 23 | `process_noise_shift_test` | `RFIS` | 298.074070 | +12.676230 | `BatchWLS` | 14763.414896 | -2.765980 |
| observed_mask_fixed_soft_full_retraining | 29 | `maneuver_shift_test` | `EKF` | 414.341592 | -0.050195 | `VA_RFIS` | 6278.731355 | +19.193263 |
| observed_mask_fixed_soft_full_retraining | 29 | `process_noise_shift_test` | `RFIS` | 393.148374 | +11.684984 | `VA_RFIS` | 15089.245847 | -9.813347 |
| observed_mask_fixed_soft_full_retraining | 31 | `maneuver_shift_test` | `UKF` | 1912.442376 | -164.721000 | `VA_RFIS` | 11034.623001 | +1.835263 |
| observed_mask_fixed_soft_full_retraining | 31 | `process_noise_shift_test` | `EKF` | 338.787922 | +13.302773 | `EKF` | 13250.673442 | +2.785976 |
| observed_mask_fixed_soft_full_retraining | 37 | `maneuver_shift_test` | `EKF` | 633.316106 | +2.100732 | `VA_RFIS` | 4933.307650 | +0.112814 |
| observed_mask_fixed_soft_full_retraining | 37 | `process_noise_shift_test` | `RFIS` | 288.496476 | +6.832398 | `VA_RFIS` | 4108.043277 | -119.719458 |
| observed_mask_fixed_soft_full_retraining | 41 | `maneuver_shift_test` | `EKF` | 589.376802 | +11.760312 | `UKF` | 17978.446709 | +2.620019 |
| observed_mask_fixed_soft_full_retraining | 41 | `process_noise_shift_test` | `UKF` | 367.622072 | +8.756843 | `VA_RFIS` | 3984.376217 | -43.435124 |
| observed_mask_fixed_soft_full_retraining | 43 | `maneuver_shift_test` | `EKF` | 471.133579 | +1.198797 | `VA_RFIS` | 20948.287803 | -0.400461 |
| observed_mask_fixed_soft_full_retraining | 43 | `process_noise_shift_test` | `RFIS` | 328.803193 | -2.175854 | `VA_RFIS` | 12261.219319 | +0.855517 |
| observed_mask_fixed_soft_full_retraining | 47 | `maneuver_shift_test` | `EKF` | 488.919582 | -11.438697 | `UKF` | 20620.657899 | +3.047933 |
| observed_mask_fixed_soft_full_retraining | 47 | `process_noise_shift_test` | `RFIS` | 422.326080 | -12.607047 | `VA_RFIS` | 3858.532011 | -8.188014 |
| observed_mask_fixed_soft_full_retraining | 53 | `maneuver_shift_test` | `EKF` | 493.273915 | -21.043092 | `BatchWLS` | 23654.976008 | +0.229612 |
| observed_mask_fixed_soft_full_retraining | 53 | `process_noise_shift_test` | `EKF` | 273.432388 | +11.475598 | `VA_RFIS` | 3927.037347 | -28.758058 |
| observed_mask_fixed_soft_full_retraining | 59 | `maneuver_shift_test` | `EKF` | 545.706440 | -3.106843 | `VA_RFIS` | 10772.735645 | -13.483828 |
| observed_mask_fixed_soft_full_retraining | 59 | `process_noise_shift_test` | `RFIS` | 376.223628 | +8.854352 | `RFIS` | 3036.136545 | +21.364779 |
| observed_mask_fixed_soft_full_retraining | 61 | `maneuver_shift_test` | `EKF` | 634.443911 | -14.466488 | `EKF` | 5060.728728 | +16.119998 |
| observed_mask_fixed_soft_full_retraining | 61 | `process_noise_shift_test` | `RFIS` | 353.598854 | +8.067487 | `VA_RFIS` | 8709.274466 | -83.858670 |
