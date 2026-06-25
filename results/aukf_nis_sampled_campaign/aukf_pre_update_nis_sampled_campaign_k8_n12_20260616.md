# AUKF-vs-UKF pre-update R-only NIS sampled campaign

Claim boundary: Internal sampled synthetic campaign only; not operational POD; not external validation; no learned model training or checkpoint inference.

This artifact is an internal sampled synthetic campaign. It does not claim operational POD performance and does not provide external validation.

The diagnostic is `pre_update_r_only_nis`: it is computed inside the UKF/AUKF recursion before each visible station update from the actual measurement residual available at that update. It is not reconstructed from post-update stored filter histories.

## Design

- Config: `configs\experiment.yaml`
- Evaluation start index: 11 (training.window_size - 1)
- Realizations per scenario: 8
- Trajectories per realization: 12
- Selected scenarios: test, stress_test, high_drag_test, process_noise_shift_test, maneuver_shift_test, low_inclination_test, sunsync_like_test, high_inclination_test, force_model_mismatch_test, force_component_omission_test
- Total rows: 80 (finite=80, nondivergent=80)
- Row checkpoint: `results\aukf_nis_sampled_campaign\aukf_pre_update_nis_sampled_campaign_k8_n12_20260616.rows.jsonl` (append, flush, and fsync after each completed realization; resume_supported=false)

## Provenance

Invoked command:

```powershell
"\\nas\Projects\Papers\GNN State Estimation\.venv\Scripts\python.exe" scripts\run_aukf_nis_sampled_campaign.py --config configs\experiment.yaml --scenarios default-full --realizations-per-scenario 8 --trajectories-per-realization 12 --base-seed 91000 --output-prefix results\aukf_nis_sampled_campaign\aukf_pre_update_nis_sampled_campaign_k8_n12_20260616
```

- Script SHA-256: `4ea8138ffd15ba160014e8cb6179f67cd7bf014500964bdfcdad1c41390bad81`
- Config SHA-256: `e772a670d03bfb83ce9f6f37dd5b397096210f98f895d328099f962c4700f8d5`
- Git HEAD: `206a93f31ee9360ed094ee0064b0e0060c6e6e37`
- Git status lines: 1484 (truncated_in_json=True, status_sha256=`abc556edfa2c00b6c55f2cf60b5de92161ba7301f6481173035017b06d7f8ecb`)
- Python: `3.11.9`
- NumPy: `2.4.6`

## Threshold Operating Characteristics

Outcome: `aukf_worse_than_ukf`. Predictor: `rho_pre_update_r_only_nis_aukf_vs_ukf >= threshold`.
Point estimates are followed by Wilson 95% confidence intervals where defined.

| rho threshold | n | TP | FP | TN | FN | precision | sensitivity | specificity | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | 80 | 12 | 4 | 39 | 25 | 0.750 [0.505, 0.898] | 0.324 [0.196, 0.485] | 0.907 [0.784, 0.963] | 0.637 [0.528, 0.734] |
| 1.25 | 80 | 6 | 2 | 41 | 31 | 0.750 [0.409, 0.929] | 0.162 [0.077, 0.311] | 0.953 [0.845, 0.987] | 0.588 [0.478, 0.689] |
| 1.5 | 80 | 0 | 0 | 43 | 37 |  | 0.000 [0.000, 0.094] | 1.000 [0.918, 1.000] | 0.537 [0.429, 0.643] |

## Sensitivity: Finite Nondivergent Rows

| rho threshold | n | TP | FP | TN | FN | precision | sensitivity | specificity | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | 80 | 12 | 4 | 39 | 25 | 0.750 [0.505, 0.898] | 0.324 [0.196, 0.485] | 0.907 [0.784, 0.963] | 0.637 [0.528, 0.734] |
| 1.25 | 80 | 6 | 2 | 41 | 31 | 0.750 [0.409, 0.929] | 0.162 [0.077, 0.311] | 0.953 [0.845, 0.987] | 0.588 [0.478, 0.689] |
| 1.5 | 80 | 0 | 0 | 43 | 37 |  | 0.000 [0.000, 0.094] | 1.000 [0.918, 1.000] | 0.537 [0.429, 0.643] |

## Row Summary

| Scenario | Realization | Seed | rho pre-update R-only NIS | EKF RMSE m | UKF RMSE m | AUKF RMSE m | AUKF worse than UKF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| test | 0 | 91000 | 0.984 | 310.65 | 393.95 | 298.01 | False |
| test | 1 | 91101 | 0.905 | 345.40 | 361.29 | 346.66 | False |
| test | 2 | 91202 | 0.978 | 396.11 | 401.58 | 388.23 | False |
| test | 3 | 91303 | 0.949 | 434.77 | 410.91 | 406.02 | False |
| test | 4 | 91404 | 1.024 | 392.29 | 571.55 | 521.88 | False |
| test | 5 | 91505 | 1.026 | 423.62 | 467.50 | 444.30 | False |
| test | 6 | 91606 | 0.991 | 302.92 | 295.43 | 294.25 | False |
| test | 7 | 91707 | 0.956 | 389.14 | 421.03 | 417.38 | False |
| stress_test | 0 | 191000 | 0.989 | 736.94 | 938.12 | 735.44 | False |
| stress_test | 1 | 191101 | 0.765 | 785.49 | 1226.04 | 780.81 | False |
| stress_test | 2 | 191202 | 0.786 | 872.41 | 1673.01 | 862.91 | False |
| stress_test | 3 | 191303 | 0.979 | 765.79 | 1215.27 | 712.90 | False |
| stress_test | 4 | 191404 | 0.828 | 919.30 | 1824.96 | 881.90 | False |
| stress_test | 5 | 191505 | 0.929 | 828.71 | 1080.80 | 818.55 | False |
| stress_test | 6 | 191606 | 0.789 | 853.63 | 1377.67 | 908.86 | False |
| stress_test | 7 | 191707 | 0.866 | 935.65 | 1124.70 | 930.96 | False |
| high_drag_test | 0 | 291000 | 1.017 | 338.04 | 329.70 | 316.12 | False |
| high_drag_test | 1 | 291101 | 1.011 | 425.32 | 414.06 | 442.00 | True |
| high_drag_test | 2 | 291202 | 0.998 | 386.26 | 379.03 | 365.84 | False |
| high_drag_test | 3 | 291303 | 0.888 | 476.01 | 633.82 | 486.67 | False |
| high_drag_test | 4 | 291404 | 1.005 | 396.29 | 474.99 | 414.38 | False |
| high_drag_test | 5 | 291505 | 0.946 | 419.48 | 465.16 | 409.64 | False |
| high_drag_test | 6 | 291606 | 0.974 | 342.15 | 353.11 | 354.90 | True |
| high_drag_test | 7 | 291707 | 0.962 | 377.26 | 378.63 | 376.73 | False |
| process_noise_shift_test | 0 | 391000 | 1.017 | 482.19 | 477.23 | 514.47 | True |
| process_noise_shift_test | 1 | 391101 | 0.970 | 445.30 | 460.91 | 497.40 | True |
| process_noise_shift_test | 2 | 391202 | 1.350 | 529.29 | 411.82 | 415.89 | True |
| process_noise_shift_test | 3 | 391303 | 1.052 | 413.26 | 459.32 | 455.03 | False |
| process_noise_shift_test | 4 | 391404 | 1.115 | 368.94 | 401.38 | 421.05 | True |
| process_noise_shift_test | 5 | 391505 | 0.939 | 471.47 | 465.65 | 479.34 | True |
| process_noise_shift_test | 6 | 391606 | 0.982 | 378.63 | 414.27 | 445.21 | True |
| process_noise_shift_test | 7 | 391707 | 1.284 | 370.84 | 396.28 | 549.84 | True |
| maneuver_shift_test | 0 | 491000 | 1.256 | 410.28 | 504.52 | 488.71 | False |
| maneuver_shift_test | 1 | 491101 | 1.293 | 466.67 | 538.90 | 516.48 | False |
| maneuver_shift_test | 2 | 491202 | 1.124 | 557.91 | 640.59 | 711.54 | True |
| maneuver_shift_test | 3 | 491303 | 1.178 | 641.44 | 934.79 | 680.26 | False |
| maneuver_shift_test | 4 | 491404 | 1.117 | 603.28 | 532.51 | 840.12 | True |
| maneuver_shift_test | 5 | 491505 | 1.076 | 550.07 | 574.52 | 625.37 | True |
| maneuver_shift_test | 6 | 491606 | 1.191 | 570.48 | 513.22 | 508.46 | False |
| maneuver_shift_test | 7 | 491707 | 0.968 | 436.27 | 424.63 | 504.18 | True |
| low_inclination_test | 0 | 591000 | 0.939 | 336.92 | 359.61 | 322.53 | False |
| low_inclination_test | 1 | 591101 | 0.966 | 393.07 | 405.68 | 374.12 | False |
| low_inclination_test | 2 | 591202 | 0.975 | 372.65 | 376.13 | 385.25 | True |
| low_inclination_test | 3 | 591303 | 0.976 | 372.45 | 359.26 | 372.06 | True |
| low_inclination_test | 4 | 591404 | 0.994 | 416.44 | 439.79 | 411.11 | False |
| low_inclination_test | 5 | 591505 | 0.984 | 384.22 | 378.00 | 385.07 | True |
| low_inclination_test | 6 | 591606 | 1.024 | 369.28 | 367.24 | 366.63 | False |
| low_inclination_test | 7 | 591707 | 0.980 | 524.47 | 510.95 | 558.22 | True |
| sunsync_like_test | 0 | 691000 | 1.066 | 356.78 | 318.92 | 361.17 | True |
| sunsync_like_test | 1 | 691101 | 1.000 | 333.01 | 339.42 | 341.96 | True |
| sunsync_like_test | 2 | 691202 | 0.946 | 383.30 | 491.14 | 378.05 | False |
| sunsync_like_test | 3 | 691303 | 0.936 | 361.05 | 358.46 | 365.82 | True |
| sunsync_like_test | 4 | 691404 | 0.961 | 464.18 | 471.52 | 417.16 | False |
| sunsync_like_test | 5 | 691505 | 0.948 | 450.49 | 479.87 | 452.34 | False |
| sunsync_like_test | 6 | 691606 | 1.006 | 335.41 | 330.08 | 353.44 | True |
| sunsync_like_test | 7 | 691707 | 1.008 | 409.20 | 438.88 | 419.67 | False |
| high_inclination_test | 0 | 791000 | 1.004 | 455.16 | 421.25 | 492.55 | True |
| high_inclination_test | 1 | 791101 | 0.963 | 399.56 | 406.90 | 409.27 | True |
| high_inclination_test | 2 | 791202 | 1.057 | 280.31 | 509.12 | 840.49 | True |
| high_inclination_test | 3 | 791303 | 0.993 | 402.33 | 411.66 | 422.92 | True |
| high_inclination_test | 4 | 791404 | 1.038 | 452.37 | 395.65 | 431.10 | True |
| high_inclination_test | 5 | 791505 | 0.976 | 377.54 | 478.44 | 368.57 | False |
| high_inclination_test | 6 | 791606 | 0.997 | 379.53 | 375.32 | 384.04 | True |
| high_inclination_test | 7 | 791707 | 0.997 | 323.73 | 362.30 | 314.58 | False |
| force_model_mismatch_test | 0 | 891000 | 1.319 | 394.40 | 425.83 | 526.32 | True |
| force_model_mismatch_test | 1 | 891101 | 1.284 | 436.40 | 507.41 | 806.30 | True |
| force_model_mismatch_test | 2 | 891202 | 1.323 | 424.43 | 454.54 | 576.90 | True |
| force_model_mismatch_test | 3 | 891303 | 1.345 | 405.40 | 443.60 | 490.49 | True |
| force_model_mismatch_test | 4 | 891404 | 1.246 | 477.54 | 459.93 | 538.61 | True |
| force_model_mismatch_test | 5 | 891505 | 1.071 | 401.91 | 433.95 | 454.98 | True |
| force_model_mismatch_test | 6 | 891606 | 1.212 | 370.69 | 384.13 | 409.87 | True |
| force_model_mismatch_test | 7 | 891707 | 1.169 | 411.52 | 423.54 | 451.54 | True |
| force_component_omission_test | 0 | 991000 | 0.991 | 368.63 | 381.18 | 399.01 | True |
| force_component_omission_test | 1 | 991101 | 0.948 | 356.75 | 540.22 | 336.29 | False |
| force_component_omission_test | 2 | 991202 | 1.036 | 390.46 | 391.72 | 389.20 | False |
| force_component_omission_test | 3 | 991303 | 1.094 | 396.09 | 517.62 | 394.78 | False |
| force_component_omission_test | 4 | 991404 | 0.994 | 410.32 | 451.91 | 415.01 | False |
| force_component_omission_test | 5 | 991505 | 1.035 | 1263.60 | 386.09 | 405.10 | True |
| force_component_omission_test | 6 | 991606 | 0.983 | 444.77 | 462.73 | 433.00 | False |
| force_component_omission_test | 7 | 991707 | 0.955 | 356.94 | 369.12 | 362.68 | False |

## Larger Campaign Command

```powershell
python scripts/run_aukf_nis_sampled_campaign.py --config configs/experiment.yaml --scenarios default-full --realizations-per-scenario 16 --trajectories-per-realization 24 --base-seed 91000 --output-prefix results/aukf_nis_sampled_campaign/k16_n24
```
