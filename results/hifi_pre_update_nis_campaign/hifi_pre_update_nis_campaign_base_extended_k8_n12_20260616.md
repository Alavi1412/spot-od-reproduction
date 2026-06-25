# High-fidelity pre-update R-only NIS campaign

Claim boundary: Internal sampled high-fidelity synthetic campaign only; not operational POD; not external validation; no learned model training or checkpoint inference.

This artifact uses high-fidelity synthetic truth cells and evaluates only EKF, instrumented UKF, and instrumented AUKF. It does not train or run learned models, and it does not evaluate PUKF, DSA-EKF, or cross-filter posterior NIS.

The diagnostic is `pre_update_r_only_nis`: it is recorded inside the UKF/AUKF recursion before each visible station update from the actual measurement residual available at that update.

## Design

- Config: `configs\experiment.yaml`
- Realizations per cell: 8
- Trajectories per realization: 12
- Selected cells: hifi_base, hifi_extended
- Total rows: 16 (finite=16, nondivergent=16)
- Row checkpoint: `results\hifi_pre_update_nis_campaign\hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616.rows.jsonl` (append, flush, and fsync after each completed realization; resume_supported=false)
- Optional long-arc cell: `hifi_long_arc` (not selected by default because it is slower).

## Cells

| Cell | Steps | dt s | Eval start | Truth model |
| --- | ---: | ---: | ---: | --- |
| hifi_base | 120 | 20.0 | 11 | J2..J4 plus third-body truth |
| hifi_extended | 120 | 20.0 | 11 | J2..J6 plus third-body and diurnal density truth |

## Provenance

Invoked command:

```powershell
"\\nas\Projects\Papers\GNN State Estimation\.venv\Scripts\python.exe" "\\nas\Projects\Papers\GNN State Estimation\scripts\run_hifi_pre_update_nis_campaign.py" --config configs\experiment.yaml --cells hifi_base,hifi_extended --realizations-per-cell 8 --trajectories-per-realization 12 --base-seed 2026061602 --output-prefix results\hifi_pre_update_nis_campaign\hifi_pre_update_nis_campaign_base_extended_k8_n12_20260616
```

- Script SHA-256: `47e0f4120d72abe7e0554be30ed857e8bc24c60dac627075b844972bb98fcdab`
- Config SHA-256: `e772a670d03bfb83ce9f6f37dd5b397096210f98f895d328099f962c4700f8d5`
- Git HEAD: `206a93f31ee9360ed094ee0064b0e0060c6e6e37`
- Git status lines: 1495 (truncated_in_json=True, status_sha256=`7d3e3629221a8f85245f3a754e8b2b5bf376bbdf3c56fc93617f460ebb859739`)
- Python: `3.11.9`
- NumPy: `2.4.6`

## Cell Summary

| Cell | Rows | Finite | Nondivergent | Median rho | Median UKF RMSE m | Median AUKF RMSE m | AUKF worse rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hifi_base | 8 | 8 | 8 | 0.994 | 384.77 | 355.23 | 3 |
| hifi_extended | 8 | 8 | 8 | 0.990 | 391.23 | 381.36 | 4 |

## Threshold Operating Characteristics

Outcome: `aukf_worse_than_ukf`. Predictor: `rho_pre_update_r_only_nis_aukf_vs_ukf >= threshold`.
Point estimates are followed by Wilson 95% confidence intervals where defined.

| rho threshold | n | TP | FP | TN | FN | precision | sensitivity | specificity | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | 16 | 0 | 0 | 9 | 7 |  | 0.000 [0.000, 0.354] | 1.000 [0.701, 1.000] | 0.562 [0.332, 0.769] |
| 1.25 | 16 | 0 | 0 | 9 | 7 |  | 0.000 [0.000, 0.354] | 1.000 [0.701, 1.000] | 0.562 [0.332, 0.769] |
| 1.5 | 16 | 0 | 0 | 9 | 7 |  | 0.000 [0.000, 0.354] | 1.000 [0.701, 1.000] | 0.562 [0.332, 0.769] |

## Sensitivity: Finite Nondivergent Rows

| rho threshold | n | TP | FP | TN | FN | precision | sensitivity | specificity | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | 16 | 0 | 0 | 9 | 7 |  | 0.000 [0.000, 0.354] | 1.000 [0.701, 1.000] | 0.562 [0.332, 0.769] |
| 1.25 | 16 | 0 | 0 | 9 | 7 |  | 0.000 [0.000, 0.354] | 1.000 [0.701, 1.000] | 0.562 [0.332, 0.769] |
| 1.5 | 16 | 0 | 0 | 9 | 7 |  | 0.000 [0.000, 0.354] | 1.000 [0.701, 1.000] | 0.562 [0.332, 0.769] |

## Row Summary

| Cell | Realization | Seed | rho pre-update R-only NIS | EKF RMSE m | UKF RMSE m | AUKF RMSE m | AUKF worse than UKF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hifi_base | 0 | 2026061602 | 0.978 | 402.28 | 461.27 | 384.04 | False |
| hifi_base | 1 | 2026061703 | 1.017 | 316.44 | 310.47 | 330.11 | True |
| hifi_base | 2 | 2026061804 | 1.019 | 229.94 | 672.41 | 933.07 | True |
| hifi_base | 3 | 2026061905 | 0.956 | 265.95 | 291.59 | 288.34 | False |
| hifi_base | 4 | 2026062006 | 0.978 | 304.40 | 301.71 | 293.05 | False |
| hifi_base | 5 | 2026062107 | 0.987 | 381.63 | 453.47 | 380.35 | False |
| hifi_base | 6 | 2026062208 | 1.001 | 277.64 | 372.22 | 279.23 | False |
| hifi_base | 7 | 2026062309 | 1.006 | 403.48 | 397.31 | 408.08 | True |
| hifi_extended | 0 | 2026161602 | 0.970 | 388.33 | 409.71 | 375.55 | False |
| hifi_extended | 1 | 2026161703 | 0.958 | 377.39 | 435.85 | 403.21 | False |
| hifi_extended | 2 | 2026161804 | 0.964 | 383.94 | 383.82 | 380.01 | False |
| hifi_extended | 3 | 2026161905 | 1.019 | 441.81 | 398.63 | 423.91 | True |
| hifi_extended | 4 | 2026162006 | 1.010 | 334.32 | 325.26 | 327.57 | True |
| hifi_extended | 5 | 2026162107 | 0.995 | 407.39 | 440.54 | 444.25 | True |
| hifi_extended | 6 | 2026162208 | 0.988 | 389.58 | 371.79 | 382.70 | True |
| hifi_extended | 7 | 2026162309 | 0.993 | 294.85 | 292.62 | 291.60 | False |

## Larger Campaign Command

```powershell
python scripts/run_hifi_pre_update_nis_campaign.py --config configs/experiment.yaml --cells default-full --realizations-per-cell 8 --trajectories-per-realization 12 --base-seed 20260616 --output-prefix results/hifi_pre_update_nis_campaign/k8_n12
```
