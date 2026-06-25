# High-fidelity pre-update R-only NIS campaign

Claim boundary: Internal sampled high-fidelity synthetic campaign only; not operational POD; not external validation; no learned model training or checkpoint inference.

This artifact uses high-fidelity synthetic truth cells and evaluates only EKF, instrumented UKF, and instrumented AUKF. It does not train or run learned models, and it does not evaluate PUKF, DSA-EKF, or cross-filter posterior NIS.

The diagnostic is `pre_update_r_only_nis`: it is recorded inside the UKF/AUKF recursion before each visible station update from the actual measurement residual available at that update.

## Design

- Config: `configs\experiment.yaml`
- Realizations per cell: 1
- Trajectories per realization: 2
- Selected cells: hifi_base, hifi_extended
- Total rows: 2 (finite=2, nondivergent=2)
- Row checkpoint: `results\hifi_pre_update_nis_campaign\hifi_pre_update_nis_campaign_smoke_20260616.rows.jsonl` (append, flush, and fsync after each completed realization; resume_supported=false)
- Optional long-arc cell: `hifi_long_arc` (not selected by default because it is slower).

## Cells

| Cell | Steps | dt s | Eval start | Truth model |
| --- | ---: | ---: | ---: | --- |
| hifi_base | 120 | 20.0 | 11 | J2..J4 plus third-body truth |
| hifi_extended | 120 | 20.0 | 11 | J2..J6 plus third-body and diurnal density truth |

## Provenance

Invoked command:

```powershell
"\\nas\Projects\Papers\GNN State Estimation\.venv\Scripts\python.exe" scripts\run_hifi_pre_update_nis_campaign.py --config configs\experiment.yaml --cells default-smoke --realizations-per-cell 1 --trajectories-per-realization 2 --base-seed 2026061601 --output-prefix results\hifi_pre_update_nis_campaign\hifi_pre_update_nis_campaign_smoke_20260616
```

- Script SHA-256: `47e0f4120d72abe7e0554be30ed857e8bc24c60dac627075b844972bb98fcdab`
- Config SHA-256: `e772a670d03bfb83ce9f6f37dd5b397096210f98f895d328099f962c4700f8d5`
- Git HEAD: `206a93f31ee9360ed094ee0064b0e0060c6e6e37`
- Git status lines: 1491 (truncated_in_json=True, status_sha256=`91064eae707f5b73b83baf366da7a6d42c03fa30fb190c934a44892af5103cf6`)
- Python: `3.11.9`
- NumPy: `2.4.6`

## Cell Summary

| Cell | Rows | Finite | Nondivergent | Median rho | Median UKF RMSE m | Median AUKF RMSE m | AUKF worse rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hifi_base | 1 | 1 | 1 | 0.905 | 399.74 | 423.13 | 1 |
| hifi_extended | 1 | 1 | 1 | 0.979 | 279.68 | 285.75 | 1 |

## Threshold Operating Characteristics

Outcome: `aukf_worse_than_ukf`. Predictor: `rho_pre_update_r_only_nis_aukf_vs_ukf >= threshold`.
Point estimates are followed by Wilson 95% confidence intervals where defined.

| rho threshold | n | TP | FP | TN | FN | precision | sensitivity | specificity | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | 2 | 0 | 0 | 0 | 2 |  | 0.000 [0.000, 0.658] |  | 0.000 [0.000, 0.658] |
| 1.25 | 2 | 0 | 0 | 0 | 2 |  | 0.000 [0.000, 0.658] |  | 0.000 [0.000, 0.658] |
| 1.5 | 2 | 0 | 0 | 0 | 2 |  | 0.000 [0.000, 0.658] |  | 0.000 [0.000, 0.658] |

## Sensitivity: Finite Nondivergent Rows

| rho threshold | n | TP | FP | TN | FN | precision | sensitivity | specificity | accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1 | 2 | 0 | 0 | 0 | 2 |  | 0.000 [0.000, 0.658] |  | 0.000 [0.000, 0.658] |
| 1.25 | 2 | 0 | 0 | 0 | 2 |  | 0.000 [0.000, 0.658] |  | 0.000 [0.000, 0.658] |
| 1.5 | 2 | 0 | 0 | 0 | 2 |  | 0.000 [0.000, 0.658] |  | 0.000 [0.000, 0.658] |

## Row Summary

| Cell | Realization | Seed | rho pre-update R-only NIS | EKF RMSE m | UKF RMSE m | AUKF RMSE m | AUKF worse than UKF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hifi_base | 0 | 2026061601 | 0.905 | 464.32 | 399.74 | 423.13 | True |
| hifi_extended | 0 | 2026161601 | 0.979 | 211.67 | 279.68 | 285.75 | True |

## Larger Campaign Command

```powershell
python scripts/run_hifi_pre_update_nis_campaign.py --config configs/experiment.yaml --cells default-full --realizations-per-cell 8 --trajectories-per-realization 12 --base-seed 20260616 --output-prefix results/hifi_pre_update_nis_campaign/k8_n12
```
