# Edge-only local tail diagnostic

Boundary: Saved-row compact-simulator diagnostic only; not independent reproduction, not public precise-reference validation, not operational POD, not a full raw/training/all-filter rerun, and not standalone learned recursive filtering.

This diagnostic aligns saved `rows.csv` records by `(source_name, scenario, trajectory_row)`.

## Sources
- attention: `results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625\rows.csv`
- local: `results\trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625\rows.csv`
- mean: `results\trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625\rows.csv`

## Aggregate diagnostics

| Tier | Rows | Obs. steps | Attention RMSE m | Local RMSE m | Mean RMSE m | Best-single RMSE m | Local vs attention W/T/L | Local vs mean W/T/L | Local-attn p50/p95/max delta m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_eval_non_development | 230 | 6831 | 390.317 | 519.282 | 406.509 | 459.591 | 80/0/150 | 104/0/126 | 11.652/199.974/3229.907 |
| fresh_extra | 47 | 1426 | 386.373 | 706.730 | 409.067 | 445.943 | 16/0/31 | 18/0/29 | 19.382/2320.927/3229.907 |

## Tail rows: all_eval_non_development

Top rows by local selected observed-step RMSE:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_seed127_split127_20260623 / process_noise_shift_test / row 3: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_seed97_split97_20260623 / maneuver_shift_test / row 6: local 1272.451 m (BatchWLS, p=0.673), attention 1216.301 m (EKF, p=0.623), mean 1240.696 m (BatchWLS, p=0.542), local-attention delta 56.150 m

Top rows by local-minus-attention row RMSE delta:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_seed127_split127_20260623 / process_noise_shift_test / row 3: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_seed97_split97_20260623 / process_noise_shift_test / row 0: local 443.012 m (RFIS, p=0.505), attention 139.212 m (BatchWLS, p=0.926), mean 158.768 m (BatchWLS, p=0.765), local-attention delta 303.800 m

## Tail rows: fresh_extra

Top rows by local selected observed-step RMSE:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 3: local 721.242 m (EKF, p=0.365), attention 521.268 m (RFIS, p=0.703), mean 530.965 m (RFIS, p=0.707), local-attention delta 199.974 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 0: local 648.886 m (EKF, p=0.731), attention 734.219 m (EKF, p=0.844), mean 734.668 m (EKF, p=0.965), local-attention delta -85.334 m

Top rows by local-minus-attention row RMSE delta:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 3485.331 m (BatchWLS, p=0.432), attention 255.425 m (RFIS, p=0.970), mean 542.264 m (RFIS, p=0.778), local-attention delta 3229.907 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 3: local 721.242 m (EKF, p=0.365), attention 521.268 m (RFIS, p=0.703), mean 530.965 m (RFIS, p=0.707), local-attention delta 199.974 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 5: local 542.902 m (EKF, p=0.629), attention 400.432 m (RFIS, p=0.775), mean 402.093 m (RFIS, p=0.667), local-attention delta 142.470 m

The weak local aggregate is driven by saved-row tail failures; the attention-vs-mean comparison remains weak/mixed.
