# Edge-only local tail diagnostic

Boundary: Saved-row compact-simulator diagnostic only; not independent reproduction, not public precise-reference validation, not operational POD, not a full raw/training/all-filter rerun, and not standalone learned recursive filtering.

This diagnostic aligns saved `rows.csv` records by `(source_name, scenario, trajectory_row)`.

## Sources
- attention: `results\trajectory_candidate_graph_attention_nodeomit_residual_refine_ensemble3_2111_2117_2129_newfresh151157163167_20260625\rows.csv`
- local: `results\trajectory_candidate_local_nodeomit_residual_refine_ensemble3_2111_2117_2129_newfresh151157163167_20260625\rows.csv`
- mean: `results\trajectory_candidate_mean_nodeomit_residual_refine_ensemble3_2111_2117_2129_newfresh151157163167_20260625\rows.csv`

## Aggregate diagnostics

| Tier | Rows | Obs. steps | Attention RMSE m | Local RMSE m | Mean RMSE m | Best-single RMSE m | Local vs attention W/T/L | Local vs mean W/T/L | Local-attn p50/p95/max delta m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_eval_non_development | 230 | 6831 | 373.728 | 562.030 | 386.224 | 459.591 | 71/0/159 | 90/0/140 | 33.848/254.663/4053.164 |
| fresh_extra | 47 | 1426 | 364.229 | 814.283 | 380.065 | 445.943 | 14/0/33 | 17/0/30 | 34.525/2888.463/4053.164 |

## Tail rows: all_eval_non_development

Top rows by local selected observed-step RMSE:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_seed127_split127_20260623 / process_noise_shift_test / row 3: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_seed97_split97_20260623 / maneuver_shift_test / row 6: local 1262.220 m (BatchWLS, p=0.626), attention 1096.880 m (EKF, p=0.999), mean 1153.548 m (EKF, p=0.999), local-attention delta 165.339 m

Top rows by local-minus-attention row RMSE delta:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_seed127_split127_20260623 / process_noise_shift_test / row 3: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_seed97_split97_20260623 / process_noise_shift_test / row 0: local 435.212 m (RFIS, p=0.418), attention 144.754 m (BatchWLS, p=1.000), mean 132.641 m (BatchWLS, p=0.998), local-attention delta 290.458 m

## Tail rows: fresh_extra

Top rows by local selected observed-step RMSE:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 3: local 651.019 m (RFIS, p=0.496), attention 480.194 m (RFIS, p=0.999), mean 500.351 m (RFIS, p=0.967), local-attention delta 170.825 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 0: local 640.237 m (EKF, p=0.708), attention 634.478 m (EKF, p=1.000), mean 615.326 m (EKF, p=0.975), local-attention delta 5.759 m

Top rows by local-minus-attention row RMSE delta:
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / process_noise_shift_test / row 2: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 / process_noise_shift_test / row 0: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 / process_noise_shift_test / row 1: local 4283.203 m (BatchWLS, p=0.534), attention 230.039 m (RFIS, p=0.823), mean 227.577 m (RFIS, p=0.999), local-attention delta 4053.164 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 3: local 651.019 m (RFIS, p=0.496), attention 480.194 m (RFIS, p=0.999), mean 500.351 m (RFIS, p=0.967), local-attention delta 170.825 m
- adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 / maneuver_shift_test / row 2: local 555.874 m (EKF, p=0.402), attention 394.942 m (EKF, p=1.000), mean 410.296 m (EKF, p=0.996), local-attention delta 160.932 m

The weak local aggregate is driven by saved-row tail failures; the attention-vs-mean comparison remains weak/mixed.
