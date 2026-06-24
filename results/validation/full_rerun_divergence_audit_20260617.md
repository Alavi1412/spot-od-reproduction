# Full Rerun Divergence Audit

Generated UTC: `2026-06-17T14:10:44Z`
Schema: `full_rerun_divergence_audit_v2`

## Boundary
Diagnostic audit only; not a canonical table replacement, not operational validation, not independent reproduction, and not a rerun success upgrade.

Failure-conditioned summaries are diagnostic only. They are not replacement metrics, do not redefine performance, and do not rescue any method.

No learned-positive claim should be inferred from raw tiny wins or failure-conditioned rows. The full-rerun scorecard already treats candidate divergence as failing the practical/headline decision logic.

## Inputs
| Input | Path | SHA-256 |
|---|---|---|
| `metrics_summary` | `results/full_rerun_20260616/metrics_summary.json` | `ad14c74abb4b3cbcb74b4cafc7181d26ed188ff3aabc4c36008bd4fe07d69e6b` |
| `scorecard_summary` | `results/full_rerun_20260616/scorecard_summary.json` | `dd3169a9e654c2d5e3e0966cc29864b7cdfee4d53829d17d9332e3e23b228fc0` |
| `trajectory_errors` | `results/full_rerun_20260616/trajectory_errors.csv` | `51a682b28dbe4785c5a64ad4a4b1ecf86b4ef020f0976ada3da582fd4db59111` |

## Overall Counts
- Total scenarios in metrics input: `17`
- Scenarios with any metrics divergence: `2`
- Divergence is concentrated in: `dense_visibility_test`, `satnogs_observation_replay_test`
- Canonical manuscript table membership: Conservative: not inferred from these audit inputs. The full rerun is diagnostic/internal evidence and this audit is not a canonical manuscript-table replacement.

## Divergence Cases
| Scenario | Method | Candidate-diverged in scorecard | n | metrics flagged n | mask flagged n | all-traj pos RMSE [m] | median traj pos [m] | max traj pos [m] | max/median pos | Reason |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dense_visibility_test` | `UKF` | `None` | 48 | 8 | 8 | 2.078e+19 | 22654.281 | 1.171e+20 | 5.167e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `AUKF` | `True` | 48 | 4 | 4 | 1.676e+14 | 22654.281 | 8.797e+14 | 3.883e+10 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `NoGraphResidual` | `True` | 48 | 3 | 3 | 2.772e+06 | 22654.743 | 6.342e+06 | 279.953 | extreme_trajectory_rmse,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `LearnedNoiseAdaptive` | `True` | 48 | 8 | 8 | 8.480e+18 | 22654.041 | 4.517e+19 | 1.994e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `HybridGNN` | `True` | 48 | 7 | 7 | 1.079e+19 | 22655.541 | 6.760e+19 | 2.984e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `MatchedNoGraphRGR` | `True` | 48 | 7 | 7 | 8.743e+18 | 22643.772 | 4.589e+19 | 2.027e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `CapacityMatchedNoGraphRGR` | `True` | 48 | 6 | 6 | 6.743e+11 | 22523.281 | 3.453e+12 | 1.533e+08 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `InnovationHybridGNN` | `True` | 48 | 7 | 7 | 1.262e+18 | 22653.914 | 6.467e+18 | 2.855e+14 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `dense_visibility_test` | `ObservabilityContextHybridGNN` | `True` | 48 | 8 | 8 | 1.162e+19 | 22641.720 | 6.667e+19 | 2.945e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `satnogs_observation_replay_test` | `UKF` | `None` | 48 | 1 | 1 | 1.228e+18 | 5479.668 | 8.507e+18 | 1.552e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `satnogs_observation_replay_test` | `LearnedNoiseAdaptive` | `True` | 48 | 1 | 1 | 8.763e+17 | 3391.286 | 6.071e+18 | 1.790e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `satnogs_observation_replay_test` | `HybridGNN` | `True` | 48 | 1 | 1 | 543870.784 | 3229.089 | 3.758e+06 | 1163.792 | extreme_trajectory_rmse,velocity_rmse_outlier_ratio |
| `satnogs_observation_replay_test` | `MatchedNoGraphRGR` | `True` | 48 | 1 | 1 | 4.837e+11 | 3004.787 | 3.351e+12 | 1.115e+09 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `satnogs_observation_replay_test` | `InnovationHybridGNN` | `True` | 48 | 1 | 1 | 8.601e+17 | 2883.957 | 5.959e+18 | 2.066e+15 | extreme_trajectory_rmse,position_rmse_outlier_ratio,velocity_rmse_outlier_ratio |
| `satnogs_observation_replay_test` | `ObservabilityContextHybridGNN` | `True` | 48 | 1 | 1 | 1.021e+06 | 3418.674 | 7.065e+06 | 2066.565 | extreme_trajectory_rmse,velocity_rmse_outlier_ratio |

## Failure-Conditioned Diagnostic Summaries
Percentiles, maxima, and top values remain all-trajectory diagnostic context. Only the diagnostic mean in this section excludes paired rows selected by the evaluator-style extreme mask.

| Scenario | Method | mask flagged n | retained n | median [m] | p90 [m] | p95 [m] | max [m] | mean excl. mask flagged rows [m] | Count diagnostic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `dense_visibility_test` | `UKF` | 8 | 40 | 22654.281 | 4.618e+12 | 3.620e+19 | 1.171e+20 | 1.973e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `AUKF` | 4 | 44 | 22654.281 | 2.856e+07 | 5.771e+09 | 8.797e+14 | 3.180e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `NoGraphResidual` | 3 | 45 | 22654.743 | 5.324e+06 | 5.975e+06 | 6.342e+06 | 1.538e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `LearnedNoiseAdaptive` | 8 | 40 | 22654.041 | 1.062e+12 | 7.168e+18 | 4.517e+19 | 1.977e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `HybridGNN` | 7 | 41 | 22655.541 | 3.103e+08 | 1.406e+18 | 6.760e+19 | 2.418e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `MatchedNoGraphRGR` | 7 | 41 | 22643.772 | 1.060e+12 | 1.201e+19 | 4.589e+19 | 1.678e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `CapacityMatchedNoGraphRGR` | 6 | 42 | 22523.281 | 2.292e+07 | 4.296e+09 | 3.453e+12 | 1.758e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `InnovationHybridGNN` | 7 | 41 | 22653.914 | 2.613e+11 | 5.081e+17 | 6.467e+18 | 3.842e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `dense_visibility_test` | `ObservabilityContextHybridGNN` | 8 | 40 | 22641.720 | 8.254e+11 | 4.085e+12 | 6.667e+19 | 1.658e+06 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `satnogs_observation_replay_test` | `UKF` | 1 | 47 | 5479.668 | 118522.989 | 400570.203 | 8.507e+18 | 40646.028 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `satnogs_observation_replay_test` | `LearnedNoiseAdaptive` | 1 | 47 | 3391.286 | 40584.883 | 96326.769 | 6.071e+18 | 16157.330 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `satnogs_observation_replay_test` | `HybridGNN` | 1 | 47 | 3229.089 | 32684.752 | 61817.052 | 3.758e+06 | 13125.076 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `satnogs_observation_replay_test` | `MatchedNoGraphRGR` | 1 | 47 | 3004.787 | 18893.639 | 31353.638 | 3.351e+12 | 8489.130 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `satnogs_observation_replay_test` | `InnovationHybridGNN` | 1 | 47 | 2883.957 | 29722.758 | 72635.663 | 5.959e+18 | 19464.001 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |
| `satnogs_observation_replay_test` | `ObservabilityContextHybridGNN` | 1 | 47 | 3418.674 | 33405.869 | 99279.402 | 7.065e+06 | 14860.923 | Evaluator-style paired trajectory mask count matches metrics num_diverged_trajectories. This remains diagnostic only and does not redefine the decision rule. |

Failure-conditioned rows are for inspection of tail concentration only. They use the paired evaluator-style extreme mask rather than a top-N trim. They are not replacement manuscript metrics, do not redefine performance, and do not alter the canonical practical-floor decisions.
