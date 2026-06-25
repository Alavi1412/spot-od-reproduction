# Edge-only residual refine comparison intervals

Candidate: `results\trajectory_candidate_graph_attention_nodeomit_residual_refine_ensemble3_2111_2117_2129_newfresh151157163167_20260625`

Node-level candidate-disagreement aggregates are omitted; pairwise edge features remain available to message-passing graph layers.

## all_eval_non_development
- best_single_retained: candidate 373.728 m vs reference 459.591 m, gain 18.682% (row CI 14.945 to 22.400; cluster CI 15.148 to 22.401), W/T/L 196/0/34
- edge_only_local_residual_refine: candidate 373.728 m vs reference 562.030 m, gain 33.504% (row CI 16.779 to 45.680; cluster CI 17.611 to 45.291), W/T/L 159/0/71
- edge_only_mean_residual_refine: candidate 373.728 m vs reference 386.224 m, gain 3.235% (row CI -0.207 to 6.685; cluster CI 0.683 to 5.715), W/T/L 124/0/106
- original_attention_residual_refine: candidate 373.728 m vs reference 380.074 m, gain 1.670% (row CI -0.753 to 3.900; cluster CI 0.011 to 3.239), W/T/L 136/0/94

## fresh_extra
- best_single_retained: candidate 364.229 m vs reference 445.943 m, gain 18.324% (row CI 9.190 to 26.728; cluster CI 9.797 to 25.427), W/T/L 42/0/5
- edge_only_local_residual_refine: candidate 364.229 m vs reference 814.283 m, gain 55.270% (row CI 9.900 to 69.225; cluster CI 32.960 to 67.183), W/T/L 33/0/14
- edge_only_mean_residual_refine: candidate 364.229 m vs reference 380.065 m, gain 4.167% (row CI -0.956 to 11.429; cluster CI -0.688 to 10.907), W/T/L 24/0/23
- original_attention_residual_refine: candidate 364.229 m vs reference 374.438 m, gain 2.727% (row CI -0.603 to 7.026; cluster CI -0.214 to 6.322), W/T/L 29/0/18

## holdout_seed_ge_67
- best_single_retained: candidate 376.194 m vs reference 463.124 m, gain 18.770% (row CI 14.736 to 22.800; cluster CI 15.116 to 22.857), W/T/L 154/0/29
- edge_only_local_residual_refine: candidate 376.194 m vs reference 473.583 m, gain 20.564% (row CI 8.289 to 34.833; cluster CI 8.565 to 34.533), W/T/L 126/0/57
- edge_only_mean_residual_refine: candidate 376.194 m vs reference 387.832 m, gain 3.001% (row CI -0.826 to 6.898; cluster CI 0.267 to 5.644), W/T/L 100/0/83
- original_attention_residual_refine: candidate 376.194 m vs reference 381.547 m, gain 1.403% (row CI -1.417 to 3.980; cluster CI -0.455 to 3.136), W/T/L 107/0/76

## future_seed_ge_109
- best_single_retained: candidate 389.506 m vs reference 477.693 m, gain 18.461% (row CI 9.746 to 26.217; cluster CI 11.703 to 26.202), W/T/L 49/0/11
- edge_only_local_residual_refine: candidate 389.506 m vs reference 561.761 m, gain 30.663% (row CI 3.634 to 51.663; cluster CI 4.426 to 51.158), W/T/L 42/0/18
- edge_only_mean_residual_refine: candidate 389.506 m vs reference 397.535 m, gain 2.020% (row CI -4.832 to 8.708; cluster CI 0.662 to 3.341), W/T/L 30/0/30
- original_attention_residual_refine: candidate 389.506 m vs reference 391.480 m, gain 0.504% (row CI -4.910 to 4.930; cluster CI -0.276 to 1.403), W/T/L 32/0/28
