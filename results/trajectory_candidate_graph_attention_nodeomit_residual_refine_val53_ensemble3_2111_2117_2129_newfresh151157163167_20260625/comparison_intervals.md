# Edge-only residual refine comparison intervals

Candidate: `results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625`

Node-level candidate-disagreement aggregates are omitted; pairwise edge features remain available to message-passing graph layers.

## all_eval_non_development
- best_single_retained: candidate 390.317 m vs reference 459.591 m, gain 15.073% (row CI 11.451 to 18.575; cluster CI 11.739 to 18.307), W/T/L 154/0/76
- edge_only_local_residual_refine: candidate 390.317 m vs reference 519.282 m, gain 24.835% (row CI 11.172 to 36.097; cluster CI 12.108 to 35.630), W/T/L 150/0/80
- edge_only_mean_residual_refine: candidate 390.317 m vs reference 406.509 m, gain 3.983% (row CI 1.522 to 6.654; cluster CI 1.563 to 6.385), W/T/L 124/0/106

## fresh_extra
- best_single_retained: candidate 386.373 m vs reference 445.943 m, gain 13.358% (row CI 5.395 to 20.688; cluster CI 6.347 to 18.328), W/T/L 29/0/18
- edge_only_local_residual_refine: candidate 386.373 m vs reference 706.730 m, gain 45.329% (row CI 6.233 to 61.137; cluster CI 23.030 to 59.308), W/T/L 31/0/16
- edge_only_mean_residual_refine: candidate 386.373 m vs reference 409.067 m, gain 5.548% (row CI 0.441 to 11.744; cluster CI 0.774 to 11.825), W/T/L 27/0/20

## holdout_seed_ge_67
- best_single_retained: candidate 391.351 m vs reference 463.124 m, gain 15.498% (row CI 11.449 to 19.397; cluster CI 11.722 to 19.241), W/T/L 125/0/58
- edge_only_local_residual_refine: candidate 391.351 m vs reference 457.190 m, gain 14.401% (row CI 5.159 to 26.123; cluster CI 5.853 to 25.573), W/T/L 119/0/64
- edge_only_mean_residual_refine: candidate 391.351 m vs reference 405.832 m, gain 3.568% (row CI 0.916 to 6.505; cluster CI 0.805 to 6.197), W/T/L 97/0/86

## future_seed_ge_109
- best_single_retained: candidate 405.443 m vs reference 477.693 m, gain 15.125% (row CI 7.772 to 22.094; cluster CI 7.657 to 21.841), W/T/L 43/0/17
- edge_only_local_residual_refine: candidate 405.443 m vs reference 521.835 m, gain 22.304% (row CI 1.089 to 42.423; cluster CI 3.372 to 41.321), W/T/L 39/0/21
- edge_only_mean_residual_refine: candidate 405.443 m vs reference 420.074 m, gain 3.483% (row CI -0.296 to 8.590; cluster CI -0.097 to 7.475), W/T/L 33/0/27
