# Targeted Retraining Replay

Status: PASS

Model: ObservabilityContextHybridGNN
Seed: 118103
Training exit code: 0

Claim boundary: This artifact demonstrates one bounded from-scratch learned-estimator training replay on deterministic materialized splits under the predeclared replay rule. It does not reproduce the full paper tables, seed suites, or main-result claims.

## Compute Requirement

The learned-estimator replay required accelerated execution; runtime, hardware, and software-version details are redacted.

## Stage Histories

- nominal_pretrain_replay: epochs=3, final_train_loss=-1.5996568997701008, final_val_loss=-1.5986842683383398
- mixed_train_replay: epochs=3, final_train_loss=-1.599725450944463, final_val_loss=-1.523287608165934
- stress_focus_replay: epochs=2, final_train_loss=-1.5997755789175265, final_val_loss=-1.1646566140187251

## Checkpoints

- results/retraining_replay/targeted_retraining_replay/artifacts/checkpoints/replay_observabilitycontexthybridgnn_nominal_pretrain_replay.pt: 83c0065b85ac6e534b2b4ad0a994ff4e18a8349656de93c751232686efd0e69e
- results/retraining_replay/targeted_retraining_replay/artifacts/checkpoints/replay_observabilitycontexthybridgnn_mixed_train_replay.pt: 00988ac6b3913809ffc84a866b43246e0e2e1799da66a77580d15a5c8685dc41
- results/retraining_replay/targeted_retraining_replay/artifacts/checkpoints/replay_observabilitycontexthybridgnn.pt: 25fe697961503fbadfc0750b42551868dc48d644f5cc24b86defaff31e3e5c53

## Criteria

- training_step_returned_zero: True
- model_history_present: True
- train_loss_finite: True
- validation_loss_finite: True
- checkpoint_produced: True
- checkpoint_sha256_recorded: True
- deterministic_config_captured: True
- isolated_output_dir_under_results_retraining_replay: True
- canonical_checkpoint_digest_unchanged: True
- accelerated_compute_required_and_used: True
- predeclaration_digest_recorded: True

