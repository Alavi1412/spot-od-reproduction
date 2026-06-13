# Targeted Retraining Replay Public Report

Status: PASS

Model: ObservabilityContextHybridGNN
Seed: 118103

Claim boundary: This artifact demonstrates one bounded from-scratch learned-estimator training replay on deterministic materialized splits under the predeclared replay rule. It does not reproduce the full paper tables, seed suites, or main-result claims.

## Source Evidence

- Raw JSON SHA-256: 856561e379ee8bb2537179129f317071ed3aecadfc04f4c9615f9ae9806f8285
- Raw markdown SHA-256: a3b862d81564babf3545b50a460edd069dec23a71357e6cbd10e56db69cd55bc

## Predeclaration

- Path: release/predeclarations/targeted_curriculum_retraining_replay_20260525.json
- SHA-256: d6c919ea2d711b0150cb5a2129609d7d9b4a50113b34e92457467d016b7dbd02

## Execution Attestation

- accelerated_compute_required: true
- accelerated_compute_used: true
- non_accelerated_execution_allowed: false

## Data Slices

| Split | Source trajectories | Slice trajectories | Source SHA-256 | Slice SHA-256 |
|---|---:|---:|---|---|
| satnogs_observation_replay_val | 16 | 16 | 2654a344748036649196a4e4a350efee6e8f17c0d24bbce599166c0964227efb | 2654a344748036649196a4e4a350efee6e8f17c0d24bbce599166c0964227efb |
| stress_train | 96 | 96 | 18ab41b8e7ea9217b617e2c67c11afd1fad9f8c1e809c99e1c3f376d6a2e78ef | 18ab41b8e7ea9217b617e2c67c11afd1fad9f8c1e809c99e1c3f376d6a2e78ef |
| stress_val | 24 | 24 | f67be0317d6d3be41ab2cfe1db15170fff760f6768c026dad402983124ca542b | f67be0317d6d3be41ab2cfe1db15170fff760f6768c026dad402983124ca542b |
| train | 160 | 160 | 915432d5774a46790a7583cc7b39d16f773571fefda6022058273bfb2032b3eb | 915432d5774a46790a7583cc7b39d16f773571fefda6022058273bfb2032b3eb |
| val | 32 | 32 | f031a105caf1cb8c83c3cc015679a510ac9fa10b50a47e0e6215ce3a85034f41 | f031a105caf1cb8c83c3cc015679a510ac9fa10b50a47e0e6215ce3a85034f41 |

## Stage Histories

Note: The stress_focus_replay stage records a positive but finite validation loss at epoch 1 followed by a finite negative validation loss at epoch 2. This is a bounded finite-loss curriculum-transition note; the public replay claim is finite execution, checkpoint production, and provenance, not performance or stability evidence.

- nominal_pretrain_replay: epochs=3, final_train_loss=-1.5996568997701008, final_val_loss=-1.5986842683383398, best_val_loss=-1.5990827083587646
- mixed_train_replay: epochs=3, final_train_loss=-1.599725450944463, final_val_loss=-1.523287608165934, best_val_loss=-1.5567349570052105
- stress_focus_replay: epochs=2, final_train_loss=-1.5997755789175265, final_val_loss=-1.1646566140187251, best_val_loss=-1.1646566140187251

## Checkpoints

- results/retraining_replay/targeted_retraining_replay/artifacts/checkpoints/replay_observabilitycontexthybridgnn_nominal_pretrain_replay.pt: 83c0065b85ac6e534b2b4ad0a994ff4e18a8349656de93c751232686efd0e69e
- results/retraining_replay/targeted_retraining_replay/artifacts/checkpoints/replay_observabilitycontexthybridgnn_mixed_train_replay.pt: 00988ac6b3913809ffc84a866b43246e0e2e1799da66a77580d15a5c8685dc41
- results/retraining_replay/targeted_retraining_replay/artifacts/checkpoints/replay_observabilitycontexthybridgnn.pt: 25fe697961503fbadfc0750b42551868dc48d644f5cc24b86defaff31e3e5c53

## Canonical Checkpoint Digest

- before_sha256: 5cf6fa3d91e7e9cfe6b55b80ca75774aab4a7bbb319e623d7975985587e5429c
- after_sha256: 5cf6fa3d91e7e9cfe6b55b80ca75774aab4a7bbb319e623d7975985587e5429c
- unchanged: true
- file_count: 32

## Criteria

- accelerated_compute_required_and_used: true
- canonical_checkpoint_digest_unchanged: true
- checkpoint_produced: true
- checkpoint_sha256_recorded: true
- deterministic_config_captured: true
- isolated_output_dir_under_results_retraining_replay: true
- model_history_present: true
- predeclaration_digest_recorded: true
- train_loss_finite: true
- training_step_returned_zero: true
- validation_loss_finite: true
