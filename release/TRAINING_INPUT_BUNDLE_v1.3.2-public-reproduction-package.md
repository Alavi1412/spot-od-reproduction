# SPOT-OD v1.3.2 public reproduction training-input bundle

This bundle supplies the upstream retained-candidate input directories needed by
the v1.3.2 public reproduction package provenance command for the v1.3.1
validation-selected edge-only attention residual-refinement evidence.

Extract this ZIP at the repository root so the
`results/adaptive_candidate_fusion_observed_fixed_soft_*` paths exist beside
`scripts/run_trajectory_candidate_graph_selector_poc.py`.

Scope boundary: checkpoint-free retained-candidate input arrays and metadata
only. This is not raw-data generation, not full raw/training/all-filter
reproduction, not public precise-reference validation, not independent
third-party reproduction, and not operational validation.

Checkpoints are omitted. The GNN training loader consumes
`adaptive_candidate_fusion_predictions.npz` under each source/scenario
directory.

Payload file count: `290`
Total ZIP file count including these manifest files: `292`
Source directory count: `29`
Payload bytes before ZIP compression: `23741532`

## Source directories

- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625`
- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625`
- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625`
- `results/adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed167_split167_20260625`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed101_split101_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed103_split103_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed107_split107_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed109_split109_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed113_split113_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed127_split127_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed131_split131_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed137_split137_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed23_split23_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed29_split29_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed31_split31_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed37_split37_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed41_split41_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed43_split43_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed47_split47_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed53_split53_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed59_split59_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed61_split61_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed67_split67_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed71_split71_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed73_split73_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed79_split79_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed83_split83_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed89_split89_20260623`
- `results/adaptive_candidate_fusion_observed_fixed_soft_seed97_split97_20260623`
