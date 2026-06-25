# SPOT-OD v1.3.2 public reproduction package repair

This release turns the reviewed v1.3.1 package repair into a coherent
GitHub/Zenodo public reproduction package.

Release tag: `v1.3.2-public-reproduction-package`.

## What changed

- Keeps the v1.3.1 validation-selected edge-only retained-candidate
  residual-refinement scientific readback unchanged.
- Packages the runtime import members needed for extracted-script execution:
  `src/`, `scripts/_bootstrap.py`, `scripts/__init__.py`, `pyproject.toml`,
  `requirements.txt`, and the repository `README.md`.
- Adds v1.3.2 metadata, release README, manifest, citation metadata, and a
  GitHub/Zenodo metadata record that does not invent a v1.3.2 DOI before Zenodo
  import.
- Adds a separate checkpoint-free training-input ZIP built from the retained
  candidate input bundle used by the provenance command.
- Adds a v1.3.2 verifier covering the main ZIP and the training-input ZIP.

## Scientific readback

No scientific metrics changed from v1.3.1:

- All non-development rows: 390.317 m versus 459.591 m for the best retained
  candidate, a 15.073% gain.
- Fresh source-generation seeds 151/157/163/167: 386.373 m versus 445.943 m
  for the best retained candidate, a 13.358% gain.
- Versus matched edge-only local/no-message control: 24.835%
  all-non-development and 45.329% fresh advantages.
- Versus edge-only mean graph: 3.983% all-non-development and 5.548% fresh
  advantages.

The future-only attention-vs-mean check remains weak/mixed. Graph-path isolation
is only against the matched edge-only no-message local control.

## Release assets

- Main package: `release/spot_od_v1_3_2_public_reproduction_package.zip`
- Training inputs:
  `release/spot_od_v1_3_2_public_reproduction_training_inputs.zip`

Asset byte sizes and SHA-256 digests are reported after archive construction and
are not embedded inside these packaged release notes.

## Provenance command

The selected attention run uses the same retained-candidate provenance command
documented for v1.3.1. To rerun it from public package materials, extract both
the main package and the training-input package at the same repository root, then
run:

```powershell
.\.venv\Scripts\python.exe scripts\run_trajectory_candidate_graph_selector_poc.py --source-glob "results/adaptive_candidate_fusion_observed_fixed_soft_seed*_split*_20260623" --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed167_split167_20260625 --output-dir results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --scenarios process_noise_shift_test,maneuver_shift_test --candidate-methods EKF,UKF,AUKF,BatchWLS,RFIS,VA_RFIS --baseline-candidate-methods EKF,UKF,AUKF,BatchWLS,RFIS,VA_RFIS --development-seed-max-exclusive 67 --development-validation-seed-min 53 --holdout-seed-min 67 --future-seed-min 109 --epochs 500 --hidden-dim 64 --graph-layers 2 --graph-layer-type attention --prediction-mode residual_refine --node-disagreement-features omit --residual-loss-weight 1e-5 --learning-rate 0.001 --weight-decay 0.002 --dropout 0.2 --seed 2111 --ensemble-size 3 --ensemble-seeds 2111,2117,2129 --device cuda --batch-size 64
```

The command is a training provenance route. The release verifier does not run
training; it only performs integrity, metadata, metric-readback, and extracted
`--help` smoke checks.

## Verification

```powershell
python scripts\verify_v132_public_reproduction_package.py --archive release\spot_od_v1_3_2_public_reproduction_package.zip --training-archive release\spot_od_v1_3_2_public_reproduction_training_inputs.zip
python -m pytest tests\test_v132_public_reproduction_package_verification.py tests\test_v131_release_package_verification.py -q
```

The v1.3.2 verifier checks:

- unsafe ZIP paths, duplicate members, `__pycache__`, and `.pyc` files;
- v1.3.2 metadata coherence and absence of a claimed pre-import v1.3.2 DOI;
- `src/gnn_state_estimation/`, `scripts/_bootstrap.py`, and
  `scripts/__init__.py`;
- extracted `python scripts/run_trajectory_candidate_graph_selector_poc.py
  --help`;
- unchanged validation-selected metric readback from packaged result JSON files;
- training-input manifests, source directories, scenario prediction arrays, and
  checkpoint omission.

## DOI status

Zenodo is expected to mint the v1.3.2 version DOI after importing the GitHub
release. This package does not claim a v1.3.2 DOI before that import.

- Previous Zenodo version DOI: `10.5281/zenodo.20844596`
- Zenodo concept DOI: `10.5281/zenodo.20768672`
- Historical cited package DOI retained in metadata: `10.5281/zenodo.20840386`

## Scope limits

This release is retained-candidate compact-simulator evidence only. It is not
independent-machine reproduction, not public precise-reference validation, not a
full raw/training/all-filter rerun, not standalone learned recursive filtering,
not broad learned orbit-determination validation, not operational precise orbit
determination, and not operational learned orbit determination.
