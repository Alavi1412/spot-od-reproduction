# SPOT-OD v1.3.1 validation-selected residual-refinement ablation

This release adds the validation-selected version of the edge-only retained-candidate attention graph residual-refinement proof of concept.

## What changed

- Promotes the validation-selected edge-only attention residual-refinement ensemble as the current retained-candidate learned-output proof of concept.
- Uses development validation seed minimum 53, 82 fit/training samples, 37 validation samples, and `validation_loss` checkpoint selection.
- Keeps the v1.3.0 train-loss-selected edge-only run as historical/exploratory evidence only.
- Adds the validation-selected comparison intervals, local/no-message tail diagnostic, and compact figure.

## Main result

Validation-selected edge-only attention residual-refinement:

- All non-development rows: 390.317 m versus 459.591 m for the best retained candidate, a 15.073% gain. Row/bootstrap interval: [11.451, 18.575]%; source-scenario/bootstrap interval: [11.739, 18.307]%.
- Fresh source-generation seeds 151/157/163/167: 386.373 m versus 445.943 m for the best retained candidate, a 13.358% gain. Row/bootstrap interval: [5.395, 20.688]%; source-scenario/bootstrap interval: [6.347, 18.328]%.
- Versus matched edge-only local/no-message control: 24.835% all-non-development advantage and 45.329% fresh advantage.
- Versus edge-only mean graph: 3.983% all-non-development advantage and 5.548% fresh advantage.

The result is a retained-candidate compact-simulator proof of concept. It is not a standalone learned recursive filter, not operational learned orbit determination, and not public precise-reference validation.

## Key artifacts

- Validation-selected attention graph run: `results/trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- Validation-selected local/no-message control: `results/trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- Validation-selected mean graph control: `results/trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625/`
- Local-control tail diagnostic: `results/trajectory_candidate_edge_only_local_tail_diagnostic_val53_20260625/`
- Figure: `paper/figures/trajectory_residual_refine_gain_distribution_val53.png`
- Table: `paper/tables/main_trajectory_graph_selector_ensemble_poc.tex`
- Manuscript: `paper/main.tex`

## Package scope

This zip is an inspection/downstream-replay package for the selected PoC outputs. It includes final result rows, summaries, checkpoints, comparison intervals, the validation-selected figure/table, metadata, release documentation, and focused tests. It does not include the upstream retained-candidate input directories named `results/adaptive_candidate_fusion_observed_fixed_soft_*` that are required to rerun training from scratch.

## Provenance and downstream replay commands

The validation-selected attention run was generated with the following provenance command. This command requires upstream retained-candidate input directories outside this zip:

```powershell
.\.venv\Scripts\python.exe scripts\run_trajectory_candidate_graph_selector_poc.py --source-glob "results/adaptive_candidate_fusion_observed_fixed_soft_seed*_split*_20260623" --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed151_split151_20260625 --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed157_split157_20260625 --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed163_split163_20260625 --extra-source-dir results\adaptive_candidate_fusion_observed_fixed_soft_obsckpt_seed167_split167_20260625 --output-dir results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --scenarios process_noise_shift_test,maneuver_shift_test --candidate-methods EKF,UKF,AUKF,BatchWLS,RFIS,VA_RFIS --baseline-candidate-methods EKF,UKF,AUKF,BatchWLS,RFIS,VA_RFIS --development-seed-max-exclusive 67 --development-validation-seed-min 53 --holdout-seed-min 67 --future-seed-min 109 --epochs 500 --hidden-dim 64 --graph-layers 2 --graph-layer-type attention --prediction-mode residual_refine --node-disagreement-features omit --residual-loss-weight 1e-5 --learning-rate 0.001 --weight-decay 0.002 --dropout 0.2 --seed 2111 --ensemble-size 3 --ensemble-seeds 2111,2117,2129 --device cuda --batch-size 64
```

Intervals were regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\build_trajectory_residual_refine_comparison_intervals.py --attention-dir results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --local-dir results\trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --mean-dir results\trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --skip-original-attention --bootstrap-samples 20000 --bootstrap-seed 20260625 --output-json results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625\comparison_intervals.json --output-md results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625\comparison_intervals.md
```

The figure was regenerated with:

```powershell
.\.venv\Scripts\python.exe scripts\build_trajectory_residual_refine_figure.py --graph-dir results\trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --local-dir results\trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --mean-dir results\trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_newfresh151157163167_20260625 --output paper\figures\trajectory_residual_refine_gain_distribution_val53.png
```

## DOI and archive status

- GitHub release: <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.1-validation-selected-residual-refine>
- Tag commit: `c4882f1b367426c0966e906b9332f64d44d2279f`
- GitHub release asset: `spot_od_v1_3_1_validation_selected_residual_refine.zip`
- GitHub release asset integrity: corrective replacement asset byte size and SHA-256 are reported externally after archive construction, not embedded in these packaged notes.
- Zenodo record: <https://zenodo.org/records/20844596>
- Zenodo version DOI: `10.5281/zenodo.20844596`
- Zenodo concept DOI: `10.5281/zenodo.20768672`
- Zenodo archived source file: `Alavi1412/spot-od-reproduction-v1.3.1-validation-selected-residual-refine.zip`
- Zenodo archived source file bytes: `212,947,668`
- Zenodo archived source file MD5: `863e5077d4d29a827c6fcfd1181dce34`
- Prior v1.3.0 DOI: `10.5281/zenodo.20842573`

## Scope limits

This release is retained-candidate compact-simulator evidence only. It is not independent-machine reproduction, not public precise-reference validation, not a full raw/training/all-filter rerun, not standalone learned recursive filtering, not broad learned orbit-determination validation, not operational precise orbit determination, and not operational learned orbit determination.
