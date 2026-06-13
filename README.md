# GNN State Estimation / SPOT-OD

This repository contains the SPOT-OD GNN state-estimation paper artifacts, tests,
release-packet metadata, and reproducibility evidence.

## Canonical Environment

Use the checked-out project virtual environment unless recreating the environment
from scratch:

```powershell
.\.venv\Scripts\python.exe --version
```

Current verified environment:

- Python 3.11.9
- CUDA device selected by `scripts\environment_check.py`: `cuda:0`
- GPU reported by the environment check: NVIDIA GeForce RTX 3080 Laptop GPU

Do not treat plain `python` on PATH as canonical unless the project venv is
activated. A system Python 3.14 interpreter may not have the scientific runtime
dependencies installed.

## Verification Commands

Run these from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\environment_check.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\compile_paper.py
.\.venv\Scripts\python.exe scripts\verify_release_packet_sync.py
.\.venv\Scripts\python.exe tools_task227_final_submission_validation.py
.\.venv\Scripts\python.exe -m json.tool results\release_packet.json
```

Current expected results:

- `scripts\environment_check.py`: exits 0 and writes `results\runtime\env_report.json`.
- `pytest`: 43 tests pass in the project virtual environment; the current validation log is `results\validation\pytest.log`.
- `scripts\compile_paper.py`: exits 0 and writes `paper\main.pdf`.
- `results\task227_final_submission_validation_stakeholder_artifact.md`: current deterministic submission-QA snapshot; reports `PASSED`, `95` cited keys, `103` BibTeX entries, `0` unresolved citation/reference warnings, a 33-page PDF readback, and release-packet sync against the current manuscript/PDF state.
- `results\task227_pdf_evidence.json`: current PDF evidence record for `paper\main.pdf`, including SHA-256 `cccbd74497d486b10fa2e9ee3fbac51833c6cc3089ab6c3b7b67eb753eb60745`, 719991 bytes, and 33 pages.
- Release-packet validators: active generated tables and external figure includes match the current `paper\main.tex` parse; the current main manuscript has 9 generated table inputs and 1 external figure include.

## Evidence Boundary

The current verification confirms that the repository tests, environment report,
paper build, and release-packet metadata are internally consistent within each
snapshot. It does not claim a fresh full experiment rerun beyond the repository's
existing cached and generated artifacts.

`results\task227_final_submission_validation_stakeholder_artifact.md`,
`results\task227_final_submission_validation_result_payload.json`, and
`results\task227_pdf_evidence.json` are the canonical current PDF/readback surfaces.
Older build logs such as `results\validation\pdf_build.log` are historical
diagnostics and must not override the current validation snapshot.
<!-- REPRODUCIBILITY-EVIDENCE-PLAN:START -->
## Reproducibility instruction evidence plan

This project treats the current workspace, not older manager notes, as the source of truth for reproducibility evidence. Active manuscript evidence is limited to uncommented `\input{...}` and `\includegraphics{...}` statements in `paper\main.tex`; diagnostic, historical, and supplement-planned files are tracked as release artifacts but must not be claimed as main-manuscript evidence unless they are actively included.

### Exact active verification commands
Run these from the repository root (`D:\GNN State Estimation`) with the checked-out project virtual environment; do not substitute plain `python` unless the venv is activated:

```powershell
.\.venv\Scripts\python.exe scripts\environment_check.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\compile_paper.py
.\.venv\Scripts\python.exe scripts\verify_release_packet_sync.py
.\.venv\Scripts\python.exe tools_task227_final_submission_validation.py
.\.venv\Scripts\python.exe -m json.tool results\release_packet.json
```

Expected current verification status: environment check exits 0 and writes `results\runtime\env_report.json`; pytest is the repository test-suite gate; `scripts\compile_paper.py` writes `paper\main.pdf`; release-packet validators confirm that the active manuscript evidence parsed from `paper\main.tex` remains 9 generated table inputs, 0 inline tables, and 1 active figure include.

### Full experiment-regeneration recipe
The following commands are the documented full-regeneration route when hardware and time permit. The current manuscript does not claim this full sequence was freshly rerun for this documentation patch.

```powershell
.\.venv\Scripts\python.exe scripts\generate_dataset.py --config configs\experiment.yaml
.\.venv\Scripts\python.exe scripts\train_models.py ^
  --config configs\experiment.yaml --device cuda:0
.\.venv\Scripts\python.exe scripts\evaluate_models.py ^
  --config configs\experiment.yaml --device cuda:0 ^
  --scenarios test,stress_test
.\.venv\Scripts\python.exe scripts\render_publication_figures.py
.\.venv\Scripts\python.exe scripts\build_paper_assets.py
.\.venv\Scripts\python.exe scripts\compile_paper.py
```

### Environment report and command-log evidence
- `results\runtime\env_report.json`
- `results\manifests\evaluation.json`
- `results\release_packet.json`
- `paper\RELEASE_PACKET.md`
- `paper\main.log`
- `results\validation\pytest.log`
- `results\validation\pdf_build.log`
- `results\validation\submission_validation.md`
- `results\main_pdf_sha256.txt`
- `results\release_validation.log`
- `results\validation\release_packet_sync.log`
- `results\task227_final_release_evidence_index.json`

### Active main-manuscript tables
- `paper\tables\main_abbreviation_glossary.tex`
- `paper\tables\main_framework_portability.tex`
- `paper\tables\main_findings_summary.tex`
- `paper\tables\main_k32_replication.tex`
- `paper\tables\main_aukf_mechanism.tex`
- `paper\tables\main_structural_recoverability.tex`
- `paper\tables\main_drag_scale_cascade.tex`
- `paper\tables\main_long_arc_result.tex`
- `paper\tables\main_dbar_withdrawal.tex`

### Active main-manuscript figures
- `paper\figures\aukf_r_inflation_mechanism.png`

### Release artifact index rule
The release packet distinguishes active main-manuscript artifacts, supplemental/planned artifacts, and historical or diagnostic artifacts. Current counts are:
- `main_generated_table_count`: 9
- `main_inline_table_count`: 0
- `main_figure_count`: 1
- `main_figure_environment_count`: 1
- `main_figure_include_count`: 1

### Stale historical tools
The following tools are retained only as historical maintenance, patching, or diagnostic helpers and are not part of the active reproducibility pipeline unless regenerated, explicitly logged, and promoted into the active command list:
- `debug_validator.py`
- `ensure_repro_language.py`
- `find_lines.py`
- `fix_provenance_repro_update.py`
- `patch_all.py`
- `patch_paper.py`
- `patch_release_packet.py`
- `patch_repo.py`
- `patch_repro_plan.py`
- `patch_task218.py`
- `record_provenance_update.py`
- `update_paper.py`
- `verify_provenance_repro_update.py`
- `results\apply_release_packet_sync_fix.py`
- `results\check_release_packet_sync_outputs.py`
- `results\ensure_release_packet_sync_artifacts.py`
- `results\extract_main_line_evidence.py`
- `results\patch_manuscript_release_consistency_current.py`
- `results\patch_release_packet_independent_review.py`
- `results\release_packet_sync_compat_fix.py`
- `results\release_packet_sync_patch.py`
- `results\release_packet_sync_validate.py`
- `results\release_packet_sync_verifier.py`
- `results\release_packet_truth_sync.py`
- `results\repair_v15_release_packet_metadata.py`
- `results\sync_release_packet_truth.py`
- `docs\verify_release_packet_sync.py`

`paper\RELEASE_PACKET.md` is the exhaustive release-artifact index; this README intentionally gives the concise operator-facing command recipe and points to the release packet for complete artifact inventory.

### Citation-backed reproducibility basis
The evidence plan follows standard computational reproducibility practice: preserve executable commands, environment metadata, data/artifact indexes, and enough provenance for independent reruns and audit (Peng, 2011; Sandve et al., 2013; ACM Artifact Review and Badging, 2020).
<!-- REPRODUCIBILITY-EVIDENCE-PLAN:END -->
