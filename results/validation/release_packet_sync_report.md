# Release Packet Synchronization Report

Generated: `2026-06-01T06:48:54.450129Z`

## Status

PASS — release packet evidence was synchronized from current workspace readback.

## Claim boundary

Only active, uncommented paper/main.tex inputs/includes are direct main-manuscript evidence. Release, diagnostic, historical, and supplement-planned artifacts remain tracked but must not be described as included in the current main manuscript unless paper/main.tex includes them.

## Source files read

- `README.md`
- `paper\main.tex`
- `results\release_packet.json`
- `results\validation\submission_validation.json`
- `results\validation\release_evidence_index.json`

## Active manuscript evidence

- Active generated table inputs: 8
- Active inline table environments: 0
- Active figure includes: 1

### Tables

- line 41: `\input{tables/main_abbreviation_glossary.tex}` -> `paper\tables\main_abbreviation_glossary.tex`
- line 87: `\input{tables/main_framework_portability.tex}` -> `paper\tables\main_framework_portability.tex`
- line 160: `\input{tables/main_k32_replication.tex}` -> `paper\tables\main_k32_replication.tex`
- line 198: `\input{tables/main_aukf_mechanism.tex}` -> `paper\tables\main_aukf_mechanism.tex`
- line 225: `\input{tables/main_structural_recoverability.tex}` -> `paper\tables\main_structural_recoverability.tex`
- line 229: `\input{tables/main_drag_scale_cascade.tex}` -> `paper\tables\main_drag_scale_cascade.tex`
- line 235: `\input{tables/main_long_arc_result.tex}` -> `paper\tables\main_long_arc_result.tex`
- line 247: `\input{tables/main_dbar_withdrawal.tex}` -> `paper\tables\main_dbar_withdrawal.tex`

### Figures

- line 193: `\includegraphics[width=\linewidth]{figures/aukf_r_inflation_mechanism.png}` -> `paper\figures\aukf_r_inflation_mechanism.png`

## PDF evidence

- Path: `paper\main.pdf`
- Pages: 33
- Page source: `pdfinfo`
- SHA-256: `dfad35dddae7ed181072c75bb3c2c07158549980866e4149f71e9c94b567ff8e`
- Bytes: 720790

## Validation readback

- Submission validation: `results\validation\submission_validation.json` status=`pass` overall_passed=`True`
- Release evidence index: `results\validation\release_evidence_index.json` status=`pass` overall_passed=`True`

## Reviewer-facing exclusions

Historical review notes are excluded from this reviewer-facing packet because they are not canonical evidence and may contain local/runtime/process metadata.

## Changed files

- `results\validation\release_packet_sync_report.json`
- `results\validation\release_packet_sync_report.md`
- `results\release_packet.json`
- `tools\sync_release_packet.py`

## Reproducibility note

`source_files_read` is populated only by successful file reads in `tools/sync_release_packet.py`; it is not a manually asserted list.
