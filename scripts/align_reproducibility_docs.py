from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIVE_COMMANDS = [
    r".\.venv\Scripts\python.exe scripts\environment_check.py",
    r".\.venv\Scripts\python.exe -m pytest",
    r".\.venv\Scripts\python.exe scripts\compile_paper.py",
    r".\.venv\Scripts\python.exe scripts\verify_release_packet_sync.py",
    r".\.venv\Scripts\python.exe tools_task227_final_submission_validation.py",
    r".\.venv\Scripts\python.exe -m json.tool results\release_packet.json",
]

FULL_REGEN_COMMANDS = [
    r".\.venv\Scripts\python.exe scripts\generate_dataset.py --config configs\experiment.yaml",
    r".\.venv\Scripts\python.exe scripts\train_models.py --config configs\experiment.yaml --device cuda:0",
    r".\.venv\Scripts\python.exe scripts\evaluate_models.py --config configs\experiment.yaml --device cuda:0 --scenarios test,stress_test",
    r".\.venv\Scripts\python.exe scripts\render_publication_figures.py",
    r".\.venv\Scripts\python.exe scripts\build_paper_assets.py",
    r".\.venv\Scripts\python.exe scripts\compile_paper.py",
]

STALE_TOOLS = [
    "debug_validator.py",
    "ensure_repro_language.py",
    "find_lines.py",
    "fix_provenance_repro_update.py",
    "patch_all.py",
    "patch_paper.py",
    "patch_release_packet.py",
    "patch_repo.py",
    "patch_repro_plan.py",
    "patch_task218.py",
    "record_provenance_update.py",
    "update_paper.py",
    "verify_provenance_repro_update.py",
    r"results\apply_release_packet_sync_fix.py",
    r"results\check_release_packet_sync_outputs.py",
    r"results\ensure_release_packet_sync_artifacts.py",
    r"results\extract_main_line_evidence.py",
    r"results\patch_manuscript_release_consistency_current.py",
    r"results\patch_release_packet_independent_review.py",
    r"results\release_packet_sync_compat_fix.py",
    r"results\release_packet_sync_patch.py",
    r"results\release_packet_sync_validate.py",
    r"results\release_packet_sync_verifier.py",
    r"results\release_packet_truth_sync.py",
    r"results\repair_v15_release_packet_metadata.py",
    r"results\sync_release_packet_truth.py",
    r"docs\verify_release_packet_sync.py",
]

START_MD = "<!-- REPRODUCIBILITY-EVIDENCE-PLAN:START -->"
END_MD = "<!-- REPRODUCIBILITY-EVIDENCE-PLAN:END -->"


def command_block(commands: list[str]) -> str:
    return "\n".join(commands)


def stale_list_md() -> str:
    return "\n".join(f"- `{item}`" for item in STALE_TOOLS)


def stale_list_tex() -> str:
    return "\n".join(f"\\item \\artifactpath{{{item.replace(chr(92), '/')}}}" for item in STALE_TOOLS)


def common_md_block(readme: bool) -> str:
    scope = (
        "`paper\\RELEASE_PACKET.md` is the exhaustive release-artifact index; this README intentionally gives the concise operator-facing command recipe and points to the release packet for complete artifact inventory."
        if readme
        else "This RELEASE_PACKET is the exhaustive release-artifact index. The README intentionally contains the shorter operator-facing recipe, but it uses the same active-command list and the same stale-tooling policy."
    )
    return f"""{START_MD}
## Reproducibility instruction evidence plan

This project treats the current workspace, not older manager notes, as the source of truth for reproducibility evidence. Active manuscript evidence is limited to uncommented `\\input{{...}}` and `\\includegraphics{{...}}` statements in `paper\\main.tex`; diagnostic, historical, and supplement-planned files are tracked as release artifacts but must not be claimed as main-manuscript evidence unless they are actively included.

### Exact active verification commands
Run these from the repository root (`D:\\GNN State Estimation`) with the checked-out project virtual environment; do not substitute plain `python` unless the venv is activated:

```powershell
{command_block(ACTIVE_COMMANDS)}
```

Expected current verification status: environment check exits 0 and writes `results\\runtime\\env_report.json`; pytest is the repository test-suite gate; `scripts\\compile_paper.py` writes `paper\\main.pdf`; release-packet validators confirm that the active manuscript evidence remains 3 generated tables, 0 inline tables, and 4 active figures.

### Full experiment-regeneration recipe
The following commands are the documented full-regeneration route when hardware and time permit. The current manuscript does not claim this full sequence was freshly rerun for this documentation patch.

```powershell
{command_block(FULL_REGEN_COMMANDS)}
```

### Environment report and command-log evidence
- `results\\runtime\\env_report.json`
- `results\\manifests\\evaluation.json`
- `results\\release_packet.json`
- `paper\\RELEASE_PACKET.md`
- `paper\\main.log`
- `results\\validation\\pytest.log`
- `results\\release_validation.log`
- `results\\verify_release_packet_sync.log`
- `results\\task227_final_release_evidence_index.json`

### Active main-manuscript tables
- line 120: `paper\\tables\\main_results.tex`
- line 163: `paper\\tables\\significance.tex`
- line 170: `paper\\tables\\ablation.tex`

### Active main-manuscript figures
- line 131: `paper\\figures\\per_step_rmse.png`
- line 138: `paper\\figures\\position_error_ecdf.png`
- line 148: `paper\\figures\\visibility_bucket_rmse.png`
- line 157: `paper\\figures\\uncertainty_calibration.png`

### Release artifact index rule
The release packet distinguishes active main-manuscript artifacts, supplemental/planned artifacts, and historical or diagnostic artifacts. Current counts are:
- `main_generated_table_count`: 3
- `main_inline_table_count`: 0
- `main_figure_count`: 4
- `main_figure_environment_count`: 4
- `main_figure_include_count`: 4

### Stale historical tools
The following tools are retained only as historical maintenance, patching, or diagnostic helpers and are not part of the active reproducibility pipeline unless regenerated, explicitly logged, and promoted into the active command list:
{stale_list_md()}

{scope}

### Citation-backed reproducibility basis
The evidence plan follows standard computational reproducibility practice: preserve executable commands, environment metadata, data/artifact indexes, and enough provenance for independent reruns and audit (Peng, 2011; Sandve et al., 2013; ACM Artifact Review and Badging, 2020).
{END_MD}
"""


def replace_between(text: str, start: str, end: str, block: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(text):
        text = pattern.sub(lambda _match: block.strip(), text)
    else:
        text = text.rstrip() + "\n\n" + block.strip()
    return text.rstrip() + "\n"


def write_markdown(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(replace_between(text, START_MD, END_MD, block), encoding="utf-8")


def write_evidence_plan() -> None:
    tex = rf"""\section{{Reproducibility and Evidence Plan}}
\label{{sec:reproducibility-evidence-plan}}

The reproducibility instructions use the current repository state as the authority for executable evidence. Active paper evidence is restricted to uncommented \texttt{{\\input}} and \texttt{{\\includegraphics}} statements in \texttt{{paper/main.tex}}; diagnostic, historical, and supplement-planned artifacts are retained in the release packet but are not claimed as active main-manuscript evidence unless they are included here. This follows reproducible-research guidance to preserve commands, environments, provenance, and auditable artifacts~\cite{{peng2011reproducible,sandve2013ten,acm2020artifact}}.

\subsection{{Reproducibility Instructions}}
\paragraph{{Exact active verification commands.}}
Run these from the repository root, \texttt{{D:\\GNN State Estimation}}, with the checked-out project virtual environment; do not substitute plain \texttt{{python}} unless the venv is activated.
\begin{{verbatim}}
{command_block(ACTIVE_COMMANDS)}
\end{{verbatim}}
The synchronized manuscript evidence remains 3 active generated tables, 0 inline tables, and 4 active figures.

\paragraph{{Full experiment-regeneration recipe.}}
When hardware and time permit, the documented full-regeneration route is:
\begin{{verbatim}}
{command_block(FULL_REGEN_COMMANDS)}
\end{{verbatim}}
The current documentation patch does not claim that this full sequence was freshly rerun.

\subsection{{Evidence Plan and Claim Boundaries}}
Only active, uncommented inputs and figure includes in \texttt{{paper/main.tex}} are treated as main-manuscript evidence. The current main manuscript includes three generated tables: \artifactpath{{paper/tables/main_results.tex}}, \artifactpath{{paper/tables/significance.tex}}, and \artifactpath{{paper/tables/ablation.tex}}. It includes four figures: \artifactpath{{paper/figures/per_step_rmse.png}}, \artifactpath{{paper/figures/position_error_ecdf.png}}, \artifactpath{{paper/figures/visibility_bucket_rmse.png}}, and \artifactpath{{paper/figures/uncertainty_calibration.png}}.

The release packet \texttt{{results/release\_packet.json}} is an index of generated tables, figures, validation artifacts, and manuscript-inclusion status. Safe manuscript language should state that the proposed method achieves the reported values on the evaluated benchmark and metrics, not that it is universally superior. Significance claims must be limited to listed tests. Robustness and calibration claims should be phrased as diagnostics unless supported by numerical calibration metrics, per-bucket sample counts, confidence intervals, repeated seeds, or external-scenario evaluations.

\subsection{{Environment Report and Command-Log Evidence}}
\begin{{itemize}}
\item \artifactpath{{results/runtime/env_report.json}}
\item \artifactpath{{results/manifests/evaluation.json}}
\item \artifactpath{{results/release_packet.json}}
\item \artifactpath{{paper/RELEASE_PACKET.md}}
\item \artifactpath{{paper/main.log}}
\item \artifactpath{{results/validation/pytest.log}}
\item \artifactpath{{results/release_validation.log}}
\item \artifactpath{{results/verify_release_packet_sync.log}}
\item \artifactpath{{results/task227_final_release_evidence_index.json}}
\end{{itemize}}

\subsection{{Stale Historical Tools}}
The following tools are retained only as historical maintenance, patching, or diagnostic helpers and are not part of the active reproducibility pipeline unless regenerated, explicitly logged, and promoted into the active command list:
\begin{{itemize}}
{stale_list_tex()}
\end{{itemize}}
"""
    # This block inventories paths, scripts, virtual-env command lines, and
    # release-audit mechanics. It is internal reproducibility tooling and must
    # NEVER be written into a paper-facing file. paper/evidence_plan.tex is the
    # hand-authored manuscript reproducibility prose; the internal inventory is
    # emitted to a results artifact instead.
    out = ROOT / "results" / "reproducibility_evidence_plan.tex"
    out.write_text(tex, encoding="utf-8")
    paper_eplan = ROOT / "paper" / "evidence_plan.tex"
    if paper_eplan.exists():
        body = paper_eplan.read_text(encoding="utf-8")
        forbidden = [
            "REPRODUCIBILITY-EVIDENCE-PLAN",
            "\\artifactpath{",
            ".venv\\Scripts",
            "Stale Historical Tools",
        ]
        leaked = [m for m in forbidden if m in body]
        if leaked:
            raise SystemExit(
                f"Refusing to continue: paper/evidence_plan.tex contains "
                f"internal reproducibility-tooling content {leaked}. The "
                f"manuscript reproducibility statement must be hand-authored "
                f"prose only; internal inventory lives in {out}.")


def clean_main_tex() -> None:
    # Non-destructive: ensure paper/main.tex includes the hand-authored
    # evidence_plan exactly once and carries no internal-tooling markers. Never
    # delete or relocate the existing include/section.
    path = ROOT / "paper" / "main.tex"
    text = path.read_text(encoding="utf-8")
    if "REPRODUCIBILITY-EVIDENCE-PLAN" in text:
        text = re.sub(
            r"\n?% REPRODUCIBILITY-EVIDENCE-PLAN:START.*?% REPRODUCIBILITY-EVIDENCE-PLAN:END\s*",
            "\n", text, flags=re.DOTALL)
    if text.count(r"\input{evidence_plan}") == 0:
        text = text.replace("\n\\section{Conclusion}",
                            "\n\\input{evidence_plan}\n\n\\section{Conclusion}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    write_markdown(ROOT / "README.md", common_md_block(readme=True))
    write_markdown(ROOT / "paper" / "RELEASE_PACKET.md", common_md_block(readme=False))
    write_evidence_plan()
    clean_main_tex()

    main_text = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert main_text.count(r"\input{evidence_plan}") == 1
    assert "REPRODUCIBILITY-EVIDENCE-PLAN" not in main_text
    # Active venv commands belong in repo/release docs and the internal results
    # artifact -- NOT in the paper-facing manuscript reproducibility prose.
    for rel in ["README.md", "paper/RELEASE_PACKET.md", "results/reproducibility_evidence_plan.tex"]:
        t = (ROOT / rel).read_text(encoding="utf-8")
        for cmd in ACTIVE_COMMANDS:
            assert cmd in t, f"missing {cmd} in {rel}"
    # The manuscript reproducibility prose must stay free of internal tooling.
    eplan = (ROOT / "paper" / "evidence_plan.tex").read_text(encoding="utf-8")
    for cmd in ACTIVE_COMMANDS:
        assert cmd not in eplan, f"venv command leaked into paper/evidence_plan.tex: {cmd}"
    assert ".venv\\Scripts" not in eplan and "\\artifactpath{" not in eplan
    print("Aligned reproducibility documentation; manuscript evidence plan left hand-authored.")


if __name__ == "__main__":
    main()
