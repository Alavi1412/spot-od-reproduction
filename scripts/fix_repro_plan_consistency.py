from pathlib import Path

README_COMMANDS = [
    r'.\.venv\Scripts\python.exe scripts\environment_check.py',
    r'.\.venv\Scripts\python.exe -m pytest',
    r'.\.venv\Scripts\python.exe scripts\compile_paper.py',
    r'.\.venv\Scripts\python.exe scripts\verify_release_packet_sync.py',
    r'.\.venv\Scripts\python.exe tools_task227_final_submission_validation.py',
    r'.\.venv\Scripts\python.exe -m json.tool results\release_packet.json',
]
REGEN_COMMANDS = [
    r'.\.venv\Scripts\python.exe scripts\generate_dataset.py --config configs\experiment.yaml',
    r'.\.venv\Scripts\python.exe scripts\train_models.py --config configs\experiment.yaml --device cuda:0',
    r'.\.venv\Scripts\python.exe scripts\evaluate_models.py --config configs\experiment.yaml --device cuda:0 --scenarios test,stress_test',
    r'.\.venv\Scripts\python.exe scripts\render_publication_figures.py',
    r'.\.venv\Scripts\python.exe scripts\build_paper_assets.py',
    r'.\.venv\Scripts\python.exe scripts\compile_paper.py',
]
ACTIVE_TABLES = [
    (111, r'paper\tables\main_results.tex'),
    (154, r'paper\tables\significance.tex'),
    (161, r'paper\tables\ablation.tex'),
]
ACTIVE_FIGURES = [
    (122, r'paper\figures\per_step_rmse.png'),
    (129, r'paper\figures\position_error_ecdf.png'),
    (139, r'paper\figures\visibility_bucket_rmse.png'),
    (148, r'paper\figures\uncertainty_calibration.png'),
]
STALE_TOOLS = [
    'debug_validator.py', 'ensure_repro_language.py', 'find_lines.py',
    'fix_provenance_repro_update.py', 'patch_all.py', 'patch_paper.py',
    'patch_release_packet.py', 'patch_repo.py', 'patch_task218.py',
    'record_provenance_update.py', 'update_paper.py',
    'verify_provenance_repro_update.py',
    r'results\release_packet_sync_validate.py', r'docs\verify_release_packet_sync.py'
]
EVIDENCE = [
    r'results\runtime\env_report.json',
    r'results\manifests\evaluation.json',
    r'results\release_packet.json',
    r'paper\RELEASE_PACKET.md',
    r'paper\main.log',
    r'results\pytest_full_attempt.txt',
    r'results\release_validation.log',
    r'results\verify_release_packet_sync.log',
    r'results\task227_final_release_evidence_index.json',
]

def md_block():
    lines=[]
    lines.append('## Reproducibility instruction evidence plan')
    lines.append('')
    lines.append('This project treats the current workspace, not older manager notes, as the source of truth for reproducibility evidence. Active manuscript evidence is limited to uncommented `\\input{...}` and `\\includegraphics{...}` statements in `paper/main.tex`; diagnostic, historical, and supplement-planned files are tracked as release artifacts but must not be claimed as main-manuscript evidence unless they are actively included.')
    lines.append('')
    lines.append('### Exact active verification commands')
    lines.append('Run these from the repository root with the checked-out project virtual environment; do not substitute plain `python` unless the venv is activated:')
    lines.append('')
    lines.append('```powershell')
    lines.extend(README_COMMANDS)
    lines.append('```')
    lines.append('')
    lines.append('Expected current verification status from `README.md`: environment check exits 0 and writes `results\\runtime\\env_report.json`; pytest reports 43 passing tests; `scripts\\compile_paper.py` writes `paper\\main.pdf`; release-packet validators confirm 3 active generated tables, 0 inline tables, and 4 active figures.')
    lines.append('')
    lines.append('### Full experiment-regeneration recipe')
    lines.append('The following commands are the documented full-regeneration route when hardware and time permit. The current manuscript does not claim this full sequence was freshly rerun for this documentation patch.')
    lines.append('')
    lines.append('```powershell')
    lines.extend(REGEN_COMMANDS)
    lines.append('```')
    lines.append('')
    lines.append('### Environment report and command-log evidence')
    for item in EVIDENCE:
        lines.append(f'- `{item}`')
    lines.append('')
    lines.append('### Active main-manuscript tables')
    for line,path in ACTIVE_TABLES:
        lines.append(f'- line {line}: `{path}`')
    lines.append('')
    lines.append('### Active main-manuscript figures')
    for line,path in ACTIVE_FIGURES:
        lines.append(f'- line {line}: `{path}`')
    lines.append('')
    lines.append('### Release artifact index rule')
    lines.append('The release packet distinguishes (i) active main-manuscript artifacts, (ii) supplemental/planned artifacts, and (iii) historical or diagnostic artifacts. Current counts are:')
    lines.append('- `main_generated_table_count`: 3')
    lines.append('- `main_inline_table_count`: 0')
    lines.append('- `main_figure_count`: 4')
    lines.append('- `main_figure_environment_count`: 4')
    lines.append('- `main_figure_include_count`: 4')
    lines.append('')
    lines.append('### Stale historical tools')
    lines.append('The following tools are retained only as historical maintenance, patching, or diagnostic helpers and are not part of the active reproducibility pipeline unless regenerated, explicitly logged, and promoted into the active command list:')
    for item in STALE_TOOLS:
        lines.append(f'- `{item}`')
    lines.append('')
    lines.append('### Citation-backed reproducibility basis')
    lines.append('The evidence plan follows standard computational reproducibility practice: preserve executable commands, environment metadata, data/artifact indexes, and enough provenance for independent reruns and audit (Peng, 2011; Sandve et al., 2013; ACM Artifact Review and Badging, 2020).')
    return '\n'.join(lines) + '\n'

def tex_escape(s):
    return s.replace('\\', '/').replace('_', '\\_')

def tex_block():
    cmd_items='\n'.join(f'\\item \\texttt{{{tex_escape(c)}}}' for c in README_COMMANDS)
    regen_items='\n'.join(f'\\item \\texttt{{{tex_escape(c)}}}' for c in REGEN_COMMANDS)
    ev_items='\n'.join(f'\\item \\texttt{{{tex_escape(e)}}}' for e in EVIDENCE)
    table_items='\n'.join(f'\\item line {line}: \\texttt{{{tex_escape(path)}}}' for line,path in ACTIVE_TABLES)
    fig_items='\n'.join(f'\\item line {line}: \\texttt{{{tex_escape(path)}}}' for line,path in ACTIVE_FIGURES)
    stale_items='\n'.join(f'\\item \\texttt{{{tex_escape(s)}}}' for s in STALE_TOOLS)
    return f'''% REPRODUCIBILITY-EVIDENCE-PLAN:START
\\subsection{{Reproducibility instruction evidence plan}}
The reproducibility instructions use the current repository state as the authority for executable evidence. Active paper evidence is restricted to uncommented \\texttt{{\\\\input}} and \\texttt{{\\\\includegraphics}} statements in \\texttt{{paper/main.tex}}; diagnostic, historical, and supplement-planned artifacts are retained in the release packet but are not claimed as active main-manuscript evidence unless they are included here. This follows reproducible-research guidance to preserve commands, environments, provenance, and auditable artifacts~\\cite{{peng2011reproducible,sandve2013ten,acm2020artifact}}.

\\paragraph{{Exact active verification commands.}}
Run these from the repository root with the checked-out project virtual environment; do not substitute plain \\texttt{{python}} unless the venv is activated.
\\begin{{itemize}}
{cmd_items}
\\end{{itemize}}
The expected current verification status recorded in \\texttt{{README.md}} is: environment check exit 0 with \\texttt{{results/runtime/env\\_report.json}}, 43 passing pytest tests, \\texttt{{paper/main.pdf}} generated by \\texttt{{scripts/compile\\_paper.py}}, and release-packet validators confirming 3 active generated tables, 0 inline tables, and 4 active figures.

\\paragraph{{Full experiment-regeneration recipe.}}
When hardware and time permit, the documented full-regeneration route is:
\\begin{{itemize}}
{regen_items}
\\end{{itemize}}
The current revision does not claim this full sequence was freshly rerun for this documentation patch.

\\paragraph{{Environment report and command-log evidence.}}
\\begin{{itemize}}
{ev_items}
\\end{{itemize}}

\\paragraph{{Current active manuscript artifacts.}}
The active generated-table entries are:
\\begin{{itemize}}
{table_items}
\\end{{itemize}}
The active figure entries are:
\\begin{{itemize}}
{fig_items}
\\end{{itemize}}
The current release-packet count summary is \\texttt{{main\\_generated\\_table\\_count}}=3; \\texttt{{main\\_inline\\_table\\_count}}=0; \\texttt{{main\\_figure\\_count}}=4; \\texttt{{main\\_figure\\_environment\\_count}}=4; \\texttt{{main\\_figure\\_include\\_count}}=4.

\\paragraph{{Stale historical tools.}}
The following tools are retained only as historical maintenance, patching, or diagnostic helpers and are not part of the active reproducibility pipeline unless regenerated, explicitly logged, and promoted into the active command list:
\\begin{{itemize}}
{stale_items}
\\end{{itemize}}
% REPRODUCIBILITY-EVIDENCE-PLAN:END'''

def replace_between(text, start, end, replacement):
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        return before + start + '\n' + replacement + end + after
    return text.rstrip() + '\n' + start + '\n' + replacement + end + '\n'

md = md_block()
for p in [Path('README.md'), Path('paper/RELEASE_PACKET.md')]:
    txt=p.read_text(encoding='utf-8')
    p.write_text(replace_between(txt, '<!-- REPRODUCIBILITY-EVIDENCE-PLAN:START -->', '<!-- REPRODUCIBILITY-EVIDENCE-PLAN:END -->', md), encoding='utf-8')
main=Path('paper/main.tex')
main_txt=main.read_text(encoding='utf-8')
main.write_text(replace_between(main_txt, '% REPRODUCIBILITY-EVIDENCE-PLAN:START', '% REPRODUCIBILITY-EVIDENCE-PLAN:END', tex_block()), encoding='utf-8')
Path('results/reproducibility_instruction_evidence_plan.md').write_text(md + '\nPatched files: README.md, paper/RELEASE_PACKET.md, paper/main.tex, paper/references.bib\n', encoding='utf-8')
print('fixed reproducibility plan consistency')
