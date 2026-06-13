from pathlib import Path
import json, re, glob
rp_path=Path('results/release_packet.json')
rp={}
if rp_path.exists():
    try: rp=json.loads(rp_path.read_text(encoding='utf-8'))
    except Exception as e: rp={'_parse_error':str(e)}
main_path=Path('paper/main.tex')
main=main_path.read_text(encoding='utf-8') if main_path.exists() else ''
active_tables=[]; active_figures=[]
for i,line in enumerate(main.splitlines(),1):
    s=line.strip()
    if s.startswith('%'): continue
    mt=re.search(r'\\input\{([^}]+)\}', line)
    if mt and ('tables/' in mt.group(1)):
        p=mt.group(1)
        if not p.startswith('paper/'):
            p='paper/'+p
        if not p.endswith('.tex'):
            p+='.tex'
        active_tables.append((i,p.replace('/','\\')))
    mf=re.search(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', line)
    if mf:
        p=mf.group(1)
        if not p.startswith('paper/'):
            p='paper/'+p
        active_figures.append((i,p.replace('/','\\')))

def exists(p): return Path(p).exists()
commands=[]
candidates=[
 ('Environment report','python scripts/report_environment.py','scripts/report_environment.py'),
 ('Regenerate release packet','python scripts/generate_release_packet.py','scripts/generate_release_packet.py'),
 ('Validate release packet','python scripts/validate_release_packet.py','scripts/validate_release_packet.py'),
 ('Generate paper tables','python scripts/generate_paper_tables.py','scripts/generate_paper_tables.py'),
 ('Generate figures','python scripts/generate_figures.py','scripts/generate_figures.py'),
 ('Run tests','python -m pytest','tests'),
]
seen=set()
for label,cmd,req in candidates:
    if exists(req) and cmd not in seen:
        commands.append((label,cmd)); seen.add(cmd)
if exists('Makefile'):
    txt=Path('Makefile').read_text(encoding='utf-8', errors='ignore')
    for target in ['release','paper','tables','figures','test']:
        if re.search(rf'^{re.escape(target)}\s*:', txt, re.M):
            commands.append((f'Make target {target}', f'make {target}'))
artifacts=[]
for p in ['results/release_packet.json','results/environment_report.json','results/command_log.json','results/command_logs.json','results/reproducibility_command_log.md','paper/RELEASE_PACKET.md']:
    if exists(p): artifacts.append(p)
for pat in ['results/*environment*','results/*command*log*','results/*release*']:
    for q in glob.glob(pat):
        if Path(q).is_file() and q not in artifacts:
            artifacts.append(q)
historical=[]
for pat in ['scripts/*histor*','scripts/*diagnostic*','scripts/*debug*','scripts/*legacy*','scripts/*old*','scripts/*archive*']:
    for q in glob.glob(pat):
        if Path(q).is_file(): historical.append(q.replace('/','\\'))
def find(o,k):
    if isinstance(o,dict):
        if k in o: return o[k]
        for v in o.values():
            r=find(v,k)
            if r is not None: return r
    elif isinstance(o,list):
        for v in o:
            r=find(v,k)
            if r is not None: return r
    return None
counts={}
for key in ['main_generated_table_count','main_inline_table_count','main_figure_count','main_figure_environment_count','main_figure_include_count']:
    v=find(rp,key)
    if v is not None: counts[key]=v
if not counts:
    counts={'active_tables_from_main_tex':len(active_tables),'active_figures_from_main_tex':len(active_figures)}
cmd_lines='\n'.join([f'- **{label}:** `{cmd}`' for label,cmd in commands]) or '- No runnable command script was discovered in the current workspace; use the artifact files below as the auditable evidence source until a command runner is added.'
art_lines='\n'.join([f'- `{p.replace("/", "\\")}`' for p in artifacts]) or '- No release/evidence artifacts were discovered.'
tab_lines='\n'.join([f'- line {i}: `{p}`' for i,p in active_tables]) or '- None discovered from active, uncommented `paper/main.tex` lines.'
fig_lines='\n'.join([f'- line {i}: `{p}`' for i,p in active_figures]) or '- None discovered from active, uncommented `paper/main.tex` lines.'
hist_lines='\n'.join([f'- `{p}`' for p in historical]) or '- No stale/legacy/diagnostic script file was discovered by name. Treat any unlisted historical command in older notes as non-authoritative unless it is re-run and logged in `results/`.'
count_lines='\n'.join([f'- `{k}`: {v}' for k,v in counts.items()])
md=f'''## Reproducibility instruction evidence plan

This project treats the current workspace, not older manager notes, as the source of truth for reproducibility evidence. Active manuscript evidence is limited to uncommented `\\input{{...}}` and `\\includegraphics{{...}}` statements in `paper/main.tex`; diagnostic, historical, and supplement-planned files are tracked as release artifacts but must not be claimed as main-manuscript evidence unless they are actively included.

### Active commands to run or audit
{cmd_lines}

### Environment and command-log evidence
{art_lines}

### Active main-manuscript tables
{tab_lines}

### Active main-manuscript figures
{fig_lines}

### Release artifact index rule
The release packet should distinguish (i) active main-manuscript artifacts, (ii) supplemental/planned artifacts, and (iii) historical or diagnostic artifacts. Counts currently read from `results\\release_packet.json` or `paper/main.tex` are:
{count_lines}

### Stale historical tools
{hist_lines}

### Citation-backed reproducibility basis
The evidence plan follows standard computational reproducibility practice: preserve executable commands, environment metadata, data/artifact indexes, and enough provenance for independent reruns and audit (Peng, 2011; Sandve et al., 2013; ACM Artifact Review and Badging, 2020).
'''
for path in [Path('README.md'), Path('paper/RELEASE_PACKET.md')]:
    txt=path.read_text(encoding='utf-8') if path.exists() else ''
    marker_start='<!-- REPRODUCIBILITY-EVIDENCE-PLAN:START -->'
    marker_end='<!-- REPRODUCIBILITY-EVIDENCE-PLAN:END -->'
    block=f'\n{marker_start}\n{md}{marker_end}\n'
    if marker_start in txt and marker_end in txt:
        txt=re.sub(re.escape(marker_start)+r'.*?'+re.escape(marker_end), block.strip(), txt, flags=re.S)
    else:
        txt=txt.rstrip()+block
    path.write_text(txt,encoding='utf-8')
def tex_escape(s):
    return s.replace('\\','/').replace('_','\\_')
latex_cmds='\n'.join([f'\\item \\textbf{{{label}:}} \\texttt{{{tex_escape(cmd)}}}' for label,cmd in commands]) or '\\item No runnable command script was discovered in the current workspace; use the archived evidence artifacts until a command runner is added.'
latex_arts='\n'.join([f'\\item \\texttt{{{tex_escape(p)}}}' for p in artifacts]) or '\\item No release/evidence artifacts were discovered.'
latex_tabs='\n'.join([f'\\item line {i}: \\texttt{{{tex_escape(p)}}}' for i,p in active_tables]) or '\\item None discovered.'
latex_figs='\n'.join([f'\\item line {i}: \\texttt{{{tex_escape(p)}}}' for i,p in active_figures]) or '\\item None discovered.'
latex_counts='; '.join([f'\\texttt{{{tex_escape(k)}}}={v}' for k,v in counts.items()])
latex=f'''
% REPRODUCIBILITY-EVIDENCE-PLAN:START
\\subsection{{Reproducibility instruction evidence plan}}
The reproducibility instructions use the current repository state as the authority for executable evidence. Active paper evidence is restricted to uncommented \\texttt{{\\\\input}} and \\texttt{{\\\\includegraphics}} statements in \\texttt{{paper/main.tex}}; diagnostic, historical, and supplement-planned artifacts are retained in the release packet but are not claimed as active main-manuscript evidence unless they are included here. This follows reproducible-research guidance to preserve commands, environments, provenance, and auditable artifacts~\\cite{{peng2011reproducible,sandve2013ten,acm2020artifact}}.

\\paragraph{{Active commands.}}
\\begin{{itemize}}
{latex_cmds}
\\end{{itemize}}

\\paragraph{{Environment and command-log evidence.}}
\\begin{{itemize}}
{latex_arts}
\\end{{itemize}}

\\paragraph{{Current active manuscript artifacts.}}
The active generated-table entries are:
\\begin{{itemize}}
{latex_tabs}
\\end{{itemize}}
The active figure entries are:
\\begin{{itemize}}
{latex_figs}
\\end{{itemize}}
The current release-packet count summary is {latex_counts}. Historical or diagnostic tools must be labeled as stale unless they are re-run, logged, and promoted into the active command list above.
% REPRODUCIBILITY-EVIDENCE-PLAN:END
'''
if main_path.exists():
    txt=main_path.read_text(encoding='utf-8')
    if '% REPRODUCIBILITY-EVIDENCE-PLAN:START' in txt and '% REPRODUCIBILITY-EVIDENCE-PLAN:END' in txt:
        txt=re.sub(r'% REPRODUCIBILITY-EVIDENCE-PLAN:START.*?% REPRODUCIBILITY-EVIDENCE-PLAN:END', latex.strip(), txt, flags=re.S)
    else:
        idx=txt.find('\\input{evidence_plan}')
        if idx!=-1:
            txt=txt[:idx]+latex+'\n'+txt[idx:]
        else:
            txt=txt.replace('\\end{document}', latex+'\n\\end{document}')
    main_path.write_text(txt,encoding='utf-8')
bib_candidates=[Path('paper/references.bib'),Path('paper/bibliography.bib'),Path('paper/main.bib'),Path('references.bib')]
bib=None
for b in bib_candidates:
    if b.exists(): bib=b; break
entries={
'peng2011reproducible':'''@article{peng2011reproducible,
  title={Reproducible research in computational science},
  author={Peng, Roger D.},
  journal={Science},
  volume={334},
  number={6060},
  pages={1226--1227},
  year={2011},
  doi={10.1126/science.1213847}
}
''',
'sandve2013ten':'''@article{sandve2013ten,
  title={Ten simple rules for reproducible computational research},
  author={Sandve, Geir Kjetil and Nekrutenko, Anton and Taylor, James and Hovig, Eivind},
  journal={PLoS Computational Biology},
  volume={9},
  number={10},
  pages={e1003285},
  year={2013},
  doi={10.1371/journal.pcbi.1003285}
}
''',
'acm2020artifact':'''@misc{acm2020artifact,
  title={{Artifact Review and Badging Version 1.1}},
  author={{Association for Computing Machinery}},
  year={2020},
  howpublished={\\url{https://www.acm.org/publications/policies/artifact-review-and-badging-current}}
}
'''}
if bib:
    bt=bib.read_text(encoding='utf-8')
    add=[v for k,v in entries.items() if k not in bt]
    if add:
        bib.write_text(bt.rstrip()+'\n\n'+'\n'.join(add),encoding='utf-8')
Path('results').mkdir(exist_ok=True)
Path('results/reproducibility_instruction_evidence_plan.md').write_text(md+f'\nPatched files: README.md, paper/RELEASE_PACKET.md, paper/main.tex' + (f', {bib}' if bib else ''),encoding='utf-8')
print('patched reproducibility plan')
print('commands:', commands)
print('artifacts:', artifacts)
print('active_tables:', active_tables)
print('active_figures:', active_figures)
print('bib:', bib)
