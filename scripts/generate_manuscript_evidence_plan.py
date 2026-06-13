import json, re
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('.')
PAPER = ROOT / 'paper'
RESULTS = ROOT / 'results'
DOCS = ROOT / 'docs'
DOCS.mkdir(exist_ok=True)

main_path = PAPER / 'main.tex'
main_text = main_path.read_text(encoding='utf-8') if main_path.exists() else ''

def uncommented_lines(text):
    out=[]
    for line in text.splitlines():
        stripped=line.lstrip()
        if stripped.startswith('%'):
            continue
        out.append(re.sub(r'(?<!\\)%.*$', '', line))
    return '\n'.join(out)

active = uncommented_lines(main_text)
all_inputs = re.findall(r'\\input\{([^}]+)\}', active)
# evidence_plan is the generated audit, not an experimental result table
inputs = [i for i in all_inputs if not i.replace('\\','/').endswith('evidence_plan') and not i.replace('\\','/').endswith('evidence_plan.tex')]
includes = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', active)

release_path = RESULTS / 'release_packet.json'
release = {}
if release_path.exists():
    release = json.loads(release_path.read_text(encoding='utf-8'))

input_paths=[]
for inp in inputs:
    p = inp
    if not p.endswith('.tex'):
        p = p + '.tex'
    pp = (PAPER / p) if not p.startswith('paper') else Path(p)
    input_paths.append(str(pp).replace('/', '\\'))

active_tables=[]
for sp in input_paths:
    p = Path(sp.replace('\\','/'))
    if p.exists():
        tex=p.read_text(encoding='utf-8')
        caption = re.search(r'\\caption\{([^}]*)\}', tex, re.S)
        label = re.search(r'\\label\{([^}]*)\}', tex)
        rows=[ln.strip() for ln in tex.splitlines() if '&' in ln and not ln.strip().startswith('%')]
        active_tables.append({
            'path': sp,
            'label': label.group(1) if label else None,
            'caption': ' '.join(caption.group(1).split()) if caption else None,
            'data_row_count_estimate': len(rows),
            'status': 'active_uncommented_input_in_paper_main_tex'
        })
    else:
        active_tables.append({'path': sp, 'status': 'active_input_missing_on_disk'})

ms = release.get('manuscript_inclusion_status', {}) if isinstance(release, dict) else {}
claim_boundary = release.get('claim_boundary') or ms.get('claim_boundary') or 'Only active, uncommented paper/main.tex inputs/includes are treated as direct main-manuscript evidence in this pass.'
figure_status_map = ms.get('figures', {}) if isinstance(ms.get('figures', {}), dict) else {}
canonical_figures = release.get('canonical_artifacts', {}).get('figures', []) if isinstance(release.get('canonical_artifacts', {}), dict) else []
all_release_figures = []
seen=set()
for source in [figure_status_map.keys(), canonical_figures]:
    for fig in source:
        if fig not in seen:
            seen.add(fig); all_release_figures.append(fig)

norm_includes = {x.replace('\\','/').replace('paper/','') for x in includes} | {x.replace('\\','/') for x in includes}
fig_entries=[]
for fig in all_release_figures:
    norm = fig.replace('\\','/')
    fig_entries.append({
        'path': fig,
        'release_status': figure_status_map.get(fig, figure_status_map.get(norm, 'canonical_not_in_main')),
        'main_manuscript_status': 'active_include' if norm in norm_includes else 'not_active_in_main_tex'
    })

missing_experiments=[]
for f in fig_entries:
    st=str(f.get('release_status',''))
    if st in ('supplement_planned','historical_auxiliary','release_only_diagnostic','canonical_not_in_main'):
        missing_experiments.append({
            'artifact': f.get('path'),
            'current_status': st,
            'needed_for_claims_about': Path(str(f.get('path',''))).stem.replace('_',' '),
            'required_action': 'Promote into paper/main.tex/supplement with caption and claim text, or keep out of claims; rerun/validate if numbers are stale or historical.'
        })

claim_audit = [
    {
        'claim': 'The current main manuscript directly includes generated quantitative tables for main results, statistical significance, and ablation evidence.',
        'evidence': [t['path'] for t in active_tables],
        'support_level': 'supported_by_active_main_tex_inputs',
        'boundary': 'Supported only to the extent the table files contain the cited values; this pass does not recompute experiments.'
    },
    {
        'claim': 'The current main manuscript has no active figure includes.',
        'evidence': includes if includes else ['paper/main.tex active includegraphics scan returned zero active figure includes'],
        'support_level': 'supported_by_current_main_tex_scan',
        'boundary': 'Release or supplement-planned figures must not be described as main-manuscript figures until uncommented in paper/main.tex.'
    },
    {
        'claim': 'Diagnostic, historical, and supplement-planned figures remain tracked artifacts but are not direct main-manuscript evidence at this revision point.',
        'evidence': fig_entries,
        'support_level': 'supported_by_results_release_packet_json_manuscript_inclusion_status',
        'boundary': 'Use these only in supplement/planned-work language unless paper/main.tex is patched to include them.'
    },
    {
        'claim': 'Claims about robustness profiles, calibration, visibility buckets, or per-step trajectories require promotion into the manuscript/supplement or regenerated validation.',
        'evidence': missing_experiments,
        'support_level': 'conditional_or_missing_from_main_manuscript',
        'boundary': 'Do not overclaim from non-included artifacts.'
    }
]

table_figure_plan = {
    'claim_boundary': claim_boundary,
    'active_main_tables': active_tables,
    'active_main_figures': includes,
    'release_figures': fig_entries,
    'missing_or_gated_experiments': missing_experiments,
    'recommended_manuscript_actions': [
        'Keep the three active generated result tables as the current main-manuscript quantitative evidence unless additional inputs/includes are deliberately added.',
        'Do not refer to release-only diagnostic figures as main-manuscript figures.',
        'Move supplement-planned figures into a supplement section only after captions and claim text are evidence-locked.',
        'Treat historical auxiliary figures as archival until regenerated or explicitly validated against the current release packet.'
    ]
}

payload = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'source_files_read': ['paper/main.tex', 'results/release_packet.json'] + [t['path'] for t in active_tables],
    'table_figure_plan': table_figure_plan,
    'claim_audit': claim_audit,
    'missing_experiments': missing_experiments,
    'backward_compatibility_note': 'Created/updated results/manuscript_evidence_plan.json as a companion structured artifact and patched paper/main.tex by adding a LaTeX input; release_packet.json schema was not rewritten.'
}

(RESULTS / 'manuscript_evidence_plan.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

md=[]
md.append('# Manuscript Evidence Plan — Evidence-Locked Reconstruction Pass')
md.append('')
md.append(f"Generated: {payload['generated_at_utc']}")
md.append('')
md.append('## Claim boundary')
md.append(claim_boundary)
md.append('')
md.append('## Active main-manuscript tables')
for t in active_tables:
    md.append(f"- `{t['path']}` — status: {t.get('status')}; label: {t.get('label')}; caption: {t.get('caption')}; data-row estimate: {t.get('data_row_count_estimate')}")
md.append('')
md.append('## Active main-manuscript figures')
if includes:
    for f in includes: md.append(f'- `{f}`')
else:
    md.append('- None found by active uncommented `\\includegraphics` scan of `paper/main.tex`.')
md.append('')
md.append('## Release/supplement/historical figure disposition')
for f in fig_entries:
    md.append(f"- `{f.get('path')}` — release status: {f.get('release_status')}; main status: {f.get('main_manuscript_status')}")
md.append('')
md.append('## Claim audit')
for i,c in enumerate(claim_audit,1):
    md.append(f"### Claim {i}: {c['claim']}")
    md.append(f"- Support level: {c['support_level']}")
    md.append(f"- Boundary: {c['boundary']}")
    ev=c['evidence']
    if isinstance(ev, list):
        for e in ev[:30]: md.append(f"  - Evidence: `{e}`")
    else:
        md.append(f"  - Evidence: `{ev}`")
md.append('')
md.append('## Missing experiments / gated claims')
for m in missing_experiments:
    md.append(f"- `{m.get('artifact')}` — current status: {m.get('current_status')}; needed for: {m.get('needed_for_claims_about','manual review')}; action: {m.get('required_action')}")
md.append('')
md.append('## Backward compatibility')
md.append(payload['backward_compatibility_note'])
(DOCS / 'manuscript_evidence_plan.md').write_text('\n'.join(md)+'\n', encoding='utf-8')

# Internal artifact/path/inclusion bookkeeping is written ONLY to results/*
# and docs/* artifacts. It must NEVER be written into any paper-facing file
# (paper/main.tex, paper/evidence_plan.tex): the manuscript's reproducibility
# prose is hand-authored and must not contain path/script/release-audit
# inventories. paper/evidence_plan.tex and paper/main.tex are intentionally
# left untouched by this generator.
PAPER_FACING = [PAPER / 'main.tex', PAPER / 'evidence_plan.tex']
paper_facing_before = {
    str(p): p.read_bytes() if p.exists() else None for p in PAPER_FACING
}

(RESULTS / 'manuscript_evidence_plan_inventory.md').write_text(
    '\n'.join(md) + '\n', encoding='utf-8')

# Guard: assert no paper-facing file was modified by this generator.
for p in PAPER_FACING:
    after = p.read_bytes() if p.exists() else None
    if after != paper_facing_before[str(p)]:
        raise SystemExit(
            f'Invariant violated: {p} was modified by the evidence-plan '
            f'generator. This generator must not write paper-facing files.')

print('WROTE docs/manuscript_evidence_plan.md')
print('WROTE results/manuscript_evidence_plan.json')
print('WROTE results/manuscript_evidence_plan_inventory.md')
print('PAPER-FACING FILES UNTOUCHED (paper/main.tex, paper/evidence_plan.tex)')
print('ACTIVE_RESULT_TABLE_INPUTS=', inputs)
print('ACTIVE_FIGURES=', includes)
print('RELEASE_FIGURE_STATUS_COUNT=', len(fig_entries))
print('MISSING_OR_GATED_COUNT=', len(missing_experiments))
