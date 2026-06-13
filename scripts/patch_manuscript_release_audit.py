from pathlib import Path
import json, re, hashlib, datetime

ROOT = Path('.')
MAIN = ROOT / 'paper' / 'main.tex'
EPLAN = ROOT / 'paper' / 'evidence_plan.tex'
PKT = ROOT / 'results' / 'release_packet.json'
AUDIT_JSON = ROOT / 'results' / 'manuscript_release_consistency_audit.json'
AUDIT_MD = ROOT / 'results' / 'manuscript_release_consistency_audit.md'

# Paper-facing files. The manuscript-release consistency note must NEVER be
# injected into any of these: release/manuscript-inclusion bookkeeping,
# artifact/path inventories, and static-include audit mechanics are internal
# and live only in results/*.json and results/*.md.
PAPER_FACING = [MAIN, EPLAN]

for p in [MAIN, EPLAN, PKT]:
    if not p.exists():
        raise FileNotFoundError(f'Required file missing: {p}')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Guard: refuse to run if a paper-facing file still carries the old internal
# consistency patch or any artifact/path inventory leaked from this audit.
LEAK_MARKERS = [
    '% BEGIN MANUSCRIPT RELEASE CONSISTENCY PATCH',
    '% END MANUSCRIPT RELEASE CONSISTENCY PATCH',
    'Manuscript-release consistency note',
    'static include audit',
    'generated table inputs',
    'active figure includes',
]
for p in PAPER_FACING:
    t = p.read_text(encoding='utf-8')
    leaked = [m for m in LEAK_MARKERS if m in t]
    if leaked:
        raise SystemExit(
            f'Refusing to proceed: paper-facing file {p} contains internal '
            f'release-audit content {leaked}. Release consistency must live in '
            f'results/*.json and results/*.md, not in the manuscript. Remove the '
            f'internal block from {p} before re-running this audit.')

before = {str(p).replace('/', '\\'): sha256(p) for p in [MAIN, EPLAN, PKT]}

main_text = MAIN.read_text(encoding='utf-8')
lines = main_text.splitlines()


def active(line):
    return not line.lstrip().startswith('%')


input_re = re.compile(r'\\input\{([^}]+)\}')
include_re = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
active_inputs = []
active_figures = []
for i, line in enumerate(lines, 1):
    if not active(line):
        continue
    for m in input_re.finditer(line):
        p = m.group(1)
        norm = ('paper/' + p if not p.startswith('paper/') else p).replace('/', '\\')
        if not norm.endswith('.tex'):
            norm += '.tex'
        active_inputs.append({'line': i, 'content': line.strip(), 'path': norm})
    for m in include_re.finditer(line):
        p = m.group(1)
        norm = ('paper/' + p if not p.startswith('paper/') else p).replace('/', '\\')
        active_figures.append({'line': i, 'content': line.strip(), 'path': norm})

active_table_inputs = [x for x in active_inputs if '\\tables\\' in x['path'] or x['path'].startswith('paper\\tables')]
figure_envs = sum(1 for line in lines if active(line) and re.search(r'\\begin\{figure\}', line))
inline_tables = sum(1 for line in lines if active(line) and re.search(r'\\begin\{table\}', line))

pkt = json.loads(PKT.read_text(encoding='utf-8'))
old_counts = None
old_status = pkt.get('manuscript_inclusion_status')
if isinstance(old_status, dict):
    old_counts = old_status.get('counts')

counts = {
    'main_generated_table_count': len(active_table_inputs),
    'main_inline_table_count': inline_tables,
    'main_figure_environment_count': figure_envs,
    'main_figure_include_count': len(active_figures),
    'main_figure_count': len(active_figures),
}
claim_boundary = ('Only active, uncommented paper/main.tex inputs/includes are direct main-manuscript evidence. '
                  'Release, diagnostic, historical, and supplement-planned artifacts remain tracked but must not be described as included in the current main manuscript unless paper/main.tex includes them.')
status = {
    'claim_boundary': claim_boundary,
    'counts': counts,
    'active_table_line_evidence': active_table_inputs,
    'active_figure_line_evidence': active_figures,
}

# JSON-side ledger only. Preserve existing structured fields while adding/updating
# a compatibility-safe status block. No paper-facing file is touched.
pkt['manuscript_inclusion_status'] = status
pkt['main_generated_table_count'] = counts['main_generated_table_count']
pkt['main_inline_table_count'] = counts['main_inline_table_count']
pkt['main_figure_count'] = counts['main_figure_count']
pkt['main_generated_table_inputs'] = [x['path'] for x in active_table_inputs]
pkt['main_figure_includes'] = [x['path'] for x in active_figures]
pkt['claim_boundary'] = claim_boundary

after_paper_hashes = {str(p).replace('/', '\\'): sha256(p) for p in PAPER_FACING}
paper_unmodified = all(before[k] == after_paper_hashes[k] for k in after_paper_hashes)
audit = {
    'generated_at_utc': datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
    'task': 'Patch release-ledger manuscript-inclusion bookkeeping in results/*.json and results/*.md only',
    'changed_files': [
        'results\\release_packet.json',
        'results\\manuscript_release_consistency_audit.json',
        'results\\manuscript_release_consistency_audit.md',
    ],
    'unchanged_reference_files': ['paper\\main.tex', 'paper\\evidence_plan.tex'],
    'paper_facing_files_unmodified': paper_unmodified,
    'consistency_note_location': 'results/release_packet.json (manuscript_inclusion_status) and results/manuscript_release_consistency_audit.{json,md}',
    'paper_facing_injection': 'none -- internal release-audit content is never written into paper/*.tex',
    'before_counts_from_release_packet': old_counts,
    'after_counts_from_static_include_audit': counts,
    'static_include_audit': status,
    'claim_support_audit': {
        'finding': 'The ledger uses active, uncommented paper/main.tex inputs/includes as the boundary for main-manuscript evidence; this boundary is recorded only in results artifacts, not in the manuscript.',
        'release_only_artifact_rule': claim_boundary,
        'main_tex_modified': False,
        'evidence_plan_tex_modified': False,
    },
    'hashes_before': before,
}

pkt['current_state_patch_audit'] = audit
PKT.write_text(json.dumps(pkt, indent=2, sort_keys=False), encoding='utf-8')

after = {str(p).replace('/', '\\'): sha256(p) for p in [MAIN, EPLAN, PKT]}
audit['hashes_after'] = after
AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding='utf-8')

md = []
md.append('# Manuscript and Release Ledger Consistency Audit\n')
md.append(f"Generated: {audit['generated_at_utc']}\n")
md.append('\nThis consistency note is internal release-ledger bookkeeping. It is '
          'recorded only in `results/release_packet.json` and these results '
          'artifacts; it is never written into any paper-facing file '
          '(`paper/main.tex`, `paper/evidence_plan.tex`).\n')
md.append('\n## Changed files\n')
for f in audit['changed_files']:
    md.append(f'- `{f}`\n')
md.append('\n## Unchanged paper-facing reference files\n')
for f in audit['unchanged_reference_files']:
    md.append(f'- `{f}` (hash unchanged: '
              f"{'yes' if before[f] == after[f] else 'NO'})\n")
md.append('\n## Before/after counts\n')
md.append(f"- Before counts from `results/release_packet.json`: `{json.dumps(old_counts, sort_keys=True)}`\n")
md.append(f"- After static include audit counts: `{json.dumps(counts, sort_keys=True)}`\n")
md.append('\n## Static include audit\n')
md.append('### Active generated table inputs\n')
for x in active_table_inputs:
    md.append(f"- line {x['line']}: `{x['content']}` -> `{x['path']}`\n")
md.append('\n### Active figure includes\n')
for x in active_figures:
    md.append(f"- line {x['line']}: `{x['content']}` -> `{x['path']}`\n")
md.append('\n## Claim-support audit\n')
md.append(f"- {claim_boundary}\n")
md.append('- `paper/main.tex` and `paper/evidence_plan.tex` were used as evidence and were not modified by this audit.\n')
AUDIT_MD.write_text(''.join(md), encoding='utf-8')
print(json.dumps({
    'ok': True,
    'counts': counts,
    'changed_files': audit['changed_files'],
    'paper_facing_files_unmodified': paper_unmodified,
}, indent=2))
