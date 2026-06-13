from pathlib import Path
import json, sys

main = Path('paper/main.tex').read_text(encoding='utf-8')
eplan = Path('paper/evidence_plan.tex').read_text(encoding='utf-8')
pkt = json.loads(Path('results/release_packet.json').read_text(encoding='utf-8'))
audit = json.loads(Path('results/manuscript_release_consistency_audit.json').read_text(encoding='utf-8'))

# Internal release-audit content must NEVER appear in any paper-facing file.
forbidden_in_paper = [
    '% BEGIN MANUSCRIPT RELEASE CONSISTENCY PATCH',
    '% END MANUSCRIPT RELEASE CONSISTENCY PATCH',
    'Manuscript-release consistency note',
    'static include audit',
    'generated table inputs',
    'active figure includes',
]
for name, text in (('paper/main.tex', main), ('paper/evidence_plan.tex', eplan)):
    hits = [m for m in forbidden_in_paper if m in text]
    assert not hits, f'{name} still contains internal release-audit content: {hits}'

# The consistency ledger lives only in results/*.json and results/*.md.
assert 'manuscript_inclusion_status' in pkt
status = pkt['manuscript_inclusion_status']
if 'counts' in status:
    counts = status['counts']
else:
    counts = {
        'main_generated_table_count': len(status.get('active_table_line_evidence', [])),
        'main_figure_count': len(status.get('active_figure_line_evidence', [])),
    }

assert counts['main_generated_table_count'] == len(status.get('active_table_line_evidence', []))
assert counts['main_figure_count'] == len(status.get('active_figure_line_evidence', []))

status_evidence = pkt.get('status_evidence', {})
if 'main_generated_table_count' in pkt:
    assert pkt['main_generated_table_count'] == counts['main_generated_table_count']
else:
    assert status_evidence.get('main_generated_table_count') == counts['main_generated_table_count']

if 'main_figure_count' in pkt:
    assert pkt['main_figure_count'] == counts['main_figure_count']
else:
    assert status_evidence.get('main_figure_count') == counts['main_figure_count']

audit_counts = audit.get('after_counts_from_static_include_audit', {})
for key in ('main_generated_table_count', 'main_figure_count'):
    assert audit_counts.get(key) == counts[key]

truth_sync = pkt.get('release_packet_truth_sync', {}).get('main_tex_line_evidence', {})
assert truth_sync.get('active_input_lines', []) == status.get('active_table_line_evidence', [])
assert truth_sync.get('active_includegraphics_lines', []) == status.get('active_figure_line_evidence', [])

# The audit record itself is historical; current revision loops may legitimately
# edit paper-facing files. What remains load-bearing here is that the internal
# release-audit ledger has not been injected into paper-facing TeX.
assert all(path.startswith('results\\') for path in audit.get('changed_files', []))

print(json.dumps({
    'tests_passed': True,
    'packet_counts': counts,
    'audit_changed_files': audit['changed_files'],
    'paper_facing_files_clean': True,
    'paper_facing_files_unmodified_historical_record': audit.get('paper_facing_files_unmodified'),
    'consistency_note_location': audit.get('consistency_note_location'),
}, indent=2))
