import json
import pathlib
import re

main_path = pathlib.Path('paper/main.tex')
packet_path = pathlib.Path('results/release_packet.json')
release_md_path = pathlib.Path('paper/RELEASE_PACKET.md')


def strip_comment(line: str) -> str:
    out = []
    escaped = False
    for ch in line:
        if ch == '%' and not escaped:
            break
        out.append(ch)
        if ch == '\\' and not escaped:
            escaped = True
        else:
            escaped = False
    return ''.join(out)


def norm(path_str: str) -> str:
    return path_str.replace('/', '\\').replace('\\\\', '\\').strip()


main = main_path.read_text(encoding='utf-8')
packet = json.loads(packet_path.read_text(encoding='utf-8'))
release_md = release_md_path.read_text(encoding='utf-8')

inputs = []
figures = []
inline_tables = []
input_lines = []
figure_lines = []
figure_env_lines = []

for line_no, raw_line in enumerate(main.splitlines(), start=1):
    line = strip_comment(raw_line)
    for raw in re.findall(r'\\input\{([^}]*)\}', line):
        raw = raw.strip()
        if raw.startswith('tables/') or raw.startswith('tables\\'):
            stem = raw if raw.endswith('.tex') else raw + '.tex'
            path = norm('paper/' + stem.replace('\\', '/'))
            inputs.append(path)
            input_lines.append({'line': line_no, 'content': line.strip(), 'path': path})
    for raw in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}', line):
        raw = raw.strip()
        if raw.startswith('figures/') or raw.startswith('figures\\'):
            if '.' not in pathlib.Path(raw).name:
                raw += '.png'
            path = norm('paper/' + raw.replace('\\', '/'))
            figures.append(path)
            figure_lines.append({'line': line_no, 'content': line.strip(), 'path': path})
    if re.findall(r'\\begin\{figure\*?\}', line):
        figure_env_lines.append({'line': line_no, 'content': line.strip(), 'path': norm('paper/main.tex')})
    inline_tables.extend(re.findall(r'\\begin\{table\*?\}', line))

# Inline tables are estimator/notation tables authored directly in paper/main.tex
# (identified by their \label{tab:...}) rather than pulled in via \input{tables/...}.
# The release packet records them as "paper/main.tex#<label>" using exactly this
# scan (see scripts/build_paper_assets.py build_release_packet); mirror it here so
# the inline inventory is verified against main.tex instead of assumed empty.
inline_table_refs = [f"paper/main.tex#{m}" for m in re.findall(r'\\label\{(tab:[^}]+)\}', main)]

status = packet['status_evidence']
current = packet['current_main_manuscript_artifacts']
manuscript_status = packet.get('manuscript_inclusion_status', {})
release_metadata = packet.get('release_metadata', {})
truth_sync = packet.get('release_packet_truth_sync', {})
truth_line_evidence = truth_sync.get('main_tex_line_evidence', {}) if isinstance(truth_sync, dict) else {}
duplicate_line_evidence_match = (
    packet.get('active_table_line_evidence') == input_lines
    and packet.get('active_figure_line_evidence') == figure_lines
    and manuscript_status.get('active_table_line_evidence') == input_lines
    and manuscript_status.get('active_figure_line_evidence') == figure_lines
    and release_metadata.get('active_table_line_evidence') == input_lines
    and release_metadata.get('active_figure_line_evidence') == figure_lines
    and truth_line_evidence.get('active_input_lines') == input_lines
    and truth_line_evidence.get('active_includegraphics_lines') == figure_lines
    and truth_line_evidence.get('active_figure_environment_lines') == figure_env_lines
)
result = {
    'main_tex_input_tables': inputs,
    'main_tex_input_table_lines': input_lines,
    'main_tex_input_table_count': len(inputs),
    'main_tex_inline_table_count': len(inline_tables),
    'main_tex_includegraphics': figures,
    'main_tex_includegraphics_lines': figure_lines,
    'main_tex_figure_environment_lines': figure_env_lines,
    'main_tex_figure_count': len(figures),
    'json_valid': True,
    'release_md_has_current_table_section': '## Current Main-Manuscript Generated Table Inputs' in release_md,
    'release_md_has_current_figure_section': '## Current Main-Manuscript Figure Includes' in release_md,
    'release_md_claim_boundary_present': 'This release-truth synchronization changes metadata labels only' in release_md,
    'release_md_no_included_in_manuscript_heading': 'Included In The Manuscript' not in release_md,
    'json_status_evidence_counts': {
        'main_generated_table_count': status.get('main_generated_table_count'),
        'main_inline_table_count': status.get('main_inline_table_count'),
        'main_figure_count': status.get('main_figure_count'),
    },
    'json_current_main_manuscript_artifacts': current,
    'json_duplicate_line_evidence_match': duplicate_line_evidence_match,
    'json_duplicate_line_evidence': {
        'top_level_tables': packet.get('active_table_line_evidence'),
        'top_level_figures': packet.get('active_figure_line_evidence'),
        'manuscript_status_tables': manuscript_status.get('active_table_line_evidence'),
        'manuscript_status_figures': manuscript_status.get('active_figure_line_evidence'),
        'release_metadata_tables': release_metadata.get('active_table_line_evidence'),
        'release_metadata_figures': release_metadata.get('active_figure_line_evidence'),
        'truth_sync_tables': truth_line_evidence.get('active_input_lines'),
        'truth_sync_figures': truth_line_evidence.get('active_includegraphics_lines'),
    },
    'json_counts_match_main_tex': (
        status.get('main_generated_table_count') == len(inputs)
        and status.get('main_inline_table_count') == len(inline_tables)
        and status.get('main_figure_count') == len(figures)
        and status.get('active_table_input_lines') == input_lines
        and status.get('active_figure_include_lines') == figure_lines
        and status.get('active_figure_environment_lines') == figure_env_lines
        and current.get('generated_tables') == inputs
        and current.get('inline_tables') == inline_table_refs
        and status.get('main_inline_tables') == inline_table_refs
        and len(inline_table_refs) == len(inline_tables)
        and current.get('figures') == figures
        and duplicate_line_evidence_match
    ),
    'main_tex_inline_table_refs': inline_table_refs,
    'json_inline_tables_match_main_tex': (
        current.get('inline_tables') == inline_table_refs
        and status.get('main_inline_tables') == inline_table_refs
        and len(inline_table_refs) == len(inline_tables)
    ),
}
print(json.dumps(result, indent=2, sort_keys=True))
