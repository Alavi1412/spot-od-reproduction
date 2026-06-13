from __future__ import annotations
import json, re, hashlib, os, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace('/', os.sep)
    except Exception:
        return str(p)

def run(cmd, cwd=ROOT, timeout=300):
    try:
        cp = subprocess.run(cmd, cwd=str(cwd), shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        output = cp.stdout or ''
        return {
            'execution_details_redacted': True,
            'returncode': cp.returncode,
            'ok': cp.returncode == 0,
            'output_sha256': hashlib.sha256(output.encode('utf-8', errors='replace')).hexdigest(),
            'output_line_count': len(output.splitlines()),
        }
    except Exception as e:
        return {'execution_details_redacted': True, 'returncode': None, 'ok': False, 'error_type': type(e).__name__}

def active_tex_lines(tex: str, pattern: str):
    out=[]
    rx=re.compile(pattern)
    for i,line in enumerate(tex.splitlines(),1):
        stripped=line.lstrip()
        if stripped.startswith('%'):
            continue
        m=rx.search(line)
        if m:
            out.append({'line': i, 'content': line.strip(), 'target': m.group(1)})
    return out

def extract_cites(tex: str):
    keys=[]
    for m in re.finditer(r'\\(?:cite|citep|citet|citealp|autocite|parencite|textcite)(?:\[[^\]]*\])*\{([^}]+)\}', tex):
        for k in m.group(1).split(','):
            k=k.strip()
            if k:
                keys.append(k)
    return sorted(set(keys))

def bib_entries(paths):
    entries=set(); existing=[]
    for p in paths:
        if p.exists():
            existing.append(rel(p))
            text=p.read_text(encoding='utf-8', errors='replace')
            entries.update(re.findall(r'@\w+\s*\{\s*([^,\s]+)', text))
    return sorted(entries), existing

main_path = PAPER / 'main.tex'
main_tex = main_path.read_text(encoding='utf-8', errors='replace')
fig_lines = active_tex_lines(main_tex, r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
table_lines = active_tex_lines(main_tex, r'\\input\{(tables/[^}]+)\}')

active_figures=[]
for x in fig_lines:
    p = PAPER / x['target']
    active_figures.append({**x, 'path': rel(p), 'exists': p.exists(), 'sha256': sha256(p) if p.exists() else None, 'bytes': p.stat().st_size if p.exists() else None})
active_tables=[]
for x in table_lines:
    p = PAPER / x['target']
    active_tables.append({**x, 'path': rel(p), 'exists': p.exists(), 'sha256': sha256(p) if p.exists() else None, 'bytes': p.stat().st_size if p.exists() else None})

# Validate release packet consistency against active manuscript refs.
release_path = RESULTS / 'release_packet.json'
release = json.loads(release_path.read_text(encoding='utf-8')) if release_path.exists() else {}
release_text = json.dumps(release, sort_keys=True)
release_refs = {rel(PAPER / x['target']): (rel(PAPER / x['target']) in release_text or x['target'].replace('/', '\\\\') in release_text) for x in fig_lines+table_lines}

# Citation validation.
bibnames=[]
for m in re.finditer(r'\\bibliography\{([^}]+)\}', main_tex):
    bibnames += [b.strip() for b in m.group(1).split(',') if b.strip()]
if not bibnames:
    for m in re.finditer(r'\\addbibresource\{([^}]+)\}', main_tex):
        bibnames.append(m.group(1).replace('.bib',''))
bib_paths=[]
for b in bibnames or ['references','refs','main']:
    candidate = PAPER / (b if b.endswith('.bib') else b + '.bib')
    bib_paths.append(candidate)
    candidate2 = ROOT / (b if b.endswith('.bib') else b + '.bib')
    if candidate2 != candidate:
        bib_paths.append(candidate2)
entries, existing_bibs = bib_entries(bib_paths)
cites = extract_cites(main_tex)
missing_citations = sorted(set(cites) - set(entries))
unused_entries = sorted(set(entries) - set(cites))

# PDF evidence.
pdf_candidates=[PAPER/'main.pdf', ROOT/'main.pdf']
pdf_info=[]
for p in pdf_candidates:
    if p.exists():
        data=p.read_bytes()[:8]
        pdf_info.append({'path': rel(p), 'exists': True, 'bytes': p.stat().st_size, 'sha256': sha256(p), 'pdf_header': data.startswith(b'%PDF')})

# log readback
log_path=PAPER/'main.log'
log_flags={}
if log_path.exists():
    log=log_path.read_text(encoding='utf-8', errors='replace')
    log_flags={
        'has_undefined_references': 'undefined references' in log.lower() or 'undefined citation' in log.lower(),
        'has_latex_error': '! latex error' in log.lower() or '! emergency stop' in log.lower(),
        'output_pdf_line_present': any('Output written on' in line for line in log.splitlines()),
        'warning_count': sum(1 for line in log.splitlines() if 'Warning' in line),
        'overfull_count': sum(1 for line in log.splitlines() if line.startswith('Overfull ')),
        'underfull_count': sum(1 for line in log.splitlines() if line.startswith('Underfull ')),
    }

# Command manifest/status files.
status_files=[]
for pat in ['*manifest*','*status*','*validation*']:
    for p in RESULTS.glob(pat):
        if p.is_file():
            status_files.append({'path': rel(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)})

# Optional release validator re-run.
validator = run('python scripts/validate_release.py', ROOT, 300) if (ROOT/'scripts'/'validate_release.py').exists() else {'ok': False, 'error': 'scripts/validate_release.py not found'}

all_artifacts_exist = all(x['exists'] for x in active_figures+active_tables)
release_consistent = bool(release_path.exists()) and all(release_refs.values())
pdf_ok = any(x.get('pdf_header') and x.get('bytes',0)>1000 for x in pdf_info)
citations_ok = len(cites) > 0 and len(existing_bibs) > 0 and not missing_citations
log_ok = not log_flags.get('has_undefined_references', True) and not log_flags.get('has_latex_error', True) and bool(log_flags.get('output_pdf_line_present'))

report={
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'tests_passed': all([validator.get('ok'), all_artifacts_exist, release_consistent, pdf_ok, citations_ok, log_ok]),
    'validation_results': {
        'release_validator': validator,
        'active_artifacts_exist': all_artifacts_exist,
        'release_packet_consistent_with_active_refs': release_consistent,
        'pdf_ok': pdf_ok,
        'citations_ok': citations_ok,
        'latex_log_ok': log_ok,
    },
    'pdf_evidence': {'candidates': pdf_info, 'log_path': rel(log_path), 'log_flags': log_flags, 'log_details_redacted': True},
    'release_evidence_index': {
        'release_packet_path': rel(release_path),
        'release_packet_exists': release_path.exists(),
        'release_packet_sha256': sha256(release_path) if release_path.exists() else None,
        'active_figures': active_figures,
        'active_tables': active_tables,
        'release_refs_found': release_refs,
        'counts': {'active_figure_count': len(active_figures), 'active_generated_table_count': len(active_tables)}
    },
    'citations': {'citation_keys_in_main_tex': cites, 'bibliography_files': existing_bibs, 'bib_entry_count': len(entries), 'missing_citations': missing_citations, 'unused_bib_entries': unused_entries[:100]},
    'sources': {'main_tex': rel(main_path), 'bibliography_files': existing_bibs, 'release_packet': rel(release_path), 'status_files': status_files},
    'status_evidence': {'command_manifest_or_status_files': status_files, 'validator_status': validator.get('ok'), 'latex_log_flags': log_flags}
}
out=RESULTS/'submission_validation_evidence.json'
out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps({'wrote': rel(out), 'tests_passed': report['tests_passed'], 'summary': report['validation_results']}, indent=2))
sys.exit(0 if report['tests_passed'] else 1)
