from __future__ import annotations
import json, hashlib, re
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'validation'
OUT.mkdir(parents=True, exist_ok=True)
def rel(p): return str(p.relative_to(ROOT)).replace('/','\\')
def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
    return h.hexdigest()
def info(path):
    p=ROOT/path
    return {'path':str(path).replace('/','\\'), 'exists':p.exists(), 'bytes':p.stat().st_size if p.exists() else None, 'sha256':sha(p) if p.exists() and p.is_file() else None}
def load_json(path):
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}
def validation_summary(data):
    pdf=data.get('pdf_evidence') or data.get('pdf') or {}
    cite=data.get('citation_validation') or {}
    release=data.get('release_evidence_index') or data.get('release_packet_validation') or {}
    warnings=data.get('latex_warnings') or {}
    return {
        'status': data.get('status'),
        'overall_passed': data.get('overall_passed'),
        'pass_fail': data.get('pass_fail', {}),
        'pdf': {
            'path': pdf.get('path'),
            'exists': pdf.get('exists'),
            'bytes': pdf.get('bytes'),
            'sha256': pdf.get('sha256'),
            'page_count': pdf.get('page_count') or pdf.get('pages'),
            'page_source': pdf.get('page_source'),
        },
        'citation_validation': {
            'status': cite.get('status'),
            'cited_key_count': cite.get('cited_key_count'),
            'bib_entry_count': cite.get('bib_entry_count'),
            'bbl_bibitem_count': cite.get('bbl_bibitem_count'),
            'finding_counts': {k: len(v) for k, v in (cite.get('findings') or {}).items() if isinstance(v, list)},
        },
        'release_packet_validation': {
            'status': release.get('status'),
            'path': release.get('path'),
            'checked_artifact_count': release.get('checked_artifact_count'),
            'missing_artifact_count': len(release.get('missing_artifacts') or []),
            'sha256': release.get('sha256'),
        },
        'latex_warning_summary': {
            'count': warnings.get('count'),
            'count_basis': warnings.get('count_basis'),
            'raw_line_count': warnings.get('raw_line_count'),
            'deduplicated_line_count': warnings.get('deduplicated_line_count'),
            'supplement_warning_count': warnings.get('supplement_warning_count'),
            'category_counts': warnings.get('category_counts', {}),
            'documents': {
                name: {
                    'log_path': row.get('log_path'),
                    'log_exists': row.get('log_exists'),
                    'count': row.get('count'),
                    'count_basis': row.get('count_basis'),
                    'raw_line_count': row.get('raw_line_count'),
                    'deduplicated_line_count': row.get('deduplicated_line_count'),
                    'category_counts': row.get('category_counts', {}),
                }
                for name, row in (warnings.get('documents') or {}).items()
                if isinstance(row, dict)
            },
        },
        'active_include_counts': data.get('active_include_counts') or data.get('counts'),
        'supporting_artifacts': data.get('supporting_artifacts', {}),
    }
def pytest_summary(text):
    clean=text.replace('\x00','')
    passed_match=re.findall(r'\b(\d+)\s+passed\b', clean, re.I)
    failed=bool(re.search(r'\bfailed\b|\berror\b', clean, re.I))
    return {
        'log':'results\\validation\\pytest.log',
        'passed': bool(passed_match) and not failed,
        'passed_count': int(passed_match[-1]) if passed_match else None,
        'failed_or_error_detected': failed,
        'line_count': len(clean.splitlines()),
    }
def pdf_build_summary(text):
    clean=text.replace('\x00','')
    page_matches=re.findall(r'Output written on .*? \((\d+) pages?,', clean)
    return {
        'log':'results\\validation\\pdf_build.log',
        'blocked':'BLOCKED:' in clean,
        'exit_codes': re.findall(r'\[exit_code\] (\d+)', clean),
        'page_count': int(page_matches[-1]) if page_matches else None,
        'warning_count': len(re.findall(r'Warning', clean)),
        'overfull_count': len(re.findall(r'^Overfull ', clean, re.M)),
        'underfull_count': len(re.findall(r'^Underfull ', clean, re.M)),
        'line_count': len(clean.splitlines()),
    }
artifacts=[
    Path('results/release_packet.json'),
    Path('results/validation/submission_validation.json'),
    Path('results/validation/citation_validation.json'),
    Path('results/validation/table_figure_plan.json'),
    Path('results/validation/claim_audit.json'),
    Path('results/validation/submission_validation_command.log'),
    Path('results/validation/pdf_build.log'),
    Path('results/validation/pytest.log'),
    Path('paper/main.pdf'),
    Path('paper/tables/main_results.tex'),
    Path('paper/tables/significance.tex'),
    Path('paper/tables/ablation.tex'),
    Path('paper/figures/per_step_rmse.png'),
    Path('paper/figures/position_error_ecdf.png'),
    Path('paper/figures/visibility_bucket_rmse.png'),
    Path('paper/figures/uncertainty_calibration.png'),
]
index={'generated_at':datetime.now(timezone.utc).isoformat(), 'root_label':'submission_root', 'artifacts':[info(p) for p in artifacts]}
# Parse statuses when available.
sv=ROOT/'results/validation/submission_validation.json'
if sv.exists():
    index['submission_validation']=validation_summary(load_json(sv))
pl=ROOT/'results/validation/pytest.log'
if pl.exists():
    text=pl.read_text(encoding='utf-8', errors='replace')
    index['pytest']=pytest_summary(text)
bl=ROOT/'results/validation/pdf_build.log'
if bl.exists():
    text=bl.read_text(encoding='utf-8', errors='replace')
    index['pdf_build']=pdf_build_summary(text)
out=OUT/'release_evidence_index.json'
out.write_text(json.dumps(index, indent=2), encoding='utf-8')
print(json.dumps({'wrote':rel(out), 'artifact_count':len(index['artifacts']), 'all_indexed_exist':all(a['exists'] for a in index['artifacts'])}, indent=2))
