from pathlib import Path
import json
import re

required_keys = [
    'revach2022kalmannet', 'tapley2004statistical', 'vallado2006revisiting',
    'scarselli2009graph', 'gilmer2017neural', 'battaglia2016interaction',
    'battaglia2018relational', 'sanchezgonzalez2020learning', 'chen2018neural',
    'kipf2018neural'
]
root = Path('.')
bib_text = (root / 'paper' / 'references.bib').read_text(encoding='utf-8') if (root / 'paper' / 'references.bib').exists() else ''
lit_text = (root / 'literature_review_gnn_state_estimation.md').read_text(encoding='utf-8') if (root / 'literature_review_gnn_state_estimation.md').exists() else ''
audit_text = (root / 'docs' / 'repro_build_blocker_audit_2026-05-13.md').read_text(encoding='utf-8') if (root / 'docs' / 'repro_build_blocker_audit_2026-05-13.md').exists() else ''
log_text = (root / 'paper' / 'main.log').read_text(encoding='utf-8', errors='replace') if (root / 'paper' / 'main.log').exists() else ''
pytest_text = (root / 'results' / 'pytest_full_attempt.txt').read_text(encoding='utf-8', errors='replace') if (root / 'results' / 'pytest_full_attempt.txt').exists() else ''
requirements_text = (root / 'requirements.txt').read_text(encoding='utf-8') if (root / 'requirements.txt').exists() else ''
release_text = (root / 'results' / 'release_packet_validation.json').read_text(encoding='utf-8') if (root / 'results' / 'release_packet_validation.json').exists() else ''

present = {key: bool(re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,', bib_text, flags=re.IGNORECASE)) for key in required_keys}
main_log_citation_warnings = re.findall(r"Citation `[^']+' on page .*? undefined|LaTeX Warning: Citation .*? undefined", log_text)
main_log_reference_warnings = [line.strip() for line in log_text.splitlines() if 'undefined' in line.lower()][:50]
pytest_missing = sorted(set(re.findall(r"ModuleNotFoundError: No module named '([^']+)'", pytest_text)))
req_declared = {name: (name in requirements_text.lower()) for name in ['pandas', 'torch', 'sgp4']}
report = {
    'working_directory_note': 'Verification scripts were run from the project working directory.',
    'required_bibliography_keys_present': present,
    'all_required_bibliography_keys_present': all(present.values()),
    'literature_review_has_open_questions_section': '## Open questions for this repository' in lit_text,
    'literature_review_source_matrix_rows': lit_text.count('| S'),
    'audit_has_bibliography_sync_correction': '## Bibliography synchronization correction' in audit_text,
    'paper_main_log_exists': bool(log_text),
    'paper_main_log_undefined_citation_warning_count': len(main_log_citation_warnings),
    'paper_main_log_undefined_warning_examples': main_log_reference_warnings[:10],
    'pytest_full_attempt_exists': bool(pytest_text),
    'pytest_missing_imports_detected_in_saved_attempt': pytest_missing,
    'requirements_declares_blocked_imports': req_declared,
    'release_packet_validation_file_exists': bool(release_text),
    'files_updated_or_verified': [
        'paper/references.bib',
        'literature_review_gnn_state_estimation.md',
        'docs/repro_build_blocker_audit_2026-05-13.md',
        'results/repro_audit_verification_2026-05-13.json'
    ]
}
out = root / 'results' / 'repro_audit_verification_2026-05-13.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
print(json.dumps(report, indent=2, sort_keys=True))
