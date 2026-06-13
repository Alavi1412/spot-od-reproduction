from pathlib import Path
text = Path('paper/main.tex').read_text(encoding='utf-8')
checks = {
    'single_begin_abstract': text.count('\\begin{abstract}') == 1,
    'single_end_abstract': text.count('\\end{abstract}') == 1,
    'abstract_mentions_aukf': 'AUKF is the strongest aggregate performer' in text,
    'abstract_mentions_pure_gnn_negative': 'Pure GNN is a negative result' in text,
    'contributions_present': '\\paragraph{Evidence-bounded contributions.}' in text,
    'limitations_present': '\\section{Limitations}' in text,
    'conclusion_present': '\\section{Conclusion}' in text,
    'repro_citations_present': all(k in text for k in ['peng2011reproducible','sandve2013ten','acm2020artifact']),
    'main_results_input_present': '\\input{tables/main_results.tex}' in text,
    'significance_input_present': '\\input{tables/significance.tex}' in text,
    'ablation_input_present': '\\input{tables/ablation.tex}' in text,
    'four_main_figures_present': all(s in text for s in ['figures/per_step_rmse.png','figures/position_error_ecdf.png','figures/visibility_bucket_rmse.png','figures/uncertainty_calibration.png']),
}
missing_files = [p for p in ['paper/tables/main_results.tex','paper/tables/significance.tex','paper/tables/ablation.tex','paper/figures/per_step_rmse.png','paper/figures/position_error_ecdf.png','paper/figures/visibility_bucket_rmse.png','paper/figures/uncertainty_calibration.png'] if not Path(p).exists()]
checks['active_artifact_files_exist'] = not missing_files
failed = [k for k,v in checks.items() if not v]
print('Framing verification checks:')
for k,v in checks.items():
    print(f'- {k}: {v}')
if missing_files:
    print('Missing files:', missing_files)
if failed:
    raise SystemExit('FAILED: ' + ', '.join(failed))
print('PASSED')
