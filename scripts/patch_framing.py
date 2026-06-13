from pathlib import Path
import re

path = Path('paper/main.tex')
text = path.read_text(encoding='utf-8')

abstract = r'''\begin{abstract}
Satellite state estimation under sparse optical and radio visibility remains a significant challenge for both classical dynamical filters and modern learned estimators. This paper presents SPOT-OD, an evidence-locked benchmark and methodological synthesis for sparse-visibility orbit determination. The study evaluates reliability-gated bounded residual correction over EKF/UKF/AUKF filtering priors against classical and learned baselines, while treating strong adaptive filters as first-class comparators rather than weak baselines. The supported result is deliberately conservative: prior-guided learned hybrids show statistically significant stress-regime gains over fixed-noise UKF in the active evaluation, but they do not uniformly dominate a tuned Adaptive Unscented Kalman Filter (AUKF). AUKF is the strongest aggregate performer in the displayed main-manuscript evidence, and the Pure GNN is a negative result relative to dynamics-aware baselines. The contribution is therefore not a claim of foundational algorithmic invention, universal GNN superiority, or flight-ready autonomy; it is a reproducible benchmark, a bounded hybrid-estimator formulation, and a claim-audited evaluation that identifies where learned residuals help, where they fail, and which validation gaps remain before operational conclusions would be justified.
\end{abstract}'''
text, n_abs = re.subn(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda _m: abstract, text, count=1, flags=re.S)
if n_abs != 1:
    raise SystemExit(f'Expected one abstract environment, replaced {n_abs}')

contrib = r'''\paragraph{Evidence-bounded contributions.}
This manuscript makes four scoped, citation-backed contributions. First, it positions graph learning as a complement to probabilistic orbit filtering rather than as a replacement for EKF/UKF/AUKF structure, consistent with classical filtering and statistical orbit-determination foundations~\cite{kalman1960,jazwinski1970stochastic,julier1997unscented,wan2000unscented,tapley2004statistical}. Second, it reports the strongest aggregate evidence in favor of AUKF while making the Pure GNN's weaker performance an explicit negative result rather than hiding it behind selective plots. Third, it limits the learned-method claim to stress-regime gains over fixed-noise UKF and explicitly states that the active evidence does not show uniform superiority over AUKF. Fourth, it ties the paper's main claims to the active tables and figures and preserves reproducibility provenance following computational reproducibility and artifact-evaluation guidance~\cite{peng2011reproducible,sandve2013ten,acm2020artifact}.

'''
marker = r'\paragraph{Evidence-bounded contributions.}'
if marker not in text:
    insert_before = r'\paragraph{Contributions.}'
    if insert_before not in text:
        raise SystemExit('Could not find Contributions paragraph for insertion')
    text = text.replace(insert_before, contrib + insert_before, 1)
else:
    text = re.sub(r'\\paragraph\{Evidence-bounded contributions\.\}.*?(?=\n\n\\paragraph\{Contributions\.\})', lambda _m: contrib.rstrip(), text, count=1, flags=re.S)

# Rename the limitations section so downstream checks and readers see the explicit section requested.
text = text.replace(r'\section{Limitations and Missing Experiments}', r'\section{Limitations}', 1)

# Strengthen the opening limitations paragraph without removing the detailed missing-experiment list.
old_lim = ("The evidence boundary for SPOT-OD remains narrower than the broader research ambition. "
           "The current results should be read as benchmark evidence for the documented simulation protocol, evaluation windows, baselines, and active manuscript artifacts. "
           "They do not validate the estimator for flight operations, hardware-in-the-loop deployment, safety-critical autonomy, or all sparse-visibility orbital regimes.")
new_lim = ("The evidence boundary for SPOT-OD remains narrower than the broader research ambition. "
           "The current results should be read as benchmark evidence for the documented simulation protocol, evaluation windows, baselines, and active manuscript artifacts. "
           "AUKF is the strongest aggregate performer in the active main-manuscript evidence, and the Pure GNN is a negative result for direct neural replacement under these conditions. "
           "These results do not validate the estimator for flight operations, hardware-in-the-loop deployment, safety-critical autonomy, or all sparse-visibility orbital regimes.")
if old_lim in text:
    text = text.replace(old_lim, new_lim, 1)

conclusion = r'''\section{Conclusion}
The active SPOT-OD evidence supports a cautious and disciplined conclusion. Learned residual components provide statistically significant stress-regime gains over fixed-noise UKF in the displayed comparisons, but they do not uniformly replace or dominate well-tuned adaptive filters. AUKF is the strongest aggregate performer in the current benchmark, and the Pure GNN is a negative result for standalone neural state prediction relative to dynamics-aware baselines.

The methodological contribution is therefore best framed as reliability-gated bounded residual correction around filtering priors, not as foundational algorithmic invention or flight-ready autonomy. The ablation evidence also leaves graph-specific superiority unisolated: learned gains may arise from the temporal backbone, filtering prior, learned noise adaptation, residual bounds, or their interaction. Future work must expand topology generalization, process-noise sensitivity, station-subset robustness, parameter-matched graph/no-graph ablations, and real-data validation before making operational deployment claims.
'''
text, n_conc = re.subn(r'\\section\{Conclusion\}.*?(?=\n\\bibliographystyle)', lambda _m: conclusion.rstrip() + '\n', text, count=1, flags=re.S)
if n_conc != 1:
    raise SystemExit(f'Expected one Conclusion section, replaced {n_conc}')

path.write_text(text, encoding='utf-8')
print('patched paper/main.tex')
print('abstract replacements', n_abs, 'conclusion replacements', n_conc)
