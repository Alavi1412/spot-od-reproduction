from pathlib import Path
import re

root = Path('.')
bib_path = root / 'paper' / 'references.bib'
bib = bib_path.read_text(encoding='utf-8') if bib_path.exists() else ''

entries = {
    'revach2022kalmannet': r'''@article{revach2022kalmannet,
  title        = {KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics},
  author       = {Revach, Guy and Shlezinger, Nir and Ni, Xiaoyong and Lopez Escoriza, Adria and van Sloun, Ruud J. G. and Eldar, Yonina C.},
  journal      = {IEEE Transactions on Signal Processing},
  volume       = {70},
  pages        = {1532--1547},
  year         = {2022},
  doi          = {10.1109/TSP.2022.3158588},
  url          = {https://doi.org/10.1109/TSP.2022.3158588}
}
''',
    'tapley2004statistical': r'''@book{tapley2004statistical,
  title        = {Statistical Orbit Determination},
  author       = {Tapley, Byron D. and Schutz, Bob E. and Born, George H.},
  publisher    = {Elsevier Academic Press},
  address      = {Burlington, MA},
  year         = {2004},
  isbn         = {9780126836301},
  doi          = {10.1016/B978-0-12-683630-1.X5000-X},
  url          = {https://doi.org/10.1016/B978-0-12-683630-1.X5000-X}
}
'''
}

changed = False
for key, entry in entries.items():
    if not re.search(r'@\w+\s*\{\s*' + re.escape(key) + r'\s*,', bib, flags=re.IGNORECASE):
        if bib and not bib.endswith('\n'):
            bib += '\n'
        bib += '\n' + entry
        changed = True
if changed:
    bib_path.write_text(bib, encoding='utf-8')

# Add a timestamp-free synchronization note to the blocker audit without rewriting the existing matrix.
audit_path = root / 'docs' / 'repro_build_blocker_audit_2026-05-13.md'
audit_note = '''

## Bibliography synchronization correction

Fresh remediation added the missing primary/domain bibliography records that the prior review identified as absent from `paper/references.bib`:

| Key | Source | Why it is required | Repository status |
|---|---|---|---|
| `revach2022kalmannet` | Revach et al., "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics," *IEEE Transactions on Signal Processing*, 70:1532--1547, 2022, DOI: 10.1109/TSP.2022.3158588. | Primary neural state-estimation baseline for partially known dynamics and learned Kalman filtering. | Present in `paper/references.bib`. |
| `tapley2004statistical` | Tapley, Schutz, and Born, *Statistical Orbit Determination*, Elsevier Academic Press, 2004, ISBN 9780126836301. | Domain reference for orbit-determination measurement modeling, filtering, covariance, and observability context. | Present in `paper/references.bib`. |

This correction narrows the earlier blocker from "missing primary sources in the bibliography" to a verification item: rebuild the LaTeX manuscript in an environment with the declared TeX/Biber toolchain and check that no `Citation ... undefined` warnings remain. Undefined figure/table/reference warnings, if any, should be tracked separately from bibliography synchronization.
'''
if audit_path.exists():
    audit = audit_path.read_text(encoding='utf-8')
    if '## Bibliography synchronization correction' not in audit:
        audit_path.write_text(audit.rstrip() + audit_note + '\n', encoding='utf-8')

print('bibliography_sync=ok')
print('references_bib=' + str(bib_path))
print('added_or_present=' + ','.join(entries.keys()))
