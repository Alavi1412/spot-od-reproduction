from pathlib import Path
import re

bib_path = Path('paper/references.bib')
bib = bib_path.read_text(encoding='utf-8') if bib_path.exists() else ''
tex = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in Path('paper').glob('*.tex'))
keys = set()
for m in re.finditer(r'\\(?:cite|citep|citet|citealp|parencite|textcite|autocite|footcite)(?:\[[^\]]*\])*\{([^}]+)\}', tex):
    for key in m.group(1).split(','):
        key = key.strip()
        if key:
            keys.add(key)
existing = set(m.group(1).strip() for m in re.finditer(r'@\w+\s*\{\s*([^,]+),', bib))

def add_alias(alias, canonical_entry):
    global bib
    if alias in existing:
        return False
    entry = re.sub(r'@([A-Za-z]+)\s*\{\s*[^,]+,', lambda mm: '@' + mm.group(1) + '{' + alias + ',', canonical_entry, count=1)
    if bib and not bib.endswith('\n'):
        bib += '\n'
    bib += '\n' + entry
    existing.add(alias)
    return True
revach_entry = r'''@article{revach2022kalmannet,
  title        = {KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics},
  author       = {Revach, Guy and Shlezinger, Nir and Ni, Xiaoyong and Lopez Escoriza, Adria and van Sloun, Ruud J. G. and Eldar, Yonina C.},
  journal      = {IEEE Transactions on Signal Processing},
  volume       = {70},
  pages        = {1532--1547},
  year         = {2022},
  doi          = {10.1109/TSP.2022.3158588},
  url          = {https://doi.org/10.1109/TSP.2022.3158588}
}
'''
tapley_entry = r'''@book{tapley2004statistical,
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
changed = False
for key in sorted(keys - existing):
    low = key.lower()
    if 'kalman' in low or 'revach' in low:
        changed |= add_alias(key, revach_entry)
    elif 'tapley' in low or 'statisticalorbit' in low or 'orbitdetermination' in low:
        changed |= add_alias(key, tapley_entry)
if changed:
    bib_path.write_text(bib, encoding='utf-8')
print('citation_alias_repair=ok')
print('tex_citation_keys=' + ','.join(sorted(keys)))
print('remaining_missing=' + ','.join(sorted(keys - existing)))
