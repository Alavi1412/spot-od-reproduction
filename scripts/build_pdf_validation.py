from __future__ import annotations
import subprocess, sys, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'validation'
OUT.mkdir(parents=True, exist_ok=True)
log=[]
cmds=[]
if shutil.which('pdflatex') and shutil.which('bibtex'):
    cmds=[['pdflatex','-interaction=nonstopmode','main.tex'],['bibtex','main'],['pdflatex','-interaction=nonstopmode','main.tex'],['pdflatex','-interaction=nonstopmode','main.tex']]
else:
    missing=[x for x in ['pdflatex','bibtex'] if not shutil.which(x)]
    log.append('BLOCKED: missing executable(s): '+', '.join(missing))
    (OUT/'pdf_build.log').write_text('\n'.join(log), encoding='utf-8')
    print('\n'.join(log))
    sys.exit(2)
rc=0
for cmd in cmds:
    log.append('$ '+' '.join(cmd))
    p=subprocess.run(cmd, cwd=ROOT/'paper', text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, errors='replace')
    log.append(p.stdout)
    log.append(f'[exit_code] {p.returncode}')
    if p.returncode!=0:
        rc=p.returncode
        break
(OUT/'pdf_build.log').write_text('\n'.join(log), encoding='utf-8')
print('\n'.join(log[-20:]))
sys.exit(rc)
