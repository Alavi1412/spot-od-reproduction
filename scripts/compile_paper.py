#!/usr/bin/env python
"""Compile the paper from the correct working directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--paper-dir", type=str, default="paper")
    p.add_argument("--tex-file", type=str, default="main.tex")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--skip-bibtex", action="store_true")
    p.add_argument(
        "--with-supplement",
        action="store_true",
        help=(
            "Compile paper/supplement.tex first (so its .aux exists for xr "
            "cross-references) and then compile the requested --tex-file."
        ),
    )
    return p


def _run_compile_pipeline(paper_dir: Path, tex_file: str, runs: int, skip_bibtex: bool) -> None:
    """Run a self-contained pdflatex + bibtex + pdflatex pipeline for one tex
    file. Factored out so the supplement and main documents can each be
    compiled in sequence by ``--with-supplement``.
    """
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file]
    stem = Path(tex_file).stem
    aux_like = [paper_dir / f"{stem}.aux", paper_dir / f"{stem}.out"]
    first_pass_count = 1 if not skip_bibtex and (paper_dir / "references.bib").exists() else 0
    total_runs = first_pass_count + max(runs, 1)
    run_idx = 0
    if first_pass_count:
        run_idx += 1
        print(f"[compile {run_idx}/{total_runs}] " + " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=paper_dir, check=True)
        except subprocess.CalledProcessError:
            for path in aux_like:
                remove_if_exists(path)
            subprocess.run(cmd, cwd=paper_dir, check=True)
        bib_cmd = ["bibtex", stem]
        print("[bibtex] " + " ".join(bib_cmd))
        subprocess.run(bib_cmd, cwd=paper_dir, check=True)
    for idx in range(max(runs, 1)):
        run_idx += 1
        print(f"[compile {run_idx}/{total_runs}] " + " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=paper_dir, check=True)
        except subprocess.CalledProcessError:
            if run_idx == 1:
                for path in aux_like:
                    remove_if_exists(path)
                subprocess.run(cmd, cwd=paper_dir, check=True)
            else:
                raise


def main() -> None:
    args = build_parser().parse_args()
    paper_dir = Path(args.paper_dir)
    tex_file = args.tex_file
    if not (paper_dir / tex_file).exists():
        raise FileNotFoundError(f"Could not find {(paper_dir / tex_file)!s}")
    if args.with_supplement and (paper_dir / "supplement.tex").exists():
        # Two-pass cross-document compile: each document's xr setup needs the
        # other's .aux file to resolve \ref{} across the main/supplement
        # split, so compile both, then re-compile both so the cross-references
        # converge.
        _run_compile_pipeline(paper_dir, "supplement.tex", args.runs, args.skip_bibtex)
        _run_compile_pipeline(paper_dir, tex_file, args.runs, args.skip_bibtex)
        _run_compile_pipeline(paper_dir, "supplement.tex", 1, True)
        _run_compile_pipeline(paper_dir, tex_file, 1, True)
        print(
            f"Compiled {(paper_dir / 'supplement.tex')!s} and "
            f"{(paper_dir / tex_file)!s} successfully (cross-document refs converged)."
        )
        return
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_file]
    stem = Path(tex_file).stem
    aux_like = [paper_dir / f"{stem}.aux", paper_dir / f"{stem}.out"]
    first_pass_count = 1 if not args.skip_bibtex and (paper_dir / "references.bib").exists() else 0
    total_runs = first_pass_count + max(args.runs, 1)
    run_idx = 0
    if first_pass_count:
        run_idx += 1
        print(f"[compile {run_idx}/{total_runs}] " + " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=paper_dir, check=True)
        except subprocess.CalledProcessError:
            for path in aux_like:
                remove_if_exists(path)
            subprocess.run(cmd, cwd=paper_dir, check=True)
        bib_cmd = ["bibtex", stem]
        print("[bibtex] " + " ".join(bib_cmd))
        subprocess.run(bib_cmd, cwd=paper_dir, check=True)
    for idx in range(max(args.runs, 1)):
        run_idx += 1
        print(f"[compile {run_idx}/{total_runs}] " + " ".join(cmd))
        try:
            subprocess.run(cmd, cwd=paper_dir, check=True)
        except subprocess.CalledProcessError:
            if run_idx == 1:
                for path in aux_like:
                    remove_if_exists(path)
                subprocess.run(cmd, cwd=paper_dir, check=True)
            else:
                raise
    print(f"Compiled {(paper_dir / tex_file)!s} successfully.")


if __name__ == "__main__":
    main()
