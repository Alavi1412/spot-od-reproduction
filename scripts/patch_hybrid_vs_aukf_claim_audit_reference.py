#!/usr/bin/env python
"""Patch manuscript/table text to make Hybrid-vs-AUKF evidence boundary auditable."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "main.tex"
SIGNIFICANCE = ROOT / "paper" / "tables" / "significance.tex"

sig_text = SIGNIFICANCE.read_text(encoding="utf-8")
old_caption = """  \\caption{Paired statistical comparison for hybrid estimators on trajectory-wise position RMSE. The AUKF rows are explicit tuned-adaptive-filter comparator tests, not secondary diagnostics; all rows use the same one-sided Wilcoxon convention (alternative: candidate better) and paired-bootstrap mean-gain confidence interval.}"""
new_caption = """  \\caption{Paired statistical comparison for hybrid estimators on trajectory-wise position RMSE. The AUKF rows are explicit tuned-adaptive-filter comparator tests, not secondary diagnostics; all rows use the same one-sided Wilcoxon convention (alternative: candidate better) and paired-bootstrap mean-gain confidence interval. Mean gain is baseline minus candidate, so negative AUKF-row values are direct evidence against candidate superiority.}"""
if old_caption not in sig_text and new_caption not in sig_text:
    raise SystemExit("significance caption target not found")
if old_caption in sig_text:
    sig_text = sig_text.replace(old_caption, new_caption)
    SIGNIFICANCE.write_text(sig_text, encoding="utf-8")

main_text = MAIN.read_text(encoding="utf-8")
old_claim = """The major claims in this revision are tied to explicit repository artifacts. The claim that AUKF is the strongest displayed aggregate method is supported by \\artifactpath{paper/tables/main_results.tex} and is limited to the displayed nominal/stress RMSE rows. The claim that RGR-GF and IDP-RGR-GF have statistically supported stress-regime gains is supported by \\artifactpath{paper/tables/significance.tex} and is limited to comparisons against fixed-noise UKF, not AUKF. The claim that graph-specific superiority is not isolated is supported by \\artifactpath{paper/tables/ablation.tex}. The active figures provide temporal, distributional, visibility-bucket, and calibration diagnostics without changing these aggregate and statistical conclusions. Protocol details are sourced from \\artifactpath{configs/experiment.yaml}, \\artifactpath{results/data/dataset_manifest.json}, \\artifactpath{results/metrics_summary.json}, and implementation files; they are included to make the method auditable, not to expand the quantitative claim boundary."""
new_claim = """The major claims in this revision are tied to explicit repository artifacts. The claim that AUKF is the strongest displayed aggregate method is supported by \\artifactpath{paper/tables/main_results.tex} and is limited to the displayed nominal/stress RMSE rows. The claim that RGR-GF and IDP-RGR-GF have statistically supported stress-regime gains is supported by \\artifactpath{paper/tables/significance.tex} and is limited to comparisons against fixed-noise UKF, not AUKF. The companion audit in \\artifactpath{results/hybrid_vs_aukf_statistical_audit.md} recomputes the AUKF rows from \\artifactpath{results/trajectory_errors.csv}; it is negative evidence for hybrid superiority over tuned AUKF, not an additional positive result. The claim that graph-specific superiority is not isolated is supported by \\artifactpath{paper/tables/ablation.tex}. The active figures provide temporal, distributional, visibility-bucket, and calibration diagnostics without changing these aggregate and statistical conclusions. Protocol details are sourced from \\artifactpath{configs/experiment.yaml}, \\artifactpath{results/data/dataset_manifest.json}, \\artifactpath{results/metrics_summary.json}, and implementation files; they are included to make the method auditable, not to expand the quantitative claim boundary."""
if old_claim not in main_text and new_claim not in main_text:
    raise SystemExit("main claim-audit paragraph target not found")
if old_claim in main_text:
    main_text = main_text.replace(old_claim, new_claim)
    MAIN.write_text(main_text, encoding="utf-8")

print("patched Hybrid-vs-AUKF claim-audit references")
