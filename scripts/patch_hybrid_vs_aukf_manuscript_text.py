#!/usr/bin/env python
"""Patch paper/main.tex with explicit Hybrid-vs-AUKF claim-boundary text."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "main.tex"

old = """Table~\\ref{tab:significance} supports only bounded significance claims. In the stress scenario, RGR-GF improves over UKF by a mean 487.19 m with 95\\% CI [145.50, 997.24] and Wilcoxon $p=0.0001823$. IDP-RGR-GF improves over UKF by a mean 454.87 m with 95\\% CI [143.81, 913.99] and Wilcoxon $p=0.001593$. These are stress-regime gains against fixed-noise UKF.

The same table does not support a claim that the hybrids beat AUKF. RGR-GF versus AUKF in stress has mean gain $-16.28$ m and $p=0.9095$; IDP-RGR-GF versus AUKF has mean gain $-48.60$ m and $p=0.9248$. Nominal comparisons are also not significant in favor of the hybrids. The safe conclusion is that hybrid residual/filter variants improve over UKF under stress conditions while AUKF remains the strongest displayed aggregate comparator.
"""

new = """Table~\\ref{tab:significance} supports only bounded significance claims. In the stress scenario, RGR-GF improves over UKF by a mean 487.19 m with 95\\% CI [145.50, 997.24] and Wilcoxon $p=0.0001823$. IDP-RGR-GF improves over UKF by a mean 454.87 m with 95\\% CI [143.81, 913.99] and Wilcoxon $p=0.001593$. These are stress-regime gains against fixed-noise UKF.

The AUKF rows are the decisive guardrail against overclaiming: they use the same paired trajectory-wise Wilcoxon/bootstrap convention as the UKF rows, but they test the hybrids against the tuned adaptive filter rather than the fixed-noise UKF. The table does not support a claim that the hybrids beat AUKF. RGR-GF versus AUKF in stress has mean gain $-16.28$ m and $p=0.9095$; IDP-RGR-GF versus AUKF has mean gain $-48.60$ m and $p=0.9248$. Nominal comparisons are also not significant in favor of the hybrids. The safe conclusion is that hybrid residual/filter variants improve over UKF under stress conditions while AUKF remains the strongest displayed aggregate comparator.
"""

text = MAIN.read_text(encoding="utf-8")
if old not in text:
    raise SystemExit("target manuscript paragraph not found; no patch applied")
MAIN.write_text(text.replace(old, new), encoding="utf-8")
print("patched", MAIN.relative_to(ROOT))
