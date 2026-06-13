"""Small, dependency-free binary-classification statistics.

Used to report DBAR (and any other small-sample binary indicator) with a
no-information / majority-class baseline, Wilson-score confidence intervals,
and an explicit power / sample-size statement, so a headline accuracy is never
presented without its incremental value over a trivial classifier.
"""

from __future__ import annotations

import math

# Two-sided normal quantiles for the common confidence levels.
_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


def _z(conf: float) -> float:
    return _Z.get(round(conf, 2), 1.959963984540054)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k/n``.

    Returns ``(lo, hi)`` clamped to ``[0, 1]``; an empty sample yields
    ``(0.0, 1.0)`` (maximally uninformative, never a spurious point estimate).
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _z(conf)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def required_n_one_proportion(
    p0: float, p1: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Approx. one-sided one-proportion test sample size to detect ``p1>p0``.

    Normal approximation; rounded up. Returns a large sentinel when the effect
    is degenerate (``p1 <= p0``), i.e. "not detectable at any feasible N".
    """
    if p1 <= p0:
        return 10**9
    z_a = _z(1.0 - 2.0 * alpha)  # one-sided alpha via two-sided table entry
    z_b = _z(1.0 - 2.0 * (1.0 - power))
    num = z_a * math.sqrt(p0 * (1.0 - p0)) + z_b * math.sqrt(p1 * (1.0 - p1))
    return int(math.ceil((num / (p1 - p0)) ** 2))


def binary_classification_report(
    tp: int, tn: int, fp: int, fn: int, conf: float = 0.95
) -> dict:
    """Confusion-derived metrics with Wilson CIs and a majority baseline.

    The ``no_information`` block reports the always-predict-the-majority-class
    accuracy (the trivial classifier) and the indicator's *incremental* value
    over it, so a headline accuracy is never reported in isolation.
    """
    n = tp + tn + fp + fn
    n_pos = tp + fn
    n_neg = tn + fp
    n_correct = tp + tn
    acc = n_correct / n if n else float("nan")
    sens = tp / n_pos if n_pos else float("nan")
    spec = tn / n_neg if n_neg else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    majority = max(n_pos, n_neg) / n if n else float("nan")
    majority_class = (
        "no_fire (negative)" if n_neg >= n_pos else "fire (positive)"
    )
    acc_lo, acc_hi = wilson_ci(n_correct, n, conf)
    sens_lo, sens_hi = wilson_ci(tp, n_pos, conf)
    spec_lo, spec_hi = wilson_ci(tn, n_neg, conf)
    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "confidence": conf,
        "accuracy": round(acc, 4),
        "accuracy_ci": [round(acc_lo, 4), round(acc_hi, 4)],
        "sensitivity": round(sens, 4) if n_pos else None,
        "sensitivity_ci": [round(sens_lo, 4), round(sens_hi, 4)]
        if n_pos
        else None,
        "specificity": round(spec, 4) if n_neg else None,
        "specificity_ci": [round(spec_lo, 4), round(spec_hi, 4)]
        if n_neg
        else None,
        "precision": round(ppv, 4) if (tp + fp) else None,
        "no_information": {
            "majority_class": majority_class,
            "majority_class_accuracy": round(majority, 4),
            "accuracy_minus_majority": round(acc - majority, 4),
            "n_correct_above_majority": int(
                round(n_correct - majority * n)
            ),
            "beats_majority": bool(acc > majority),
        },
        "power": {
            "n_for_80pct_power_vs_majority_alpha05": (
                required_n_one_proportion(majority, acc, 0.05, 0.80)
            ),
            "positive_class_underpowered": bool(n_pos < 20),
            "note": (
                "Sensitivity rests on the positive class only; with a small "
                "positive class its Wilson interval is wide and no inferential "
                "claim is made."
            ),
        },
    }
