"""Tests for the dependency-free binary-classification statistics helper.

These pin the no-information / majority-class baseline arithmetic, the Wilson
interval behaviour, and the one-proportion power sentinel, since DBAR
reporting now depends on them.
"""

from __future__ import annotations

from gnn_state_estimation.utils.classification_stats import (
    binary_classification_report,
    required_n_one_proportion,
    wilson_ci,
)


def test_wilson_ci_bounds_and_degenerate() -> None:
    lo, hi = wilson_ci(5, 10, 0.95)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # Empty sample is maximally uninformative, never a spurious point est.
    assert wilson_ci(0, 0) == (0.0, 1.0)
    # All-correct still yields an interval strictly inside [0, 1].
    lo2, hi2 = wilson_ci(12, 12, 0.95)
    assert lo2 > 0.5 and hi2 <= 1.0


def test_dbar_insim_majority_baseline_arithmetic() -> None:
    """The committed in-sim confusion 10/42/6/2 must reproduce 86.7% and an
    80% always-no-fire majority baseline (the reviewer-mandated check)."""
    rep = binary_classification_report(tp=10, tn=42, fp=6, fn=2)
    assert rep["n"] == 60
    assert rep["n_positive"] == 12 and rep["n_negative"] == 48
    assert abs(rep["accuracy"] - 0.8667) < 1e-3
    ni = rep["no_information"]
    assert ni["majority_class"].startswith("no_fire")
    assert abs(ni["majority_class_accuracy"] - 0.80) < 1e-9
    assert abs(ni["accuracy_minus_majority"] - 0.0667) < 1e-3
    assert ni["n_correct_above_majority"] == 4
    assert ni["beats_majority"] is True
    # Accuracy Wilson interval must contain the 80% trivial baseline.
    lo, hi = rep["accuracy_ci"]
    assert lo < 0.80 < hi
    # Small positive class flagged underpowered.
    assert rep["power"]["positive_class_underpowered"] is True


def test_real_sp3_dbar_below_trivial_baseline() -> None:
    """Real-SP3 confusion 0/6/2/2 -> 60%, below the 80% majority baseline."""
    rep = binary_classification_report(tp=0, tn=6, fp=2, fn=2)
    assert abs(rep["accuracy"] - 0.60) < 1e-9
    ni = rep["no_information"]
    assert abs(ni["majority_class_accuracy"] - 0.80) < 1e-9
    assert ni["beats_majority"] is False
    assert ni["accuracy_minus_majority"] < 0.0


def test_required_n_degenerate_and_increasing() -> None:
    # No detectable effect when p1 <= p0.
    assert required_n_one_proportion(0.80, 0.80) >= 10**9
    assert required_n_one_proportion(0.80, 0.70) >= 10**9
    # Detecting 86.7% vs 80% needs a large (hundreds) sample.
    n = required_n_one_proportion(0.80, 0.8667, 0.05, 0.80)
    assert 100 < n < 10**6
