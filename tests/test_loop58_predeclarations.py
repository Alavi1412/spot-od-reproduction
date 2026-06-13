"""Schema checks for the loop58 predeclared rule artefacts.

This test suite verifies the static shape of the loop58 predeclared rule
artefact for the KalmanNet SPOT-OD training-budget adequacy diagnostic.
The schema checks here do not exercise the underlying run; they verify
the predeclaration file commits the disjoint seeds inherited from the
loop57 transposition rule, the predeclared optimiser-step snapshot
schedule, and the inherited decision predicate so the audit boundary is
inspectable from the artefact alone.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str) -> dict:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def test_kalmannet_spot_od_budget_adequacy_loop58_schema() -> None:
    """The loop58 budget-adequacy rule commits the predeclared snapshot
    schedule beyond the loop57 single-budget point, inherits the loop57
    train/validation/test data layout byte-identical, and inherits the
    loop57 decision predicate without retuning.
    """
    art = _load(
        "release/predeclarations/kalmannet_spot_od_budget_adequacy_loop58.json"
    )
    assert art["predeclared_on_utc"] == "2026-05-20"
    loop57 = _load(
        "release/predeclarations/kalmannet_spot_od_faithful_transposition_loop57.json"
    )

    # Data layout must inherit the loop57 transposition rule's seeds so the
    # diagnostic operates on the byte-identical train/validation/test draws
    # that the loop57 result was reported on.
    data = art["data"]
    loop57_data = loop57["data"]
    for split in ("train_split", "validation_split", "test_split"):
        assert (
            data[split]["rng_seed"]
            == loop57_data[split]["rng_seed"]
        ), f"loop58 {split} seed must match loop57"
    # Train and test trajectory counts inherit the loop57 rule. The
    # validation count inherits the loop57 *actually-run* validation draw
    # (24 trajectories), which the prior loop57 result artifact records.
    for split in ("train_split", "test_split"):
        assert (
            data[split]["n_trajectories"]
            == loop57_data[split]["n_trajectories"]
        ), f"loop58 {split} size must match loop57"

    # The predeclared snapshot schedule must include the loop57 single-budget
    # point (300 steps) as a snapshot so the diagnostic reproduces the prior
    # outcome at the same milestone, and must include at least one snapshot
    # that materially extends beyond the loop57 budget.
    budget = art["predeclared_optimisation_budget"]
    schedule = list(budget["predeclared_snapshot_schedule"])
    assert int(loop57["predeclared_optimisation_budget"]["n_steps"]) in schedule
    assert max(schedule) > int(loop57["predeclared_optimisation_budget"]["n_steps"])
    assert int(budget["n_steps_total"]) >= max(schedule)

    # Optimiser batch size matches the loop57 rule (only the budget is
    # varied). The loop57 rule records the learning rate and weight decay
    # textually inside the design-changes block; the loop58 rule records
    # them numerically in the optimisation-budget block so the diagnostic
    # is reproducible without parsing the prose.
    assert budget["n_batch"] == loop57["predeclared_optimisation_budget"]["n_batch"]
    assert float(budget["lr"]) == 1.0e-4
    assert float(budget["wd"]) == 1.0e-4
    assert int(budget["in_mult_KNet"]) == 5
    assert int(budget["out_mult_KNet"]) == 4

    # Decision predicate inherits the loop57 three-condition rule.
    dp = art["decision_predicate"].lower()
    for needle in (
        "strictly lowest mean",
        "strictly below zero",
        "3% practical-significance floor",
    ):
        assert needle in dp

    # Honest-negative interpretation: the rule must require an honest
    # report regardless of the sign of the outcome, no retuning allowed.
    hn = art["honest_negative"].lower()
    assert "no rule retuning of the loop57 transposition is allowed" in hn

    # Non-paper-facing artefact paths must be recorded so the rule is
    # inspectable from the artefact alone.
    art_paths = art["non_paper_artifacts"]
    assert art_paths["test_harness"].endswith("budget_adequacy.py")
    assert art_paths["result_directory"].endswith("budget_adequacy_loop58")
