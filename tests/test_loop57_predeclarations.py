"""Schema checks for the loop57 predeclared rule artefacts.

This test suite verifies the static shape of the two loop57 predeclared
rule artefacts: the long-arc n=64 extension and the faithful KalmanNet
SPOT-OD transposition. The schema checks here do not exercise the
underlying runs; they verify the predeclaration files commit the
required disjoint seeds, populations, and decision predicates so the
audit boundary is inspectable from the artefacts alone.
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str) -> dict:
    return json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))


def test_long_arc_n64_extension_loop57_schema() -> None:
    """The loop57 long-arc n=64 extension inherits the loop47 rule
    unchanged in scope and decision predicate, raises the test population
    to n=64, and records the loop47 36-trajectory prefix explicitly.
    """
    art = _load(
        "release/predeclarations/long_arc_hifi_n64_extension_loop57.json"
    )
    assert art["predeclared_on_utc"] == "2026-05-20"
    assert art["inherits_from"].endswith("long_arc_hifi_rule_loop47.json")
    ep = art["evaluation_protocol"]
    assert ep["n_trajectories_planned"] == 64
    assert ep["n_trajectories_loop47_prefix"] == 36
    assert ep["n_trajectories_loop57_extension"] == 28
    # The test seed is inherited from loop47 so the first 36 trajectories
    # are byte-identical to the loop47 result; the loop47 rule and the
    # loop57 extension must therefore agree on the test seed.
    loop47 = _load("release/predeclarations/long_arc_hifi_rule_loop47.json")
    assert ep["test_seed_inherited_from_loop47"] == loop47["evaluation_protocol"]["test_seed"]
    # The candidate hyperparameters used on the extension are the loop47
    # validation-selection result (no rule retuning).
    sel = ep["candidate_hyperparameters_inherited_from_loop47_validation_selection"]
    assert sel["selected_grid_point_label"] == "L4"
    assert sel["init_drag_scale_std"] == 1.0
    assert sel["drag_scale_sigma_ss"] == 1.0
    assert sel["drag_scale_tau_s"] == 3600.0
    # The decision predicate is inherited from the loop47 rule.
    assert "decision_predicate_inherited_from_loop47" in art


def test_kalmannet_spot_od_faithful_transposition_loop57_schema() -> None:
    """The loop57 faithful KalmanNet SPOT-OD transposition rule commits
    the four predeclared design changes jointly, disjoint train/val/test
    seeds, and the predeclared decision predicate (strictly lowest mean,
    strictly negative CI, gap above the 3% practical-significance floor).
    """
    art = _load(
        "release/predeclarations/kalmannet_spot_od_faithful_transposition_loop57.json"
    )
    assert art["predeclared_on_utc"] == "2026-05-20"
    dc = art["predeclared_design_changes"]
    # The four design choices must each appear as a distinct predeclared key.
    for key in (
        "orbital_scale_normalization",
        "sequence_length_and_curriculum_rematching",
        "sparse_observation_architectural_adaptation",
        "learning_rate_and_budget_recalibration",
    ):
        assert key in dc
        assert isinstance(dc[key], str) and len(dc[key]) > 40
    data = art["data"]
    seeds = {
        data["train_split"]["rng_seed"],
        data["validation_split"]["rng_seed"],
        data["test_split"]["rng_seed"],
    }
    # Disjoint seeds for train, validation, and held-out test.
    assert len(seeds) == 3
    # The test split must be disjoint from every prior in-manuscript test
    # seed; spot-check against a handful of canonical seeds.
    test_seed = data["test_split"]["rng_seed"]
    assert test_seed not in {42, 770000, 20260645, 20260545, 20260847, 770566}
    budget = art["predeclared_optimisation_budget"]
    assert int(budget["n_steps"]) >= 100
    assert int(budget["n_batch"]) >= 8
    assert "validation-best step" in budget["model_selection_rule"].lower()
    assert "KalmanNet-SPOT-OD" in art["comparators"]
    # The decision predicate must require all three conditions.
    dp = art["decision_predicate"].lower()
    for needle in (
        "strictly the lowest",
        "strictly negative",
        "3% practical-significance floor",
    ):
        assert needle in dp
