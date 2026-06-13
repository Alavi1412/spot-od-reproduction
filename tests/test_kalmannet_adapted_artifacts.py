"""Schema/regression tests for the adapted KalmanNet SPOT-OD artifact.

These cover the non-paper-facing ``results/kalmannet_adapted`` JSON produced by
``scripts/run_kalmannet_adapted_spot_od.py``. The artifact is skipped when not
generated in the current environment (the training run is GPU/long-running),
but when present it must conform to the schema so the honest win/lose outcome
cannot be silently misreported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = REPO_ROOT / "results" / "kalmannet_adapted"


def _load_primary() -> dict | None:
    candidates = sorted(RESULT_DIR.glob("kalmannet_adapted_loop163*.json"))
    if not candidates:
        return None
    # Prefer the headline tag if present.
    headline = RESULT_DIR / "kalmannet_adapted_loop163.json"
    path = headline if headline.exists() else candidates[0]
    return json.loads(path.read_text())


def test_adapted_artifact_schema_when_present() -> None:
    payload = _load_primary()
    if payload is None:
        pytest.skip("Adapted KalmanNet artifact not generated in this environment.")
    assert payload["scenario"] == "kalmannet_adapted_spot_od"
    assert payload["schema_version"] == "kalmannet_adapted_spot_od_v1"
    cfg = payload["config"]
    assert cfg["m"] == 6
    assert cfg["n"] in (32, 40)
    assert cfg["eval_start_step"] == 11
    # The decisive adaptation (differentiable BPTT) must be recorded.
    assert "differentiable_torch_f_h_bptt" in payload["adaptations_enabled"]
    # Classical baselines present with the standard four filters.
    cb = payload["classical_baselines"]
    assert cb, "no classical baseline variant recorded"
    for variant, block in cb.items():
        means = block["observed_step_rmse_mean_m"]
        assert {"EKF", "UKF", "AUKF", "PUKF"}.issubset(set(means))
    # A pure-propagation (no-skill) reference must be recorded for context.
    assert payload["pure_propagation_reference_mean_m"] > 0.0
    # The decision block must classify the outcome with the fixed criterion.
    dec = payload["decision"]
    assert dec["outcome_class"] in {
        "win_above_floor",
        "win_below_floor",
        "loss_ci_strictly_positive",
        "inconclusive_ci_contains_zero",
    }
    assert isinstance(dec["is_win"], bool)
    # Internal consistency: is_win implies CI strictly negative and floor met.
    if dec["is_win"]:
        assert dec["ci_strictly_negative"] and dec["floor_exceeded"]


def test_adapted_vendor_commit_pinned_when_present() -> None:
    payload = _load_primary()
    if payload is None:
        pytest.skip("Adapted KalmanNet artifact not generated in this environment.")
    commit_file = REPO_ROOT / "external" / "third_party" / "KalmanNet_TSP_COMMIT"
    pinned = commit_file.read_text().strip() if commit_file.exists() else ""
    # The adapted baseline keeps the upstream gain network at the pinned commit.
    assert payload["vendor_commit"] == pinned, (payload["vendor_commit"], pinned)
