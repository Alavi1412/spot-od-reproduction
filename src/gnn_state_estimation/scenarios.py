"""Scenario simulation-config resolution shared by generation and baselines.

A benchmark scenario's *truth* simulation config is ``base simulation`` deep-
merged with the scenario ``overrides``. When a scenario additionally declares an
``estimator_overrides`` key, the recursive filters, prior banks, evaluation
baselines, batch WLS, and RFIS instead see ``base simulation`` deep-merged with
``estimator_overrides`` -- a deliberate dynamics/measurement model mismatch
against the truth that generated the data.

When a scenario does **not** declare ``estimator_overrides`` the estimator
config is, by construction, the exact same dict the truth config produces, so
legacy scenarios keep bit-for-bit identical behavior.

Public-station handling (``public_catalog_replay`` /
``public_observation_replay``) is applied to both the truth and the estimator
config. Those selectors are deterministic functions of the scenario snapshot
paths and filters and do not depend on the dynamics/measurement overrides, so
the resolved station bank is identical on both sides -- estimators always see
the same physical stations the measurements were synthesized at.
"""

from __future__ import annotations

import copy
from typing import Any

from .observation_replay import apply_public_observation_station_bank
from .public_data import apply_public_station_selection

__all__ = [
    "deep_update",
    "scenario_kind",
    "has_estimator_overrides",
    "truth_sim_config",
    "estimator_sim_config",
]


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def scenario_kind(scenario_cfg: dict[str, Any]) -> str:
    return str(scenario_cfg.get("kind", "synthetic"))


def has_estimator_overrides(scenario_cfg: dict[str, Any]) -> bool:
    """Whether the scenario opts in to a truth/estimator model mismatch.

    Opt-in is by *presence* of the ``estimator_overrides`` key, not its
    truthiness: an explicit empty mapping means "estimators use the nominal
    base simulation" (the typical force-model-mismatch configuration), which is
    still meaningfully different from the absent case (estimator == truth).
    """
    return "estimator_overrides" in scenario_cfg


def _apply_station_selection(
    sim_cfg: dict[str, Any], scenario_cfg: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    kind = scenario_kind(scenario_cfg)
    if kind == "public_catalog_replay":
        return apply_public_station_selection(sim_cfg, scenario_cfg)
    if kind == "public_observation_replay":
        return apply_public_observation_station_bank(sim_cfg, scenario_cfg)
    return sim_cfg, None


def truth_sim_config(
    base_sim: dict[str, Any],
    scenario_cfg: dict[str, Any],
    *,
    with_station_meta: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
    """Simulation config used to synthesize the scenario's ground truth."""
    sim_cfg = deep_update(base_sim, scenario_cfg.get("overrides", {}) or {})
    sim_cfg, meta = _apply_station_selection(sim_cfg, scenario_cfg)
    if with_station_meta:
        return sim_cfg, meta
    return sim_cfg


def estimator_sim_config(
    base_sim: dict[str, Any],
    scenario_cfg: dict[str, Any],
    *,
    with_station_meta: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
    """Simulation config seen by recursive filters / WLS / RFIS / evaluation.

    Equals :func:`truth_sim_config` unless the scenario declares
    ``estimator_overrides``, in which case the estimator side is
    ``base simulation`` deep-merged with ``estimator_overrides`` only (the
    truth ``overrides`` are intentionally *not* applied to the estimator).
    """
    if has_estimator_overrides(scenario_cfg):
        overrides = scenario_cfg.get("estimator_overrides") or {}
    else:
        overrides = scenario_cfg.get("overrides", {}) or {}
    sim_cfg = deep_update(base_sim, overrides)
    sim_cfg, meta = _apply_station_selection(sim_cfg, scenario_cfg)
    if with_station_meta:
        return sim_cfg, meta
    return sim_cfg
