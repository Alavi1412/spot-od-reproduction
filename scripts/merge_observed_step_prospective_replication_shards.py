"""Merge completed observed-step prospective replication scenario shards.

The simulation builder can materialize one scenario at a time with
``--only-scenario-index``. This utility combines those completed shard JSON
artifacts into the canonical full artifact only after validating that the
shards are compatible and complete.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_OUT_PATH = (
    ROOT
    / "results"
    / "observed_step_prospective_replication"
    / "observed_step_prospective_replication.json"
)

SCHEMA_VERSION = "observed_step_prospective_replication_v1"
EXPECTED_STATUS = "completed"
CLASSICAL_METHODS = frozenset(("EKF", "UKF", "AUKF"))
ALLOWED_BEST_METHODS = CLASSICAL_METHODS | frozenset(("RGR-GF",))
VARIABLE_TOP_LEVEL_KEYS = frozenset(("num_scenarios", "scenarios", "summary"))
EXPECTED_CANONICAL_SCENARIO_INDICES = (0, 1, 2)
EXPECTED_CANONICAL_FROZEN_RULE_VALUES = {
    "realization_base_seed": 880000,
    "num_realizations_per_scenario": 32,
    "trajectories_per_realization": 24,
    "bootstrap_samples": 5000,
    "primary_metric": "observed_step_position_rmse_m",
    "reference_metric": "all_step_position_rmse_m",
    "frozen_before_evaluation": True,
    "not_external_preregistration": True,
    "inference_only": True,
    "no_selection_tuning_or_retraining": True,
}


class ValidationError(ValueError):
    """Raised when shards cannot safely be merged."""


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _is_int_not_bool(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _load_shard(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValidationError(
            f"{_display_path(path)}: could not read shard: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(
            f"{_display_path(path)}: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not _is_mapping(payload):
        raise ValidationError(f"{_display_path(path)}: shard root must be a JSON object")
    return payload


def _require_key(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"{context}: missing required key {key!r}")
    return mapping[key]


def _validate_shard_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    context = _display_path(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(
            f"{context}: schema_version must be {SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    if payload.get("status") != EXPECTED_STATUS:
        raise ValidationError(
            f"{context}: status must be {EXPECTED_STATUS!r}, got {payload.get('status')!r}"
        )

    scenarios = _require_key(payload, "scenarios", context)
    if not isinstance(scenarios, list):
        raise ValidationError(f"{context}: scenarios must be a list")
    if len(scenarios) != 1:
        raise ValidationError(
            f"{context}: expected exactly one scenario, got {len(scenarios)}"
        )

    num_scenarios = payload.get("num_scenarios")
    if num_scenarios is not None and num_scenarios != 1:
        raise ValidationError(
            f"{context}: num_scenarios must be 1 for a shard, got {num_scenarios!r}"
        )

    frozen_rule = _require_key(payload, "frozen_rule", context)
    if not _is_mapping(frozen_rule):
        raise ValidationError(f"{context}: frozen_rule must be a JSON object")
    for key in (
        "num_realizations_per_scenario",
        "trajectories_per_realization",
        "bootstrap_samples",
    ):
        _require_key(frozen_rule, key, f"{context}: frozen_rule")

    scenario = scenarios[0]
    if not _is_mapping(scenario):
        raise ValidationError(f"{context}: scenario entry must be a JSON object")

    scenario_index = _require_key(scenario, "scenario_index", context)
    if not _is_int_not_bool(scenario_index) or scenario_index < 0:
        raise ValidationError(
            f"{context}: scenario_index must be a non-negative integer, got {scenario_index!r}"
        )

    learned_positive = _require_key(
        scenario, "learned_positive_under_frozen_rule", context
    )
    if not isinstance(learned_positive, bool):
        raise ValidationError(
            f"{context}: learned_positive_under_frozen_rule must be boolean, "
            f"got {learned_positive!r}"
        )

    best_method = _require_key(scenario, "best_method_primary", context)
    if best_method not in ALLOWED_BEST_METHODS:
        raise ValidationError(
            f"{context}: best_method_primary must be one of "
            f"{sorted(ALLOWED_BEST_METHODS)}, got {best_method!r}"
        )

    best_classical = scenario.get("best_classical_primary")
    if best_classical is not None and best_classical not in CLASSICAL_METHODS:
        raise ValidationError(
            f"{context}: best_classical_primary must be one of "
            f"{sorted(CLASSICAL_METHODS)}, got {best_classical!r}"
        )

    for scenario_key, rule_key in (
        ("n_realizations", "num_realizations_per_scenario"),
        ("trajectories_per_realization", "trajectories_per_realization"),
    ):
        actual = _require_key(scenario, scenario_key, context)
        expected = frozen_rule[rule_key]
        if actual != expected:
            raise ValidationError(
                f"{context}: scenario {scenario_key!r} ({actual!r}) does not match "
                f"frozen_rule {rule_key!r} ({expected!r})"
            )

    return scenario


def _validate_matching_metadata(
    loaded: Sequence[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    ref_path, ref_payload = loaded[0]
    ref_keys = set(ref_payload) - VARIABLE_TOP_LEVEL_KEYS
    ref_metadata = {key: ref_payload[key] for key in ref_keys}

    for path, payload in loaded[1:]:
        metadata_keys = set(payload) - VARIABLE_TOP_LEVEL_KEYS
        if metadata_keys != ref_keys:
            missing = sorted(ref_keys - metadata_keys)
            extra = sorted(metadata_keys - ref_keys)
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if extra:
                detail.append(f"extra {extra}")
            raise ValidationError(
                f"{_display_path(path)}: top-level metadata keys do not match "
                f"{_display_path(ref_path)} ({'; '.join(detail)})"
            )
        for key in sorted(ref_keys):
            if payload[key] != ref_metadata[key]:
                raise ValidationError(
                    f"{_display_path(path)}: top-level metadata key {key!r} does not "
                    f"match {_display_path(ref_path)}"
                )

    return ref_metadata


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _validate_canonical_requirements(
    out_path: Path,
    scenarios: Sequence[dict[str, Any]],
) -> None:
    if not _same_path(out_path, CANONICAL_OUT_PATH):
        return

    scenario_indices = tuple(sorted(row["scenario_index"] for row in scenarios))
    if scenario_indices != EXPECTED_CANONICAL_SCENARIO_INDICES:
        raise ValidationError(
            "canonical output requires exactly scenario_index values "
            f"{list(EXPECTED_CANONICAL_SCENARIO_INDICES)}, got {list(scenario_indices)}"
        )


def _validate_canonical_frozen_rule(out_path: Path, frozen_rule: dict[str, Any]) -> None:
    if not _same_path(out_path, CANONICAL_OUT_PATH):
        return

    for key, expected in EXPECTED_CANONICAL_FROZEN_RULE_VALUES.items():
        actual = frozen_rule.get(key)
        if actual != expected:
            raise ValidationError(
                f"canonical output requires frozen_rule.{key} == {expected!r}, "
                f"got {actual!r}"
            )


def _build_summary(
    scenarios: Sequence[dict[str, Any]],
    frozen_rule: dict[str, Any],
) -> dict[str, Any]:
    learned_positive_count = sum(
        1 for row in scenarios if row["learned_positive_under_frozen_rule"]
    )
    classical_best_count = sum(
        1 for row in scenarios if row["best_method_primary"] in CLASSICAL_METHODS
    )
    return {
        "n_scenarios": len(scenarios),
        "num_realizations_per_scenario": frozen_rule["num_realizations_per_scenario"],
        "trajectories_per_realization": frozen_rule["trajectories_per_realization"],
        "scenarios_with_learned_positive_under_frozen_rule": learned_positive_count,
        "scenarios_with_classical_best_on_primary": classical_best_count,
        "verdict": (
            "larger independent observed-step replication under the frozen "
            "rule: no learned positive under the decision predicate"
            if learned_positive_count == 0
            else "learned positive observed under the frozen decision predicate"
        ),
    }


def build_merged_artifact(
    shard_paths: Sequence[Path | str],
    out_path: Path | str,
) -> dict[str, Any]:
    """Validate shard files and return the merged artifact without writing it."""
    paths = [Path(path) for path in shard_paths]
    if not paths:
        raise ValidationError("at least one --shard path is required")

    loaded = [(path, _load_shard(path)) for path in paths]
    scenarios = [_validate_shard_payload(path, payload) for path, payload in loaded]

    seen_indices: set[int] = set()
    for path, scenario in zip(paths, scenarios):
        idx = scenario["scenario_index"]
        if idx in seen_indices:
            raise ValidationError(
                f"{_display_path(path)}: duplicate scenario_index {idx}"
            )
        seen_indices.add(idx)

    metadata = _validate_matching_metadata(loaded)
    sorted_scenarios = sorted(
        (copy.deepcopy(row) for row in scenarios),
        key=lambda row: row["scenario_index"],
    )
    out = Path(out_path)
    frozen_rule = metadata["frozen_rule"]
    _validate_canonical_requirements(out, sorted_scenarios)
    _validate_canonical_frozen_rule(out, frozen_rule)

    first_payload = loaded[0][1]
    result = {
        key: copy.deepcopy(first_payload[key])
        for key in first_payload
        if key not in VARIABLE_TOP_LEVEL_KEYS
    }
    result["num_scenarios"] = len(sorted_scenarios)
    result["scenarios"] = sorted_scenarios
    result["summary"] = _build_summary(sorted_scenarios, frozen_rule)
    return result


def _write_json_atomic(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def merge_shards(
    shard_paths: Sequence[Path | str],
    out_path: Path | str,
) -> dict[str, Any]:
    """Validate, merge, and write shard files."""
    out = Path(out_path)
    payload = build_merged_artifact(shard_paths, out)
    _write_json_atomic(out, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        action="append",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more one-scenario shard JSON paths. May be repeated.",
    )
    parser.add_argument("--out-path", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_paths = [Path(path) for group in args.shard for path in group]
    try:
        payload = merge_shards(shard_paths, args.out_path)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload["summary"], indent=2))
    print(
        f"wrote {args.out_path} "
        f"({payload['num_scenarios']} scenarios from {len(shard_paths)} shards)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
