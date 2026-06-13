"""Loop-163 larger-K (K=96) all-scenario internal prospective observed-step replication.

This script manages the local freeze record and supporting utilities for a new
independently seeded, larger-K internal prospective replication of the primary
observed-step endpoint across all three simulator scenarios (nominal, stress,
force-model mismatch).  It mirrors the loop-160 K=32 replication harness but
triples the per-scenario realization count to K=96 (matching the powered
stress-only replication's depth, now extended to every scenario) under a fresh
disjoint base seed.

The actual scenario shards are run with the existing
``build_observed_step_prospective_replication.py`` builder and merged with the
loop163-specific ``merge-shards`` subcommand of this script (NOT the generic
``merge_observed_step_prospective_replication_shards.py`` utility, which rejects
these shards because shard_s1 carries a stress-focused rule_type string and each
shard lists only its own scenario in frozen_rule.scenarios).

The fixed, previously trained RGR-GF estimator is used in inference only: no
model selection, tuning, or retraining occurs after the freeze record is
written.

Subcommands
-----------
freeze
    Write the local freeze record to
    ``results/observed_step_internal_prospective_replication_loop163_k96/preregistration.json``
    before any evaluation. The record is a confidential internal evidence record
    held with the submission evidence package; it is not external preregistration.

print-shard-commands
    Print the three exact CLI commands for running one scenario shard each with
    the canonical loop-163 base seed, K=96 realizations, rule path, and output
    paths.  Does not run the experiment.  Also prints the loop163-specific
    merge-shards command and the export-csv command.

export-csv
    Read the completed merged JSON artifact and write durable CSV summaries to
    the same results folder:

      * ``scenario_summary.csv`` --- one row per scenario with mean RMSE values
        and paired CI for each method.
      * ``per_realization_observed_step.csv`` --- per-realization RMSE for each
        method and scenario.

validate-freeze
    Re-read the written freeze record and verify its SHA-256 self-hash is
    internally consistent.

merge-shards
    Validate and merge the three loop-163 scenario shard JSON files into the
    canonical merged artifact.  Tolerates scenario-specific differences in
    frozen_rule.scenarios and frozen_rule.rule_type across shards (the
    properties that cause the generic merge utility to reject these shards).
    The merged frozen_rule is rebuilt authoritatively from the preregistration
    record.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = (
    ROOT
    / "results"
    / "observed_step_internal_prospective_replication_loop163_k96"
)
PREREGISTRATION_PATH = OUT_DIR / "preregistration.json"
ARTIFACT_PATH = (
    OUT_DIR / "observed_step_internal_prospective_replication_loop163_k96.json"
)
BUILDER_SCRIPT = ROOT / "scripts" / "build_observed_step_prospective_replication.py"
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
K32_RULE_PATH = (
    ROOT
    / "release"
    / "predeclarations"
    / "observed_step_prospective_replication_loop71.json"
)

# Canonical loop-163 K=96 parameters.
BASE_SEED = 1630000
NUM_REALIZATIONS = 96
TRAJECTORIES = 24
BOOTSTRAP_SAMPLES = 5000
SCHEMA_VERSION = "observed_step_internal_prospective_replication_loop163_k96_rule_v1"
ARTIFACT_ROLE = "additional_internal_prospective_replication_loop163_k96"

# Scenario set (identical to the prior prospective replications).
SCENARIOS = [
    {
        "name": "test",
        "label": "Nominal",
        "regime": "nominal sparse-visibility synthetic split",
    },
    {
        "name": "stress_test",
        "label": "Measurement-noise stress",
        "regime": "inflated measurement noise/outliers",
    },
    {
        "name": "force_model_mismatch_test",
        "label": "Controlled force-model mismatch",
        "regime": (
            "truth inflates drag/SRP/process noise; "
            "estimators keep the compact model"
        ),
    },
]

# Prior base seeds against which disjointness is asserted.
_PRIOR_SEED_BASES: dict[str, int] = {
    "training_validation_cohort_lower": 41,
    "training_validation_cohort_upper": 55,
    "endpoint_fixation_k8_k16": 770000,
    "k32_central_replication": 880000,
    "k96_stress_replication": 991117,
    "scenario_resampling": 90000,
    "loop160_k32_internal_replication": 1160000,
}

# ---------------------------------------------------------------------------
# Merge-shards constants
# ---------------------------------------------------------------------------

# Schema version written by the shard builder (NOT the freeze-record schema).
SHARD_SCHEMA_VERSION = "observed_step_prospective_replication_v1"

# Canonical expected values for the three scenarios.
EXPECTED_SCENARIO_NAMES: list[str] = [
    "test",
    "stress_test",
    "force_model_mismatch_test",
]
EXPECTED_SCENARIO_INDICES: frozenset[int] = frozenset({0, 1, 2})
CLASSICAL_METHODS_MERGE: frozenset[str] = frozenset(("EKF", "UKF", "AUKF"))

# Frozen-rule fields that every shard must agree on (scenario-specific fields
# such as frozen_rule.scenarios and frozen_rule.rule_type are intentionally
# excluded so shards with a stress-focused rule_type string are accepted).
INVARIANT_FROZEN_RULE_FIELDS: dict[str, Any] = {
    "realization_base_seed": BASE_SEED,
    "num_realizations_per_scenario": NUM_REALIZATIONS,
    "trajectories_per_realization": TRAJECTORIES,
    "bootstrap_samples": BOOTSTRAP_SAMPLES,
    "frozen_before_evaluation": True,
    "not_external_preregistration": True,
    "inference_only": True,
    "no_selection_tuning_or_retraining": True,
}


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str | None:
    """Return hex SHA-256 of a file, or None if the file does not exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_canonical_json(record: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON representation (sorted keys, UTF-8)."""
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    """Return a repo-relative forward-slash path string."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_checkpoint_name() -> str | None:
    """Try to read the checkpoint name from configs/experiment.yaml."""
    if not CONFIG_PATH.exists():
        return None
    try:
        # Lightweight manual YAML parse to avoid a heavy dependency.
        text = CONFIG_PATH.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("checkpoint_name:"):
                val = stripped.split(":", 1)[1].strip().strip("\"'")
                if val:
                    return val
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Freeze record builder
# ---------------------------------------------------------------------------

def _build_freeze_record(checkpoint_name: str | None) -> dict[str, Any]:
    """Build the full freeze-record dict (without the self-hash field)."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    source_hashes: dict[str, Any] = {}
    for label, path in [
        ("configs/experiment.yaml", CONFIG_PATH),
        (
            "scripts/build_observed_step_prospective_replication.py",
            BUILDER_SCRIPT,
        ),
        (
            "release/predeclarations/observed_step_prospective_replication_loop71.json",
            K32_RULE_PATH,
        ),
    ]:
        h = _sha256_file(path)
        source_hashes[label] = {
            "sha256": h,
            "exists_at_freeze_time": path.exists(),
        }

    checkpoint_path = (
        ROOT / "results" / "checkpoints" / checkpoint_name
        if checkpoint_name
        else None
    )
    checkpoint_hash = _sha256_file(checkpoint_path) if checkpoint_path else None

    seed_disjointness_text = (
        f"base seed {BASE_SEED} is disjoint from the 41-55 "
        "training/validation cohort, model-selection validation splits, "
        "the earlier observed-step endpoint-fixation base seed 770000, "
        "the K=32 central replication base seed 880000, "
        "the stress-only K=96 replication base seed 991117, "
        "the scenario-resampling base seed 90000, "
        "the loop-160 K=32 internal replication base seed 1160000, "
        "and prior endpoint-extension shard seeds 770539-770566"
    )

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": now_utc,
        "evidence_scope": (
            "confidential internal evidence record held with the submission "
            "evidence package; not external preregistration"
        ),
        "not_external_preregistration": True,
        "frozen_before_evaluation": True,
        "endpoint_hierarchy": {
            "primary_metric": "observed_step_position_rmse_m",
            "reference_metric": "all_step_position_rmse_m",
            "endpoint_selection_note": (
                "observed-step RMSE was selected on training-cohort data "
                "in a post-hoc recomputation; the K=8 endpoint-fixation "
                "support record, the K=32 central replication, and the K=96 "
                "stress-only powered replication preceded this larger-K "
                "all-scenario independently seeded internal replication"
            ),
            "primary_metric_description": (
                "mean observed-step position RMSE over independent "
                "realizations; observed steps are those with at least one "
                "visible station from the window-start step onward"
            ),
            "reference_metric_description": (
                "all-step position RMSE retained as a propagation-dominated "
                "reference; not part of the decision predicate"
            ),
        },
        "decision_rule": {
            "decision_predicate": (
                "For each scenario, a learned positive requires the fixed "
                "RGR-GF estimator to have the lowest mean observed-step "
                "position RMSE and the 95% percentile bootstrap CI for the "
                "paired RGR-GF-minus-best-classical observed-step gap to be "
                "strictly below zero. The all-step position RMSE is reported "
                "only as a propagation-dominated reference."
            ),
            "learned_positive_criterion": (
                "RGR-GF must be the lowest-mean method for the scenario and "
                "the upper endpoint of the 95% paired-gap bootstrap CI must "
                "be strictly below zero"
            ),
            "bootstrap_method": "percentile bootstrap over independent realizations",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
        },
        "seed_plan": {
            "base_seed": BASE_SEED,
            "seed_disjointness": seed_disjointness_text,
            "prior_seed_bases": _PRIOR_SEED_BASES,
            "per_realization_seed_formula": (
                "seed = base_seed + 1000 * (scenario_index + 1) + realization_index"
            ),
            "per_realization_seed_ranges": {
                "test": [BASE_SEED + 1000, BASE_SEED + 1000 + NUM_REALIZATIONS - 1],
                "stress_test": [
                    BASE_SEED + 2000,
                    BASE_SEED + 2000 + NUM_REALIZATIONS - 1,
                ],
                "force_model_mismatch_test": [
                    BASE_SEED + 3000,
                    BASE_SEED + 3000 + NUM_REALIZATIONS - 1,
                ],
            },
            "statistical_unit": (
                "independent realization with an independently seeded "
                "trajectory population and measurement-noise draw"
            ),
        },
        "scenarios": SCENARIOS,
        "num_realizations_per_scenario": NUM_REALIZATIONS,
        "trajectories_per_realization": TRAJECTORIES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "practical_floor_rule": {
            "floor_pct": 3.0,
            "description": (
                "3% of the per-scenario best-classical observed-step position "
                "RMSE; effects below this floor are treated as practically "
                "non-significant regardless of statistical detectability. This "
                "is a mission-agnostic audit threshold for the compact "
                "simulator, not a universal mission requirement."
            ),
        },
        "model_artifact_reference": {
            "role": (
                "fixed, previously trained RGR-GF estimator evaluated without "
                "any per-realization refitting"
            ),
            "checkpoint_name": checkpoint_name,
            "checkpoint_sha256": checkpoint_hash,
            "inference_only": True,
            "no_model_selection": True,
            "no_tuning": True,
            "no_retraining": True,
        },
        "source_input_hashes": source_hashes,
        "interpretation_boundary": (
            "This record defines a larger-K (K=96 per scenario) all-scenario "
            "internal prospective independently seeded replication under the "
            "established observed-step hierarchy and the same frozen decision "
            "predicate used by the K=32 central replication. The rule is fixed "
            "locally before any loop-163 realization is generated or evaluated. "
            "This record is not an external preregistration, does not "
            "constitute public-reference validation or operational validation, "
            "and permits no model selection, tuning, or retraining after the "
            "record is written."
        ),
        "prior_replications": {
            "loop160_k32_internal": {
                "base_seed": 1160000,
                "num_realizations": 32,
                "role": (
                    "additional internal K=32 all-scenario replication; "
                    "fixed before the loop-160 draw was generated"
                ),
            },
            "k32_central": {
                "base_seed": 880000,
                "num_realizations": 32,
                "rule_record": _relative(K32_RULE_PATH),
                "role": (
                    "central anchor; fixed before the K=32 draw was generated"
                ),
            },
            "k96_stress_only": {
                "base_seed": 991117,
                "num_realizations": 96,
                "role": (
                    "stress-focused floor-power check; fixed before the K=96 draw"
                ),
            },
            "k8_endpoint_fixation": {
                "base_seed": 770000,
                "num_realizations": 8,
                "role": "endpoint-fixation support; no external timestamp",
            },
        },
    }
    return record


def write_freeze_record(out_path: Path = PREREGISTRATION_PATH) -> dict[str, Any]:
    """Build and write the freeze record, appending its self-hash.

    The self-hash is computed over the record content without the
    ``canonical_rule_sha256`` field so the hash is deterministic given the
    other inputs. Returns the written record dict.
    """
    checkpoint_name = _load_checkpoint_name()
    record = _build_freeze_record(checkpoint_name)
    # Compute self-hash over record content before adding the hash field.
    self_hash = _sha256_canonical_json(record)
    record["canonical_rule_sha256"] = self_hash
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {_relative(out_path)}")
    print(f"  schema_version : {record['schema_version']}")
    print(f"  created_at_utc : {record['created_at_utc']}")
    print(f"  base_seed      : {record['seed_plan']['base_seed']}")
    print(f"  K              : {record['num_realizations_per_scenario']}")
    print(f"  canonical_sha256: {self_hash[:16]}...")
    return record


def validate_freeze_record(path: Path = PREREGISTRATION_PATH) -> bool:
    """Re-read the freeze record and verify the self-hash is consistent.

    Returns True on success, False on failure.
    """
    if not path.exists():
        print(f"freeze record not found: {_relative(path)}", file=sys.stderr)
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = record.pop("canonical_rule_sha256", None)
    if stored_hash is None:
        print("missing canonical_rule_sha256 field", file=sys.stderr)
        return False
    computed_hash = _sha256_canonical_json(record)
    if stored_hash != computed_hash:
        print(
            f"hash mismatch: stored={stored_hash[:16]}... "
            f"computed={computed_hash[:16]}...",
            file=sys.stderr,
        )
        return False
    print(f"freeze record hash verified: {stored_hash[:16]}...")
    return True


# ---------------------------------------------------------------------------
# Shard command printer
# ---------------------------------------------------------------------------

def _shard_out_path(scenario_index: int) -> str:
    return _relative(OUT_DIR / f"shard_s{scenario_index}.json")


def print_shard_commands(
    args_device: str = "auto",
    filter_workers: int = 12,
) -> None:
    """Print the exact CLI commands to run each of the three scenario shards."""
    rule_path = _relative(PREREGISTRATION_PATH)
    builder = _relative(BUILDER_SCRIPT)
    print(
        "# Loop-163 larger-K (K=96) all-scenario internal prospective replication\n"
        "# shard commands.\n"
        "# Run freeze first, then execute these commands sequentially or in parallel,\n"
        "# then run the loop163-specific merge-shards subcommand (see below).\n"
        "#\n"
        f"# Each shard scores one scenario with {NUM_REALIZATIONS} independent\n"
        f"# realizations and {TRAJECTORIES} trajectories per realization\n"
        f"# (base seed {BASE_SEED}).\n"
    )
    for s_idx in range(len(SCENARIOS)):
        out = _shard_out_path(s_idx)
        print(
            f"python {builder} \\\n"
            f"  --base-seed {BASE_SEED} \\\n"
            f"  --num-realizations {NUM_REALIZATIONS} \\\n"
            f"  --trajectories {TRAJECTORIES} \\\n"
            f"  --only-scenario-index {s_idx} \\\n"
            f"  --fixed-rule-path {rule_path} \\\n"
            f"  --artifact-role {ARTIFACT_ROLE} \\\n"
            f"  --out-path {out} \\\n"
            f"  --device {args_device} \\\n"
            f"  --filter-workers {filter_workers}\n"
        )
    this_script = _relative(Path(__file__))
    shard_args = " \\\n  ".join(
        f"--shard {_shard_out_path(i)}" for i in range(len(SCENARIOS))
    )
    artifact_out = _relative(ARTIFACT_PATH)
    freeze_path = _relative(PREREGISTRATION_PATH)
    print(
        "# Merge command (after all shards complete) -- use the loop163-specific\n"
        "# merge-shards subcommand; it tolerates scenario-specific frozen_rule\n"
        "# differences (rule_type, scenarios) that cause the generic merge utility\n"
        "# to reject these shards.\n"
        "#\n"
        "# Default (uses canonical shard paths and preregistration.json):\n"
        f"python {this_script} merge-shards\n"
        "#\n"
        "# Or with explicit paths:\n"
        f"python {this_script} merge-shards \\\n"
        f"  {shard_args} \\\n"
        f"  --out-path {artifact_out} \\\n"
        f"  --freeze-record {freeze_path}\n"
    )
    print(
        "# Export CSV (after merge completes):\n"
        f"python {this_script} export-csv\n"
    )


# ---------------------------------------------------------------------------
# CSV exporter
# ---------------------------------------------------------------------------

def export_csv(artifact_path: Path = ARTIFACT_PATH) -> None:
    """Read the merged artifact and write scenario_summary.csv and
    per_realization_observed_step.csv to the results folder."""
    if not artifact_path.exists():
        print(
            f"artifact not found: {_relative(artifact_path)}", file=sys.stderr
        )
        sys.exit(1)

    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = data.get("scenarios", [])
    if not rows:
        print("no scenario rows in artifact", file=sys.stderr)
        sys.exit(1)

    out_dir = artifact_path.parent
    methods = ["EKF", "UKF", "AUKF", "RGR-GF"]

    # ------------------------------------------------------------------
    # scenario_summary.csv
    # ------------------------------------------------------------------
    summary_path = out_dir / "scenario_summary.csv"
    summary_fieldnames = [
        "scenario_name",
        "scenario_label",
        "n_realizations",
        "best_classical_primary",
        "ekf_obs_step_rmse_m",
        "ukf_obs_step_rmse_m",
        "aukf_obs_step_rmse_m",
        "rgr_gf_obs_step_rmse_m",
        "rgr_gf_minus_best_classical_mean_m",
        "rgr_gf_minus_best_classical_ci_low_m",
        "rgr_gf_minus_best_classical_ci_high_m",
        "learned_positive_under_frozen_rule",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in rows:
            obs = row.get("observed_step_pos_rmse_m") or row.get(
                "primary_observed_step_pos_rmse_m", {}
            )
            writer.writerow(
                {
                    "scenario_name": row.get("name", ""),
                    "scenario_label": row.get("label", ""),
                    "n_realizations": row.get("n_realizations", ""),
                    "best_classical_primary": row.get("best_classical_primary", ""),
                    "ekf_obs_step_rmse_m": obs.get("EKF", ""),
                    "ukf_obs_step_rmse_m": obs.get("UKF", ""),
                    "aukf_obs_step_rmse_m": obs.get("AUKF", ""),
                    "rgr_gf_obs_step_rmse_m": obs.get("RGR-GF", ""),
                    "rgr_gf_minus_best_classical_mean_m": row.get(
                        "rgr_gf_minus_best_classical_primary_mean_m", ""
                    ),
                    "rgr_gf_minus_best_classical_ci_low_m": row.get(
                        "rgr_gf_minus_best_classical_primary_ci_low_m", ""
                    ),
                    "rgr_gf_minus_best_classical_ci_high_m": row.get(
                        "rgr_gf_minus_best_classical_primary_ci_high_m", ""
                    ),
                    "learned_positive_under_frozen_rule": row.get(
                        "learned_positive_under_frozen_rule", ""
                    ),
                }
            )
    print(f"wrote {_relative(summary_path)} ({len(rows)} scenarios)")

    # ------------------------------------------------------------------
    # per_realization_observed_step.csv
    # ------------------------------------------------------------------
    per_real_path = out_dir / "per_realization_observed_step.csv"
    per_real_fieldnames = [
        "scenario_name",
        "scenario_label",
        "realization_index",
        "method",
        "observed_step_pos_rmse_m",
    ]
    with per_real_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=per_real_fieldnames)
        writer.writeheader()
        for row in rows:
            per_real = row.get(
                "per_realization_observed_step_m",
                row.get("per_realization_primary_m", {}),
            )
            n = row.get("n_realizations", 0)
            for method in methods:
                vals = per_real.get(method, [])
                for r_idx in range(int(n)):
                    v = vals[r_idx] if r_idx < len(vals) else ""
                    writer.writerow(
                        {
                            "scenario_name": row.get("name", ""),
                            "scenario_label": row.get("label", ""),
                            "realization_index": r_idx,
                            "method": method,
                            "observed_step_pos_rmse_m": v,
                        }
                    )
    total_rows = sum(
        int(r.get("n_realizations", 0)) * len(methods) for r in rows
    )
    print(f"wrote {_relative(per_real_path)} ({total_rows} data rows)")


# ---------------------------------------------------------------------------
# Loop-163-specific shard merge
# ---------------------------------------------------------------------------


class MergeValidationError(ValueError):
    """Raised when loop-163 shards cannot safely be merged."""


def _validate_loop163_shard(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate a single loop-163 shard payload and return the scenario row.

    Accepts shards whose frozen_rule.scenarios and frozen_rule.rule_type differ
    across shards (e.g. shard_s1 carries a stress-focused rule_type string).
    Only the invariant frozen_rule fields in INVARIANT_FROZEN_RULE_FIELDS are
    checked.
    """
    ctx = _relative(path)

    sv = payload.get("schema_version")
    if sv != SHARD_SCHEMA_VERSION:
        raise MergeValidationError(
            f"{ctx}: schema_version must be {SHARD_SCHEMA_VERSION!r}, got {sv!r}"
        )

    status = payload.get("status")
    if status != "completed":
        raise MergeValidationError(
            f"{ctx}: status must be 'completed', got {status!r}"
        )

    role = payload.get("artifact_role")
    if role != ARTIFACT_ROLE:
        raise MergeValidationError(
            f"{ctx}: artifact_role must be {ARTIFACT_ROLE!r}, got {role!r}"
        )

    frozen_rule = payload.get("frozen_rule")
    if not isinstance(frozen_rule, dict):
        raise MergeValidationError(f"{ctx}: frozen_rule must be a JSON object")

    for key, expected_val in INVARIANT_FROZEN_RULE_FIELDS.items():
        actual = frozen_rule.get(key)
        if actual != expected_val:
            raise MergeValidationError(
                f"{ctx}: frozen_rule.{key} must be {expected_val!r}, got {actual!r}"
            )

    scenarios_list = payload.get("scenarios")
    if not isinstance(scenarios_list, list) or len(scenarios_list) != 1:
        n = (
            len(scenarios_list)
            if isinstance(scenarios_list, list)
            else type(scenarios_list)
        )
        raise MergeValidationError(
            f"{ctx}: expected exactly one scenario per shard, got {n}"
        )

    scenario = scenarios_list[0]
    if not isinstance(scenario, dict):
        raise MergeValidationError(f"{ctx}: scenario entry must be a JSON object")

    s_idx = scenario.get("scenario_index")
    if not isinstance(s_idx, int) or isinstance(s_idx, bool) or s_idx < 0:
        raise MergeValidationError(
            f"{ctx}: scenario_index must be a non-negative integer, got {s_idx!r}"
        )

    n_real = scenario.get("n_realizations")
    if n_real != NUM_REALIZATIONS:
        raise MergeValidationError(
            f"{ctx}: n_realizations must be {NUM_REALIZATIONS}, got {n_real!r}"
        )

    lp = scenario.get("learned_positive_under_frozen_rule")
    if not isinstance(lp, bool):
        raise MergeValidationError(
            f"{ctx}: learned_positive_under_frozen_rule must be boolean, "
            f"got {lp!r} (type {type(lp).__name__})"
        )

    if lp:
        bm = scenario.get("best_method_primary")
        if bm != "RGR-GF":
            raise MergeValidationError(
                f"{ctx}: when learned_positive_under_frozen_rule is true, "
                f"best_method_primary must be 'RGR-GF', got {bm!r}"
            )
        ci_high = scenario.get("rgr_gf_minus_best_classical_primary_ci_high_m")
        if ci_high is None or not isinstance(ci_high, (int, float)) or ci_high >= 0.0:
            raise MergeValidationError(
                f"{ctx}: when learned_positive_under_frozen_rule is true, "
                f"rgr_gf_minus_best_classical_primary_ci_high_m must be < 0, "
                f"got {ci_high!r}"
            )

    return scenario


def _build_merged_frozen_rule(
    freeze_record: dict[str, Any],
    freeze_record_path: Path,
) -> dict[str, Any]:
    """Build the top-level frozen_rule for the merged artifact.

    Reads canonical values from the preregistration record so the merged rule
    covers all three scenarios. The scenario-specific frozen_rule fields from
    the individual shards are intentionally discarded; the preregistration is
    the authoritative source.
    """
    dr = freeze_record.get("decision_rule", {})
    sp = freeze_record.get("seed_plan", {})
    ep = freeze_record.get("endpoint_hierarchy", {})
    mar = freeze_record.get("model_artifact_reference", {})
    freeze_canonical_sha = freeze_record.get("canonical_rule_sha256", "")

    return {
        "rule_type": (
            "larger-K (K=96) all-scenario independent endpoint replication "
            "under a frozen observed-step rule"
        ),
        "fixed_rule_path": _relative(freeze_record_path),
        "freeze_record_canonical_sha256": freeze_canonical_sha,
        "not_external_preregistration": freeze_record.get(
            "not_external_preregistration", True
        ),
        "frozen_before_evaluation": freeze_record.get("frozen_before_evaluation", True),
        "primary_metric": ep.get("primary_metric", "observed_step_position_rmse_m"),
        "reference_metric": ep.get("reference_metric", "all_step_position_rmse_m"),
        "decision_predicate": dr.get("decision_predicate", ""),
        "realization_base_seed": sp.get("base_seed", BASE_SEED),
        "seed_disjointness": sp.get("seed_disjointness", ""),
        "scenarios": list(EXPECTED_SCENARIO_NAMES),
        "num_realizations_per_scenario": freeze_record.get(
            "num_realizations_per_scenario", NUM_REALIZATIONS
        ),
        "trajectories_per_realization": freeze_record.get(
            "trajectories_per_realization", TRAJECTORIES
        ),
        "bootstrap_samples": freeze_record.get("bootstrap_samples", BOOTSTRAP_SAMPLES),
        "statistical_unit": sp.get("statistical_unit", "independent realization"),
        "inference_only": True,
        "no_selection_tuning_or_retraining": True,
        "interpretation_boundary": freeze_record.get("interpretation_boundary", ""),
        "fixed_released_estimator": mar.get(
            "role", "fixed previously trained RGR-GF estimator"
        ),
    }


def _build_merge_summary(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the top-level summary block for the merged artifact."""
    learned_pos = sum(
        1 for s in scenarios if s.get("learned_positive_under_frozen_rule")
    )
    classical_best = sum(
        1 for s in scenarios if s.get("best_method_primary") in CLASSICAL_METHODS_MERGE
    )
    verdict = (
        "larger-K (K=96) all-scenario independent observed-step replication "
        "under the frozen rule: no learned positive under the decision predicate"
        if learned_pos == 0
        else "learned positive observed under the frozen decision predicate"
    )
    return {
        "n_scenarios": len(scenarios),
        "K": NUM_REALIZATIONS,
        "trajectories_per_realization": TRAJECTORIES,
        "scenarios_with_learned_positive_under_frozen_rule": learned_pos,
        "scenarios_with_classical_best_on_primary": classical_best,
        "verdict": verdict,
    }


def merge_shards_loop163(
    shard_paths: list[Path] | None = None,
    out_path: Path = ARTIFACT_PATH,
    freeze_record_path: Path = PREREGISTRATION_PATH,
) -> dict[str, Any]:
    """Validate and merge the three loop-163 scenario shards.

    Unlike the generic merge utility this function tolerates scenario-specific
    differences in frozen_rule.scenarios and frozen_rule.rule_type across
    shards (shard_s1 carries a stress-focused rule_type string).  The merged
    frozen_rule is rebuilt authoritatively from the preregistration record.

    Parameters
    ----------
    shard_paths:
        Explicit shard file paths.  Defaults to the three canonical loop-163
        shard files in the results folder.
    out_path:
        Destination for the merged artifact JSON.
    freeze_record_path:
        Path to the preregistration.json freeze record.
    """
    if shard_paths is None:
        shard_paths = [
            OUT_DIR / f"shard_s{i}.json" for i in range(len(SCENARIOS))
        ]

    if not freeze_record_path.exists():
        raise MergeValidationError(
            f"freeze record not found: {_relative(freeze_record_path)}"
        )

    # Verify the freeze record self-hash before using any of its payload.
    if not validate_freeze_record(freeze_record_path):
        raise MergeValidationError(
            f"freeze record self-hash verification failed: "
            f"{_relative(freeze_record_path)}; "
            "the record may have been tampered with after the freeze was written"
        )

    freeze_text = freeze_record_path.read_text(encoding="utf-8")
    freeze_record = json.loads(freeze_text)

    # Validate and collect scenario rows.
    seen_indices: set[int] = set()
    indexed_scenarios: list[tuple[int, dict[str, Any]]] = []
    for path in shard_paths:
        if not path.exists():
            raise MergeValidationError(
                f"shard not found: {_relative(path)}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MergeValidationError(
                f"{_relative(path)}: invalid JSON: {exc}"
            ) from exc
        scenario = _validate_loop163_shard(path, payload)
        idx: int = scenario["scenario_index"]
        if idx in seen_indices:
            raise MergeValidationError(
                f"{_relative(path)}: duplicate scenario_index {idx}"
            )
        seen_indices.add(idx)
        indexed_scenarios.append((idx, copy.deepcopy(scenario)))

    if seen_indices != EXPECTED_SCENARIO_INDICES:
        missing = sorted(EXPECTED_SCENARIO_INDICES - seen_indices)
        raise MergeValidationError(
            f"expected scenario_index set {sorted(EXPECTED_SCENARIO_INDICES)}, "
            f"got {sorted(seen_indices)}; missing indices: {missing}"
        )

    indexed_scenarios.sort(key=lambda t: t[0])
    sorted_scenarios = [s for _, s in indexed_scenarios]

    actual_names = [s.get("name") for s in sorted_scenarios]
    if actual_names != EXPECTED_SCENARIO_NAMES:
        raise MergeValidationError(
            f"expected scenario names {EXPECTED_SCENARIO_NAMES} (in "
            f"scenario_index order), got {actual_names}"
        )

    merged_frozen_rule = _build_merged_frozen_rule(freeze_record, freeze_record_path)

    result: dict[str, Any] = {
        "status": "completed",
        "schema_version": SHARD_SCHEMA_VERSION,
        "artifact_role": ARTIFACT_ROLE,
        "frozen_rule": merged_frozen_rule,
        "statistical_unit": (
            "independent realization (independent trajectory population and "
            "measurement-noise draw); per-scenario estimate is the mean over "
            f"{NUM_REALIZATIONS} independent realizations with a percentile "
            "bootstrap CI on the paired RGR-GF-minus-best-classical "
            "observed-step gap"
        ),
        "source": (
            "new independent realizations generated and scored at build time; "
            "classical filters plus the fixed previously trained RGR-GF "
            "estimator in inference only"
        ),
        "num_scenarios": len(sorted_scenarios),
        "scenarios": sorted_scenarios,
        "summary": _build_merge_summary(sorted_scenarios),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"wrote {_relative(out_path)} ({len(sorted_scenarios)} scenarios merged)")
    print(json.dumps(result["summary"], indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # freeze
    freeze_p = subparsers.add_parser(
        "freeze",
        help="Write the local freeze record before any evaluation.",
    )
    freeze_p.add_argument(
        "--out-path",
        type=Path,
        default=PREREGISTRATION_PATH,
        help="Output path for the freeze record JSON.",
    )

    # print-shard-commands
    shard_p = subparsers.add_parser(
        "print-shard-commands",
        help="Print the exact CLI commands for each scenario shard.",
    )
    shard_p.add_argument(
        "--device",
        default="auto",
        help="Device argument forwarded to the shard builder (default: auto).",
    )
    shard_p.add_argument(
        "--filter-workers",
        type=int,
        default=12,
        help="Filter-worker count forwarded to the shard builder (default: 12).",
    )

    # export-csv
    export_p = subparsers.add_parser(
        "export-csv",
        help="Export CSV summaries from the completed merged artifact.",
    )
    export_p.add_argument(
        "--artifact-path",
        type=Path,
        default=ARTIFACT_PATH,
        help="Path to the merged artifact JSON.",
    )

    # validate-freeze
    validate_p = subparsers.add_parser(
        "validate-freeze",
        help="Re-read the freeze record and verify its self-hash.",
    )
    validate_p.add_argument(
        "--path",
        type=Path,
        default=PREREGISTRATION_PATH,
        help="Path to the freeze record JSON.",
    )

    # merge-shards
    merge_p = subparsers.add_parser(
        "merge-shards",
        help=(
            "Validate and merge the loop-163 scenario shards into the canonical "
            "merged artifact. Tolerates scenario-specific frozen_rule.scenarios "
            "and frozen_rule.rule_type differences across shards."
        ),
    )
    merge_p.add_argument(
        "--shard",
        action="append",
        type=Path,
        dest="shard_paths",
        default=None,
        metavar="PATH",
        help=(
            "Path to a shard JSON (may be repeated up to three times). "
            "Defaults to the three canonical loop-163 shard files."
        ),
    )
    merge_p.add_argument(
        "--out-path",
        type=Path,
        default=ARTIFACT_PATH,
        help="Destination for the merged artifact JSON.",
    )
    merge_p.add_argument(
        "--freeze-record",
        type=Path,
        default=PREREGISTRATION_PATH,
        help="Path to the preregistration.json freeze record.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.subcommand == "freeze":
        write_freeze_record(out_path=args.out_path)
        return 0

    if args.subcommand == "print-shard-commands":
        print_shard_commands(
            args_device=args.device,
            filter_workers=int(args.filter_workers),
        )
        return 0

    if args.subcommand == "export-csv":
        export_csv(artifact_path=args.artifact_path)
        return 0

    if args.subcommand == "validate-freeze":
        ok = validate_freeze_record(path=args.path)
        return 0 if ok else 1

    if args.subcommand == "merge-shards":
        try:
            merge_shards_loop163(
                shard_paths=args.shard_paths,
                out_path=args.out_path,
                freeze_record_path=args.freeze_record,
            )
        except MergeValidationError as exc:
            print(f"merge validation failed: {exc}", file=sys.stderr)
            return 2
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
