"""Larger independent observed-step endpoint replication.

This builder reuses the observed-step scoring path from
``build_observed_step_preregistration.py`` but writes a separate artifact for
a larger independent draw. The rule is fixed before evaluating this draw, but
the artifact intentionally does not describe itself as externally
preregistered.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
from concurrent.futures import Executor, ProcessPoolExecutor
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch

from gnn_state_estimation.evaluation import run_model_inference_batched
from gnn_state_estimation.utils.runtime import resolve_device

try:
    from build_observed_step_preregistration import (
        REFERENCE_METRIC,
        SCENARIOS,
        all_step_pos_rmse,
        build_innovation_features,
        build_prior_bank_feature_array,
        generate_dataset,
        generate_noisy_init,
        load_rgr_gf,
        load_yaml,
        observed_step_pos_rmse,
        parse_baseline_config,
        parse_dataset_config,
        parse_train_config,
        percentile_bootstrap_ci,
        resolve_sim_configs,
        run_filter_baselines,
    )
except ModuleNotFoundError:  # pragma: no cover
    from scripts.build_observed_step_preregistration import (
        REFERENCE_METRIC,
        SCENARIOS,
        all_step_pos_rmse,
        build_innovation_features,
        build_prior_bank_feature_array,
        generate_dataset,
        generate_noisy_init,
        load_rgr_gf,
        load_yaml,
        observed_step_pos_rmse,
        parse_baseline_config,
        parse_dataset_config,
        parse_train_config,
        percentile_bootstrap_ci,
        resolve_sim_configs,
        run_filter_baselines,
    )

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "observed_step_prospective_replication"
OUT_PATH = OUT_DIR / "observed_step_prospective_replication.json"
RULE_PATH = (
    ROOT
    / "release"
    / "predeclarations"
    / "observed_step_prospective_replication_loop71.json"
)
DEFAULT_SCHEMA_VERSION = "observed_step_prospective_replication_v1"
DEFAULT_ARTIFACT_ROLE = "larger_independent_endpoint_replication"

PRIMARY_METRIC = "observed_step_position_rmse_m"
BASE_SEED = 880000
DEFAULT_REALIZATIONS_PER_SCENARIO = 32
DEFAULT_TRAJECTORIES_PER_REALIZATION = 24
DEFAULT_BOOTSTRAP_SAMPLES = 5000
_MAX_WINDOWS_FILTER_WORKERS = 61


def _effective_filter_workers(requested: int, n_trajectories: int) -> int:
    if requested < 1:
        raise ValueError("filter workers must be >= 1")
    effective = min(int(requested), max(int(n_trajectories), 1))
    if os.name == "nt":
        effective = min(effective, _MAX_WINDOWS_FILTER_WORKERS)
    return effective


def _filter_trajectory_worker(task: tuple) -> tuple[int, dict[str, np.ndarray]]:
    """Run the existing baseline implementation for one trajectory.

    ``x0_estimates`` is mandatory for the parallel path, so this worker never
    draws initial-state noise; the seed is forwarded only to preserve the
    serial call signature.
    """
    (
        traj_idx,
        state,
        measurement,
        visible,
        time_s,
        dataset_cfg,
        baseline_cfg,
        seed,
        x0_estimate,
    ) = task
    filters = run_filter_baselines(
        states=state[np.newaxis, ...],
        measurements=measurement[np.newaxis, ...],
        visibility=visible[np.newaxis, ...],
        times=time_s[np.newaxis, ...],
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        seed=seed,
        x0_estimates=x0_estimate[np.newaxis, ...],
    )
    return int(traj_idx), {key: value[0] for key, value in filters.items()}


def _run_filter_baselines_optional_parallel(
    *,
    states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times: np.ndarray,
    dataset_cfg,
    baseline_cfg,
    seed: int,
    x0_estimates: np.ndarray | None,
    filter_workers: int,
    executor: Executor | None = None,
) -> dict[str, np.ndarray]:
    workers = int(filter_workers)
    n_traj = int(states.shape[0])
    if workers < 1:
        raise ValueError("filter_workers must be >= 1")
    if workers == 1 or n_traj <= 1:
        return run_filter_baselines(
            states=states,
            measurements=measurements,
            visibility=visibility,
            times=times,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            seed=seed,
            x0_estimates=x0_estimates,
        )
    if x0_estimates is None:
        raise ValueError(
            "parallel filter path requires precomputed x0_estimates to avoid "
            "worker-side random initial-state draws"
        )
    if int(x0_estimates.shape[0]) != n_traj:
        raise ValueError(
            f"x0_estimates first dimension must match trajectories: "
            f"{x0_estimates.shape[0]} != {n_traj}"
        )

    effective_workers = _effective_filter_workers(workers, n_traj)
    if effective_workers <= 1:
        return run_filter_baselines(
            states=states,
            measurements=measurements,
            visibility=visibility,
            times=times,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            seed=seed,
            x0_estimates=x0_estimates,
        )

    tasks = [
        (
            i,
            states[i],
            measurements[i],
            visibility[i],
            times[i],
            dataset_cfg,
            baseline_cfg,
            seed,
            x0_estimates[i],
        )
        for i in range(n_traj)
    ]

    if executor is None:
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp_context,
        ) as local_executor:
            indexed_filters = list(
                local_executor.map(_filter_trajectory_worker, tasks, chunksize=1)
            )
    else:
        indexed_filters = list(
            executor.map(_filter_trajectory_worker, tasks, chunksize=1)
        )

    indexed_filters.sort(key=lambda item: item[0])
    if len(indexed_filters) != n_traj:
        raise RuntimeError(
            f"parallel filter path returned {len(indexed_filters)} trajectories, "
            f"expected {n_traj}"
        )
    keys = tuple(indexed_filters[0][1].keys())
    return {
        key: np.stack([filters[key] for _, filters in indexed_filters], axis=0)
        for key in keys
    }


def _rounded_mean(values: dict[str, np.ndarray]) -> dict[str, float]:
    return {m: round(float(np.mean(v)), 2) for m, v in values.items()}


def _rounded_std(values: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for m, v in values.items():
        ddof = 1 if v.size > 1 else 0
        out[m] = round(float(np.std(v, ddof=ddof)), 2)
    return out


def _rounded_series(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {m: [round(float(x), 2) for x in arr.tolist()] for m, arr in values.items()}


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_inference_determinism(device: torch.device) -> None:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _station_ecef_for_batched_inference(
    station_ecef: np.ndarray,
    n_trajectories: int,
) -> np.ndarray:
    """Normalize static station geometry for batched model calls."""
    station = np.asarray(station_ecef)
    if station.ndim == 2:
        return station
    if station.ndim == 3 and station.shape[0] == n_trajectories:
        return station
    if station.ndim == 3 and station.shape[0] == 1:
        return np.repeat(station, n_trajectories, axis=0)
    raise ValueError(
        "station_ecef must be [stations, 3], [1, stations, 3], or "
        f"[n_trajectories, stations, 3]; got shape {station.shape}"
    )


def _relative_rule_path(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _is_canonical_k32(args: argparse.Namespace, selected_scenarios: list[str]) -> bool:
    return (
        int(args.num_realizations) == DEFAULT_REALIZATIONS_PER_SCENARIO
        and int(args.trajectories) == DEFAULT_TRAJECTORIES_PER_REALIZATION
        and int(args.base_seed) == BASE_SEED
        and int(args.only_scenario_index) < 0
        and selected_scenarios == [s[0] for s in SCENARIOS]
        and str(args.artifact_role) == DEFAULT_ARTIFACT_ROLE
        and _relative_rule_path(str(args.fixed_rule_path))
        == str(RULE_PATH.relative_to(ROOT)).replace("\\", "/")
    )


def _rule_block(
    args: argparse.Namespace,
    checkpoint_name: str,
    selected_scenarios: list[str],
) -> dict:
    k = int(args.num_realizations)
    n = int(args.trajectories)
    canonical_k32 = _is_canonical_k32(args, selected_scenarios)
    rule_type = (
        "larger independent endpoint replication under a frozen K=32 "
        "decision rule and established observed-step hierarchy"
        if canonical_k32
        else (
            "stress-focused powered replication under a frozen observed-step rule"
            if selected_scenarios == ["stress_test"]
            else "independent endpoint replication under a frozen observed-step rule"
        )
    )
    interpretation_boundary = (
        "This artifact is a larger independent replication under the "
        "established observed-step hierarchy and a frozen K=32 decision "
        "rule fixed before this new draw was evaluated. It is "
        "not an external preregistration and supports "
        "only the endpoint ordering for the specified simulator "
        "scenarios."
        if canonical_k32
        else (
            "This artifact is an internal frozen-rule replication under an "
            "already selected observed-step endpoint. It does not constitute "
            "external preregistration, public-reference validation, or "
            "operational validation, and it permits no selection, tuning, or "
            "retraining after the rule is fixed."
        )
    )
    seed_disjointness = (
        f"base seed {int(args.base_seed)} is disjoint from the 41-55 "
        "training/validation cohort, model-selection validation splits, "
        "the earlier observed-step endpoint-fixation base seed 770000, "
        "and the scenario-resampling base seed 90000"
        if canonical_k32
        else (
            f"base seed {int(args.base_seed)} is disjoint from the 41-55 "
            "training/validation cohort, model-selection validation splits, "
            "the earlier observed-step endpoint-fixation base seed 770000, "
            "the K=32 replication base seed 880000, the scenario-resampling "
            "base seed 90000, and prior endpoint-extension shard seeds "
            "770539-770566"
        )
    )
    rule = {
        "rule_type": rule_type,
        "fixed_rule_path": _relative_rule_path(str(args.fixed_rule_path)),
        "not_external_preregistration": True,
        "frozen_before_evaluation": True,
        "primary_metric": PRIMARY_METRIC,
        "reference_metric": REFERENCE_METRIC,
        "decision_predicate": (
            "For each scenario, a learned positive requires the released "
            "RGR-GF estimator to have the lowest mean observed-step position "
            "RMSE and the 95% percentile bootstrap CI for the paired "
            "RGR-GF-minus-best-classical observed-step gap to be strictly "
            "below zero. The all-step position RMSE is reported only as a "
            "reference metric."
        ),
        "realization_base_seed": int(args.base_seed),
        "seed_disjointness": seed_disjointness,
        "scenarios": selected_scenarios,
        "num_realizations_per_scenario": k,
        "trajectories_per_realization": n,
        "bootstrap_samples": int(args.bootstrap_samples),
        "statistical_unit": "independent realization",
        "inference_only": True,
        "no_selection_tuning_or_retraining": True,
        "interpretation_boundary": interpretation_boundary,
    }
    if canonical_k32:
        rule["fixed_released_checkpoint"] = checkpoint_name
    else:
        rule["fixed_released_estimator"] = "fixed previously trained RGR-GF estimator"
    if (
        args.power_required_realizations is not None
        or args.power_floor_effect_m is not None
    ):
        required = (
            int(args.power_required_realizations)
            if args.power_required_realizations is not None
            else None
        )
        rule["power_design"] = {
            "stress_floor_power_requirement_realizations": required,
            "realizations_requested": k,
            "exceeds_floor_power_requirement": (
                required is not None and k >= required
            ),
            "floor_effect_reference_m": (
                float(args.power_floor_effect_m)
                if args.power_floor_effect_m is not None
                else None
            ),
            "target_power": 0.80,
            "one_sided_alpha": 0.05,
        }
    return rule


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument(
        "--num-realizations",
        type=int,
        default=DEFAULT_REALIZATIONS_PER_SCENARIO,
    )
    ap.add_argument(
        "--trajectories",
        type=int,
        default=DEFAULT_TRAJECTORIES_PER_REALIZATION,
    )
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    ap.add_argument(
        "--only-scenario-index",
        type=int,
        default=-1,
        help=(
            "Optional shard index. Keeps the original scenario index so "
            "per-realization seeds match the full run."
        ),
    )
    ap.add_argument("--out-path", default=str(OUT_PATH))
    ap.add_argument("--fixed-rule-path", default=str(RULE_PATH))
    ap.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    ap.add_argument("--artifact-role", default=DEFAULT_ARTIFACT_ROLE)
    ap.add_argument("--power-required-realizations", type=int, default=None)
    ap.add_argument("--power-floor-effect-m", type=float, default=None)
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for learned RGR-GF inference. auto uses CUDA when available.",
    )
    ap.add_argument(
        "--inference-batch-size",
        type=int,
        default=64,
        help="Trajectory batch size for batched learned-model inference.",
    )
    ap.add_argument(
        "--filter-workers",
        type=int,
        default=1,
        help=(
            "Classical-filter worker processes across trajectories. "
            "Default 1 preserves the existing serial filter path."
        ),
    )
    args = ap.parse_args()
    if int(args.inference_batch_size) < 1:
        ap.error("--inference-batch-size must be >= 1")
    if int(args.filter_workers) < 1:
        ap.error("--filter-workers must be >= 1")

    cfg = load_yaml(Path(args.config))
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    device = resolve_device(str(args.device))
    _configure_inference_determinism(device)
    model = load_rgr_gf(cfg, train_cfg, device)

    k = int(args.num_realizations)
    n = int(args.trajectories)
    requested_filter_workers = int(args.filter_workers)
    filter_workers = _effective_filter_workers(requested_filter_workers, n)
    classical_keys = ["EKF", "UKF", "AUKF"]
    method_keys = classical_keys + ["RGR-GF"]
    checkpoint_name = str(cfg["models"]["HybridGNN"]["checkpoint_name"])

    scenario_rows = []
    only_idx = int(args.only_scenario_index)
    selected_scenarios = [
        (s_idx, key, label, regime)
        for s_idx, (key, label, regime) in enumerate(SCENARIOS)
        if only_idx < 0 or s_idx == only_idx
    ]
    if not selected_scenarios:
        raise ValueError(f"No scenario matched --only-scenario-index={only_idx}")

    filter_worker_text = f"filter_workers={filter_workers}"
    if filter_workers != requested_filter_workers:
        filter_worker_text += f" requested_filter_workers={requested_filter_workers}"
    print(
        f"device={device} inference=batched batch_size={int(args.inference_batch_size)} "
        f"{filter_worker_text} scenarios={len(selected_scenarios)} K={k} N={n}",
        flush=True,
    )
    run_t0 = time.perf_counter()
    for scenario_ord, (s_idx, key, label, regime) in enumerate(
        selected_scenarios, start=1
    ):
        scenario_t0 = time.perf_counter()
        print(f"[scenario {scenario_ord}/{len(selected_scenarios)}] {label}", flush=True)
        est_sim, truth_sim = resolve_sim_configs(cfg, key)
        truth_dc = parse_dataset_config(truth_sim)
        est_dc = parse_dataset_config(est_sim)
        primary = {m: [] for m in method_keys}
        reference = {m: [] for m in method_keys}
        filter_executor = None
        if filter_workers > 1:
            mp_context = multiprocessing.get_context("spawn")
            filter_executor = ProcessPoolExecutor(
                max_workers=filter_workers,
                mp_context=mp_context,
            )
        for r in range(k):
            realization_t0 = time.perf_counter()
            seed = int(args.base_seed) + 1000 * (s_idx + 1) + r
            stage_t0 = time.perf_counter()
            data = generate_dataset(truth_dc, n, seed=seed)
            data_sec = time.perf_counter() - stage_t0
            states, meas, vis, times = (
                data["states"],
                data["measurements"],
                data["visibility"],
                data["times"],
            )
            stage_t0 = time.perf_counter()
            rng = np.random.default_rng(seed)
            x0 = np.stack(
                [
                    generate_noisy_init(
                        states[i, 0],
                        rng,
                        baseline_cfg.init_pos_std_m,
                        baseline_cfg.init_vel_std_mps,
                    )
                    for i in range(states.shape[0])
                ]
            )
            prior_filters = _run_filter_baselines_optional_parallel(
                states=states,
                measurements=meas,
                visibility=vis,
                times=times,
                dataset_cfg=est_dc,
                baseline_cfg=baseline_cfg,
                seed=seed,
                x0_estimates=x0,
                filter_workers=filter_workers,
                executor=filter_executor,
            )
            filters_sec = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()
            innovation_features = build_innovation_features(
                dataset_cfg=est_dc,
                measurements=meas,
                visibility=vis,
                times=times,
                ekf_prior=prior_filters["ekf"],
            )
            prior_bank_stats = build_prior_bank_feature_array(
                prior_filters["ekf"],
                prior_filters["ukf"],
                prior_filters["aukf"],
                dataset_cfg=est_dc,
                station_ecef=data["station_ecef"],
                visibility=vis,
                times=times,
                use_observability_context=False,
            )
            features_sec = time.perf_counter() - stage_t0
            _sync_if_cuda(device)
            stage_t0 = time.perf_counter()
            with torch.inference_mode():
                rgr_gf = run_model_inference_batched(
                    model=model,
                    states=states,
                    measurements=meas,
                    visibility=vis,
                    station_ecef=_station_ecef_for_batched_inference(
                        data["station_ecef"], states.shape[0]
                    ),
                    window_size=int(train_cfg.window_size),
                    ekf_prior=prior_filters["ekf"],
                    ukf_prior=prior_filters["ukf"],
                    aukf_prior=prior_filters["aukf"],
                    innovation_features=innovation_features,
                    prior_bank_stats=prior_bank_stats,
                    batch_size=int(args.inference_batch_size),
                )
            _sync_if_cuda(device)
            model_sec = time.perf_counter() - stage_t0
            stage_t0 = time.perf_counter()
            preds = {
                "EKF": prior_filters["ekf"],
                "UKF": prior_filters["ukf"],
                "AUKF": prior_filters["aukf"],
                "RGR-GF": rgr_gf,
            }
            for method in method_keys:
                primary[method].append(
                    observed_step_pos_rmse(states, preds[method], vis, eval_start)
                )
                reference[method].append(
                    all_step_pos_rmse(states, preds[method], eval_start)
                )
            score_sec = time.perf_counter() - stage_t0
            total_sec = time.perf_counter() - realization_t0
            print(
                f"  r={r + 1}/{k} seed={seed} "
                f"data={data_sec:.1f}s filters={filters_sec:.1f}s "
                f"filter_workers={filter_workers} "
                f"features={features_sec:.1f}s model={model_sec:.1f}s "
                f"score={score_sec:.1f}s total={total_sec:.1f}s",
                flush=True,
            )
        if filter_executor is not None:
            filter_executor.shutdown()

        primary_arr = {m: np.asarray(primary[m], dtype=np.float64) for m in method_keys}
        reference_arr = {
            m: np.asarray(reference[m], dtype=np.float64) for m in method_keys
        }
        primary_mean = {m: float(np.mean(primary_arr[m])) for m in method_keys}
        best_classical = min(classical_keys, key=lambda m: primary_mean[m])
        best_overall = min(method_keys, key=lambda m: primary_mean[m])
        paired_gap = primary_arr["RGR-GF"] - primary_arr[best_classical]
        ci_low, ci_high = percentile_bootstrap_ci(
            paired_gap,
            seed=int(args.base_seed) + 17 * (s_idx + 1),
            n_boot=int(args.bootstrap_samples),
        )
        learned_positive = bool(best_overall == "RGR-GF" and ci_high < 0.0)

        row = {
            "name": key,
            "label": label,
            "regime": regime,
            "scenario_index": s_idx,
            "n_realizations": k,
            "trajectories_per_realization": n,
            "observed_step_pos_rmse_m": _rounded_mean(primary_arr),
            "observed_step_pos_rmse_std_m": _rounded_std(primary_arr),
            "all_step_pos_rmse_m": _rounded_mean(reference_arr),
            "primary_observed_step_pos_rmse_m": _rounded_mean(primary_arr),
            "primary_observed_step_pos_rmse_std_m": _rounded_std(primary_arr),
            "reference_all_step_pos_rmse_m": _rounded_mean(reference_arr),
            "best_method_primary": best_overall,
            "best_classical_primary": best_classical,
            "rgr_gf_minus_best_classical_primary_mean_m": round(
                float(np.mean(paired_gap)), 2
            ),
            "rgr_gf_minus_best_classical_primary_ci_low_m": round(float(ci_low), 2),
            "rgr_gf_minus_best_classical_primary_ci_high_m": round(float(ci_high), 2),
            "learned_positive_under_frozen_rule": learned_positive,
            "decision_predicate_satisfied": learned_positive,
            "per_realization_observed_step_m": _rounded_series(primary_arr),
            "per_realization_reference_all_step_m": _rounded_series(reference_arr),
        }
        scenario_rows.append(row)
        scenario_sec = time.perf_counter() - scenario_t0
        print(
            f"{label}: observed-step best={best_overall} "
            f"EKF={primary_mean['EKF']:.1f} UKF={primary_mean['UKF']:.1f} "
            f"AUKF={primary_mean['AUKF']:.1f} RGR-GF={primary_mean['RGR-GF']:.1f} "
            f"| RGR-GF-best_classical={float(np.mean(paired_gap)):.1f} "
            f"CI[{ci_low:.1f},{ci_high:.1f}] learned_positive={learned_positive} "
            f"scenario_time={scenario_sec:.1f}s",
            flush=True,
        )

    learned_positive_count = sum(
        1 for r in scenario_rows if r["learned_positive_under_frozen_rule"]
    )
    classical_best_count = sum(
        1 for r in scenario_rows if r["best_method_primary"] in classical_keys
    )
    selected_scenario_keys = [key for _, key, _, _ in selected_scenarios]
    source = (
        "new independent realizations generated and scored at build time; "
        "classical filters plus the released compact RGR-GF checkpoint in "
        "inference only"
        if _is_canonical_k32(args, selected_scenario_keys)
        else (
            "new independent realizations generated and scored at build time; "
            "classical filters plus the fixed previously trained RGR-GF "
            "estimator in inference only"
        )
    )
    is_stress_only_powered = (
        selected_scenario_keys == ["stress_test"]
        and "stress" in str(args.artifact_role).lower()
    )
    no_positive_verdict = (
        "stress-only powered observed-step replication under the frozen rule: "
        "no learned positive under the decision predicate"
        if is_stress_only_powered
        else (
            "larger independent observed-step replication under the frozen "
            "rule: no learned positive under the decision predicate"
        )
    )
    result = {
        "status": "completed",
        "schema_version": str(args.schema_version),
        "artifact_role": str(args.artifact_role),
        "frozen_rule": _rule_block(
            args,
            checkpoint_name,
            selected_scenario_keys,
        ),
        "statistical_unit": (
            f"independent realization (independent trajectory population and "
            f"measurement-noise draw); per-scenario estimate is the mean over "
            f"{k} independent realizations with a percentile bootstrap CI on "
            f"the paired RGR-GF-minus-best-classical observed-step gap"
        ),
        "source": source,
        "num_scenarios": len(scenario_rows),
        "scenarios": scenario_rows,
        "summary": {
            "n_scenarios": len(scenario_rows),
            "num_realizations_per_scenario": k,
            "trajectories_per_realization": n,
            "scenarios_with_learned_positive_under_frozen_rule": learned_positive_count,
            "scenarios_with_classical_best_on_primary": classical_best_count,
            "verdict": (
                no_positive_verdict
                if learned_positive_count == 0
                else "learned positive observed under the frozen decision predicate"
            ),
        },
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(
        f"wrote {out_path} ({len(scenario_rows)} scenarios, K={k}, "
        f"filter_workers={filter_workers}, "
        f"elapsed={time.perf_counter() - run_t0:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
