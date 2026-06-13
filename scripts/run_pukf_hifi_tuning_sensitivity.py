#!/usr/bin/env python
"""Bounded PUKF validation-grid sensitivity on the higher-fidelity slice.

This script addresses the reviewer concern that PUKF is less tuned than AUKF.
It performs a post-hoc, explicitly non-replacing, 16-point validation-grid
stress for the process-noise-adaptive UKF on the higher-fidelity force-mismatch
slice, selects the lowest validation observed-step RMSE, and evaluates only
that selected PUKF on the retained held-out higher-fidelity test population.

The result is a comparator-comparability sensitivity analysis. It does not
replace the timestamped predeclared PUKF decision record.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_hifi_force_mismatch as hifi

from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import (
    AdaptiveUKFConfig,
    EKFConfig,
    ProcessNoiseAdaptiveUKFConfig,
    UKFConfig,
    run_adaptive_ukf,
    run_ekf,
    run_process_noise_adaptive_ukf,
    run_ukf,
)
from gnn_state_estimation.simulation import parse_dataset_config
from gnn_state_estimation.utils.io import load_yaml


@dataclass(frozen=True)
class Population:
    states: np.ndarray
    measurements: np.ndarray
    visibility: np.ndarray
    times: np.ndarray
    x0_estimates: np.ndarray


@dataclass(frozen=True)
class GridPoint:
    label: str
    window_size: int
    smoothing: float
    q_scale_warn: float
    q_scale_alarm: float
    q_scale_max: float
    nis_warn_ratio: float = 1.5
    nis_alarm_ratio: float = 2.0
    nis_per_update_expected: float = 4.0


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _grid() -> list[GridPoint]:
    points: list[GridPoint] = []
    alarm_specs = [
        (3.0, 2.0),
        (6.0, 2.5),
        (10.0, 3.0),
        (18.0, 5.0),
    ]
    for window_size in (5, 10):
        for smoothing in (0.2, 0.4):
            for alarm, warn in alarm_specs:
                points.append(
                    GridPoint(
                        label=f"W{window_size}_S{str(smoothing).replace('.', 'p')}_A{int(alarm)}",
                        window_size=window_size,
                        smoothing=smoothing,
                        q_scale_warn=warn,
                        q_scale_alarm=alarm,
                        q_scale_max=alarm,
                    )
                )
    return points


def _make_population(
    *,
    cfg: dict[str, Any],
    seed: int,
    n_trajectories: int,
    epoch_unix: float,
) -> tuple[Population, dict[str, Any]]:
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector
    rng = np.random.default_rng(seed)
    steps = dyn.steps
    dt = dyn.dt_s
    times = np.tile(np.arange(steps, dtype=np.float64) * dt, (n_trajectories, 1))
    states_all = np.zeros((n_trajectories, steps, 6), dtype=np.float64)
    meas_all = np.zeros((n_trajectories, steps, len(stations), 4), dtype=np.float64)
    vis_all = np.zeros((n_trajectories, steps, len(stations)), dtype=np.float64)
    x0_all = np.zeros((n_trajectories, 6), dtype=np.float64)

    for i in range(n_trajectories):
        x0 = hifi._sample_orbit(dataset_cfg.orbit_sampling, rng)
        states = hifi._propagate_hifi_trajectory(
            x0,
            dt=dt,
            steps=steps,
            epoch0_unix=epoch_unix,
            ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
            drag_rho_ref=dyn.drag_rho_ref,
            drag_h_ref_m=dyn.drag_h_ref_m,
            drag_scale_height_m=dyn.drag_scale_height_m,
        )
        meas, vis = hifi._generate_observations(
            states=states,
            times=times[i],
            stations=stations,
            noise_std=meas_std,
            outlier_prob=dataset_cfg.measurement_noise.outlier_prob,
            outlier_scale=dataset_cfg.measurement_noise.outlier_scale,
            dropout_prob=dataset_cfg.measurement_noise.random_dropout_prob,
            rng=rng,
        )
        perturb = np.array(
            [rng.normal(0.0, baseline_cfg.ukf.init_pos_std_m) for _ in range(3)]
            + [rng.normal(0.0, baseline_cfg.ukf.init_vel_std_mps) for _ in range(3)],
            dtype=np.float64,
        )
        states_all[i] = states
        meas_all[i] = meas
        vis_all[i] = vis
        x0_all[i] = states[0] + perturb

    meta = {
        "generation_protocol": (
            "Deterministic higher-fidelity force-mismatch population generated "
            "by _make_population in scripts/run_pukf_hifi_tuning_sensitivity.py "
            "using the same trajectory and observation helpers as "
            "scripts/run_hifi_force_mismatch.py."
        ),
        "seed": int(seed),
        "n_trajectories": int(n_trajectories),
        "epoch_unix": float(epoch_unix),
        "steps": int(steps),
        "dt_s": float(dt),
        "station_count": len(stations),
        "measurement_std": [float(x) for x in meas_std],
        "trajectory_id_policy": (
            "No external trajectory IDs are stored by the source population; "
            "pairing uses deterministic row order with trajectory_index 0..n-1."
        ),
    }
    return Population(states_all, meas_all, vis_all, times, x0_all), meta


def _estimator_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    dyn = dataset_cfg.dynamics
    return {
        "ballistic_coeff_m2_per_kg": dyn.ballistic_coeff_m2_per_kg,
        "drag_rho_ref": dyn.drag_rho_ref,
        "drag_h_ref_m": dyn.drag_h_ref_m,
        "drag_scale_height_m": dyn.drag_scale_height_m,
        "enable_third_body": dyn.enable_third_body,
        "enable_srp": dyn.enable_srp,
        "srp_area_to_mass_m2_per_kg": dyn.srp_area_to_mass_m2_per_kg,
        "srp_cr": dyn.srp_cr,
        "sun_initial_phase_rad": dyn.sun_initial_phase_rad,
        "moon_initial_phase_rad": dyn.moon_initial_phase_rad,
    }


def _pukf_config(cfg: dict[str, Any], point: GridPoint) -> ProcessNoiseAdaptiveUKFConfig:
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    return ProcessNoiseAdaptiveUKFConfig(
        q_pos_m=baseline_cfg.ukf.q_pos_m,
        q_vel_mps=baseline_cfg.ukf.q_vel_mps,
        init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
        init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
        alpha=baseline_cfg.ukf.alpha,
        beta=baseline_cfg.ukf.beta,
        kappa=baseline_cfg.ukf.kappa,
        window_size=point.window_size,
        nis_per_update_expected=point.nis_per_update_expected,
        nis_warn_ratio=point.nis_warn_ratio,
        nis_alarm_ratio=point.nis_alarm_ratio,
        q_scale_warn=point.q_scale_warn,
        q_scale_alarm=point.q_scale_alarm,
        q_scale_max=point.q_scale_max,
        smoothing=point.smoothing,
        angle_deweight_elev_cap_deg=getattr(
            baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
        ),
    )


def _run_pukf(
    *,
    cfg: dict[str, Any],
    pop: Population,
    point: GridPoint,
) -> tuple[np.ndarray, dict[str, float]]:
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    meas_std = dataset_cfg.measurement_noise.std_vector
    stations = dataset_cfg.stations
    kwargs = _estimator_kwargs(cfg)
    pukf_cfg = _pukf_config(cfg, point)
    pred = np.zeros_like(pop.states)
    q_scales: list[float] = []
    for i in range(pop.states.shape[0]):
        pred[i], _, recs = run_process_noise_adaptive_ukf(
            measurements=pop.measurements[i],
            visibility=pop.visibility[i],
            times_s=pop.times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=pop.x0_estimates[i],
            cfg=pukf_cfg,
            **kwargs,
        )
        q_scales.extend(float(rec["q_scale_used"]) for rec in recs)
    rmse = hifi._per_traj_observed_pos_rmse(pop.states, pred, pop.visibility, eval_start=11)
    q_arr = np.asarray(q_scales, dtype=np.float64)
    summary = {
        "observed_step_rmse_mean_m": float(np.nanmean(rmse)),
        "finite_trajectory_count": int(np.sum(np.isfinite(rmse))),
        "mean_q_scale": float(np.mean(q_arr)) if q_arr.size else float("nan"),
        "median_q_scale": float(np.median(q_arr)) if q_arr.size else float("nan"),
        "max_q_scale": float(np.max(q_arr)) if q_arr.size else float("nan"),
    }
    return rmse, summary


def _run_validation_classical(cfg: dict[str, Any], pop: Population) -> dict[str, float]:
    dataset_cfg = parse_dataset_config(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    meas_std = dataset_cfg.measurement_noise.std_vector
    stations = dataset_cfg.stations
    kwargs = _estimator_kwargs(cfg)
    configs = {
        "EKF": EKFConfig(
            q_pos_m=baseline_cfg.ekf.q_pos_m,
            q_vel_mps=baseline_cfg.ekf.q_vel_mps,
            init_pos_std_m=baseline_cfg.ekf.init_pos_std_m,
            init_vel_std_mps=baseline_cfg.ekf.init_vel_std_mps,
            gating_threshold=baseline_cfg.ekf.gating_threshold,
            angle_deweight_elev_cap_deg=getattr(
                baseline_cfg.ekf, "angle_deweight_elev_cap_deg", None
            ),
        ),
        "UKF": UKFConfig(
            q_pos_m=baseline_cfg.ukf.q_pos_m,
            q_vel_mps=baseline_cfg.ukf.q_vel_mps,
            init_pos_std_m=baseline_cfg.ukf.init_pos_std_m,
            init_vel_std_mps=baseline_cfg.ukf.init_vel_std_mps,
            alpha=baseline_cfg.ukf.alpha,
            beta=baseline_cfg.ukf.beta,
            kappa=baseline_cfg.ukf.kappa,
            angle_deweight_elev_cap_deg=getattr(
                baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
            ),
        ),
        "AUKF": AdaptiveUKFConfig(
            q_pos_m=baseline_cfg.aukf.q_pos_m,
            q_vel_mps=baseline_cfg.aukf.q_vel_mps,
            init_pos_std_m=baseline_cfg.aukf.init_pos_std_m,
            init_vel_std_mps=baseline_cfg.aukf.init_vel_std_mps,
            alpha=baseline_cfg.aukf.alpha,
            beta=baseline_cfg.aukf.beta,
            kappa=baseline_cfg.aukf.kappa,
            adapt_rate=baseline_cfg.aukf.adapt_rate,
            min_r_scale=baseline_cfg.aukf.min_r_scale,
            max_r_scale=baseline_cfg.aukf.max_r_scale,
            huber_kappa=baseline_cfg.aukf.huber_kappa,
            nis_soft_gate=baseline_cfg.aukf.nis_soft_gate,
            angle_deweight_elev_cap_deg=getattr(
                baseline_cfg.aukf, "angle_deweight_elev_cap_deg", None
            ),
        ),
    }
    means: dict[str, float] = {}
    for name, estimator_cfg in configs.items():
        pred = np.zeros_like(pop.states)
        for i in range(pop.states.shape[0]):
            common = dict(
                measurements=pop.measurements[i],
                visibility=pop.visibility[i],
                times_s=pop.times[i],
                stations=stations,
                meas_std_vector=meas_std,
                x0_est=pop.x0_estimates[i],
                cfg=estimator_cfg,
                **kwargs,
            )
            if name == "EKF":
                pred[i], _ = run_ekf(**common)
            elif name == "UKF":
                pred[i], _ = run_ukf(**common)
            else:
                pred[i], _ = run_adaptive_ukf(**common)
        rmse = hifi._per_traj_observed_pos_rmse(pop.states, pred, pop.visibility, eval_start=11)
        means[name] = float(np.nanmean(rmse))
    return means


def _paired_summary(candidate: np.ndarray, baseline: np.ndarray, *, seed: int) -> dict[str, Any]:
    if candidate.shape != baseline.shape:
        raise AssertionError(
            f"paired arrays must have identical shape, got {candidate.shape} and {baseline.shape}"
        )
    diffs = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    finite = diffs[np.isfinite(diffs)]
    if finite.size <= 0:
        raise AssertionError("paired comparison has zero finite trajectory denominators")
    mean, lo, hi = hifi._paired_bootstrap_ci(diffs, n_boot=5000, seed=seed)
    return {
        "mean_diff_m": mean,
        "ci_lo_m": lo,
        "ci_hi_m": hi,
        "n_paired": int(finite.size),
        "candidate_better_count": int(np.sum(finite < 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--validation-seed", type=int, default=20260519)
    parser.add_argument("--test-seed", type=int, default=20260520)
    parser.add_argument("--validation-trajectories", type=int, default=16)
    parser.add_argument("--test-trajectories", type=int, default=48)
    parser.add_argument("--epoch-unix", type=float, default=1_736_640_000.0)
    parser.add_argument(
        "--test-classical-csv",
        default="results/hifi_force_mismatch/hifi_force_mismatch.csv",
    )
    parser.add_argument(
        "--test-reference-json",
        default="results/hifi_force_mismatch/hifi_force_mismatch.json",
    )
    parser.add_argument(
        "--output-json",
        default="results/pukf_tuning_sensitivity/pukf_hifi_grid_sensitivity.json",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)
    t0 = time.perf_counter()
    validation_pop, validation_pop_meta = _make_population(
        cfg=cfg,
        seed=args.validation_seed,
        n_trajectories=args.validation_trajectories,
        epoch_unix=args.epoch_unix,
    )
    validation_classical = _run_validation_classical(cfg, validation_pop)

    grid_rows: list[dict[str, Any]] = []
    selected: tuple[GridPoint, dict[str, float]] | None = None
    for point in _grid():
        _, summary = _run_pukf(cfg=cfg, pop=validation_pop, point=point)
        row = {**asdict(point), **summary}
        grid_rows.append(row)
        if selected is None or summary["observed_step_rmse_mean_m"] < selected[1]["observed_step_rmse_mean_m"]:
            selected = (point, summary)
        print(
            json.dumps(
                {
                    "grid_point": point.label,
                    "validation_observed_step_rmse_m": summary["observed_step_rmse_mean_m"],
                }
            ),
            flush=True,
        )
    assert selected is not None
    selected_point, selected_validation = selected

    test_pop, test_pop_meta = _make_population(
        cfg=cfg,
        seed=args.test_seed,
        n_trajectories=args.test_trajectories,
        epoch_unix=args.epoch_unix,
    )
    selected_test_rmse, selected_test_summary = _run_pukf(
        cfg=cfg, pop=test_pop, point=selected_point
    )
    finite_selected = int(np.sum(np.isfinite(selected_test_rmse)))
    if finite_selected != int(selected_test_summary["finite_trajectory_count"]):
        raise AssertionError(
            "Selected-test finite trajectory count does not match the paired array denominator"
        )

    test_csv = Path(args.test_classical_csv)
    if not test_csv.exists():
        raise SystemExit(f"Missing held-out classical CSV: {test_csv}")
    df = pd.read_csv(test_csv)
    required_columns = [
        "trajectory_index",
        "EKF_observed_pos_rmse_m",
        "UKF_observed_pos_rmse_m",
        "AUKF_observed_pos_rmse_m",
        "PUKF_observed_pos_rmse_m",
    ]
    missing_columns = [name for name in required_columns if name not in df.columns]
    if missing_columns:
        raise AssertionError(
            f"Held-out classical CSV is missing required columns: {missing_columns}"
        )
    expected_n = int(test_pop.states.shape[0])
    if int(len(df)) != expected_n:
        raise AssertionError(
            f"Held-out classical CSV has {len(df)} rows, expected {expected_n} "
            "from regenerated deterministic population"
        )
    expected_index = np.arange(expected_n, dtype=np.int64)
    actual_index = df["trajectory_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_index, expected_index):
        raise AssertionError(
            "Held-out classical CSV trajectory_index is not the deterministic row order 0..n-1"
        )
    baseline_arrays = {
        "EKF": df["EKF_observed_pos_rmse_m"].to_numpy(dtype=np.float64),
        "UKF": df["UKF_observed_pos_rmse_m"].to_numpy(dtype=np.float64),
        "AUKF": df["AUKF_observed_pos_rmse_m"].to_numpy(dtype=np.float64),
        "PUKF_predeclared": df["PUKF_observed_pos_rmse_m"].to_numpy(dtype=np.float64),
    }
    heldout_means = {k: float(np.nanmean(v)) for k, v in baseline_arrays.items()}
    heldout_means["PUKF_validation_selected"] = selected_test_summary[
        "observed_step_rmse_mean_m"
    ]
    classical_names = ["EKF", "UKF", "AUKF"]
    best_classical = min(classical_names, key=lambda name: heldout_means[name])
    paired = {
        name: _paired_summary(selected_test_rmse, arr, seed=args.test_seed + i)
        for i, (name, arr) in enumerate(baseline_arrays.items())
    }
    gap = paired[best_classical]
    selected_positive = bool(
        heldout_means["PUKF_validation_selected"] < heldout_means[best_classical]
        and math.isfinite(gap["ci_hi_m"])
        and gap["ci_hi_m"] < 0.0
    )

    reference_json = Path(args.test_reference_json)
    reference_digest = _sha256_file(reference_json)
    reference_payload = (
        json.loads(reference_json.read_text(encoding="utf-8"))
        if reference_json.exists()
        else {}
    )
    reference_population = {
        "path": str(reference_json),
        "sha256": reference_digest,
        "rng_seed": reference_payload.get("rng_seed"),
        "n_trajectories": reference_payload.get("n_trajectories"),
        "steps": reference_payload.get("steps"),
        "dt_s": reference_payload.get("dt_s"),
        "epoch_unix": reference_payload.get("epoch_unix"),
        "eval_start_step": reference_payload.get("eval_start_step"),
    }
    reference_expectations = {
        "rng_seed": int(args.test_seed),
        "n_trajectories": int(args.test_trajectories),
        "steps": int(test_pop_meta["steps"]),
        "dt_s": float(test_pop_meta["dt_s"]),
        "epoch_unix": float(args.epoch_unix),
    }
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected in reference_expectations.items():
        actual = reference_payload.get(key)
        if isinstance(expected, float):
            ok = actual is not None and math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        else:
            ok = actual == expected
        if not ok:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise AssertionError(
            "Held-out reference JSON does not match regenerated deterministic "
            f"population metadata: {mismatches}"
        )
    csv_digest = _sha256_file(test_csv)
    payload: dict[str, Any] = {
        "schema_version": "pukf_hifi_tuning_sensitivity_v1",
        "analysis_label": "reviewer-requested PUKF tuning-comparability sensitivity",
        "analysis_scope": (
            "Post-hoc bounded validation-grid stress for PUKF on the higher-fidelity "
            "force-mismatch slice. It is a comparator-comparability sensitivity and "
            "does not replace the predeclared PUKF decision record."
        ),
        "validation": {
            "seed": int(args.validation_seed),
            "n_trajectories": int(args.validation_trajectories),
            "classical_observed_step_rmse_mean_m": validation_classical,
            "grid_points": grid_rows,
            "selected_grid_point": asdict(selected_point),
            "selected_validation_summary": selected_validation,
        },
        "heldout_test": {
            "seed": int(args.test_seed),
            "n_trajectories": int(args.test_trajectories),
            "reference_classical_csv": str(test_csv),
            "reference_classical_csv_sha256": csv_digest,
            "reference_result_json": str(reference_json),
            "reference_result_sha256": reference_digest,
            "reference_population_metadata": reference_population,
            "population_alignment": {
                "status": "pass",
                "trajectory_ids_available": False,
                "row_order_assertion": (
                    "CSV trajectory_index equals 0..n-1 and row count equals "
                    "the regenerated deterministic held-out population."
                ),
                "generation_protocol_assertion": (
                    "Reference JSON rng_seed, n_trajectories, steps, dt_s, and "
                    "epoch_unix match the regenerated population metadata."
                ),
                "csv_rows": int(len(df)),
                "regenerated_population_trajectories": expected_n,
                "finite_selected_pukf_trajectories": finite_selected,
                "paired_denominator_assertion": (
                    "Every paired comparator summary asserts a nonzero finite "
                    "trajectory denominator before bootstrapping."
                ),
            },
            "observed_step_rmse_mean_m": heldout_means,
            "best_classical": best_classical,
            "pukf_selected_minus_comparator_paired_m": paired,
            "validation_selected_pukf_positive_vs_best_classical": selected_positive,
            "qualitative_conclusion": (
                "validation-selected PUKF is a learned-positive-style comparator win"
                if selected_positive
                else "validation-selected PUKF does not beat the best classical comparator"
            ),
        },
        "population": {
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "validation": validation_pop_meta,
            "heldout_test": test_pop_meta,
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["heldout_test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
