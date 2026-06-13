"""Loop-25 M1: independent-realization validation of the DBAR diagnostic.

The loop-24 DBAR (Dynamics-Bias Adaptation-Risk) indicator was exhibited on
three hand-constructed regimes, one realization each. A reviewer correctly
objected that n=3 single-realization points cannot support either "validated"
or "insensitive to the exact threshold". This script replaces the n=3
exhibition with a genuine sampling design.

For ``--realizations-per-family`` independent realizations in each of three
regime families it:

* draws an independent random seed (fresh trajectory population and an
  independent measurement-noise / initialisation draw),
* for the two non-nominal families draws an independent random *severity*
  from a predeclared range (so each realization is a distinct regime, not a
  re-run of one fixed regime),
* runs the EKF, the fixed-noise UKF, and the instrumented adaptive UKF
  in-process on that realization (no multiprocessing baseline pool; the
  per-trajectory filter loops are the only cost),
* computes the predeclared DBAR statistics ``R_eff`` and
  ``rho_NIS = median(adaptive R-only NIS) / median(fixed-noise UKF R-only NIS)``,
* records, from the realization itself, whether R-adaptation was actually
  counterproductive (AUKF observed-step position RMSE materially worse than the
  best of the three filters, predeclared margin 5 %), and
* applies the predeclared rule ``DBAR fires iff R_eff > TAU_R AND
  rho_NIS >= TAU_RHO``.

Regime families (design label; ground truth is *measured per realization*, not
assumed from the family):

* ``nominal``       -- perfect shared model, no injected bias.
* ``meas_stress``   -- truth measurement noise inflated by a random factor; the
                       estimator keeps the nominal noise model. R-adaptation is
                       the *correct* response here, so DBAR must not fire even
                       though R_eff is large -- the decisive control.
* ``dynamics_bias`` -- truth dynamics (drag area-to-mass, reference density,
                       SRP, process noise) randomly perturbed while the
                       estimator keeps the nominal compact model.

The diagnostic is scored by its classification accuracy against the *measured*
outcome over all independent realizations, and a threshold-sensitivity grid
reports how that accuracy varies as (TAU_R, TAU_RHO) are swept around the
predeclared (1.5, 1.5), so the robustness claim is backed by evidence rather
than asserted.

No model is trained and no cached artifact is consumed; every number is scored
on a freshly generated independent realization.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - entrypoint dependent
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from gnn_state_estimation.evaluation import generate_noisy_init, parse_baseline_config
from gnn_state_estimation.filters.ekf import run_ekf
from gnn_state_estimation.filters.ukf import (
    predicted_innovation_nis,
    run_adaptive_ukf_instrumented,
    run_ukf,
)
from gnn_state_estimation.simulation import generate_dataset, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.classification_stats import (
    binary_classification_report,
    required_n_one_proportion,
)
from gnn_state_estimation.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "adaptation_risk_diagnostic"
OUT_PATH = OUT_DIR / "dbar_independent_sweep.json"

# Predeclared DBAR rule (identical to scripts/build_adaptation_risk_diagnostic.py;
# round thresholds fixed a priori, not fitted to outcomes).
TAU_R = 1.5
TAU_RHO = 1.5
# Predeclared materiality margin: R-adaptation counts as having been
# counterproductive only when the AUKF observed-step RMSE exceeds the best
# filter's by more than this fraction.
MATERIALITY_MARGIN = 0.05

# Predeclared severity ranges for the two non-nominal families. Each
# realization draws one severity uniformly from these ranges, so the family is
# a *continuum* of independent regimes spanning mild to severe.
MEAS_STRESS_FACTOR = (1.5, 3.5)          # multiplies the angular/range/rr noise
MEAS_STRESS_OUTLIER_PROB = (0.02, 0.06)
MEAS_STRESS_OUTLIER_SCALE = (6.0, 12.0)

DYN_BC_FACTOR = (1.8, 3.0)               # x nominal ballistic_coeff (0.018)
DYN_PROC_NOISE = (0.10, 0.60)            # absolute process_noise_std
DYN_RHO_FACTOR = (1.3, 2.2)              # x nominal drag_rho_ref
DYN_SRP_AM_FACTOR = (1.8, 3.5)           # x nominal srp_area_to_mass
DYN_SRP_CR_FACTOR = (1.10, 1.30)         # x nominal srp_cr

FAMILIES = ("nominal", "meas_stress", "dynamics_bias")


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def sample_severity(family: str, rng: np.random.Generator) -> dict[str, Any]:
    """Independent random severity for one realization of a family."""
    if family == "nominal":
        return {}
    if family == "meas_stress":
        f = float(rng.uniform(*MEAS_STRESS_FACTOR))
        return {
            "kind": "meas_stress",
            "noise_factor": f,
            "outlier_prob": float(rng.uniform(*MEAS_STRESS_OUTLIER_PROB)),
            "outlier_scale": float(rng.uniform(*MEAS_STRESS_OUTLIER_SCALE)),
        }
    # dynamics_bias
    return {
        "kind": "dynamics_bias",
        "bc_factor": float(rng.uniform(*DYN_BC_FACTOR)),
        "process_noise_std": float(rng.uniform(*DYN_PROC_NOISE)),
        "rho_factor": float(rng.uniform(*DYN_RHO_FACTOR)),
        "srp_am_factor": float(rng.uniform(*DYN_SRP_AM_FACTOR)),
        "srp_cr_factor": float(rng.uniform(*DYN_SRP_CR_FACTOR)),
    }


def build_truth_sim(base_sim: dict[str, Any], severity: dict[str, Any]) -> dict[str, Any]:
    """Truth-side simulation config for a realization. The estimator always
    keeps ``base_sim`` (the nominal compact model)."""
    kind = severity.get("kind")
    if kind == "meas_stress":
        mn = base_sim["measurement_noise"]
        return deep_update(
            base_sim,
            {
                "measurement_noise": {
                    "range_std_m": mn["range_std_m"] * severity["noise_factor"],
                    "az_std_deg": mn["az_std_deg"] * severity["noise_factor"],
                    "el_std_deg": mn["el_std_deg"] * severity["noise_factor"],
                    "range_rate_std_mps": mn["range_rate_std_mps"]
                    * severity["noise_factor"],
                    "outlier_prob": severity["outlier_prob"],
                    "outlier_scale": severity["outlier_scale"],
                }
            },
        )
    if kind == "dynamics_bias":
        dyn = base_sim["dynamics"]
        return deep_update(
            base_sim,
            {
                "dynamics": {
                    "ballistic_coeff_m2_per_kg": dyn["ballistic_coeff_m2_per_kg"]
                    * severity["bc_factor"],
                    "process_noise_std": severity["process_noise_std"],
                    "drag_rho_ref": dyn["drag_rho_ref"] * severity["rho_factor"],
                    "srp_area_to_mass_m2_per_kg": dyn["srp_area_to_mass_m2_per_kg"]
                    * severity["srp_am_factor"],
                    "srp_cr": dyn["srp_cr"] * severity["srp_cr_factor"],
                }
            },
        )
    return copy.deepcopy(base_sim)


def observed_step_pos_rmse(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> float:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    err = states[:, eval_start:][observed, :3] - preds[:, eval_start:][observed, :3]
    if err.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def median_r_only_nis(
    preds: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times: np.ndarray,
    stations,
    meas_std_vector: np.ndarray,
) -> float:
    vals: list[float] = []
    for i in range(preds.shape[0]):
        for entry in predicted_innovation_nis(
            pred_states=preds[i],
            measurements=measurements[i],
            visibility=visibility[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std_vector,
        ):
            vals.append(entry["nis_r"])
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def run_realization(
    base_sim: dict[str, Any],
    severity: dict[str, Any],
    baseline_cfg,
    eval_start: int,
    n_traj: int,
    seed: int,
) -> dict[str, Any]:
    """Generate one independent realization and score the DBAR statistics."""
    truth_sim = build_truth_sim(base_sim, severity)
    truth_dc = parse_dataset_config(truth_sim)
    est_dc = parse_dataset_config(base_sim)  # estimator keeps the nominal model
    dyn = est_dc.dynamics
    meas_std = est_dc.measurement_noise.std_vector
    stations = est_dc.stations

    data = generate_dataset(truth_dc, n_traj, seed=seed)
    states = data["states"]
    meas = data["measurements"]
    vis = data["visibility"]
    times = data["times"]

    rng = np.random.default_rng(seed + 7)
    x0 = np.stack(
        [
            generate_noisy_init(
                states[i, 0], rng, baseline_cfg.init_pos_std_m, baseline_cfg.init_vel_std_mps
            )
            for i in range(states.shape[0])
        ]
    )

    dyn_kwargs = dict(
        ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
        drag_rho_ref=dyn.drag_rho_ref,
        drag_h_ref_m=dyn.drag_h_ref_m,
        drag_scale_height_m=dyn.drag_scale_height_m,
        enable_third_body=dyn.enable_third_body,
        enable_srp=dyn.enable_srp,
        srp_area_to_mass_m2_per_kg=dyn.srp_area_to_mass_m2_per_kg,
        srp_cr=dyn.srp_cr,
        sun_initial_phase_rad=dyn.sun_initial_phase_rad,
        moon_initial_phase_rad=dyn.moon_initial_phase_rad,
    )

    ekf_pred = np.zeros_like(states)
    ukf_pred = np.zeros_like(states)
    aukf_pred = np.zeros_like(states)
    r_eff_vals: list[float] = []

    for i in range(states.shape[0]):
        ekf_pred[i], _ = run_ekf(
            measurements=meas[i],
            visibility=vis[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0[i],
            cfg=baseline_cfg.ekf,
            **dyn_kwargs,
        )
        ukf_pred[i], _ = run_ukf(
            measurements=meas[i],
            visibility=vis[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0[i],
            cfg=baseline_cfg.ukf,
            **dyn_kwargs,
        )
        x_hist, _p, records = run_adaptive_ukf_instrumented(
            measurements=meas[i],
            visibility=vis[i],
            times_s=times[i],
            stations=stations,
            meas_std_vector=meas_std,
            x0_est=x0[i],
            cfg=baseline_cfg.aukf,
            **dyn_kwargs,
        )
        aukf_pred[i] = x_hist
        for rec in records:
            r_eff_vals.append(float(rec["r_eff_scale_mean"]))

    r_eff = float(np.mean(r_eff_vals)) if r_eff_vals else float("nan")
    med_aukf = median_r_only_nis(aukf_pred, meas, vis, times, stations, meas_std)
    med_ukf = median_r_only_nis(ukf_pred, meas, vis, times, stations, meas_std)
    rho_nis = med_aukf / med_ukf if med_ukf else float("nan")

    obs = {
        "EKF": observed_step_pos_rmse(states, ekf_pred, vis, eval_start),
        "UKF": observed_step_pos_rmse(states, ukf_pred, vis, eval_start),
        "AUKF": observed_step_pos_rmse(states, aukf_pred, vis, eval_start),
    }
    best_method = min(obs, key=obs.get)
    best_rmse = obs[best_method]
    # Secondary, best-of-three excess (continuity with the loop-24 n=3 table).
    aukf_excess_vs_best = (
        (obs["AUKF"] - best_rmse) / best_rmse if best_rmse else float("nan")
    )
    aukf_worst_of_three = bool(aukf_excess_vs_best > MATERIALITY_MARGIN)
    # Primary, mechanism-aligned ground truth: innovation-consistency
    # R-adaptation was *counterproductive* on this realization iff turning the
    # adaptation on (fixed-noise UKF -> adaptive UKF, same sigma-point filter)
    # made the observed-step RMSE materially worse. This is exactly what the
    # DBAR statistic is constructed to predict, and it removes the near-tie
    # EKF confound that contaminates a three-way min in the nominal regime.
    aukf_vs_twin = (
        (obs["AUKF"] - obs["UKF"]) / obs["UKF"] if obs["UKF"] else float("nan")
    )
    adaptation_counterproductive = bool(aukf_vs_twin > MATERIALITY_MARGIN)
    fired = bool((r_eff > TAU_R) and (rho_nis >= TAU_RHO))

    return {
        "seed": int(seed),
        "n_trajectories": int(states.shape[0]),
        "severity": severity,
        "r_eff": round(r_eff, 4),
        "median_r_only_nis_aukf": round(med_aukf, 4),
        "median_r_only_nis_ukf": round(med_ukf, 4),
        "rho_nis": round(rho_nis, 4),
        "observed_step_pos_rmse_m": {k: round(v, 2) for k, v in obs.items()},
        "best_observed_method": best_method,
        "aukf_excess_vs_best_pct": round(100.0 * aukf_excess_vs_best, 2),
        "aukf_worst_of_three": aukf_worst_of_three,
        "aukf_vs_fixed_twin_pct": round(100.0 * aukf_vs_twin, 2),
        "adaptation_counterproductive": adaptation_counterproductive,
        "dbar_fired": fired,
        "dbar_correct": bool(fired == adaptation_counterproductive),
        "dbar_correct_worst_of_three": bool(fired == aukf_worst_of_three),
    }


_WORKER_CTX: dict[str, Any] = {}


def _load_ctx(config_path: str) -> dict[str, Any]:
    """Per-process cached parse of the experiment config.

    Parsing is deterministic and identical to the serial path; caching it
    just avoids re-reading the YAML for every realization handled by a
    worker. The scientific inputs (base_sim, baseline_cfg, eval_start) are
    byte-for-byte the same objects the serial loop builds."""
    ctx = _WORKER_CTX.get(config_path)
    if ctx is None:
        cfg = load_yaml(Path(config_path))
        base_sim = copy.deepcopy(cfg["simulation"])
        baseline_cfg = parse_baseline_config(cfg["baselines"])
        train_cfg = parse_train_config(cfg["training"])
        eval_start = max(int(train_cfg.window_size) - 1, 0)
        ctx = {
            "base_sim": base_sim,
            "baseline_cfg": baseline_cfg,
            "eval_start": eval_start,
        }
        _WORKER_CTX[config_path] = ctx
    return ctx


def _worker(task: tuple[str, str, int, dict[str, Any], int]) -> dict[str, Any]:
    """Process-pool entry point for one independent realization.

    ``task = (config_path, family, seed, severity, n_traj)``. Realizations
    are independent and each is fully determined by its seed and severity,
    so dispatching them to a pool yields exactly the result the serial loop
    would; only the execution order is parallel."""
    config_path, family, seed, severity, n_traj = task
    ctx = _load_ctx(config_path)
    row = run_realization(
        ctx["base_sim"],
        severity,
        ctx["baseline_cfg"],
        ctx["eval_start"],
        n_traj,
        seed,
    )
    row["family"] = family
    return row


def out_of_sample_predeclared(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the FROZEN predeclared rule to an independent data partition.

    The thresholds are fixed a priori and are *never* selected from these
    data, so the only honest generalisation question is whether the predeclared
    rule's accuracy holds on a partition disjoint from any inspection. The rows
    are split by the parity of their realisation seed (an arbitrary,
    outcome-independent partition); the predeclared (TAU_R, TAU_RHO) rule is
    evaluated on each half separately. Nothing is tuned -- this addresses the
    predeclaration-vs-in-sample-optimality concern by removing any reliance on
    a grid maximum.
    """

    def _confusion(part: list[dict[str, Any]]) -> dict[str, Any]:
        tp = sum(
            1
            for r in part
            if r["dbar_fired"] and r["adaptation_counterproductive"]
        )
        tn = sum(
            1
            for r in part
            if not r["dbar_fired"]
            and not r["adaptation_counterproductive"]
        )
        fp = sum(
            1
            for r in part
            if r["dbar_fired"]
            and not r["adaptation_counterproductive"]
        )
        fn = sum(
            1
            for r in part
            if not r["dbar_fired"]
            and r["adaptation_counterproductive"]
        )
        n = len(part)
        return {
            "n": n,
            "confusion": {
                "true_fire": tp,
                "true_no_fire": tn,
                "false_fire": fp,
                "false_no_fire": fn,
            },
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "report": binary_classification_report(tp, tn, fp, fn),
        }

    half_a = [r for r in rows if int(r.get("seed", 0)) % 2 == 0]
    half_b = [r for r in rows if int(r.get("seed", 0)) % 2 == 1]
    return {
        "partition": "parity of the independent realisation seed (no tuning)",
        "threshold_source": "predeclared a priori (TAU_R=1.5, TAU_RHO=1.5)",
        "split_even_seed": _confusion(half_a),
        "split_odd_seed": _confusion(half_b),
        "interpretation": (
            "The predeclared rule is applied unchanged to two disjoint, "
            "outcome-independent halves; no threshold is selected from the "
            "data, so the per-half accuracies are an out-of-sample check of "
            "the fixed rule rather than of a grid optimum."
        ),
    }


def threshold_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classification accuracy over a grid of (TAU_R, TAU_RHO) around the
    predeclared (1.5, 1.5).

    This grid is reported for transparency only. The thresholds are fixed a
    priori and are *not* selected from the grid: the predeclared point is
    deliberately reported alongside the grid argmax (and the gap to it) so the
    reader can see the rule was not cherry-picked to the in-sample optimum.
    The robustness evidence is the predeclared rule's accuracy and its
    out-of-sample split (:func:`out_of_sample_predeclared`), not grid
    optimality."""
    tau_r_grid = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    tau_rho_grid = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    n = len(rows)
    grid = []
    accs = []
    for tr in tau_r_grid:
        for trho in tau_rho_grid:
            correct = 0
            for row in rows:
                fired = (row["r_eff"] > tr) and (row["rho_nis"] >= trho)
                if fired == row["adaptation_counterproductive"]:
                    correct += 1
            acc = correct / n if n else float("nan")
            accs.append(acc)
            grid.append(
                {"tau_r": tr, "tau_rho": trho, "accuracy": round(acc, 4)}
            )
    accs_arr = np.asarray(accs, dtype=np.float64)
    base_acc = next(
        g["accuracy"] for g in grid if g["tau_r"] == TAU_R and g["tau_rho"] == TAU_RHO
    )
    # Band over which accuracy stays within 0.05 of the predeclared-threshold
    # accuracy.
    stable = [g for g in grid if abs(g["accuracy"] - base_acc) <= 0.05]
    grid_argmax = max(grid, key=lambda g: g["accuracy"])
    return {
        "tau_r_grid": tau_r_grid,
        "tau_rho_grid": tau_rho_grid,
        "predeclared_accuracy": round(base_acc, 4),
        "grid_min_accuracy": round(float(accs_arr.min()), 4),
        "grid_max_accuracy": round(float(accs_arr.max()), 4),
        "grid_mean_accuracy": round(float(accs_arr.mean()), 4),
        "grid_argmax_thresholds": {
            "tau_r": grid_argmax["tau_r"],
            "tau_rho": grid_argmax["tau_rho"],
        },
        "grid_argmax_accuracy": round(float(grid_argmax["accuracy"]), 4),
        "predeclared_is_grid_argmax": bool(
            grid_argmax["tau_r"] == TAU_R
            and grid_argmax["tau_rho"] == TAU_RHO
        ),
        "predeclared_minus_grid_argmax_accuracy": round(
            base_acc - float(grid_argmax["accuracy"]), 4
        ),
        "framing": (
            "Transparency grid only; thresholds are predeclared a priori and "
            "are NOT selected as the grid optimum. The predeclared point is "
            "reported with its gap to the grid argmax so the reader can verify "
            "the rule was not cherry-picked; robustness rests on the "
            "predeclared rule and its out-of-sample split, not grid "
            "optimality."
        ),
        "n_grid_points": len(grid),
        "n_grid_points_within_0p05_of_predeclared": len(stable),
        "tau_r_stable_min": min((g["tau_r"] for g in stable), default=None),
        "tau_r_stable_max": max((g["tau_r"] for g in stable), default=None),
        "tau_rho_stable_min": min((g["tau_rho"] for g in stable), default=None),
        "tau_rho_stable_max": max((g["tau_rho"] for g in stable), default=None),
        "grid": grid,
    }


def build_result(
    rows: list[dict[str, Any]], K: int, N: int, elapsed: float
) -> dict[str, Any]:
    """Assemble the full artifact from per-realization records.

    Pure function of ``rows`` (each row already carries the predeclared-rule
    decision, the measured counterproductivity label, the seed, and family),
    so it reproduces deterministically and can be re-derived from a stored
    sweep without re-running the expensive realisation loop.
    """
    n = len(rows)
    n_correct = sum(1 for r in rows if r["dbar_correct"])
    fired = [r for r in rows if r["dbar_fired"]]
    no_fire = [r for r in rows if not r["dbar_fired"]]

    by_family = {}
    for fam in FAMILIES:
        fr = [r for r in rows if r["family"] == fam]
        by_family[fam] = {
            "n": len(fr),
            "n_dbar_fired": sum(1 for r in fr if r["dbar_fired"]),
            "n_adaptation_counterproductive": sum(
                1 for r in fr if r["adaptation_counterproductive"]
            ),
            "n_correct": sum(1 for r in fr if r["dbar_correct"]),
            "accuracy": round(
                sum(1 for r in fr if r["dbar_correct"]) / len(fr), 4
            )
            if fr
            else float("nan"),
        }

    rho_fire_min = min((r["rho_nis"] for r in fired), default=float("nan"))
    rho_nofire_max = max(
        (r["rho_nis"] for r in no_fire), default=float("nan")
    )

    # Confusion against the measured "R-adaptation was counterproductive"
    # label.
    tp = sum(
        1
        for r in rows
        if r["dbar_fired"] and r["adaptation_counterproductive"]
    )
    tn = sum(
        1
        for r in rows
        if not r["dbar_fired"] and not r["adaptation_counterproductive"]
    )
    fp = sum(
        1
        for r in rows
        if r["dbar_fired"] and not r["adaptation_counterproductive"]
    )
    fn = sum(
        1
        for r in rows
        if not r["dbar_fired"] and r["adaptation_counterproductive"]
    )
    n_correct_worst_of_three = sum(
        1 for r in rows if r["dbar_correct_worst_of_three"]
    )
    report = binary_classification_report(tp, tn, fp, fn)

    return {
        "status": "completed",
        "schema_version": "dbar_independent_sweep_v1",
        "diagnostic_name": "DBAR (Dynamics-Bias Adaptation-Risk indicator)",
        "statistic": "rho_NIS = median(adaptive R-only NIS) / median(fixed-noise UKF R-only NIS)",
        "predeclared_thresholds": {"tau_r_eff": TAU_R, "tau_rho_nis": TAU_RHO},
        "materiality_margin": MATERIALITY_MARGIN,
        "rule": "DBAR fires iff R_eff > tau_r_eff AND rho_NIS >= tau_rho_nis",
        "ground_truth": (
            "measured per realization: innovation-consistency R-adaptation "
            "was counterproductive iff the adaptive UKF observed-step "
            "position RMSE exceeds its own fixed-noise twin (the UKF, same "
            "sigma-point filter) by more than the predeclared materiality "
            "margin. This is exactly the effect the DBAR statistic is "
            "constructed to predict. A secondary best-of-{EKF,UKF,AUKF} label "
            "is also reported for continuity with the three-regime "
            "exhibition."
        ),
        "design": {
            "families": list(FAMILIES),
            "realizations_per_family": K,
            "trajectories_per_realization": N,
            "n_independent_realizations": n,
            "independent": (
                "each realization is an independently seeded trajectory "
                "population with an independent measurement-noise / "
                "initialisation draw; non-nominal families also draw an "
                "independent random severity from a predeclared range"
            ),
            "severity_ranges": {
                "meas_stress_noise_factor": MEAS_STRESS_FACTOR,
                "meas_stress_outlier_prob": MEAS_STRESS_OUTLIER_PROB,
                "meas_stress_outlier_scale": MEAS_STRESS_OUTLIER_SCALE,
                "dyn_bias_bc_factor": DYN_BC_FACTOR,
                "dyn_bias_process_noise_std": DYN_PROC_NOISE,
                "dyn_bias_rho_factor": DYN_RHO_FACTOR,
                "dyn_bias_srp_am_factor": DYN_SRP_AM_FACTOR,
                "dyn_bias_srp_cr_factor": DYN_SRP_CR_FACTOR,
            },
        },
        "summary": {
            "n_independent_realizations": n,
            "n_correct": n_correct,
            "classification_accuracy": round(n_correct / n, 4)
            if n
            else float("nan"),
            "classification_accuracy_worst_of_three": round(
                n_correct_worst_of_three / n, 4
            )
            if n
            else float("nan"),
            "confusion": {
                "true_fire": tp,
                "true_no_fire": tn,
                "false_fire": fp,
                "false_no_fire": fn,
            },
            "min_rho_nis_among_fired": round(rho_fire_min, 4),
            "max_rho_nis_among_no_fire": round(rho_nofire_max, 4),
            "separation_margin_rho_nis": round(
                rho_fire_min - rho_nofire_max, 4
            ),
            "classification_report": report,
            "no_information_baseline": report["no_information"],
            "incremental_accuracy_over_majority_class": report[
                "no_information"
            ]["accuracy_minus_majority"],
            "by_family": by_family,
        },
        "out_of_sample_predeclared": out_of_sample_predeclared(rows),
        "threshold_sensitivity": threshold_sensitivity(rows),
        "runtime_s": round(elapsed, 1),
        "realizations": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--realizations-per-family", type=int, default=12)
    ap.add_argument("--trajectories", type=int, default=14)
    ap.add_argument("--base-seed", type=int, default=70000)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent realizations are dispatched to this many worker "
        "processes. Each realization is fully determined by its seed and "
        "severity, so the artifact is byte-identical to the serial "
        "(--workers 1) path regardless of completion order; only wall-clock "
        "time changes. 0/1 keeps the serial loop.",
    )
    ap.add_argument(
        "--done-sentinel",
        default=None,
        help="Optional path; an empty file is written here only after the "
        "artifact has been fully written, so a background launcher can wait "
        "on file existence instead of polling a buffered stdout log.",
    )
    ap.add_argument(
        "--reprocess",
        action="store_true",
        help="Re-derive the summary, classification report, out-of-sample "
        "split, and threshold grid from the per-realization records already "
        "stored in the artifact, without re-running the realisation loop. "
        "Deterministic: the stored records were produced with fixed seeds.",
    )
    args = ap.parse_args()

    if args.reprocess:
        prev = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        rows = prev["realizations"]
        K = int(prev["design"]["realizations_per_family"])
        N = int(prev["design"]["trajectories_per_realization"])
        elapsed = float(prev.get("runtime_s", 0.0))
        result = build_result(rows, K, N, elapsed)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result["summary"], indent=2))
        print(
            f"\nreprocessed {OUT_PATH.relative_to(ROOT)} "
            f"({len(rows)} stored realizations; no sweep re-run)"
        )
        return 0

    cfg = load_yaml(Path(args.config))
    base_sim = copy.deepcopy(cfg["simulation"])
    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    K = int(args.realizations_per_family)
    N = int(args.trajectories)

    # Precompute the full (family, seed, severity) task list by replaying the
    # SAME per-family RNG draw order as the serial loop. Scaling K only extends
    # each family's deterministic sequence, so the first realizations remain
    # identical to any smaller run. The task order also fixes the artifact's
    # row order independently of pool completion order.
    tasks: list[tuple[str, str, int, dict[str, Any], int]] = []
    for fam_idx, family in enumerate(FAMILIES):
        sev_rng = np.random.default_rng(args.base_seed + 100 * (fam_idx + 1))
        for r in range(K):
            seed = args.base_seed + 1000 * (fam_idx + 1) + r
            severity = sample_severity(family, sev_rng)
            tasks.append((str(args.config), family, seed, severity, N))

    workers = max(1, int(args.workers))
    n_tasks = len(tasks)
    rows_by_seed: dict[int, dict[str, Any]] = {}
    done = 0
    start = time.perf_counter()

    def _log(row: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        print(
            f"[{row['family']} {done}/{n_tasks}] seed={row['seed']} "
            f"R_eff={row['r_eff']:.2f} rho={row['rho_nis']:.2f} "
            f"fired={row['dbar_fired']} "
            f"adapt_bad={row['adaptation_counterproductive']} "
            f"(twin {row['aukf_vs_fixed_twin_pct']:+.1f}%) "
            f"correct={row['dbar_correct']}",
            flush=True,
        )

    if workers <= 1:
        for task in tasks:
            row = _worker(task)
            rows_by_seed[row["seed"]] = row
            _log(row)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for row in ex.map(_worker, tasks, chunksize=1):
                rows_by_seed[row["seed"]] = row
                _log(row)

    # Reassemble in the deterministic task order so the artifact is
    # byte-identical to the serial path regardless of completion order.
    rows: list[dict[str, Any]] = [
        rows_by_seed[task[2]] for task in tasks
    ]
    elapsed = time.perf_counter() - start

    result = build_result(rows, K, N, elapsed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["threshold_sensitivity"], indent=2)[:1200])
    print(
        f"\nwrote {OUT_PATH.relative_to(ROOT)} "
        f"({len(rows)} independent realizations)"
    )
    if args.done_sentinel:
        sentinel = Path(args.done_sentinel)
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(
            f"completed {len(rows)} realizations in {elapsed:.1f}s\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
