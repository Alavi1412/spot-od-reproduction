#!/usr/bin/env python
"""Validation-selected EnKF inflation audit.

This is a bounded internal audit for the reviewer concern that the paper only
reported an untuned EnKF. It tunes exactly one EnKF hyperparameter,
multiplicative forecast-anomaly inflation, on validation splits only
(``val`` and ``stress_val``). The selected inflation is then frozen and scored
once on the force-model-mismatch, nominal, and stress test splits against the
cached EKF/UKF/AUKF priors on the same observed-step position-RMSE endpoint.

The audit is deliberately narrow: it does not tune localization, ensemble
size, process noise, initial covariance, particle filters, Gaussian-mixture
filters, or any operational real-data OD procedure.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon  # type: ignore[import-untyped]

try:
    from run_enkf_comparator import (
        _deep_update,
        _paired_bootstrap_ci,
        _per_traj_observed_pos_rmse,
    )
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts.run_enkf_comparator import (
        _deep_update,
        _paired_bootstrap_ci,
        _per_traj_observed_pos_rmse,
    )

from gnn_state_estimation.dataset import load_dataset_npz
from gnn_state_estimation.evaluation import parse_baseline_config
from gnn_state_estimation.filters import EnKFConfig, run_enkf
from gnn_state_estimation.scenarios import estimator_sim_config
from gnn_state_estimation.simulation import DatasetConfig, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml


CANDIDATE_GRID = (1.00, 1.03, 1.06, 1.10)
VALIDATION_SPLITS = ("val", "stress_val")
TEST_SPLITS = (
    ("force_model_mismatch_test", "Force mismatch (primary)"),
    ("test", "Nominal"),
    ("stress_test", "Stress"),
)
SCENARIO_LABELS = {
    "val": "Nominal validation",
    "stress_val": "Stress validation",
    "force_model_mismatch_test": "Force mismatch (primary)",
    "test": "Nominal",
    "stress_test": "Stress",
}
SPLIT_SEED_OFFSETS = {
    "val": 10_000,
    "stress_val": 20_000,
    "force_model_mismatch_test": 30_000,
    "test": 40_000,
    "stress_test": 50_000,
}


def _resolve_estimator_sim_for_split(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    """Estimator-side simulation config for validation and test splits.

    This follows ``scripts/run_enkf_comparator.py`` for test scenarios and
    extends the stress override convention to ``stress_val`` so validation
    scoring uses the same stress-side measurement model as ``stress_test``.
    """
    sim_cfg = copy.deepcopy(cfg["simulation"])
    if split in {"stress_train", "stress_val", "stress_test"}:
        sim_cfg = _deep_update(sim_cfg, cfg.get("stress_simulation_overrides", {}))
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(split)
    if scenario_cfg is None:
        return sim_cfg
    return estimator_sim_config(sim_cfg, scenario_cfg)


def _mean_count(values: np.ndarray) -> tuple[float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), 0
    return float(np.mean(finite)), int(finite.size)


def _one_sided_wilcoxon_candidate_better(diffs: np.ndarray) -> float:
    finite = np.asarray(diffs, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or np.allclose(finite, 0.0):
        return float("nan")
    try:
        return float(wilcoxon(finite, alternative="less", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _all_method_finite_mask(per_traj: dict[str, np.ndarray]) -> np.ndarray:
    mask: np.ndarray | None = None
    for values in per_traj.values():
        this = np.isfinite(values)
        mask = this if mask is None else (mask & this)
    if mask is None:
        return np.zeros(0, dtype=bool)
    return mask


def _run_enkf_split(
    cfg: dict[str, Any],
    split: str,
    th: dict[str, Any],
    inflation: float,
    eval_start: int,
    baseline_cfg: Any,
    traj_limit: int,
) -> dict[str, Any]:
    est_sim = _resolve_estimator_sim_for_split(cfg, split)
    dataset_cfg: DatasetConfig = parse_dataset_config(est_sim)
    dyn = dataset_cfg.dynamics
    stations = dataset_cfg.stations
    meas_std = dataset_cfg.measurement_noise.std_vector

    data_path = Path(cfg["data"]["output_dir"]) / f"{split}.npz"
    arrays = load_dataset_npz(data_path)
    if arrays.x0_estimates is None:
        raise ValueError(f"{data_path} missing x0_estimates")

    states = arrays.states
    meas = arrays.measurements
    vis = arrays.visibility
    times = arrays.times
    n_traj = states.shape[0]
    if traj_limit > 0:
        n_traj = min(traj_limit, n_traj)

    base_seed = int(th["seed"]) + int(SPLIT_SEED_OFFSETS.get(split, 0))
    enkf_pred = np.zeros_like(states[:n_traj])
    pos_spread_means: list[float] = []

    t0 = time.perf_counter()
    for i in range(n_traj):
        enkf_cfg = EnKFConfig(
            q_pos_m=float(th["q_pos_m"]),
            q_vel_mps=float(th["q_vel_mps"]),
            init_pos_std_m=float(th["init_pos_std_m"]),
            init_vel_std_mps=float(th["init_vel_std_mps"]),
            ensemble_size=int(th["ensemble_size"]),
            inflation=float(inflation),
            seed=base_seed + i,
            angle_deweight_elev_cap_deg=getattr(
                baseline_cfg.ukf, "angle_deweight_elev_cap_deg", None
            ),
        )
        x_hist, _p_hist, diag = run_enkf(
            measurements=meas[i],
            visibility=vis[i],
            times_s=times[i],
            stations=stations,
            ballistic_coeff_m2_per_kg=dyn.ballistic_coeff_m2_per_kg,
            meas_std_vector=meas_std,
            x0_est=arrays.x0_estimates[i],
            cfg=enkf_cfg,
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
        enkf_pred[i] = x_hist
        pos_spread_means.append(float(diag["mean_pos_spread_m"]))

    elapsed = time.perf_counter() - t0
    enkf_rmse = _per_traj_observed_pos_rmse(
        states[:n_traj], enkf_pred, vis[:n_traj], eval_start
    )
    mean_rmse, n_finite = _mean_count(enkf_rmse)
    return {
        "split": split,
        "data_path": str(data_path),
        "n_trajectories": int(n_traj),
        "n_finite_enkf": int(n_finite),
        "enkf_observed_pos_rmse_mean_m": float(mean_rmse),
        "enkf_per_traj_observed_pos_rmse_m": enkf_rmse,
        "enkf_diagnostics": {
            "ensemble_size": int(th["ensemble_size"]),
            "inflation": float(inflation),
            "mean_pos_spread_m": (
                float(np.mean(pos_spread_means)) if pos_spread_means else float("nan")
            ),
        },
        "elapsed_seconds": float(elapsed),
        "arrays": arrays,
        "states": states[:n_traj],
        "visibility": vis[:n_traj],
        "eval_start_step": int(eval_start),
        "stations": int(len(stations)),
    }


def _score_validation_candidate(
    cfg: dict[str, Any],
    th: dict[str, Any],
    inflation: float,
    eval_start: int,
    baseline_cfg: Any,
    traj_limit: int,
    n_boot: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    weighted_num = 0.0
    weighted_den = 0
    for split in VALIDATION_SPLITS:
        res = _run_enkf_split(
            cfg=cfg,
            split=split,
            th=th,
            inflation=inflation,
            eval_start=eval_start,
            baseline_cfg=baseline_cfg,
            traj_limit=traj_limit,
        )
        per_traj = res.pop("enkf_per_traj_observed_pos_rmse_m")
        res.pop("arrays")
        res.pop("states")
        res.pop("visibility")
        mean_ci, lo, hi = _paired_bootstrap_ci(
            per_traj, n_boot=n_boot, seed=int(th["seed"]) + SPLIT_SEED_OFFSETS[split] + 777
        )
        n_finite = int(res["n_finite_enkf"])
        if n_finite > 0 and np.isfinite(res["enkf_observed_pos_rmse_mean_m"]):
            weighted_num += float(res["enkf_observed_pos_rmse_mean_m"]) * n_finite
            weighted_den += n_finite
        splits[split] = {
            **res,
            "enkf_observed_pos_rmse_bootstrap_ci95_m": [float(lo), float(hi)],
            "enkf_observed_pos_rmse_bootstrap_mean_m": float(mean_ci),
        }
        for i, value in enumerate(per_traj):
            csv_rows.append(
                {
                    "phase": "validation_selection",
                    "split": split,
                    "trajectory_index": i,
                    "inflation": float(inflation),
                    "estimator": "EnKF",
                    "observed_pos_rmse_m": float(value) if np.isfinite(value) else None,
                    "selected": False,
                }
            )

    weighted = float(weighted_num / weighted_den) if weighted_den else float("nan")
    return (
        {
            "inflation": float(inflation),
            "weighted_mean_observed_pos_rmse_m": weighted,
            "weighted_finite_trajectory_count": int(weighted_den),
            "split_scores": splits,
        },
        csv_rows,
    )


def _score_test_split(
    cfg: dict[str, Any],
    split: str,
    th: dict[str, Any],
    inflation: float,
    eval_start: int,
    baseline_cfg: Any,
    traj_limit: int,
    n_boot: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    res = _run_enkf_split(
        cfg=cfg,
        split=split,
        th=th,
        inflation=inflation,
        eval_start=eval_start,
        baseline_cfg=baseline_cfg,
        traj_limit=traj_limit,
    )
    arrays = res.pop("arrays")
    states = res.pop("states")
    visibility = res.pop("visibility")
    enkf_rmse = res.pop("enkf_per_traj_observed_pos_rmse_m")

    per_traj: dict[str, np.ndarray] = {"EnKF": enkf_rmse}
    cached_priors = {
        "EKF": arrays.ekf_prior,
        "UKF": arrays.ukf_prior,
        "AUKF": arrays.aukf_prior,
    }
    for name, prior in cached_priors.items():
        if prior is None:
            raise ValueError(f"{split}.npz missing {name.lower()}_prior")
        per_traj[name] = _per_traj_observed_pos_rmse(
            states, prior[: states.shape[0]], visibility, eval_start
        )

    individual_means = {name: _mean_count(values)[0] for name, values in per_traj.items()}
    individual_counts = {name: _mean_count(values)[1] for name, values in per_traj.items()}
    all_mask = _all_method_finite_mask(per_traj)
    paired_n = int(np.sum(all_mask))
    paired_means = {
        name: float(np.mean(values[all_mask])) if paired_n else float("nan")
        for name, values in per_traj.items()
    }

    paired: dict[str, Any] = {}
    for offset, (name, values) in enumerate(per_traj.items()):
        if name == "EnKF":
            continue
        enkf_pair, other_pair = _finite_pair(per_traj["EnKF"], values)
        diffs = enkf_pair - other_pair
        mean_d, lo, hi = _paired_bootstrap_ci(
            diffs, n_boot=n_boot, seed=int(th["seed"]) + SPLIT_SEED_OFFSETS[split] + offset
        )
        paired[name] = {
            "mean_enkf_minus_other_m": float(mean_d),
            "ci95_m": [float(lo), float(hi)],
            "ci_lo_m": float(lo),
            "ci_hi_m": float(hi),
            "wilcoxon_p_one_sided_enkf_better": _one_sided_wilcoxon_candidate_better(diffs),
            "n_paired": int(diffs.size),
            "enkf_better_count": int(np.sum(diffs < 0.0)),
        }

    non_enkf = {name: value for name, value in paired_means.items() if name != "EnKF"}
    best_non_enkf = min(non_enkf, key=lambda name: non_enkf[name])
    best_non_enkf_mean = float(non_enkf[best_non_enkf])
    best_diffs = per_traj["EnKF"][all_mask] - per_traj[best_non_enkf][all_mask]
    gap_mean, gap_lo, gap_hi = _paired_bootstrap_ci(
        best_diffs, n_boot=n_boot, seed=int(th["seed"]) + SPLIT_SEED_OFFSETS[split] + 999
    )
    floor_fraction = float(th.get("practical_significance_floor", 0.03))
    floor_abs = floor_fraction * best_non_enkf_mean
    enkf_is_lowest = bool(
        np.isfinite(paired_means["EnKF"])
        and paired_means["EnKF"] < min(non_enkf.values())
    )
    ci_strictly_negative = bool(np.isfinite(gap_hi) and gap_hi < 0.0)
    floor_exceeded = bool(np.isfinite(gap_mean) and gap_mean < 0.0 and -gap_mean > floor_abs)
    positive = bool(enkf_is_lowest and ci_strictly_negative and floor_exceeded)

    csv_rows: list[dict[str, Any]] = []
    for i in range(states.shape[0]):
        for estimator, values in per_traj.items():
            csv_rows.append(
                {
                    "phase": "frozen_test_evaluation",
                    "split": split,
                    "trajectory_index": i,
                    "inflation": float(inflation),
                    "estimator": estimator,
                    "observed_pos_rmse_m": (
                        float(values[i]) if np.isfinite(values[i]) else None
                    ),
                    "selected": True,
                }
            )

    return (
        {
            **res,
            "inflation": float(inflation),
            "observed_step_rmse_mean_m": individual_means,
            "observed_step_rmse_finite_count": individual_counts,
            "all_method_paired_n": paired_n,
            "all_method_paired_observed_step_rmse_mean_m": paired_means,
            "paired_enkf_vs_other": paired,
            "decision": {
                "best_non_enkf_estimator": best_non_enkf,
                "best_non_enkf_mean_m": best_non_enkf_mean,
                "practical_significance_floor_fraction": floor_fraction,
                "practical_significance_floor_abs_m": float(floor_abs),
                "enkf_minus_best_non_enkf_mean_m": float(gap_mean),
                "enkf_minus_best_non_enkf_ci95_m": [float(gap_lo), float(gap_hi)],
                "enkf_minus_best_non_enkf_ci_lo_m": float(gap_lo),
                "enkf_minus_best_non_enkf_ci_hi_m": float(gap_hi),
                "enkf_is_strictly_lowest_paired_mean": enkf_is_lowest,
                "ci_strictly_negative_for_enkf": ci_strictly_negative,
                "floor_exceeded": floor_exceeded,
                "positive_criterion_met": positive,
            },
        },
        csv_rows,
    )


def _fmt_num(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{float(value):+.{digits}f}"


def _fmt_ci(lo: float, hi: float, digits: int = 0) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "[--,--]"
    return f"[{_fmt_signed(lo, digits)},{_fmt_signed(hi, digits)}]"


def build_tex_table(payload: dict[str, Any]) -> str:
    selected = float(payload["selected_inflation"])
    selection_rows = []
    for cand in payload["validation_selection"]["candidate_scores"]:
        mark = r"\textbf{selected}" if float(cand["inflation"]) == selected else ""
        selection_rows.append(
            "    "
            + " & ".join(
                [
                    f"{float(cand['inflation']):.2f}",
                    _fmt_num(cand["split_scores"]["val"]["enkf_observed_pos_rmse_mean_m"]),
                    _fmt_num(
                        cand["split_scores"]["stress_val"]["enkf_observed_pos_rmse_mean_m"]
                    ),
                    _fmt_num(cand["weighted_mean_observed_pos_rmse_m"]),
                    mark,
                ]
            )
            + r" \\"
        )

    test_rows = []
    for split, label in TEST_SPLITS:
        res = payload["test_split_results"][split]
        means = res["all_method_paired_observed_step_rmse_mean_m"]
        dec = res["decision"]
        lo, hi = dec["enkf_minus_best_non_enkf_ci95_m"]
        test_rows.append(
            "    "
            + " & ".join(
                [
                    label,
                    str(res["all_method_paired_n"]),
                    f"{selected:.2f}",
                    _fmt_num(means["EnKF"]),
                    _fmt_num(means["EKF"]),
                    _fmt_num(means["UKF"]),
                    _fmt_num(means["AUKF"]),
                    (
                        f"{_fmt_signed(dec['enkf_minus_best_non_enkf_mean_m'])} "
                        f"vs {dec['best_non_enkf_estimator']}, "
                        f"CI {_fmt_ci(lo, hi)}"
                    ),
                ]
            )
            + r" \\"
        )

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"  \centering",
            (
                r"  \caption{Validation-selected EnKF inflation audit. "
                r"Multiplicative inflation is selected only on the "
                r"nominal and stress validation splits from the four-point grid "
                r"$\{1.00,1.03,1.06,1.10\}$ using finite-trajectory weighted "
                r"mean observed-step position RMSE, then frozen before one "
                r"evaluation on the force-mismatch, nominal, and stress test "
                r"splits. The completed audit selected 1.00, i.e., no added "
                r"inflation; larger grid values degraded the weighted validation "
                r"score. Test rows use the all-method paired finite subset; "
                r"the rightmost gap is EnKF minus the best cached EKF/UKF/AUKF "
                r"prior with a paired trajectory-bootstrap 95\% confidence "
                r"interval. This is an internal validation-selected inflation "
                r"audit, not localization, particle/Gaussian-mixture filtering, "
                r"external validation, or operational POD.}"
            ),
            r"  \label{tab:validation_tuned_enkf_comparator}",
            r"  \resizebox{\linewidth}{!}{%",
            r"  \begin{tabular}{@{}lrrrr@{}}",
            r"    \toprule",
            r"    Inflation & Val EnKF & Stress-val EnKF & Weighted selection score & Decision \\",
            r"    \midrule",
            *selection_rows,
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"  \vspace{0.5em}",
            r"  \resizebox{\linewidth}{!}{%",
            r"  \begin{tabular}{@{}lrrrrrrl@{}}",
            r"    \toprule",
            r"    Test slice & $n_{\mathrm{pair}}$ & Inflation & EnKF & EKF & UKF & AUKF & EnKF$-$best classical \\",
            r"    \midrule",
            *test_rows,
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table}",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument(
        "--source-rule",
        default="results/loop163_enkf_comparator/enkf_predeclared_rule_loop163.json",
        help="Untuned EnKF rule whose q/init/ensemble settings are inherited.",
    )
    p.add_argument(
        "--output-dir",
        default="results/validation_tuned_enkf_comparator",
        help="Directory for default JSON/CSV outputs.",
    )
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument(
        "--tex-output",
        default="paper/tables/validation_tuned_enkf_comparator.tex",
    )
    p.add_argument("--trajectory-limit", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(Path(args.config))
    source_rule = json.loads(Path(args.source_rule).read_text(encoding="utf-8"))
    th = dict(source_rule["thresholds"])

    baseline_cfg = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)

    candidate_scores: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for inflation in CANDIDATE_GRID:
        score, rows = _score_validation_candidate(
            cfg=cfg,
            th=th,
            inflation=float(inflation),
            eval_start=eval_start,
            baseline_cfg=baseline_cfg,
            traj_limit=int(args.trajectory_limit),
            n_boot=int(args.bootstrap_samples),
        )
        candidate_scores.append(score)
        csv_rows.extend(rows)
        print(
            json.dumps(
                {
                    "phase": "validation_selection",
                    "inflation": inflation,
                    "weighted_mean_observed_pos_rmse_m": score[
                        "weighted_mean_observed_pos_rmse_m"
                    ],
                },
                indent=2,
            )
        )

    selected_score = min(
        candidate_scores,
        key=lambda item: (
            float(item["weighted_mean_observed_pos_rmse_m"]),
            float(item["inflation"]),
        ),
    )
    selected_inflation = float(selected_score["inflation"])
    for row in csv_rows:
        if row["phase"] == "validation_selection":
            row["selected"] = bool(float(row["inflation"]) == selected_inflation)

    test_results: dict[str, Any] = {}
    for split, _label in TEST_SPLITS:
        res, rows = _score_test_split(
            cfg=cfg,
            split=split,
            th=th,
            inflation=selected_inflation,
            eval_start=eval_start,
            baseline_cfg=baseline_cfg,
            traj_limit=int(args.trajectory_limit),
            n_boot=int(args.bootstrap_samples),
        )
        test_results[split] = res
        csv_rows.extend(rows)
        print(
            json.dumps(
                {
                    "phase": "frozen_test_evaluation",
                    "split": split,
                    "selected_inflation": selected_inflation,
                    "paired_means_m": res[
                        "all_method_paired_observed_step_rmse_mean_m"
                    ],
                    "decision": res["decision"],
                },
                indent=2,
            )
        )

    primary_decision = test_results["force_model_mismatch_test"]["decision"]
    payload: dict[str, Any] = {
        "task_id": "EVID-ENKF-VALIDATION-INFLATION",
        "schema_version": "validation_tuned_enkf_comparator_v1",
        "audit_type": "internal_validation_selected_enkf_inflation_audit",
        "candidate_grid": list(CANDIDATE_GRID),
        "selected_inflation": selected_inflation,
        "selection_boundary": {
            "selection_input_splits": list(VALIDATION_SPLITS),
            "selection_input_files": [
                str(Path(cfg["data"]["output_dir"]) / f"{split}.npz")
                for split in VALIDATION_SPLITS
            ],
            "excluded_from_selection": [
                "force_model_mismatch_test",
                "test",
                "stress_test",
                "force_component_omission_test",
            ],
            "selection_metric": (
                "finite-trajectory-count-weighted mean observed-step position "
                "RMSE over val and stress_val only"
            ),
            "test_or_force_mismatch_data_used_for_selection": False,
            "baseline_priors_used_for_selection": False,
            "tie_break_rule": "choose the smaller inflation if weighted validation scores tie",
        },
        "claim_boundary": (
            "Internal validation-selected EnKF inflation audit only. The result "
            "may select 1.00, meaning no added inflation; it "
            "does not claim external validation, operational POD, localized EnKF, "
            "particle filtering, Gaussian-mixture filtering, ensemble-size tuning, "
            "or process-noise/init-covariance tuning."
        ),
        "decision_rule": {
            "selection_rule": (
                "Choose the inflation in [1.00, 1.03, 1.06, 1.10] with the "
                "lowest finite-trajectory-count-weighted validation mean over "
                "val and stress_val; do not inspect test/force-mismatch splits."
            ),
            "test_positive_rule": (
                "On a test split, the validation-selected EnKF setting is a positive only "
                "if its all-method paired mean observed-step position RMSE is "
                "strictly lower than EKF/UKF/AUKF, the paired-bootstrap 95% CI "
                "for EnKF-minus-best-classical is strictly negative, and the "
                "mean gap exceeds the inherited 3% practical floor."
            ),
            "primary_split": "force_model_mismatch_test",
            "context_splits": ["test", "stress_test"],
        },
        "source_untuned_enkf_rule": {
            "path": args.source_rule,
            "inherited_settings": {
                key: th[key]
                for key in [
                    "ensemble_size",
                    "q_pos_m",
                    "q_vel_mps",
                    "init_pos_std_m",
                    "init_vel_std_mps",
                    "seed",
                    "practical_significance_floor",
                ]
                if key in th
            },
        },
        "eval_start_step": int(eval_start),
        "bootstrap_samples": int(args.bootstrap_samples),
        "trajectory_limit": int(args.trajectory_limit),
        "validation_selection": {
            "candidate_grid": list(CANDIDATE_GRID),
            "selection_input_splits": list(VALIDATION_SPLITS),
            "candidate_scores": candidate_scores,
            "selected_candidate": selected_score,
        },
        "test_split_results": test_results,
        "primary_decision": primary_decision,
        "overall_primary_positive": bool(primary_decision["positive_criterion_met"]),
        "output_files": {},
    }

    out_dir = Path(args.output_dir)
    out_json = Path(args.output_json) if args.output_json else out_dir / "validation_tuned_enkf_comparator.json"
    out_csv = Path(args.output_csv) if args.output_csv else out_dir / "validation_tuned_enkf_comparator.csv"
    out_tex = Path(args.tex_output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    payload["output_files"] = {
        "json": str(out_json),
        "csv": str(out_csv),
        "tex_table": str(out_tex),
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    out_tex.write_text(build_tex_table(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "selected_inflation": selected_inflation,
                "primary_decision": primary_decision,
                "json": str(out_json),
                "csv": str(out_csv),
                "tex_table": str(out_tex),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
