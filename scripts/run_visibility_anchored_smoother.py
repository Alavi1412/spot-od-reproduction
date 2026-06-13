#!/usr/bin/env python
"""Build and score a visibility-anchored RFIS composite from existing artifacts.

Motivation
----------
The robust fixed-interval variational smoother (RFIS) reference improves
all-step position RMSE over the best classical method on the regime-shift
scenarios, but RFIS *alone* is far worse than a causal EKF on the
measurement-informed observed steps: spreading the fit across the full arc
trades observed-step accuracy for a small all-step margin. That observed-step
degradation is the motivation for anchoring.

Construction (visibility-anchored RFIS, "VA-RFIS")
-------------------------------------------------
For every evaluated state time the composite uses a *predeclared* rule that
consults only the station-visibility mask, never ground truth:

* if at least one station is visible at that time -> use the causal EKF prior
  (the predeclared observed-step anchor);
* if zero stations are visible -> use the offline RFIS multistart estimate.

The composite is therefore an offline construction (RFIS uses future
measurements in the zero-visible gaps) that preserves the EKF observed-step
estimate while substituting the smoother only where no measurement exists.
Method selection uses the visibility mask and the predeclared anchor only;
ground-truth states are read solely to score the predictions.

This script reuses already-computed artifacts (it does not refit anything):

* RFIS multistart predictions:  <rfis-dir>/<scenario>/rfis_predictions.npz
* batch-WLS predictions:        <wls-dir>/<scenario>/batch_wls_predictions.npz
* dataset priors + visibility:  <data-dir>/<scenario>.npz
* the evaluation window (``eval_start_step``) and trajectory count are read
  from the RFIS scenario summary so the composite is scored on exactly the
  same window and arcs as the published RFIS reference.

Outputs (paper-facing JSON/CSV) contain only scenario, the composite rule,
metrics, and an honest verdict: no local paths, environment, or
code-structure prose.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn_state_estimation.evaluation import score_predictions

# Methods scored alongside the composite. EKF/UKF/AUKF are the recursive
# causal filters carried in the dataset; BatchWLS and RFIS are the two offline
# classical OD references already computed on the shift scenarios.
_RECURSIVE_METHODS = ("EKF", "UKF", "AUKF")
_CLASSICAL_METHODS = _RECURSIVE_METHODS + ("BatchWLS",)
# A composite metric is treated as exactly matching its anchor when it agrees
# to this relative tolerance (the construction makes the agreement exact up to
# floating-point summation order).
_EQUALITY_RTOL = 1.0e-9


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        type=str,
        default="process_noise_shift_test,maneuver_shift_test",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results/data",
        help="Directory holding <scenario>.npz dataset files with EKF/UKF/AUKF "
        "priors and the station-visibility mask.",
    )
    parser.add_argument(
        "--rfis-dir",
        type=str,
        default="results/variational_smoother_multistart_shift_full",
        help="Directory holding the RFIS multistart predictions and per-scenario summaries.",
    )
    parser.add_argument(
        "--wls-dir",
        type=str,
        default="results/batch_wls_shift_baseline",
        help="Directory holding the batch-WLS predictions per scenario.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visibility_anchored_smoother_shift",
    )
    return parser


def _masked_pos_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Position RMSE over masked steps (sum-of-squares-per-point convention).

    Identical to the observed/zero-visible RMSE used by the RFIS and batch-WLS
    baselines so the composite is scored on the same definition.
    """
    if not np.any(mask):
        return float("nan")
    err = y_true[mask, :3] - y_pred[mask, :3]
    return float(np.sqrt(np.mean(np.sum(err * err, axis=-1))))


def _observed_metrics(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> dict[str, float]:
    eval_vis = visibility[:, eval_start:]
    observed = np.sum(eval_vis, axis=-1) >= 0.5
    zero_visible = ~observed
    y_true = states[:, eval_start:]
    y_pred = preds[:, eval_start:]
    return {
        "observed_step_pos_rmse_m": _masked_pos_rmse(y_true, y_pred, observed),
        "zero_visible_pos_rmse_m": _masked_pos_rmse(y_true, y_pred, zero_visible),
        "observed_steps": int(np.sum(observed)),
        "zero_visible_steps": int(np.sum(zero_visible)),
    }


def _trajectory_rmse_values(y_true: np.ndarray, y_pred: np.ndarray, eval_start: int) -> np.ndarray:
    err = y_true[:, eval_start:, :3] - y_pred[:, eval_start:, :3]
    return np.sqrt(np.mean(np.sum(err * err, axis=-1), axis=1))


def _method_summary(
    states: np.ndarray, preds: np.ndarray, visibility: np.ndarray, eval_start: int
) -> dict[str, float]:
    all_step = score_predictions(states[:, eval_start:], preds[:, eval_start:])
    obs = _observed_metrics(states, preds, visibility, eval_start)
    traj = _trajectory_rmse_values(states, preds, eval_start)
    return {
        "all_step_pos_rmse_m": float(all_step["pos_rmse_m"]),
        "all_step_vel_rmse_mps": float(all_step["vel_rmse_mps"]),
        "median_traj_pos_rmse_m": float(np.median(traj)),
        "max_traj_pos_rmse_m": float(np.max(traj)),
        "failure_rate_100km": float(np.mean(traj > 100_000.0)),
        **obs,
    }


def _gain_percent(reference: float, candidate: float) -> float:
    """Signed % improvement of candidate over reference (positive = better)."""
    if not np.isfinite(reference) or abs(reference) < 1e-9 or not np.isfinite(candidate):
        return float("nan")
    return float(100.0 * (reference - candidate) / reference)


def _approx_equal(a: float, b: float, rtol: float = _EQUALITY_RTOL) -> bool:
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return bool(abs(a - b) <= rtol * max(1.0, abs(b)))


def _load_prediction(path: Path, key: str) -> np.ndarray:
    data = np.load(path)
    if key not in data.files:
        raise ValueError(f"{path.name} is missing the '{key}' array.")
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 6:
        raise ValueError(f"{path.name}['{key}'] has unexpected shape {arr.shape}.")
    return arr


def run_scenario(
    scenario: str,
    *,
    data_dir: Path,
    rfis_dir: Path,
    wls_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rfis_summary_path = rfis_dir / scenario / "rfis_summary.json"
    if not rfis_summary_path.exists():
        raise FileNotFoundError(f"missing RFIS summary for scenario {scenario!r}")
    rfis_summary = json.loads(rfis_summary_path.read_text(encoding="utf-8"))
    eval_start = int(rfis_summary["eval_start_step"])

    dataset = np.load(data_dir / f"{scenario}.npz", allow_pickle=True)
    for key in ("states", "visibility", "ekf_prior", "ukf_prior", "aukf_prior"):
        if key not in dataset.files:
            raise ValueError(f"{scenario}.npz is missing '{key}'.")
    states = np.asarray(dataset["states"], dtype=np.float64)
    visibility = np.asarray(dataset["visibility"], dtype=np.float64)
    ekf_prior = np.asarray(dataset["ekf_prior"], dtype=np.float64)
    ukf_prior = np.asarray(dataset["ukf_prior"], dtype=np.float64)
    aukf_prior = np.asarray(dataset["aukf_prior"], dtype=np.float64)

    rfis = _load_prediction(rfis_dir / scenario / "rfis_predictions.npz", "rfis")
    wls = _load_prediction(wls_dir / scenario / "batch_wls_predictions.npz", "batch_wls")

    # Align on the common trajectory count across every artifact.
    n_traj = min(
        states.shape[0],
        visibility.shape[0],
        ekf_prior.shape[0],
        ukf_prior.shape[0],
        aukf_prior.shape[0],
        rfis.shape[0],
        wls.shape[0],
    )
    states = states[:n_traj]
    visibility = visibility[:n_traj]
    ekf_prior = ekf_prior[:n_traj]
    ukf_prior = ukf_prior[:n_traj]
    aukf_prior = aukf_prior[:n_traj]
    rfis = rfis[:n_traj]
    wls = wls[:n_traj]

    # Predeclared, ground-truth-free anchoring rule: EKF where any station is
    # visible, RFIS where zero stations are visible. Built over the full arc;
    # only the evaluated window [eval_start:] is scored downstream.
    observed_mask = np.sum(visibility, axis=-1) >= 0.5  # (n_traj, T)
    va_rfis = np.where(observed_mask[..., None], ekf_prior, rfis)

    method_arrays = {
        "VA_RFIS": va_rfis,
        "RFIS": rfis,
        "EKF": ekf_prior,
        "UKF": ukf_prior,
        "AUKF": aukf_prior,
        "BatchWLS": wls,
    }
    methods = {
        name: _method_summary(states, arr, visibility, eval_start)
        for name, arr in method_arrays.items()
    }

    va = methods["VA_RFIS"]
    rfis_m = methods["RFIS"]
    ekf_m = methods["EKF"]

    def best_by(metric: str) -> tuple[str, float]:
        name = min(_CLASSICAL_METHODS, key=lambda m: methods[m][metric])
        return name, float(methods[name][metric])

    best_cls_all, best_cls_all_val = best_by("all_step_pos_rmse_m")
    best_cls_obs, best_cls_obs_val = best_by("observed_step_pos_rmse_m")

    gain_va_vs_best_cls_all = _gain_percent(best_cls_all_val, va["all_step_pos_rmse_m"])
    gain_va_vs_best_cls_obs = _gain_percent(best_cls_obs_val, va["observed_step_pos_rmse_m"])
    gain_va_vs_batchwls_all = _gain_percent(
        methods["BatchWLS"]["all_step_pos_rmse_m"], va["all_step_pos_rmse_m"]
    )
    gain_va_vs_ekf_obs = _gain_percent(
        ekf_m["observed_step_pos_rmse_m"], va["observed_step_pos_rmse_m"]
    )
    gain_va_vs_ekf_all = _gain_percent(
        ekf_m["all_step_pos_rmse_m"], va["all_step_pos_rmse_m"]
    )
    gain_rfis_vs_best_cls_all = _gain_percent(
        best_cls_all_val, rfis_m["all_step_pos_rmse_m"]
    )
    gain_rfis_vs_best_cls_obs = _gain_percent(
        best_cls_obs_val, rfis_m["observed_step_pos_rmse_m"]
    )

    # Construction sanity: on observed steps the composite *is* EKF, and in the
    # zero-visible gaps it *is* RFIS, so these metrics must match their anchors
    # exactly (up to summation order). Recorded, not assumed.
    va_obs_equals_ekf = _approx_equal(
        va["observed_step_pos_rmse_m"], ekf_m["observed_step_pos_rmse_m"]
    )
    va_zero_equals_rfis = _approx_equal(
        va["zero_visible_pos_rmse_m"], rfis_m["zero_visible_pos_rmse_m"]
    )

    preserves_observed = va_obs_equals_ekf
    improves_all_step_vs_best_classical = bool(
        np.isfinite(gain_va_vs_best_cls_all) and gain_va_vs_best_cls_all > 0.0
    )
    materially_improves_all_step = bool(
        np.isfinite(gain_va_vs_best_cls_all) and gain_va_vs_best_cls_all >= 1.0
    )
    if improves_all_step_vs_best_classical and preserves_observed:
        verdict = "improves_all_step_over_best_classical_while_preserving_ekf_observed_step"
    elif improves_all_step_vs_best_classical:
        verdict = "improves_all_step_over_best_classical_only"
    else:
        verdict = "no_all_step_improvement_over_best_classical"

    scenario_dir = output_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scenario_dir / "va_rfis_predictions.npz", va_rfis=va_rfis)

    summary = {
        "scenario": scenario,
        "trajectories": int(n_traj),
        "eval_start_step": int(eval_start),
        "method": "visibility_anchored_rfis_composite",
        "composite_rule": (
            "Predeclared, ground-truth-free: at evaluated state times with at "
            "least one visible station use the causal EKF prior (observed-step "
            "anchor); at zero-visible times use the offline RFIS multistart "
            "estimate. Selection uses only the visibility mask and the "
            "predeclared anchor; ground truth is used solely for scoring. The "
            "composite is offline because RFIS uses future measurements in the "
            "zero-visible gaps."
        ),
        "rfis_provenance": rfis_dir.as_posix(),
        "wls_provenance": wls_dir.as_posix(),
        "best_classical_all_step_method": best_cls_all,
        "best_classical_observed_method": best_cls_obs,
        "va_rfis_gain_vs_best_classical_all_step_percent": gain_va_vs_best_cls_all,
        "va_rfis_gain_vs_best_classical_observed_percent": gain_va_vs_best_cls_obs,
        "va_rfis_gain_vs_batchwls_all_step_percent": gain_va_vs_batchwls_all,
        "va_rfis_gain_vs_ekf_observed_percent": gain_va_vs_ekf_obs,
        "va_rfis_gain_vs_ekf_all_step_percent": gain_va_vs_ekf_all,
        "rfis_gain_vs_best_classical_all_step_percent": gain_rfis_vs_best_cls_all,
        "rfis_gain_vs_best_classical_observed_percent": gain_rfis_vs_best_cls_obs,
        "va_rfis_observed_step_equals_ekf": bool(va_obs_equals_ekf),
        "va_rfis_zero_visible_equals_rfis": bool(va_zero_equals_rfis),
        "preserves_ekf_observed_step": bool(preserves_observed),
        "improves_all_step_vs_best_classical": improves_all_step_vs_best_classical,
        "materially_improves_all_step_vs_best_classical": materially_improves_all_step,
        "honest_verdict": verdict,
        "methods": methods,
    }
    (scenario_dir / "va_rfis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{scenario}: VA-RFIS all-step={va['all_step_pos_rmse_m']:.4f} m "
        f"obs-step={va['observed_step_pos_rmse_m']:.4f} m "
        f"zero-visible={va['zero_visible_pos_rmse_m']:.4f} m | "
        f"best classical all-step={best_cls_all_val:.4f} m ({best_cls_all}) "
        f"all-step gain={gain_va_vs_best_cls_all:+.2f}% "
        f"obs-step gain vs EKF={gain_va_vs_ekf_obs:+.4f}% "
        f"verdict={verdict}",
        flush=True,
    )
    return summary


def flatten_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scalar_keys = [
        "scenario",
        "trajectories",
        "eval_start_step",
        "method",
        "best_classical_all_step_method",
        "best_classical_observed_method",
        "va_rfis_gain_vs_best_classical_all_step_percent",
        "va_rfis_gain_vs_best_classical_observed_percent",
        "va_rfis_gain_vs_batchwls_all_step_percent",
        "va_rfis_gain_vs_ekf_observed_percent",
        "va_rfis_gain_vs_ekf_all_step_percent",
        "rfis_gain_vs_best_classical_all_step_percent",
        "rfis_gain_vs_best_classical_observed_percent",
        "va_rfis_observed_step_equals_ekf",
        "va_rfis_zero_visible_equals_rfis",
        "preserves_ekf_observed_step",
        "improves_all_step_vs_best_classical",
        "materially_improves_all_step_vs_best_classical",
        "honest_verdict",
    ]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        base = {key: summary[key] for key in scalar_keys}
        for method, metrics in summary["methods"].items():
            prefix = method.lower()
            for key, value in metrics.items():
                base[f"{prefix}_{key}"] = value
        rows.append(base)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir)
    rfis_dir = Path(args.rfis_dir)
    wls_dir = Path(args.wls_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for scenario in [name.strip() for name in args.scenarios.split(",") if name.strip()]:
        summaries.append(
            run_scenario(
                scenario,
                data_dir=data_dir,
                rfis_dir=rfis_dir,
                wls_dir=wls_dir,
                output_dir=output_dir,
            )
        )

    rows = flatten_summary_rows(summaries)
    pd.DataFrame(rows).to_csv(output_dir / "va_rfis_summary.csv", index=False)
    (output_dir / "va_rfis_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"wrote": "va_rfis_summary.csv", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
