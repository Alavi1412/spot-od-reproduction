"""Loop-23 B5 / reviewer M5: scenario-level resampling over orbital/geometry/

noise regimes, with the *scenario* as the statistical unit rather than only the
training seed.

This is a deterministic derivation from the already-computed primary-seed
``results/metrics_summary.json``: for each deterministic scenario the
observed-step (>=1 visible station) position RMSE is reconstructed by pooling
the one-visible and two-or-more-visible visibility buckets that the main
evaluator already records per method. No model is retrained or re-run, so the
table regenerates exactly and offline from the released metrics artifact.

The scenario population spans drag, process-noise, maneuver-like, and three
distinct orbital-inclination/geometry regimes, plus the nominal and
measurement-noise-stress anchors. It complements the 15-seed cohort (which
varies only training under one scenario) by measuring whether the learned
residual generalises across orbital/geometry/noise regimes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics_summary.json"
OUT_DIR = ROOT / "results" / "scenario_resampling"
OUT_PATH = OUT_DIR / "scenario_resampling.json"

# Deterministic scenarios (configs/experiment.yaml benchmark_suite.scenarios +
# the nominal/stress anchors). Replay-only scenarios are intentionally excluded
# because they are reported in their own tables.
SCENARIOS = [
    ("test", "Nominal", "nominal optical/radio sparse-visibility"),
    ("stress_test", "Measurement-noise stress", "inflated measurement noise/outliers"),
    ("high_drag_test", "High drag", "inflated ballistic coefficient + process noise"),
    ("process_noise_shift_test", "Process-noise shift", "elevated process-noise std"),
    ("maneuver_shift_test", "Maneuver-like shift", "large process noise + dropout/outliers"),
    ("low_inclination_test", "Low inclination", "low-inclination orbital regime"),
    ("sunsync_like_test", "Sun-synchronous-like", "sun-synchronous-like geometry"),
    ("high_inclination_test", "High inclination", "high-inclination orbital regime"),
]

# metrics key -> paper display name
METHODS = [("EKF", "EKF"), ("UKF", "UKF"), ("AUKF", "AUKF"), ("HybridGNN", "RGR-GF")]


def observed_step_rmse(block: dict) -> float:
    """Pool the one-visible and >=2-visible buckets into observed-step RMSE.

    Each ``vis_k_pos_rmse_m`` is an RMS over its bucket's evaluated
    trajectory-steps; pooling by step counts is exact for RMS quantities.
    """
    n1 = float(block.get("vis_1_count", 0) or 0)
    r1 = float(block.get("vis_1_pos_rmse_m", 0.0) or 0.0)
    n2 = float(block.get("vis_2plus_count", 0) or 0)
    r2 = float(block.get("vis_2plus_pos_rmse_m", 0.0) or 0.0)
    n = n1 + n2
    if n <= 0:
        return float("nan")
    return math.sqrt((n1 * r1 * r1 + n2 * r2 * r2) / n)


def main() -> int:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    scenario_rows = []
    for key, label, regime in SCENARIOS:
        blk = metrics.get(key)
        if not blk or not all(mk in blk for mk, _ in METHODS):
            continue
        vals = {disp: observed_step_rmse(blk[mk]) for mk, disp in METHODS}
        rgr = vals["RGR-GF"]
        best_method = min(vals, key=vals.get)
        scenario_rows.append(
            {
                "name": key,
                "label": label,
                "regime": regime,
                "ekf_obs_pos_rmse_m": round(vals["EKF"], 2),
                "ukf_obs_pos_rmse_m": round(vals["UKF"], 2),
                "aukf_obs_pos_rmse_m": round(vals["AUKF"], 2),
                "rgr_gf_obs_pos_rmse_m": round(rgr, 2),
                "rgr_minus_ekf_m": round(rgr - vals["EKF"], 2),
                "rgr_minus_ukf_m": round(rgr - vals["UKF"], 2),
                "rgr_minus_aukf_m": round(rgr - vals["AUKF"], 2),
                "best_method": best_method,
            }
        )

    n = len(scenario_rows)
    beats = {
        "aukf": sum(1 for r in scenario_rows if r["rgr_minus_aukf_m"] < 0),
        "ekf": sum(1 for r in scenario_rows if r["rgr_minus_ekf_m"] < 0),
        "ukf": sum(1 for r in scenario_rows if r["rgr_minus_ukf_m"] < 0),
    }
    best_counts = {disp: 0 for _, disp in METHODS}
    for r in scenario_rows:
        best_counts[r["best_method"]] += 1

    def mean(field: str) -> float:
        return round(sum(r[field] for r in scenario_rows) / n, 2) if n else float("nan")

    result = {
        "status": "completed",
        "schema_version": "scenario_resampling_v1",
        "primary_metric": "observed_step_position_rmse_m",
        "statistical_unit": "deterministic scenario (single primary seed per scenario)",
        "source": "results/metrics_summary.json (visibility-bucket pooling; no recompute)",
        "num_scenarios": n,
        "scenarios": scenario_rows,
        "summary": {
            "n_scenarios": n,
            "rgr_gf_beats_aukf_scenarios": beats["aukf"],
            "rgr_gf_beats_ekf_scenarios": beats["ekf"],
            "rgr_gf_beats_ukf_scenarios": beats["ukf"],
            "mean_rgr_minus_aukf_m": mean("rgr_minus_aukf_m"),
            "mean_rgr_minus_ekf_m": mean("rgr_minus_ekf_m"),
            "mean_rgr_minus_ukf_m": mean("rgr_minus_ukf_m"),
            "best_method_scenario_counts": best_counts,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} ({n} scenarios)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
