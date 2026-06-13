"""Loop-24 M1: a concrete, predeclared, validated adaptive-filter diagnostic.

The controlled force-model-mismatch result and its mechanism table show *that*
innovation-consistency noise adaptation degrades the AUKF when the dominant
innovation source is an unmodelled dynamics bias. That is a known
covariance-matching limitation (Mehra 1970; Sage & Husa 1969; Jazwinski 1970;
Stacey & D'Amico 2021). The contribution operationalised here is different from
re-narrating the mechanism: a single predeclared decision statistic that a
practitioner can compute online from any adaptive-vs-fixed-noise filter pair,
together with a validation that it fires only when R-adaptation is actually
counterproductive -- and, critically, *not* under even severe measurement-noise
stress, where R-adaptation is the correct move and the AUKF wins.

DBAR -- Dynamics-Bias Adaptation-Risk indicator
------------------------------------------------
From the visible-update stream of an adaptive filter and its fixed-noise twin:

* ``R_eff``  = mean effective measurement-noise scale the adaptive filter
  applied (1.0 == no adaptation).
* ``rho_NIS`` = median(adaptive R-only NIS) / median(fixed-noise UKF R-only NIS):
  whether adaptation *whitened* the standardized residual (<=1) or failed to
  (>1, the fingerprint of a systematic dynamics bias that widening R cannot
  remove).

Predeclared rule (round thresholds fixed a priori, not fitted to outcomes):

    DBAR fires  <=>  R_eff > TAU_R (=1.5)  AND  rho_NIS >= TAU_RHO (=1.5)

A fire means the dominant residual is a dynamics/force-model bias, so
innovation-consistency R-adaptation is counterproductive and process-/dynamics-
uncertainty inflation (or the tighter-gain causal filter) should be preferred.

Validation regimes (each an independently generated split; summaries produced by
``scripts/analyze_force_mismatch_adaptation.py``):

* nominal ``test``                -- perfect shared model (no dynamics bias)
* ``stress_test``                 -- inflated *measurement* noise (no dynamics
                                     bias; R-adaptation is appropriate, AUKF
                                     is the best stress comparator)
* ``force_model_mismatch_test``   -- true dynamics/process-noise bias

The diagnostic is correct iff it fires on the third regime only and matches the
realised AUKF outcome (worst under bias; competitive/best otherwise).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "adaptation_risk_diagnostic"
OUT_PATH = OUT_DIR / "adaptation_risk_diagnostic.json"

TAU_R = 1.5
TAU_RHO = 1.5
# Predeclared materiality margin: AUKF is "materially worst" only when its
# observed-step RMSE exceeds the best filter's by more than this fraction. A
# sub-margin cluster (all filters within MARGIN of each other) means adaptation
# is harmless, which is the correct non-fire outcome.
MATERIALITY_MARGIN = 0.05

# (regime label, dynamics-bias ground truth, summary json)
REGIMES = [
    ("Nominal (shared model)", False, ROOT / "results" / "dbar_nominal_summary.json"),
    ("Measurement-noise stress", False, ROOT / "results" / "dbar_measstress_summary.json"),
    (
        "Force-model mismatch",
        True,
        ROOT / "results" / "force_model_mismatch_adaptation_summary.json",
    ),
]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_regime(label: str, dyn_bias: bool, summary_path: Path) -> dict:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    adapt = s.get("aukf_adaptation_mechanism", {})
    rnis = s.get("cross_filter_r_only_nis", {})
    obs = s.get("observed_step_pos_rmse", {})

    r_eff = _f(adapt.get("mean_r_eff_scale"))
    med_aukf = _f(rnis.get("AUKF", {}).get("median_r_only_nis"))
    med_ukf = _f(rnis.get("UKF", {}).get("median_r_only_nis"))
    rho_nis = med_aukf / med_ukf if med_ukf else float("nan")

    obs_rmse = {
        m: _f(obs.get(m, {}).get("observed_step_pos_rmse_m"))
        for m in ("EKF", "UKF", "AUKF")
    }
    best_method = min(obs_rmse, key=obs_rmse.get)
    best_rmse = obs_rmse[best_method]
    aukf_excess_vs_best = (
        (obs_rmse["AUKF"] - best_rmse) / best_rmse if best_rmse else float("nan")
    )
    aukf_is_best = best_method == "AUKF"
    # "Materially worst": AUKF degraded beyond the predeclared margin (a
    # sub-margin cluster means adaptation is harmless, not harmful).
    aukf_materially_worst = aukf_excess_vs_best > MATERIALITY_MARGIN

    fired = (r_eff > TAU_R) and (rho_nis >= TAU_RHO)
    # The diagnostic is correct when its fire decision matches the dynamics-bias
    # ground truth, and the realised AUKF outcome corroborates it: AUKF
    # materially worst under a fired bias regime; AUKF not materially harmed
    # (competitive or best) when it does not fire.
    consistent_with_outcome = (fired and aukf_materially_worst) or (
        not fired and not aukf_materially_worst
    )
    correct = (fired == dyn_bias) and consistent_with_outcome

    return {
        "regime": label,
        "scenario": s.get("scenario"),
        "dynamics_bias_ground_truth": bool(dyn_bias),
        "estimator_truth_model_mismatch": bool(
            s.get("estimator_truth_model_mismatch", False)
        ),
        "r_eff": round(r_eff, 3),
        "median_r_only_nis_aukf": round(med_aukf, 3),
        "median_r_only_nis_ukf": round(med_ukf, 3),
        "rho_nis": round(rho_nis, 3),
        "observed_step_pos_rmse_m": {k: round(v, 2) for k, v in obs_rmse.items()},
        "best_observed_method": best_method,
        "aukf_is_best": bool(aukf_is_best),
        "aukf_excess_vs_best_pct": round(100.0 * aukf_excess_vs_best, 2),
        "aukf_materially_worst": bool(aukf_materially_worst),
        "dbar_fired": bool(fired),
        "diagnostic_correct": bool(correct),
    }


def main() -> int:
    rows = []
    for label, dyn_bias, path in REGIMES:
        if not path.exists():
            raise SystemExit(
                f"missing diagnostic summary {path.relative_to(ROOT)}; run "
                f"scripts/analyze_force_mismatch_adaptation.py for that scenario first"
            )
        rows.append(evaluate_regime(label, dyn_bias, path))

    fired = [r for r in rows if r["dbar_fired"]]
    no_fire = [r for r in rows if not r["dbar_fired"]]
    rho_fire_min = min((r["rho_nis"] for r in fired), default=float("nan"))
    rho_nofire_max = max((r["rho_nis"] for r in no_fire), default=float("nan"))

    result = {
        "status": "completed",
        "schema_version": "adaptation_risk_diagnostic_v1",
        "diagnostic_name": "DBAR (Dynamics-Bias Adaptation-Risk indicator)",
        "statistic": "rho_NIS = median(adaptive R-only NIS) / median(fixed-noise UKF R-only NIS)",
        "predeclared_thresholds": {"tau_r_eff": TAU_R, "tau_rho_nis": TAU_RHO},
        "rule": "DBAR fires iff R_eff > tau_r_eff AND rho_NIS >= tau_rho_nis",
        "n_regimes": len(rows),
        "regimes": rows,
        "summary": {
            "all_regimes_classified_correctly": all(
                r["diagnostic_correct"] for r in rows
            ),
            "fired_regimes": [r["regime"] for r in fired],
            "no_fire_regimes": [r["regime"] for r in no_fire],
            "min_rho_nis_among_fired": round(rho_fire_min, 3),
            "max_rho_nis_among_no_fire": round(rho_nofire_max, 3),
            "separation_margin_rho_nis": round(rho_fire_min - rho_nofire_max, 3),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} ({len(rows)} regimes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
