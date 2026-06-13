"""Tests for the predeclared, characterized DBAR adaptation-risk heuristic.

These are cheap: the rule logic is exercised on tiny crafted summary dicts, and
the committed artifact (if present) is checked to classify all three
illustrative regimes correctly. No model is trained or re-run here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_adaptation_risk_diagnostic",
    _ROOT / "scripts" / "build_adaptation_risk_diagnostic.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def _summary(*, r_eff: float, med_aukf: float, med_ukf: float, obs: dict) -> dict:
    return {
        "scenario": "synthetic",
        "estimator_truth_model_mismatch": True,
        "aukf_adaptation_mechanism": {"mean_r_eff_scale": r_eff},
        "cross_filter_r_only_nis": {
            "AUKF": {"median_r_only_nis": med_aukf},
            "UKF": {"median_r_only_nis": med_ukf},
        },
        "observed_step_pos_rmse": {
            m: {"observed_step_pos_rmse_m": v} for m, v in obs.items()
        },
    }


class TestDBARRule(unittest.TestCase):
    def test_predeclared_thresholds_are_round_and_fixed(self) -> None:
        # The contribution's credibility rests on a-priori, untuned thresholds.
        self.assertEqual(mod.TAU_R, 1.5)
        self.assertEqual(mod.TAU_RHO, 1.5)
        self.assertEqual(mod.MATERIALITY_MARGIN, 0.05)

    def test_fires_only_under_dynamics_bias(self, tmp=None) -> None:
        # Dynamics bias: R inflated AND adaptation fails to whiten (rho>=1.5),
        # AUKF materially worst -> fires, correct.
        bias = _summary(
            r_eff=3.3,
            med_aukf=4.7,
            med_ukf=1.8,
            obs={"EKF": 448.0, "UKF": 469.0, "AUKF": 526.0},
        )
        out = mod.evaluate_regime("bias", True, _write(bias))
        self.assertTrue(out["dbar_fired"])
        self.assertTrue(out["diagnostic_correct"])

    def test_does_not_fire_under_measurement_noise_stress(self) -> None:
        # The decisive control: R inflated but adaptation whitens (rho<1.5) and
        # AUKF is the best filter -> must NOT fire, and that is correct.
        stress = _summary(
            r_eff=2.8,
            med_aukf=2.35,
            med_ukf=2.44,
            obs={"EKF": 1006.0, "UKF": 1883.0, "AUKF": 915.0},
        )
        out = mod.evaluate_regime("meas-stress", False, _write(stress))
        self.assertFalse(out["dbar_fired"])
        self.assertTrue(out["diagnostic_correct"])

    def test_nominal_cluster_is_harmless_not_a_fire(self) -> None:
        # Sub-margin cluster: AUKF nominally "worst" by <5% is harmless, so no
        # fire is the correct outcome.
        nominal = _summary(
            r_eff=1.48,
            med_aukf=1.63,
            med_ukf=1.54,
            obs={"EKF": 382.76, "UKF": 387.44, "AUKF": 392.69},
        )
        out = mod.evaluate_regime("nominal", False, _write(nominal))
        self.assertFalse(out["dbar_fired"])
        self.assertLess(out["aukf_excess_vs_best_pct"], 5.0)
        self.assertTrue(out["diagnostic_correct"])

    def test_committed_artifact_classifies_all_regimes_correctly(self) -> None:
        art = _ROOT / "results" / "adaptation_risk_diagnostic" / "adaptation_risk_diagnostic.json"
        if not art.exists():
            self.skipTest("diagnostic artifact not built")
        d = json.loads(art.read_text(encoding="utf-8"))
        self.assertEqual(d["schema_version"], "adaptation_risk_diagnostic_v1")
        self.assertEqual(d["n_regimes"], 3)
        self.assertTrue(d["summary"]["all_regimes_classified_correctly"])
        # Exactly one regime (the true dynamics bias) fires.
        self.assertEqual(len(d["summary"]["fired_regimes"]), 1)
        # Wide separating margin so the conclusion is threshold-insensitive.
        self.assertGreater(d["summary"]["separation_margin_rho_nis"], 0.5)


_TMP_FILES: list[Path] = []


def _write(payload: dict) -> Path:
    import tempfile

    fh = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(payload, fh)
    fh.close()
    p = Path(fh.name)
    _TMP_FILES.append(p)
    return p


def tearDownModule() -> None:  # noqa: N802 (unittest hook name)
    for p in _TMP_FILES:
        try:
            p.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
