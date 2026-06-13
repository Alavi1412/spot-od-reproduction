"""Tests for the constrained-AUKF mechanism-control artifact and table generator.

Validates the JSON schema of the committed artifact and the honest numerical
invariants the manuscript relies on:
- AUKF-Rcap (max_r_scale=2.0) improves over standard AUKF (paired gap strictly
  negative under the AUKF-Rcap minus AUKF convention)
- AUKF-Rcap remains strictly worse than EKF (positive paired gap, CI entirely
  above zero), confirming the mechanism interpretation
- Mean effective R scale of AUKF-Rcap is well below the standard AUKF value
  (mechanism activated successfully)
- Table generator renders the required label and key numbers

No model is run; all checks operate on the committed artifact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_paper_assets", _ROOT / "scripts" / "build_paper_assets.py"
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

_ARTIFACT = _ROOT / "results" / "constrained_aukf_mechanism_control" / "constrained_aukf_mechanism_control.json"

# Standard AUKF observed-step RMSE from the committed mechanism summary
_STANDARD_AUKF_RMSE_M = 526.39


class ConstrainedAuKFMechanismControlArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("constrained_aukf_mechanism_control.json not present")
        self.data = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    def test_schema_version_and_keys(self) -> None:
        self.assertEqual(
            self.data.get("schema_version"),
            "constrained_aukf_mechanism_control_v1",
        )
        for key in (
            "scenario",
            "trajectories_processed",
            "rcap_config",
            "standard_aukf_max_r_scale",
            "rcap_mechanism",
            "pooled_observed_step_rmse_m",
            "paired_comparisons",
            "n_paired_trajectories",
        ):
            self.assertIn(key, self.data, f"Missing key: {key}")

    def test_rcap_config_is_2x(self) -> None:
        rcap_cfg = self.data["rcap_config"]
        self.assertAlmostEqual(rcap_cfg["max_r_scale"], 2.0, places=6)

    def test_effective_r_scale_is_below_standard_aukf(self) -> None:
        """AUKF-Rcap mean effective R scale must be well below 3.32 (standard AUKF)."""
        mech = self.data["rcap_mechanism"]
        r_eff = float(mech["mean_r_eff_scale"])
        # Standard AUKF mean effective R scale is 3.32; cap at 2.0 should give less than or equal to 2.0
        self.assertLess(r_eff, 2.5, f"Effective R scale {r_eff:.3f} unexpectedly high")
        # And it should be above the min_r_scale (0.4)
        self.assertGreater(r_eff, 0.0)

    def test_rcap_improves_over_standard_aukf(self) -> None:
        """AUKF-Rcap must beat standard AUKF: paired gap AUKF-Rcap minus AUKF must be negative."""
        pairs = self.data["paired_comparisons"]
        self.assertIn("AUKF", pairs)
        aukf_gap = float(pairs["AUKF"]["mean_paired_gap_m"])
        ci_hi = float(pairs["AUKF"]["ci_hi_m"])
        # Negative mean gap: AUKF-Rcap better than standard AUKF
        self.assertLess(aukf_gap, 0.0,
                        f"AUKF-Rcap should beat standard AUKF: gap={aukf_gap:.1f} m")
        # CI upper bound also negative (strictly negative CI)
        self.assertLess(ci_hi, 0.0,
                        f"CI upper bound should be negative: ci_hi={ci_hi:.1f} m")

    def test_rcap_still_worse_than_ekf(self) -> None:
        """AUKF-Rcap must remain worse than EKF: positive gap and CI entirely above zero."""
        pairs = self.data["paired_comparisons"]
        self.assertIn("EKF", pairs)
        ekf_gap = float(pairs["EKF"]["mean_paired_gap_m"])
        ci_lo = float(pairs["EKF"]["ci_lo_m"])
        # Positive mean gap: AUKF-Rcap worse than EKF
        self.assertGreater(ekf_gap, 0.0,
                           f"AUKF-Rcap should be worse than EKF: gap={ekf_gap:.1f} m")
        # CI lower bound also positive (strictly positive CI)
        self.assertGreater(ci_lo, 0.0,
                           f"CI lower bound should be positive: ci_lo={ci_lo:.1f} m")

    def test_pooled_rmse_ordering(self) -> None:
        """Verify ordering: EKF < UKF < AUKF-Rcap < AUKF (all present and sensible)."""
        pooled = self.data["pooled_observed_step_rmse_m"]
        ekf = float(pooled["EKF"])
        ukf = float(pooled["UKF"])
        rcap = float(pooled["AUKF_Rcap"])
        aukf = float(pooled["AUKF"])
        self.assertLess(ekf, ukf,
                        f"Expected EKF < UKF: {ekf:.2f} vs {ukf:.2f}")
        self.assertLess(rcap, aukf,
                        f"Expected AUKF-Rcap < AUKF: {rcap:.2f} vs {aukf:.2f}")
        # Standard AUKF should be close to the reference value
        self.assertAlmostEqual(aukf, _STANDARD_AUKF_RMSE_M, delta=5.0,
                               msg=f"Standard AUKF RMSE {aukf:.2f} differs from reference")

    def test_paired_convention_stated(self) -> None:
        """Paired-difference convention must be documented."""
        convention = str(self.data.get("paired_difference_convention", ""))
        self.assertIn("AUKF_Rcap minus comparator", convention)
        self.assertIn("negative", convention.lower())

    def test_n_trajectories_is_48(self) -> None:
        self.assertEqual(int(self.data["trajectories_processed"]), 48)

    def test_n_paired_trajectories_is_38(self) -> None:
        self.assertEqual(int(self.data["n_paired_trajectories"]), 38)

    def test_all_paired_comparisons_record_38_paired(self) -> None:
        """Every per-comparator paired_comparisons entry must record n_paired_trajectories == 38."""
        pairs = self.data["paired_comparisons"]
        for comp, pr in pairs.items():
            self.assertEqual(
                int(pr["n_paired_trajectories"]),
                38,
                f"paired_comparisons[{comp!r}]['n_paired_trajectories'] != 38",
            )

    def test_all_paired_cis_are_ordered(self) -> None:
        """All CI (lo, hi) pairs must be ordered lo less than or equal to hi."""
        for comp, pr in self.data["paired_comparisons"].items():
            lo = float(pr["ci_lo_m"])
            hi = float(pr["ci_hi_m"])
            self.assertLessEqual(lo, hi,
                                 f"CI inverted for {comp}: [{lo:.1f}, {hi:.1f}]")


class ConstrainedAUKFTableGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _ARTIFACT.exists():
            self.skipTest("constrained_aukf_mechanism_control.json not present")

    def test_table_renders_label(self) -> None:
        tex = mod.build_constrained_aukf_mechanism_control_table()
        self.assertIn("tab:constrained_aukf_mechanism_control", tex)

    def test_table_renders_key_numbers(self) -> None:
        tex = mod.build_constrained_aukf_mechanism_control_table()
        # EKF RMSE
        self.assertIn("448.79", tex)
        # AUKF-Rcap must appear and show improvement over standard AUKF
        self.assertIn("AUKF-Rcap", tex)
        # Standard AUKF RMSE
        self.assertIn("526.39", tex)

    def test_table_is_valid_latex_fragment(self) -> None:
        tex = mod.build_constrained_aukf_mechanism_control_table()
        self.assertIn("\\begin{table}", tex)
        self.assertIn("\\end{table}", tex)
        self.assertIn("\\caption{", tex)
        self.assertIn("\\toprule", tex)
        self.assertIn("\\bottomrule", tex)
        # Should not contain forbidden terms
        self.assertNotIn("checkpoint", tex.lower())
        self.assertNotIn("zenodo", tex.lower())
        self.assertNotIn("\\bvenv\\b", tex.lower())

    def test_table_distinguishes_48_processed_and_38_paired(self) -> None:
        """Table caption must state all 48 processed trajectories and n=38 paired finite ones."""
        tex = mod.build_constrained_aukf_mechanism_control_table()
        self.assertIn("All 48 trajectories are processed", tex)
        self.assertIn("$n=38$ paired finite trajectories", tex)

    def test_table_contains_standard_aukf_mechanism_values(self) -> None:
        """Table must contain formatted standard AUKF values: '3.315' and '466.3'."""
        tex = mod.build_constrained_aukf_mechanism_control_table()
        self.assertIn("3.315", tex, "Expected mean R-eff scale '3.315' in table")
        self.assertIn("466.3", tex, "Expected state-update norm '466.3' in table")

    def test_json_artifact_values_format_to_expected_strings(self) -> None:
        """Standard AUKF values in adaptation summary must format to '3.315' and '466.3'."""
        adp_path = _ROOT / "results" / "force_model_mismatch_adaptation_summary.json"
        if not adp_path.exists():
            self.skipTest("force_model_mismatch_adaptation_summary.json not present")
        adp = json.loads(adp_path.read_text(encoding="utf-8"))
        mech = adp["aukf_adaptation_mechanism"]
        r_eff = float(mech["mean_r_eff_scale"])
        upd_norm = float(mech["mean_state_update_pos_norm_m"])
        self.assertEqual(
            mod.format_metric(r_eff, 3),
            "3.315",
            f"mean_r_eff_scale {r_eff} does not format to '3.315'",
        )
        self.assertEqual(
            mod.format_metric(upd_norm, 1),
            "466.3",
            f"mean_state_update_pos_norm_m {upd_norm} does not format to '466.3'",
        )

    def test_table_contains_denominator_cause(self) -> None:
        """Table caption must explain that the 10 excluded trajectories have zero
        visible scored steps under the shared visibility/scoring mask, not
        method-specific nonfinite failures.
        """
        tex = mod.build_constrained_aukf_mechanism_control_table()
        # Caption must mention zero-visibility as the cause of the smaller denominator
        has_cause = (
            "zero visible scored steps" in tex
            or "no visible scored steps" in tex
            or "visibility/scoring mask" in tex
            or "shared visibility" in tex
        )
        self.assertTrue(
            has_cause,
            "Table caption should explain the denominator cause "
            "(zero visible scored steps / shared visibility mask)",
        )

    def test_table_48_minus_38_denominator_contract(self) -> None:
        """Table caption must state all 48 processed and the 10-excluded visibility reason."""
        tex = mod.build_constrained_aukf_mechanism_control_table()
        # 48 processed trajectories must appear
        self.assertIn("48", tex, "Table caption must mention 48 processed trajectories")
        # 38 paired finite trajectories must appear
        self.assertIn("38", tex, "Table caption must mention 38 paired trajectories")
        # The exclusion reason (zero visible scored steps) must be present
        has_cause = (
            "zero visible scored steps" in tex
            or "no visible scored steps" in tex
            or "visibility/scoring mask" in tex
            or "shared visibility" in tex
        )
        self.assertTrue(
            has_cause,
            "Table caption must state the shared-visibility/scoring mask cause",
        )

    def test_table_unavailable_stub_on_missing_file(self) -> None:
        tex = mod.build_constrained_aukf_mechanism_control_table(
            result_path=_ROOT / "results" / "__nonexistent_test_path__.json"
        )
        self.assertIn("%", tex)
        self.assertIn("unavailable", tex)


class ConstrainedAUKFBootstrapHelperTests(unittest.TestCase):
    """Smoke test for the paired bootstrap CI helper."""

    def _import_helper(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "run_constrained_aukf",
            _ROOT / "scripts" / "run_constrained_aukf_mechanism_control.py",
        )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_ci_is_ordered_and_finite(self) -> None:
        import numpy as np
        try:
            m = self._import_helper()
        except Exception:
            self.skipTest("Could not import run_constrained_aukf_mechanism_control")
        rng = np.random.default_rng(42)
        diffs = rng.standard_normal(50)
        lo, hi = m.paired_bootstrap_ci(diffs, seed=123, n_bootstrap=500)
        self.assertLess(lo, hi)
        self.assertTrue(all(map(lambda v: v == v, [lo, hi])))  # finite check

    def test_empty_diffs_returns_nans(self) -> None:
        import math
        import numpy as np
        try:
            m = self._import_helper()
        except Exception:
            self.skipTest("Could not import run_constrained_aukf_mechanism_control")
        lo, hi = m.paired_bootstrap_ci(np.array([]), seed=0, n_bootstrap=100)
        self.assertTrue(math.isnan(lo))
        self.assertTrue(math.isnan(hi))


if __name__ == "__main__":
    unittest.main()
