"""Truth vs estimator scenario-config resolution.

Covers the shared resolver in ``gnn_state_estimation.scenarios`` and verifies
that ``scripts/generate_dataset.py`` synthesizes truth from the scenario
``overrides`` while the recursive-filter priors / innovation features it bakes
into the dataset see the estimator config -- so a scenario that declares
``estimator_overrides`` does not leak its truth force model into the estimator
baselines, while a scenario without it keeps the legacy (estimator == truth)
behavior bit-for-bit.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn_state_estimation.scenarios import (  # noqa: E402
    estimator_sim_config,
    has_estimator_overrides,
    truth_sim_config,
)
from gnn_state_estimation.utils.io import load_yaml  # noqa: E402


def load_script_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_SIM = {
    "orbit_sampling": {"altitude_min_km": 450.0, "altitude_max_km": 950.0},
    "dynamics": {
        "dt_s": 20.0,
        "steps": 4,
        "ballistic_coeff_m2_per_kg": 0.018,
        "process_noise_std": 0.0,
        "drag_rho_ref": 4.0e-11,
        "srp_area_to_mass_m2_per_kg": 0.02,
        "srp_cr": 1.35,
    },
    "measurement_noise": {"range_std_m": 30.0},
    "stations": [{"name": "A", "lat_deg": 0.0, "lon_deg": 0.0, "alt_m": 0.0, "min_elevation_deg": 8.0}],
}


class ScenarioResolverTests(unittest.TestCase):
    def test_without_estimator_overrides_estimator_equals_truth(self) -> None:
        scenario = {"size": 48, "overrides": {"dynamics": {"ballistic_coeff_m2_per_kg": 0.028}}}
        self.assertFalse(has_estimator_overrides(scenario))
        truth = truth_sim_config(BASE_SIM, scenario)
        est = estimator_sim_config(BASE_SIM, scenario)
        # Legacy behavior: estimator config is exactly the truth config.
        self.assertEqual(truth, est)
        self.assertEqual(est["dynamics"]["ballistic_coeff_m2_per_kg"], 0.028)
        # Base simulation is never mutated by resolution.
        self.assertEqual(BASE_SIM["dynamics"]["ballistic_coeff_m2_per_kg"], 0.018)

    def test_empty_estimator_overrides_pins_estimator_to_base(self) -> None:
        scenario = {
            "size": 48,
            "overrides": {"dynamics": {"ballistic_coeff_m2_per_kg": 0.045, "process_noise_std": 0.45}},
            "estimator_overrides": {},
        }
        self.assertTrue(has_estimator_overrides(scenario))
        truth = truth_sim_config(BASE_SIM, scenario)
        est = estimator_sim_config(BASE_SIM, scenario)
        # Truth carries the mismatch; estimator falls back to the nominal base.
        self.assertEqual(truth["dynamics"]["ballistic_coeff_m2_per_kg"], 0.045)
        self.assertEqual(truth["dynamics"]["process_noise_std"], 0.45)
        self.assertEqual(est["dynamics"], BASE_SIM["dynamics"])
        self.assertNotEqual(truth["dynamics"], est["dynamics"])

    def test_nonempty_estimator_overrides_do_not_inherit_truth_overrides(self) -> None:
        scenario = {
            "size": 48,
            "overrides": {"dynamics": {"ballistic_coeff_m2_per_kg": 0.05, "process_noise_std": 0.5}},
            "estimator_overrides": {"dynamics": {"process_noise_std": 0.1}},
        }
        est = estimator_sim_config(BASE_SIM, scenario)
        # Estimator == base + estimator_overrides only: the truth ballistic
        # coefficient must NOT leak in.
        self.assertEqual(est["dynamics"]["process_noise_std"], 0.1)
        self.assertEqual(est["dynamics"]["ballistic_coeff_m2_per_kg"], 0.018)
        self.assertEqual(truth_sim_config(BASE_SIM, scenario)["dynamics"]["process_noise_std"], 0.5)

    def test_public_station_selection_preserved_and_identical_both_sides(self) -> None:
        cfg = load_yaml(str(REPO_ROOT / "configs" / "experiment.yaml"))
        base = cfg["simulation"]
        for name in ("public_catalog_replay_test", "satnogs_observation_replay_test"):
            scenario = cfg["benchmark_suite"]["scenarios"][name]
            truth, truth_meta = truth_sim_config(base, scenario, with_station_meta=True)
            est, est_meta = estimator_sim_config(base, scenario, with_station_meta=True)
            # The public station bank is applied (replaces the base stations)
            # and is identical on both sides so estimators see exactly the
            # stations the measurements were synthesized at.
            self.assertNotEqual(truth["stations"], base["stations"])
            self.assertEqual(truth["stations"], est["stations"])
            self.assertEqual(truth_meta, est_meta)

    def test_configured_force_model_mismatch_scenario(self) -> None:
        cfg = load_yaml(str(REPO_ROOT / "configs" / "experiment.yaml"))
        base = cfg["simulation"]
        scenario = cfg["benchmark_suite"]["scenarios"]["force_model_mismatch_test"]
        self.assertEqual(int(scenario["size"]), 48)
        self.assertTrue(has_estimator_overrides(scenario))
        truth = truth_sim_config(base, scenario)["dynamics"]
        est = estimator_sim_config(base, scenario)["dynamics"]
        # Truth meaningfully differs in drag, process noise, and area-to-mass.
        self.assertGreater(truth["ballistic_coeff_m2_per_kg"], base["dynamics"]["ballistic_coeff_m2_per_kg"])
        self.assertGreater(truth["process_noise_std"], base["dynamics"]["process_noise_std"])
        self.assertGreater(truth["srp_area_to_mass_m2_per_kg"], base["dynamics"]["srp_area_to_mass_m2_per_kg"])
        self.assertGreater(truth["drag_rho_ref"], base["dynamics"]["drag_rho_ref"])
        # Estimator intentionally uses the nominal compact (base) model.
        self.assertEqual(est, base["dynamics"])

    def test_configured_force_component_omission_scenario(self) -> None:
        cfg = load_yaml(str(REPO_ROOT / "configs" / "experiment.yaml"))
        base = cfg["simulation"]
        scenario = cfg["benchmark_suite"]["scenarios"]["force_component_omission_test"]
        self.assertEqual(int(scenario["size"]), 48)
        self.assertTrue(has_estimator_overrides(scenario))
        truth = truth_sim_config(base, scenario)["dynamics"]
        est = estimator_sim_config(base, scenario)["dynamics"]
        # Truth keeps the perturbing forces present (and at least as strong as
        # the base) and applies a moderate drag / process stress.
        self.assertTrue(truth["enable_third_body"])
        self.assertTrue(truth["enable_srp"])
        self.assertGreater(truth["ballistic_coeff_m2_per_kg"], base["dynamics"]["ballistic_coeff_m2_per_kg"])
        self.assertGreater(truth["process_noise_std"], base["dynamics"]["process_noise_std"])
        self.assertGreaterEqual(truth["srp_area_to_mass_m2_per_kg"], base["dynamics"]["srp_area_to_mass_m2_per_kg"])
        self.assertGreater(truth["drag_rho_ref"], base["dynamics"]["drag_rho_ref"])
        # ... but a *moderate* stress: strictly lighter than the heavy
        # force_model_mismatch_test split so the two probes stay distinct.
        heavy = cfg["benchmark_suite"]["scenarios"]["force_model_mismatch_test"]
        heavy_truth = truth_sim_config(base, heavy)["dynamics"]
        self.assertLess(truth["ballistic_coeff_m2_per_kg"], heavy_truth["ballistic_coeff_m2_per_kg"])
        self.assertLess(truth["process_noise_std"], heavy_truth["process_noise_std"])
        # Estimator deliberately omits third-body and SRP and falls back to the
        # nominal compact drag / process-noise values.
        self.assertFalse(est["enable_third_body"])
        self.assertFalse(est["enable_srp"])
        self.assertEqual(est["ballistic_coeff_m2_per_kg"], base["dynamics"]["ballistic_coeff_m2_per_kg"])
        self.assertEqual(est["process_noise_std"], base["dynamics"]["process_noise_std"])
        self.assertEqual(est["drag_rho_ref"], base["dynamics"]["drag_rho_ref"])
        # Estimator force model is strictly simpler than the truth's.
        self.assertNotEqual(truth, est)


class GenerateDatasetWiringTests(unittest.TestCase):
    """``materialize_scenario`` must feed truth config to data synthesis and
    estimator config to the filter priors / innovation features it persists."""

    @staticmethod
    def _parseable_base() -> dict:
        # parse_dataset_config (run inside materialize_scenario) needs the full
        # simulation schema, so start from the shipped config and shrink the
        # propagation length to keep the test light.
        cfg = load_yaml(str(REPO_ROOT / "configs" / "experiment.yaml"))
        base = cfg["simulation"]
        base["dynamics"]["steps"] = 4
        return base

    def _run(self, scenario_cfg: dict) -> tuple[object, dict]:
        gd = load_script_module("generate_dataset", "scripts/generate_dataset.py")
        base_sim = self._parseable_base()
        captured: dict = {}

        def fake_generate_dataset(ds_cfg, num_trajectories, seed):
            captured["truth_ds_cfg"] = ds_cfg
            n, t = num_trajectories, int(ds_cfg.dynamics.steps)
            s = len(ds_cfg.stations)
            return {
                "states": np.zeros((n, t, 6), dtype=np.float64),
                "measurements": np.zeros((n, t, s, 4), dtype=np.float64),
                "visibility": np.zeros((n, t, s), dtype=np.float64),
                "times": np.tile(np.arange(t, dtype=np.float64), (n, 1)),
            }

        def fake_compute_filter_priors(*, data, sim_cfg, baseline_cfg, seed, x0_estimates=None):
            captured["estimator_sim_cfg"] = sim_cfg
            n, t = data["states"].shape[0], data["states"].shape[1]
            filt = {k: np.zeros((n, t, 6), dtype=np.float64) for k in ("ekf", "ukf", "aukf")}
            return filt, np.zeros((n, 6), dtype=np.float64), np.zeros((n, t, 4), dtype=np.float64)

        def fake_innovation(*args, **kwargs):
            return np.zeros((1, 1, 1), dtype=np.float64)

        gd.generate_dataset = fake_generate_dataset
        gd.compute_filter_priors = fake_compute_filter_priors
        gd.compute_innovation_features = fake_innovation
        gd.save_split = lambda *a, **k: None

        summary = gd.materialize_scenario(
            name="unit_scenario",
            scenario_cfg=scenario_cfg,
            base_sim=base_sim,
            baseline_cfg=None,
            out_dir=REPO_ROOT / "results" / "_unit_tmp",
            base_seed=42,
        )
        return captured, summary

    def test_estimator_overrides_does_not_leak_truth_into_priors(self) -> None:
        captured, summary = self._run(
            {
                "size": 3,
                "seed_offset": 0,
                "overrides": {"dynamics": {"ballistic_coeff_m2_per_kg": 0.045, "process_noise_std": 0.45}},
                "estimator_overrides": {},
            }
        )
        truth_dyn = captured["truth_ds_cfg"].dynamics
        est_dyn = captured["estimator_sim_cfg"]["dynamics"]
        # Truth synthesis sees the mismatch ...
        self.assertEqual(truth_dyn.ballistic_coeff_m2_per_kg, 0.045)
        self.assertEqual(truth_dyn.process_noise_std, 0.45)
        # ... while the persisted filter priors see only the nominal base.
        self.assertEqual(est_dyn["ballistic_coeff_m2_per_kg"], 0.018)
        self.assertEqual(est_dyn["process_noise_std"], 0.0)
        self.assertTrue(summary.get("truth_estimator_model_mismatch", False))
        self.assertIn("estimator_overrides", summary)

    def test_without_estimator_overrides_priors_match_truth(self) -> None:
        captured, summary = self._run(
            {
                "size": 3,
                "seed_offset": 0,
                "overrides": {"dynamics": {"ballistic_coeff_m2_per_kg": 0.028}},
            }
        )
        truth_dyn = captured["truth_ds_cfg"].dynamics
        est_dyn = captured["estimator_sim_cfg"]["dynamics"]
        # Legacy scenarios: priors are computed on the very same config as truth.
        self.assertEqual(truth_dyn.ballistic_coeff_m2_per_kg, 0.028)
        self.assertEqual(est_dyn["ballistic_coeff_m2_per_kg"], 0.028)
        self.assertNotIn("estimator_overrides", summary)
        self.assertNotIn("truth_estimator_model_mismatch", summary)


if __name__ == "__main__":
    unittest.main()
