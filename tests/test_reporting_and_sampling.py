import importlib.util
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def load_script_module(module_name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReportingAndSamplingTests(unittest.TestCase):
    def test_calibration_table_uses_configured_ci_label(self) -> None:
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        metrics = {
            "test": {
                "GNN": {
                    "pos_uncertainty_nll": 0.5,
                    "pos_uncertainty_ece": 0.1,
                    "pos_uncertainty_cov68": 0.7,
                    "pos_uncertainty_cov95": 0.94,
                    "pos_uncertainty_sigma_mean_m": 42.0,
                    "pos_uncertainty_nll_bootstrap_ci": {"ci_low": 0.4, "ci_high": 0.6},
                    "pos_uncertainty_ece_bootstrap_ci": {"ci_low": 0.08, "ci_high": 0.12},
                    "pos_uncertainty_cov68_bootstrap_ci": {"ci_low": 0.65, "ci_high": 0.75},
                    "pos_uncertainty_cov95_bootstrap_ci": {"ci_low": 0.92, "ci_high": 0.96},
                    "pos_uncertainty_sigma_mean_m_bootstrap_ci": {"ci_low": 40.0, "ci_high": 44.0},
                    "pos_uncertainty_bootstrap_ci_percent": 90.0,
                }
            },
            "stress_test": {},
        }
        table = assets.build_calibration_table(metrics)
        self.assertIn("[90\\% CI]", table)
        self.assertNotIn("[95\\% CI]", table)

    def test_station_subset_indices_seeded_random(self) -> None:
        outage = load_script_module("run_station_outage_sweep", "scripts/run_station_outage_sweep.py")
        idx = outage.select_subset_indices(n_total=20, max_trajectories=5, seed=1234)
        self.assertEqual(idx.shape[0], 5)
        self.assertTrue(np.all(idx[:-1] < idx[1:]))
        self.assertFalse(np.array_equal(idx, np.arange(5)))

    def test_robustness_subset_indices_seeded_random(self) -> None:
        robust = load_script_module("run_robustness_sweep", "scripts/run_robustness_sweep.py")
        idx = robust.select_subset_indices(n_total=25, max_trajectories=7, seed=5678)
        self.assertEqual(idx.shape[0], 7)
        self.assertTrue(np.all(idx[:-1] < idx[1:]))
        self.assertFalse(np.array_equal(idx, np.arange(7)))

    def test_uncertainty_bootstrap_is_trajectory_clustered(self) -> None:
        eval_mod = load_script_module("evaluate_models", "scripts/evaluate_models.py")

        def run_case(t_steps: int) -> float:
            states = np.zeros((2, t_steps, 6), dtype=np.float64)
            pred = np.zeros_like(states)
            pred[1, :, :3] = 3.0  # always under-covered for 68% and 95%
            logvar = np.zeros_like(states)  # sigma=1 for all channels
            metrics, _ = eval_mod.uncertainty_diagnostics(
                states=states,
                pred=pred,
                logvar=logvar,
                n_bootstrap=400,
                ci_percent=95.0,
                seed=42,
            )
            ci = metrics["pos_uncertainty_cov68_bootstrap_ci"]
            return float(ci["ci_high"] - ci["ci_low"])

        width_short = run_case(t_steps=1)
        width_long = run_case(t_steps=100)
        # Under trajectory-cluster bootstrap, duplicating time points per trajectory
        # should not dramatically tighten the CI.
        self.assertGreater(width_short, 0.8)
        self.assertGreater(width_long, 0.8)
        self.assertLess(abs(width_short - width_long), 0.15)

    def test_seed_table_withholds_misaligned_auxiliary_run(self) -> None:
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        csv_path = REPO_ROOT / "results" / "seed_sweep" / "seed_sweep_metrics.csv"
        summary_paths = [
            REPO_ROOT / "results" / "seed_suite_innovation_public" / "benchmark_seed_summary.csv",
            REPO_ROOT / "results" / "seed_suite" / "benchmark_seed_summary.csv",
        ]
        backups = {
            path: path.read_text(encoding="utf-8")
            for path in summary_paths
            if path.exists()
        }
        for path in summary_paths:
            if path.exists():
                path.unlink()
        metrics = {
            "test": {"HybridGNN": {"pos_rmse_m": 6176.24}},
            "stress_test": {"HybridGNN": {"pos_rmse_m": 6804.69}},
        }
        try:
            table = assets.build_seed_table(csv_path, metrics)
            self.assertIn("withheld from the canonical packet", table)
        finally:
            for path, content in backups.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

    def test_seed_table_prefers_benchmark_seed_summary_when_present(self) -> None:
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        summary_path = REPO_ROOT / "results" / "seed_suite_innovation_public" / "benchmark_seed_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        backup = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None
        try:
            summary_path.write_text(
                "\n".join(
                    [
                        "scenario,metric,mean,std,ci_low,ci_high,n_seeds",
                        "test,pos_rmse_m,6172.25,0.82,6171.50,6173.13,3",
                        "stress_test,pos_rmse_m,6836.50,14.24,6822.40,6850.88,3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            table = assets.build_seed_table(REPO_ROOT / "results" / "seed_sweep" / "seed_sweep_metrics.csv", metrics={})
            self.assertIn("Repeated-seed position-RMSE summary", table)
            self.assertIn("6172.25", table)
            self.assertIn("6836.50", table)
        finally:
            if backup is None:
                summary_path.unlink(missing_ok=True)
            else:
                summary_path.write_text(backup, encoding="utf-8")

    def test_seed_suite_filters_progress_rows_to_model_and_requested_scenarios(self) -> None:
        sweep = load_script_module("run_benchmark_seed_sweep", "scripts/run_benchmark_seed_sweep.py")
        df = pd.DataFrame(
            [
                {"seed": 41, "scenario": "test", "model": "InnovationHybridGNN"},
                {"seed": 41, "scenario": "high_drag_test", "model": "InnovationHybridGNN"},
                {"seed": 42, "scenario": "stress_test", "model": "InnovationHybridGNN"},
                {"seed": 99, "scenario": "test", "model": "OtherModel"},
            ]
        )
        filtered = sweep.filter_progress_rows(
            df,
            model_name="InnovationHybridGNN",
            requested_scenarios=["test", "stress_test"],
        )
        self.assertEqual(filtered.shape[0], 2)
        self.assertEqual(set(filtered["scenario"]), {"test", "stress_test"})
        self.assertEqual(set(filtered["model"]), {"InnovationHybridGNN"})

    def test_seed_suite_payload_reports_completed_scope(self) -> None:
        sweep = load_script_module("run_benchmark_seed_sweep", "scripts/run_benchmark_seed_sweep.py")
        df = pd.DataFrame(
            [
                {"seed": 41, "scenario": "test", "model": "InnovationHybridGNN"},
                {"seed": 43, "scenario": "stress_test", "model": "InnovationHybridGNN"},
                {"seed": 42, "scenario": "test", "model": "InnovationHybridGNN"},
            ]
        )
        payload = sweep.build_seed_suite_payload(
            df=df,
            model_name="InnovationHybridGNN",
            requested_seeds=[43],
            requested_scenarios=["test", "stress_test"],
            output_dir=REPO_ROOT / "results" / "seed_suite",
        )
        self.assertEqual(payload["requested_seeds"], [43])
        self.assertEqual(payload["completed_seeds"], [41, 42, 43])
        self.assertEqual(payload["completed_scenarios"], ["test", "stress_test"])
        self.assertEqual(payload["n_rows"], 3)

    def test_coverage_runtime_table_expands_for_public_replays(self) -> None:
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        metrics = {
            "test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.79, "fraction_steps_one_visibility": 0.20, "fraction_steps_two_plus_visibility": 0.01, "mean_visible_stations_per_step": 0.21}, "runtime_sec": {"baseline_cache_build_sec": 1.0, "GNN_inference_sec": 2.0, "HybridGNN_inference_sec": 3.0, "InnovationHybridGNN_inference_sec": 4.0}, "evaluation_window": {"start_step_inclusive": 11, "evaluated_steps": 109}}},
            "stress_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.80, "fraction_steps_one_visibility": 0.19, "fraction_steps_two_plus_visibility": 0.01, "mean_visible_stations_per_step": 0.20}, "runtime_sec": {"baseline_cache_build_sec": 1.5, "GNN_inference_sec": 2.5, "HybridGNN_inference_sec": 3.5, "InnovationHybridGNN_inference_sec": 4.5}, "evaluation_window": {"start_step_inclusive": 11, "evaluated_steps": 109}}},
            "public_catalog_replay_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.76, "fraction_steps_one_visibility": 0.15, "fraction_steps_two_plus_visibility": 0.09, "mean_visible_stations_per_step": 0.32}, "runtime_sec": {"baseline_cache_build_sec": 5.0, "GNN_inference_sec": 6.0, "HybridGNN_inference_sec": 7.0, "InnovationHybridGNN_inference_sec": 8.0}, "evaluation_window": {"start_step_inclusive": 11, "evaluated_steps": 109}}},
            "satnogs_observation_replay_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.92, "fraction_steps_one_visibility": 0.08, "fraction_steps_two_plus_visibility": 0.0, "mean_visible_stations_per_step": 0.08}, "runtime_sec": {"baseline_cache_build_sec": 9.0, "GNN_inference_sec": 10.0, "HybridGNN_inference_sec": 11.0, "InnovationHybridGNN_inference_sec": 12.0}, "evaluation_window": {"start_step_inclusive": 11, "evaluated_steps": 109}}},
            "satnogs_observation_replay_stress_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.94, "fraction_steps_one_visibility": 0.05, "fraction_steps_two_plus_visibility": 0.01, "mean_visible_stations_per_step": 0.07}, "runtime_sec": {"baseline_cache_build_sec": 13.0, "GNN_inference_sec": 14.0, "HybridGNN_inference_sec": 15.0, "InnovationHybridGNN_inference_sec": 16.0}, "evaluation_window": {"start_step_inclusive": 11, "evaluated_steps": 109}}},
        }
        table = assets.build_coverage_runtime_table(metrics)
        self.assertIn("Public replay", table)
        self.assertIn("Obs. replay", table)
        self.assertIn("Obs. replay stress", table)
        self.assertIn("0.0900", table)

    def test_benchmark_suite_table_excludes_satnogs_obs_replay_and_keeps_divergence_labels(self) -> None:
        # SatNOGS observation-window replay is now validated in a dedicated
        # time-aligned table; the benchmark suite must not carry the superseded
        # failure-only SatNOGS rows, but must still surface EKF and the
        # Diverged label for genuinely divergent non-SatNOGS scenarios.
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        metrics = {
            "high_drag_test": {
                "EKF": {"pos_rmse_m": 10.0, "diverged": False},
                "UKF": {"pos_rmse_m": 1.0e20, "diverged": True},
                "AUKF": {"pos_rmse_m": 12.0, "diverged": False},
                "KalmanNetLike": {"pos_rmse_m": 13.0, "diverged": False},
                "NoGraphResidual": {"pos_rmse_m": 14.0, "diverged": False},
                "LearnedNoiseAdaptive": {"pos_rmse_m": 15.0, "diverged": False},
                "HybridGNN": {"pos_rmse_m": 16.0, "diverged": False},
                "InnovationHybridGNN": {"pos_rmse_m": 17.0, "diverged": False},
            },
            "satnogs_observation_replay_test": {
                "EKF": {"pos_rmse_m": 20.0, "diverged": False},
                "UKF": {"pos_rmse_m": 1.0e20, "diverged": True},
                "AUKF": {"pos_rmse_m": 1.0e18, "diverged": True},
                "KalmanNetLike": {"pos_rmse_m": 21.0, "diverged": False},
                "NoGraphResidual": {"pos_rmse_m": 1.0e16, "diverged": True},
                "LearnedNoiseAdaptive": {"pos_rmse_m": 1.0e16, "diverged": True},
                "HybridGNN": {"pos_rmse_m": 1.0e16, "diverged": True},
                "InnovationHybridGNN": {"pos_rmse_m": 1.0e16, "diverged": True},
            },
        }
        table = assets.build_benchmark_suite_table(metrics)
        self.assertIn("EKF RMSE [m]", table)
        self.assertIn("Diverged", table)
        self.assertIn("High-drag shift", table)
        self.assertNotIn("SatNOGS observation replay &", table)
        self.assertIn("tab:satnogs_timefix_validation", table)

    def test_satnogs_timefix_validation_table_reports_classical_and_wls(self) -> None:
        import json
        import tempfile

        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        metrics = {
            "satnogs_observation_replay_test": {
                "EKF": {"pos_rmse_m": 4485.78, "diverged": False, "num_diverged_trajectories": 0},
                "UKF": {"pos_rmse_m": 1.2e18, "diverged": True, "num_diverged_trajectories": 1},
                "AUKF": {"pos_rmse_m": 129964.73, "diverged": False, "num_diverged_trajectories": 0},
                "_best_classical_method": "EKF",
            },
            "satnogs_observation_replay_stress_test": {
                "EKF": {"pos_rmse_m": 3460.63, "diverged": False, "num_diverged_trajectories": 0},
                "UKF": {"pos_rmse_m": 3790.67, "diverged": False, "num_diverged_trajectories": 0},
                "AUKF": {"pos_rmse_m": 3439.72, "diverged": False, "num_diverged_trajectories": 0},
                "_best_classical_method": "AUKF",
            },
        }
        csv_text = (
            "scenario,trajectories,fit_success_rate,best_recursive_observed_method,"
            "wls_gain_vs_best_recursive_observed_percent,wls_gain_vs_best_recursive_all_step_percent,"
            "batchwls_all_step_pos_rmse_m,batchwls_observed_step_pos_rmse_m\n"
            "satnogs_observation_replay_test,48,1.0,EKF,98.66,93.12,308.80,127.09\n"
            "satnogs_observation_replay_stress_test,16,1.0,AUKF,82.14,83.50,567.49,142.42\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "metrics.json"
            cpath = Path(tmp) / "wls.csv"
            jpath.write_text(json.dumps(metrics), encoding="utf-8")
            cpath.write_text(csv_text, encoding="utf-8")
            table = assets.build_satnogs_timefix_validation_table(jpath, cpath)
        self.assertIn("tab:satnogs_timefix_validation", table)
        self.assertIn("SatNOGS observation replay", table)
        self.assertIn("SatNOGS observation replay stress", table)
        # Time-aligned test slice: EKF finite at km scale, UKF divergent, WLS strong.
        self.assertIn("4485.78", table)
        self.assertIn("Diverged (1 traj.)", table)
        self.assertIn("308.80", table)
        self.assertIn("127.09", table)
        self.assertIn("567.49", table)
        # Honest scope caveats must be present in the caption.
        self.assertIn("not decoded real RF-measurement orbit determination", table)
        self.assertIn("not a flight validation", table)
        self.assertIn("Learned estimators are not scored on this slice", table)

    def test_public_data_summary_table_uses_manifest_and_coverage(self) -> None:
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        metrics = {
            "semi_real_replay_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.70, "fraction_steps_one_visibility": 0.20, "fraction_steps_two_plus_visibility": 0.10}}},
            "public_catalog_replay_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.76, "fraction_steps_one_visibility": 0.15, "fraction_steps_two_plus_visibility": 0.09}}},
            "satnogs_observation_replay_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.80, "fraction_steps_one_visibility": 0.15, "fraction_steps_two_plus_visibility": 0.05}}},
            "satnogs_observation_replay_stress_test": {"_meta": {"coverage": {"fraction_steps_zero_visibility": 0.90, "fraction_steps_one_visibility": 0.08, "fraction_steps_two_plus_visibility": 0.02}}},
        }
        dataset_manifest = {
            "semi_real_replay_test": {"samples": 48, "distinct_source_satellites": 48, "distinct_station_bank_members": 8, "coverage": metrics["semi_real_replay_test"]["_meta"]["coverage"]},
            "public_catalog_replay_test": {"samples": 48, "distinct_source_satellites": 48, "distinct_station_bank_members": 8, "coverage": metrics["public_catalog_replay_test"]["_meta"]["coverage"]},
            "satnogs_observation_replay_test": {"samples": 48, "distinct_source_satellites": 37, "distinct_station_bank_members": 15, "coverage": metrics["satnogs_observation_replay_test"]["_meta"]["coverage"], "observation_station_bank": {"source_pool_count": 80}},
            "satnogs_observation_replay_stress_test": {"samples": 16, "distinct_source_satellites": 13, "distinct_station_bank_members": 9, "coverage": metrics["satnogs_observation_replay_stress_test"]["_meta"]["coverage"], "observation_station_bank": {"source_pool_count": 80}},
        }
        public_manifest = {"catalog": {"count": 15110}, "observations": {"count": 149}}
        table = assets.build_public_data_summary_table(metrics, dataset_manifest, public_manifest)
        self.assertIn("Public-data and replay-slice summary", table)
        self.assertIn("15110", table)
        self.assertIn("80", table)
        self.assertIn("SatNOGS observation replay stress", table)

    def test_benchmark_task_tables_humanize_metric_labels(self) -> None:
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        task_table = assets.build_benchmark_task_table(
            {
                "tasks": [
                    {
                        "name": "method_selection",
                        "train_scenarios": ["test"],
                        "eval_scenarios": ["satnogs_observation_replay_test"],
                        "metrics": ["mean_regret_m", "oracle_match_rate"],
                    }
                ]
            }
        )
        self.assertIn("Method selection", task_table)
        self.assertIn("mean regret", task_table)
        self.assertIn("oracle match", task_table)
        self.assertNotIn("mean\\_regret\\_m", task_table)

    def test_method_selection_table_is_withheld_after_timefix(self) -> None:
        # The public-observation-window method-selection summary was not
        # re-derived from the time-aligned SatNOGS replay, so the table is
        # intentionally withheld and absent from the assembled manuscript. Even
        # when a stale large-regret summary CSV is present, the builder must
        # return the withheld notice rather than reintroduce the superseded
        # numbers, and that notice must be scientific prose only -- no file or
        # path references that would leak code structure into the paper.
        assets = load_script_module("build_paper_assets", "scripts/build_paper_assets.py")
        summary_path = REPO_ROOT / "results" / "benchmark_tasks" / "method_selection_summary_test.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            "\n".join(
                [
                    "selector,scope,n_trajectories,mean_selected_rmse_m,divergence_avoidance_rate,oracle_match_rate,mean_regret_m",
                    "Always AUKF,combined,64,1.0,0.9,0.2,10677034662.204304",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            table = assets.build_method_selection_table(summary_path)
        finally:
            summary_path.unlink(missing_ok=True)
        # The withheld notice is a LaTeX comment block explaining the omission.
        self.assertIn("not reported in this manuscript", table)
        self.assertIn("time-aligned SatNOGS", table)
        self.assertTrue(
            all(
                line.startswith("%")
                for line in table.splitlines()
                if line.strip()
            ),
            msg="withheld notice must be emitted entirely as LaTeX comments",
        )
        # The notice must be scientific prose only: no file/path or code
        # structure references may leak into the paper artifact. This guards
        # against the stale "tables/...tex / \\input / paper/main.tex" comment
        # block ever returning on regeneration.
        for forbidden in ("tables/", "paper/", "\\input", "main.tex", ".tex"):
            self.assertNotIn(
                forbidden,
                table,
                msg=f"withheld notice must not reference {forbidden!r}",
            )
        # The stale numeric row and rendered table machinery must not return.
        self.assertNotIn("1.07e+10", table)
        self.assertNotIn("\\begin{table}", table)
        self.assertNotIn("tab:method_selection", table)
        # And the table must remain absent from the assembled manuscript.
        main_tex = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        self.assertNotIn("tables/method_selection", main_tex)


if __name__ == "__main__":
    unittest.main()
