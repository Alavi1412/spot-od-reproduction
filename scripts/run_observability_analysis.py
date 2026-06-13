#!/usr/bin/env python
"""Compute measurement-information diagnostics for SPOT-OD scenarios."""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from gnn_state_estimation.coordinates import StationGeometry, line_of_sight_measurement
from gnn_state_estimation.dataset import STATE_SCALE
from gnn_state_estimation.simulation import MeasurementNoiseConfig
from gnn_state_estimation.utils.io import load_yaml


DEFAULT_SCENARIOS = (
    "test",
    "stress_test",
    "public_catalog_replay_test",
    "satnogs_observation_replay_test",
    "satnogs_observation_replay_stress_test",
)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def scenario_measurement_std(cfg: dict[str, Any], scenario_name: str) -> np.ndarray:
    sim_cfg = cfg["simulation"]
    merged = dict(sim_cfg["measurement_noise"])
    if scenario_name == "stress_test":
        merged = deep_update(
            merged,
            cfg.get("stress_simulation_overrides", {}).get("measurement_noise", {}),
        )
    suite_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario_name, {})
    overrides = suite_cfg.get("overrides", {})
    if "measurement_noise" in overrides:
        merged = deep_update(merged, overrides["measurement_noise"])
    return MeasurementNoiseConfig(**merged).std_vector.astype(np.float64)


def stations_from_npz(data: np.lib.npyio.NpzFile) -> tuple[StationGeometry, ...]:
    llh = np.asarray(data["station_llh"], dtype=np.float64)[0]
    names = [f"station_{idx}" for idx in range(llh.shape[0])]
    if "station_name" in data.files:
        names = [str(name) for name in np.asarray(data["station_name"])[0]]
    return tuple(
        StationGeometry(
            name=names[idx],
            lat_deg=float(np.rad2deg(llh[idx, 0])),
            lon_deg=float(np.rad2deg(llh[idx, 1])),
            alt_m=float(llh[idx, 2]),
            min_elevation_deg=-90.0,
        )
        for idx in range(llh.shape[0])
    )


def wrapped_angle_difference(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def scaled_measurement_jacobian(
    state: np.ndarray,
    station: StationGeometry,
    time_s: float,
    measurement_std: np.ndarray,
    *,
    fd_relative_step: float = 1.0e-6,
) -> np.ndarray:
    """Finite-difference d(measurement/std)/d(state/state_scale)."""
    state_scale = STATE_SCALE.astype(np.float64)
    jac = np.zeros((4, 6), dtype=np.float64)
    for col, scale in enumerate(state_scale):
        step = max(float(scale) * fd_relative_step, 1.0e-6)
        plus = np.asarray(state, dtype=np.float64).copy()
        minus = np.asarray(state, dtype=np.float64).copy()
        plus[col] += step
        minus[col] -= step
        z_plus, _ = line_of_sight_measurement(plus, station, float(time_s))
        z_minus, _ = line_of_sight_measurement(minus, station, float(time_s))
        diff = z_plus - z_minus
        diff[1] = wrapped_angle_difference(float(diff[1]))
        jac[:, col] = (diff / (2.0 * step)) * scale / measurement_std
    return jac


def compute_trajectory_information(
    *,
    states: np.ndarray,
    visibility: np.ndarray,
    times: np.ndarray,
    stations: tuple[StationGeometry, ...],
    measurement_std: np.ndarray,
    eval_start: int,
    rank_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n_traj = int(states.shape[0])
    for traj_id in range(n_traj):
        gramian = np.zeros((6, 6), dtype=np.float64)
        visible_station_epochs = 0
        coverage = np.asarray(visibility[traj_id, eval_start:], dtype=np.float64).sum(axis=1)
        for step in range(eval_start, states.shape[1]):
            active = np.where(visibility[traj_id, step] >= 0.5)[0]
            for station_idx in active:
                jac = scaled_measurement_jacobian(
                    states[traj_id, step],
                    stations[int(station_idx)],
                    float(times[traj_id, step]),
                    measurement_std,
                )
                gramian += jac.T @ jac
                visible_station_epochs += 1
        eigvals = np.linalg.eigvalsh(0.5 * (gramian + gramian.T))
        eigvals = np.clip(eigvals, 0.0, None)
        rank = int(np.sum(eigvals > rank_threshold))
        positive = eigvals[eigvals > rank_threshold]
        condition = float(positive[-1] / positive[0]) if positive.size >= 2 else math.inf
        trace = float(np.trace(gramian))
        rows.append(
            {
                "traj_id": traj_id,
                "visible_station_epochs": int(visible_station_epochs),
                "measurement_rows": int(4 * visible_station_epochs),
                "fraction_zero_visibility": float(np.mean(coverage == 0.0)),
                "fraction_one_visibility": float(np.mean(coverage == 1.0)),
                "fraction_two_plus_visibility": float(np.mean(coverage >= 2.0)),
                "rank": rank,
                "rank6": int(rank >= 6),
                "info_log10_pdet": float(np.sum(np.log10(eigvals + 1.0e-12))),
                "info_log10_trace": float(np.log10(trace + 1.0e-12)),
                "info_log10_condition": float(np.log10(condition)) if math.isfinite(condition) else math.inf,
                "min_eig": float(eigvals[0]),
                "max_eig": float(eigvals[-1]),
            }
        )
    return pd.DataFrame(rows)


def summarize_observability(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_df in trajectory_df.groupby("scenario", sort=False):
        rows.append(
            {
                "scenario": scenario_name,
                "n_trajectories": int(scenario_df.shape[0]),
                "median_visible_station_epochs": float(scenario_df["visible_station_epochs"].median()),
                "median_measurement_rows": float(scenario_df["measurement_rows"].median()),
                "rank6_fraction": float(scenario_df["rank6"].mean()),
                "median_rank": float(scenario_df["rank"].median()),
                "median_info_log10_pdet": float(scenario_df["info_log10_pdet"].median()),
                "median_info_log10_trace": float(scenario_df["info_log10_trace"].median()),
                "median_info_log10_condition": float(scenario_df["info_log10_condition"].replace([np.inf, -np.inf], np.nan).median()),
                "median_fraction_zero_visibility": float(scenario_df["fraction_zero_visibility"].median()),
                "median_fraction_two_plus_visibility": float(scenario_df["fraction_two_plus_visibility"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_correlation_frame(
    trajectory_df: pd.DataFrame,
    *,
    trajectory_errors_path: Path,
    method_selection_details_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trajectory_errors_path.exists():
        traj_errors = pd.read_csv(trajectory_errors_path)
        for scenario_name, scenario_df in trajectory_df.groupby("scenario", sort=False):
            ekf = traj_errors[
                (traj_errors["scenario"] == scenario_name) & (traj_errors["method"] == "EKF")
            ][["traj_id", "traj_pos_rmse_m"]]
            merged = scenario_df.merge(ekf, on="traj_id", how="inner", validate="one_to_one")
            if merged.shape[0] >= 3 and merged["traj_pos_rmse_m"].nunique() > 1:
                corr, pvalue = spearmanr(merged["info_log10_pdet"], merged["traj_pos_rmse_m"])
                rows.append(
                    {
                        "scenario": scenario_name,
                        "target": "EKF trajectory RMSE",
                        "n_trajectories": int(merged.shape[0]),
                        "spearman_r": float(corr),
                        "pvalue": float(pvalue),
                    }
                )
    if method_selection_details_path.exists():
        selection = pd.read_csv(method_selection_details_path)
        selection = selection[selection["selector"] == "Always EKF"][["scenario", "traj_id", "regret_m"]]
        for scenario_name, scenario_df in trajectory_df.groupby("scenario", sort=False):
            merged = scenario_df.merge(
                selection[selection["scenario"] == scenario_name],
                on=["scenario", "traj_id"],
                how="inner",
                validate="one_to_one",
            )
            if merged.shape[0] >= 3 and merged["regret_m"].nunique() > 1:
                corr, pvalue = spearmanr(merged["info_log10_pdet"], merged["regret_m"])
                rows.append(
                    {
                        "scenario": scenario_name,
                        "target": "Always-EKF oracle regret",
                        "n_trajectories": int(merged.shape[0]),
                        "spearman_r": float(corr),
                        "pvalue": float(pvalue),
                    }
                )
    return pd.DataFrame(rows)


def build_observability_figure(
    trajectory_df: pd.DataFrame,
    *,
    trajectory_errors_path: Path,
    out_path: Path,
) -> None:
    if not trajectory_errors_path.exists():
        return
    traj_errors = pd.read_csv(trajectory_errors_path)
    ekf = traj_errors[traj_errors["method"] == "EKF"][["scenario", "traj_id", "traj_pos_rmse_m"]]
    plot_df = trajectory_df.merge(ekf, on=["scenario", "traj_id"], how="inner", validate="one_to_one")
    if plot_df.empty:
        return
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(7.2, 4.3))
    sns.scatterplot(
        data=plot_df,
        x="info_log10_pdet",
        y="traj_pos_rmse_m",
        hue="scenario",
        style="scenario",
        s=38,
        alpha=0.82,
    )
    plt.yscale("log")
    plt.xlabel("State-scaled measurement information, log10 pseudo-determinant")
    plt.ylabel("EKF trajectory position RMSE [m]")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--output-dir", type=str, default="results/observability")
    parser.add_argument("--rank-threshold", type=float, default=1.0e-6)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    repo_root = Path(args.config).resolve().parent.parent
    data_dir = repo_root / cfg["data"]["output_dir"]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_start = max(int(cfg["training"]["window_size"]) - 1, 0)

    frames: list[pd.DataFrame] = []
    for scenario_name in [item.strip() for item in args.scenarios.split(",") if item.strip()]:
        data_path = data_dir / f"{scenario_name}.npz"
        if not data_path.exists():
            continue
        with np.load(data_path, allow_pickle=True) as data:
            measurement_std = scenario_measurement_std(cfg, scenario_name)
            scenario_frame = compute_trajectory_information(
                states=np.asarray(data["states"], dtype=np.float64),
                visibility=np.asarray(data["visibility"], dtype=np.float64),
                times=np.asarray(data["times"], dtype=np.float64),
                stations=stations_from_npz(data),
                measurement_std=measurement_std,
                eval_start=eval_start,
                rank_threshold=float(args.rank_threshold),
            )
        scenario_frame.insert(0, "scenario", scenario_name)
        frames.append(scenario_frame)

    trajectory_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary_df = summarize_observability(trajectory_df) if not trajectory_df.empty else pd.DataFrame()
    correlation_df = build_correlation_frame(
        trajectory_df,
        trajectory_errors_path=repo_root / "results/trajectory_errors.csv",
        method_selection_details_path=repo_root / "results/benchmark_tasks/method_selection_details.csv",
    )

    trajectory_df.to_csv(output_dir / "observability_trajectory.csv", index=False)
    summary_df.to_csv(output_dir / "observability_summary.csv", index=False)
    correlation_df.to_csv(output_dir / "observability_correlations.csv", index=False)
    build_observability_figure(
        trajectory_df,
        trajectory_errors_path=repo_root / "results/trajectory_errors.csv",
        out_path=output_dir / "observability_vs_ekf_error.png",
    )


if __name__ == "__main__":
    main()
