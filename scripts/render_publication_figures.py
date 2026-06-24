#!/usr/bin/env python
"""Render manuscript-readable figures from existing result artifacts.

The evaluator keeps broad multi-scenario plots for release diagnostics. Those
plots can become extremely wide when included in the manuscript. This script
builds compact nominal/stress panels for the active paper figures while keeping
the underlying numeric evidence in the existing CSV/JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


SCENARIOS = {
    "test": "Nominal",
    "stress_test": "Stress",
}

METHOD_LABELS = {
    "EKF": "EKF",
    "UKF": "UKF",
    "AUKF": "AUKF",
    "GNN": "Pure GNN",
    "KalmanNetLike": "KalmanNet-like",
    "NoGraphResidual": "No-Graph Residual",
    "LearnedNoiseAdaptive": "Learned Noise Adaptive",
    "HybridGNN": "RGR-GF",
    "InnovationHybridGNN": "IDP-RGR-GF",
}

TRACKING_METHODS = [
    "EKF",
    "UKF",
    "AUKF",
    "LearnedNoiseAdaptive",
    "HybridGNN",
    "InnovationHybridGNN",
]

CALIBRATION_METHODS = [
    "GNN",
    "KalmanNetLike",
    "NoGraphResidual",
    "LearnedNoiseAdaptive",
    "HybridGNN",
    "InnovationHybridGNN",
]

BUCKETS = [
    ("vis_0_pos_rmse_m", "0 visible"),
    ("vis_1_pos_rmse_m", "1 visible"),
    ("vis_2plus_pos_rmse_m", "2+ visible"),
]

GRAPH_ANCHOR_PAIR_GATE_SWEEP_DIR = "graph_anchor_pair_gate_seed_sweep_20260623"


def _read_csv(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_metrics(results_dir: Path) -> dict:
    path = results_dir / "metrics_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json(results_dir: Path, name: str) -> dict:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_panel_data(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    out = df[df["scenario"].isin(SCENARIOS) & df["method"].isin(methods)].copy()
    out["scenario_label"] = out["scenario"].map(SCENARIOS)
    out["method_label"] = out["method"].map(METHOD_LABELS)
    order = [METHOD_LABELS[m] for m in methods if m in METHOD_LABELS]
    out["method_label"] = pd.Categorical(out["method_label"], categories=order, ordered=True)
    return out.sort_values(["scenario", "method_label"])


def _finish_shared_legend(fig: plt.Figure, axes, *, ncol: int = 3) -> None:
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.tight_layout(rect=(0, 0.13, 1, 1))


def _save(fig: plt.Figure, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return _image_info(path)


def _image_info(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path),
        "width_px": int(width),
        "height_px": int(height),
        "aspect_ratio": round(width / height, 3),
        "bytes": path.stat().st_size,
    }


def _graph_anchor_pair_gate_fallback_summary_rows() -> pd.DataFrame:
    try:
        from build_paper_assets import _GRAPH_ANCHOR_PAIR_GATE_SUMMARY_FALLBACK
    except ModuleNotFoundError:  # pragma: no cover - import mode depends on entrypoint
        from scripts.build_paper_assets import _GRAPH_ANCHOR_PAIR_GATE_SUMMARY_FALLBACK

    return pd.DataFrame(_GRAPH_ANCHOR_PAIR_GATE_SUMMARY_FALLBACK)


def _graph_anchor_pair_gate_summary_rows(results_dir: Path) -> pd.DataFrame:
    summary_path = (
        results_dir
        / GRAPH_ANCHOR_PAIR_GATE_SWEEP_DIR
        / "graph_anchor_pair_gate_seed_sweep_summary.csv"
    )
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return _graph_anchor_pair_gate_fallback_summary_rows()


def _save_graph_anchor_pair_gate_seed_sweep_aggregate(
    summary_rows: pd.DataFrame,
    output_path: Path,
) -> Path:
    from matplotlib.patches import Patch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = summary_rows.copy()
    plot_rows["seed"] = pd.to_numeric(plot_rows["seed"])
    plot_rows["gain_vs_best_candidate_all_step_percent"] = pd.to_numeric(
        plot_rows["gain_vs_best_candidate_all_step_percent"]
    )
    scenario_rank = {"process_noise_shift_test": 0, "maneuver_shift_test": 1}
    plot_rows["_scenario_rank"] = plot_rows["scenario"].map(scenario_rank).fillna(99).astype(int)
    plot_rows = plot_rows.sort_values(["seed", "_scenario_rank"]).reset_index(drop=True)
    gains = plot_rows["gain_vs_best_candidate_all_step_percent"].astype(float).to_numpy()
    labels = [
        f"s{int(row.seed)}\n{'process' if row.scenario == 'process_noise_shift_test' else 'maneuver'}"
        for row in plot_rows.itertuples(index=False)
    ]
    colors = {
        "process_noise_shift_test": "#4C78A8",
        "maneuver_shift_test": "#F58518",
    }
    bar_colors = [colors.get(str(scenario), "#777777") for scenario in plot_rows["scenario"]]

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    x = np.arange(len(plot_rows), dtype=float)
    bars = ax.bar(x, gains, color=bar_colors, width=0.74)
    for bar, gain in zip(bars, gains):
        y = gain + (0.55 if gain >= 0.0 else -0.85)
        va = "bottom" if gain >= 0.0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{gain:+.1f}",
            ha="center",
            va=va,
            fontsize=7.5,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Gain vs best candidate (%)")
    ax.set_title("GraphAnchorPairGate five-seed sweep: all scenario-seed rows", fontsize=10.5)
    ax.grid(axis="y", alpha=0.24)
    ax.margins(x=0.02)
    ymax = max(20.0, float(np.nanmax(gains)) + 3.0)
    ymin = min(-5.0, float(np.nanmin(gains)) - 3.0)
    ax.set_ylim(ymin, ymax)
    ax.legend(
        handles=[
            Patch(facecolor=colors["process_noise_shift_test"], label="process-noise shift"),
            Patch(facecolor=colors["maneuver_shift_test"], label="maneuver shift"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.01,
        0.01,
        "All-step center-window RMSE; retained CSV rows only. Seed 19 process shift is the single loss.",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _render_graph_anchor_pair_gate_seed_sweep_aggregate(
    results_dir: Path,
    paper_fig_dir: Path,
) -> dict:
    summary_rows = _graph_anchor_pair_gate_summary_rows(results_dir)
    output_path = paper_fig_dir / "graph_anchor_pair_gate_seed_sweep_aggregate.png"
    try:
        from build_graph_anchor_pair_gate_seed_sweep_artifacts import (
            save_seed_sweep_aggregate_figure,
        )
    except ModuleNotFoundError:
        try:
            from scripts.build_graph_anchor_pair_gate_seed_sweep_artifacts import (
                save_seed_sweep_aggregate_figure,
            )
        except ModuleNotFoundError:
            save_seed_sweep_aggregate_figure = _save_graph_anchor_pair_gate_seed_sweep_aggregate

    with plt.rc_context(plt.rcParamsDefault):
        save_seed_sweep_aggregate_figure(summary_rows, output_path)
    return _image_info(output_path)


def _render_per_step(results_dir: Path, paper_fig_dir: Path) -> dict:
    df = _filter_panel_data(_read_csv(results_dir, "per_step_errors.csv"), TRACKING_METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        panel = df[df["scenario"] == scenario]
        sns.lineplot(data=panel, x="step", y="pos_rmse_m", hue="method_label", ax=ax, linewidth=1.7)
        ax.set_title(SCENARIOS[scenario])
        ax.set_xlabel("Evaluation step")
        ax.set_ylabel("Position RMSE [m]")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    _finish_shared_legend(fig, axes, ncol=3)
    return _save(fig, paper_fig_dir / "per_step_rmse.png")


def _render_ecdf(results_dir: Path, paper_fig_dir: Path) -> dict:
    df = _filter_panel_data(_read_csv(results_dir, "trajectory_errors.csv"), TRACKING_METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        panel = df[df["scenario"] == scenario]
        sns.ecdfplot(data=panel, x="traj_pos_rmse_m", hue="method_label", ax=ax, linewidth=1.7)
        ax.set_title(SCENARIOS[scenario])
        ax.set_xscale("log")
        ax.set_xlabel("Trajectory position RMSE [m]")
        ax.set_ylabel("ECDF")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    _finish_shared_legend(fig, axes, ncol=3)
    return _save(fig, paper_fig_dir / "position_error_ecdf.png")


def _visibility_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for scenario, scenario_label in SCENARIOS.items():
        scenario_payload = metrics.get(scenario, {})
        for method in TRACKING_METHODS:
            method_payload = scenario_payload.get(method, {})
            for key, bucket_label in BUCKETS:
                if key not in method_payload:
                    continue
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": scenario_label,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "bucket": bucket_label,
                        "pos_rmse_m": float(method_payload[key]),
                    }
                )
    return pd.DataFrame(rows)


def _render_visibility(results_dir: Path, paper_fig_dir: Path) -> dict:
    df = _visibility_frame(_read_metrics(results_dir))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    palette = sns.color_palette(n_colors=len(TRACKING_METHODS))
    for ax, scenario in zip(axes, SCENARIOS):
        panel = df[df["scenario"] == scenario]
        sns.barplot(
            data=panel,
            x="bucket",
            y="pos_rmse_m",
            hue="method_label",
            ax=ax,
            palette=palette,
            errorbar=None,
        )
        ax.set_title(SCENARIOS[scenario])
        ax.set_xlabel("Visible stations")
        ax.set_ylabel("Position RMSE [m]")
        ax.set_yscale("log")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    _finish_shared_legend(fig, axes, ncol=3)
    return _save(fig, paper_fig_dir / "visibility_bucket_rmse.png")


def _render_calibration(results_dir: Path, paper_fig_dir: Path) -> dict:
    df = _filter_panel_data(_read_csv(results_dir, "uncertainty_calibration.csv"), CALIBRATION_METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        panel = df[df["scenario"] == scenario]
        sns.lineplot(
            data=panel,
            x="nominal_coverage",
            y="empirical_coverage",
            hue="method_label",
            marker="o",
            ax=ax,
            linewidth=1.5,
            markersize=4,
        )
        ax.plot([0.45, 1.0], [0.45, 1.0], color="black", linestyle="--", linewidth=1.0, label="Ideal")
        ax.set_title(SCENARIOS[scenario])
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.45, 1.0)
        ax.set_xlabel("Nominal central coverage")
        ax.set_ylabel("Empirical central coverage")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    _finish_shared_legend(fig, axes, ncol=4)
    return _save(fig, paper_fig_dir / "uncertainty_calibration.png")


def _render_aukf_r_inflation_mechanism(results_dir: Path, paper_fig_dir: Path) -> dict:
    summary = _read_json(results_dir, "force_model_mismatch_adaptation_summary.json")
    updates_path = results_dir / "force_model_mismatch_adaptation_updates.csv"
    updates = pd.read_csv(updates_path) if updates_path.exists() else pd.DataFrame()

    adapt = summary.get("aukf_adaptation_mechanism", {})
    rnis = summary.get("cross_filter_r_only_nis", {})
    obs = summary.get("observed_step_pos_rmse", {})
    gate = float(adapt.get("nis_soft_gate", 16.0))

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8))
    colors = {
        "EKF": "#2a9d8f",
        "UKF": "#457b9d",
        "AUKF": "#e76f51",
        "proposal": "#f4a261",
        "neutral": "#6c757d",
    }

    ax = axes[0, 0]
    if not updates.empty and "pre_adapt_nis" in updates:
        finite = updates["pre_adapt_nis"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            upper = max(gate * 2.5, float(finite.quantile(0.98)))
            sns.histplot(finite.clip(upper=upper), bins=28, ax=ax, color="#457b9d", edgecolor="white")
    ax.axvline(gate, color=colors["AUKF"], linestyle="--", linewidth=1.6)
    ax.text(
        0.97,
        0.90,
        (
            f"mean {float(adapt.get('mean_pre_adapt_nis', float('nan'))):.2f}\n"
            f"median {float(adapt.get('median_pre_adapt_nis', float('nan'))):.2f}\n"
            f"soft-gate hits {float(adapt.get('percent_updates_exceeding_soft_gate', float('nan'))):.1f}%"
        ),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c9c9c9"},
    )
    ax.set_title("1. Dynamics-bias innovations enter the update")
    ax.set_xlabel("Pre-adaptation NIS")
    ax.set_ylabel("Visible updates")
    ax.grid(True, linewidth=0.3, alpha=0.45)

    ax = axes[0, 1]
    stages = ["pre", "proposal", "post", "effective"]
    r_values = [
        float(adapt.get("mean_r_scale_pre", float("nan"))),
        float(adapt.get("mean_r_proposal_scale", float("nan"))),
        float(adapt.get("mean_r_scale_post", float("nan"))),
        float(adapt.get("mean_r_eff_scale", float("nan"))),
    ]
    stage_colors = [colors["neutral"], colors["proposal"], "#8ab17d", colors["AUKF"]]
    ax.plot(stages, r_values, color="#333333", linewidth=1.4, zorder=1)
    ax.scatter(stages, r_values, s=120, color=stage_colors, edgecolor="white", linewidth=1.0, zorder=2)
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.1)
    for x, y in zip(stages, r_values):
        ax.text(x, y + 0.16, f"{y:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(r_values) * 1.28)
    ax.set_title("2. AUKF interprets residuals as R evidence")
    ax.set_xlabel("AUKF adaptation stage")
    ax.set_ylabel("Mean R scale")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)

    ax = axes[1, 0]
    methods = ["EKF", "UKF", "AUKF"]
    nis_values = [
        float(rnis.get(method, {}).get("median_r_only_nis", float("nan")))
        for method in methods
    ]
    ax.bar(methods, nis_values, color=[colors[m] for m in methods], edgecolor="white")
    for x, y in zip(methods, nis_values):
        ax.text(x, y + 0.13, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(nis_values) * 1.22)
    ax.set_title("3. R-only consistency is most stressed for AUKF")
    ax.set_xlabel("Filter")
    ax.set_ylabel("Median R-only NIS")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)

    ax = axes[1, 1]
    rmse_values = [
        float(obs.get(method, {}).get("observed_step_pos_rmse_m", float("nan")))
        for method in methods
    ]
    ax.bar(methods, rmse_values, color=[colors[m] for m in methods], edgecolor="white")
    for x, y in zip(methods, rmse_values):
        ax.text(x, y + 8, f"{y:.0f} m", ha="center", va="bottom", fontsize=9)
    ax.annotate(
        "damped corrections\ntrack bias least well",
        xy=("AUKF", rmse_values[-1]),
        xytext=(0.58, 0.38),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.0},
        fontsize=9,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#c9c9c9"},
    )
    ax.set_ylim(0, max(rmse_values) * 1.22)
    ax.set_title("4. Observed-step RMSE consequence")
    ax.set_xlabel("Filter")
    ax.set_ylabel("Observed-step position RMSE [m]")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.45)

    fig.suptitle(
        "AUKF R-inflation mechanism under controlled force-model mismatch",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _save(fig, paper_fig_dir / "aukf_r_inflation_mechanism.png")


def _try_render(name: str, render_fn, results_dir: Path, paper_fig_dir: Path, errors: dict[str, str]) -> dict | None:
    try:
        return render_fn(results_dir, paper_fig_dir)
    except FileNotFoundError as exc:
        errors[name] = f"missing source artifact: {exc}"
    except Exception as exc:  # pragma: no cover - validation path
        errors[name] = repr(exc)
    return None


def render_publication_figures(
    *,
    results_dir: Path = Path("results"),
    paper_fig_dir: Path = Path("paper/figures"),
    report_path: Path = Path("results/publication_figure_render_report.json"),
) -> dict:
    sns.set_theme(style="whitegrid", context="paper")
    figures: dict[str, dict] = {}
    errors: dict[str, str] = {}
    renderers = {
        "per_step_rmse": _render_per_step,
        "position_error_ecdf": _render_ecdf,
        "visibility_bucket_rmse": _render_visibility,
        "uncertainty_calibration": _render_calibration,
        "aukf_r_inflation_mechanism": _render_aukf_r_inflation_mechanism,
        "graph_anchor_pair_gate_seed_sweep_aggregate": (
            _render_graph_anchor_pair_gate_seed_sweep_aggregate
        ),
    }
    for name, renderer in renderers.items():
        record = _try_render(name, renderer, results_dir, paper_fig_dir, errors)
        if record is not None:
            figures[name] = record
    report = {
        "source_artifacts": [
            str(results_dir / "per_step_errors.csv"),
            str(results_dir / "trajectory_errors.csv"),
            str(results_dir / "metrics_summary.json"),
            str(results_dir / "uncertainty_calibration.csv"),
            str(results_dir / "force_model_mismatch_adaptation_summary.json"),
            str(results_dir / "force_model_mismatch_adaptation_updates.csv"),
            str(
                results_dir
                / GRAPH_ANCHOR_PAIR_GATE_SWEEP_DIR
                / "graph_anchor_pair_gate_seed_sweep_summary.csv"
            ),
        ],
        "scenarios": SCENARIOS,
        "tracking_methods": {method: METHOD_LABELS[method] for method in TRACKING_METHODS},
        "calibration_methods": {method: METHOD_LABELS[method] for method in CALIBRATION_METHODS},
        "figures": figures,
        "render_errors": errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--paper-fig-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument("--report-path", type=Path, default=Path("results/publication_figure_render_report.json"))
    args = parser.parse_args()
    report = render_publication_figures(
        results_dir=args.results_dir,
        paper_fig_dir=args.paper_fig_dir,
        report_path=args.report_path,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
