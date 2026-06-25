#!/usr/bin/env python
"""Build the main-text retained-candidate residual-refinement figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


GRAPH_DIR = Path(
    "results/"
    "trajectory_candidate_graph_attention_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
MEAN_DIR = Path(
    "results/"
    "trajectory_candidate_mean_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
LOCAL_DIR = Path(
    "results/"
    "trajectory_candidate_local_nodeomit_residual_refine_val53_ensemble3_2111_2117_2129_"
    "newfresh151157163167_20260625"
)
OUTPUT_PATH = Path("paper/figures/trajectory_residual_refine_gain_distribution_val53.png")

TIERS = (
    ("all_eval_non_development", "All non-dev"),
    ("fresh_extra", "Fresh seeds"),
)

METHODS = (
    ("best", "Best retained", "#6C757D"),
    ("local", "Edge-only local", "#2A9D8F"),
    ("mean", "Edge-only mean", "#457B9D"),
    ("attention", "Edge-only attention", "#E76F51"),
)


@dataclass(frozen=True)
class TierAggregate:
    rows: int
    observed_steps: int
    best_rmse_m: float
    local_rmse_m: float | None = None
    mean_rmse_m: float | None = None
    attention_rmse_m: float | None = None
    gain_vs_best_percent: float | None = None
    row_gains_percent: tuple[float, ...] = ()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _has_tier(row: dict[str, str], tier: str) -> bool:
    return tier in str(row.get("tier_flags", "")).split(";")


def _rmse(sse_values: list[float], step_values: list[int]) -> float:
    steps = sum(step_values)
    if steps <= 0:
        return float("nan")
    return math.sqrt(sum(sse_values) / steps)


def _aggregate_from_rows(rows: list[dict[str, str]], tier: str) -> TierAggregate:
    tier_rows = [row for row in rows if _has_tier(row, tier)]
    if not tier_rows:
        raise ValueError(f"No rows matched tier {tier!r}")

    selected_sse = [float(row["selected_observed_step_sse"]) for row in tier_rows]
    selected_steps = [int(float(row["selected_observed_steps"])) for row in tier_rows]
    best_sse = [float(row["best_single_trajectory_observed_step_sse"]) for row in tier_rows]
    best_steps = [int(float(row["best_single_trajectory_observed_steps"])) for row in tier_rows]
    gains = tuple(float(row["gain_vs_best_single_trajectory_percent"]) for row in tier_rows)
    selected_rmse = _rmse(selected_sse, selected_steps)
    best_rmse = _rmse(best_sse, best_steps)
    return TierAggregate(
        rows=len(tier_rows),
        observed_steps=sum(selected_steps),
        best_rmse_m=best_rmse,
        attention_rmse_m=selected_rmse,
        gain_vs_best_percent=100.0 * (best_rmse - selected_rmse) / best_rmse,
        row_gains_percent=gains,
    )


def _validate_summary(summary: dict, tier: str, aggregate: TierAggregate) -> None:
    tier_summary = summary.get("aggregate_tiers", {}).get(tier)
    if not isinstance(tier_summary, dict):
        raise ValueError(f"summary.json missing aggregate_tiers.{tier}")
    checks = {
        "rows": float(aggregate.rows),
        "observed_steps": float(aggregate.observed_steps),
        "best_single_observed_step_rmse_m": aggregate.best_rmse_m,
        "selector_observed_step_rmse_m": aggregate.attention_rmse_m,
    }
    for key, actual in checks.items():
        expected = float(tier_summary[key])
        tolerance = 1.0e-6 if key in {"rows", "observed_steps"} else 5.0e-4
        if abs(actual - expected) > tolerance:
            raise ValueError(
                f"{tier} {key} mismatch: rows.csv={actual:.9g}, summary.json={expected:.9g}"
            )


def _load_variant(dir_path: Path) -> tuple[dict[str, TierAggregate], dict]:
    rows_path = dir_path / "rows.csv"
    summary_path = dir_path / "summary.json"
    if not rows_path.exists():
        raise FileNotFoundError(rows_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows = _read_rows(rows_path)
    summary = _read_json(summary_path)
    aggregates = {tier: _aggregate_from_rows(rows, tier) for tier, _ in TIERS}
    for tier, aggregate in aggregates.items():
        _validate_summary(summary, tier, aggregate)
    return aggregates, summary


def _combined_aggregates(
    graph_dir: Path,
    local_dir: Path,
    mean_dir: Path,
) -> tuple[dict[str, TierAggregate], dict]:
    graph_aggs, graph_summary = _load_variant(graph_dir)
    local_aggs, local_summary = _load_variant(local_dir)
    mean_aggs, mean_summary = _load_variant(mean_dir)
    combined: dict[str, TierAggregate] = {}
    for tier, _ in TIERS:
        graph = graph_aggs[tier]
        local = local_aggs[tier]
        mean = mean_aggs[tier]
        if (
            graph.rows != local.rows
            or graph.rows != mean.rows
            or graph.observed_steps != local.observed_steps
            or graph.observed_steps != mean.observed_steps
        ):
            raise ValueError(f"{tier} graph/local/mean aggregate size mismatch")
        if abs(graph.best_rmse_m - local.best_rmse_m) > 5.0e-4 or abs(graph.best_rmse_m - mean.best_rmse_m) > 5.0e-4:
            raise ValueError(f"{tier} graph/local/mean best-retained RMSE mismatch")
        combined[tier] = TierAggregate(
            rows=graph.rows,
            observed_steps=graph.observed_steps,
            best_rmse_m=graph.best_rmse_m,
            local_rmse_m=local.attention_rmse_m,
            mean_rmse_m=mean.attention_rmse_m,
            attention_rmse_m=graph.attention_rmse_m,
            gain_vs_best_percent=graph.gain_vs_best_percent,
            row_gains_percent=graph.row_gains_percent,
        )
    metadata = {
        "graph_prediction_mode": graph_summary.get("prediction_mode"),
        "graph_layers": graph_summary.get("graph_layers"),
        "graph_layer_type": graph_summary.get("graph_layer_type"),
        "graph_message_passing_enabled": graph_summary.get("message_passing_enabled"),
        "local_graph_layers": local_summary.get("graph_layers"),
        "local_message_passing_enabled": local_summary.get("message_passing_enabled"),
        "mean_graph_layers": mean_summary.get("graph_layers"),
        "mean_graph_layer_type": mean_summary.get("graph_layer_type"),
        "mean_message_passing_enabled": mean_summary.get("message_passing_enabled"),
    }
    return combined, metadata


def _format_rmse(value: float) -> str:
    return f"{value:.0f}"


def _plot_bars(ax: plt.Axes, aggregates: dict[str, TierAggregate]) -> None:
    x = np.arange(len(TIERS), dtype=float)
    width = 0.18
    offsets = {"best": -1.5 * width, "local": -0.5 * width, "mean": 0.5 * width, "attention": 1.5 * width}
    all_values: list[float] = []
    for method, label, color in METHODS:
        values = []
        for tier, _ in TIERS:
            agg = aggregates[tier]
            values.append(
                {
                    "best": agg.best_rmse_m,
                    "local": agg.local_rmse_m,
                    "mean": agg.mean_rmse_m,
                    "attention": agg.attention_rmse_m,
                }[method]
            )
        all_values.extend(float(value) for value in values)
        bars = ax.bar(
            x + offsets[method],
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value) + 7.0,
                _format_rmse(float(value)),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in TIERS])
    ax.set_ylabel("Observed-step RMSE [m]\n(lower is better)")
    ymax = max(all_values)
    ax.set_ylim(0, math.ceil((ymax + 80.0) / 50.0) * 50.0)
    ax.grid(axis="y", linewidth=0.35, alpha=0.45)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=7.6)
    ax.set_title("Edge-only residual-refinement ablation RMSE", fontsize=10.5)


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    fields = ("source_name", "seed", "split", "scenario", "trajectory_row", "trajectory_index")
    return tuple(str(row[field]) for field in fields)


def _paired_attention_advantages(attention_dir: Path, reference_dir: Path, tier: str) -> np.ndarray:
    attention_rows = [row for row in _read_rows(attention_dir / "rows.csv") if _has_tier(row, tier)]
    reference_by_key = {_row_key(row): row for row in _read_rows(reference_dir / "rows.csv")}
    advantages = []
    for row in attention_rows:
        reference = reference_by_key[_row_key(row)]
        attention_rmse = float(row["selected_observed_step_rmse_m"])
        reference_rmse = float(reference["selected_observed_step_rmse_m"])
        advantages.append(100.0 * (reference_rmse - attention_rmse) / reference_rmse)
    return np.asarray(advantages, dtype=float)


def _plot_advantage_distribution(
    ax: plt.Axes,
    graph_dir: Path,
    local_dir: Path,
    mean_dir: Path,
) -> None:
    rng = np.random.default_rng(20260625)
    variants = [
        ("All non-dev vs local", "all_eval_non_development", local_dir, "#2A9D8F", 3.05),
        ("All non-dev vs mean", "all_eval_non_development", mean_dir, "#457B9D", 2.65),
        ("Fresh vs local", "fresh_extra", local_dir, "#2A9D8F", 1.25),
        ("Fresh vs mean", "fresh_extra", mean_dir, "#457B9D", 0.85),
    ]
    box_data = []
    positions = []
    colors = []
    labels = []
    for label, tier, reference_dir, color, position in variants:
        advantages = _paired_attention_advantages(graph_dir, reference_dir, tier)
        box_data.append(advantages)
        positions.append(position)
        colors.append(color)
        labels.append(f"{label}\n(n={len(advantages)})")
        jitter = rng.normal(0.0, 0.035, size=len(advantages))
        ax.scatter(
            advantages,
            np.full_like(advantages, position) + jitter,
            s=9,
            color=color,
            alpha=0.28,
            linewidths=0,
        )
    box = ax.boxplot(
        box_data,
        positions=positions,
        vert=False,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.42)
        patch.set_edgecolor(color)
    ax.axvline(0.0, color="#333333", linewidth=0.9)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Trajectory-row advantage of edge-only attention over reference [%]")
    ax.set_xlim(-170, 100)
    ax.grid(axis="x", linewidth=0.35, alpha=0.45)
    ax.set_title("Attention clearly separates from local; attention-vs-mean is weaker", fontsize=10.5)


def build_figure(graph_dir: Path, local_dir: Path, mean_dir: Path, output_path: Path) -> dict:
    aggregates, metadata = _combined_aggregates(graph_dir, local_dir, mean_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.1, 5.15),
        gridspec_kw={"height_ratios": [1.0, 1.15]},
        constrained_layout=False,
    )
    _plot_bars(axes[0], aggregates)
    _plot_advantage_distribution(axes[1], graph_dir, local_dir, mean_dir)

    all_non_dev = aggregates["all_eval_non_development"]
    fresh = aggregates["fresh_extra"]
    all_attention_vs_local = (
        100.0
        * (float(all_non_dev.local_rmse_m) - float(all_non_dev.attention_rmse_m))
        / float(all_non_dev.local_rmse_m)
    )
    fresh_attention_vs_local = (
        100.0 * (float(fresh.local_rmse_m) - float(fresh.attention_rmse_m)) / float(fresh.local_rmse_m)
    )
    all_attention_vs_mean = (
        100.0
        * (float(all_non_dev.mean_rmse_m) - float(all_non_dev.attention_rmse_m))
        / float(all_non_dev.mean_rmse_m)
    )
    fresh_attention_vs_mean = (
        100.0 * (float(fresh.mean_rmse_m) - float(fresh.attention_rmse_m)) / float(fresh.mean_rmse_m)
    )
    fig.suptitle(
        "Edge-only retained-candidate residual-refinement ablation",
        fontsize=11.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.01,
        0.012,
        (
            "Attention over no-message local: "
            f"{all_attention_vs_local:.2f}% all non-dev, {fresh_attention_vs_local:.2f}% fresh. "
            "Attention over mean graph is weaker/mixed: "
            f"{all_attention_vs_mean:.2f}% all non-dev, {fresh_attention_vs_mean:.2f}% fresh."
        ),
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.965), h_pad=1.05)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", metadata={"Software": "SPOT-OD"})
    plt.close(fig)
    return {
        "output_path": str(output_path),
        "source_artifacts": [
            str(graph_dir / "rows.csv"),
            str(graph_dir / "summary.json"),
            str(local_dir / "rows.csv"),
            str(local_dir / "summary.json"),
            str(mean_dir / "rows.csv"),
            str(mean_dir / "summary.json"),
        ],
        "tiers": {
            tier: {
                "rows": aggregates[tier].rows,
                "observed_steps": aggregates[tier].observed_steps,
                "best_retained_rmse_m": aggregates[tier].best_rmse_m,
                "edge_only_local_residual_rmse_m": aggregates[tier].local_rmse_m,
                "edge_only_mean_residual_rmse_m": aggregates[tier].mean_rmse_m,
                "edge_only_attention_residual_rmse_m": aggregates[tier].attention_rmse_m,
            }
            for tier, _ in TIERS
        },
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=GRAPH_DIR)
    parser.add_argument("--local-dir", type=Path, default=LOCAL_DIR)
    parser.add_argument("--mean-dir", type=Path, default=MEAN_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = build_figure(args.graph_dir, args.local_dir, args.mean_dir, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
