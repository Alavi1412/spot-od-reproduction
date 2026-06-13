"""Benchmark-native public replay tasks for stability and method selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


PUBLIC_PACKET_SCENARIOS = (
    "semi_real_replay_test",
    "public_catalog_replay_test",
    "satnogs_observation_replay_val",
    "satnogs_observation_replay_test",
    "satnogs_observation_replay_stress_test",
)

DEFAULT_SELECTOR_METHODS = (
    "EKF",
    "UKF",
    "AUKF",
    "KalmanNetLike",
    "NoGraphResidual",
    "LearnedNoiseAdaptive",
    "HybridGNN",
    "InnovationHybridGNN",
)

DEFAULT_FEATURE_NAMES = (
    "mean_visible_stations",
    "fraction_zero_visibility",
    "fraction_one_visibility",
    "fraction_two_plus_visibility",
    "mean_innovation_energy",
    "max_innovation_energy",
    "mean_visibility_mismatch",
    "mean_prior_gap_pos_m",
    "max_prior_gap_pos_m",
    "mean_prior_gap_vel_mps",
    "is_public_catalog",
    "is_public_observation",
    "is_stress",
)


@dataclass(frozen=True)
class LogisticRiskModel:
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float
    constant_probability: float | None = None


def trajectory_diverged_mask(
    traj_pos_rmse_m: np.ndarray,
    traj_vel_rmse_mps: np.ndarray,
    *,
    pos_threshold_m: float = 1.0e8,
    vel_threshold_mps: float = 1.0e5,
) -> np.ndarray:
    pos = np.asarray(traj_pos_rmse_m, dtype=np.float64).reshape(-1)
    vel = np.asarray(traj_vel_rmse_mps, dtype=np.float64).reshape(-1)
    return (~np.isfinite(pos)) | (~np.isfinite(vel)) | (np.abs(pos) > pos_threshold_m) | (np.abs(vel) > vel_threshold_mps)


def _safe_logit(prob: float, eps: float = 1.0e-4) -> float:
    p = float(np.clip(prob, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _average_visible_stat(values: np.ndarray, visible_mask: np.ndarray, reducer: str) -> np.ndarray:
    out = np.zeros(values.shape[0], dtype=np.float64)
    for idx in range(values.shape[0]):
        active = np.asarray(values[idx])[np.asarray(visible_mask[idx], dtype=bool)]
        if active.size == 0:
            out[idx] = 0.0
            continue
        if reducer == "mean":
            out[idx] = float(np.mean(active))
        elif reducer == "max":
            out[idx] = float(np.max(active))
        else:  # pragma: no cover - guarded by caller
            raise ValueError(f"Unsupported reducer: {reducer}")
    return out


def compute_trajectory_feature_frame(
    arrays: Any,
    *,
    scenario_name: str,
    eval_start: int,
) -> pd.DataFrame:
    vis = np.asarray(arrays.visibility[:, eval_start:], dtype=np.float64)
    coverage = vis.sum(axis=2)
    n_traj = coverage.shape[0]
    features: dict[str, Any] = {
        "scenario": [scenario_name] * n_traj,
        "traj_id": np.arange(n_traj, dtype=np.int64),
        "mean_visible_stations": np.mean(coverage, axis=1),
        "fraction_zero_visibility": np.mean(coverage == 0, axis=1),
        "fraction_one_visibility": np.mean(coverage == 1, axis=1),
        "fraction_two_plus_visibility": np.mean(coverage >= 2, axis=1),
        "is_public_catalog": np.full(n_traj, 1.0 if scenario_name == "public_catalog_replay_test" else 0.0),
        "is_public_observation": np.full(n_traj, 1.0 if scenario_name.startswith("satnogs_observation_replay") else 0.0),
        "is_stress": np.full(n_traj, 1.0 if "stress" in scenario_name else 0.0),
    }

    if arrays.innovation_features is not None:
        innov = np.asarray(arrays.innovation_features[:, eval_start:], dtype=np.float64)
        visible_mask = vis >= 0.5
        features["mean_innovation_energy"] = _average_visible_stat(innov[..., 4], visible_mask, "mean")
        features["max_innovation_energy"] = _average_visible_stat(innov[..., 4], visible_mask, "max")
        features["mean_visibility_mismatch"] = np.mean(np.abs(innov[..., 5] - vis), axis=(1, 2))
    else:
        features["mean_innovation_energy"] = np.zeros(n_traj, dtype=np.float64)
        features["max_innovation_energy"] = np.zeros(n_traj, dtype=np.float64)
        features["mean_visibility_mismatch"] = np.zeros(n_traj, dtype=np.float64)

    pairwise_pos_gaps: list[np.ndarray] = []
    pairwise_vel_gaps: list[np.ndarray] = []
    priors = {
        "ekf": arrays.ekf_prior,
        "ukf": arrays.ukf_prior,
        "aukf": arrays.aukf_prior,
    }
    prior_pairs = [("ekf", "ukf"), ("ekf", "aukf"), ("ukf", "aukf")]
    for left_name, right_name in prior_pairs:
        left = priors[left_name]
        right = priors[right_name]
        if left is None or right is None:
            continue
        pos_gap = np.linalg.norm(left[:, eval_start:, :3] - right[:, eval_start:, :3], axis=-1)
        vel_gap = np.linalg.norm(left[:, eval_start:, 3:] - right[:, eval_start:, 3:], axis=-1)
        pairwise_pos_gaps.append(pos_gap)
        pairwise_vel_gaps.append(vel_gap)
    if pairwise_pos_gaps:
        pos_stack = np.stack(pairwise_pos_gaps, axis=-1)
        vel_stack = np.stack(pairwise_vel_gaps, axis=-1)
        features["mean_prior_gap_pos_m"] = np.mean(pos_stack, axis=(1, 2))
        features["max_prior_gap_pos_m"] = np.max(pos_stack, axis=(1, 2))
        features["mean_prior_gap_vel_mps"] = np.mean(vel_stack, axis=(1, 2))
    else:
        features["mean_prior_gap_pos_m"] = np.zeros(n_traj, dtype=np.float64)
        features["max_prior_gap_pos_m"] = np.zeros(n_traj, dtype=np.float64)
        features["mean_prior_gap_vel_mps"] = np.zeros(n_traj, dtype=np.float64)

    return pd.DataFrame(features)


def build_stability_labels(
    traj_df: pd.DataFrame,
    *,
    pos_threshold_m: float = 1.0e8,
    vel_threshold_mps: float = 1.0e5,
) -> pd.DataFrame:
    frame = traj_df.copy()
    frame["is_unstable"] = trajectory_diverged_mask(
        frame["traj_pos_rmse_m"].to_numpy(dtype=np.float64),
        frame["traj_vel_rmse_mps"].to_numpy(dtype=np.float64),
        pos_threshold_m=pos_threshold_m,
        vel_threshold_mps=vel_threshold_mps,
    ).astype(int)
    frame["is_stable"] = 1 - frame["is_unstable"]
    return frame


def prepare_method_feature_frame(
    feature_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    *,
    methods: tuple[str, ...] = DEFAULT_SELECTOR_METHODS,
) -> pd.DataFrame:
    labels = stability_df[stability_df["method"].isin(methods)].copy()
    merged = labels.merge(feature_df, on=["scenario", "traj_id"], how="left", validate="many_to_one")
    return merged


def fit_logistic_risk_model(
    df: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    label_col: str = "is_unstable",
    l2_weight: float = 1.0,
) -> LogisticRiskModel:
    X = df.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy(dtype=np.float64)
    const_prob = float(np.clip(np.mean(y), 1.0e-4, 1.0 - 1.0e-4))
    if np.unique(y).size < 2:
        return LogisticRiskModel(
            feature_names=feature_names,
            feature_mean=np.zeros(X.shape[1], dtype=np.float64),
            feature_scale=np.ones(X.shape[1], dtype=np.float64),
            weights=np.zeros(X.shape[1], dtype=np.float64),
            bias=_safe_logit(const_prob),
            constant_probability=const_prob,
        )

    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    Xn = (X - mean) / scale

    def objective(params: np.ndarray) -> float:
        weights = params[:-1]
        bias = params[-1]
        logits = Xn @ weights + bias
        probs = np.clip(expit(logits), 1.0e-6, 1.0 - 1.0e-6)
        loss = -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))
        loss += 0.5 * l2_weight * float(np.mean(weights**2))
        return float(loss)

    init = np.zeros(X.shape[1] + 1, dtype=np.float64)
    init[-1] = _safe_logit(const_prob)
    result = minimize(objective, init, method="L-BFGS-B")
    params = result.x if result.success else init
    return LogisticRiskModel(
        feature_names=feature_names,
        feature_mean=mean.astype(np.float64),
        feature_scale=scale.astype(np.float64),
        weights=params[:-1].astype(np.float64),
        bias=float(params[-1]),
        constant_probability=None,
    )


def predict_logistic_risk(model: LogisticRiskModel, df: pd.DataFrame) -> np.ndarray:
    if model.constant_probability is not None:
        return np.full(df.shape[0], float(model.constant_probability), dtype=np.float64)
    X = df.loc[:, list(model.feature_names)].to_numpy(dtype=np.float64)
    Xn = (X - model.feature_mean) / model.feature_scale
    return expit(Xn @ model.weights + model.bias)


def empirical_visibility_risk(
    df: pd.DataFrame,
    *,
    empirical_unstable_rate: float,
) -> np.ndarray:
    base = _safe_logit(empirical_unstable_rate)
    logits = (
        base
        + 4.0 * df["fraction_zero_visibility"].to_numpy(dtype=np.float64)
        - 1.5 * df["mean_visible_stations"].to_numpy(dtype=np.float64)
        + 0.30 * df["mean_innovation_energy"].to_numpy(dtype=np.float64)
        + 1.00 * df["mean_visibility_mismatch"].to_numpy(dtype=np.float64)
        + 2.0e-5 * df["max_prior_gap_pos_m"].to_numpy(dtype=np.float64)
        + 0.90 * df["is_stress"].to_numpy(dtype=np.float64)
    )
    return expit(logits)


def binary_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    score = np.clip(np.asarray(y_score, dtype=np.float64).reshape(-1), 1.0e-6, 1.0 - 1.0e-6)
    metrics = {
        "n_samples": float(y.size),
        "positive_rate": float(np.mean(y)) if y.size else float("nan"),
        "predicted_positive_rate": float(np.mean(score >= 0.5)) if y.size else float("nan"),
        "brier": float(np.mean((score - y) ** 2)) if y.size else float("nan"),
    }
    if y.size == 0 or np.unique(y).size < 2:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
        return metrics
    pos_mask = y > 0.5
    n_pos = int(np.sum(pos_mask))
    n_neg = int(y.size - n_pos)
    ranks = rankdata(score)
    pos_rank_sum = float(np.sum(ranks[pos_mask]))
    metrics["auroc"] = float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))

    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(n_pos), 1.0)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    metrics["auprc"] = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return metrics


def fit_stability_models(
    train_df: pd.DataFrame,
    *,
    methods: tuple[str, ...] = DEFAULT_SELECTOR_METHODS,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for method in methods:
        subset = train_df[train_df["method"] == method].copy()
        if subset.empty:
            continue
        empirical_rate = float(np.clip(subset["is_unstable"].mean(), 1.0e-4, 1.0 - 1.0e-4))
        models[method] = {
            "logistic": fit_logistic_risk_model(subset, feature_names=feature_names),
            "empirical_rate": empirical_rate,
        }
    return models


def score_stability_predictors(
    eval_df: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    *,
    methods: tuple[str, ...] = DEFAULT_SELECTOR_METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        subset = eval_df[eval_df["method"] == method].copy()
        if subset.empty or method not in models:
            continue
        rule_score = empirical_visibility_risk(subset, empirical_unstable_rate=float(models[method]["empirical_rate"]))
        logistic_score = predict_logistic_risk(models[method]["logistic"], subset)
        for baseline_name, score in (("Empirical visibility risk", rule_score), ("Method-specific logistic risk", logistic_score)):
            baseline_detail = subset.loc[:, ["scenario", "traj_id", "method", "is_unstable"]].assign(
                baseline=baseline_name,
                unstable_probability=score,
            )
            detail_rows.append(
                baseline_detail
            )
            combined_metrics = binary_classification_metrics(subset["is_unstable"].to_numpy(dtype=np.float64), score)
            summary_rows.append(
                {
                    "baseline": baseline_name,
                    "method": method,
                    "scope": "combined",
                    **combined_metrics,
                }
            )
            for scenario_name, scenario_df in baseline_detail.groupby("scenario", sort=False):
                scenario_metrics = binary_classification_metrics(
                    scenario_df["is_unstable"].to_numpy(dtype=np.float64),
                    scenario_df["unstable_probability"].to_numpy(dtype=np.float64),
                )
                summary_rows.append(
                    {
                        "baseline": baseline_name,
                        "method": method,
                        "scope": scenario_name,
                        **scenario_metrics,
                    }
                )
    detail_df = pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if not detail_df.empty:
        overall_rows: list[dict[str, Any]] = []
        for baseline_name, baseline_df in detail_df.groupby("baseline", sort=False):
            overall_rows.append(
                {
                    "baseline": baseline_name,
                    "method": "ALL",
                    "scope": "combined",
                    **binary_classification_metrics(
                        baseline_df["is_unstable"].to_numpy(dtype=np.float64),
                        baseline_df["unstable_probability"].to_numpy(dtype=np.float64),
                    ),
                }
            )
            for scenario_name, scenario_df in baseline_df.groupby("scenario", sort=False):
                overall_rows.append(
                    {
                        "baseline": baseline_name,
                        "method": "ALL",
                        "scope": scenario_name,
                        **binary_classification_metrics(
                            scenario_df["is_unstable"].to_numpy(dtype=np.float64),
                            scenario_df["unstable_probability"].to_numpy(dtype=np.float64),
                        ),
                    }
                )
        summary_df = pd.concat([summary_df, pd.DataFrame(overall_rows)], ignore_index=True)
    return detail_df, summary_df


def build_oracle_selector_targets(
    stability_df: pd.DataFrame,
    *,
    methods: tuple[str, ...] = DEFAULT_SELECTOR_METHODS,
) -> pd.DataFrame:
    subset = stability_df[stability_df["method"].isin(methods)].copy()
    subset["effective_rmse_m"] = subset["traj_pos_rmse_m"].where(subset["is_stable"] > 0, np.inf)
    stable_subset = subset[np.isfinite(subset["effective_rmse_m"])].copy()
    if stable_subset.empty:
        raise ValueError("No stable trajectory/method pairs available for oracle construction.")
    idx = stable_subset.groupby(["scenario", "traj_id"], sort=False)["effective_rmse_m"].idxmin()
    oracle = stable_subset.loc[idx, ["scenario", "traj_id", "method", "traj_pos_rmse_m"]].copy()
    oracle = oracle.rename(columns={"method": "oracle_method", "traj_pos_rmse_m": "oracle_pos_rmse_m"})
    return oracle.sort_values(["scenario", "traj_id"]).reset_index(drop=True)


def summarize_training_costs(
    train_df: pd.DataFrame,
    *,
    methods: tuple[str, ...] = DEFAULT_SELECTOR_METHODS,
) -> dict[str, float]:
    costs: dict[str, float] = {}
    for method in methods:
        subset = train_df[(train_df["method"] == method) & (train_df["is_stable"] > 0)].copy()
        if subset.empty:
            continue
        costs[method] = float(np.mean(subset["traj_pos_rmse_m"]))
    return costs


def build_selection_detail_frame(
    eval_features: pd.DataFrame,
    eval_stability: pd.DataFrame,
    oracle_df: pd.DataFrame,
    *,
    selector_name: str,
    selected_methods: np.ndarray,
) -> pd.DataFrame:
    selected = pd.DataFrame(
        {
            "scenario": eval_features["scenario"].to_numpy(),
            "traj_id": eval_features["traj_id"].to_numpy(dtype=np.int64),
            "selected_method": selected_methods,
        }
    )
    method_results = eval_stability.loc[:, ["scenario", "traj_id", "method", "traj_pos_rmse_m", "is_stable"]].rename(
        columns={
            "method": "selected_method",
            "traj_pos_rmse_m": "selected_pos_rmse_m",
            "is_stable": "selected_is_stable",
        }
    )
    detail = selected.merge(method_results, on=["scenario", "traj_id", "selected_method"], how="left", validate="one_to_one")
    detail = detail.merge(oracle_df, on=["scenario", "traj_id"], how="left", validate="one_to_one")
    detail["selector"] = selector_name
    detail["oracle_match"] = (detail["selected_method"] == detail["oracle_method"]).astype(int)
    detail["regret_m"] = detail["selected_pos_rmse_m"] - detail["oracle_pos_rmse_m"]
    return detail


def summarize_selector_details(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for selector_name, selector_df in detail_df.groupby("selector", sort=False):
        for scope_name, scope_df in [("combined", selector_df)] + list(selector_df.groupby("scenario", sort=False)):
            rows.append(
                {
                    "selector": selector_name,
                    "scope": scope_name,
                    "n_trajectories": int(scope_df.shape[0]),
                    "mean_selected_rmse_m": float(scope_df["selected_pos_rmse_m"].mean()),
                    "divergence_avoidance_rate": float(scope_df["selected_is_stable"].mean()),
                    "oracle_match_rate": float(scope_df["oracle_match"].mean()),
                    "mean_regret_m": float(scope_df["regret_m"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_selector_outputs(
    train_features: pd.DataFrame,
    train_stability: pd.DataFrame,
    eval_features: pd.DataFrame,
    eval_stability: pd.DataFrame,
    *,
    selector_methods: tuple[str, ...] = DEFAULT_SELECTOR_METHODS,
    divergence_penalty_m: float = 1.0e8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = fit_stability_models(train_stability.merge(train_features, on=["scenario", "traj_id"], how="left"), methods=selector_methods)
    risk_costs = summarize_training_costs(train_stability, methods=selector_methods)
    oracle = build_oracle_selector_targets(eval_stability, methods=selector_methods)

    baseline_details: list[pd.DataFrame] = []
    feature_index = eval_features.loc[:, ["scenario", "traj_id"]].copy()

    for baseline_name, method_name in (
        ("Always EKF", "EKF"),
        ("Always AUKF", "AUKF"),
        ("Always KalmanNet-like", "KalmanNetLike"),
    ):
        selected_methods = np.full(feature_index.shape[0], method_name, dtype=object)
        baseline_details.append(
            build_selection_detail_frame(
                eval_features,
                eval_stability,
                oracle,
                selector_name=baseline_name,
                selected_methods=selected_methods,
            )
        )

    if {"EKF", "KalmanNetLike", "InnovationHybridGNN"}.issubset(set(selector_methods)):
        regime_selected = np.where(
            eval_features["is_stress"].to_numpy(dtype=np.float64) > 0.5,
            "InnovationHybridGNN",
            "KalmanNetLike",
        ).astype(object)
        regime_selected = np.where(
            eval_features["is_public_observation"].to_numpy(dtype=np.float64) > 0.5,
            regime_selected,
            "EKF",
        ).astype(object)
        baseline_details.append(
            build_selection_detail_frame(
                eval_features,
                eval_stability,
                oracle,
                selector_name="Public-regime conservative selector",
                selected_methods=regime_selected,
            )
        )

    if models:
        expected_costs_rule: list[np.ndarray] = []
        expected_costs_logistic: list[np.ndarray] = []
        method_order: list[str] = []
        merged_eval = eval_features.copy()
        for method in selector_methods:
            if method not in models:
                continue
            empirical_rate = float(models[method]["empirical_rate"])
            rule_risk = empirical_visibility_risk(merged_eval, empirical_unstable_rate=empirical_rate)
            logistic_risk = predict_logistic_risk(models[method]["logistic"], merged_eval)
            stable_cost = float(risk_costs.get(method, divergence_penalty_m))
            expected_costs_rule.append(rule_risk * divergence_penalty_m + (1.0 - rule_risk) * stable_cost)
            expected_costs_logistic.append(logistic_risk * divergence_penalty_m + (1.0 - logistic_risk) * stable_cost)
            method_order.append(method)
        if method_order:
            rule_stack = np.stack(expected_costs_rule, axis=1)
            logistic_stack = np.stack(expected_costs_logistic, axis=1)
            rule_selected = np.array(method_order, dtype=object)[np.argmin(rule_stack, axis=1)]
            logistic_selected = np.array(method_order, dtype=object)[np.argmin(logistic_stack, axis=1)]
            baseline_details.append(
                build_selection_detail_frame(
                    eval_features,
                    eval_stability,
                    oracle,
                    selector_name="Stability-weighted empirical selector",
                    selected_methods=rule_selected,
                )
            )
            baseline_details.append(
                build_selection_detail_frame(
                    eval_features,
                    eval_stability,
                    oracle,
                    selector_name="Stability-weighted logistic selector",
                    selected_methods=logistic_selected,
                )
            )

    oracle_details = oracle.assign(
        selector="Oracle stable selector",
        selected_method=oracle["oracle_method"],
        selected_pos_rmse_m=oracle["oracle_pos_rmse_m"],
        selected_is_stable=1,
        oracle_match=1,
        regret_m=0.0,
    )
    baseline_details.append(
        oracle_details.loc[
            :,
            [
                "scenario",
                "traj_id",
                "selector",
                "selected_method",
                "selected_pos_rmse_m",
                "selected_is_stable",
                "oracle_method",
                "oracle_pos_rmse_m",
                "oracle_match",
                "regret_m",
            ],
        ]
    )

    detail_df = pd.concat(baseline_details, ignore_index=True)
    summary_df = summarize_selector_details(detail_df)
    return detail_df, summary_df


def build_packet_registry(
    cfg: dict[str, Any],
    dataset_manifest: dict[str, Any],
    public_manifest: dict[str, Any],
) -> dict[str, Any]:
    benchmark_cfg = cfg.get("benchmark_tasks", {})
    packets: list[dict[str, Any]] = []
    for scenario_name in PUBLIC_PACKET_SCENARIOS:
        if scenario_name not in dataset_manifest:
            continue
        summary = dataset_manifest[scenario_name]
        if scenario_name == "semi_real_replay_test":
            tier = "archived_catalog_replay"
            tasks = ["state_estimation"]
        elif scenario_name == "public_catalog_replay_test":
            tier = "public_catalog_replay"
            tasks = ["state_estimation"]
        elif scenario_name == "satnogs_observation_replay_val":
            tier = "public_observation_validation"
            tasks = ["stability_prediction_train", "method_selection_train"]
        elif scenario_name == "satnogs_observation_replay_test":
            tier = "public_observation_test"
            tasks = ["state_estimation", "stability_prediction_eval", "method_selection_eval"]
        else:
            tier = "public_observation_stress"
            tasks = ["state_estimation", "stability_prediction_eval", "method_selection_eval"]
        packets.append(
            {
                "scenario": scenario_name,
                "tier": tier,
                "tasks": tasks,
                "kind": summary.get("kind"),
                "samples": int(summary.get("samples", 0)),
                "distinct_source_satellites": int(summary.get("distinct_source_satellites", 0)),
                "distinct_station_bank_members": int(summary.get("distinct_station_bank_members", 0)),
                "coverage": summary.get("coverage", {}),
                "source_pool_count": int(summary.get("observation_station_bank", {}).get("source_pool_count", 0)),
            }
        )
    return {
        "benchmark_name": benchmark_cfg.get("name", "SPOT-OD"),
        "benchmark_subtitle": benchmark_cfg.get("subtitle", "Sparse public observation and tracking orbit-determination benchmark"),
        "public_manifest": {
            "catalog_count": int(public_manifest.get("catalog", {}).get("count", 0)),
            "observation_count": int(public_manifest.get("observations", {}).get("count", 0)),
        },
        "packets": packets,
        "state_estimation_scenarios": benchmark_cfg.get("state_estimation_scenarios", []),
        "task_train_scenarios": benchmark_cfg.get("task_train_scenarios", []),
        "task_eval_scenarios": benchmark_cfg.get("task_eval_scenarios", []),
        "selector_methods": benchmark_cfg.get("selector_methods", list(DEFAULT_SELECTOR_METHODS)),
    }
