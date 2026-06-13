"""Trained trajectory-level guards for learned state estimators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .benchmark_tasks import trajectory_diverged_mask


@dataclass(frozen=True)
class GuardedSelectorConfig:
    hidden_dim: int = 48
    epochs: int = 800
    learning_rate: float = 0.003
    weight_decay: float = 1.0e-4
    val_fraction: float = 0.25
    patience: int = 80
    seed: int = 42
    divergence_penalty_m: float = 1.0e8
    max_cost_m: float = 1.0e8


class CostPredictorNet(nn.Module):
    """Small MLP that predicts log trajectory cost for each candidate method."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedGuardedSelector:
    methods: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    model: CostPredictorNet
    config: GuardedSelectorConfig
    history: pd.DataFrame


def _feature_matrix(feature_df: pd.DataFrame, feature_names: tuple[str, ...]) -> np.ndarray:
    missing = [name for name in feature_names if name not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing selector feature columns: {missing}")
    x = feature_df.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    return (x - mean) / scale, mean.astype(np.float32), scale.astype(np.float32)


def _standardize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def build_cost_matrix(
    features: pd.DataFrame,
    trajectory_errors: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    divergence_penalty_m: float = 1.0e8,
    max_cost_m: float = 1.0e8,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Align feature rows with per-method trajectory costs."""

    keys = ["scenario", "traj_id"]
    base = features.loc[:, keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    cost_columns: list[np.ndarray] = []
    for method in methods:
        subset = trajectory_errors[trajectory_errors["method"] == method].copy()
        subset = subset.loc[:, keys + ["traj_pos_rmse_m", "traj_vel_rmse_mps"]]
        merged = base.merge(subset, on=keys, how="left", validate="one_to_one")
        pos = merged["traj_pos_rmse_m"].to_numpy(dtype=np.float64)
        vel = merged["traj_vel_rmse_mps"].to_numpy(dtype=np.float64)
        diverged = trajectory_diverged_mask(pos, vel)
        cost = np.where(diverged, divergence_penalty_m, pos)
        cost = np.nan_to_num(cost, nan=divergence_penalty_m, posinf=divergence_penalty_m, neginf=divergence_penalty_m)
        cost = np.clip(cost, 0.0, max_cost_m)
        cost_columns.append(cost.astype(np.float32))
    cost_matrix = np.stack(cost_columns, axis=1)
    oracle_idx = np.argmin(cost_matrix, axis=1)
    aligned = base.merge(features, on=keys, how="left", validate="one_to_one")
    aligned["oracle_method"] = np.asarray(methods, dtype=object)[oracle_idx]
    aligned["oracle_pos_rmse_m"] = cost_matrix[np.arange(cost_matrix.shape[0]), oracle_idx]
    return aligned, cost_matrix


def fit_guarded_selector(
    feature_df: pd.DataFrame,
    cost_matrix: np.ndarray,
    *,
    methods: tuple[str, ...],
    feature_names: tuple[str, ...],
    config: GuardedSelectorConfig,
    device: torch.device | str = "cpu",
) -> TrainedGuardedSelector:
    """Train a cost-prediction guard and return the fitted selector."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device)

    x_raw = _feature_matrix(feature_df, feature_names)
    x_scaled, feature_mean, feature_scale = _standardize_fit(x_raw)
    y_raw = np.log1p(np.clip(cost_matrix.astype(np.float32), 0.0, config.max_cost_m))
    y_scaled, target_mean, target_scale = _standardize_fit(y_raw)

    n_samples = x_scaled.shape[0]
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(n_samples)
    n_val = int(round(config.val_fraction * n_samples)) if n_samples > 4 else 0
    val_idx = order[:n_val]
    train_idx = order[n_val:] if n_val > 0 else order

    x_tensor = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(y_scaled.astype(np.float32)).to(device)
    train_idx_t = torch.from_numpy(train_idx.astype(np.int64)).to(device)
    val_idx_t = torch.from_numpy(val_idx.astype(np.int64)).to(device) if n_val > 0 else None

    model = CostPredictorNet(
        input_dim=len(feature_names),
        output_dim=len(methods),
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.SmoothL1Loss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    rows: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_tensor[train_idx_t])
        loss = loss_fn(pred, y_tensor[train_idx_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = float(loss_fn(model(x_tensor[train_idx_t]), y_tensor[train_idx_t]).item())
            if val_idx_t is not None and val_idx_t.numel() > 0:
                val_loss = float(loss_fn(model(x_tensor[val_idx_t]), y_tensor[val_idx_t]).item())
            else:
                val_loss = train_loss
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1.0e-7:
            best_val = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    model.eval()
    return TrainedGuardedSelector(
        methods=methods,
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        model=model,
        config=config,
        history=pd.DataFrame(rows),
    )


@torch.no_grad()
def predict_costs(selector: TrainedGuardedSelector, feature_df: pd.DataFrame) -> np.ndarray:
    x = _feature_matrix(feature_df, selector.feature_names)
    x = _standardize_apply(x, selector.feature_mean, selector.feature_scale).astype(np.float32)
    pred_scaled = selector.model(torch.from_numpy(x)).cpu().numpy()
    pred_log_cost = pred_scaled * selector.target_scale + selector.target_mean
    return np.expm1(pred_log_cost)


def select_methods(selector: TrainedGuardedSelector, feature_df: pd.DataFrame) -> pd.DataFrame:
    costs = predict_costs(selector, feature_df)
    idx = np.argmin(costs, axis=1)
    out = feature_df.loc[:, ["scenario", "traj_id"]].copy()
    out["selected_method"] = np.asarray(selector.methods, dtype=object)[idx]
    out["predicted_cost_m"] = costs[np.arange(costs.shape[0]), idx]
    for method_idx, method in enumerate(selector.methods):
        out[f"predicted_cost_{method}_m"] = costs[:, method_idx]
    return out


def evaluate_selection(selection: pd.DataFrame, trajectory_errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["scenario", "traj_id"]
    detail = selection.merge(
        trajectory_errors.loc[:, keys + ["method", "traj_pos_rmse_m", "traj_vel_rmse_mps"]],
        left_on=keys + ["selected_method"],
        right_on=keys + ["method"],
        how="left",
        validate="one_to_one",
    )
    detail = detail.drop(columns=["method"])
    detail["selected_diverged"] = trajectory_diverged_mask(
        detail["traj_pos_rmse_m"].to_numpy(dtype=np.float64),
        detail["traj_vel_rmse_mps"].to_numpy(dtype=np.float64),
    ).astype(int)

    rows: list[dict[str, Any]] = []
    for scope, group in [("combined", detail)] + list(detail.groupby("scenario", sort=False)):
        pos = group["traj_pos_rmse_m"].to_numpy(dtype=np.float64)
        vel = group["traj_vel_rmse_mps"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "scope": scope,
                "n_trajectories": int(group.shape[0]),
                "aggregate_pos_rmse_m": float(np.sqrt(np.mean(pos**2))),
                "mean_traj_pos_rmse_m": float(np.mean(pos)),
                "aggregate_vel_rmse_mps": float(np.sqrt(np.mean(vel**2))),
                "mean_traj_vel_rmse_mps": float(np.mean(vel)),
                "divergence_rate": float(np.mean(group["selected_diverged"])),
            }
        )
    return detail, pd.DataFrame(rows)


def save_guarded_selector(selector: TrainedGuardedSelector, path: str | Path) -> None:
    payload = {
        "methods": selector.methods,
        "feature_names": selector.feature_names,
        "feature_mean": selector.feature_mean,
        "feature_scale": selector.feature_scale,
        "target_mean": selector.target_mean,
        "target_scale": selector.target_scale,
        "model_state_dict": selector.model.state_dict(),
        "config": selector.config.__dict__,
        "history": selector.history.to_dict(orient="records"),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_guarded_selector(path: str | Path) -> TrainedGuardedSelector:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    methods = tuple(str(x) for x in payload["methods"])
    feature_names = tuple(str(x) for x in payload["feature_names"])
    config = GuardedSelectorConfig(**payload["config"])
    model = CostPredictorNet(
        input_dim=len(feature_names),
        output_dim=len(methods),
        hidden_dim=config.hidden_dim,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return TrainedGuardedSelector(
        methods=methods,
        feature_names=feature_names,
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
        feature_scale=np.asarray(payload["feature_scale"], dtype=np.float32),
        target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
        target_scale=np.asarray(payload["target_scale"], dtype=np.float32),
        model=model,
        config=config,
        history=pd.DataFrame(payload.get("history", [])),
    )
