"""Evaluation metrics for state estimation."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=axis))


def position_velocity_rmse(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    pos_rmse = float(rmse(y_true[..., :3], y_pred[..., :3]))
    vel_rmse = float(rmse(y_true[..., 3:], y_pred[..., 3:]))
    return {"pos_rmse_m": pos_rmse, "vel_rmse_mps": vel_rmse}


def median_absolute_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    return {
        "med_abs_pos_m": float(np.median(abs_err[..., :3])),
        "med_abs_vel_mps": float(np.median(abs_err[..., 3:])),
    }
