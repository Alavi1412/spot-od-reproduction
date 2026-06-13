"""Learned residual-correction model for the bounded real ILRS SLR pilot.

This module provides a *leak-free* feature builder and a small, deterministic,
bounded neural residual regressor used to score a learned estimator on real
satellite-laser-ranging normal points.  It is deliberately compact and
self-contained:

* Features are restricted to quantities available without ground truth (time
  since the first epoch, ranging-station identity, an orbital-phase angle from
  the public element set, and the model-predicted range), so a learned
  correction trained on the earlier fit arc and evaluated on the later held-out
  arc is a genuine temporal-extrapolation test, not an information leak.
* The regressor is a tiny bounded multilayer perceptron.  Training is fully
  deterministic (fixed seed, CPU-or-GPU agnostic) and the predicted correction
  is hard-bounded to a multiple of the fit-arc residual scale so a learned
  extrapolation cannot blow up the held-out comparison.
* If the optional neural backend is unavailable the model degrades to a
  deterministic closed-form ridge regressor so the pilot stays reproducible
  offline; the active backend is reported in the result schema.

Nothing here forces accelerated execution: the neural backend uses an
accelerator only when one is already present and otherwise runs on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Stations carried by the supported pilot subset, in a fixed order so the
# one-hot station encoding is stable across runs and splits.
_STATION_ORDER: tuple[str, ...] = ("YARL", "MATM", "WETL", "HERL")


def orbital_period_seconds(tle_line2: str, default: float = 13360.0) -> float:
    """Orbital period (s) from the mean motion in TLE line 2.

    The mean motion (revolutions per day) occupies columns 53--63 of a
    standard two-line element set.  ``default`` (the LAGEOS-2 period) is used
    only if the field cannot be parsed.
    """
    try:
        mean_motion_rev_per_day = float(tle_line2[52:63])
        if mean_motion_rev_per_day <= 0.0:
            return float(default)
        return 86400.0 / mean_motion_rev_per_day
    except (ValueError, IndexError):
        return float(default)


def build_feature_matrix(
    station_codes: list[str],
    epochs_unix: np.ndarray,
    predicted_ranges_m: np.ndarray,
    t0_unix: float,
    time_span_s: float,
    period_s: float,
) -> np.ndarray:
    """Assemble the leak-free feature matrix (one row per normal point).

    Columns: normalized time-since-epoch, four station one-hot indicators, the
    sine and cosine of the orbital-phase angle, and the model-predicted range.
    Every column is computable without ground truth or held-out information.
    """
    epochs = np.asarray(epochs_unix, dtype=np.float64)
    pred = np.asarray(predicted_ranges_m, dtype=np.float64)
    span = float(time_span_s) if time_span_s > 0.0 else 1.0
    rel = epochs - float(t0_unix)
    t_norm = rel / span
    phase = 2.0 * np.pi * rel / float(period_s if period_s > 0.0 else 1.0)
    onehot = np.zeros((epochs.size, len(_STATION_ORDER)), dtype=np.float64)
    for i, code in enumerate(station_codes):
        if code in _STATION_ORDER:
            onehot[i, _STATION_ORDER.index(code)] = 1.0
    return np.column_stack(
        [t_norm, onehot, np.sin(phase), np.cos(phase), pred]
    ).astype(np.float64)


@dataclass
class _Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "_Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


def _fit_torch(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_all: np.ndarray,
    *,
    seed: int,
    bound: float,
) -> tuple[np.ndarray, str] | None:
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - optional backend
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_scale = float(np.sqrt(np.mean(y_tr**2))) or 1.0
    xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(
        (y_tr / y_scale).reshape(-1, 1), dtype=torch.float32, device=device
    )
    xa = torch.tensor(x_all, dtype=torch.float32, device=device)

    hidden = 32
    model = torch.nn.Sequential(
        torch.nn.Linear(x_tr.shape[1], hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()
    model.train()
    for _ in range(600):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        raw = model(xa).cpu().numpy().reshape(-1) * y_scale
    corr = bound * np.tanh(raw / bound)
    return corr.astype(np.float64), f"torch:{device.type}"


def _fit_ridge(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_all: np.ndarray,
    *,
    bound: float,
    alpha: float = 1.0,
) -> tuple[np.ndarray, str]:
    """Deterministic closed-form ridge fallback (offline-reproducible)."""
    xb_tr = np.column_stack([x_tr, np.ones(x_tr.shape[0])])
    xb_all = np.column_stack([x_all, np.ones(x_all.shape[0])])
    n_feat = xb_tr.shape[1]
    reg = alpha * np.eye(n_feat)
    reg[-1, -1] = 0.0  # do not penalize the bias term
    beta = np.linalg.solve(xb_tr.T @ xb_tr + reg, xb_tr.T @ y_tr)
    raw = xb_all @ beta
    corr = bound * np.tanh(raw / bound)
    return corr.astype(np.float64), "ridge"


def learned_residual_correction(
    features: np.ndarray,
    residuals: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int = 0,
    clip_k: float = 4.0,
) -> tuple[np.ndarray, str]:
    """Fit a bounded residual model on the fit arc and predict for all points.

    ``residuals`` are the signed range residuals (predicted minus observed) of
    the underlying orbit (SGP4 prior or fitted WLS solution).  The model is
    trained only on ``train_idx`` and applied to every point; the returned
    correction is hard-bounded to ``clip_k`` times the fit-arc residual RMS.

    Returns the per-point correction and a backend tag.
    """
    feats = np.asarray(features, dtype=np.float64)
    resid = np.asarray(residuals, dtype=np.float64)
    std = _Standardizer.fit(feats[train_idx])
    x_tr = std.transform(feats[train_idx])
    x_all = std.transform(feats)
    y_tr = resid[train_idx]
    train_rms = float(np.sqrt(np.mean(y_tr**2))) or 1.0
    bound = clip_k * train_rms

    result = _fit_torch(x_tr, y_tr, x_all, seed=seed, bound=bound)
    if result is None:
        result = _fit_ridge(x_tr, y_tr, x_all, bound=bound)
    return result
