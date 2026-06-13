"""Innovation-feature construction for innovation-conditioned hybrids."""

from __future__ import annotations

import numpy as np

from .coordinates import StationGeometry, line_of_sight_measurement


INNOVATION_FEATURE_DIM = 6


def wrap_angle_pi(x: float) -> float:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def compute_innovation_features(
    prior_states: np.ndarray,
    measurements: np.ndarray,
    visibility: np.ndarray,
    times_s: np.ndarray,
    stations: tuple[StationGeometry, ...],
    meas_std_vector: np.ndarray,
    clip: float = 12.0,
) -> np.ndarray:
    """Return normalized innovation features per trajectory/time/station.

    Channels:
    0-3: normalized innovation for [range, azimuth, elevation, range-rate]
    4: RMS normalized innovation magnitude
    5: predicted visibility flag from the prior state
    """
    prior = np.asarray(prior_states, dtype=np.float64)
    meas = np.asarray(measurements, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    std = np.asarray(meas_std_vector, dtype=np.float64).reshape(4)
    inv_std = 1.0 / np.clip(std, 1e-9, None)

    n_traj, t_steps, s_count = meas.shape[:3]
    out = np.zeros((n_traj, t_steps, s_count, INNOVATION_FEATURE_DIM), dtype=np.float32)

    for i in range(n_traj):
        for t in range(t_steps):
            for s_idx, station in enumerate(stations):
                z_pred, pred_visible = line_of_sight_measurement(prior[i, t], station, float(times[i, t]))
                out[i, t, s_idx, 5] = 1.0 if pred_visible else 0.0
                if vis[i, t, s_idx] < 0.5:
                    continue
                resid = meas[i, t, s_idx] - z_pred
                resid[1] = wrap_angle_pi(float(resid[1]))
                resid[2] = wrap_angle_pi(float(resid[2]))
                norm_resid = np.clip(resid * inv_std, -clip, clip)
                out[i, t, s_idx, :4] = norm_resid.astype(np.float32)
                out[i, t, s_idx, 4] = float(np.sqrt(np.mean(norm_resid**2)))
    return out
