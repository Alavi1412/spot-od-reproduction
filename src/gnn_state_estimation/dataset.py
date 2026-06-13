"""PyTorch datasets and normalization for state-estimation learning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


STATE_SCALE = np.array([1.0e7, 1.0e7, 1.0e7, 1.0e4, 1.0e4, 1.0e4], dtype=np.float32)
MEAS_SCALE = np.array([1.0e7, np.pi, 0.5 * np.pi, 1.0e4], dtype=np.float32)


@dataclass(frozen=True)
class DatasetArrays:
    states: np.ndarray
    measurements: np.ndarray
    visibility: np.ndarray
    station_ecef: np.ndarray
    times: np.ndarray
    ekf_prior: np.ndarray | None = None
    ukf_prior: np.ndarray | None = None
    aukf_prior: np.ndarray | None = None
    secondary_prior: np.ndarray | None = None
    x0_estimates: np.ndarray | None = None
    innovation_features: np.ndarray | None = None
    prior_bank_stats: np.ndarray | None = None
    sample_weight: np.ndarray | None = None
    regime_name: np.ndarray | None = None


def load_dataset_npz(path: str | Path) -> DatasetArrays:
    data = np.load(path)
    ekf_prior = data["ekf_prior"] if "ekf_prior" in data.files else None
    ukf_prior = data["ukf_prior"] if "ukf_prior" in data.files else None
    aukf_prior = data["aukf_prior"] if "aukf_prior" in data.files else None
    secondary_prior = None
    if "secondary_prior" in data.files:
        secondary_prior = data["secondary_prior"]
    elif "aukf_prior" in data.files:
        secondary_prior = data["aukf_prior"]
    x0_estimates = data["x0_estimates"] if "x0_estimates" in data.files else None
    innovation_features = data["innovation_features"] if "innovation_features" in data.files else None
    prior_bank_stats = data["prior_bank_stats"] if "prior_bank_stats" in data.files else None
    sample_weight = data["sample_weight"] if "sample_weight" in data.files else None
    regime_name = data["regime_name"] if "regime_name" in data.files else None
    return DatasetArrays(
        states=data["states"],
        measurements=data["measurements"],
        visibility=data["visibility"],
        station_ecef=data["station_ecef"][0],
        times=data["times"],
        ekf_prior=ekf_prior,
        ukf_prior=ukf_prior,
        aukf_prior=aukf_prior,
        secondary_prior=secondary_prior,
        x0_estimates=x0_estimates,
        innovation_features=innovation_features,
        prior_bank_stats=prior_bank_stats,
        sample_weight=sample_weight,
        regime_name=regime_name,
    )


def with_innovation_features(arrays: DatasetArrays, innovation_features: np.ndarray) -> DatasetArrays:
    return replace(arrays, innovation_features=innovation_features)


def with_secondary_prior(arrays: DatasetArrays, secondary_prior: np.ndarray) -> DatasetArrays:
    return replace(arrays, secondary_prior=secondary_prior)


def with_prior_bank_stats(arrays: DatasetArrays, prior_bank_stats: np.ndarray) -> DatasetArrays:
    return replace(arrays, prior_bank_stats=prior_bank_stats)


def with_sample_weight(arrays: DatasetArrays, sample_weight: np.ndarray) -> DatasetArrays:
    return replace(arrays, sample_weight=sample_weight)


def with_full_priors(
    arrays: DatasetArrays,
    *,
    ekf_prior: np.ndarray | None = None,
    ukf_prior: np.ndarray | None = None,
    aukf_prior: np.ndarray | None = None,
) -> DatasetArrays:
    return replace(
        arrays,
        ekf_prior=arrays.ekf_prior if ekf_prior is None else ekf_prior,
        ukf_prior=arrays.ukf_prior if ukf_prior is None else ukf_prior,
        aukf_prior=arrays.aukf_prior if aukf_prior is None else aukf_prior,
        secondary_prior=arrays.secondary_prior if arrays.secondary_prior is not None else aukf_prior,
    )


def concatenate_dataset_arrays(items: list[DatasetArrays]) -> DatasetArrays:
    if not items:
        raise ValueError("Cannot concatenate an empty dataset list.")
    base_station = items[0].station_ecef
    if any(arr.station_ecef.shape != base_station.shape or not np.allclose(arr.station_ecef, base_station) for arr in items[1:]):
        raise ValueError("All datasets must share the same station geometry.")

    def _cat(name: str) -> np.ndarray | None:
        values = [getattr(arr, name) for arr in items]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            raise ValueError(f"Dataset field {name} is missing for part of the concatenation.")
        return np.concatenate(values, axis=0)

    return DatasetArrays(
        states=np.concatenate([arr.states for arr in items], axis=0),
        measurements=np.concatenate([arr.measurements for arr in items], axis=0),
        visibility=np.concatenate([arr.visibility for arr in items], axis=0),
        station_ecef=base_station,
        times=np.concatenate([arr.times for arr in items], axis=0),
        ekf_prior=_cat("ekf_prior"),
        ukf_prior=_cat("ukf_prior"),
        aukf_prior=_cat("aukf_prior"),
        secondary_prior=_cat("secondary_prior"),
        x0_estimates=_cat("x0_estimates"),
        innovation_features=_cat("innovation_features"),
        prior_bank_stats=_cat("prior_bank_stats"),
        sample_weight=_cat("sample_weight"),
        regime_name=_cat("regime_name"),
    )


def compute_prior_bank_stats(*priors: np.ndarray | None) -> np.ndarray:
    bank = [p for p in priors if p is not None]
    if not bank:
        raise ValueError("At least one prior array is required.")
    prior_bank = np.stack(bank, axis=-2)
    spread = np.std(prior_bank, axis=-2)
    mean_abs_dev = np.mean(np.abs(prior_bank - np.mean(prior_bank, axis=-2, keepdims=True)), axis=-2)
    max_abs_pairwise = np.max(np.abs(prior_bank - np.median(prior_bank, axis=-2, keepdims=True)), axis=-2)
    scale = np.concatenate([STATE_SCALE, STATE_SCALE, STATE_SCALE], axis=0)
    stats = np.concatenate([spread, mean_abs_dev, max_abs_pairwise], axis=-1)
    return (stats / scale).astype(np.float32)


def scale_state(x: np.ndarray) -> np.ndarray:
    return x / STATE_SCALE


def unscale_state(x: np.ndarray) -> np.ndarray:
    return x * STATE_SCALE


def scale_measurements(z: np.ndarray) -> np.ndarray:
    return z / MEAS_SCALE


class WindowedSatelliteDataset(Dataset):
    """Sliding-window dataset across trajectories and time."""

    def __init__(
        self,
        arrays: DatasetArrays,
        window_size: int,
        require_prior: bool = False,
    ) -> None:
        self.arr = arrays
        self.window_size = int(window_size)
        self.require_prior = require_prior
        if require_prior and arrays.ekf_prior is None:
            raise ValueError("EKF prior is required but not found in dataset.")

        n_traj, t_steps = arrays.states.shape[0], arrays.states.shape[1]
        self.indices: list[tuple[int, int]] = []
        for i in range(n_traj):
            for t in range(self.window_size - 1, t_steps):
                self.indices.append((i, t))

        station_xyz = arrays.station_ecef.astype(np.float32) / 6_378_137.0
        self.station_xyz = torch.from_numpy(station_xyz)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i, t = self.indices[idx]
        t0 = t - self.window_size + 1

        meas = self.arr.measurements[i, t0 : t + 1].astype(np.float32)
        vis = self.arr.visibility[i, t0 : t + 1].astype(np.float32)[..., None]
        meas = scale_measurements(meas)

        target = scale_state(self.arr.states[i, t].astype(np.float32))
        sample = {
            "measurements": torch.from_numpy(meas),
            "visibility": torch.from_numpy(vis),
            "station_xyz": self.station_xyz,
            "target": torch.from_numpy(target),
        }
        if self.arr.ekf_prior is not None:
            prior = scale_state(self.arr.ekf_prior[i, t].astype(np.float32))
            sample["ekf_prior"] = torch.from_numpy(prior)
        if self.arr.secondary_prior is not None:
            secondary_prior = scale_state(self.arr.secondary_prior[i, t].astype(np.float32))
            sample["secondary_prior"] = torch.from_numpy(secondary_prior)
        if self.arr.innovation_features is not None:
            innov = self.arr.innovation_features[i, t0 : t + 1].astype(np.float32)
            sample["innovation_features"] = torch.from_numpy(innov)
        if self.arr.ekf_prior is not None and self.arr.ukf_prior is not None and self.arr.aukf_prior is not None:
            prior_bank = np.stack(
                [
                    scale_state(self.arr.ekf_prior[i, t].astype(np.float32)),
                    scale_state(self.arr.ukf_prior[i, t].astype(np.float32)),
                    scale_state(self.arr.aukf_prior[i, t].astype(np.float32)),
                ],
                axis=0,
            )
            sample["prior_bank"] = torch.from_numpy(prior_bank)
        if self.arr.prior_bank_stats is not None:
            sample["prior_bank_stats"] = torch.from_numpy(self.arr.prior_bank_stats[i, t].astype(np.float32))
        if self.arr.sample_weight is not None:
            sample["sample_weight"] = torch.tensor(float(self.arr.sample_weight[i]), dtype=torch.float32)
        return sample
