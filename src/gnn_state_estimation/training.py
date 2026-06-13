"""Training utilities for graph and hybrid estimators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    DatasetArrays,
    WindowedSatelliteDataset,
    compute_prior_bank_stats,
    with_full_priors,
    with_innovation_features,
    with_prior_bank_stats,
)
from .evaluation import BaselineConfig, run_filter_baselines
from .innovation import compute_innovation_features
from .models.graph_estimator import TemporalGraphEstimator
from .observability import OBSERVABILITY_CONTEXT_DIM, compute_observability_context_features, stations_from_ecef
from .simulation import DatasetConfig
from .utils.runtime import resolve_device


@dataclass(frozen=True)
class TrainConfig:
    window_size: int
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    gnn_layers: int
    gru_layers: int
    dropout: float
    gradient_clip_norm: float
    early_stopping_patience: int
    num_workers: int
    device: str = "auto"
    residual_reg_weight: float = 0.05
    residual_activity_weight: float = 0.0
    residual_activity_floor: float = 0.0
    fusion_entropy_weight: float = 0.0
    fusion_entropy_floor: float = 0.0
    visibility_consistency_weight: float = 0.0
    use_amp: bool = True
    prior_stats_dim: int = 18


def parse_train_config(cfg: dict[str, Any]) -> TrainConfig:
    return TrainConfig(**cfg)


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return values.mean()
    w = sample_weight.reshape(-1).clamp(min=1e-8)
    return torch.sum(values.reshape(-1) * w) / torch.sum(w)


def _base_loss(
    pred_state: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None,
    mse_weight: float = 0.6,
) -> torch.Tensor:
    sq = (pred_state - target) ** 2
    nll = 0.5 * (torch.exp(-pred_logvar) * sq + pred_logvar)
    per_item = mse_weight * sq.mean(dim=1) + (1.0 - mse_weight) * nll.mean(dim=1)
    return _weighted_mean(per_item, sample_weight)


def _loss_with_regularizers(
    out: dict[str, torch.Tensor],
    target: torch.Tensor,
    sample_weight: torch.Tensor | None,
    visibility: torch.Tensor,
    cfg: TrainConfig,
    fallback_prior: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = _base_loss(out["state"], out["logvar"], target, sample_weight)
    stats: dict[str, float] = {"base_loss": float(loss.detach().item())}

    reference_prior = out.get("fused_prior")
    if reference_prior is None:
        reference_prior = fallback_prior
    if reference_prior is not None and cfg.residual_reg_weight > 0.0:
        residual_reg = torch.mean((out["state"] - reference_prior) ** 2, dim=1)
        loss = loss + cfg.residual_reg_weight * _weighted_mean(residual_reg, sample_weight)
        stats["residual_reg"] = float(residual_reg.mean().detach().item())

    residual = out.get("residual")
    if residual is not None and cfg.residual_activity_weight > 0.0:
        activity = torch.mean(torch.abs(residual), dim=1)
        activity_penalty = torch.relu(
            torch.full_like(activity, float(cfg.residual_activity_floor)) - activity
        )
        loss = loss + cfg.residual_activity_weight * _weighted_mean(activity_penalty, sample_weight)
        stats["residual_activity"] = float(activity.mean().detach().item())

    fusion_weights = out.get("fusion_weights")
    if fusion_weights is not None and cfg.fusion_entropy_weight > 0.0:
        probs = fusion_weights.clamp(min=1e-8, max=1.0)
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean(dim=-1)
        entropy = entropy / np.log(float(probs.shape[-1]))
        entropy_penalty = torch.relu(
            torch.full_like(entropy, float(cfg.fusion_entropy_floor)) - entropy
        )
        loss = loss + cfg.fusion_entropy_weight * _weighted_mean(entropy_penalty, sample_weight)
        stats["fusion_entropy"] = float(entropy.mean().detach().item())

    if residual is not None and cfg.visibility_consistency_weight > 0.0:
        vis_rate = visibility.mean(dim=(1, 2, 3))
        vis_penalty = (1.0 - vis_rate) * torch.mean(residual**2, dim=1)
        loss = loss + cfg.visibility_consistency_weight * _weighted_mean(vis_penalty, sample_weight)
        stats["visibility_penalty"] = float(vis_penalty.mean().detach().item())

    noise_scale = out.get("noise_scale")
    if noise_scale is not None:
        stats["noise_scale_mean"] = float(noise_scale.mean().detach().item())

    return loss, stats


def _epoch_step(
    model: TemporalGraphEstimator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    steps = 0
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for batch in pbar:
        meas = batch["measurements"].to(device)
        vis = batch["visibility"].to(device)
        station_xyz = batch["station_xyz"].to(device)
        target = batch["target"].to(device)
        sample_weight = batch.get("sample_weight")
        sample_weight = sample_weight.to(device) if sample_weight is not None else None

        prior = batch.get("ekf_prior")
        prior = prior.to(device) if prior is not None else None
        secondary_prior = batch.get("secondary_prior")
        secondary_prior = secondary_prior.to(device) if secondary_prior is not None else None
        innovation_features = batch.get("innovation_features")
        innovation_features = innovation_features.to(device) if innovation_features is not None else None
        prior_bank = batch.get("prior_bank")
        prior_bank = prior_bank.to(device) if prior_bank is not None else None
        prior_bank_stats = batch.get("prior_bank_stats")
        prior_bank_stats = prior_bank_stats.to(device) if prior_bank_stats is not None else None

        with torch.set_grad_enabled(is_train):
            autocast_device = "cuda" if use_amp else "cpu"
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                out = model(
                    measurements=meas,
                    visibility=vis,
                    station_xyz=station_xyz,
                    ekf_prior=prior,
                    secondary_prior=secondary_prior,
                    innovation_features=innovation_features,
                    prior_bank=prior_bank,
                    prior_bank_stats=prior_bank_stats,
                )
                loss, aux_stats = _loss_with_regularizers(
                    out=out,
                    target=target,
                    sample_weight=sample_weight,
                    visibility=vis,
                    cfg=cfg,
                    fallback_prior=prior,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()

        total += float(loss.item())
        steps += 1
        postfix = {"loss": f"{(total / max(steps, 1)):.4f}"}
        if "residual_activity" in aux_stats:
            postfix["res"] = f"{aux_stats['residual_activity']:.4f}"
        if "fusion_entropy" in aux_stats:
            postfix["entropy"] = f"{aux_stats['fusion_entropy']:.3f}"
        pbar.set_postfix(postfix)
    if steps == 0:
        return float("nan")
    return total / steps


def _validation_loss_across_loaders(
    model: TemporalGraphEstimator,
    loaders: list[DataLoader],
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    weighted_total = 0.0
    total_items = 0
    for loader in loaders:
        loss = _epoch_step(
            model=model,
            loader=loader,
            optimizer=None,
            device=device,
            cfg=cfg,
        )
        if not math.isfinite(loss):
            continue
        n_items = len(loader.dataset)
        weighted_total += float(loss) * n_items
        total_items += n_items
    if total_items == 0:
        return float("nan")
    return weighted_total / total_items


def _ensure_arrays_for_model(
    arrays: DatasetArrays,
    *,
    dataset_cfg: DatasetConfig | None,
    baseline_cfg: BaselineConfig | None,
    seed: int,
    model_kwargs: dict[str, Any],
) -> DatasetArrays:
    need_innovation = bool(model_kwargs.get("use_innovation_features", False))
    need_prior_bank = bool(model_kwargs.get("use_prior_bank_fusion", False))
    need_dual_prior = bool(model_kwargs.get("use_dual_prior_fusion", False))
    need_observability_context = bool(model_kwargs.get("use_observability_context", False))
    if not (need_innovation or need_prior_bank or need_dual_prior or need_observability_context):
        return arrays
    if dataset_cfg is None or baseline_cfg is None:
        raise ValueError("dataset_cfg and baseline_cfg are required for prior/innovation-conditioned models.")

    out = arrays
    need_ukf = need_prior_bank and out.ukf_prior is None
    need_aukf = (need_prior_bank or need_dual_prior) and out.aukf_prior is None
    need_ekf = out.ekf_prior is None
    if need_ekf or need_ukf or need_aukf:
        if out.x0_estimates is None:
            raise ValueError("x0_estimates are required to compute missing filter priors.")
        filters = run_filter_baselines(
            states=out.states,
            measurements=out.measurements,
            visibility=out.visibility,
            times=out.times,
            dataset_cfg=dataset_cfg,
            baseline_cfg=baseline_cfg,
            seed=seed,
            x0_estimates=out.x0_estimates,
        )
        out = with_full_priors(
            out,
            ekf_prior=filters.get("ekf", out.ekf_prior),
            ukf_prior=filters.get("ukf", out.ukf_prior),
            aukf_prior=filters.get("aukf", out.aukf_prior),
        )

    if need_innovation and out.innovation_features is None:
        if out.ekf_prior is None:
            raise ValueError("EKF prior is required for innovation features.")
        out = with_innovation_features(
            out,
            compute_innovation_features(
                prior_states=out.ekf_prior,
                measurements=out.measurements,
                visibility=out.visibility,
                times_s=out.times,
                stations=dataset_cfg.stations,
                meas_std_vector=dataset_cfg.measurement_noise.std_vector,
            ),
        )
    if need_prior_bank and (out.prior_bank_stats is None or need_observability_context):
        prior_stats = compute_prior_bank_stats(out.ekf_prior, out.ukf_prior, out.aukf_prior)
        if need_observability_context:
            stations = dataset_cfg.stations
            if len(stations) != int(out.station_ecef.shape[0]):
                stations = stations_from_ecef(out.station_ecef)
            obs_stats = compute_observability_context_features(
                prior_states=out.ekf_prior,
                visibility=out.visibility,
                times_s=out.times,
                stations=stations,
                meas_std_vector=dataset_cfg.measurement_noise.std_vector,
            )
            prior_stats = np.concatenate([prior_stats, obs_stats], axis=-1)
        out = with_prior_bank_stats(out, prior_stats)
    if need_dual_prior and out.secondary_prior is None and out.aukf_prior is not None:
        out = with_full_priors(out, aukf_prior=out.aukf_prior)
    return out


def train_model(
    train_arrays: DatasetArrays,
    val_arrays: DatasetArrays | list[DatasetArrays],
    cfg: TrainConfig,
    output_dir: str | Path,
    seed: int,
    use_ekf_prior: bool,
    model_kwargs: dict[str, Any] | None = None,
    dataset_cfg: DatasetConfig | None = None,
    baseline_cfg: BaselineConfig | None = None,
    checkpoint_name: str | None = None,
    initial_checkpoint: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[TemporalGraphEstimator, dict[str, list[float]], Path]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if model_kwargs is None:
        model_kwargs = {}
    device = device or resolve_device(cfg.device)

    train_arrays = _ensure_arrays_for_model(
        train_arrays,
        dataset_cfg=dataset_cfg,
        baseline_cfg=baseline_cfg,
        seed=seed,
        model_kwargs=model_kwargs,
    )
    if isinstance(val_arrays, list):
        prepared_val_arrays = [
            _ensure_arrays_for_model(
                arrays,
                dataset_cfg=dataset_cfg,
                baseline_cfg=baseline_cfg,
                seed=seed + 10_000 + idx,
                model_kwargs=model_kwargs,
            )
            for idx, arrays in enumerate(val_arrays)
        ]
    else:
        prepared_val_arrays = [
            _ensure_arrays_for_model(
                val_arrays,
                dataset_cfg=dataset_cfg,
                baseline_cfg=baseline_cfg,
                seed=seed + 10_000,
                model_kwargs=model_kwargs,
            )
        ]

    train_ds = WindowedSatelliteDataset(train_arrays, window_size=cfg.window_size, require_prior=use_ekf_prior)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loaders = [
        DataLoader(
            WindowedSatelliteDataset(arrays, window_size=cfg.window_size, require_prior=use_ekf_prior),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        for arrays in prepared_val_arrays
    ]

    merged_kwargs = dict(model_kwargs)
    if bool(merged_kwargs.get("use_prior_bank_fusion", False)):
        merged_kwargs.setdefault("prior_bank_size", 3)
        default_prior_stats_dim = cfg.prior_stats_dim
        if bool(merged_kwargs.get("use_observability_context", False)):
            default_prior_stats_dim += OBSERVABILITY_CONTEXT_DIM
        merged_kwargs.setdefault("prior_stats_dim", default_prior_stats_dim)

    model = TemporalGraphEstimator(
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
        use_ekf_prior=use_ekf_prior,
        **merged_kwargs,
    ).to(device)

    if initial_checkpoint is not None and Path(initial_checkpoint).exists():
        ckpt = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_name = "best_hybrid.pt" if use_ekf_prior else "best_gnn.pt"
    best_path = out_dir / (checkpoint_name or default_name)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience = 0

    for epoch in range(cfg.num_epochs):
        train_loss = _epoch_step(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
        )
        if len(val_loaders) == 1:
            val_loss = _epoch_step(
                model=model,
                loader=val_loaders[0],
                optimizer=None,
                device=device,
                cfg=cfg,
            )
        else:
            val_loss = _validation_loss_across_loaders(
                model=model,
                loaders=val_loaders,
                device=device,
                cfg=cfg,
            )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[epoch {epoch + 1:03d}] train={train_loss:.6f} val={val_loss:.6f}")

        score = val_loss if np.isfinite(val_loss) else train_loss
        if score < best_val:
            best_val = score
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg.__dict__,
                    "model_kwargs": merged_kwargs,
                    "use_ekf_prior": use_ekf_prior,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, history, best_path
