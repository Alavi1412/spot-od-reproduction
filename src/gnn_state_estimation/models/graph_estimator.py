"""Spatio-temporal estimators for satellite state estimation."""

from __future__ import annotations

import torch
import torch.nn as nn


class GraphMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_fc = nn.Linear(hidden_dim, hidden_dim)
        self.msg_fc = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, S, H]
        s_count = h.size(1)
        if s_count == 1:
            neigh = h
        else:
            neigh = (h.sum(dim=1, keepdim=True) - h) / float(s_count - 1)
        z = torch.cat([self.self_fc(h), self.msg_fc(neigh)], dim=-1)
        out = self.update(z)
        out = self.dropout(out)
        return self.norm(h + out)


class LocalNodeLayer(nn.Module):
    """Per-station replacement for GraphMessageLayer without cross-node mixing."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_fc = nn.Linear(hidden_dim, hidden_dim)
        self.local_fc = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.self_fc(h), self.local_fc(h)], dim=-1)
        out = self.update(z)
        out = self.dropout(out)
        return self.norm(h + out)


class TemporalGraphEstimator(nn.Module):
    """Graph/sequence estimator over station-time measurements.

    The class supports three families of models under one interface:
    1. Pure sequence/graph regressors with no explicit prior.
    2. Single-prior residual estimators (EKF-like or dual-prior legacy path).
    3. Multi-prior fusion estimators that blend EKF/UKF/AUKF priors before
       applying a bounded residual correction.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        gnn_layers: int = 4,
        gru_layers: int = 2,
        dropout: float = 0.15,
        use_ekf_prior: bool = False,
        residual_scale: float = 0.02,
        use_gating: bool = True,
        bounded_residual: bool = True,
        use_innovation_features: bool = False,
        use_context_budget: bool = False,
        use_dual_prior_fusion: bool = False,
        use_graph: bool = True,
        use_prior_bank_fusion: bool = False,
        prior_bank_size: int = 0,
        prior_stats_dim: int = 0,
        use_observability_context: bool = False,
        predict_noise_scale: bool = False,
        fusion_temperature: float = 1.0,
        use_local_layers_when_no_graph: bool = False,
        kalmannet_gain: bool = False,
        kalmannet_innov_dim: int = 4,
        kalmannet_gain_scale: float = 1.0e-3,
        kalmannet_correction_clip: float = 5.0e-3,
    ) -> None:
        super().__init__()
        self.use_ekf_prior = use_ekf_prior
        self.residual_scale = residual_scale
        self.use_gating = use_gating
        self.bounded_residual = bounded_residual
        self.use_innovation_features = use_innovation_features
        self.use_context_budget = use_context_budget
        self.use_dual_prior_fusion = use_dual_prior_fusion
        self.use_graph = use_graph
        self.use_prior_bank_fusion = use_prior_bank_fusion
        self.use_local_layers_when_no_graph = bool(use_local_layers_when_no_graph)
        self.prior_bank_size = int(prior_bank_size)
        self.prior_stats_dim = int(prior_stats_dim)
        self.use_observability_context = use_observability_context
        self.predict_noise_scale = predict_noise_scale
        self.fusion_temperature = max(float(fusion_temperature), 1e-3)
        self.kalmannet_gain = bool(kalmannet_gain)
        self.kalmannet_innov_dim = int(kalmannet_innov_dim)
        self.kalmannet_gain_scale = float(kalmannet_gain_scale)
        self.kalmannet_correction_clip = float(kalmannet_correction_clip)
        if self.use_dual_prior_fusion and not self.use_ekf_prior:
            raise ValueError("Dual-prior fusion requires use_ekf_prior=True.")
        if self.use_prior_bank_fusion and self.prior_bank_size < 2:
            raise ValueError("Prior-bank fusion requires prior_bank_size >= 2.")
        if self.use_graph and self.use_local_layers_when_no_graph:
            raise ValueError("Local no-graph replacement layers require use_graph=False.")
        if self.kalmannet_gain:
            # Literature-derived KalmanNet-style learned Kalman-gain baseline
            # (Revach et al., IEEE TSP 2022): keep the EKF state-space flow and
            # learn the gain that maps the current innovation to the state
            # correction, posterior = EKF prior + gain @ innovation.
            if not self.use_ekf_prior:
                raise ValueError("KalmanNetGain mode requires use_ekf_prior=True.")
            if not self.use_innovation_features:
                raise ValueError("KalmanNetGain mode requires use_innovation_features=True.")
            if self.use_prior_bank_fusion or self.use_dual_prior_fusion:
                raise ValueError(
                    "KalmanNetGain mode is incompatible with prior-bank/dual-prior fusion."
                )
            if not (1 <= self.kalmannet_innov_dim <= 6):
                raise ValueError("kalmannet_innov_dim must be in [1, 6].")
            if not (self.kalmannet_correction_clip > 0.0):
                raise ValueError(
                    "kalmannet_correction_clip must be > 0 (normalized state units)."
                )
        innovation_dim = 6 if use_innovation_features else 0
        context_dim = 5 if use_context_budget else 0
        in_dim = 4 + 1 + 3 + innovation_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.graph_layers = nn.ModuleList(
            [GraphMessageLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(gnn_layers)]
            if (self.use_graph or not self.use_local_layers_when_no_graph)
            else []
        )
        self.local_layers = nn.ModuleList(
            [LocalNodeLayer(hidden_dim=hidden_dim, dropout=dropout) for _ in range(gnn_layers)]
            if (not self.use_graph and self.use_local_layers_when_no_graph)
            else []
        )
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        legacy_prior_dim = 0
        if not self.use_prior_bank_fusion:
            if use_ekf_prior:
                legacy_prior_dim += 6
            if self.use_dual_prior_fusion:
                legacy_prior_dim += 6
        prior_bank_flat_dim = self.prior_bank_size * 6 if self.use_prior_bank_fusion else 0
        head_in = 2 * hidden_dim + legacy_prior_dim + context_dim + prior_bank_flat_dim + self.prior_stats_dim

        self.shared_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.state_head = nn.Linear(hidden_dim // 2, 6)
        self.logvar_head = nn.Linear(hidden_dim // 2, 6)
        if self.kalmannet_gain:
            # Recurrent-feature-driven Kalman-gain head: maps the temporal
            # summary to a [6 x innov_dim] gain matrix. Zero-initialised so the
            # estimator starts exactly at the EKF prior (gain = 0) and learns
            # corrections from there, matching KalmanNet's warm-started flow.
            self.gain_head = nn.Linear(hidden_dim // 2, 6 * self.kalmannet_innov_dim)
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)
        if (self.use_ekf_prior or self.use_prior_bank_fusion) and self.use_gating:
            self.gate_head = nn.Linear(hidden_dim // 2, 6)
            nn.init.constant_(self.gate_head.bias, -1.6)
        if self.use_dual_prior_fusion:
            self.prior_gate_head = nn.Linear(hidden_dim // 2, 6)
            nn.init.constant_(self.prior_gate_head.bias, -1.2)
        if self.use_prior_bank_fusion:
            self.prior_bank_head = nn.Linear(hidden_dim // 2, self.prior_bank_size * 6)
        if (self.use_ekf_prior or self.use_prior_bank_fusion) and self.use_context_budget:
            self.budget_head = nn.Sequential(
                nn.Linear(head_in, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            nn.init.constant_(self.budget_head[-1].bias, -0.3)
        if self.predict_noise_scale:
            self.noise_scale_head = nn.Linear(hidden_dim // 2, 4)

    def forward(
        self,
        measurements: torch.Tensor,
        visibility: torch.Tensor,
        station_xyz: torch.Tensor,
        ekf_prior: torch.Tensor | None = None,
        secondary_prior: torch.Tensor | None = None,
        innovation_features: torch.Tensor | None = None,
        prior_bank: torch.Tensor | None = None,
        prior_bank_stats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # measurements: [B, W, S, 4], visibility: [B, W, S, 1], station_xyz: [B, S, 3] or [S, 3]
        bsz, window, _, _ = measurements.shape
        if station_xyz.dim() == 2:
            station_xyz = station_xyz.unsqueeze(0).expand(bsz, -1, -1)
        station_xyz = station_xyz.unsqueeze(1).expand(-1, window, -1, -1)

        inputs = [measurements, visibility, station_xyz]
        if self.use_innovation_features:
            if innovation_features is None:
                raise ValueError("Innovation features must be provided when use_innovation_features=True.")
            inputs.append(innovation_features)
        x = torch.cat(inputs, dim=-1)
        h = self.input_proj(x)

        pooled = []
        vis = visibility.clamp(min=0.0, max=1.0)
        for t in range(window):
            h_t = h[:, t]
            if self.use_graph:
                for layer in self.graph_layers:
                    h_t = layer(h_t)
            elif self.use_local_layers_when_no_graph:
                for layer in self.local_layers:
                    h_t = layer(h_t)
            w = vis[:, t]
            denom = w.sum(dim=1, keepdim=False).clamp(min=1.0)
            pooled_t = (h_t * w).sum(dim=1) / denom
            pooled.append(pooled_t)
        seq = torch.stack(pooled, dim=1)

        seq_out, _ = self.temporal(seq)
        summary = seq_out[:, -1, :]
        context_stats = None
        if self.use_context_budget:
            if innovation_features is None:
                raise ValueError("Innovation features are required for context-budget gating.")
            innov_energy = innovation_features[..., 4:5]
            pred_vis = innovation_features[..., 5:6]
            vis_sum = vis.sum(dim=(1, 2, 3)).clamp(min=1.0)
            mean_energy = (innov_energy * vis).sum(dim=(1, 2, 3)) / vis_sum
            max_energy = (innov_energy * vis).amax(dim=(1, 2, 3))
            visibility_rate = vis.mean(dim=(1, 2, 3))
            pred_visibility_rate = pred_vis.mean(dim=(1, 2, 3))
            visibility_mismatch = torch.abs(pred_vis - vis).mean(dim=(1, 2, 3))
            context_stats = torch.stack(
                [mean_energy, max_energy, visibility_rate, pred_visibility_rate, visibility_mismatch],
                dim=-1,
            )
            summary = torch.cat([summary, context_stats], dim=-1)

        fused_prior: torch.Tensor | None = None
        fusion_weights: torch.Tensor | None = None
        prior_gate: torch.Tensor | None = None

        if self.use_prior_bank_fusion:
            if prior_bank is None:
                raise ValueError("prior_bank must be provided when use_prior_bank_fusion=True.")
            if prior_bank.shape[1] != self.prior_bank_size:
                raise ValueError(
                    f"Expected prior_bank_size={self.prior_bank_size}, got {prior_bank.shape[1]}."
                )
            summary = torch.cat([summary, prior_bank.reshape(bsz, -1)], dim=-1)
            if self.prior_stats_dim > 0:
                if prior_bank_stats is None:
                    raise ValueError("prior_bank_stats must be provided when prior_stats_dim > 0.")
                summary = torch.cat([summary, prior_bank_stats], dim=-1)
        else:
            if self.use_ekf_prior:
                if ekf_prior is None:
                    raise ValueError("EKF prior must be provided when use_ekf_prior=True.")
                summary = torch.cat([summary, ekf_prior], dim=-1)
                if self.use_dual_prior_fusion:
                    if secondary_prior is None:
                        raise ValueError("Secondary prior must be provided when use_dual_prior_fusion=True.")
                    summary = torch.cat([summary, secondary_prior], dim=-1)

        feat = self.shared_head(summary)
        raw_state = self.state_head(feat)
        logvar = self.logvar_head(feat).clamp(min=-8.0, max=4.0)

        if self.kalmannet_gain:
            if ekf_prior is None:
                raise ValueError("KalmanNetGain mode requires the EKF prior tensor.")
            if innovation_features is None:
                raise ValueError("KalmanNetGain mode requires innovation features.")
            # Aggregate the current-step normalized measurement innovation
            # across stations (visibility-weighted): channels 0..d-1 are the
            # [range, az, el, range-rate] normalized residuals.
            innov_meas = innovation_features[..., : self.kalmannet_innov_dim]
            cur_innov = innov_meas[:, -1]            # [B, S, d]
            cur_w = vis[:, -1]                       # [B, S, 1]
            denom = cur_w.sum(dim=1).clamp(min=1.0)  # [B, 1]
            innovation_vector = (cur_innov * cur_w).sum(dim=1) / denom  # [B, d]
            raw_gain = self.gain_head(feat).view(bsz, 6, self.kalmannet_innov_dim)
            if self.bounded_residual:
                gain = self.kalmannet_gain_scale * torch.tanh(raw_gain)
            else:
                gain = self.kalmannet_gain_scale * raw_gain
            # Explicit Kalman-style update: posterior = prior + gain @ innovation.
            # No additive bias, so a zero innovation yields an exact zero
            # correction relative to the EKF prior.
            raw_correction = torch.bmm(gain, innovation_vector.unsqueeze(-1)).squeeze(-1)
            # The state is normalized by STATE_SCALE (position 1.0 == 1e7 m,
            # velocity 1.0 == 1e4 m/s). A raw learned gain acting on the
            # innovation channels (each normalized by measurement sigma and
            # clipped to +/-12) can imply physically absurd, kilometer-scale
            # corrections that destabilize the observed step. Bound the
            # per-component correction in normalized state units with a smooth,
            # saturating clamp: |correction_i| < kalmannet_correction_clip
            # strictly, even with a saturated raw gain and innovation, while the
            # derivative is 1 at the origin so the explicit Kalman update is
            # recovered exactly in the small-correction regime. tanh(0)=0 keeps
            # the zero-innovation => exact-zero-correction contract intact (no
            # additive bias is introduced by the bound).
            c = self.kalmannet_correction_clip
            correction = c * torch.tanh(raw_correction / c)
            state = ekf_prior + correction
            return {
                "state": state,
                "logvar": logvar,
                "budget": torch.ones((bsz, 1), dtype=state.dtype, device=state.device),
                "gate": torch.ones_like(state),
                "residual": correction,
                "fused_prior": ekf_prior,
                "learned_gain": gain,
                "innovation_vector": innovation_vector,
                "correction": correction,
                "raw_correction": raw_correction,
                "correction_clip": torch.full(
                    (bsz, 1), c, dtype=state.dtype, device=state.device
                ),
            }

        if self.use_prior_bank_fusion:
            logits = self.prior_bank_head(feat).view(bsz, 6, self.prior_bank_size) / self.fusion_temperature
            fusion_weights = torch.softmax(logits, dim=-1)
            prior_bank_t = prior_bank.transpose(1, 2)
            fused_prior = torch.sum(fusion_weights * prior_bank_t, dim=-1)
            prior_gate = fusion_weights.mean(dim=-1)
        elif self.use_ekf_prior:
            if self.use_dual_prior_fusion:
                prior_gate = torch.sigmoid(self.prior_gate_head(feat))
                fused_prior = prior_gate * ekf_prior + (1.0 - prior_gate) * secondary_prior
            else:
                fused_prior = ekf_prior
                prior_gate = torch.ones_like(ekf_prior)

        if self.bounded_residual:
            residual = self.residual_scale * torch.tanh(raw_state)
        else:
            residual = self.residual_scale * raw_state

        noise_scale = None
        if self.predict_noise_scale:
            noise_scale = torch.clamp(0.25 + torch.nn.functional.softplus(self.noise_scale_head(feat)), 0.25, 8.0)
            residual = residual * (1.0 + 0.1 * (noise_scale.mean(dim=-1, keepdim=True) - 1.0))

        if fused_prior is not None:
            if self.use_context_budget:
                budget = torch.sigmoid(self.budget_head(summary))
                residual = budget * residual
            else:
                budget = torch.ones((bsz, 1), dtype=residual.dtype, device=residual.device)
            if self.use_gating:
                gate = torch.sigmoid(self.gate_head(feat))
            else:
                gate = torch.ones_like(residual)
            state = fused_prior + gate * residual
        else:
            budget = torch.ones((bsz, 1), dtype=raw_state.dtype, device=raw_state.device)
            gate = torch.ones_like(raw_state)
            state = raw_state

        out = {
            "state": state,
            "logvar": logvar,
            "budget": budget,
            "gate": gate,
            "residual": residual,
        }
        if fused_prior is not None:
            out["fused_prior"] = fused_prior
        if prior_gate is not None:
            out["prior_gate"] = prior_gate
        if fusion_weights is not None:
            out["fusion_weights"] = fusion_weights
        if noise_scale is not None:
            out["noise_scale"] = noise_scale
        if context_stats is not None:
            out["context_stats"] = context_stats
        return out


def heteroscedastic_loss(
    pred_state: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.6,
) -> torch.Tensor:
    sq = (pred_state - target) ** 2
    nll = 0.5 * (torch.exp(-pred_logvar) * sq + pred_logvar)
    return mse_weight * sq.mean() + (1.0 - mse_weight) * nll.mean()
