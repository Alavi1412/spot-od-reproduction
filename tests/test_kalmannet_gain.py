"""Tests for the literature-derived KalmanNet-style learned-gain baseline.

These cover the new ``KalmanNetGain`` mode of ``TemporalGraphEstimator``
(Revach et al., IEEE TSP 2022 inspired): it must require the EKF prior and
innovation features, expose the explicit ``posterior = prior + gain @ innov``
update with diagnostics, and reduce to the EKF prior when the innovation is
zero. It also exercises the lightweight ``--models`` training filter.
"""

import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnn_state_estimation.models import TemporalGraphEstimator
from scripts.train_models import (
    build_parser,
    parse_models_filter,
    should_train_model,
)


def _make_gain_model(**overrides) -> TemporalGraphEstimator:
    kwargs = dict(
        hidden_dim=16,
        gnn_layers=1,
        gru_layers=1,
        dropout=0.0,
        use_ekf_prior=True,
        use_innovation_features=True,
        use_graph=False,
        kalmannet_gain=True,
        kalmannet_innov_dim=4,
        kalmannet_gain_scale=1.0e-3,
        kalmannet_correction_clip=5.0e-3,
        bounded_residual=True,
    )
    kwargs.update(overrides)
    model = TemporalGraphEstimator(**kwargs)
    model.eval()
    return model


class KalmanNetGainModeTests(unittest.TestCase):
    def test_gain_mode_requires_ekf_prior(self) -> None:
        with self.assertRaises(ValueError):
            TemporalGraphEstimator(
                hidden_dim=16,
                gnn_layers=1,
                gru_layers=1,
                dropout=0.0,
                use_ekf_prior=False,
                use_innovation_features=True,
                kalmannet_gain=True,
            )

    def test_gain_mode_requires_innovation_features(self) -> None:
        with self.assertRaises(ValueError):
            TemporalGraphEstimator(
                hidden_dim=16,
                gnn_layers=1,
                gru_layers=1,
                dropout=0.0,
                use_ekf_prior=True,
                use_innovation_features=False,
                kalmannet_gain=True,
            )

    def test_gain_mode_incompatible_with_prior_bank_fusion(self) -> None:
        with self.assertRaises(ValueError):
            TemporalGraphEstimator(
                hidden_dim=16,
                gnn_layers=1,
                gru_layers=1,
                dropout=0.0,
                use_ekf_prior=True,
                use_innovation_features=True,
                kalmannet_gain=True,
                use_prior_bank_fusion=True,
                prior_bank_size=3,
            )

    def test_rejects_non_positive_correction_clip(self) -> None:
        with self.assertRaises(ValueError):
            TemporalGraphEstimator(
                hidden_dim=16,
                gnn_layers=1,
                gru_layers=1,
                dropout=0.0,
                use_ekf_prior=True,
                use_innovation_features=True,
                kalmannet_gain=True,
                kalmannet_correction_clip=0.0,
            )

    def test_saturated_gain_and_innovation_respect_correction_bound(self) -> None:
        """The stabilization contract: even with a fully saturated raw gain and
        a saturated/over-range innovation, the per-component correction in
        normalized state units must stay strictly within the configured bound.

        This guards against the prior 0.1 gain scale, which (with innovations
        normalized by measurement sigma and clipped to +/-12) could imply
        tens-of-thousands-of-km position corrections.
        """
        torch.manual_seed(3)
        clip = 5.0e-3
        model = _make_gain_model(kalmannet_correction_clip=clip)
        # Drive the gain head hard so kalmannet_gain_scale * tanh(raw) saturates.
        with torch.no_grad():
            model.gain_head.weight.normal_(0.0, 50.0)
            model.gain_head.bias.normal_(0.0, 50.0)

        bsz, window, n_stations = 4, 5, 4
        ekf_prior = torch.randn(bsz, 6)
        innovation_features = torch.zeros(bsz, window, n_stations, 6)
        # Push the residual channels well past the +/-12 upstream clip to prove
        # the bound holds regardless of innovation magnitude.
        innovation_features[..., :4] = 50.0
        out = model(
            measurements=torch.randn(bsz, window, n_stations, 4),
            visibility=torch.ones(bsz, window, n_stations, 1),
            station_xyz=torch.randn(bsz, n_stations, 3),
            ekf_prior=ekf_prior,
            innovation_features=innovation_features,
        )
        max_abs_correction = out["correction"].abs().max().item()
        # Strictly bounded by the configured clip (tanh < 1); allow a tiny
        # numerical tolerance on the boundary.
        self.assertLessEqual(max_abs_correction, clip + 1e-9)
        self.assertLess(max_abs_correction, clip)
        # The applied posterior remains exactly prior + bounded correction.
        np.testing.assert_allclose(
            out["state"].detach().numpy(),
            (ekf_prior + out["correction"]).detach().numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out["correction_clip"].detach().numpy(),
            np.full((bsz, 1), clip, dtype=np.float32),
            rtol=1e-6,
        )

    def test_forward_pass_shapes(self) -> None:
        torch.manual_seed(0)
        model = _make_gain_model()
        bsz, window, n_stations, innov_dim = 3, 5, 4, 4
        out = model(
            measurements=torch.randn(bsz, window, n_stations, 4),
            visibility=torch.ones(bsz, window, n_stations, 1),
            station_xyz=torch.randn(bsz, n_stations, 3),
            ekf_prior=torch.randn(bsz, 6),
            innovation_features=torch.randn(bsz, window, n_stations, 6),
        )
        self.assertEqual(tuple(out["state"].shape), (bsz, 6))
        self.assertEqual(tuple(out["logvar"].shape), (bsz, 6))
        self.assertEqual(tuple(out["learned_gain"].shape), (bsz, 6, innov_dim))
        self.assertEqual(tuple(out["innovation_vector"].shape), (bsz, innov_dim))
        self.assertEqual(tuple(out["correction"].shape), (bsz, 6))
        self.assertEqual(tuple(out["raw_correction"].shape), (bsz, 6))
        self.assertEqual(tuple(out["correction_clip"].shape), (bsz, 1))
        self.assertEqual(tuple(out["residual"].shape), (bsz, 6))
        self.assertEqual(tuple(out["budget"].shape), (bsz, 1))
        self.assertIn("fused_prior", out)

    def test_zero_innovation_gives_zero_correction(self) -> None:
        """With a zero innovation the posterior must equal the EKF prior.

        Randomise the gain head so the property holds structurally (no additive
        bias on the correction), not just at the zero-initialised start.
        """
        torch.manual_seed(1)
        model = _make_gain_model()
        with torch.no_grad():
            model.gain_head.weight.normal_(0.0, 1.0)
            model.gain_head.bias.normal_(0.0, 1.0)

        bsz, window, n_stations = 4, 5, 4
        ekf_prior = torch.randn(bsz, 6)
        # Innovation features all zero (channels 0-3 carry the residual).
        innovation_features = torch.zeros(bsz, window, n_stations, 6)
        out = model(
            measurements=torch.randn(bsz, window, n_stations, 4),
            visibility=torch.ones(bsz, window, n_stations, 1),
            station_xyz=torch.randn(bsz, n_stations, 3),
            ekf_prior=ekf_prior,
            innovation_features=innovation_features,
        )
        np.testing.assert_allclose(
            out["correction"].detach().numpy(),
            np.zeros((bsz, 6), dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            out["state"].detach().numpy(),
            ekf_prior.detach().numpy(),
            atol=1e-6,
        )

    def test_nonzero_innovation_moves_off_prior(self) -> None:
        torch.manual_seed(2)
        model = _make_gain_model()
        with torch.no_grad():
            model.gain_head.weight.normal_(0.0, 1.0)
            model.gain_head.bias.normal_(0.0, 1.0)

        bsz, window, n_stations = 2, 5, 4
        ekf_prior = torch.randn(bsz, 6)
        innovation_features = torch.zeros(bsz, window, n_stations, 6)
        innovation_features[..., :4] = 0.7  # nonzero normalized residuals
        out = model(
            measurements=torch.randn(bsz, window, n_stations, 4),
            visibility=torch.ones(bsz, window, n_stations, 1),
            station_xyz=torch.randn(bsz, n_stations, 3),
            ekf_prior=ekf_prior,
            innovation_features=innovation_features,
        )
        correction_norm = torch.linalg.norm(out["correction"]).item()
        self.assertGreater(correction_norm, 1e-6)
        # Update is the explicit prior + correction.
        np.testing.assert_allclose(
            out["state"].detach().numpy(),
            (ekf_prior + out["correction"]).detach().numpy(),
            atol=1e-6,
        )


class TrainModelsFilterTests(unittest.TestCase):
    def test_parse_models_filter(self) -> None:
        self.assertIsNone(parse_models_filter(None))
        self.assertIsNone(parse_models_filter(""))
        self.assertIsNone(parse_models_filter("  , ,"))
        self.assertEqual(parse_models_filter("KalmanNetGain"), {"KalmanNetGain"})
        self.assertEqual(
            parse_models_filter(" KalmanNetGain , HybridGNN ,"),
            {"KalmanNetGain", "HybridGNN"},
        )

    def test_parser_accepts_models_flag(self) -> None:
        args = build_parser().parse_args(["--models", "KalmanNetGain"])
        self.assertEqual(args.models, "KalmanNetGain")
        default_args = build_parser().parse_args([])
        self.assertIsNone(default_args.models)

    def test_should_train_model_filter_skips_unrequested_enabled(self) -> None:
        models_cfg = {
            "KalmanNetGain": {"enabled": True},
            "HybridGNN": {"enabled": True},
            "DisabledModel": {"enabled": False},
        }
        filt = parse_models_filter("KalmanNetGain")
        trained = [
            name
            for name, spec in models_cfg.items()
            if should_train_model(name, spec, filt)
        ]
        # Only the requested model trains; other enabled models are skipped.
        self.assertEqual(trained, ["KalmanNetGain"])

        # Without a filter, every enabled model trains (disabled still skipped).
        trained_all = [
            name
            for name, spec in models_cfg.items()
            if should_train_model(name, spec, None)
        ]
        self.assertEqual(trained_all, ["KalmanNetGain", "HybridGNN"])


if __name__ == "__main__":
    unittest.main()
