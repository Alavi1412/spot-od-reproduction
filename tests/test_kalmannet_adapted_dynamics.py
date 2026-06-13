"""Numerical-equivalence tests for the differentiable torch SPOT-OD dynamics.

The adapted KalmanNet baseline relies on torch-autograd implementations of the
state-transition and line-of-sight measurement functions. These tests assert
that the torch versions reproduce the numpy reference implementations (the same
code that generates the truth trajectories and synthetic measurements) to
tight tolerance, and that the torch graph is genuinely differentiable end to
end (the property that the faithful numpy-bridge transposition lacked).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gnn_state_estimation.coordinates import (  # noqa: E402
    StationGeometry,
    ecef_to_enu_matrix,
    line_of_sight_measurement,
    station_to_ecef,
)
from gnn_state_estimation.dynamics import kepler_to_cartesian, rk4_step  # noqa: E402
from gnn_state_estimation.kalmannet_adapted.torch_dynamics import (  # noqa: E402
    SPOTODTorchDynamics,
    los_measurement_torch,
    rk4_step_torch,
)

R_EARTH_M = 6378.1363e3
STATE_SCALE = np.array([1.0e7, 1.0e7, 1.0e7, 1.0e4, 1.0e4, 1.0e4], dtype=np.float64)


def _sample_states(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        alt = rng.uniform(450.0, 950.0) * 1e3
        a = R_EARTH_M + alt
        e = rng.uniform(0.0, 0.02)
        inc = np.deg2rad(rng.uniform(20.0, 98.0))
        raan = rng.uniform(0.0, 2 * np.pi)
        argp = rng.uniform(0.0, 2 * np.pi)
        nu = rng.uniform(0.0, 2 * np.pi)
        out.append(
            kepler_to_cartesian(
                semi_major_axis_m=a,
                eccentricity=e,
                inclination_rad=inc,
                raan_rad=raan,
                arg_perigee_rad=argp,
                true_anomaly_rad=nu,
            )
        )
    return np.stack(out, axis=0)


def _stations() -> tuple[StationGeometry, ...]:
    return (
        StationGeometry("Alaska", 64.84, -147.72, 140.0, 8.0),
        StationGeometry("Hawaii", 19.70, -155.08, 10.0, 8.0),
        StationGeometry("Spain", 40.43, -4.25, 700.0, 8.0),
        StationGeometry("Australia", -31.80, 115.89, 40.0, 8.0),
    )


def test_rk4_step_matches_numpy_reference() -> None:
    states = _sample_states(16)
    dt = 20.0
    ballistic = 0.018
    kwargs = dict(
        drag_rho_ref=4.0e-11, drag_h_ref_m=400e3, drag_scale_height_m=60e3
    )
    ref = np.stack(
        [
            rk4_step(s, dt=dt, ballistic_coeff_m2_per_kg=ballistic, t_s=0.0, **kwargs)
            for s in states
        ],
        axis=0,
    )
    x = torch.as_tensor(states, dtype=torch.float64)
    got = rk4_step_torch(
        x, dt, ballistic_coeff_m2_per_kg=ballistic, **kwargs
    ).numpy()
    # Position to <1 mm, velocity to <1e-6 m/s over a 20 s step on ~7000 km orbit.
    assert np.max(np.abs(got[:, :3] - ref[:, :3])) < 1e-3
    assert np.max(np.abs(got[:, 3:] - ref[:, 3:])) < 1e-6


def test_los_measurement_matches_numpy_reference() -> None:
    states = _sample_states(24, seed=11)
    stations = _stations()
    times = [0.0, 137.0, 1234.5, 2400.0]
    max_err = {"rho": 0.0, "az": 0.0, "el": 0.0, "rdot": 0.0}
    for t_s in times:
        for st in stations:
            st_ecef = torch.as_tensor(station_to_ecef(st), dtype=torch.float64)
            enu = torch.as_tensor(
                ecef_to_enu_matrix(st.lat_rad, st.lon_rad), dtype=torch.float64
            )
            x = torch.as_tensor(states, dtype=torch.float64)
            got = los_measurement_torch(x, t_s, st_ecef, enu).numpy()
            for i, s in enumerate(states):
                ref, _ = line_of_sight_measurement(s, st, t_s)
                max_err["rho"] = max(max_err["rho"], abs(got[i, 0] - ref[0]))
                # azimuth compared modulo 2*pi
                daz = abs((got[i, 1] - ref[1] + np.pi) % (2 * np.pi) - np.pi)
                max_err["az"] = max(max_err["az"], daz)
                max_err["el"] = max(max_err["el"], abs(got[i, 2] - ref[2]))
                max_err["rdot"] = max(max_err["rdot"], abs(got[i, 3] - ref[3]))
    assert max_err["rho"] < 1e-3, max_err
    assert max_err["az"] < 1e-7, max_err
    assert max_err["el"] < 1e-7, max_err
    assert max_err["rdot"] < 1e-5, max_err


def test_dynamics_container_scaled_roundtrip_and_gradient() -> None:
    stations = _stations()
    dyn = SPOTODTorchDynamics(
        stations,
        dt_s=20.0,
        ballistic_coeff_m2_per_kg=0.018,
        drag_rho_ref=4.0e-11,
        drag_h_ref_m=400e3,
        drag_scale_height_m=60e3,
        state_scale=STATE_SCALE,
        device="cpu",
        dtype=torch.float64,
    )
    states = _sample_states(8, seed=3)
    x_scaled = torch.tensor(
        states / STATE_SCALE, dtype=torch.float64, requires_grad=True
    )
    x_next = dyn.f_scaled(x_scaled)
    meas = dyn.h_scaled_all_stations(x_next, t_s=60.0)
    # Scaled f matches numpy reference after unscaling.
    ref = np.stack(
        [
            rk4_step(
                s,
                dt=20.0,
                ballistic_coeff_m2_per_kg=0.018,
                t_s=0.0,
                drag_rho_ref=4.0e-11,
                drag_h_ref_m=400e3,
                drag_scale_height_m=60e3,
            )
            for s in states
        ],
        axis=0,
    )
    got_phys = (x_next.detach().numpy()) * STATE_SCALE
    assert np.max(np.abs(got_phys[:, :3] - ref[:, :3])) < 1e-3
    # End-to-end differentiability: a scalar of the measurement has finite,
    # non-zero gradient w.r.t. the input scaled state (BPTT path intact).
    loss = (meas[..., 0] ** 2).mean()
    loss.backward()
    assert x_scaled.grad is not None
    assert torch.isfinite(x_scaled.grad).all()
    assert float(x_scaled.grad.abs().sum()) > 0.0
