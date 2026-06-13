"""Differentiable torch implementations of the SPOT-OD compact dynamics and
line-of-sight measurement model.

These mirror, term for term, the numpy reference implementations used to
generate the truth trajectories and synthetic measurements:

* ``gnn_state_estimation.dynamics.rk4_step`` with two-body + J2 + rotating
  exponential-atmosphere drag (third-body and SRP are NOT included, matching
  the compact model used by ``run_kalmannet_spot_od_transposition._propagate``
  and every classical baseline scored against this truth).
* ``gnn_state_estimation.coordinates.line_of_sight_measurement`` (ECI -> ECEF
  via Earth-rotation, station ECEF, ENU azimuth/elevation, range and
  range-rate).

The whole point is that every operation here is a torch op with autograd, so
the recurrent KalmanNet rollout keeps a differentiable graph from the loss all
the way back through the propagator and measurement model. The numpy reference
``.detach()``-es f/h, which is documented diagnosis root cause R3.

Equivalence with the numpy reference is asserted by
``tests/test_kalmannet_adapted_dynamics.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..constants import (
    EARTH_ROTATION_RATE,
    J2,
    MU_EARTH,
    R_EARTH,
)
from ..coordinates import (
    StationGeometry,
    ecef_to_enu_matrix,
    station_to_ecef,
)


# --- State transition -----------------------------------------------------


def _atmospheric_density_torch(
    altitude_m: torch.Tensor,
    rho_ref: float,
    h_ref_m: float,
    scale_height_m: float,
) -> torch.Tensor:
    h = torch.clamp(altitude_m, min=0.0)
    return rho_ref * torch.exp(-(h - h_ref_m) / scale_height_m)


def state_derivative_torch(
    state: torch.Tensor,
    *,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
) -> torch.Tensor:
    """Compact two-body + J2 + drag derivative in PHYSICAL ECI units.

    ``state`` shape: ``(..., 6)`` (position metres, velocity m/s). Returns the
    time-derivative with the same shape. Time-independent (the compact model
    has no third-body/SRP terms; the rotating atmosphere depends on position,
    not absolute time).
    """
    r = state[..., :3]
    v = state[..., 3:]
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]
    r2 = (r * r).sum(dim=-1)
    r_norm = torch.sqrt(r2)
    # Overflow guard: clamp the radius used in the gravity/J2 denominators to a
    # floor well below any real orbit (0.3 R_EARTH). For valid LEO states
    # (r ~ R_EARTH + altitude) this is a no-op, so the numpy-reference
    # equivalence is unaffected; it only prevents the -mu*r/|r|^3 singularity
    # from producing NaN/Inf when a transient bad gain drives the posterior
    # toward the geocentre during early training.
    r_floor = 0.3 * R_EARTH
    r_norm_safe = torch.clamp(r_norm, min=r_floor)
    r2_safe = r_norm_safe * r_norm_safe
    r3 = r_norm_safe * r2_safe
    r5 = r3 * r2_safe

    # Two-body gravity
    a_grav = -MU_EARTH * r / r3.unsqueeze(-1)

    # J2 perturbation
    factor = 1.5 * J2 * MU_EARTH * (R_EARTH ** 2) / r5
    common = 5.0 * (z * z) / r2_safe
    a_j2 = factor.unsqueeze(-1) * torch.stack(
        [x * (common - 1.0), y * (common - 1.0), z * (common - 3.0)], dim=-1
    )

    # Drag with rotating atmosphere
    altitude = r_norm - R_EARTH
    rho = _atmospheric_density_torch(
        altitude, drag_rho_ref, drag_h_ref_m, drag_scale_height_m
    )
    # v_atm = omega x r, omega = [0, 0, EARTH_ROTATION_RATE]
    v_atm = torch.stack(
        [-EARTH_ROTATION_RATE * y, EARTH_ROTATION_RATE * x, torch.zeros_like(z)],
        dim=-1,
    )
    v_rel = v - v_atm
    v_rel_norm = torch.sqrt((v_rel * v_rel).sum(dim=-1))
    a_drag = -0.5 * ballistic_coeff_m2_per_kg * (rho * v_rel_norm).unsqueeze(-1) * v_rel

    a_total = a_grav + a_j2 + a_drag
    return torch.cat([v, a_total], dim=-1)


def rk4_step_torch(
    state: torch.Tensor,
    dt: float,
    *,
    ballistic_coeff_m2_per_kg: float,
    drag_rho_ref: float = 4.0e-11,
    drag_h_ref_m: float = 400e3,
    drag_scale_height_m: float = 60e3,
) -> torch.Tensor:
    """Single RK4 step in PHYSICAL ECI units. ``state`` shape ``(..., 6)``."""
    kwargs = dict(
        ballistic_coeff_m2_per_kg=ballistic_coeff_m2_per_kg,
        drag_rho_ref=drag_rho_ref,
        drag_h_ref_m=drag_h_ref_m,
        drag_scale_height_m=drag_scale_height_m,
    )
    k1 = state_derivative_torch(state, **kwargs)
    k2 = state_derivative_torch(state + 0.5 * dt * k1, **kwargs)
    k3 = state_derivative_torch(state + 0.5 * dt * k2, **kwargs)
    k4 = state_derivative_torch(state + dt * k3, **kwargs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# --- Line-of-sight measurement --------------------------------------------


def _wrap_az_torch(az: torch.Tensor) -> torch.Tensor:
    """Wrap azimuth into [0, 2*pi) (matches the numpy reference convention)."""
    two_pi = 2.0 * np.pi
    return torch.remainder(az, two_pi)


def los_measurement_torch(
    state: torch.Tensor,
    t_s: torch.Tensor | float,
    station_ecef: torch.Tensor,
    enu_matrix: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Differentiable [range, azimuth, elevation, range_rate] for a single
    station.

    ``state`` shape ``(..., 6)`` in PHYSICAL ECI units; ``t_s`` scalar or
    broadcastable; ``station_ecef`` shape ``(3,)``; ``enu_matrix`` shape
    ``(3, 3)`` (ECEF->ENU rotation for the station). Returns shape ``(..., 4)``.

    Mirrors ``coordinates.line_of_sight_measurement`` including the
    inertial->ECEF velocity mapping (ground station fixed in ECEF). Azimuth is
    wrapped to ``[0, 2*pi)``; elevation uses a clamped ``arcsin``.
    """
    r_eci = state[..., :3]
    v_eci = state[..., 3:]
    if not torch.is_tensor(t_s):
        t_s = torch.as_tensor(t_s, dtype=state.dtype, device=state.device)
    theta = EARTH_ROTATION_RATE * t_s
    c = torch.cos(theta)
    s = torch.sin(theta)
    # rot_z(theta) @ r  (rows of rot_z: [c,s,0],[-s,c,0],[0,0,1])
    rx = r_eci[..., 0]
    ry = r_eci[..., 1]
    rz = r_eci[..., 2]
    r_ecef = torch.stack([c * rx + s * ry, -s * rx + c * ry, rz], dim=-1)
    # v_ecef = rot @ (v - omega x r), omega x r = [-ROT*ry, ROT*rx, 0]
    omega_cross_r = torch.stack(
        [-EARTH_ROTATION_RATE * ry, EARTH_ROTATION_RATE * rx, torch.zeros_like(rz)],
        dim=-1,
    )
    v_minus = v_eci - omega_cross_r
    vx = v_minus[..., 0]
    vy = v_minus[..., 1]
    vz = v_minus[..., 2]
    v_ecef = torch.stack([c * vx + s * vy, -s * vx + c * vy, vz], dim=-1)

    rel = r_ecef - station_ecef
    rel_v = v_ecef
    rho = torch.sqrt((rel * rel).sum(dim=-1) + eps)

    # ENU = enu_matrix @ rel (broadcasts over any leading batch/station dims)
    enu = torch.einsum("...ij,...j->...i", enu_matrix, rel)
    east = enu[..., 0]
    north = enu[..., 1]
    up = enu[..., 2]
    az = _wrap_az_torch(torch.atan2(east, north))
    el = torch.asin(torch.clamp(up / rho, -1.0, 1.0))
    rho_dot = (rel * rel_v).sum(dim=-1) / rho
    return torch.stack([rho, az, el, rho_dot], dim=-1)


# --- Convenience container -------------------------------------------------


class SPOTODTorchDynamics:
    """Holds station geometry tensors and dynamics parameters and exposes
    batched differentiable ``f`` (propagate one step) and ``h`` (all-station
    line-of-sight) on SCALED state coordinates.

    Scaled coordinates: position / ``state_scale[:3]`` (1e7 m), velocity /
    ``state_scale[3:]`` (1e4 m/s). Measurements are returned in PHYSICAL units;
    the filter layer applies its own measurement scaling/normalisation.
    """

    def __init__(
        self,
        stations: tuple[StationGeometry, ...],
        *,
        dt_s: float,
        ballistic_coeff_m2_per_kg: float,
        drag_rho_ref: float,
        drag_h_ref_m: float,
        drag_scale_height_m: float,
        state_scale: np.ndarray,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.stations = tuple(stations)
        self.n_stations = len(stations)
        self.dt = float(dt_s)
        self.ballistic = float(ballistic_coeff_m2_per_kg)
        self.rho_ref = float(drag_rho_ref)
        self.h_ref = float(drag_h_ref_m)
        self.scale_h = float(drag_scale_height_m)
        self.device = torch.device(device)
        self.dtype = dtype
        self.state_scale = torch.as_tensor(
            np.asarray(state_scale, dtype=np.float64), dtype=dtype, device=self.device
        )
        st_ecef = np.stack([station_to_ecef(s) for s in stations], axis=0)
        enu = np.stack(
            [ecef_to_enu_matrix(s.lat_rad, s.lon_rad) for s in stations], axis=0
        )
        self.station_ecef = torch.as_tensor(st_ecef, dtype=dtype, device=self.device)
        self.enu_matrix = torch.as_tensor(enu, dtype=dtype, device=self.device)
        self.min_elev_rad = torch.as_tensor(
            np.array([s.min_elevation_rad for s in stations], dtype=np.float64),
            dtype=dtype,
            device=self.device,
        )

    def to(self, device: torch.device | str) -> "SPOTODTorchDynamics":
        self.device = torch.device(device)
        self.state_scale = self.state_scale.to(self.device)
        self.station_ecef = self.station_ecef.to(self.device)
        self.enu_matrix = self.enu_matrix.to(self.device)
        self.min_elev_rad = self.min_elev_rad.to(self.device)
        return self

    def f_scaled(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """Propagate one ``dt`` step. ``x_scaled`` shape ``(B, 6)`` -> ``(B, 6)``.

        The scaled input is clamped to a generous band (``[-4, 4]``; a valid LEO
        state is ~0.7 in scaled position) before propagation so a transient bad
        gain cannot push the propagator into overflow. Valid orbits are
        unaffected (no-op clamp), preserving the numpy-reference equivalence.
        """
        x_scaled = torch.clamp(x_scaled, min=-4.0, max=4.0)
        x_phys = x_scaled * self.state_scale
        x_next = rk4_step_torch(
            x_phys,
            self.dt,
            ballistic_coeff_m2_per_kg=self.ballistic,
            drag_rho_ref=self.rho_ref,
            drag_h_ref_m=self.h_ref,
            drag_scale_height_m=self.scale_h,
        )
        return x_next / self.state_scale

    def h_scaled_all_stations(
        self, x_scaled: torch.Tensor, t_s: float
    ) -> torch.Tensor:
        """All-station line-of-sight in PHYSICAL units.

        ``x_scaled`` shape ``(B, 6)`` -> ``(B, S, 4)`` (range m, az rad, el rad,
        range-rate m/s). No visibility masking here; the caller applies the
        mask and measurement scaling.
        """
        x_phys = x_scaled * self.state_scale  # (B, 6)
        b = x_phys.shape[0]
        # Expand to (B, S, 6)
        x_bs = x_phys.unsqueeze(1).expand(b, self.n_stations, 6)
        meas = los_measurement_torch(
            x_bs,
            float(t_s),
            self.station_ecef.view(1, self.n_stations, 3),
            self.enu_matrix.view(1, self.n_stations, 3, 3),
        )  # (B, S, 4) with per-station station_ecef/enu broadcast
        return meas
