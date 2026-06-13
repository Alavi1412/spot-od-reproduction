#!/usr/bin/env python
"""Derive an astrodynamics-grounded practical-significance floor (Loop 47).

A reviewer flagged that the 3% relative practical-significance floor used in
the prior force-mismatch slices is procedurally predeclared but not
astrodynamically derived. This script derives an absolute (metres) lower
bound on the position-RMSE improvement that can be plausibly distinguished
from a sparse-visibility, single-arc line-of-sight observation pipeline,
given the manuscript's measurement-noise model and station geometry.

The derivation has four pieces, all from quantities pinned in
``configs/experiment.yaml`` rather than from any held-out test outcome:

1. **Per-update measurement-noise-limited position CRLB.** For one
   line-of-sight observation with independent range and bearing noise, the
   Cramer-Rao lower bound on the per-update position sigma in the
   line-of-sight frame is

       sigma_pos_per_update = sqrt(sigma_r^2 + (R*sigma_az)^2 + (R*sigma_el)^2),

   where ``R`` is the slant range. The representative LEO slant range is
   derived from the configuration's orbit-altitude band and station minimum
   elevation by the geometric closed form
       R = -R_e*sin(el) + sqrt(R_e^2*sin^2(el) + 2*R_e*h + h^2).

2. **Per-station geometric visibility fraction.** For a uniformly distributed
   satellite over its orbit at the mean configured altitude, the fraction of
   orbit time visible to a single ground station above the configured
   minimum-elevation cap is the spherical-cap area ratio
       f_single = (1 - cos alpha_max) / 2,
   where ``alpha_max`` is the geometric horizon half-angle bounded by the
   minimum-elevation cap. With ``N_stations`` well-separated stations the
   network fraction of orbit time visible to at least one station is taken
   as the upper-bound union estimate ``f_network = min(1, N_stations *
   f_single)`` (a conservative no-overlap assumption that does not
   double-count overlapping stations).

3. **Per-arc independent observation count.** The expected count of
   independent line-of-sight observations across an arc of length
   ``arc_steps`` at step ``dt`` is

       N_obs_per_arc = arc_steps * f_network,

   assuming one independent observation per visible step (a conservative
   lower bound: in practice multiple stations may be simultaneously visible,
   producing more updates).

4. **Per-arc Cramer-Rao floor.** Standard linearization (Tapley, Schutz, and
   Born 2004, Section 4.4) gives the arc-accumulated position-RMSE lower
   bound

       sigma_pos_per_arc = sigma_pos_per_update / sqrt(N_obs_per_arc).

   The astrodynamics-grounded practical-significance floor is then this
   absolute CRLB on the arc-accumulated 3D position uncertainty achievable
   by any measurement-noise-limited estimator under this station geometry
   and arc length. A position-RMSE improvement smaller than this floor is
   below the measurement-noise CRLB on the arc and cannot be distinguished
   from a measurement-noise fluctuation of the same arc.

The output is a JSON artefact that is read by the long-arc held-out test
harness so the practical-significance threshold enters the test as a
*derived absolute floor* in metres, not as an asserted percentage. The
derivation depends only on the configuration; no held-out test outcome
enters it.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gnn_state_estimation.constants import R_EARTH
from gnn_state_estimation.utils.io import load_yaml


def _representative_slant_range_m(
    altitude_min_km: float,
    altitude_max_km: float,
    min_elevation_deg: float,
) -> float:
    """Geometric slant range from a ground station to a target at altitude
    ``h`` whose elevation angle at the station equals ``min_elevation_deg``.

    Using the spherical-Earth geometric solution
    ``R^2 = (R_e + h)^2 - R_e^2 * cos^2(el)`` and subtracting the in-plane
    component along the line-of-sight gives

        R = -R_e * sin(el) + sqrt(R_e^2 * sin^2(el) + 2*R_e*h + h^2).

    We use the mean of the altitude band to define a single representative
    slant range; the same value is used downstream for the held-out test floor.
    """
    h = 1e3 * 0.5 * (altitude_min_km + altitude_max_km)
    el = np.deg2rad(min_elevation_deg)
    sin_el = np.sin(el)
    return float(
        -R_EARTH * sin_el
        + np.sqrt(R_EARTH**2 * sin_el**2 + 2.0 * R_EARTH * h + h**2)
    )


def _single_station_visibility_fraction(
    altitude_min_km: float,
    altitude_max_km: float,
    min_elevation_deg: float,
) -> float:
    """Spherical-cap visibility fraction for one station above the
    minimum-elevation cap, for a satellite uniformly distributed over its
    orbit at the mean configured altitude.

    The geometric horizon half-angle subtended at the satellite when its
    sub-satellite point sits at the station horizon at elevation ``el`` is
    bounded by

        alpha_max = arccos(R_e * cos(el) / (R_e + h)) - el.

    The visible solid-angle fraction is (1 - cos alpha_max) / 2.
    """
    h = 1e3 * 0.5 * (altitude_min_km + altitude_max_km)
    el = np.deg2rad(min_elevation_deg)
    arg = R_EARTH * np.cos(el) / (R_EARTH + h)
    arg = float(np.clip(arg, -1.0, 1.0))
    alpha_max = float(np.arccos(arg)) - float(el)
    if alpha_max <= 0.0:
        return 0.0
    return float((1.0 - np.cos(alpha_max)) / 2.0)


def derive_floor(
    cfg_path: Path,
    arc_steps: int,
    arc_dt_s: float,
) -> dict[str, Any]:
    cfg = load_yaml(cfg_path)
    sim = cfg["simulation"]
    orb = sim["orbit_sampling"]
    mn = sim["measurement_noise"]
    stns = sim["stations"]

    min_el_deg = float(np.median([float(s["min_elevation_deg"]) for s in stns]))
    n_stations = int(len(stns))
    slant_m = _representative_slant_range_m(
        altitude_min_km=float(orb["altitude_min_km"]),
        altitude_max_km=float(orb["altitude_max_km"]),
        min_elevation_deg=min_el_deg,
    )

    sigma_r_m = float(mn["range_std_m"])
    sigma_az_rad = np.deg2rad(float(mn["az_std_deg"]))
    sigma_el_rad = np.deg2rad(float(mn["el_std_deg"]))

    az_cross_m = slant_m * sigma_az_rad
    el_cross_m = slant_m * sigma_el_rad
    sigma_pos_per_update_m = float(
        np.sqrt(sigma_r_m**2 + az_cross_m**2 + el_cross_m**2)
    )

    f_single = _single_station_visibility_fraction(
        altitude_min_km=float(orb["altitude_min_km"]),
        altitude_max_km=float(orb["altitude_max_km"]),
        min_elevation_deg=min_el_deg,
    )
    f_network = float(min(1.0, n_stations * f_single))

    n_obs_per_arc = float(max(1.0, arc_steps * f_network))
    sigma_pos_per_arc_m = float(sigma_pos_per_update_m / np.sqrt(n_obs_per_arc))

    floor_m = float(sigma_pos_per_arc_m)

    return {
        "schema_version": "astrodynamics_floor_v2",
        "predeclared_on_utc": "2026-05-20",
        "scope": (
            "Astrodynamics-grounded absolute practical-significance floor in "
            "metres for the long-arc higher-fidelity force-mismatch slice "
            "(Loop 47). Derived solely from quantities pinned in the "
            "configuration file: measurement-noise standard deviations, "
            "orbit-altitude band, station minimum-elevation cap, station "
            "count, and the predeclared arc length. No held-out test outcome "
            "enters the derivation."
        ),
        "config_path": str(cfg_path),
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        "inputs": {
            "range_std_m": sigma_r_m,
            "az_std_deg": float(mn["az_std_deg"]),
            "el_std_deg": float(mn["el_std_deg"]),
            "altitude_min_km": float(orb["altitude_min_km"]),
            "altitude_max_km": float(orb["altitude_max_km"]),
            "median_min_elevation_deg": min_el_deg,
            "n_stations": n_stations,
            "arc_steps": int(arc_steps),
            "arc_dt_s": float(arc_dt_s),
            "arc_length_s": float(arc_steps * arc_dt_s),
            "representative_slant_range_m": slant_m,
        },
        "derivation": {
            "step_1_per_update_position_sigma_m": sigma_pos_per_update_m,
            "step_2_single_station_visibility_fraction": f_single,
            "step_3_network_visibility_fraction_upper_bound": f_network,
            "step_4_expected_independent_observations_per_arc": n_obs_per_arc,
            "step_5_per_arc_position_sigma_m": sigma_pos_per_arc_m,
            "formula_step_1": "sigma_pos = sqrt(sigma_r^2 + (R*sigma_az)^2 + (R*sigma_el)^2)",
            "formula_step_2": "f_single = (1 - cos(arccos(R_e*cos(el)/(R_e+h)) - el)) / 2",
            "formula_step_3": "f_network = min(1, N_stations * f_single)",
            "formula_step_4": "N_obs_per_arc = arc_steps * f_network",
            "formula_step_5": "sigma_pos_per_arc = sigma_pos_per_update / sqrt(N_obs_per_arc)",
        },
        "practical_significance_floor_m_absolute": floor_m,
        "interpretation": (
            "The floor is the measurement-noise-limited Cramer-Rao lower "
            "bound on the arc-accumulated 3D position-RMSE achievable by any "
            "linearised estimator under the configured station geometry and "
            "the predeclared arc length. A position-RMSE improvement smaller "
            "than this absolute floor is below the arc-accumulated CRLB and "
            "cannot be distinguished from a measurement-noise fluctuation of "
            "the same arc. The floor depends only on pinned configuration "
            "quantities and the predeclared arc length, so it is auditable "
            "independently of any held-out test outcome."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--arc-steps", type=int, default=540)
    p.add_argument("--arc-dt-s", type=float, default=20.0)
    p.add_argument(
        "--output",
        default="release/predeclarations/astrodynamics_floor_loop47.json",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    payload = derive_floor(
        Path(args.config),
        arc_steps=int(args.arc_steps),
        arc_dt_s=float(args.arc_dt_s),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
