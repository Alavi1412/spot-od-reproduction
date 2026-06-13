"""GNN-based satellite state estimation research package."""

from .constants import (
    ATMOSPHERE_SCALE_HEIGHT,
    ATMOSPHERE_SURFACE_DENSITY,
    EARTH_ROTATION_RATE,
    J2,
    MU_EARTH,
    R_EARTH,
)

__all__ = [
    "MU_EARTH",
    "R_EARTH",
    "J2",
    "EARTH_ROTATION_RATE",
    "ATMOSPHERE_SURFACE_DENSITY",
    "ATMOSPHERE_SCALE_HEIGHT",
]
