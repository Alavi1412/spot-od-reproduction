"""IERS Earth-orientation parameters (polar motion and UT1-UTC).

This module is *additive*.  It parses the public IERS ``finals2000A.all`` Earth
-orientation series (CSV distribution) and exposes a small, dependency-free
linear interpolator for polar motion ``(x_p, y_p)`` and the ``UT1-UTC`` offset
at an arbitrary UTC epoch.  These are exactly the Earth-orientation parameters
deliberately omitted from the bounded GMST-only slice and from the analytic
IAU-76/80 transform in :mod:`gnn_state_estimation.frames`; ingesting the real
public series lets the operationally-corrected real-data sanity probe close the
``polar motion`` and ``UT1-UTC`` items of the missing precise-SLR-reduction
list, and lets the correction-sensitivity audit quantify their magnitude.

The series is a small public ASCII product and is archived next to the slice
inputs on first fetch, so the slice regenerates offline thereafter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# arcseconds -> radians
_ARCSEC = np.pi / (180.0 * 3600.0)

# Modified Julian Date of the POSIX epoch (1970-01-01): MJD = JD - 2400000.5,
# JD(unix) = unix/86400 + 2440587.5  ->  MJD(unix) = unix/86400 + 40587.0.
_MJD_UNIX_OFFSET = 40587.0


def mjd_utc_from_unix(epoch_unix: float) -> float:
    """Modified Julian Date (UTC) for a POSIX timestamp."""
    return float(epoch_unix) / 86400.0 + _MJD_UNIX_OFFSET


@dataclass(frozen=True)
class EopSeries:
    """Tabulated IERS polar motion / UT1-UTC, linearly interpolated.

    ``mjd`` is strictly increasing; ``xp_arcsec`` / ``yp_arcsec`` are polar
    motion in arcseconds and ``ut1_utc_s`` is the UT1-UTC offset in seconds.
    Queries outside the tabulated range clamp to the nearest endpoint (the
    LAGEOS arcs used here lie well inside the IERS final-data span, so no
    extrapolation occurs in practice; clamping only guards degenerate calls).
    """

    mjd: np.ndarray
    xp_arcsec: np.ndarray
    yp_arcsec: np.ndarray
    ut1_utc_s: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.mjd.size < 2:
            raise ValueError("EOP series needs at least two tabulated rows")

    def _interp(self, values: np.ndarray, mjd: float) -> float:
        return float(np.interp(mjd, self.mjd, values))

    def polar_motion_rad(self, epoch_unix: float) -> tuple[float, float]:
        """Return ``(x_p, y_p)`` in radians at the given UTC epoch."""
        m = mjd_utc_from_unix(epoch_unix)
        return (
            self._interp(self.xp_arcsec, m) * _ARCSEC,
            self._interp(self.yp_arcsec, m) * _ARCSEC,
        )

    def ut1_minus_utc_s(self, epoch_unix: float) -> float:
        """Return the ``UT1-UTC`` offset in seconds at the given UTC epoch."""
        return self._interp(self.ut1_utc_s, mjd_utc_from_unix(epoch_unix))

    def covers(self, epoch_unix: float) -> bool:
        m = mjd_utc_from_unix(epoch_unix)
        return float(self.mjd[0]) <= m <= float(self.mjd[-1])


def parse_finals2000a_csv(text: str) -> EopSeries:
    """Parse the IERS ``finals2000A.all.csv`` distribution.

    The CSV is ``;``-separated with a named header.  Rows are retained while
    polar motion and UT1-UTC are populated (the IERS file pads predicted /
    empty tails with blank fields); the ``final``/``prediction`` flag itself is
    not needed because the LAGEOS window used here falls in the final-data
    span.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty EOP CSV")
    header = [h.strip() for h in lines[0].split(";")]
    idx = {name: i for i, name in enumerate(header)}
    for required in ("MJD", "x_pole", "y_pole", "UT1-UTC"):
        if required not in idx:
            raise ValueError(f"EOP CSV missing column {required!r}")
    i_mjd = idx["MJD"]
    i_xp = idx["x_pole"]
    i_yp = idx["y_pole"]
    i_ut1 = idx["UT1-UTC"]

    mjd: list[float] = []
    xp: list[float] = []
    yp: list[float] = []
    ut1: list[float] = []
    for ln in lines[1:]:
        f = ln.split(";")
        if len(f) <= i_ut1:
            continue
        try:
            m = float(f[i_mjd])
            x = float(f[i_xp])
            y = float(f[i_yp])
            u = float(f[i_ut1])
        except (ValueError, IndexError):
            continue  # blank predicted/padding tail
        mjd.append(m)
        xp.append(x)
        yp.append(y)
        ut1.append(u)
    if len(mjd) < 2:
        raise ValueError("EOP CSV produced fewer than two usable rows")
    arr_mjd = np.asarray(mjd, dtype=np.float64)
    order = np.argsort(arr_mjd, kind="stable")
    return EopSeries(
        mjd=arr_mjd[order],
        xp_arcsec=np.asarray(xp, dtype=np.float64)[order],
        yp_arcsec=np.asarray(yp, dtype=np.float64)[order],
        ut1_utc_s=np.asarray(ut1, dtype=np.float64)[order],
        source="IERS finals2000A.all (CSV distribution)",
    )


def load_eop_series(path: str | Path) -> EopSeries:
    """Load an archived IERS ``finals2000A.all.csv`` file."""
    p = Path(path)
    return parse_finals2000a_csv(p.read_text(encoding="utf-8", errors="replace"))
