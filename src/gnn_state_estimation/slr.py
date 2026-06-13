"""Parsing and geometry helpers for ILRS satellite-laser-ranging (SLR) data.

This module supports a *bounded pilot* real-measurement validation: it parses
public Consolidated Laser Ranging Data (CRD) version-2 normal-point files and
provides an approximate Earth-rotation (GMST) transform so that station
coordinates can be expressed in a pseudo-inertial frame consistent with an
SGP4/TEME prior.  It deliberately does not implement precise SLR reduction
(no relativistic, tropospheric, centre-of-mass, or polar-motion corrections);
it is sufficient for a range-only sanity-check fit and is documented as
approximate everywhere it is used.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np

from .coordinates import geodetic_to_ecef, rot_z

# Vacuum speed of light (CODATA / SI exact), metres per second.
SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True)
class SlrStation:
    """Official ILRS station description used for the supported pilot subset."""

    code: str
    cdp_id: int
    lat_deg: float
    lon_deg: float
    alt_m: float

    def ecef_m(self) -> np.ndarray:
        return geodetic_to_ecef(
            np.deg2rad(self.lat_deg), np.deg2rad(self.lon_deg), self.alt_m
        )


# Coordinates taken from the official ILRS station pages for the four stations
# that appear in the pilot LAGEOS-2 daily CRD file.  Keyed by CDP pad id (the
# 4-digit identifier carried in the CRD ``H2`` record).
SUPPORTED_STATIONS: dict[int, SlrStation] = {
    7090: SlrStation("YARL", 7090, -29.0464, 115.3467, 244.0),
    7941: SlrStation("MATM", 7941, 40.6486, 16.7046, 536.9),
    8834: SlrStation("WETL", 8834, 49.1444, 12.8780, 665.0),
    7840: SlrStation("HERL", 7840, 50.8674, 0.3361, 75.0),
}


@dataclass(frozen=True)
class NormalPoint:
    """A single CRD version-2 normal-point range observation."""

    station_code: str
    cdp_id: int
    epoch_unix: float
    epoch_iso: str
    seconds_of_day: float
    tof_two_way_s: float
    range_m: float
    raw_count: int | None
    np_window_s: float | None
    bin_rms_ps: float | None


def one_way_range_m(tof_two_way_s: float) -> float:
    """Convert a two-way time of flight (seconds) to a one-way range (metres)."""
    return float(tof_two_way_s) * SPEED_OF_LIGHT_MPS / 2.0


def _to_float(token: str) -> float | None:
    try:
        value = float(token)
    except (TypeError, ValueError):
        return None
    return value


def _to_int(token: str) -> int | None:
    value = _to_float(token)
    if value is None:
        return None
    return int(round(value))


def parse_crd_v2_normal_points(text: str) -> list[NormalPoint]:
    """Parse CRD v2 normal-point (``11``) records block-wise.

    Station code / CDP id come from the most recent ``H2`` record and the pass
    UTC date from the most recent ``H4`` record; the per-record seconds-of-day
    field is converted to a UTC timestamp with a midnight roll-over guard.
    Only records from :data:`SUPPORTED_STATIONS` are returned.
    """
    points: list[NormalPoint] = []
    cur_code: str | None = None
    cur_cdp: int | None = None
    h4_base_midnight_unix: float | None = None
    h4_start_sod: float | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0].lower()

        if tag == "h2" and len(parts) >= 3:
            cur_code = parts[1]
            cur_cdp = _to_int(parts[2])
        elif tag == "h4" and len(parts) >= 8:
            # h4 <data type> <yr mon day hr min sec> (start) <yr mon day ...> (end)
            year, month, day = _to_int(parts[2]), _to_int(parts[3]), _to_int(parts[4])
            hour, minute, sec = _to_int(parts[5]), _to_int(parts[6]), _to_int(parts[7])
            if None in (year, month, day, hour, minute, sec):
                h4_base_midnight_unix = None
                h4_start_sod = None
                continue
            midnight = _dt.datetime(
                year, month, day, tzinfo=_dt.timezone.utc
            )
            h4_base_midnight_unix = midnight.timestamp()
            h4_start_sod = hour * 3600.0 + minute * 60.0 + sec
        elif tag in ("h8", "h9"):
            cur_code = None
            cur_cdp = None
            h4_base_midnight_unix = None
            h4_start_sod = None
        elif tag == "11" and len(parts) >= 3:
            if (
                cur_cdp is None
                or h4_base_midnight_unix is None
                or cur_cdp not in SUPPORTED_STATIONS
            ):
                continue
            sod = _to_float(parts[1])
            tof = _to_float(parts[2])
            if sod is None or tof is None or tof <= 0.0:
                continue
            day_offset = 0.0
            if h4_start_sod is not None and sod + 1e-6 < h4_start_sod:
                day_offset = 86400.0  # pass crossed UTC midnight
            epoch_unix = h4_base_midnight_unix + sod + day_offset
            station = SUPPORTED_STATIONS[cur_cdp]
            np_window = _to_float(parts[5]) if len(parts) > 5 else None
            raw_count = _to_int(parts[6]) if len(parts) > 6 else None
            bin_rms = _to_float(parts[7]) if len(parts) > 7 else None
            points.append(
                NormalPoint(
                    station_code=station.code,
                    cdp_id=station.cdp_id,
                    epoch_unix=epoch_unix,
                    epoch_iso=_dt.datetime.fromtimestamp(
                        epoch_unix, tz=_dt.timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                    seconds_of_day=sod,
                    tof_two_way_s=tof,
                    range_m=one_way_range_m(tof),
                    raw_count=raw_count,
                    np_window_s=np_window,
                    bin_rms_ps=bin_rms,
                )
            )
    points.sort(key=lambda p: p.epoch_unix)
    return points


def julian_date_from_unix(epoch_unix: float) -> float:
    """Julian Date (UTC ~ UT1 approximation) for a POSIX timestamp."""
    return epoch_unix / 86400.0 + 2440587.5


def gmst_rad(epoch_unix: float) -> float:
    """Greenwich Mean Sidereal Time (rad), IAU-1982 series (Vallado eq. 3-47).

    UT1 is approximated by UTC; for a one-day pilot fit the sub-second UT1-UTC
    offset is negligible relative to the model-error budget and is documented
    as an approximation.
    """
    jd = julian_date_from_unix(epoch_unix)
    t = (jd - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    gmst_deg = (gmst_sec % 86400.0) / 240.0
    return float(np.deg2rad(gmst_deg % 360.0))


def station_pseudo_inertial_m(station: SlrStation, epoch_unix: float) -> np.ndarray:
    """Station ECEF position rotated into the SGP4/TEME-ish inertial frame.

    Uses a single GMST z-rotation (no polar motion / nutation); this is an
    approximate transform appropriate only for the bounded range-only pilot.
    """
    theta = gmst_rad(epoch_unix)
    # rot_z(theta) maps inertial -> Earth-fixed, so its transpose maps back.
    return rot_z(theta).T @ station.ecef_m()


def summarize_residuals(residuals_m: np.ndarray) -> dict[str, float]:
    """Robust residual summary used for both the prior and fitted orbits."""
    arr = np.asarray(residuals_m, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "rms_m": float("nan"),
            "mae_m": float("nan"),
            "median_abs_m": float("nan"),
            "p95_abs_m": float("nan"),
        }
    abs_arr = np.abs(arr)
    return {
        "count": int(arr.size),
        "rms_m": float(np.sqrt(np.mean(arr**2))),
        "mae_m": float(np.mean(abs_arr)),
        "median_abs_m": float(np.median(abs_arr)),
        "p95_abs_m": float(np.percentile(abs_arr, 95.0)),
    }


# ---------------------------------------------------------------------------
# Standard precise-SLR-reduction corrections (additive).  The committed parser
# and the GMST-only geometry above are left byte-for-byte unchanged so the
# committed bounded slice keeps zero regression risk.  These implement the
# tropospheric, satellite centre-of-mass, and relativistic (Shapiro) range
# corrections the loop-30 review listed as missing; the operationally-corrected
# real-data sanity probe applies them and the correction-sensitivity audit
# reports the size of each term.
# ---------------------------------------------------------------------------

# Newtonian gravitational parameter of the Earth (m^3 s^-2) and the Shapiro
# constant 2*GM/c^2 (m), used for the one-way relativistic range delay.
_GM_EARTH = 3.986004418e14
_SHAPIRO_2GM_OVER_C2 = 2.0 * _GM_EARTH / (SPEED_OF_LIGHT_MPS**2)

# Nominal LAGEOS-1/-2 satellite centre-of-mass range correction (m): the laser
# pulse reflects off the corner-cube array, so the geometric range to the
# centre of mass is the measured range minus this offset.  The 0.251 m nominal
# value is the long-standing ILRS LAGEOS constant (Otsubo & Appleby, 2003).
LAGEOS_CENTRE_OF_MASS_OFFSET_M = 0.251


@dataclass(frozen=True)
class MetRecord:
    """A CRD v2 ``20`` surface-meteorology record."""

    epoch_unix: float
    pressure_hpa: float
    temperature_k: float
    humidity_pct: float


def parse_crd_v2_meteorology(text: str) -> list[MetRecord]:
    """Parse CRD v2 ``20`` meteorology records (pressure/temperature/humidity).

    Time stamping mirrors :func:`parse_crd_v2_normal_points` (``H4`` UTC date
    plus seconds-of-day with a midnight roll-over guard) so a met record can be
    matched to a normal point by nearest epoch.
    """
    recs: list[MetRecord] = []
    base_midnight: float | None = None
    start_sod: float | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        tag = parts[0].lower()
        if tag == "h4" and len(parts) >= 8:
            y, mo, d = _to_int(parts[2]), _to_int(parts[3]), _to_int(parts[4])
            hh, mm, ss = _to_int(parts[5]), _to_int(parts[6]), _to_int(parts[7])
            if None in (y, mo, d, hh, mm, ss):
                base_midnight = None
                start_sod = None
                continue
            base_midnight = _dt.datetime(
                y, mo, d, tzinfo=_dt.timezone.utc
            ).timestamp()
            start_sod = hh * 3600.0 + mm * 60.0 + ss
        elif tag in ("h8", "h9"):
            base_midnight = None
            start_sod = None
        elif tag == "20" and len(parts) >= 5 and base_midnight is not None:
            sod = _to_float(parts[1])
            pres = _to_float(parts[2])
            temp = _to_float(parts[3])
            hum = _to_float(parts[4])
            if None in (sod, pres, temp, hum):
                continue
            day_offset = (
                86400.0
                if start_sod is not None and sod + 1e-6 < start_sod
                else 0.0
            )
            recs.append(
                MetRecord(
                    epoch_unix=base_midnight + sod + day_offset,
                    pressure_hpa=float(pres),
                    temperature_k=float(temp),
                    humidity_pct=float(hum),
                )
            )
    recs.sort(key=lambda r: r.epoch_unix)
    return recs


def parse_crd_v2_transmit_wavelength_nm(text: str) -> float | None:
    """Transmit laser wavelength (nm) from the CRD v2 ``c0`` system record."""
    for raw_line in text.splitlines():
        parts = raw_line.strip().split()
        if parts and parts[0].lower() == "c0" and len(parts) >= 3:
            wl = _to_float(parts[2])
            if wl is not None and wl > 0.0:
                return float(wl)
    return None


def nearest_met_record(
    recs: list[MetRecord], epoch_unix: float
) -> MetRecord | None:
    """Return the meteorology record nearest in time to ``epoch_unix``."""
    if not recs:
        return None
    return min(recs, key=lambda r: abs(r.epoch_unix - float(epoch_unix)))


def saturation_vapor_pressure_hpa(temperature_k: float) -> float:
    """Saturation water-vapour pressure (hPa) via the Magnus/Tetens formula."""
    tc = float(temperature_k) - 273.15
    return 6.1078 * 10.0 ** (7.5 * tc / (237.3 + tc))


def marini_murray_range_correction_m(
    elevation_rad: float,
    pressure_hpa: float,
    temperature_k: float,
    humidity_pct: float,
    latitude_rad: float,
    height_m: float,
    wavelength_um: float,
) -> float:
    """One-way tropospheric range delay (m), Marini & Murray (1973).

    The classical SLR tropospheric model (NASA X-591-73-351).  At this slice's
    bounded hundreds-of-metres fidelity the choice between Marini--Murray and
    the IERS-2010 Mendes--Pavlis model differs only at the centimetre
    elevation-dependent level, far below the Earth-orientation term the
    correction-sensitivity audit isolates as binding; Marini--Murray is used
    for its compact, transcription-robust closed form.
    """
    el = max(float(elevation_rad), np.deg2rad(1.0))
    p0 = float(pressure_hpa)
    t0 = float(temperature_k)
    phi = float(latitude_rad)
    h_km = float(height_m) / 1000.0
    e_s = saturation_vapor_pressure_hpa(t0)
    e0 = max(0.0, min(float(humidity_pct), 100.0)) / 100.0 * e_s
    # Laser-frequency (wavelength) dispersion factor.
    lam = float(wavelength_um)
    f_lambda = 0.9650 + 0.0164 / lam**2 + 0.000228 / lam**4
    # Site function of latitude and height.
    f_site = 1.0 - 0.0026 * np.cos(2.0 * phi) - 0.00031 * h_km
    a = 0.002357 * p0 + 0.000141 * e0
    k = (
        1.163
        - 0.00968 * np.cos(2.0 * phi)
        - 0.00104 * t0
        + 0.00001435 * p0
    )
    b = (1.084e-8 * p0 * t0 * k) + (
        4.734e-8 * p0 * p0 * (2.0 / (t0 * (3.0 - 1.0 / k)))
    )
    sin_e = np.sin(el)
    denom = sin_e + b / ((a + b) * (sin_e + 0.01))
    return float((f_lambda / f_site) * (a + b) / denom)


def shapiro_delay_m(
    station_eci_m: np.ndarray, satellite_eci_m: np.ndarray
) -> float:
    """One-way gravitational (Shapiro) range delay (m).

    ``dr = (2GM/c^2) ln[(r1+r2+rho)/(r1+r2-rho)]`` with ``r1``/``r2`` the
    geocentric station and satellite radii and ``rho`` the station--satellite
    distance.  At the LAGEOS radius this is a few millimetres; it is included
    so the correction-sensitivity audit can show it is negligible here.
    """
    r1 = float(np.linalg.norm(station_eci_m))
    r2 = float(np.linalg.norm(satellite_eci_m))
    rho = float(np.linalg.norm(np.asarray(satellite_eci_m) - np.asarray(station_eci_m)))
    num = r1 + r2 + rho
    den = r1 + r2 - rho
    if den <= 0.0 or num <= 0.0:
        return 0.0
    return _SHAPIRO_2GM_OVER_C2 * float(np.log(num / den))
