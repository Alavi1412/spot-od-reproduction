import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnn_state_estimation.coordinates import (
    StationGeometry,
    eci_to_ecef,
    line_of_sight_measurement,
    station_to_ecef,
)


class MeasurementModelTests(unittest.TestCase):
    def test_range_rate_is_consistent_in_ecef_frame(self) -> None:
        station = StationGeometry(
            name="TestStation",
            lat_deg=35.0,
            lon_deg=-120.0,
            alt_m=100.0,
            min_elevation_deg=-90.0,  # force visibility for consistency test
        )
        state = np.array(
            [7_000_000.0, 1_100_000.0, 1_500_000.0, -900.0, 7_300.0, 800.0],
            dtype=np.float64,
        )
        t_s = 1234.5

        z, _ = line_of_sight_measurement(state, station, t_s)
        r_ecef, v_ecef = eci_to_ecef(state[:3], state[3:], t_s)
        rel = r_ecef - station_to_ecef(station)
        rho = float(np.linalg.norm(rel))
        expected_rho_dot = float(np.dot(rel, v_ecef) / rho)

        self.assertAlmostEqual(float(z[3]), expected_rho_dot, places=9)


if __name__ == "__main__":
    unittest.main()
