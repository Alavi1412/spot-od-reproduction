import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_public_tracking_snapshots import (  # noqa: E402
    extract_observation_listing_rows,
    parse_observation_detail_html,
    select_observation_records,
)


LISTING_HTML = """
<tr class="clickable-row" data-href="/observations/13829020/">
  <td class="text-nowrap">
    <a href="/observations/13829020/" class="obs-link">
      <span class="badge badge-good">13829020</span>
    </a>
  </td>
</tr>
<tr class="clickable-row" data-href="/observations/13823759/">
  <td class="text-nowrap">
    <a href="/observations/13823759/" class="obs-link">
      <span class="badge badge-unknown">13823759</span>
    </a>
  </td>
</tr>
"""


DETAIL_HTML = """
<table class="table table-sm table-borderless table-hover">
  <tr>
    <td><span class="badge badge-secondary">Satellite</span></td>
    <td>
      <a href="#" data-toggle="modal">98338  - HADES-R</a>
    </td>
  </tr>
  <tr>
    <td><span class="badge badge-secondary">Station</span></td>
    <td>
      <a href="/stations/4040/">4040 - Jim UHF</a>
    </td>
  </tr>
</table>
<tr>
  <td><span class="badge badge-secondary">Status</span></td>
  <td>
    <span id="rating-status">
      <span class="badge badge-good">Good</span>
    </span>
  </td>
</tr>
<div id="waterfall-status-badge">With Signal</div>
<svg id="polar"
     data-tle1="1 98338U 25024B   26105.63455840  .00018184  00000-0  62988-3 0  9998"
     data-tle2="2 98338  97.4418 104.7449 0015878 160.0207 200.1778 15.25615080 78085"
     data-timeframe-start="2026-04-16T05:00:01+00:00"
     data-timeframe-end="2026-04-16T05:08:31+00:00"
     data-groundstation-lat="59.3293"
     data-groundstation-lon="18.0686"
     data-groundstation-alt="35"></svg>
"""


class PublicTrackingFetchTests(unittest.TestCase):
    def test_extract_observation_listing_rows(self) -> None:
        rows = extract_observation_listing_rows(LISTING_HTML)
        self.assertEqual(rows, [(13829020, "good"), (13823759, "unknown")])

    def test_parse_observation_detail_html(self) -> None:
        station_lookup = {
            4040: {"id": 4040, "name": "Jim UHF", "lat": 59.3293, "lng": 18.0686, "altitude": 35, "min_horizon": 12}
        }
        record = parse_observation_detail_html(
            DETAIL_HTML,
            observation_id=13829020,
            status_badge="good",
            station_lookup=station_lookup,
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["norad_cat_id"], 98338)
        self.assertEqual(record["station_name"], "Jim UHF")
        self.assertEqual(record["status"], "good")
        self.assertEqual(record["vetted_status"], "with signal")
        self.assertEqual(record["min_horizon"], 12.0)

    def test_select_observation_records_honors_per_satellite_cap(self) -> None:
        records = [
            {"id": 1, "norad_cat_id": 10},
            {"id": 2, "norad_cat_id": 10},
            {"id": 3, "norad_cat_id": 10},
            {"id": 4, "norad_cat_id": 11},
        ]
        selected = select_observation_records(records, max_records=4, max_per_satellite=2)
        self.assertEqual([row["id"] for row in selected], [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
