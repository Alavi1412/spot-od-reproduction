"""Fast smoke + schema tests for the multi-revolution SGP4-truth benchmark.

These keep the unit suite fast by running a single target over a deliberately
short arc; the full multi-revolution benchmark is exercised by the standalone
script and its archived result JSON, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml
from sgp4.api import Satrec

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_multi_rev_sgp4_benchmark",
    _ROOT / "scripts" / "run_multi_rev_sgp4_benchmark.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
# Register before exec so dataclass introspection can resolve the module.
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


def _cfg() -> dict:
    return yaml.safe_load(
        (_ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8")
    )


def _first_leo_sat(catalog) -> Satrec:
    leo = mod.filter_tle_catalog(
        catalog,
        min_altitude_km=200.0,
        max_altitude_km=2000.0,
        max_eccentricity=0.05,
        min_mean_motion_rev_per_day=11.0,
    )
    assert leo, "expected at least one multi-revolution LEO target"
    return Satrec.twoline2rv(leo[0].line1, leo[0].line2)


def test_run_target_short_arc_schema_and_finiteness() -> None:
    cfg = _cfg()
    mod._STATIONS = mod._stations_from_cfg(cfg)
    model = mod._compact_model_from_cfg(cfg)
    catalog = mod.load_tle_catalog(_ROOT / "configs" / "archived_tles.json")
    sat = _first_leo_sat(catalog)

    block = mod.run_target(
        "SMOKE", sat, "deadbeef", cfg, model,
        arc_hours=0.4, dt_s=60.0, train_frac=0.6, max_nfev=6, seed=7,
    )

    assert block["steps"] == 24
    assert block["num_train_steps"] + block["num_held_out_steps"] == 24
    for name, kind in mod.ESTIMATOR_KINDS.items():
        assert name in block["estimators"], name
        m = block["estimators"][name]
        # Held-out RMSE must be a real number (finite or +inf for a blow-up),
        # never NaN, for every estimator on a populated arc.
        assert not np.isnan(m["held_out_pos_rmse_m"]), name
        assert "_held_err" in block and name in block["_held_err"]
    # The SGP4 truth must span a real LEO regime.
    assert 150.0 < block["altitude_km"] < 2000.0
    assert block["mean_motion_rev_per_day"] > 11.0


def test_compact_model_matches_config() -> None:
    cfg = _cfg()
    model = mod._compact_model_from_cfg(cfg)
    dyn = cfg["simulation"]["dynamics"]
    assert model.ballistic_coeff_m2_per_kg == float(dyn["ballistic_coeff_m2_per_kg"])
    assert model.enable_third_body == bool(dyn["enable_third_body"])
    assert model.srp_cr == float(dyn["srp_cr"])


def test_pos_rmse_masking() -> None:
    truth = np.zeros((4, 6))
    est = np.zeros((4, 6))
    est[:, 0] = [0.0, 3.0, 0.0, 4.0]
    mask = np.array([False, True, False, True])
    # sqrt(mean([9, 16])) = sqrt(12.5)
    assert abs(mod._pos_rmse(truth, est, mask) - np.sqrt(12.5)) < 1e-9
    assert np.isnan(mod._pos_rmse(truth, est, np.zeros(4, dtype=bool)))


def test_table_builder_handles_missing_and_real_result() -> None:
    from scripts.build_paper_assets import build_multi_rev_sgp4_benchmark_table

    missing = build_multi_rev_sgp4_benchmark_table(
        Path("results/does_not_exist_multi_rev.json")
    )
    assert missing.startswith("%")

    real = Path("results/multi_rev_sgp4/multi_rev_sgp4_benchmark.json")
    if real.exists():
        tex = build_multi_rev_sgp4_benchmark_table(real)
        assert "\\label{tab:multi_rev_sgp4_benchmark}" in tex
        # Paper-facing wording must call SGP4 a *reference*, not "truth"
        # (reviewer M1 / loop-23 A1); the old "SGP4 truth" framing is removed.
        assert "SGP4-reference" in tex or "SGP4 reference" in tex
        assert "SGP4 truth" not in tex and "SGP4-truth" not in tex
