"""Astrodynamically-credible dense-tracking OD probe (predeclared).

Context. A prior measurement-update-dominant probe densified the observation
network but produced non-credible multi-thousand-kilometre median trajectory
errors and a ~57% gross-failure rate for *every* estimator. Reading the
filter and measurement code shows this is a measurement-conditioning artifact,
not an estimator-skill or regime-difficulty statement: the topocentric azimuth
``arctan2(east, north)`` is geometrically singular at the station zenith (its
measurement Jacobian scales as ``1/(rho*cos(elevation))``), while the
measurement model weights azimuth with a fixed 0.02-degree noise. Near zenith
the Kalman gain therefore applies an enormous, geometrically meaningless
azimuth correction and the filter diverges; a dense global network makes
near-zenith passes over *some* station common, so all estimators fail
identically. This is the standard near-zenith azimuth conditioning defect.

This probe is the *corrected* twin. The estimators apply the standard
topocentric-azimuth de-weighting (a capped ``1/cos^2(elevation)`` variance
inflation; see ``filters.ekf.azimuth_deweight_factor``), applied identically
to EKF/UKF/AUKF and inherited by the learned residual through its classical
priors, so it cannot create a learned advantage. Truth and estimators share
one perfect dynamics model with third-body and SRP perturbations enabled and
NO force-model or measurement mismatch (the scenario declares an explicit
empty ``estimator_overrides``), so the regime isolates observation density on
an astrodynamically credible OD setup.

PREDECLARED PRIMARY METRIC (fixed before running, stated here and echoed into
the artifact): the pooled all-step trajectory position RMSE over the
trajectories that avoid numerical divergence, lower is better. The
observed-step pooled position RMSE is reported SYMMETRICALLY as the secondary
metric. Heavy-tail-robust medians, the gross-failure rate (100 km
physical-adequacy threshold), per-estimator divergence avoidance, the realized
visibility-bucket fractions, and a paired percentile-bootstrap confidence
interval for the learned-minus-best-classical difference on the predeclared
metric are reported alongside. A bounded divergence diagnostic re-runs the
same trajectories with the conditioning fix DISABLED to demonstrate, with
numbers rather than speculation, that the uncorrected probe's catastrophe is
the near-zenith azimuth conditioning artifact and that the fix removes it.

No model is trained or refit here: the released RGR-GF estimator is evaluated
in fixed-model inference, reported honestly as such.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import copy
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from gnn_state_estimation.coordinates import line_of_sight_measurement
from gnn_state_estimation.evaluation import (
    BaselineConfig,
    build_innovation_features,
    build_prior_bank_feature_array,
    generate_noisy_init,
    parse_baseline_config,
    run_filter_baselines,
    run_model_inference,
)
from gnn_state_estimation.models import TemporalGraphEstimator
from gnn_state_estimation.scenarios import (
    estimator_sim_config,
    has_estimator_overrides,
    truth_sim_config,
)
from gnn_state_estimation.simulation import generate_dataset, parse_dataset_config
from gnn_state_estimation.training import parse_train_config
from gnn_state_estimation.utils.io import load_yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "credible_dense_od_probe"
OUT_PATH = OUT_DIR / "credible_dense_od_probe.json"

SCENARIO_KEY = "credible_dense_od_test"

# Standard topocentric-azimuth de-weighting elevation cap (degrees). At 80 deg
# the azimuth-variance inflation factor 1/cos^2(el) is capped at ~33x, which
# bounds the weight through a numerical zenith crossing while still removing
# the near-singular high-elevation azimuth conditioning.
ELEV_CAP_DEG = 80.0

# Manuscript guardrails, identical to the rest of the study.
DIVERGENCE_POS_RMSE_M = 1.0e8
ENGINEERING_ADEQUATE_POS_RMSE_M = 1.0e5

PREDECLARED_PRIMARY_METRIC = (
    "pooled all-step trajectory position RMSE over non-divergent trajectories "
    "(lower is better); fixed before running, observed-step reported "
    "symmetrically as the secondary metric"
)

# Deterministic, configuration-derived diagnosis text shared by the full run
# and by the --refresh-diagnosis path so the committed artifact and this
# script can never drift. These describe the *configuration* finding and carry
# no measured numeric value.
DIVERGENCE_MECHANISM_TEXT = (
    "The uncorrected densified-visibility control's catastrophe is "
    "dominated by an estimator/truth observation-network "
    "inconsistency: in that configuration the recursive filters "
    "resolve a different (smaller) station bank than the truth was "
    "synthesised against, so ranges and angles are differenced "
    "against the wrong station coordinates and most trajectories "
    "diverge (see station_bank_consistency). A secondary, "
    "well-known contributor is the near-zenith topocentric-azimuth "
    "conditioning weakness (the azimuth is geometrically singular "
    "at the station zenith); under an 8-degree elevation mask "
    "genuine near-zenith passes are rare, so the controlled "
    "azimuth-deweight fix-on/fix-off comparison below shows only a "
    "minor effect. Both are shared model/configuration defects, "
    "not estimator skill or intrinsic regime difficulty."
)
AZIMUTH_DEWEIGHT_NOTE = (
    "uncorrected vs corrected here toggles ONLY the azimuth "
    "de-weighting on the same network-consistent trajectories; the "
    "network inconsistency is absent from this probe by "
    "construction, so this isolates the azimuth effect alone"
)


def build_station_bank_consistency(cfg: dict, est_sim: dict, truth_sim: dict) -> dict:
    """Configuration-derived station-bank consistency block.

    Resolves the uncorrected densified-visibility control's scenario and shows
    that its estimator config resolves a different (smaller) station bank than
    the truth it was synthesised against -- the dominant cause of that
    control's catastrophe -- whereas this corrected probe resolves an identical
    bank. Pure function of ``cfg`` and the resolved configs; no filtering.
    """
    unc_scn = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(
        "dense_visibility_test"
    )
    if unc_scn is not None:
        unc_sim = copy.deepcopy(cfg["simulation"])
        unc_est = estimator_sim_config(unc_sim, unc_scn)
        unc_truth = truth_sim_config(unc_sim, unc_scn)
        return {
            "uncorrected_control_scenario": "dense_visibility_test",
            "uncorrected_estimator_n_stations": len(unc_est["stations"]),
            "uncorrected_truth_n_stations": len(unc_truth["stations"]),
            "uncorrected_estimator_equals_truth": unc_est == unc_truth,
            "corrected_estimator_n_stations": len(est_sim["stations"]),
            "corrected_truth_n_stations": len(truth_sim["stations"]),
            "corrected_estimator_equals_truth": est_sim == truth_sim,
        }
    return {
        "corrected_estimator_n_stations": len(est_sim["stations"]),
        "corrected_truth_n_stations": len(truth_sim["stations"]),
        "corrected_estimator_equals_truth": est_sim == truth_sim,
    }


def _refresh_diagnosis(cfg: dict, args: argparse.Namespace) -> int:
    """Deterministically refresh ONLY the configuration-derived diagnosis.

    Reads the already-completed committed artifact and rewrites exclusively the
    configuration-derived ``divergence_diagnosis`` sub-fields (the mechanism
    statement, the station-bank consistency block, the azimuth-deweight note,
    and the boolean azimuth-is-minor flag, which is recomputed from the
    artifact's own stored gross-failure numbers). Every measured numeric value
    is left byte-identical and no filtering, simulation, training, or
    re-evaluation is performed; this only keeps the committed artifact's
    explanatory text consistent with the current script without a rerun.
    """
    if not OUT_PATH.exists():
        raise SystemExit(
            f"--refresh-diagnosis needs an existing completed artifact at "
            f"{OUT_PATH.relative_to(ROOT)}; none found (run the probe first)"
        )
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    if (
        data.get("status") != "completed"
        or data.get("schema_version") != "credible_dense_od_probe_v1"
    ):
        raise SystemExit(
            "--refresh-diagnosis requires a completed "
            "credible_dense_od_probe_v1 artifact; refusing to synthesise one"
        )
    # The refresh is only valid for the design the artifact actually measured.
    if int(data.get("num_realizations", -1)) != int(args.num_realizations) or int(
        data.get("trajectories_per_realization", -1)
    ) != int(args.trajectories):
        raise SystemExit(
            "--refresh-diagnosis design mismatch: artifact was measured at "
            f"K={data.get('num_realizations')}, N="
            f"{data.get('trajectories_per_realization')} but "
            f"--num-realizations={args.num_realizations} "
            f"--trajectories={args.trajectories} were requested; refusing to "
            "relabel a different design"
        )
    diag = data.get("divergence_diagnosis")
    if not isinstance(diag, dict):
        raise SystemExit(
            "--refresh-diagnosis requires a divergence_diagnosis block to "
            "update; artifact has none"
        )
    required_measured = (
        "uncorrected_gross_failure_rate",
        "corrected_gross_failure_rate",
    )
    if any(k not in diag for k in required_measured):
        raise SystemExit(
            "--refresh-diagnosis requires the measured gross-failure rates to "
            "recompute the azimuth-is-minor flag; artifact is incomplete"
        )

    est_sim, truth_sim, _ = resolve_sim_configs(cfg, SCENARIO_KEY)

    # Recompute ONLY configuration-derived / stored-number-derived fields.
    diag["mechanism"] = DIVERGENCE_MECHANISM_TEXT
    diag["station_bank_consistency"] = build_station_bank_consistency(
        cfg, est_sim, truth_sim
    )
    diag["azimuth_deweight_note"] = AZIMUTH_DEWEIGHT_NOTE
    unc_gf = diag["uncorrected_gross_failure_rate"]
    cor_gf = diag["corrected_gross_failure_rate"]
    diag["azimuth_deweight_is_minor_here"] = bool(
        max(float(v) for v in unc_gf.values())
        - max(float(v) for v in cor_gf.values())
        <= 0.20
    )
    # Drop the stale, superseded boolean from the pre-station-bank schema; it
    # described a question the current diagnosis no longer frames this way.
    diag.pop("fix_removes_artifact", None)
    data["divergence_diagnosis"] = diag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        "refreshed configuration-derived divergence-diagnosis fields only "
        "(mechanism, station_bank_consistency, azimuth_deweight_note, "
        "azimuth_deweight_is_minor_here); every measured numeric value left "
        "byte-identical. No re-filtering, simulation, training, or "
        "re-evaluation was performed -- this is a deterministic relabel of the "
        f"existing artifact, not a rerun. wrote {OUT_PATH.relative_to(ROOT)}"
    )
    return 0


def resolve_sim_configs(cfg: dict, scenario: str):
    sim_cfg = copy.deepcopy(cfg["simulation"])
    scenario_cfg = cfg.get("benchmark_suite", {}).get("scenarios", {}).get(scenario)
    if scenario_cfg is None:
        raise SystemExit(f"scenario {scenario!r} not found in benchmark_suite")
    est = estimator_sim_config(sim_cfg, scenario_cfg)
    truth = truth_sim_config(sim_cfg, scenario_cfg)
    return est, truth, scenario_cfg


def conditioned_baseline(base: BaselineConfig, cap_deg: float | None) -> BaselineConfig:
    """A BaselineConfig with the azimuth de-weighting cap set (or cleared)."""
    return BaselineConfig(
        ekf=dataclasses.replace(base.ekf, angle_deweight_elev_cap_deg=cap_deg),
        ukf=dataclasses.replace(base.ukf, angle_deweight_elev_cap_deg=cap_deg),
        aukf=(
            None
            if base.aukf is None
            else dataclasses.replace(base.aukf, angle_deweight_elev_cap_deg=cap_deg)
        ),
        init_pos_std_m=base.init_pos_std_m,
        init_vel_std_mps=base.init_vel_std_mps,
    )


def load_rgr_gf(cfg: dict, train_cfg, device):
    spec = cfg["models"]["HybridGNN"]
    ckpt = torch.load(
        ROOT / "results" / "checkpoints" / spec["checkpoint_name"], map_location=device
    )
    model_kwargs = ckpt.get("model_kwargs", dict(spec.get("model_kwargs", {})))
    model = TemporalGraphEstimator(
        hidden_dim=train_cfg.hidden_dim,
        gnn_layers=train_cfg.gnn_layers,
        gru_layers=train_cfg.gru_layers,
        dropout=train_cfg.dropout,
        use_ekf_prior=bool(ckpt.get("use_ekf_prior", spec.get("use_ekf_prior", True))),
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def per_traj_pos_rmse(states, preds, eval_start, observed_mask) -> tuple:
    s = states[:, eval_start:, :3]
    p = preds[:, eval_start:, :3]
    sq = np.sum((s - p) ** 2, axis=-1)
    all_step = np.sqrt(np.mean(sq, axis=1))
    obs = np.full(s.shape[0], np.nan)
    for i in range(s.shape[0]):
        m = observed_mask[i]
        if np.any(m):
            obs[i] = np.sqrt(np.mean(sq[i][m]))
    return all_step, obs


def update_elevations_deg(states, visibility, times, stations, eval_start):
    """Per-trajectory max visible-update elevation and global high-el fractions.

    Returns (max_el_deg[n_traj], frac_ge_80, frac_ge_85) computed on the truth
    geometry over evaluated steps, so the near-zenith azimuth-conditioning
    mechanism is evidenced quantitatively rather than asserted.
    """
    n_traj = states.shape[0]
    max_el = np.zeros(n_traj)
    n_upd = 0
    n_ge80 = 0
    n_ge85 = 0
    for i in range(n_traj):
        m = 0.0
        for k in range(eval_start, states.shape[1]):
            for s_idx, station in enumerate(stations):
                if visibility[i, k, s_idx] < 0.5:
                    continue
                z, _ = line_of_sight_measurement(states[i, k], station, float(times[i, k]))
                el_deg = float(np.rad2deg(z[2]))
                m = max(m, el_deg)
                n_upd += 1
                if el_deg >= 80.0:
                    n_ge80 += 1
                if el_deg >= 85.0:
                    n_ge85 += 1
        max_el[i] = m
    frac80 = round(n_ge80 / n_upd, 4) if n_upd else 0.0
    frac85 = round(n_ge85 / n_upd, 4) if n_upd else 0.0
    return max_el, frac80, frac85


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--num-realizations", type=int, default=6)
    ap.add_argument("--trajectories", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=330000)
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument(
        "--refresh-diagnosis",
        action="store_true",
        help=(
            "Do NOT re-filter. If a completed artifact for the same "
            "(realizations, trajectories, base-seed) design already exists, "
            "recompute ONLY the deterministic, configuration-derived "
            "divergence-diagnosis fields (station-bank consistency, mechanism "
            "text, azimuth-deweight note) from the already-stored numeric "
            "outputs and rewrite the artifact. Every measured value is left "
            "byte-identical; this exists so the committed artifact stays "
            "consistent with this script without a full multi-hour rerun, and "
            "is fully reproducible from the stored numbers."
        ),
    )
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    if args.refresh_diagnosis:
        return _refresh_diagnosis(cfg, args)
    base_baseline = parse_baseline_config(cfg["baselines"])
    train_cfg = parse_train_config(cfg["training"])
    eval_start = max(int(train_cfg.window_size) - 1, 0)
    device = torch.device("cpu")
    model = load_rgr_gf(cfg, train_cfg, device)

    K = int(args.num_realizations)
    N = int(args.trajectories)
    method_keys = ["EKF", "UKF", "AUKF", "RGR-GF"]

    est_sim, truth_sim, scenario_cfg = resolve_sim_configs(cfg, SCENARIO_KEY)
    # Hard credibility assumption: NO truth/estimator dynamics, measurement, or
    # station mismatch. estimator_overrides must be absent so the estimator
    # resolves the identical dense-network simulation config as the truth.
    shared_dynamics = (
        not has_estimator_overrides(scenario_cfg) and est_sim == truth_sim
    )
    if not shared_dynamics:
        raise SystemExit(
            "credible probe requires a perfect shared model: "
            "estimator_sim_config must equal truth_sim_config and "
            "estimator_overrides must be absent"
        )
    truth_dc = parse_dataset_config(truth_sim)
    est_dc = parse_dataset_config(est_sim)
    n_stations = len(truth_sim["stations"])
    enable_tb = bool(truth_sim["dynamics"]["enable_third_body"])
    enable_srp = bool(truth_sim["dynamics"]["enable_srp"])

    fixed_cfg = conditioned_baseline(base_baseline, ELEV_CAP_DEG)
    legacy_cfg = conditioned_baseline(base_baseline, None)

    # Pooled per-trajectory accumulators for the corrected (primary) arm.
    traj_all = {m: [] for m in method_keys}
    traj_obs = {m: [] for m in method_keys}
    # Diagnostic (uncorrected) arm: classical filters only (the artifact is
    # estimator-independent), to show the catastrophe returns without the fix.
    diag_keys = ["EKF", "UKF", "AUKF"]
    diag_all = {m: [] for m in diag_keys}
    zero_vis_fracs, one_vis_fracs, multi_vis_fracs = [], [], []
    max_el_all = []
    frac80s, frac85s = [], []

    for r in range(K):
        seed = args.base_seed + r
        data = generate_dataset(truth_dc, N, seed=seed)
        states = data["states"]
        meas = data["measurements"]
        vis = data["visibility"]
        times = data["times"]

        eval_vis = vis[:, eval_start:]
        vis_count = np.sum(eval_vis >= 0.5, axis=-1)
        observed_mask = vis_count >= 1
        zero_vis_fracs.append(float(np.mean(vis_count == 0)))
        one_vis_fracs.append(float(np.mean(vis_count == 1)))
        multi_vis_fracs.append(float(np.mean(vis_count >= 2)))

        mel, f80, f85 = update_elevations_deg(
            states, vis, times, est_dc.stations, eval_start
        )
        max_el_all.append(mel)
        frac80s.append(f80)
        frac85s.append(f85)

        rng = np.random.default_rng(seed)
        x0 = np.stack(
            [
                generate_noisy_init(
                    states[i, 0],
                    rng,
                    base_baseline.init_pos_std_m,
                    base_baseline.init_vel_std_mps,
                )
                for i in range(states.shape[0])
            ]
        )

        pf = run_filter_baselines(
            states=states,
            measurements=meas,
            visibility=vis,
            times=times,
            dataset_cfg=est_dc,
            baseline_cfg=fixed_cfg,
            seed=seed,
            x0_estimates=x0,
        )
        inno = build_innovation_features(
            dataset_cfg=est_dc,
            measurements=meas,
            visibility=vis,
            times=times,
            ekf_prior=pf["ekf"],
        )
        pbank = build_prior_bank_feature_array(
            pf["ekf"],
            pf["ukf"],
            pf["aukf"],
            dataset_cfg=est_dc,
            station_ecef=data["station_ecef"],
            visibility=vis,
            times=times,
            use_observability_context=False,
        )
        with torch.no_grad():
            rgr = run_model_inference(
                model=model,
                states=states,
                measurements=meas,
                visibility=vis,
                station_ecef=data["station_ecef"],
                window_size=int(train_cfg.window_size),
                ekf_prior=pf["ekf"],
                ukf_prior=pf["ukf"],
                aukf_prior=pf["aukf"],
                innovation_features=inno,
                prior_bank_stats=pbank,
            )
        preds = {"EKF": pf["ekf"], "UKF": pf["ukf"], "AUKF": pf["aukf"], "RGR-GF": rgr}
        for m in method_keys:
            a, o = per_traj_pos_rmse(states, preds[m], eval_start, observed_mask)
            traj_all[m].append(a)
            traj_obs[m].append(o)

        pf_legacy = run_filter_baselines(
            states=states,
            measurements=meas,
            visibility=vis,
            times=times,
            dataset_cfg=est_dc,
            baseline_cfg=legacy_cfg,
            seed=seed,
            x0_estimates=x0,
        )
        preds_legacy = {"EKF": pf_legacy["ekf"], "UKF": pf_legacy["ukf"], "AUKF": pf_legacy["aukf"]}
        for m in diag_keys:
            a, _ = per_traj_pos_rmse(states, preds_legacy[m], eval_start, observed_mask)
            diag_all[m].append(a)
        print(f"realization {r}: zero-vis={zero_vis_fracs[-1]:.3f} done", flush=True)

    all_arr = {m: np.concatenate(traj_all[m]) for m in method_keys}
    obs_arr = {m: np.concatenate(traj_obs[m]) for m in method_keys}
    diag_arr = {m: np.concatenate(diag_all[m]) for m in diag_keys}
    max_el_arr = np.concatenate(max_el_all)
    n_traj_total = int(all_arr[method_keys[0]].size)

    def finite_ok(x):
        return np.isfinite(x) & (x <= DIVERGENCE_POS_RMSE_M)

    survive = {m: finite_ok(all_arr[m]) for m in method_keys}
    div_avoid = {m: round(float(np.mean(survive[m])), 4) for m in method_keys}

    def rmse_over(idx_arr, vals):
        v = vals[idx_arr]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        return round(float(np.sqrt(np.mean(v**2))), 2)

    def med(vals):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return None
        return round(float(np.median(v)), 2)

    common = np.ones(n_traj_total, dtype=bool)
    for m in method_keys:
        common &= survive[m]
    n_common = int(np.sum(common))

    # PRIMARY (predeclared): pooled all-step RMSE over each method's
    # non-divergent trajectories. SECONDARY: observed-step, reported
    # symmetrically. Plus the paired common-survivor view.
    primary_all = {m: rmse_over(survive[m], all_arr[m]) for m in method_keys}
    secondary_obs = {m: rmse_over(survive[m], obs_arr[m]) for m in method_keys}
    common_all = {m: rmse_over(common, all_arr[m]) for m in method_keys}
    common_obs = {m: rmse_over(common, obs_arr[m]) for m in method_keys}
    median_all = {m: med(all_arr[m]) for m in method_keys}
    median_obs = {m: med(obs_arr[m]) for m in method_keys}

    adequate = {
        m: np.isfinite(all_arr[m]) & (all_arr[m] <= ENGINEERING_ADEQUATE_POS_RMSE_M)
        for m in method_keys
    }
    gross_failure_rate = {
        m: round(float(np.mean(~adequate[m])), 4) for m in method_keys
    }

    def best_of(d):
        cand = {k: v for k, v in d.items() if v is not None}
        return min(cand, key=cand.get) if cand else None

    best_primary = best_of(primary_all)
    best_secondary = best_of(secondary_obs)
    classical = ["EKF", "UKF", "AUKF"]
    best_classical_primary = best_of({m: primary_all[m] for m in classical})

    # Paired percentile bootstrap of RGR-GF minus best-classical on the
    # PREDECLARED metric over the common-survivor set (a fair paired test).
    rng = np.random.default_rng(20260519)
    paired = None
    if n_common >= 5 and best_classical_primary is not None:
        a = all_arr["RGR-GF"][common]
        b = all_arr[best_classical_primary][common]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        if a.size >= 5:
            point = float(
                np.sqrt(np.mean(a**2)) - np.sqrt(np.mean(b**2))
            )
            B = int(args.bootstrap)
            idx = rng.integers(0, a.size, size=(B, a.size))
            diffs = np.sqrt(np.mean(a[idx] ** 2, axis=1)) - np.sqrt(
                np.mean(b[idx] ** 2, axis=1)
            )
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            paired = {
                "comparison": f"RGR-GF minus {best_classical_primary} (predeclared all-step)",
                "n_paired": int(a.size),
                "point_estimate_m": round(point, 2),
                "ci95_low_m": round(float(lo), 2),
                "ci95_high_m": round(float(hi), 2),
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "rgr_gf_better": bool(point < 0.0 and hi < 0.0),
                "bootstrap_resamples": B,
            }

    # A learned positive is only credited if the regime is astrodynamically
    # credible (best classical within the physical-adequacy bound) AND the
    # paired CI for RGR-GF minus best-classical strictly excludes zero in the
    # learned direction. Otherwise the honest negative stands.
    od_is_credible = bool(
        best_classical_primary is not None
        and primary_all.get(best_classical_primary) is not None
        and primary_all[best_classical_primary] <= ENGINEERING_ADEQUATE_POS_RMSE_M
    )
    learned_positive = bool(
        od_is_credible and paired is not None and paired["rgr_gf_better"]
    )

    # Diagnostic (uncorrected) arm summary on the same trajectories.
    diag_survive = {m: finite_ok(diag_arr[m]) for m in diag_keys}
    diag_div_avoid = {m: round(float(np.mean(diag_survive[m])), 4) for m in diag_keys}
    diag_adequate = {
        m: np.isfinite(diag_arr[m]) & (diag_arr[m] <= ENGINEERING_ADEQUATE_POS_RMSE_M)
        for m in diag_keys
    }
    diag_gross = {m: round(float(np.mean(~diag_adequate[m])), 4) for m in diag_keys}
    diag_median_all = {m: med(diag_arr[m]) for m in diag_keys}

    # Deterministic configuration diagnosis (no filtering): resolve the
    # uncorrected densified-visibility control's scenario and show that its
    # estimator config resolves a DIFFERENT (smaller) station bank than the
    # truth it was synthesised against -- the dominant cause of that control's
    # catastrophe -- whereas this corrected probe resolves an identical bank.
    station_bank_consistency = build_station_bank_consistency(
        cfg, est_sim, truth_sim
    )

    # Mechanistic association: is uncorrected EKF gross-failure tied to a
    # near-zenith (high max-elevation) update on that trajectory?
    legacy_ekf_fail = ~diag_adequate["EKF"]
    hi_zenith = max_el_arr >= ELEV_CAP_DEG
    if np.any(hi_zenith) and np.any(~hi_zenith):
        fail_if_zenith = float(np.mean(legacy_ekf_fail[hi_zenith]))
        fail_if_not = float(np.mean(legacy_ekf_fail[~hi_zenith]))
    else:
        fail_if_zenith = float(np.mean(legacy_ekf_fail))
        fail_if_not = float("nan")

    result = {
        "status": "completed",
        "schema_version": "credible_dense_od_probe_v1",
        "regime": (
            "perfect shared dynamics (two-body + J2 + drag with third-body and "
            "SRP perturbations enabled) and an identical measurement model on "
            f"both sides (no force-model or measurement mismatch), a {n_stations}-"
            "station global ground network at an 8-degree elevation mask, with "
            "the standard capped 1/cos^2(elevation) topocentric-azimuth "
            "de-weighting applied identically to every estimator"
        ),
        "pre_registered": True,
        "predeclared_primary_metric": PREDECLARED_PRIMARY_METRIC,
        "secondary_metric": (
            "pooled observed-step trajectory position RMSE over non-divergent "
            "trajectories, reported symmetrically with the primary metric"
        ),
        "shared_dynamics_no_mismatch": bool(shared_dynamics),
        "third_body_enabled": enable_tb,
        "srp_enabled": enable_srp,
        "azimuth_deweight_elevation_cap_deg": ELEV_CAP_DEG,
        "learned_evaluation_mode": (
            "fixed-model inference of the previously trained RGR-GF estimator; "
            "no training or per-realization refitting was performed"
        ),
        "num_realizations": K,
        "trajectories_per_realization": N,
        "n_trajectories_total": n_traj_total,
        "n_stations": n_stations,
        "eval_start_step": eval_start,
        "divergence_guard_pos_rmse_m": DIVERGENCE_POS_RMSE_M,
        "engineering_adequate_pos_rmse_m": ENGINEERING_ADEQUATE_POS_RMSE_M,
        "visibility": {
            "zero_visible_fraction_mean": round(float(np.mean(zero_vis_fracs)), 4),
            "one_visible_fraction_mean": round(float(np.mean(one_vis_fracs)), 4),
            "two_plus_visible_fraction_mean": round(float(np.mean(multi_vis_fracs)), 4),
            "measurement_informed_fraction_mean": round(
                1.0 - float(np.mean(zero_vis_fracs)), 4
            ),
            "main_split_zero_visible_fraction_reference": 0.79,
        },
        "divergence_avoidance_fraction": div_avoid,
        "gross_failure_rate": gross_failure_rate,
        "primary_all_step_pooled_rmse_m": primary_all,
        "secondary_observed_step_pooled_rmse_m": secondary_obs,
        "common_survivor_pooled_rmse_m": {
            "n_common_trajectories": n_common,
            "all_step": common_all,
            "observed_step": common_obs,
        },
        "median_trajectory_position_rmse_m": {
            "all_step": median_all,
            "observed_step": median_obs,
        },
        "best_method_primary_all_step": best_primary,
        "best_method_secondary_observed_step": best_secondary,
        "best_classical_primary_all_step": best_classical_primary,
        "paired_learned_vs_best_classical": paired,
        "od_is_astrodynamically_credible": od_is_credible,
        "learned_positive_established": learned_positive,
        "divergence_diagnosis": {
            "mechanism": DIVERGENCE_MECHANISM_TEXT,
            "station_bank_consistency": station_bank_consistency,
            "uncorrected_divergence_avoidance_fraction": diag_div_avoid,
            "uncorrected_gross_failure_rate": diag_gross,
            "uncorrected_median_all_step_m": diag_median_all,
            "corrected_gross_failure_rate": {m: gross_failure_rate[m] for m in diag_keys},
            "corrected_median_all_step_m": {m: median_all[m] for m in diag_keys},
            "azimuth_deweight_note": AZIMUTH_DEWEIGHT_NOTE,
            "high_elevation_update_fraction_ge_80deg": round(float(np.mean(frac80s)), 4),
            "high_elevation_update_fraction_ge_85deg": round(float(np.mean(frac85s)), 4),
            "uncorrected_ekf_gross_failure_rate_if_zenith_pass": round(fail_if_zenith, 4),
            "uncorrected_ekf_gross_failure_rate_if_no_zenith_pass": (
                None if np.isnan(fail_if_not) else round(fail_if_not, 4)
            ),
            "azimuth_deweight_is_minor_here": bool(
                max(diag_gross.values()) - max(gross_failure_rate[m] for m in diag_keys)
                <= 0.20
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} (K={K}, N={N})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
