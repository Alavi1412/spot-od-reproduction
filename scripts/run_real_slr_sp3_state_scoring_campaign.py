#!/usr/bin/env python
"""Bounded public precise-reference state-scoring campaign.

This script consolidates the existing public LAGEOS CRD/SP3 artifacts into a
single no-test-leakage state-scoring record.  It deliberately does not relabel
the correction/provenance audit as validation.  Instead, it records what can be
claimed from the current public data:

* a validation-selected sparse-SLR classical filter scored on a strictly later
  test week against the independent SP3 precise-orbit product;
* a validation-selected controlled dynamics candidate set that includes the
  held-out learned residual calibrator, also scored on the later test week;
* the older sparse-SLR learned calibrator no-leakage check, retained as a
  bounded negative because it uses real CRD normal points and SP3 state scoring
  but not a train/validation/test temporal selector.

The output is intended as machine-readable claim-boundary evidence.  It is not
an operational POD validation and not a central learned-versus-classical
validation of the simulator conclusion.
"""

from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover - entrypoint dependent
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gnn_state_estimation.utils.io import dump_json
from gnn_state_estimation.utils.runtime import utc_now_iso


DEFAULT_HIFI_JSON = Path(
    "results/real_slr_sp3_hifi/real_slr_sp3_hifi_validation.json"
)
DEFAULT_CAL_JSON = Path(
    "results/real_slr_sp3_od/sp3_residual_calibrator.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "results/real_slr_sp3_state_scoring_campaign/"
    "real_slr_sp3_state_scoring_campaign.json"
)
DEFAULT_TABLE = Path("paper/tables/real_slr_sp3_state_scoring_campaign.tex")

BOOTSTRAP_SEED = 20260522
BOOTSTRAP_N = 5000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(x: float | None, ndigits: int = 2):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), ndigits)


def _bootstrap_ci(values: np.ndarray) -> list:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, values.size, size=(BOOTSTRAP_N, values.size))
    means = values[idx].mean(axis=1)
    return [
        round(float(np.percentile(means, 2.5)), 2),
        round(float(np.percentile(means, 97.5)), 2),
    ]


def paired_gap_summary(
    rows: list[dict], a_key: str, b_key: str, *, field: str
) -> dict:
    """Paired gap a-b; positive means candidate ``a`` has larger RMSE."""
    pairs = []
    for row in rows:
        vals = row.get(field, {})
        a = vals.get(a_key)
        b = vals.get(b_key)
        if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
            pairs.append(float(a) - float(b))
    arr = np.asarray(pairs, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean_gap_m": _round(arr.mean()),
        "median_gap_m": _round(np.median(arr)),
        "n_a_lower_rmse": int((arr < 0.0).sum()),
        "fraction_a_lower_rmse": round(float((arr < 0.0).mean()), 3),
        "bootstrap95_mean_gap_m": _bootstrap_ci(arr),
        "gap_convention": (
            "a_minus_b; positive means candidate a has larger held-out RMSE"
        ),
    }


def select_lowest(validation_means: dict[str, float]) -> str:
    finite = {
        k: float(v)
        for k, v in validation_means.items()
        if v is not None and np.isfinite(v)
    }
    if not finite:
        raise ValueError("no finite validation means")
    return min(finite, key=lambda k: (finite[k], k))


def _split_arcs(hifi: dict, split: str) -> list[dict]:
    return [
        a for a in hifi.get("arcs", [])
        if a.get("status") == "completed" and a.get("split") == split
    ]


def sparse_slr_temporal_selector(hifi: dict) -> dict:
    """Validation-selected classical sparse-SLR state-scoring arm."""
    labels = [
        "UKF (compact)",
        "UKF (higher-fidelity)",
        "AUKF (higher-fidelity)",
    ]
    val_pool = hifi["sparse_slr_operational_realism"]["val"]
    test_pool = hifi["sparse_slr_operational_realism"]["test"]
    validation_means = {
        label: val_pool[label]["mean_arc_rms_m"] for label in labels
    }
    test_means = {
        label: test_pool[label]["mean_arc_rms_m"] for label in labels
    }
    selected = select_lowest(validation_means)
    test_best = select_lowest(test_means)
    test_arcs = _split_arcs(hifi, "test")
    return {
        "status": "completed",
        "uses_public_crd_normal_points": True,
        "uses_independent_sp3_reference": True,
        "candidate_pool_includes_learned_model": False,
        "selection_rule": "choose the lowest mean arc RMSE on the validation week",
        "selection_split": "validation week 260502",
        "test_split": "strictly later week 260509",
        "validation_mean_arc_rms_m": validation_means,
        "test_mean_arc_rms_m": test_means,
        "selected_candidate": selected,
        "selected_candidate_family": "classical",
        "selected_test_mean_arc_rms_m": test_means[selected],
        "test_best_candidate": test_best,
        "test_best_mean_arc_rms_m": test_means[test_best],
        "selected_minus_test_best_m": _round(
            test_means[selected] - test_means[test_best]
        ),
        "selected_vs_test_best_paired_gap": paired_gap_summary(
            test_arcs,
            selected,
            test_best,
            field="held_out_position_rmse_m",
        ),
        "interpretation": (
            "Honest public CRD/SP3 state-scoring arm for classical filters. "
            "It cannot adjudicate learned-versus-classical performance because "
            "the sparse-SLR temporal candidate pool contains no learned filter."
        ),
    }


def controlled_dynamics_temporal_selector(hifi: dict) -> dict:
    """Validation-selected controlled propagation arm including learned model."""
    cpd = hifi["controlled_pure_dynamics"]
    cal = hifi["learned_calibrator"]
    selected_lam = cal.get("selected_ridge_lambda")
    lam_key = f"{float(selected_lam):.0e}" if selected_lam is not None else None
    cal_val = None
    if lam_key is not None:
        cal_val = cal.get(
            "validation_ridge_grid_calibrated_mean_rms_m", {}
        ).get(lam_key)
    validation_means = {
        "compact_classical": cpd["val"]["compact_mean_rms_m"],
        "higher_fidelity_classical": cpd["val"]["hifi_mean_rms_m"],
        "learned_calibrated_hifi": cal_val,
    }
    test_means = {
        "compact_classical": cpd["test"]["compact_mean_rms_m"],
        "higher_fidelity_classical": cpd["test"]["hifi_mean_rms_m"],
        "learned_calibrated_hifi": cal["test_controlled_pd"].get(
            "calibrated_hifi_mean_rms_m"
        ),
    }
    selected = select_lowest(validation_means)
    test_best = select_lowest(test_means)
    hifi_vs_compact = cpd["test"].get("hifi_vs_compact", {})
    learned_vs_hifi = cal["test_controlled_pd"].get(
        "calibrated_vs_hifi", {}
    )
    selected_vs_best = {
        "n": hifi_vs_compact.get("n"),
        "mean_gap_m": _round(test_means[selected] - test_means[test_best]),
        "bootstrap95_mean_gap_m": (
            hifi_vs_compact.get("bootstrap95_mean_improvement_m")
            if selected == "compact_classical"
            and test_best == "higher_fidelity_classical"
            else [None, None]
        ),
        "gap_convention": (
            "selected_minus_test_best; positive means the validation-selected "
            "candidate has larger held-out RMSE"
        ),
    }
    return {
        "status": "completed",
        "uses_public_crd_normal_points": False,
        "uses_independent_sp3_reference": True,
        "candidate_pool_includes_learned_model": True,
        "selection_rule": (
            "choose the lowest validation-week mean RMSE after selecting the "
            "learned ridge strength only on the validation week"
        ),
        "selection_split": "validation week 260502",
        "test_split": "strictly later week 260509",
        "validation_mean_rms_m": validation_means,
        "test_mean_rms_m": test_means,
        "selected_candidate": selected,
        "selected_candidate_family": (
            "learned" if selected == "learned_calibrated_hifi" else "classical"
        ),
        "selected_test_mean_rms_m": test_means[selected],
        "test_best_candidate": test_best,
        "test_best_mean_rms_m": test_means[test_best],
        "selected_vs_test_best_paired_gap": selected_vs_best,
        "learned_candidate_vs_higher_fidelity_classical": {
            "n": learned_vs_hifi.get("n"),
            "mean_improvement_m": learned_vs_hifi.get("mean_improvement_m"),
            "bootstrap95_mean_improvement_m": learned_vs_hifi.get(
                "bootstrap95_mean_improvement_m"
            ),
            "improvement_convention": (
                "higher_fidelity_classical_rmse minus "
                "learned_calibrated_hifi_rmse; positive would favor learned"
            ),
        },
        "interpretation": (
            "This is a leakage-controlled public-SP3 state propagation arm "
            "that includes a learned candidate, but it is not sparse-SLR OD "
            "because candidate scoring starts from precise SP3 states."
        ),
    }


def sparse_slr_learned_no_leakage_check(cal: dict) -> dict:
    """Older CRD/SP3 learned check, summarized with paired uncertainty."""
    arcs = cal.get("arcs", [])

    def learned_gap(proto: str) -> dict:
        label = (
            "loao_calibrated_rmse_m"
            if proto == "loao"
            else "looo_calibrated_rmse_m"
        )
        rows = []
        for arc in arcs:
            unc = arc.get("uncalibrated_rmse_m", {}).get("UKF (fixed-noise)")
            learned = arc.get(label, {}).get("Calibrated-UKF (fixed-noise)")
            if (
                unc is not None
                and learned is not None
                and np.isfinite(unc)
                and np.isfinite(learned)
            ):
                rows.append({"rmse": {"learned": learned, "classical": unc}})
        return paired_gap_summary(
            rows, "learned", "classical", field="rmse"
        )

    pooled = cal["pooled_held_out_position_rmse_m"]
    verdict = cal["verdict"]
    return {
        "status": cal.get("status"),
        "uses_public_crd_normal_points": True,
        "uses_independent_sp3_reference": True,
        "candidate_pool_includes_learned_model": True,
        "selection_rule": (
            "learned calibrator hyperparameters predeclared; no temporal "
            "validation selector in this older ten-arc arm"
        ),
        "protocols": {
            "leave_one_arc_out": {
                "classical_reference_mean_rmse_m": pooled["uncalibrated"][
                    "UKF (fixed-noise)"
                ],
                "learned_mean_rmse_m": pooled["loao_calibrated"][
                    "Calibrated-UKF (fixed-noise)"
                ],
                "learned_minus_classical_paired_gap": learned_gap("loao"),
                "beats_classical_reference": verdict["loao"][
                    "beats_best_uncalibrated_reference"
                ],
            },
            "leave_one_object_out": {
                "classical_reference_mean_rmse_m": pooled["uncalibrated"][
                    "UKF (fixed-noise)"
                ],
                "learned_mean_rmse_m": pooled["looo_calibrated"][
                    "Calibrated-UKF (fixed-noise)"
                ],
                "learned_minus_classical_paired_gap": learned_gap("looo"),
                "beats_classical_reference": verdict["looo"][
                    "beats_best_uncalibrated_reference"
                ],
            },
        },
        "claimed_as_positive_contribution": verdict[
            "claimed_as_positive_contribution"
        ],
        "interpretation": (
            "No-leakage CRD/SP3 learned-vs-classical check.  It is negative "
            "and not a temporally validation-selected campaign."
        ),
    }


def build_result(hifi_path: Path, cal_path: Path) -> dict:
    hifi = load_json(hifi_path)
    cal = load_json(cal_path)
    sparse = sparse_slr_temporal_selector(hifi)
    controlled = controlled_dynamics_temporal_selector(hifi)
    learned_check = sparse_slr_learned_no_leakage_check(cal)
    return {
        "schema_version": "real_slr_sp3_state_scoring_campaign_v1",
        "generated_utc": utc_now_iso(),
        "status": "completed",
        "source_artifacts": [
            {
                "artifact_id": hifi_path.as_posix(),
                "sha256": sha256_file(hifi_path),
            },
            {
                "artifact_id": cal_path.as_posix(),
                "sha256": sha256_file(cal_path),
            },
        ],
        "public_corpus": {
            "targets": hifi.get("targets", []),
            "num_sparse_slr_arcs": hifi.get("num_arcs_completed"),
            "sp3_analysis_center": hifi.get("sp3_analysis_center"),
            "split_weeks": hifi.get("split_weeks"),
        },
        "selection_integrity": {
            "test_set_information_used_for_selection": False,
            "temporal_test_week": "260509",
            "validation_week": "260502",
            "learned_ridge_selected_on_validation_only": True,
            "paired_uncertainty_where_available": True,
        },
        "campaigns": {
            "sparse_slr_temporal_selector": sparse,
            "controlled_sp3_dynamics_temporal_selector": controlled,
            "sparse_slr_learned_no_leakage_check": learned_check,
        },
        "headline_readout": {
            "sparse_slr_selected_candidate": sparse["selected_candidate"],
            "sparse_slr_selected_test_mean_arc_rms_m": sparse[
                "selected_test_mean_arc_rms_m"
            ],
            "sparse_slr_test_best_candidate": sparse["test_best_candidate"],
            "controlled_selected_candidate": controlled["selected_candidate"],
            "controlled_selected_test_mean_rms_m": controlled[
                "selected_test_mean_rms_m"
            ],
            "controlled_test_best_candidate": controlled[
                "test_best_candidate"
            ],
            "learned_sparse_slr_positive": learned_check[
                "claimed_as_positive_contribution"
            ],
            "learned_controlled_positive": (
                controlled["learned_candidate_vs_higher_fidelity_classical"][
                    "mean_improvement_m"
                ]
                is not None
                and controlled[
                    "learned_candidate_vs_higher_fidelity_classical"
                ]["mean_improvement_m"]
                > 0.0
                and controlled[
                    "learned_candidate_vs_higher_fidelity_classical"
                ]["bootstrap95_mean_improvement_m"][0]
                is not None
                and controlled[
                    "learned_candidate_vs_higher_fidelity_classical"
                ]["bootstrap95_mean_improvement_m"][0]
                > 0.0
            ),
            "paper_strength_class": "bounded_public_probe",
        },
        "claim_boundary": {
            "defensible_status": "bounded_public_precise_reference_probe",
            "true_external_validation_campaign_feasible_in_this_loop": False,
            "is_operational_validation": False,
            "is_central_learned_vs_classical_validation": False,
            "is_centimetre_slr_or_flight_validation": False,
            "does_not_relabel_provenance_as_validation": True,
            "appropriate_use": (
                "Use only as a bounded public precise-reference state-scoring "
                "probe.  It strengthens external-state-scoring transparency "
                "but is not strong enough to support a central operational "
                "validation claim."
            ),
            "why_not_full_external_validation": [
                (
                    "The sparse-SLR temporal selector contains only classical "
                    "filters; the learned sparse-SLR check is no-leakage but "
                    "not a train/validation/test temporal selector."
                ),
                (
                    "The controlled learned candidate is selected honestly but "
                    "starts from SP3 states, so it is a propagation-state "
                    "scoring arm rather than real-measurement OD."
                ),
                (
                    "The test evidence is small and LAGEOS-specific, with "
                    "wide paired intervals and no independent analysis-centre "
                    "cross-check."
                ),
            ],
        },
    }


def _fmt(x) -> str:
    if x is None:
        return "--"
    return f"{float(x):.2f}"


def write_table(result: dict, path: Path) -> None:
    sparse = result["campaigns"]["sparse_slr_temporal_selector"]
    controlled = result["campaigns"]["controlled_sp3_dynamics_temporal_selector"]
    learned = result["campaigns"]["sparse_slr_learned_no_leakage_check"]
    loao = learned["protocols"]["leave_one_arc_out"]
    looo = learned["protocols"]["leave_one_object_out"]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Bounded public precise-reference state-scoring "
            r"campaign. Selection uses only validation information where a "
            r"temporal selector is available, and the held-out week is scored "
            r"against the independent SP3 precise-orbit product. The sparse-SLR "
            r"temporal selector uses real CRD normal points but has only "
            r"classical candidates; the learned real-measurement check is "
            r"no-leakage but not a temporal validation selector; the controlled "
            r"learned arm is state propagation from SP3 states rather than "
            r"real-measurement OD. The result is therefore a bounded public "
            r"probe, not operational validation.}"
        ),
        r"  \label{tab:real_slr_sp3_state_scoring_campaign}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        (
            r"    Arm & Validation-selected candidate & Test mean [m] & "
            r"Test best [m] & Paired gap [m] \\"
        ),
        r"    \midrule",
        (
            "    Sparse-SLR temporal selector & "
            f"{sparse['selected_candidate']} & "
            f"{_fmt(sparse['selected_test_mean_arc_rms_m'])} & "
            f"{sparse['test_best_candidate']}: "
            f"{_fmt(sparse['test_best_mean_arc_rms_m'])} & "
            f"{_fmt(sparse['selected_minus_test_best_m'])} "
            f"[{_fmt(sparse['selected_vs_test_best_paired_gap']['bootstrap95_mean_gap_m'][0])}, "
            f"{_fmt(sparse['selected_vs_test_best_paired_gap']['bootstrap95_mean_gap_m'][1])}] \\\\"
        ),
        (
            "    Controlled SP3-state selector & "
            f"{controlled['selected_candidate'].replace('_', ' ')} & "
            f"{_fmt(controlled['selected_test_mean_rms_m'])} & "
            f"{controlled['test_best_candidate'].replace('_', ' ')}: "
            f"{_fmt(controlled['test_best_mean_rms_m'])} & "
            f"{_fmt(controlled['selected_vs_test_best_paired_gap']['mean_gap_m'])} "
            f"[{_fmt(controlled['selected_vs_test_best_paired_gap']['bootstrap95_mean_gap_m'][0])}, "
            f"{_fmt(controlled['selected_vs_test_best_paired_gap']['bootstrap95_mean_gap_m'][1])}] \\\\"
        ),
        (
            "    Learned sparse-SLR LOAO check & Calibrated UKF & "
            f"{_fmt(loao['learned_mean_rmse_m'])} & "
            f"UKF: {_fmt(loao['classical_reference_mean_rmse_m'])} & "
            f"{_fmt(loao['learned_minus_classical_paired_gap']['mean_gap_m'])} "
            f"[{_fmt(loao['learned_minus_classical_paired_gap']['bootstrap95_mean_gap_m'][0])}, "
            f"{_fmt(loao['learned_minus_classical_paired_gap']['bootstrap95_mean_gap_m'][1])}] \\\\"
        ),
        (
            "    Learned sparse-SLR LOOO check & Calibrated UKF & "
            f"{_fmt(looo['learned_mean_rmse_m'])} & "
            f"UKF: {_fmt(looo['classical_reference_mean_rmse_m'])} & "
            f"{_fmt(looo['learned_minus_classical_paired_gap']['mean_gap_m'])} "
            f"[{_fmt(looo['learned_minus_classical_paired_gap']['bootstrap95_mean_gap_m'][0])}, "
            f"{_fmt(looo['learned_minus_classical_paired_gap']['bootstrap95_mean_gap_m'][1])}] \\\\"
        ),
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  }",
        (
            r"  \\[2pt] {\footnotesize Gap is selected or learned candidate "
            r"minus the listed classical/test-best reference; positive values "
            r"mean larger held-out RMSE. LOAO = leave one arc out; LOOO = "
            r"leave one object out.}"
        ),
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hifi-json", type=Path, default=DEFAULT_HIFI_JSON)
    p.add_argument("--calibrator-json", type=Path, default=DEFAULT_CAL_JSON)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    p.add_argument(
        "--no-table",
        action="store_true",
        help="Write only the machine-readable JSON artifact.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    result = build_result(args.hifi_json, args.calibrator_json)
    dump_json(result, args.output_json)
    if not args.no_table:
        write_table(result, args.table)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_json": str(args.output_json),
                "paper_strength_class": result["headline_readout"][
                    "paper_strength_class"
                ],
                "sparse_slr_selected": result["headline_readout"][
                    "sparse_slr_selected_candidate"
                ],
                "controlled_selected": result["headline_readout"][
                    "controlled_selected_candidate"
                ],
                "learned_sparse_slr_positive": result["headline_readout"][
                    "learned_sparse_slr_positive"
                ],
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
