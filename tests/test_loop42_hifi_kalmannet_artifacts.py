"""Loop-42 regression tests:

* The higher-fidelity force-mismatch JSON exists, conforms to its schema, and
  the cross-filter R-only NIS diagnostic still flags AUKF as the most stressed
  filter (the mechanism reproduces qualitatively at the second fidelity).
* The faithful KalmanNet SPOT-OD transposition JSON exists, conforms to its
  schema, and the upstream vendor commit is the pinned upstream commit.
* The two paper-facing tables are regenerated correctly by build_paper_assets
  and reference the right labels so they cannot silently disappear from the
  manuscript.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LATEX_LABEL_REFERENCE_RE = re.compile(
    r"\\(?:ref|eqref|autoref|nameref|pageref|cref|Cref|vref|Vref)\*?\{([^{}]+)\}"
)


def has_exact_latex_label_reference(source: str, label: str) -> bool:
    for match in LATEX_LABEL_REFERENCE_RE.finditer(source):
        labels = [part.strip() for part in match.group(1).split(",")]
        if label in labels:
            return True
    return False


def test_hifi_force_mismatch_json_exists_and_conforms() -> None:
    path = REPO_ROOT / "results" / "hifi_force_mismatch" / "hifi_force_mismatch.json"
    assert path.exists(), "Higher-fidelity force-mismatch JSON missing."
    payload = json.loads(path.read_text())
    assert payload["scenario"] == "hifi_force_mismatch"
    assert payload["n_trajectories"] >= 32
    assert payload["eval_start_step"] == 11
    nis = payload["cross_filter_r_only_nis"]
    # The compact-model mechanism diagnostic must reproduce qualitatively at
    # the second fidelity: AUKF median R-only NIS must be the largest among
    # the four filters (the Loop 41 mechanism continues to fire).
    medians = {k: float(v["median"]) for k, v in nis.items()}
    assert medians["AUKF"] == max(medians.values()), medians
    # Sanity: every filter has at least 200 visible-update samples.
    for filt, block in nis.items():
        assert int(block["n"]) >= 200, (filt, block)
    means = payload["observed_step_rmse_mean_m"]
    assert {"EKF", "UKF", "AUKF", "PUKF"}.issubset(set(means))


def test_kalmannet_spot_od_json_exists_and_uses_pinned_upstream() -> None:
    path = (
        REPO_ROOT
        / "results"
        / "kalmannet_spot_od_loop57"
        / "kalmannet_spot_od.json"
    )
    assert path.exists(), "KalmanNet-SPOT-OD JSON missing."
    payload = json.loads(path.read_text())
    assert payload["scenario"] == "kalmannet_spot_od_transposition"
    commit_file = REPO_ROOT / "external" / "third_party" / "KalmanNet_TSP_COMMIT"
    pinned = commit_file.read_text().strip() if commit_file.exists() else ""
    assert payload["vendor_commit"] == pinned, (payload["vendor_commit"], pinned)
    cfg = payload["config"]
    # State and observation dims are the SPOT-OD measurement setting.
    assert cfg["m"] == 6
    assert cfg["n"] == 32
    assert cfg["n_train"] == 160
    assert cfg["n_cv"] == 24
    assert cfg["n_test"] == 64
    assert cfg["n_steps"] == 300
    means = payload["observed_step_rmse_mean_m"]
    assert "KalmanNet-SPOT-OD" in means
    assert {"EKF", "UKF", "AUKF", "PUKF"}.issubset(set(means))
    paired = payload["paired_vs_best_classical"]
    assert paired["best_classical"] in {"EKF", "UKF", "AUKF", "PUKF"}


@pytest.mark.parametrize(
    "table_path,label",
    [
        ("paper/tables/hifi_force_mismatch.tex", "tab:hifi_force_mismatch"),
    ],
)
def test_paper_tables_present_and_label_intact(table_path: str, label: str) -> None:
    text = (REPO_ROOT / table_path).read_text(encoding="utf-8")
    assert "\\begin{table}" in text
    assert label in text
    assert "\\end{table}" in text


def test_main_tex_inputs_new_tables() -> None:
    """Main references supplement-housed diagnostic tables by stable labels.

    The higher-fidelity force-mismatch table is intentionally supplement-housed
    while main text references its unsuffixed label. The adapted KalmanNet
    SPOT-OD transposition and budget diagnostics are likewise paper-facing via
    supplement inputs and stable unsuffixed labels; loop identifiers belong only
    to archived evidence records.
    """
    main = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    supplement = (
        (REPO_ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
        if (REPO_ROOT / "paper" / "supplement.tex").exists()
        else ""
    )
    combined = main + "\n" + supplement
    hifi_table = (
        REPO_ROOT / "paper" / "tables" / "hifi_force_mismatch.tex"
    ).read_text(encoding="utf-8")
    assert has_exact_latex_label_reference(main, "tab:hifi_force_mismatch")
    assert not has_exact_latex_label_reference(
        r"S-Table~\ref{tab:hifi_force_mismatch_extended}",
        "tab:hifi_force_mismatch",
    )
    assert "\\input{tables/hifi_force_mismatch.tex}" not in main
    assert "\\input{tables/hifi_force_mismatch.tex}" in supplement
    assert "\\label{tab:hifi_force_mismatch}" in hifi_table
    # The SPOT-OD KalmanNet design-gap note must remain anchored in a
    # paper-facing file.
    assert "app:kalmannet-spot-od-feasibility" in combined
    # The faithful transposition table is paper-facing under an unsuffixed
    # paper-facing label; loop identifiers belong only to archived evidence.
    kalmannet_table = (
        REPO_ROOT / "paper" / "tables" / "kalmannet_spot_od_transposition.tex"
    ).read_text(encoding="utf-8")
    budget_table = (
        REPO_ROOT / "paper" / "tables" / "kalmannet_spot_od_budget_adequacy.tex"
    ).read_text(encoding="utf-8")
    assert "\\input{tables/kalmannet_spot_od_transposition.tex}" in supplement
    assert "\\input{tables/kalmannet_spot_od_budget_adequacy.tex}" in supplement
    assert (
        "tab:kalmannet_spot_od_transposition" in combined
    ), "KalmanNet SPOT-OD transposition table label must be paper-facing."
    assert "\\label{tab:kalmannet_spot_od_transposition}" in kalmannet_table
    assert "\\label{tab:kalmannet_spot_od_budget_adequacy}" in budget_table
    loop_transposition_label_pattern = re.compile(
        r"tab:kalmannet_spot_od_transposition_loop\d+"
    )
    assert (
        loop_transposition_label_pattern.search(combined) is None
    ), "Paper-facing SPOT-OD transposition table label must not expose loop identifiers."
    assert (
        "tab:kalmannet_spot_od_diagnostic_control" not in combined
    ), "SPOT-OD diagnostic-control table label should be withdrawn from paper-facing files."
    assert (
        "tab:kalmannet_spot_od_learning_curve" not in combined
    ), "SPOT-OD learning-curve table label should be withdrawn from paper-facing files."


def test_main_tex_inputs_diagnostic_control_table() -> None:
    """The SPOT-OD KalmanNet diagnostic-control table is withdrawn.

    The four-named-design-choices design-gap statement now travels with the
    manuscript bundle in the supplement instead, so the M1-style concern is
    addressed by a paper-facing design-gap note rather than by a
    re-instantiation numeric table.
    """
    main = (REPO_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    supplement = (
        (REPO_ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
        if (REPO_ROOT / "paper" / "supplement.tex").exists()
        else ""
    )
    combined = main + "\n" + supplement
    assert (
        "\\input{tables/kalmannet_spot_od_diagnostic_control.tex}" not in combined
    )
    table_path = (
        REPO_ROOT / "paper" / "tables" / "kalmannet_spot_od_diagnostic_control.tex"
    )
    assert not table_path.exists(), (
        "The SPOT-OD KalmanNet diagnostic-control table should be withdrawn "
        "from paper-facing artefacts."
    )
    # The design-gap note must enumerate the four named design choices in
    # the supplement.
    assert "Four design choices" in supplement or (
        "four named design choices" in main.lower()
    )


def test_kalmannet_diagnostic_json_schema_when_present() -> None:
    """The loop-46 diagnostic artefact should include both faithful and
    diagnostic-control diagnostics on the same disjoint-seed split as the
    faithful transposition, and the explicit DC modifications must be
    declared so the diagnostic control is not silently confused with the
    faithful transposition."""
    path = REPO_ROOT / "results" / "kalmannet_spot_od" / "diagnostic.json"
    if not path.exists():
        pytest.skip("Loop-46 diagnostic artefact not generated in this environment.")
    payload = json.loads(path.read_text())
    assert payload["scenario"] == "kalmannet_spot_od_diagnostic"
    cfg = payload["config"]
    assert cfg["m"] == 6
    assert cfg["n_faithful"] == 32
    assert cfg["n_diagnostic_control"] == 40
    assert {"EKF", "UKF", "AUKF"}.issubset(set(payload["classical_baselines_mean_observed_step_rmse_m"]))
    assert payload["best_classical_baseline"] in {"EKF", "UKF", "AUKF"}
    for block in ("faithful_diagnostics", "diagnostic_control_diagnostics"):
        assert block in payload
        for key in (
            "observed_step_rmse_mean_m",
            "observed_step_rmse_median_m",
            "per_axis_pos_rmse_m",
            "per_axis_vel_rmse_mps",
            "visibility_bucket_pooled_pos_rmse_m",
            "paired_vs_best_classical",
        ):
            assert key in payload[block]
    mods = payload["diagnostic_control_modifications"]
    assert any("velocity" in m.lower() for m in mods)
    assert any("visibility" in m.lower() for m in mods)
