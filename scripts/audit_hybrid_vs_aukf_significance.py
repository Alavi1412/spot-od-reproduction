#!/usr/bin/env python
"""Audit Hybrid/RGR-GF vs AUKF paired significance evidence.

This script is intentionally read-only with respect to experiment inputs. It
recomputes the manuscript Hybrid-vs-AUKF rows from results/trajectory_errors.csv
using the same paired trajectory-wise convention documented in paper/main.tex:

    diff = baseline trajectory position RMSE - candidate trajectory position RMSE

Positive mean gain therefore favors the hybrid candidate. Wilcoxon tests use the
one-sided alternative "candidate better". If SciPy is installed, the script uses
scipy.stats.wilcoxon(..., alternative="greater"); otherwise it uses a
standard normal-approximation implementation of the Wilcoxon signed-rank test
with average ranks for ties and no continuity correction, matching SciPy's
large-sample default for n=64. Bootstrap confidence intervals are percentile
paired-bootstrap intervals over shared trajectory indices.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_CSV = ROOT / "results" / "trajectory_errors.csv"
OUT_JSON = ROOT / "results" / "hybrid_vs_aukf_statistical_audit.json"
OUT_MD = ROOT / "results" / "hybrid_vs_aukf_statistical_audit.md"

METHOD_LABELS = {
    "HybridGNN": "RGR-GF",
    "InnovationHybridGNN": "IDP-RGR-GF",
}
SCENARIOS = ("test", "stress_test")
CANDIDATES = ("HybridGNN", "InnovationHybridGNN")
BASELINE = "AUKF"
BOOTSTRAP_RESAMPLES = 3000
CI_PERCENT = 95.0
BOOTSTRAP_SEED = 321


def load_trajectory_errors(path: Path) -> dict[tuple[str, str], dict[int, float]]:
    rows: dict[tuple[str, str], dict[int, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"scenario", "method", "traj_id", "traj_pos_rmse_m"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required trajectory CSV columns: {sorted(missing)}")
        for row in reader:
            key = (row["scenario"], row["method"])
            rows.setdefault(key, {})[int(row["traj_id"])] = float(row["traj_pos_rmse_m"])
    return rows


def paired_arrays(rows: dict[tuple[str, str], dict[int, float]], scenario: str, baseline: str, candidate: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    b = rows.get((scenario, baseline), {})
    c = rows.get((scenario, candidate), {})
    shared_ids = sorted(set(b).intersection(c))
    if not shared_ids:
        raise ValueError(f"No shared trajectory ids for {scenario} {candidate} vs {baseline}")
    return (
        np.asarray([b[i] for i in shared_ids], dtype=np.float64),
        np.asarray([c[i] for i in shared_ids], dtype=np.float64),
        shared_ids,
    )


def bootstrap_ci(diff: np.ndarray, *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = int(diff.size)
    samples = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diff[idx]))
    alpha = (100.0 - CI_PERCENT) / 2.0
    return {
        "mean_diff": float(np.mean(diff)),
        "ci_low": float(np.percentile(samples, alpha)),
        "ci_high": float(np.percentile(samples, 100.0 - alpha)),
    }


def _average_ranks(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return 1-indexed average ranks and tie group sizes for numeric values."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    tie_sizes: list[int] = []
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        # Ranks are 1-indexed; average of i+1 through j.
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        if j - i > 1:
            tie_sizes.append(j - i)
        i = j
    return ranks, tie_sizes


def _wilcoxon_greater_normal(baseline_arr: np.ndarray, candidate_arr: np.ndarray) -> tuple[float, float, str]:
    """Wilcoxon signed-rank, one-sided greater, normal approximation.

    Uses zero_method='wilcox' behavior: exact zero differences are discarded.
    This is the same large-sample approximation family used by SciPy for n>50
    when method='auto'. No continuity correction is applied.
    """
    diff = baseline_arr - candidate_arr
    diff = diff[diff != 0.0]
    if diff.size == 0:
        raise ValueError("All paired differences are zero; Wilcoxon statistic is undefined")
    ranks, tie_sizes = _average_ranks(np.abs(diff))
    w_plus = float(np.sum(ranks[diff > 0.0]))
    n = float(diff.size)
    mean = n * (n + 1.0) / 4.0
    var = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0
    if tie_sizes:
        var -= sum(t**3 - t for t in tie_sizes) / 48.0
    if var <= 0.0:
        raise ValueError(f"Non-positive Wilcoxon variance after tie correction: {var}")
    z = (w_plus - mean) / math.sqrt(var)
    # One-sided P(W+ >= observed), normal approximation.
    p_value = 0.5 * math.erfc(z / math.sqrt(2.0))
    return w_plus, float(p_value), "normal_approx_no_continuity_scipy_fallback"


def wilcoxon_greater(baseline_arr: np.ndarray, candidate_arr: np.ndarray) -> tuple[float, float, str]:
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon  # type: ignore
    except ModuleNotFoundError:
        return _wilcoxon_greater_normal(baseline_arr, candidate_arr)
    stat, p_value = scipy_wilcoxon(baseline_arr, candidate_arr, alternative="greater")
    return float(stat), float(p_value), "scipy.stats.wilcoxon"


def summarize_pair(rows: dict[tuple[str, str], dict[int, float]], scenario: str, candidate: str) -> dict[str, object]:
    baseline_arr, candidate_arr, shared_ids = paired_arrays(rows, scenario, BASELINE, candidate)
    diff = baseline_arr - candidate_arr
    stat, p_value, backend = wilcoxon_greater(baseline_arr, candidate_arr)
    return {
        "scenario": scenario,
        "candidate_method": candidate,
        "candidate_label": METHOD_LABELS[candidate],
        "baseline_method": BASELINE,
        "paired_unit": "trajectory-wise position RMSE over the common 109-step evaluation window",
        "n_trajectories": int(diff.size),
        "shared_traj_id_min": int(min(shared_ids)),
        "shared_traj_id_max": int(max(shared_ids)),
        "wins": int(np.sum(diff > 0.0)),
        "losses": int(np.sum(diff < 0.0)),
        "ties": int(np.sum(diff == 0.0)),
        "win_rate_percent": float(100.0 * np.mean(diff > 0.0)),
        "mean_gain_m": float(np.mean(diff)),
        "median_gain_m": float(np.median(diff)),
        "bootstrap_ci": bootstrap_ci(diff, seed=BOOTSTRAP_SEED),
        "wilcoxon_alternative": "candidate better; baseline trajectory RMSE > candidate trajectory RMSE",
        "wilcoxon_backend": backend,
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p": float(p_value),
        "interpretation": "supports_candidate_superiority" if float(p_value) < 0.05 and float(np.mean(diff)) > 0.0 else "does_not_support_candidate_superiority",
    }


def format_float(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def build_markdown(results: Iterable[dict[str, object]]) -> str:
    results = list(results)
    backends = sorted({str(r["wilcoxon_backend"]) for r in results})
    lines: list[str] = []
    lines.append("# Hybrid-vs-AUKF Statistical Evidence Audit")
    lines.append("")
    lines.append("This audit recomputes the active manuscript AUKF comparator rows from `results/trajectory_errors.csv`.")
    lines.append("")
    lines.append("## Method")
    lines.append("- Paired unit: one trajectory's position RMSE over the shared 109-step evaluation window.")
    lines.append("- Difference convention: `AUKF trajectory RMSE - candidate trajectory RMSE`; positive values favor the candidate.")
    lines.append("- Wilcoxon signed-rank test: one-sided, alternative = candidate better. The script uses `scipy.stats.wilcoxon(..., alternative=\"greater\")` when SciPy is installed; otherwise it uses an internal large-sample normal approximation with average ranks for ties and no continuity correction.")
    lines.append(f"- Wilcoxon backend used in this run: {', '.join(backends)}.")
    lines.append(f"- Confidence interval: {CI_PERCENT:.0f}% percentile paired bootstrap over shared trajectory indices, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}.")
    lines.append("- Sources: `scripts/audit_hybrid_vs_aukf_significance.py`, `results/trajectory_errors.csv`, `paper/tables/significance.tex`, `paper/main.tex`.")
    lines.append("")
    lines.append("## Recomputed AUKF comparator rows")
    lines.append("| Scenario | Candidate | n | Mean gain [m] | 95% CI [m] | Win rate [%] | Wilcoxon p | Interpretation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        ci = r["bootstrap_ci"]  # type: ignore[index]
        lines.append(
            "| {scenario} | {label} vs AUKF | {n} | {mean} | [{lo}, {hi}] | {win} | {p:.4g} | {interp} |".format(
                scenario=r["scenario"],
                label=r["candidate_label"],
                n=r["n_trajectories"],
                mean=format_float(float(r["mean_gain_m"])),
                lo=format_float(float(ci["ci_low"])),  # type: ignore[index]
                hi=format_float(float(ci["ci_high"])),  # type: ignore[index]
                win=format_float(float(r["win_rate_percent"])),
                p=float(r["wilcoxon_p"]),
                interp=str(r["interpretation"]).replace("_", " "),
            )
        )
    lines.append("")
    lines.append("## Claim impact")
    lines.append("The recomputed AUKF rows do not support a Hybrid/RGR-GF superiority claim over AUKF. Both stress-test comparisons have negative mean gains and large one-sided p-values, while nominal comparisons also fail the directional test. The manuscript should therefore keep the conservative claim boundary: stress-regime gains are supported against fixed-noise UKF, not tuned AUKF.")
    lines.append("")
    lines.append("## Open questions")
    lines.append("1. Would the same conclusion hold under additional random seeds and larger trajectory counts?")
    lines.append("2. Would parameter-matched graph/no-graph and adaptive-filter tuning sweeps change the AUKF-vs-hybrid ordering?")
    lines.append("3. How sensitive are the AUKF rows to bootstrap seed, multiple-comparison correction, and alternative nonparametric effect-size summaries?")
    lines.append("4. Do public-replay and high-fidelity force-model scenarios show the same no-superiority pattern, or do they require separate paired evidence before making any claim?")
    lines.append("")
    lines.append("## Primary/statistical sources")
    lines.append("- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. *Biometrics Bulletin*, 1(6), 80–83. DOI: 10.2307/3001968.")
    lines.append("- Efron, B. and Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.")
    lines.append("- Julier, S. J. and Uhlmann, J. K. (1997). New Extension of the Kalman Filter to Nonlinear Systems. SPIE. DOI: 10.1117/12.280797.")
    lines.append("- Wan, E. A. and van der Merwe, R. (2000). The Unscented Kalman Filter for Nonlinear Estimation. IEEE. DOI: 10.1109/ASSPCC.2000.882463.")
    lines.append("- Mehra, R. K. (1970). On the Identification of Variances and Adaptive Kalman Filtering. *IEEE TAC*, 15(2), 175–184. DOI: 10.1109/TAC.1970.1099422.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = load_trajectory_errors(TRAJECTORY_CSV)
    results = [summarize_pair(rows, scenario, candidate) for scenario in SCENARIOS for candidate in CANDIDATES]
    payload = {
        "artifact": "hybrid_vs_aukf_statistical_audit",
        "trajectory_source": str(TRAJECTORY_CSV.relative_to(ROOT)),
        "output_markdown": str(OUT_MD.relative_to(ROOT)),
        "paired_convention": "baseline trajectory RMSE minus candidate trajectory RMSE; positive favors candidate",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "ci_percent": CI_PERCENT,
        "rows": results,
        "claim_boundary": "No Hybrid/RGR-GF superiority over AUKF is supported by these paired trajectory-wise tests.",
        "citations": ["wilcoxon1945", "efron1994", "julier1997unscented", "wan2000unscented", "mehra1970adaptive"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_MD.write_text(build_markdown(results), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
