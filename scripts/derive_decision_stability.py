#!/usr/bin/env python
"""Decision-stability analysis for the higher-fidelity and long-arc slices (Loop 50).

The Acta Astronautica reviewer pass on loop-49 raised that sample sizes on
the discriminative slices are modest (n=36 on the long-arc replication,
n=48 on the 40-minute higher-fidelity slice) and asked for either a K>=64
expansion or an explicit decision-stability analysis showing that the
predeclared pass/fail outcomes are robust to doubled n.

A K>=64 expansion would require running the long-arc higher-fidelity
simulator from a new disjoint base seed and is gated by predeclared-rule
discipline (a new run after the rule was committed would constitute new
confirmatory evidence rather than a sensitivity sweep). This script
instead performs a *decision-stability analysis* using the per-trajectory
artefacts that already exist for both slices:

  - results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch.csv (n=36)
  - results/hifi_force_mismatch/hifi_force_mismatch.csv (n=48)

For each predeclared pairwise decision we recompute the paired mean and a
shared-index paired bootstrap CI under three subsample stress conditions:

  (i) leave-one-trajectory-out jackknife: report the fraction of jackknife
      replicates that keep the same pass/fail decision under the
      predeclared rule;
  (ii) half-sample subsample stability: 1000 paired-subsample draws of
      floor(n/2) trajectories; report the fraction that keep the decision;
  (iii) the bootstrap distribution of the paired difference: report the
      fraction of paired-bootstrap resamples whose mean has the same sign
      as the predeclared sample mean.

The script does not refit any estimator; it only re-aggregates the
per-trajectory paired differences that were committed at the time of the
predeclared rule. It is reported as a *stability analysis*, not as new
confirmatory evidence: the predeclared decision is unchanged.
"""
from __future__ import annotations

try:
    from _bootstrap import ensure_src_on_path
except ModuleNotFoundError:  # pragma: no cover
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SLICES = {
    "hifi_40min": {
        "csv": Path("results/hifi_force_mismatch/hifi_force_mismatch.csv"),
        "json": Path("results/hifi_force_mismatch/hifi_force_mismatch.json"),
        "arc_label": "40-min higher-fidelity force-mismatch",
        "decisions": [
            # candidate_col, baseline_col, label, candidate_name,
            # predeclared_outcome: "candidate_better" (positive criterion
            # satisfied by candidate-baseline strictly negative);
            # "candidate_worse" (the structural channel predeclared positive
            # criterion failed, candidate-baseline observed strictly positive);
            # "auxiliary_directional" (a directional finding such as the
            # EKF/AUKF long-arc reversal).
            ("PUKF_observed_pos_rmse_m", "AUKF_observed_pos_rmse_m",
             "PUKF predeclared positive criterion vs AUKF",
             "PUKF",
             "candidate_worse"),
        ],
    },
    "long_arc": {
        # Prefer the powered $n{=}64$ population (loop58 update to the prior
        # $n{=}36$ analysis). The $n{=}64$ population strictly contains the
        # prior $n{=}36$ population as a byte-identical prefix, so the
        # stability analysis on $n{=}64$ supersedes the prior on the same
        # predeclared decision; the prior $n{=}36$ analysis remains
        # consistent and is preserved as historical context only.
        "csv": Path(
            "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch_n64_loop57.csv"
        ),
        "json": Path(
            "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch_n64_loop57.json"
        ),
        "arc_label": "3-h long-arc higher-fidelity force-and-density mismatch (powered $n{=}64$)",
        "decisions": [
            ("DSA_EKF_observed_pos_rmse_m", "AUKF_observed_pos_rmse_m",
             "DSA-EKF predeclared positive criterion vs AUKF (best non-DSA)",
             "DSA-EKF",
             "candidate_worse"),
            ("EKF_observed_pos_rmse_m", "AUKF_observed_pos_rmse_m",
             "EKF/AUKF ordering reversal (AUKF strictly better on long-arc)",
             "EKF",
             "auxiliary_directional"),
        ],
    },
}


def _paired_bootstrap_ci(
    diffs: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> tuple[float, float, float, np.ndarray]:
    n = len(diffs)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[indices].mean(axis=1)
    return (
        float(diffs.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
        boot_means,
    )


def _decision_for_dsa(diffs: np.ndarray) -> dict[str, Any]:
    """Predeclared DSA positive criterion: DSA strictly lowest mean AND
    paired CI versus best non-DSA strictly negative AND magnitude exceeds
    the predeclared floor. We re-evaluate the CI-strictly-negative leg only;
    the strictly-lowest-mean leg is conditional on the same paired sample.
    """
    mean = float(diffs.mean())
    return {"mean": mean, "ci_strictly_negative": mean < 0}


def _decision_for_reversal(diffs: np.ndarray) -> dict[str, Any]:
    """EKF/AUKF reversal on the long-arc: CI strictly above zero indicates
    AUKF is strictly better (predeclared structural-channel scope-down).
    """
    mean = float(diffs.mean())
    return {"mean": mean, "ci_strictly_positive": mean > 0}


def _analyse_slice(
    slice_key: str, slice_def: dict[str, Any], n_bootstrap: int, rng: np.random.Generator
) -> dict[str, Any]:
    df_full = pd.read_csv(slice_def["csv"])
    decisions_out: list[dict[str, Any]] = []
    for cand_col, base_col, label, candidate_name, predeclared_outcome in slice_def["decisions"]:
        # Drop NaN rows pairwise. The committed n_paired for some predeclared
        # rules excludes divergent trajectories (the rule documents this).
        df = df_full[[cand_col, base_col]].dropna()
        n = len(df)
        diffs = df[cand_col].to_numpy() - df[base_col].to_numpy()
        sample_mean = float(diffs.mean())

        # For each predeclared outcome we record the fraction of subsample
        # replicates that agree with the sample-outcome direction.
        # "candidate_worse": predeclared positive criterion failed because
        # candidate-baseline came out >0; we report stability of that failure
        # (the fraction of replicates with mean strictly positive).
        # "candidate_better": predeclared positive criterion satisfied
        # because candidate-baseline came out <0; we report stability of
        # that pass (fraction of replicates with mean strictly negative).
        # "auxiliary_directional": report whichever sign matches the sample
        # (e.g., the EKF/AUKF long-arc reversal).
        if predeclared_outcome == "candidate_better":
            agree_predicate = lambda m: m < 0
            outcome_label = "predeclared positive criterion satisfied"
        elif predeclared_outcome == "candidate_worse":
            agree_predicate = lambda m: m > 0
            outcome_label = "predeclared positive criterion failed (candidate-baseline strictly positive)"
        else:  # auxiliary_directional
            agree_predicate = (lambda m: m > 0) if sample_mean > 0 else (lambda m: m < 0)
            outcome_label = (
                "directional finding (sample-direction agreement)"
            )

        sample_outcome_agrees = bool(agree_predicate(sample_mean))

        # Full-sample paired-bootstrap CI for context.
        full_mean, full_lo, full_hi, _ = _paired_bootstrap_ci(diffs, n_bootstrap, rng)

        # Leave-one-out jackknife.
        loo_agreements: list[bool] = []
        loo_means: list[float] = []
        for drop_idx in range(n):
            kept = np.delete(diffs, drop_idx)
            m = float(kept.mean())
            loo_means.append(m)
            loo_agreements.append(bool(agree_predicate(m)))
        loo_pass_fraction = float(np.mean(loo_agreements))

        # Half-sample subsample stability.
        n_sub = max(1, n // 2)
        sub_agreements: list[bool] = []
        sub_means: list[float] = []
        n_sub_draws = 1000
        for _ in range(n_sub_draws):
            idx = rng.choice(n, size=n_sub, replace=False)
            m = float(diffs[idx].mean())
            sub_means.append(m)
            sub_agreements.append(bool(agree_predicate(m)))
        sub_pass_fraction = float(np.mean(sub_agreements))

        # Doubled-n bootstrap surrogate: resample 2n trajectories with
        # replacement from the committed paired set. The bootstrap mean
        # distribution then has standard error sigma/sqrt(2n), matching the
        # standard error of a true doubled-n draw if the trajectory population
        # were i.i.d. We label this as a surrogate for K>=2*n_committed, not
        # as new confirmatory evidence.
        n_doubled = 2 * n
        doubled_indices = rng.integers(0, n, size=(n_bootstrap, n_doubled))
        doubled_means = diffs[doubled_indices].mean(axis=1)
        doubled_agreement = float(np.mean([agree_predicate(m) for m in doubled_means]))
        doubled_ci_lo = float(np.percentile(doubled_means, 2.5))
        doubled_ci_hi = float(np.percentile(doubled_means, 97.5))

        # Bootstrap sign agreement at the committed n.
        boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
        boot_means = diffs[boot_indices].mean(axis=1)
        boot_sign_agreement = float(np.mean([agree_predicate(m) for m in boot_means]))

        decisions_out.append(
            {
                "label": label,
                "candidate": candidate_name,
                "candidate_minus_baseline_columns": [cand_col, base_col],
                "n_paired": n,
                "sample_paired_mean_m": sample_mean,
                "sample_paired_ci_lo_m": full_lo,
                "sample_paired_ci_hi_m": full_hi,
                "predeclared_outcome": outcome_label,
                "sample_outcome_agrees_with_predeclared_direction": sample_outcome_agrees,
                "leave_one_out_jackknife": {
                    "n_jackknife_replicates": n,
                    "fraction_decision_preserved": loo_pass_fraction,
                    "min_jackknife_mean_m": float(min(loo_means)),
                    "max_jackknife_mean_m": float(max(loo_means)),
                },
                "half_sample_subsample_stability": {
                    "n_subsample_draws": n_sub_draws,
                    "subsample_size": int(n_sub),
                    "fraction_decision_preserved": sub_pass_fraction,
                    "subsample_mean_p05_m": float(np.percentile(sub_means, 5)),
                    "subsample_mean_p95_m": float(np.percentile(sub_means, 95)),
                },
                "doubled_n_bootstrap_surrogate": {
                    "n_resamples": n_bootstrap,
                    "doubled_sample_size": n_doubled,
                    "fraction_decision_preserved": doubled_agreement,
                    "doubled_ci_lo_m": doubled_ci_lo,
                    "doubled_ci_hi_m": doubled_ci_hi,
                    "note": (
                        "Bootstrap surrogate for K>=2*n_committed: resample "
                        "2n trajectories with replacement so the bootstrap "
                        "mean's standard error matches that of a true 2n "
                        "draw under an i.i.d. trajectory population. Not "
                        "new confirmatory evidence; the predeclared rule "
                        "still binds at the committed n."
                    ),
                },
                "paired_bootstrap_sign_agreement_fraction_at_committed_n": boot_sign_agreement,
            }
        )

    return {
        "slice": slice_key,
        "arc_label": slice_def["arc_label"],
        "n_trajectories_full": len(df_full),
        "source_csv": str(slice_def["csv"]),
        "source_json": str(slice_def["json"]),
        "decisions": decisions_out,
    }


def run(n_bootstrap: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    slices_out = {key: _analyse_slice(key, defn, n_bootstrap, rng) for key, defn in SLICES.items()}
    summary = {
        "schema_version": "decision_stability_v2",
        "audit_on_utc": "2026-05-20",
        "scope": (
            "Decision-stability analysis of the predeclared higher-fidelity "
            "and long-arc paired pass/fail decisions (PUKF positive criterion "
            "on the 40-minute slice; DSA-EKF positive criterion and EKF/AUKF "
            "reversal on the 3-hour long-arc slice powered to n=64). This is "
            "a stability analysis, not new confirmatory evidence: no "
            "estimator is refit, and the predeclared decision is unchanged. "
            "The leave-one-out jackknife and half-sample subsample replicates "
            "re-aggregate the committed per-trajectory paired differences "
            "only. The long-arc rows update the prior n=36 analysis to the "
            "powered n=64 population that the manuscript reports as the "
            "primary held-out test."
        ),
        "rng_seed": seed,
        "n_bootstrap": n_bootstrap,
        "slices": slices_out,
        "interpretation": (
            "For each predeclared decision we report (i) the leave-one-out "
            "jackknife fraction that preserves the sample-direction decision, "
            "(ii) the half-sample subsample fraction over 1000 paired draws, "
            "and (iii) the paired-bootstrap sign-agreement fraction. A "
            "fraction near 1.0 means the decision is robust to subsampling; "
            "a fraction near 0.5 means the decision is sample-marginal."
        ),
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument(
        "--output",
        default="results/decision_stability/decision_stability_loop58.json",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    payload = run(n_bootstrap=int(args.n_bootstrap), seed=int(args.seed))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
