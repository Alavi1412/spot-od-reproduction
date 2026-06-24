# SPOT-OD v1.2.5 Zenodo metadata repair release notes

Release tag: `v1.2.5-zenodo-metadata-repair`

GitHub release URL:
<https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.5-zenodo-metadata-repair>

Zenodo concept DOI: `10.5281/zenodo.20768672`

## Purpose

This is a metadata-only follow-up release to repair the GitHub-to-Zenodo import
metadata after the v1.2.4 Zenodo record inherited stale
GraphAnchorPairGate/v1.2.1 title, version, and description text.

This release adds no new scientific results, experiments, assets, model runs,
or evidence upgrades.

## v1.2.4 GitHub release facts

The GitHub release for `v1.2.4-acf-prospective-guard` is coherent:

- URL: <https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.4-acf-prospective-guard>
- Asset: `spot_od_v1_2_4_acf_prospective_guard_artifact.zip`
- Size: 278,230,689 bytes
- SHA-256: `FB91EB9179D2F0AF236B70A31D4DC10AD0BDECD8C9F94233DAD25E81C71C9D52`

The prospective guard failed: 4/10 scenario-row wins plus 1 tie, mean
observed-step gain -6.366%, row bootstrap CI [-18.521,+3.542]%, 2/5 positive
seed-paired means, and seed-paired CI [-15.306,+2.379]%.

## v1.2.4 Zenodo metadata mismatch

Zenodo record `20836876` / DOI `10.5281/zenodo.20836876` points to the
v1.2.4 GitHub release and file, but its imported title, version, and
description text remained stale GraphAnchorPairGate/v1.2.1 metadata.

This v1.2.5 release exists to provide corrected Zenodo metadata for a follow-up
GitHub/Zenodo release without changing the v1.2.4 scientific record.

## Current clean public boundary

`v1.2.3-acf-holdout-audit`, DOI `10.5281/zenodo.20825138`, remains the current
clean citable public access and integrity boundary until a corrected v1.2.5
Zenodo DOI is verified.

## Scientific boundary

This metadata repair does not upgrade any evidence to operational
precise-reference validation, independent-machine reproduction, third-party
validation, full raw/training/all-filter rerun, learned-superiority evidence,
confirmatory learned-superiority evidence, or operational readiness.
