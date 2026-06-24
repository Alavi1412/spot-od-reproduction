# SPOT-OD v1.2.2 ACF audit release notes

Release tag: `v1.2.2-acf-audit`

GitHub release URL:
<https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.2-acf-audit>

GitHub release commit: `6fbc88745b6d96939736d59731089e99786c1f8c`

Zenodo software record:
<https://zenodo.org/records/20822968>

Zenodo DOI: `10.5281/zenodo.20822968`

DOI URL:
<https://doi.org/10.5281/zenodo.20822968>

Zenodo archived file: `Alavi1412/spot-od-reproduction-v1.2.2-acf-audit.zip`

Zenodo archived file bytes: `72,607,548`

Zenodo archived file MD5: `533b8363954cb6531f17bf4d405a5092`

GitHub release asset: `spot_od_v1_2_2_acf_audit_review_archive.zip`

GitHub release asset bytes: `59,127,034`

GitHub release asset SHA-256:
`e6b6139bb0fb5463f5091bdde05e14b82a8191d1419466cdd21c8aafa533b240`

Prior release DOI: v1.2.1 remains archived at `10.5281/zenodo.20811701`.

## What Changed

This release supersedes v1.2.1 only by adding the AdaptiveCandidateFusion audit
and table tier plus matching manuscript claim-boundary wording. It does not
upgrade the scientific metrics to operational validation.

Included ACF audit artifacts:

- `paper/tables/adaptive_candidate_fusion_full_training_poc.tex`
- `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.json`
- `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_summary.md`
- `results/adaptive_candidate_fusion_fixed_soft_training_campaigns_20260623/adaptive_candidate_fusion_fixed_soft_training_campaign_rows.csv`
- `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.json`
- `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.md`
- `results/adaptive_candidate_fusion_global_scenario_portfolio_15seed_20260624/summary.csv`
- Generator/test support in `scripts/build_paper_assets.py` and focused ACF tests.

## Claim Boundary

ACF remains validation-selected compact-simulator PoC evidence. It is not
operational precise-reference validation, independent-machine reproduction,
third-party validation, full raw/training/all-filter reproduction, or universal
learned-OD superiority.
