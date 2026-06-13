# Minimum-Tier Reproduction Check

Status: **PASS**

This check validates the release manifest inventory, recorded SHA-256
digests, review archive membership, and claim-to-artifact map coverage.
It does not rerun estimator training, table generation, or filter
estimation.

## Checks
- Manifest artifact count summary: **PASS**
- Manifest artifact presence and checksums: **PASS**
- Review archive digest and member checksums: **PASS**
- Claim-map and regeneration-tier coverage: **PASS**

## Outputs
- JSON: `results/validation/minimum_tier_reproduction_check.json`
- Markdown: `results/validation/minimum_tier_reproduction_check.md`
