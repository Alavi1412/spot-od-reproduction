# Codex Clean Archive Attestation - 2026-06-14

This artifact is a same-workspace separate-agent clean extracted archive attestation only. It records commands run by a Codex implementation worker in the current workspace/machine against the current review archive and materialized artifacts. It is not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, and not independent external reproduction.

## Archive Observed From Manifest

- Manifest: `release/SUPPLEMENTARY_MANIFEST.json`
- Archive: `release/spot_od_v1_1_0_supplement_review_archive.zip`
- Observed archive SHA-256 before this attestation was indexed:
  `e856f70f4d7d8c8f79a42fda7e73481d8d23652a3f645f904d8c4cd92c6b2ca3`
- Observed archive bytes before this attestation was indexed: `58056994`
- Observed indexed artifact coverage before this attestation was indexed: `964`

The regenerated `release/SUPPLEMENTARY_MANIFEST.json` remains authoritative for
the current review archive SHA-256. This attestation is itself indexed and
archived, so the pre-indexing archive digest is recorded as the observed
paired-manifest digest at attestation time rather than as a self-referential
fixed-point claim.

## Commands Run

| Command | Exit code | Status | Notes |
|---|---:|---|---|
| `python scripts/validate_submission.py` | 0 | pass | Refreshed `results/validation/submission_validation.json`; overall pass, 34 pages, citation pass, zero main/supplement LaTeX warnings. |
| `python scripts/verify_minimum_tier_reproduction.py` | 1 | fail | Claim map, manifest summary, and review archive checks passed, but the workspace artifact check found one pre-indexing SHA-256 mismatch for `results/validation/supplement_layout_warnings.md`. |
| `python scripts/verify_archive_extracted_reproduction.py` | 0 | pass | Extracted artifact digests, claim map, regeneration tiers, active table regeneration, and archive-extracted public OD slice rerun passed. |
| `python scripts/verify_minimum_tier_reproduction.py --check-only` | 0 | pass | Final post-regeneration no-write check passed after indexing the attestation and rebuilding `release/SUPPLEMENTARY_MANIFEST.json` plus the review archive. |

## Nested Outputs

- `submission_validation`: status `pass`, overall pass `true`, page count `34`,
  citation status `pass`, main LaTeX warning count `0`, supplement LaTeX
  warning count `0`.
- `minimum_tier_reproduction`: status `fail` before this attestation was
  indexed; artifact check `fail`; claim map `pass`; manifest summary `pass`;
  review archive `pass`; one workspace-file digest mismatch for
  `results/validation/supplement_layout_warnings.md`. The final
  post-regeneration no-write minimum-tier check passed.
- `archive_extracted_reproduction`: status `pass`; extracted artifacts `pass`;
  claim map `pass`; regeneration tiers `pass`; active table regeneration
  nested report `pass` with 10 artifacts, 10 pass, 0 mismatch, 0 blockers.
- `archive_extracted_public_od_slice_rerun`: status `pass`; 10 completed arcs;
  DBAR 6/10; zero public-claim mismatches; generated table text matched the
  submitted table. Pooled held-out position RMSE means: AUKF 341.33 m, EKF
  367.98 m, SP3-IC propagation 402.92 m, fixed-noise UKF 334.72 m.

## Boundary

This artifact is a same-workspace separate-agent clean extracted archive attestation only: not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, and not independent external reproduction.
