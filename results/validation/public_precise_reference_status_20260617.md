# Public Precise-Reference Status (2026-06-17)

Generated UTC: 2026-06-17T12:54:45.9236175Z

## Live EDC Listing Recheck

- LAGEOS-1 parent listing: `https://edc.dgfi.tum.de/pub/slr/products/orbits/lageos1/`; has 260613=False; has 260620=False.
- LAGEOS-2 parent listing: `https://edc.dgfi.tum.de/pub/slr/products/orbits/lageos2/`; has 260613=False; has 260620=False.

## Direct NSGF URL Checks

| Target | Week | URL result | Interpretation |
|---|---:|---|---|
| LAGEOS-1 | 260613 | HTTP 200, `text/html; charset=utf-8`, length 825 | Not a gzip/SP3 product |
| LAGEOS-2 | 260613 | HTTP 200, `text/html; charset=utf-8`, length 825 | Not a gzip/SP3 product |
| LAGEOS-1 | 260620 | HTTP 200, `text/html; charset=utf-8`, length 825 | Not a gzip/SP3 product |
| LAGEOS-2 | 260620 | HTTP 200, `text/html; charset=utf-8`, length 825 | Not a gzip/SP3 product |

## Prospective Public Week 260620

- Predeclaration: `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json`
- Predeclaration SHA-256: `76f784dbefe9d251e707bc524abf602eda659e21d18738ebfd728cbaf90d87ca`
- Timestamp attestation: `results/validation/real_slr_sp3_temporal_corrected_od_prospective_260620_timestamp_attestation.json`
- Timestamp proof status: Bitcoin-block-header-attested OpenTimestamps proof.
- Availability status: can_score_260620_now=False; pending/not scored.

## Interpretation

The 260620 OpenTimestamps proof is now Bitcoin-block-header-attested, but this timestamp proof is not scored validation and does not change public precise-reference availability.

As of the 2026-06-17 live check, public 260613/260620 SP3 products remain unavailable for the checked LAGEOS targets. The 260620 prospective campaign remains pending/not scored and must not be represented as scored validation, DOI/public archive evidence, independent reproduction, or operational POD.
