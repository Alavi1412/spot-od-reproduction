# Containerized Minimum-Tier Attestation - 2026-06-14

This artifact is a local Docker Desktop clean-container staged minimum-tier integrity attestation only. It is same-host and staged from the current workspace/review archive. It is not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, not independent external reproduction, and not a full scientific rerun.

## Archive Observed From Manifest

- Manifest: `release/SUPPLEMENTARY_MANIFEST.json`
- Archive: `release/spot_od_v1_1_0_supplement_review_archive.zip`
- Observed archive SHA-256 before this attestation was indexed:
  `f06b351e7c9dd6fd47066f45535f7db5e7ff254589a7bc68b32ea05575dbe4ad`
- Observed archive bytes before this attestation was indexed: `58073055`
- Observed indexed artifact coverage before this attestation was indexed: `968`

The regenerated `release/SUPPLEMENTARY_MANIFEST.json` remains authoritative for
the current review archive SHA-256. This attestation is itself indexed and
archived, so the pre-indexing archive digest is recorded as the observed
paired-manifest digest at attestation time rather than as a self-referential
fixed-point claim.

## Staged Container Check

- Runtime: Docker Desktop, Docker 27.3.1 (`desktop-linux` context)
- Local image: `python:3.11` (`69feacea6aaa`, 1.59GB)
- Staged inputs:
  `release/SUPPLEMENTARY_MANIFEST.json`,
  `release/spot_od_v1_1_0_supplement_review_archive.zip`, and
  `scripts/verify_minimum_tier_reproduction.py`
- Local staging path: intentionally redacted and represented as
  `<local-stage>`

Command template:

```text
docker run --rm -v <local-stage>:/work -w /work python:3.11 python scripts/verify_minimum_tier_reproduction.py --check-only
```

## Result

| Check | Status |
|---|---|
| Overall status | pass |
| Manifest summary check | pass |
| Artifact check | pass |
| Review archive check | pass |
| Claim-map check | pass |
| Requires retraining | false |

## Boundary

This is a local Docker Desktop clean-container staged minimum-tier integrity attestation only: same-host and staged from the current workspace/review archive; not independent-machine validation, not third-party reproduction, not DOI/public archive, not full scientific reproduction, not operational POD, not independent external reproduction, and not a full scientific rerun.
