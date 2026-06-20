# Artifacts

## Public Verification

After this repository is pushed and released, the authoritative public independent-machine signal for this package is the GitHub Actions workflow running on GitHub-hosted runners. The workflow reruns the archive-extracted verifier, writes a GitHub Actions attestation, and uploads the verifier and attestation reports.

## Release Files

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `release/spot_od_v1_1_0_supplement_review_archive.zip` | `58908807` | `9d6f34599b238749bfe1cc3e8bdda4d6a3034ee662f3e8b2f7c7cafc49831b3f` |
| `release/SUPPLEMENTARY_MANIFEST.json` | `354821` | `2d7a05dee73d83b436dcc88ebcd40f5d7caeaacbfc70ee5d170040474f99ff72` |
| `scripts/verify_archive_extracted_reproduction.py` | `72018` | `a2eac634a69ce4bf170734e4df5ef9c9294a5466d0be69eba1fa66e0fe3ececd` |

## Archive

- Archive members: `1014`
- Archive format: ZIP
- Archive member timestamp recorded by the paired manifest: `2026-05-19T00:00:00+00:00`

The paired manifest is not embedded inside the ZIP. The verifier treats it as an allowed release-level record to avoid a self-referential archive digest cycle.
