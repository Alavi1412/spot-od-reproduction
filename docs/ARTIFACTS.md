# Artifacts

## Public Verification

After this repository is pushed and released, the authoritative public independent-machine signal for this package is the GitHub Actions workflow running on GitHub-hosted runners. The workflow reruns the v1.1.0 archive-extracted verifier, reruns the v1.2.0 GraphAnchorPairGate PoC archive verifier, writes a GitHub Actions attestation, and uploads verifier and attestation reports.

## Release Files

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `release/spot_od_v1_1_0_supplement_review_archive.zip` | `58908807` | `9d6f34599b238749bfe1cc3e8bdda4d6a3034ee662f3e8b2f7c7cafc49831b3f` |
| `release/SUPPLEMENTARY_MANIFEST.json` | `354821` | `2d7a05dee73d83b436dcc88ebcd40f5d7caeaacbfc70ee5d170040474f99ff72` |
| `release/spot_od_v1_2_0_graph_anchor_gate_poc.zip` | `17708072` | `17389963787273cea7628269476409dd99d379c34c0715c56539fa59ea5bb712` |
| `release/spot_od_v1_2_0_graph_anchor_gate_poc.zip::MANIFEST.json` | `53066` | `3cbe5b44fc0d39b1666384e8461d519ffcbe74969618c5c24044271c4a2c1862` |
| `scripts/verify_archive_extracted_reproduction.py` | `72018` | `a2eac634a69ce4bf170734e4df5ef9c9294a5466d0be69eba1fa66e0fe3ececd` |
| `scripts/verify_graph_anchor_gate_poc.py` | `24628` | `60bb22aa8dd71ec5de757aa1b89e1fe258c2eb11761a7d40dd153860e3b1478e` |
| `.github/workflows/archive-extracted-reproduction.yml` | `2808` | `f8a1397379dc222513b8b9b328761638fd226617556f4a66ec5c9caea0ad83a1` |
| `.zenodo.json` | `1802` | `d808b5ee7704386c39f54ed5000b0a102cd453706899cb5cc5f79e7a2ee5b297` |
| `CITATION.cff` | `1149` | `2e078f153fbf14c2aba27f5d431ad62c9d4fa01d8752c8292a59a217b97a0f03` |

## v1.1.0 Archive

- Archive members: `1014`
- Archive format: ZIP
- Archive member timestamp recorded by the paired manifest: `2026-05-19T00:00:00+00:00`

The paired v1.1.0 manifest is not embedded inside the ZIP. The verifier treats it as an allowed release-level record to avoid a self-referential archive digest cycle.

## v1.2.0 Graph PoC Archive

- Archive file members: `129`
- Manifest-indexed payload artifacts: `128`
- Archive format: ZIP
- Embedded manifest: `MANIFEST.json`

The embedded `MANIFEST.json` indexes every payload member except itself. The manifest records its own non-self-indexing policy because embedding a digest of itself would create a self-referential digest cycle. The graph verifier still reports the embedded manifest bytes and SHA-256 after extraction.

The v1.2.0 archive contains the GraphAnchorPairGate PoC code, focused tests, direct code dependencies, retained CSV/JSON/PNG evidence, selected paper artifacts, and best GraphAnchorPairGate checkpoints. Prediction NPZ arrays, sweep logs, PID files, and last checkpoints are intentionally excluded to keep the public package compact.
