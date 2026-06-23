# Artifacts

## Public Verification

After this repository is pushed and released, the authoritative public independent-machine signal for this package is the GitHub Actions workflow running on GitHub-hosted runners. The workflow reruns the v1.1.0 archive-extracted verifier, reruns the v1.2.1 GraphAnchorPairGate PoC archive verifier, writes a GitHub Actions attestation, and uploads verifier and attestation reports.

Version v1.2.1 supersedes the published v1.2.0 graph package, DOI `10.5281/zenodo.20810341`, by correcting embedded manuscript/table/provenance text. Zenodo will assign the v1.2.1 DOI after release.

## Release Files

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `release/spot_od_v1_1_0_supplement_review_archive.zip` | `58908807` | `9d6f34599b238749bfe1cc3e8bdda4d6a3034ee662f3e8b2f7c7cafc49831b3f` |
| `release/SUPPLEMENTARY_MANIFEST.json` | `354821` | `2d7a05dee73d83b436dcc88ebcd40f5d7caeaacbfc70ee5d170040474f99ff72` |
| `release/spot_od_v1_2_1_graph_anchor_gate_poc.zip` | `17710047` | `3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c` |
| `release/spot_od_v1_2_1_graph_anchor_gate_poc.zip::MANIFEST.json` | `54474` | `c71e7bfe9bc293589dcad6e37546eb01cec2638ff3d02245db1e983a55cae805` |
| `scripts/verify_archive_extracted_reproduction.py` | `72018` | `a2eac634a69ce4bf170734e4df5ef9c9294a5466d0be69eba1fa66e0fe3ececd` |
| `scripts/verify_graph_anchor_gate_poc.py` | `24638` | `2f1be3deeee02bbc8c18a31d71458fbbc9085cc01ded5335e3e7819408cd380d` |
| `.github/workflows/archive-extracted-reproduction.yml` | `2828` | `aee747bb34a16bde9f7afa728a203c5f29b590829d022be6014eb9c005ce37d0` |
| `.zenodo.json` | `2174` | `676b7618c7b2778cd235d0408926fff94d3c3321ff88068771472e600ed33ed4` |
| `CITATION.cff` | `1365` | `58f40850b757847709d85e457c9e971a9a88dd89edb74435e037813ece676169` |
| `release/spot_od_v1_2_0_graph_anchor_gate_poc.zip` | `17708072` | `17389963787273cea7628269476409dd99d379c34c0715c56539fa59ea5bb712` |
| `release/spot_od_v1_2_0_graph_anchor_gate_poc.zip::MANIFEST.json` | `53066` | `3cbe5b44fc0d39b1666384e8461d519ffcbe74969618c5c24044271c4a2c1862` |

## v1.1.0 Archive

- Archive members: `1014`
- Archive format: ZIP
- Archive member timestamp recorded by the paired manifest: `2026-05-19T00:00:00+00:00`

The paired v1.1.0 manifest is not embedded inside the ZIP. The verifier treats it as an allowed release-level record to avoid a self-referential archive digest cycle.

## v1.2.1 Graph PoC Archive

- Archive file members: `129`
- Manifest-indexed payload artifacts: `128`
- Archive format: ZIP
- Embedded manifest: `MANIFEST.json`
- v1.2.1 DOI: not known yet; Zenodo will assign it after release.

The embedded `MANIFEST.json` indexes every payload member except itself. The manifest records its own non-self-indexing policy because embedding a digest of itself would create a self-referential digest cycle. The graph verifier still reports the embedded manifest bytes and SHA-256 after extraction.

The v1.2.1 archive contains the GraphAnchorPairGate PoC code, focused tests, direct code dependencies, retained CSV/JSON/PNG evidence, selected synchronized paper artifacts, and best GraphAnchorPairGate checkpoints. Prediction NPZ arrays, sweep logs, PID files, and last checkpoints are intentionally excluded to keep the public package compact.

## Historical v1.2.0 Graph PoC Archive

The v1.2.0 graph package remains intact as a historical published package at DOI `10.5281/zenodo.20810341`. Version v1.2.1 supersedes it only because the v1.2.0 ZIP embedded pre-sync manuscript/table text with stale pending-release wording.
