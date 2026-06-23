# SPOT-OD Reproduction And GraphAnchorPairGate PoC

[![Archive-extracted reproduction](https://github.com/Alavi1412/spot-od-reproduction/actions/workflows/archive-extracted-reproduction.yml/badge.svg)](https://github.com/Alavi1412/spot-od-reproduction/actions/workflows/archive-extracted-reproduction.yml)

This repository publishes two bounded public release artifacts for SPOT-OD:

- the unchanged v1.1.0 archive-extracted reproduction package;
- the v1.2.0 GraphAnchorPairGate proof-of-concept package.

## What This Repository Proves

The v1.1.0 verifier checks that:

- `release/spot_od_v1_1_0_supplement_review_archive.zip` extracts successfully;
- the paired `release/SUPPLEMENTARY_MANIFEST.json` indexes 1014 archive members;
- extracted members match manifest byte sizes and SHA-256 digests;
- claim-to-artifact and regeneration-tier records resolve against the extracted archive;
- active main-manuscript table/figure artifacts regenerate from archived result artifacts;
- one public LAGEOS CRD/SP3 precise-reference OD slice recomputes from archived public inputs and matches the archived public-claim summary/table within verifier tolerances.

The v1.2.0 graph verifier checks that:

- `release/spot_od_v1_2_0_graph_anchor_gate_poc.zip` extracts successfully;
- the embedded `MANIFEST.json` covers all payload members and their SHA-256 digests;
- retained seed-sweep CSVs recompute the GraphAnchorPairGate headline metrics without training;
- the retained metrics are 10 scenario-seed rows, 9 row wins, 4/5 paired seeds winning both scenarios, process-shift mean gain 7.95663495038935%, maneuver-shift mean gain 8.05274642630686%, and the seed-19 process-shift failure -2.0925251807980216%.

After this repository is pushed and released, the authoritative public independent-machine signal for this package is the GitHub Actions workflow running these verifiers on GitHub-hosted runners.

## Boundaries

The v1.1.0 archive-extracted reproduction tier is not a full raw data generation rerun, model training rerun, hyperparameter search rerun, all-filter/all-table pipeline rerun, live public-data retrieval, operational precise orbit determination validation, or third-party independent validation.

The v1.2.0 GraphAnchorPairGate PoC is local compact-simulator evidence. It uses all-step center-window position RMSE on held-out eval trajectories in `process_noise_shift_test` and `maneuver_shift_test`. It is not the primary observed-step endpoint, not operational precise-reference validation, not independent third-party reproduction, and not a full raw-data/training rerun.

## Quickstart

Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_archive_extracted_reproduction.py \
  --archive release/spot_od_v1_1_0_supplement_review_archive.zip \
  --json-out results/validation/github_actions_archive_extracted_reproduction.json \
  --md-out results/validation/github_actions_archive_extracted_reproduction.md
python scripts/verify_graph_anchor_gate_poc.py \
  --archive release/spot_od_v1_2_0_graph_anchor_gate_poc.zip \
  --json-out results/validation/github_actions_graph_anchor_gate_poc_verification.json \
  --md-out results/validation/github_actions_graph_anchor_gate_poc_verification.md
python scripts/write_github_actions_attestation.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_archive_extracted_reproduction.py --archive release\spot_od_v1_1_0_supplement_review_archive.zip --json-out results\validation\github_actions_archive_extracted_reproduction.json --md-out results\validation\github_actions_archive_extracted_reproduction.md
python scripts\verify_graph_anchor_gate_poc.py --archive release\spot_od_v1_2_0_graph_anchor_gate_poc.zip --json-out results\validation\github_actions_graph_anchor_gate_poc_verification.json --md-out results\validation\github_actions_graph_anchor_gate_poc_verification.md
python scripts\write_github_actions_attestation.py
```

## Primary Artifacts

- v1.1.0 archive: `release/spot_od_v1_1_0_supplement_review_archive.zip`
- v1.1.0 archive SHA-256: `9d6f34599b238749bfe1cc3e8bdda4d6a3034ee662f3e8b2f7c7cafc49831b3f`
- v1.1.0 archive bytes: `58908807`
- v1.1.0 archive members: `1014`
- v1.1.0 paired manifest: `release/SUPPLEMENTARY_MANIFEST.json`
- v1.1.0 manifest SHA-256: `2d7a05dee73d83b436dcc88ebcd40f5d7caeaacbfc70ee5d170040474f99ff72`
- v1.2.0 graph PoC archive: `release/spot_od_v1_2_0_graph_anchor_gate_poc.zip`
- v1.2.0 graph PoC archive SHA-256: `17389963787273cea7628269476409dd99d379c34c0715c56539fa59ea5bb712`
- v1.2.0 graph PoC archive bytes: `17708072`
- v1.2.0 graph PoC archive file members: `129`
- v1.2.0 graph PoC manifest-indexed payload artifacts: `128`
- v1.2.0 embedded manifest SHA-256: `3cbe5b44fc0d39b1666384e8461d519ffcbe74969618c5c24044271c4a2c1862`

See `docs/REPRODUCTION_BOUNDARY.md` for the scope boundary and `docs/ARTIFACTS.md` for hashes.
