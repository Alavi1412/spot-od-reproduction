# SPOT-OD Archive-Extracted Reproduction

[![Archive-extracted reproduction](https://github.com/Alavi1412/spot-od-reproduction/actions/workflows/archive-extracted-reproduction.yml/badge.svg)](https://github.com/Alavi1412/spot-od-reproduction/actions/workflows/archive-extracted-reproduction.yml)

This repository publishes the SPOT-OD supplementary review archive and a narrow verifier for the archive-extracted reproduction tier.

## What This Repository Proves

Running the verifier checks that:

- `release/spot_od_v1_1_0_supplement_review_archive.zip` extracts successfully.
- The paired `release/SUPPLEMENTARY_MANIFEST.json` indexes 1014 archive members.
- Extracted members match manifest byte sizes and SHA-256 digests.
- Claim-to-artifact and regeneration-tier records resolve against the extracted archive.
- Active main-manuscript table/figure artifacts regenerate from archived result artifacts.
- One public LAGEOS CRD/SP3 precise-reference OD slice recomputes from archived public inputs and matches the archived public-claim summary/table within the verifier tolerances.

After this repository is pushed and released, the authoritative public independent-machine signal for this package is the GitHub Actions workflow running on GitHub-hosted runners.

## What This Repository Does Not Prove

This is an archive-extracted reproduction package only. It does not claim:

- a full raw data generation rerun;
- model training or hyperparameter search reruns;
- an all-filter or all-table pipeline rerun;
- live public-data retrieval from current external services;
- operational precise orbit determination validation;
- third-party independent validation.

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
python scripts/write_github_actions_attestation.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts\verify_archive_extracted_reproduction.py --archive release\spot_od_v1_1_0_supplement_review_archive.zip --json-out results\validation\github_actions_archive_extracted_reproduction.json --md-out results\validation\github_actions_archive_extracted_reproduction.md
python scripts\write_github_actions_attestation.py
```

The GitHub Actions workflow runs the same verifier on GitHub-hosted runners, writes an attestation, uploads JSON/Markdown reports, and fails the job if the verifier or attestation fails.

## Primary Artifacts

- Archive: `release/spot_od_v1_1_0_supplement_review_archive.zip`
- Archive SHA-256: `9d6f34599b238749bfe1cc3e8bdda4d6a3034ee662f3e8b2f7c7cafc49831b3f`
- Archive bytes: `58908807`
- Archive members: `1014`
- Paired manifest: `release/SUPPLEMENTARY_MANIFEST.json`
- Manifest SHA-256: `2d7a05dee73d83b436dcc88ebcd40f5d7caeaacbfc70ee5d170040474f99ff72`

See `docs/REPRODUCTION_BOUNDARY.md` for the scope boundary and `docs/ARTIFACTS.md` for hashes.
