# Independent-Machine Reproduction Request

This file is a handoff template for a third-party operator. This is not a completed independent reproduction, not third-party validation, and not scored external validation.

## Objective

Produce reviewer-credible evidence that the submitted supplementary evidence
package can be verified from a clean machine using the supplied manifest,
review archive, and verifier scripts. The expected deliverable is a signed
operator report with machine identity, command transcripts, hashes, pass/fail
status, deviations, and output hashes.

## Public Release Reference

- Short title: SPOT-OD v1.3.1 edge-only residual-refinement release sync
- Zenodo record: pending GitHub release creation and Zenodo import
- DOI: pending GitHub release creation and Zenodo import
- DOI URL: pending GitHub release creation and Zenodo import
- Zenodo concept DOI: 10.5281/zenodo.20768672
- Prior clean Zenodo version DOI: 10.5281/zenodo.20842573
- Zenodo status: pending_github_release_zenodo_import
- Resource type: Software
- GitHub repository: https://github.com/Alavi1412/spot-od-reproduction
- GitHub release:
  https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.1-edge-only-residual-refine-sync
- Release tag: v1.3.1-edge-only-residual-refine-sync
- Release commit: pending final GitHub tag target at release creation
- Supersession note: v1.3.1 is a forward sync/correction release that preserves
  the v1.3.0 train-loss edge-only retained-candidate residual-refinement
  result. It does not assert a v1.3.1 Zenodo DOI/record
  before GitHub release creation and Zenodo import. Scientific metrics are as
  recorded, not upgraded to operational validation.
- Zenodo archived file: pending GitHub release creation and Zenodo import
- Zenodo archived file bytes: pending
- Zenodo archived file MD5: pending
- GitHub release asset: pending
- GitHub release asset bytes: pending
- GitHub release asset SHA-256: pending

Historical v1.3.0 edge-only residual-refinement package reference:

- Zenodo record: https://zenodo.org/records/20842573
- DOI: 10.5281/zenodo.20842573
- DOI URL: https://doi.org/10.5281/zenodo.20842573
- Zenodo status: published
- GitHub release:
  https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.3.0-edge-only-residual-refine
- Release tag: v1.3.0-edge-only-residual-refine
- Release commit: 9faa5b3

Historical v1.2.3 ACF holdout audit package reference:

- Zenodo record: https://zenodo.org/records/20825138
- DOI: 10.5281/zenodo.20825138
- DOI URL: https://doi.org/10.5281/zenodo.20825138
- Zenodo status: published
- GitHub release:
  https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.3-acf-holdout-audit
- Release tag: v1.2.3-acf-holdout-audit
- Release commit: 39e879d8665e489266bbf75f69634cab0e797fe8
- Zenodo archived file: Alavi1412/spot-od-reproduction-v1.2.3-acf-holdout-audit.zip
- Zenodo archived file bytes: 187,254,529
- Zenodo archived file MD5: 7eb8b43a9af90a4783482a7a3a086f92
- GitHub release asset: spot_od_v1_2_3_acf_holdout_audit_review_archive.zip
- GitHub release asset bytes: 59,140,917
- GitHub release asset SHA-256:
  11909866b2ae1a375cdebe1472305d6d1fbd0b9f97453084fc7da16b78dcf70f

Historical v1.2.2 ACF audit package reference:

- Zenodo record: https://zenodo.org/records/20822968
- DOI: 10.5281/zenodo.20822968
- DOI URL: https://doi.org/10.5281/zenodo.20822968
- Zenodo status: published
- GitHub release:
  https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.2-acf-audit
- Release tag: v1.2.2-acf-audit
- Release commit: 6fbc88745b6d96939736d59731089e99786c1f8c
- Zenodo archived file: Alavi1412/spot-od-reproduction-v1.2.2-acf-audit.zip
- Zenodo archived file bytes: 72,607,548
- Zenodo archived file MD5: 533b8363954cb6531f17bf4d405a5092
- GitHub release asset: spot_od_v1_2_2_acf_audit_review_archive.zip
- GitHub release asset bytes: 59,127,034
- GitHub release asset SHA-256:
  e6b6139bb0fb5463f5091bdde05e14b82a8191d1419466cdd21c8aafa533b240

Prior v1.2.1 public archive reference:

- Zenodo record: https://zenodo.org/records/20811701
- DOI: 10.5281/zenodo.20811701
- DOI URL: https://doi.org/10.5281/zenodo.20811701
- Zenodo status: published
- GitHub release:
  https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.1-graph-anchor-gate-poc
- Release tag: v1.2.1-graph-anchor-gate-poc
- Release commit: 2dcd542dcb72f1622dfaf1cf8981a550862312bf
- Zenodo archived file:
  Alavi1412/spot-od-reproduction-v1.2.1-graph-anchor-gate-poc.zip
- Zenodo archived file bytes: 94,265,950
- Zenodo archived file MD5:
  233d2fc7fce1bc57afdd66332a3a7dc1
- GitHub release asset: spot_od_v1_2_1_graph_anchor_gate_poc.zip
- GitHub release asset bytes: 17,710,047
- GitHub release asset SHA-256:
  3cc285f132b690695a5d2a453f7c21128b46333d183fcfca265c52d50184c69c

Historical v1.2.3 GitHub Actions verifier runs are recorded at:

- Branch run:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253565
- Tag run:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28082253538
- Status: success

Historical v1.2.2 GitHub Actions verifier runs are recorded at:

- Branch run:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28075721074
- Tag run:
  https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28075722522

The prior v1.2.1 release-triggered GitHub Actions verifier passed at
https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357
and ran the archive-extracted reproduction workflow and graph verifier on
GitHub-hosted runners. These runs are maintainer-run platform evidence for the
release verifier tiers; they are not third-party independent validation.

## Artifact Pairing Rule

Choose exactly one verification route and keep every input from that same
route. Historical published release assets and historical local packets must
not be swapped across version boundaries.

- Current v1.3.1 release route: the archive/source asset is pending GitHub
  release creation and Zenodo import. Wait for the route-paired v1.3.1 asset
  before running a public archive route. Do not invent or substitute a v1.3.1
  DOI/record before import, and do not use a historical v1.2.3 archive as a
  current v1.3.1 input.
- Historical immutable published v1.2.3 GitHub/Zenodo release route: use the files,
  manifest, verifier scripts, dependency files, and review archive from the
  exact `v1.2.3-acf-holdout-audit` release/tag/asset set together. The
  published GitHub release asset is
  `spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`, 59,140,917 bytes,
  SHA-256
  `11909866b2ae1a375cdebe1472305d6d1fbd0b9f97453084fc7da16b78dcf70f`.
  Do not pair this immutable asset with later manifests.
- Historical v1.2.3 DOI-synced working-branch/local-packet route: use the
  historical
  `release/SUPPLEMENTARY_MANIFEST.json` with the regenerated local archive at
  `release/spot_od_v1_2_3_acf_holdout_audit_review_archive.zip`, 59,142,123
  bytes, SHA-256
  `11451c2032243c972534f7de9eb40ba04c44ff69b6c45db179f2053f97ad9b7e`.
  That manifest recorded
  `review_archive.matches_published_github_release_asset: false`; do not
  substitute the published GitHub release asset for this route.

## Inputs To Obtain

First select one route from the Artifact Pairing Rule above. Obtain all inputs
from that same selected route:

- `release/SUPPLEMENTARY_MANIFEST.json`
- the paired review archive:
  `release/<route-paired-review-archive.zip>`
- `scripts/verify_minimum_tier_reproduction.py`
- `scripts/verify_archive_extracted_reproduction.py`
- `requirements.txt` and `pyproject.toml` from the same archive, tag checkout,
  or local packet

For the v1.3.1 route, the archive/source asset is pending GitHub release
creation and Zenodo import. Replace the placeholder with the v1.3.1 archive
name only after the GitHub release asset and Zenodo import exist. Until then,
the v1.3.1 DOI/record and paired archive fields are pending.

## Clean-Machine Procedure

1. Create a new empty working directory on a machine not used to produce the
   submitted artifacts.
2. Copy the route-paired `release/SUPPLEMENTARY_MANIFEST.json` and
   `release/<route-paired-review-archive.zip>` into a `release/` subdirectory.
   Do not mix the immutable published GitHub asset with a later manifest, or a
   regenerated local archive with an older release/tag manifest.
3. Extract the archive into the working directory:

```powershell
python -m zipfile -e release/<route-paired-review-archive.zip> .
```

4. Confirm the paired manifest and archive hashes against the selected route:

```powershell
python -c "import hashlib, pathlib; paths=['release/SUPPLEMENTARY_MANIFEST.json','release/<route-paired-review-archive.zip>']; [print(p, hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()) for p in paths]"
```

5. Create and activate a local Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

6. Run the minimum integrity verifier:

```powershell
.\.venv\Scripts\python -I scripts/verify_minimum_tier_reproduction.py --json-out results/validation/independent_minimum_tier_reproduction.json --md-out results/validation/independent_minimum_tier_reproduction.md
```

7. Run the archive-extracted verifier:

```powershell
.\.venv\Scripts\python -I scripts/verify_archive_extracted_reproduction.py --archive release/<route-paired-review-archive.zip> --json-out results/validation/independent_archive_extracted_reproduction.json --md-out results/validation/independent_archive_extracted_reproduction.md
```

8. Hash the generated reports:

```powershell
python -c "import hashlib, pathlib; paths=['results/validation/independent_minimum_tier_reproduction.json','results/validation/independent_minimum_tier_reproduction.md','results/validation/independent_archive_extracted_reproduction.json','results/validation/independent_archive_extracted_reproduction.md']; [print(p, hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()) for p in paths if pathlib.Path(p).is_file()]"
```

If the operator uses Linux or macOS, use the same arguments with
`.venv/bin/python` instead of `.\.venv\Scripts\python`.

## Manual External Operator Route

An external operator may start from the immutable Zenodo/GitHub release set or
from the current DOI-synced local packet, but must keep the manifest, archive,
verifier scripts, and dependency files from one selected route. The operator
should verify the selected route's hashes, run the same clean-machine
procedure, and attach the generated JSON/Markdown reports and their SHA-256
hashes to the signed report. A signed report supports only the verifier scope
actually run; it does not expand the manuscript claims.

## Report Template

- Operator name and affiliation:
- Machine owner and location:
- Machine identifier or inventory tag:
- OS name/version:
- CPU/RAM:
- Python version:
- Reproduction date/time UTC:
- Archive SHA-256:
- Manifest SHA-256:
- Git commit or release label, if available:
- Commands run, exactly as executed:
- Minimum-tier verifier pass/fail:
- Archive-extracted verifier pass/fail:
- Generated report paths and SHA-256 hashes:
- Output hash differences from submitted reports, if any:
- Public source used: Zenodo/GitHub/local transfer, with URL or transfer note:
- GitHub Actions verifier URL cited, if used as contextual release evidence:
- Deviations from the procedure:
- Network access used: yes/no, with reason:
- Local dependency changes or package-resolution notes:
- Failure logs or stderr excerpts, if any:
- Operator conclusion:
- Operator signature:
- Date:

## Interpretation Boundary

A passing report from an independent machine would support the release-package
reproduction claim for the verifier scope actually run. It would still not be
operational POD validation, not a fresh full scientific rerun unless the full
rerun tier is separately executed, not live public-data retrieval, and not
third-party validation of claims outside the executed verifier scope.
