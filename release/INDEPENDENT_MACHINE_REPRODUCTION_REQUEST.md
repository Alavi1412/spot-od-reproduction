# Independent-Machine Reproduction Request

This file is a handoff template for a third-party operator. This is not a completed independent reproduction, not third-party validation, not a minted v1.2.2 DOI or completed public archive record, and not scored external validation.

## Objective

Produce reviewer-credible evidence that the submitted supplementary evidence
package can be verified from a clean machine using the supplied manifest,
review archive, and verifier scripts. The expected deliverable is a signed
operator report with machine identity, command transcripts, hashes, pass/fail
status, deviations, and output hashes.

## Public Release Reference

- Short title: SPOT-OD v1.2.2 ACF audit release
- Zenodo record: pending
- DOI: pending; fill after Zenodo mints the v1.2.2 DOI
- DOI URL: pending
- Zenodo status: pending publication
- GitHub repository: https://github.com/Alavi1412/spot-od-reproduction
- GitHub release:
  https://github.com/Alavi1412/spot-od-reproduction/releases/tag/v1.2.2-acf-audit
- Release tag: v1.2.2-acf-audit
- Release commit: pending until the release tag is created
- Supersession note: v1.2.2 supersedes v1.2.1 only by adding the ACF
  audit/table tier and manuscript claim-boundary wording. Scientific metrics
  are as recorded, not upgraded to operational validation.
- Zenodo archived file: pending
- Zenodo archived file bytes: pending
- Zenodo archived file MD5: pending
- GitHub release asset: pending
- GitHub release asset bytes: pending
- GitHub release asset SHA-256: pending

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

No v1.2.2 release-triggered verifier run exists until the GitHub release is
created. The prior v1.2.1 release-triggered GitHub Actions verifier passed at
https://github.com/Alavi1412/spot-od-reproduction/actions/runs/28018952357
and ran the archive-extracted reproduction workflow and graph verifier on
GitHub-hosted runners. That run is maintainer-run platform evidence
for the v1.2.1 release verifier tiers; it is not third-party independent
validation.

## Inputs To Obtain

- `release/SUPPLEMENTARY_MANIFEST.json`
- `release/spot_od_v1_1_0_supplement_review_archive.zip`
- `scripts/verify_minimum_tier_reproduction.py`
- `scripts/verify_archive_extracted_reproduction.py`
- `requirements.txt` and `pyproject.toml` from the archive or release packet

## Clean-Machine Procedure

1. Create a new empty working directory on a machine not used to produce the
   submitted artifacts.
2. Copy `release/SUPPLEMENTARY_MANIFEST.json` and
   `release/spot_od_v1_1_0_supplement_review_archive.zip` into a `release/`
   subdirectory.
3. Extract the archive into the working directory:

```powershell
python -m zipfile -e release/spot_od_v1_1_0_supplement_review_archive.zip .
```

4. Confirm the paired manifest and archive hashes:

```powershell
python -c "import hashlib, pathlib; paths=['release/SUPPLEMENTARY_MANIFEST.json','release/spot_od_v1_1_0_supplement_review_archive.zip']; [print(p, hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()) for p in paths]"
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
.\.venv\Scripts\python -I scripts/verify_archive_extracted_reproduction.py --archive release/spot_od_v1_1_0_supplement_review_archive.zip --json-out results/validation/independent_archive_extracted_reproduction.json --md-out results/validation/independent_archive_extracted_reproduction.md
```

8. Hash the generated reports:

```powershell
python -c "import hashlib, pathlib; paths=['results/validation/independent_minimum_tier_reproduction.json','results/validation/independent_minimum_tier_reproduction.md','results/validation/independent_archive_extracted_reproduction.json','results/validation/independent_archive_extracted_reproduction.md']; [print(p, hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()) for p in paths if pathlib.Path(p).is_file()]"
```

If the operator uses Linux or macOS, use the same arguments with
`.venv/bin/python` instead of `.\.venv\Scripts\python`.

## Manual External Operator Route

An external operator may start from either the Zenodo archive or the GitHub
release archive, verify the hashes above, and run the same clean-machine
procedure. The operator should attach the generated JSON/Markdown reports and
their SHA-256 hashes to the signed report. A signed report supports only the
verifier scope actually run; it does not expand the manuscript claims.

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
