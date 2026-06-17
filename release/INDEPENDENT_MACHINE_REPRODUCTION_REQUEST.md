# Independent-Machine Reproduction Request

This file is a handoff template for a third-party operator. This is not a completed independent reproduction, not third-party validation, not a DOI or public archive record, and not scored external validation.

## Objective

Produce reviewer-credible evidence that the submitted supplementary evidence
package can be verified from a clean machine using the supplied manifest,
review archive, and verifier scripts. The expected deliverable is a signed
operator report with machine identity, command transcripts, hashes, pass/fail
status, deviations, and output hashes.

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

## Optional Private GitLab CI Route

This repository also provides `.gitlab-ci.yml` as a private GitLab CI
route/request for clean-run evidence on GitLab shared runners after a branch
push. This route is not completed reproduction until the
`archive_extracted_reproduction` job has passed and its job URL plus artifacts
are attached or cited.

The CI job intentionally runs only the archive-extracted reproduction verifier:

```bash
python scripts/verify_archive_extracted_reproduction.py --archive release/spot_od_v1_1_0_supplement_review_archive.zip --json-out results/validation/gitlab_ci_archive_extracted_reproduction.json --md-out results/validation/gitlab_ci_archive_extracted_reproduction.md
```

It does not run the live-worktree minimum-tier verifier, because this CI route
tests the release archive as the artifact under review and avoids requiring a
branch checkout to contain every live workspace artifact indexed by the
manifest. The job installs only the minimal archive-extracted verifier
dependencies (`numpy scipy pandas matplotlib seaborn pyyaml tqdm sgp4 pytest`)
with `MPLBACKEND=Agg`.

Expected GitLab CI artifacts:

- `results/validation/gitlab_ci_archive_extracted_reproduction.json`
- `results/validation/gitlab_ci_archive_extracted_reproduction.md`
- `results/validation/gitlab_ci_reproduction_attestation.json`
- `results/validation/gitlab_ci_reproduction_attestation.md`

A passing private GitLab CI job can support archive-extracted
independent-machine reproduction evidence only. It is not DOI/public archive
evidence, not operational POD validation, not a full scientific rerun, and not
third-party independent validation. Do not claim independent-machine
reproduction from this route unless the passed job URL and the four artifacts
above are attached or cited.

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
- GitLab CI project/pipeline/job URL, if the optional CI route was used:
- GitLab CI artifact bundle attached or cited, if the optional CI route was used:
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
rerun tier is separately executed, and not a public archive/DOI unless a
public deposit is separately created and cited.
