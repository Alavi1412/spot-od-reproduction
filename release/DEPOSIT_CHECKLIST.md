# SPOT-OD supplementary evidence package — deposit-readiness checklist

This file is **internal release-ledger bookkeeping**; it is not paper-facing
and is not referenced from the manuscript. Its purpose is to inventory what
the version-pinned supplementary evidence package must contain so that the
bundle can be uploaded to Zenodo with a citable DOI without manual
reassembly. The manuscript makes no public DOI claim at initial submission,
and public archival deposition is deferred until explicit author approval or a
venue-required release point.

## Status

- Public archival deposition: **deferred until explicit author approval or a
  venue-required release point.**
- Public identifier (DOI / archive URL): **none.**
- Package is structured to be deposit-ready offline from the included pinned
  inputs and the recorded artifact checksums; no external network access is
  required at upload time. Manifest checksums for paper-facing files are
  refreshed by `scripts/build_supplementary_manifest.py` (see
  release/SUPPLEMENTARY_MANIFEST.json).
- Submission-blocking depositor/editor action item: if the journal requires a
  public citable archive at initial submission or before editorial review can
  proceed, the depositor must create the public Zenodo deposition and update the
  manuscript and release metadata with the assigned identifier before
  submission. Until that external deposit exists, no DOI or public archive URL
  may be asserted.

## Required contents (already bundled)

1. Manuscript source (paper/main.tex + paper/supplement.tex), bibliography
   (paper/references.bib), generated tables and figures.
2. Version-pinned dependency provenance: requirements.txt and pyproject.toml
   with SHA-256 digests recorded in release/SUPPLEMENTARY_MANIFEST.json.
3. Predeclared rule artifacts: timestamped rule files under
   release/predeclarations/ (primary endpoint, 3% practical-significance
   floor, DBAR decision rule, symmetric process-noise-adaptive UKF rule,
   DMC-EKF predeclared rule, DSA-EKF predeclared rule for the higher-fidelity
   force-mismatch slice).
4. 15-seed cohort (seeds 41–55) and trained-model records.
5. Per-artifact SHA-256 digests for every produced output (recorded in
   release/SUPPLEMENTARY_MANIFEST.json).
6. Archived public-input identifiers: CelesTrak GP/TLE snapshots, SatNOGS
   network/observation/API metadata, ILRS Consolidated Laser Ranging Data
   v2 normal-point filenames and source URLs with per-file SHA-256 digests
   and byte lengths, ILRS NSGF analysis-centre SP3-c precise orbit products
   with the same provenance, and the IERS Earth-orientation series.
7. Vendored KalmanNet source release at the recorded upstream commit hash.
8. Reviewer access guide (`release/REVIEWER_START_HERE.md`) and minimum-tier
   reproduction check reports under `results/validation/`, both indexed by
   `release/SUPPLEMENTARY_MANIFEST.json` and included in the reviewer archive.
9. Independent-machine reproduction request/template
   (`release/INDEPENDENT_MACHINE_REPRODUCTION_REQUEST.md`) so an external
   operator can produce a signed clean-machine report without changing any
   manuscript claim before the report exists.
10. Official ILRS precise-reference availability probe output under
   `results/validation/`, documenting whether pending LAGEOS prospective SP3
   products are valid gzip/SP3 files before any scoring run.
11. DOI-ready deposit metadata (`release/ZENODO_METADATA.json`) with no DOI or
   public URL asserted before an actual archive deposit.

## To complete a public archival deposit

The following depositor steps are required outside the automated package
state:

- [ ] Create the Zenodo deposition for the SHA-256-pinned bundle.
- [ ] Upload the SHA-256-pinned bundle as a single versioned release.
- [ ] Record the assigned DOI/handle in
      `release/SUPPLEMENTARY_MANIFEST.json::public_identifier`.
- [ ] Update the manuscript Data Availability section to cite the DOI.
- [ ] Add the DOI/handle to the citation entry for this work (CITATION.cff)
      and resubmit the camera-ready paper with the updated reference.

Until those steps are completed by the depositor, the manuscript Data
Availability section states that no public DOI is asserted at initial
submission and that public archival deposition is deferred until explicit
author approval or a venue-required release point. Manifest checksums for
paper-facing files are recorded in release/SUPPLEMENTARY_MANIFEST.json. No
paper-facing DOI or public-repository identifier is asserted before an actual
Zenodo deposit.

## Forbidden in paper-facing text

Per the paper-facing release constraints, this checklist exists only in
release/ and is NEVER quoted, paraphrased, or referenced from any
paper-facing source (paper/main.tex, paper/supplement.tex,
paper/evidence_plan.tex). The Data Availability section in the manuscript
remains the canonical paper-facing statement.
