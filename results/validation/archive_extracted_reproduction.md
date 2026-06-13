# Archive-Extracted Reproduction Check

Status: **PASS**

## Scope Boundary
Archive-extracted integrity, active main-manuscript table-regeneration, and one public LAGEOS CRD/SP3 precise-reference OD slice recomputation from archived public inputs only; this does not rerun full raw-data generation, model retraining, all recursive filters or tables, live public-data retrieval, operational POD validation, or independent end-to-end reproduction outside the supplied archive.

## Checks
- ZIP extraction: **PASS**.
- Review archive alias restore for extracted rerun dependencies: **PASS**.
- Extracted manifest-indexed artifact presence and SHA-256 checks: **PASS**.
- Claim-to-artifact map resolution: **PASS**.
- Regeneration-tier key resolution: **PASS**.
- Active table regeneration from extracted tree: **PASS**.
- Archive-extracted public OD slice rerun: **PASS**.

## Manifest Source
- Manifest source: `paired_release_manifest`.
- Loaded from extracted archive: `False`.
- Note: The review ZIP is digest-addressed by the paired release manifest; the manifest is therefore treated as an allowed release-level record rather than a self-referential ZIP member.

## Counts
- Manifest-indexed artifacts checked after extraction: `955`.
- Extracted ZIP members: `955`.
- Claim-map entries: `21`.
- Regeneration tiers: `6`.

## Extracted Active Table Regeneration
- Attempted: `True`.
- Exit code: `0`.
- Nested status: `pass`.
- Active artifacts: `10`.
- Pass count: `10`.
- Mismatch count: `0`.
- Blocker count: `0`.

## Archive-Extracted Public OD Slice Rerun
- Attempted: `True`.
- Step: `archive_extracted_public_od_slice_rerun`.
- Exit code: `0`.
- Execution details: redacted from this reviewer-facing summary.
- Completed arcs: `10`.
- Public-claim summary fields: **PASS** (0 mismatches).
- DBAR correct/scored: `6/10`.
- Generated table text matches extracted submitted table: **PASS**.
- Companion report: `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json` and `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.md`.

## Outputs
- JSON: `results/validation/archive_extracted_reproduction.json`
- Markdown: `results/validation/archive_extracted_reproduction.md`
