# Archive-Extracted Real SLR/SP3 OD Slice Rerun

Status: **PASS**

## Scope Boundary
Archive-extracted public OD slice recomputation only: one public LAGEOS CRD/SP3 precise-reference slice is rerun from archived public inputs contained in the extracted review archive through range-only EKF/UKF/AUKF/SP3-IC recomputation and table reconstruction. This is not full scientific reproduction, not full estimator training, not all filters/tables, not live public-data retrieval, and not operational POD validation.

## Rerun
- Step: `archive_extracted_public_od_slice_rerun`
- Exit code: `0`
- Table rebuild exit code: `0`
- Execution details: redacted from this reviewer-facing summary.

## Comparisons
- Public-claim summary fields: **PASS** (0 mismatches).
- Generated table text matches extracted submitted table: **PASS**.

## Summary
- Completed arcs: `10`.
- DBAR correct/scored: `6/10`.
- DBAR confusion: `{'true_fire': 0, 'true_no_fire': 6, 'false_fire': 2, 'false_no_fire': 2}`.
- Table text matched: `True`.

## Pooled Held-Out Position RMSE
- `EKF` mean `367.98` m, median `248.67` m, best `1/10`.
- `UKF (fixed-noise)` mean `334.72` m, median `234.04` m, best `1/10`.
- `AUKF (adaptive)` mean `341.33` m, median `296.13` m, best `3/10`.
- `SP3-IC propagation` mean `402.92` m, median `260.68` m, best `5/10`.

## Outputs
- JSON: `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.json`
- Markdown: `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun.md`
- Generated table: `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun/real_slr_sp3_od.tex`
- Public-claim summary: `results/validation/archive_extracted_real_slr_sp3_od_slice_rerun/public_claim_summary.json`
