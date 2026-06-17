# Real SLR/SP3 OD Slice Rerun Validation

Status: **PASS**

## Scope Boundary
One public LAGEOS CRD/SP3 precise-reference OD slice rerun from archived public inputs through range-only EKF/UKF/AUKF/SP3-IC recomputation and table reconstruction. This is not full scientific reproduction, not full estimator training, not all filters/tables, not live public-data retrieval, and not operational POD validation.

## Rerun
- Step: `real_slr_sp3_od_slice_rerun`
- Exit code: `0`
- Execution details: redacted from this reviewer-facing summary.
- Rerun JSON: `results/validation/real_slr_sp3_od_slice_rerun/real_slr_sp3_od_validation.json`
- Rerun table: `results/validation/real_slr_sp3_od_slice_rerun/real_slr_sp3_od.tex`

## Comparisons
- Public-claim summary fields: **PASS** (0 mismatches).
- Public-claim tolerated numeric differences: `0`; max absolute delta `0.0` m; RMSE tolerance `0.25` m.
- Generated table text matches submitted table: **PASS**.
- Table tolerated numeric differences: `0`; max absolute delta `0.0`; field-aware tolerance `0.5`.

## Summary
- Completed arcs: `10`.
- DBAR correct/scored: `6/10`.
- Table text matched: `True`.
