# ILRS Precise-Reference Availability Probe

Generated UTC: `2026-06-17T18:17:53.225118Z`

## Boundary

This is an official-source availability gate, not scored validation. Unavailable products are not validation evidence, and this report does not establish independent-machine reproduction.

## Overall Status

- All required default satellite/week products available: `False`
- Usable product rule: gzip bytes plus SP3-like decompressed header.

## Satellite/Week Matrix

| Satellite | Week | Usable SP3 | Usable candidates | Classifications |
|---|---:|---:|---:|---|
| lageos1 | 260606 | True | 1 | html_or_login_not_sp3, usable_gzip_sp3 |
| lageos1 | 260613 | False | 0 | html_or_login_not_sp3 |
| lageos1 | 260620 | False | 0 | html_or_login_not_sp3 |
| lageos1 | 260627 | False | 0 | html_or_login_not_sp3 |
| lageos2 | 260606 | True | 1 | html_or_login_not_sp3, usable_gzip_sp3 |
| lageos2 | 260613 | False | 0 | html_or_login_not_sp3 |
| lageos2 | 260620 | False | 0 | html_or_login_not_sp3 |
| lageos2 | 260627 | False | 0 | html_or_login_not_sp3 |

## Schedule Readiness

- `prospective_260613`: `pending_products_unavailable`; required weeks `260606, 260613`.
- `prospective_260620`: `pending_products_unavailable`; required weeks `260613, 260620`.
- `prospective_260627`: `pending_products_unavailable`; required weeks `260620, 260627`.

## Direct Product Probes

| Source | Satellite | Week | Center | Version | HTTP | Type | Bytes | Usable | Classification |
|---|---|---:|---|---|---:|---|---:|---:|---|
| EDC | lageos1 | 260606 | nsgf | v80 | 200 | text/plain | 219284 | True | usable_gzip_sp3 |
| EDC | lageos1 | 260613 | nsgf | v80 | 200 | text/html; charset=utf-8 | 825 | False | html_or_login_not_sp3 |
| EDC | lageos1 | 260620 | nsgf | v80 | 200 | text/html; charset=utf-8 | 825 | False | html_or_login_not_sp3 |
| EDC | lageos1 | 260627 | nsgf | v80 | 200 | text/html; charset=utf-8 | 825 | False | html_or_login_not_sp3 |
| EDC | lageos2 | 260606 | nsgf | v80 | 200 | text/plain | 218997 | True | usable_gzip_sp3 |
| EDC | lageos2 | 260613 | nsgf | v80 | 200 | text/html; charset=utf-8 | 825 | False | html_or_login_not_sp3 |
| EDC | lageos2 | 260620 | nsgf | v80 | 200 | text/html; charset=utf-8 | 825 | False | html_or_login_not_sp3 |
| EDC | lageos2 | 260627 | nsgf | v80 | 200 | text/html; charset=utf-8 | 825 | False | html_or_login_not_sp3 |
| CDDIS | lageos1 | 260606 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |
| CDDIS | lageos1 | 260613 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |
| CDDIS | lageos1 | 260620 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |
| CDDIS | lageos1 | 260627 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |
| CDDIS | lageos2 | 260606 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |
| CDDIS | lageos2 | 260613 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |
| CDDIS | lageos2 | 260620 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |
| CDDIS | lageos2 | 260627 | nsgf | v80 | 200 | text/html; charset=utf-8 | 11291 | False | html_or_login_not_sp3 |

## Historical EDC Positive Controls

These known historical products test the direct EDC URL pattern. They do not make pending weeks available or scoreable.

| Satellite | Week | HTTP | Bytes | SHA-256 | Usable | Classification |
|---|---:|---:|---:|---|---:|---|
| lageos1 | 260509 | 200 | 219293 | `14a0263bf8cad4dda5cddf7295af63419a520652126fffb680ef8cdb924210a1` | True | usable_gzip_sp3 |
| lageos2 | 260509 | 200 | 219040 | `be794c46b517cdc8de73e7140b2e86b3ffb8362ee0ba19acbef3efd1bcefec90` | True | usable_gzip_sp3 |

## Local Cached Product Checks

Cached files with pending `.sp3.gz` names are not treated as products unless the gzip/SP3 gate passes. HTML placeholders under SP3 filenames remain unavailable/non-usable.

| Path | Exists | Bytes | SHA-256 | Usable | Campaign input kind | Classification |
|---|---:|---:|---|---:|---|---|
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos1.260606.v80.sp3.gz` | False | 0 | `None` | False | None | local_cache_missing |
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos1.260613.v80.sp3.gz` | True | 825 | `6a8a0a15ae1607b6ff813a4ee6acdf1c98c3b5e1f508e1c49939a6ba7beacaf2` | False | sp3_not_valid_gzip | html_or_login_not_sp3 |
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos1.260620.v80.sp3.gz` | True | 825 | `624229fb5c900ccd2717c13806d32024beb4f115c1776c57ee21f6c037b2df98` | False | sp3_not_valid_gzip | html_or_login_not_sp3 |
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos1.260627.v80.sp3.gz` | False | 0 | `None` | False | None | local_cache_missing |
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos2.260606.v80.sp3.gz` | False | 0 | `None` | False | None | local_cache_missing |
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos2.260613.v80.sp3.gz` | True | 825 | `a60dd5a93b9493007357e6f879fb59b08e25862e015bc987bbad4c7efb030810` | False | sp3_not_valid_gzip | html_or_login_not_sp3 |
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos2.260620.v80.sp3.gz` | True | 825 | `b0796cdfa5dfd1d28b20d2de5f549e8e4aa82fc6d01393d403ccc88dd6c780cc` | False | sp3_not_valid_gzip | html_or_login_not_sp3 |
| `results/real_slr_sp3_od_formal210_inputs/nsgf.orb.lageos2.260627.v80.sp3.gz` | False | 0 | `None` | False | None | local_cache_missing |

## Campaign Commands

Run these only after the schedule readiness row is `available_to_score` and all needed predeclaration boundaries are valid.

### `prospective_260613`

Predeclaration command, only if this would still be a valid pre-scoring rule:

```powershell
python scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py --schedule prospective_260613 --predeclaration release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260613.json --write-predeclaration-only
```

Scoring command:

```powershell
python scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py --schedule prospective_260613 --predeclaration release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260613.json --output-json results/real_slr_sp3_temporal_corrected_od_prospective_260613/real_slr_sp3_temporal_corrected_od_prospective_260613.json --no-table --refresh --resume
```

Note: No repository predeclaration was found for 260613 in this batch; do not treat any later-created rule as prospective unless an independent timestamped pre-scoring rule exists.

### `prospective_260620`

Scoring command:

```powershell
python scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py --schedule prospective_260620 --predeclaration release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json --output-json results/real_slr_sp3_temporal_corrected_od_prospective_260620/real_slr_sp3_temporal_corrected_od_prospective_260620.json --no-table --refresh --resume
```

Note: Existing timestamped predeclaration covers the 2026-06-15..2026-06-19 test week; run scoring only after all validation/test SP3 products are valid gzip/SP3 files.

### `prospective_260627`

Scoring command:

```powershell
python scripts/run_real_slr_sp3_temporal_corrected_od_campaign.py --schedule prospective_260627 --predeclaration release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260627_20260617.json --output-json results/real_slr_sp3_temporal_corrected_od_prospective_260627/real_slr_sp3_temporal_corrected_od_prospective_260627.json --no-table --refresh --resume
```

Note: Existing timestamped predeclaration covers the 2026-06-22..2026-06-26 test week; run scoring only after all validation/test SP3 products are valid gzip/SP3 files.

## Outputs

- JSON: `results/validation/ilrs_precise_reference_availability_20260617.json`
- Markdown: `results/validation/ilrs_precise_reference_availability_20260617.md`
