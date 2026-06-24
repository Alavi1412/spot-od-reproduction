# 260620 Prospective Rule Timestamp Attestation

Report generated UTC: 2026-06-17T12:54:45.9236175Z

## Artifact Summary

- Source path: `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json`
- Source SHA-256: `76f784dbefe9d251e707bc524abf602eda659e21d18738ebfd728cbaf90d87ca`
- Existing SHA-256 sidecar: `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.sha256.txt`
- Existing sidecar SHA-256 value: `76f784dbefe9d251e707bc524abf602eda659e21d18738ebfd728cbaf90d87ca`
- Sidecar match: yes, the sidecar SHA-256 still matches the source JSON.
- OTS path: `release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json.ots`
- Upgraded OTS byte length: 3950
- Upgraded OTS SHA-256: `b39558419976a16a8ae10ae53e0e2bbfbcf71d6391291f22226da85a37a40f83`
- Previous OTS SHA-256: `0b3d9a3be919df03cbe9ec52a52ec4de5b2791aecbad85e023a2ab1d6f6f29cd`
- Proof upgrade date: 2026-06-17
- Upgrade backup status: the `.bak` created during upgrade was removed.

## OpenTimestamps Status

The upgraded proof is a Bitcoin-block-header-attested OpenTimestamps proof for the 260620 prospective precise-reference rule. Inspection with `ots.exe info` reported the expected source file SHA-256 and Bitcoin block header attestations at these heights:

| Height | Merkle root | Block hash | Timestamp |
|---:|---|---|---:|
| 953674 | `5a8877ca1c7d9703b1628120c6207af4c3706244992887e9f72323a0e1841567` | `000000000000000000013935684f88b2a01774b01a9ab532fa0c0323d80ec0e5` | 1781464036 |
| 953697 | `3997f7b71d481d65486de96c7e6b591bbf1b6e054c12aacbc220f62dd355970e` | `000000000000000000007d757abafccff42ee3f40ff19cfbd00e778b47be7a5a` | 1781473768 |
| 953699 | `2875c7a1a4134cfcd55ddc444537ca9934d849f9b63d367d327afebb0798a38e` | `000000000000000000020ba82de662c661530143b459185fe6dff7f98a2dac77` | 1781474424 |

`ots.exe --no-bitcoin verify -f <json> <ots>` did not perform local Bitcoin-node verification because Bitcoin checking was disabled. It returned exit status 1 and reported manual verification targets for the same three block heights and Merkle roots. The main session cross-checked those three heights with the Blockstream API on 2026-06-17, and every Merkle root matched the OpenTimestamps output.

Some calendar-pending markers may still appear on non-upgraded proof branches in `ots.exe info`; the relevant upgrade for this artifact is the presence of the Bitcoin block header attestations above.

## Commands Recorded

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json'
Get-Content -LiteralPath 'release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.sha256.txt'
Get-FileHash -Algorithm SHA256 -LiteralPath 'release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json.ots'
$env:Path = 'C:\Program Files\Adobe\Adobe Photoshop 2025;' + $env:Path
.\.venv\Scripts\ots.exe info 'release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json.ots'
.\.venv\Scripts\ots.exe --no-bitcoin verify -f 'release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json' 'release/predeclarations/real_slr_sp3_temporal_corrected_od_prospective_260620_20260612.json.ots'
```

## Claim Boundary

This artifact is a Bitcoin-block-header-attested OpenTimestamps proof for a future public-week rule. It is not scored validation, not DOI/public archive, not independent reproduction, and not operational POD.

Local Bitcoin-node verification was not performed. The disabled-Bitcoin OpenTimestamps verification path only reported manual block/Merkle-root targets, which the main session cross-checked against the Blockstream API.

The 260620 dates 2026-06-15 through 2026-06-19 remain pending/not scored as of 2026-06-17 because public 260613/260620 SP3 products are still unavailable.
