# Reproduction Boundary

This repository is intentionally narrow. It publishes and verifies the archive-extracted reproduction tier for the SPOT-OD supplementary evidence package and adds a compact v1.2.0 GraphAnchorPairGate proof-of-concept package.

## v1.1.0 Archive-Extracted Reproduction

### In Scope

- Byte-level release archive and paired manifest checks.
- ZIP extraction checks for `1014` archive members.
- Manifest-indexed artifact presence, byte-size checks, and SHA-256 checks.
- Claim-to-artifact and regeneration-tier resolution against the extracted archive.
- Active manuscript table/figure artifact regeneration from archived result artifacts.
- One archived-input public LAGEOS CRD/SP3 precise-reference OD slice rerun.

### Out Of Scope

- Full raw data generation.
- Full model training, hyperparameter optimization, or checkpoint regeneration.
- All-filter or all-table reruns.
- Live retrieval from public data services.
- Operational precise orbit determination validation.
- Third-party independent validation.

The v1.1.0 verifier is evidence that the archived package is internally consistent and that the defined archive-extracted tier can be rerun. It is not evidence that every upstream experiment can be regenerated from scratch.

## v1.2.0 GraphAnchorPairGate PoC

### In Scope

- Extraction and embedded-manifest SHA-256 verification for `release/spot_od_v1_2_0_graph_anchor_gate_poc.zip`.
- Verification that the embedded `MANIFEST.json` covers every non-manifest payload member.
- Retained CSV-based recomputation of the GraphAnchorPairGate seed-sweep metrics.
- Local compact-simulator held-out eval evidence for `process_noise_shift_test` and `maneuver_shift_test`.
- All-step center-window position RMSE evidence for the GraphAnchorPairGate over the existing `RFIS:VA_RFIS` anchor pair.
- Compact package inspection of code, focused tests, selected result CSV/JSON/PNG records, selected paper artifacts, and best checkpoints.

### Out Of Scope

- Primary observed-step endpoint replacement.
- Operational precise-reference validation.
- Independent third-party reproduction.
- Full raw-data generation.
- Long model training or checkpoint regeneration.
- Hyperparameter search.
- Live public-data retrieval.
- Claims that the graph PoC is universal or that it supersedes the frozen v1.1.0 endpoint hierarchy.

The graph verifier does not run training. It recomputes the retained readout from CSVs and checks the expected values: 10 scenario-seed rows, 9 row wins, 4/5 paired seeds winning both scenarios, process-shift mean gain 7.95663495038935%, maneuver-shift mean gain 8.05274642630686%, and the seed-19 process-shift failure -2.0925251807980216%.
