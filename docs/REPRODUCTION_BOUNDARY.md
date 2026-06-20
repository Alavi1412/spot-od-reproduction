# Reproduction Boundary

This repository is intentionally narrow. It publishes and verifies the archive-extracted reproduction tier for the SPOT-OD supplementary evidence package.

## In Scope

- Byte-level release archive and paired manifest checks.
- ZIP extraction checks for `1014` archive members.
- Manifest-indexed artifact presence, byte-size checks, and SHA-256 checks.
- Claim-to-artifact and regeneration-tier resolution against the extracted archive.
- Active manuscript table/figure artifact regeneration from archived result artifacts.
- One archived-input public LAGEOS CRD/SP3 precise-reference OD slice rerun.

## Out of Scope

- Full raw data generation.
- Full model training, hyperparameter optimization, or checkpoint regeneration.
- All-filter or all-table reruns.
- Live retrieval from public data services.
- Operational precise orbit determination validation.
- Third-party independent validation.

The verifier is evidence that the archived package is internally consistent and that the defined archive-extracted tier can be rerun. It is not evidence that every upstream experiment can be regenerated from scratch.
