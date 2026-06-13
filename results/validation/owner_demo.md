# Owner Demo Readback

Generated UTC: `2026-06-13T03:51:47.500629+00:00`
Overall validation: **PASS**
Manuscript PDF: `paper\main.pdf`
PDF pages: `33`
PDF SHA-256: `3fe83b8c94c72819140e99341780f456c146dbf06b907b1aee406672f20db1c3`
Citation validation: **PASS**
Release evidence index: **PASS**
LaTeX warnings reported (main + supplement): `0`; supplement: `0`
Command manifest: **PASS**
Adversarial review artifact: **PASS**

## Honest Readback
- The paper currently passes deterministic PDF, citation, active-artifact, and release-packet checks.
- Main and supplement LaTeX warning/overfull checks are disclosed in validation artifacts.
- The validation is evidence-bounded: it proves the current artifacts are synchronized, not that all experiments were freshly regenerated in this validation step.
- Public replay and deployment-readiness claims remain intentionally limited by the manuscript.
- The pytest status is accepted only when a current pytest log with a passing summary is present.

## Primary Artifacts
- `results\validation\submission_validation.json`
- `results\validation\submission_validation.md`
- `results\validation\command_manifest.json`
- `results\validation\adversarial_review.json`
- `results\validation\task227_final_validation.json`
