# v1.2.4-acf-prospective-guard

This release packages the follow-up AdaptiveCandidateFusion fixed scenario-guard prospective rerun used to bound the post-freeze ACF diagnostics.

## Included artifact

- `spot_od_v1_2_4_acf_prospective_guard_artifact.zip`
  - Size: 278,230,689 bytes
  - SHA-256: `FB91EB9179D2F0AF236B70A31D4DC10AD0BDECD8C9F94233DAD25E81C71C9D52`

The archive contains the five fresh prospective ACF run directories for seeds 67, 71, 73, 79, and 83; their captured logs; the aggregate fixed-guard artifact under `results/acf_process_guard_policy_candidategraph1_prospective_seed67_71_73_79_83_20260624/`; and the relevant ACF run/artifact scripts and tests. The updated manuscript and supplement files are carried by the tagged GitHub source snapshot rather than by this binary artifact, avoiding a circular asset-hash dependency.

## Scientific boundary

The five-seed prospective guard rerun does **not** validate the fixed scenario guard. It records 4/10 scenario-row wins plus 1 tie, mean observed-step gain -6.366%, row bootstrap CI [-18.521,+3.542]%, and 2/5 positive seed-paired means with seed-paired CI [-15.306,+2.379]%. The process slice is hypothesis-generating but unconfirmed, while the maneuver fallback fails.

This release is reproduction-support evidence for a post-freeze compact-simulator diagnostic. It is not independent-machine reproduction, not operational precise-reference validation, not a full raw/training/all-filter rerun, and not confirmatory learned-superiority evidence.
