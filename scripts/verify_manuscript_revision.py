from __future__ import annotations

import json
import re
from pathlib import Path


root = Path.cwd()
main_path = root / "paper" / "main.tex"
supplement_path = root / "paper" / "supplement.tex"
bib_path = root / "paper" / "references.bib"
release_path = root / "results" / "release_packet.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


text = read_text(main_path)
supplement_text = read_text(supplement_path)
bib = read_text(bib_path)
paper_source_text = "\n".join([text, supplement_text])

active_table_names = sorted(
    set(re.findall(r"\\input\{tables/([^}]+\.tex)\}", paper_source_text))
)
active_table_paths = [f"paper/tables/{name}" for name in active_table_names]
active_table_source_text = "\n".join(read_text(root / rel) for rel in active_table_paths)

active_figure_names = sorted(
    set(
        re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}",
            paper_source_text,
        )
    )
)
active_figure_paths = [f"paper/{name}" for name in active_figure_names]

# Current submission checks are scoped to the main manuscript, supplement, and
# the tables/figures they actually input. paper/evidence_plan.tex is retained
# as a historical/internal artifact but is no longer a paper-facing include.
expanded_text = "\n".join([paper_source_text, active_table_source_text])
text_lower = expanded_text.lower()


def normalize_phrase(value: str) -> str:
    value = value.lower()
    value = value.replace("\\'", "")
    value = value.replace("~", " ")
    value = value.replace("--", " ")
    value = value.replace("-", " ")
    value = re.sub(r"\\[a-zA-Z]+\*?", " ", value)
    value = re.sub(r"[^a-z0-9%]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


normalized_text = normalize_phrase(expanded_text)


def phrase_present(phrase: str) -> bool:
    return normalize_phrase(phrase) in normalized_text


LATEX_LABEL_REFERENCE_RE = re.compile(
    r"\\(?:ref|eqref|autoref|nameref|pageref|cref|Cref|vref|Vref)\*?\{([^{}]+)\}"
)


def has_exact_latex_label_reference(source: str, label: str) -> bool:
    for match in LATEX_LABEL_REFERENCE_RE.finditer(source):
        labels = [part.strip() for part in match.group(1).split(",")]
        if label in labels:
            return True
    return False


required_sections = [
    "Introduction",
    "Related Work and Positioning",
    "Data Sources, Evaluation Scope, and Protocol",
    "Methods Compared",
    "Results",
    "Claim Audit",
    "Limitations and Missing Experiments",
    "Data and Reproducibility Statement",
    "Conclusion",
]

required_claim_groups = {
    "simulator_bound_self_audit": [
        "simulator-bound orbit-determination self-audit record",
    ],
    "known_aukf_mechanism": [
        "known covariance-matching failure mode",
    ],
    "aukf_r_inflation": [
        "AUKF inflates effective R",
    ],
    "drag_scale_cascade": [
        "drag-scale cascade separates EKF linearisation failure, sparse-geometry observability limits, and candidate-side",
    ],
    "learned_bounded_negative": [
        "no evaluated learned construction beats the per-scenario best tuned classical reference",
    ],
    "endpoint_choice_negative": [
        "endpoint-choice sensitivity audit finds no learned positive",
        "endpoint-choice sensitivity audit does not create a learned positive",
    ],
    "endpoint_timeline_posthoc": [
        "observed-step RMSE was selected post hoc on the training cohort",
        "observed-step RMSE was selected on training-cohort data after a post-hoc recomputation",
    ],
    "k8_timestamp_caveat": [
        "K=8 endpoint-support lacks an external timestamp",
        "K=8 endpoint-fixation support record lacks a created/finalized timestamp field",
    ],
    "k32_k96_internal_boundary": [
        "K=32 and K=96 are internal frozen-rule checks, not external preregistration",
        "K=96 internal replication under an already selected endpoint, not external preregistration",
    ],
    "not_operational_pod": [
        "not operational POD",
    ],
    "not_centimetre_slr": [
        "not centimetre SLR validation",
    ],
    "not_broad_learned_od": [
        "not a broad learned-OD refutation",
    ],
    "kalmannet_transposition_scope": [
        "documented KalmanNet transposition is retained separately as a feasibility/design-gap diagnostic",
    ],
    "kalmannet_adapted_transposition": [
        "documented adapted KalmanNet SPOT-OD transposition",
    ],
    "kalmannet_four_design_changes": [
        "four predeclared design changes",
    ],
    "no_kalmannet_architecture_refutation": [
        "not an architecture level refutation of KalmanNet on its native benchmark",
        "not architecture-level refutations",
    ],
    "hifi_mechanism_scope": [
        "higher-fidelity force-mismatch slices preserve the R-only NIS stress signature",
    ],
    "hifi_ordering_not_transfer": [
        "mechanism continues to fire but the EKF/AUKF ordering is not flipped",
    ],
    "pukf_bounded_negative": [
        "predeclared symmetric Q-adaptive UKF",
        "PUKF is therefore a predeclared bounded negative",
    ],
    "dmc_dsa_negative": [
        "DMC-EKF is not better than UKF",
    ],
    "dsa_practical_floor": [
        "DSA-EKF is below its practical floor",
    ],
    "long_arc_dsa_negative": [
        "long-arc replication confirms the DSA-EKF negative",
    ],
    "dbar_withdrawn": [
        "DBAR is withdrawn",
    ],
    "public_probe_boundary": [
        "Public ILRS/SP3 probes are measurement-pipeline/provenance checks only",
    ],
    "versioned_submission_package": [
        "versioned supplementary evidence package supplied with the submission",
    ],
    "confidential_inspection_boundary": [
        "confidential inspection materials supplied through the journal submission system",
    ],
    "no_public_identifier_claimed": [
        "No DOI or public identifier is claimed at initial submission",
    ],
    "confidential_review_channel_only": [
        "journal submission and confidential review channel only",
    ],
    "not_full_scientific_reproduction": [
        "These completed checks do not constitute full scientific reproduction",
    ],
}

missing_claim_phrases = [
    f"{name}: {' | '.join(choices)}"
    for name, choices in required_claim_groups.items()
    if not any(phrase_present(choice) for choice in choices)
]

required_table_inputs = [
    "\\input{tables/main_framework_portability.tex}",
    "\\input{tables/main_k32_replication.tex}",
    "\\input{tables/main_aukf_mechanism.tex}",
    "\\input{tables/main_structural_recoverability.tex}",
    "\\input{tables/main_drag_scale_cascade.tex}",
    "\\input{tables/main_long_arc_result.tex}",
    "\\input{tables/main_dbar_withdrawal.tex}",
    "\\input{tables/observed_step_powered_stress_replication.tex}",
    "\\input{tables/observed_step_preregistration.tex}",
    "\\input{tables/endpoint_selection_sensitivity.tex}",
    "\\input{tables/hifi_force_mismatch.tex}",
    "\\input{tables/hifi_force_mismatch_extended.tex}",
    "\\input{tables/kalmannet_spot_od_transposition.tex}",
    "\\input{tables/kalmannet_spot_od_budget_adequacy.tex}",
    "\\input{tables/kalmannet_official_reproduction.tex}",
    "\\input{tables/kalmannet_gain_inhouse_comparator.tex}",
]

hifi_table_text = read_text(root / "paper/tables/hifi_force_mismatch.tex")
kalmannet_table_text = read_text(root / "paper/tables/kalmannet_spot_od_transposition.tex")
kalmannet_budget_text = read_text(root / "paper/tables/kalmannet_spot_od_budget_adequacy.tex")
observed_step_table_text = read_text(root / "paper/tables/observed_step_preregistration.tex")

kalmannet_spot_od_artifact = (
    "results/kalmannet_spot_od_loop57/kalmannet_spot_od.json"
)
kalmannet_spot_od_payload = read_json(root / kalmannet_spot_od_artifact)
kalmannet_spot_od_config = kalmannet_spot_od_payload.get("config", {}) or {}
kalmannet_spot_od_means = (
    kalmannet_spot_od_payload.get("observed_step_rmse_mean_m", {}) or {}
)
kalmannet_spot_od_paired = (
    kalmannet_spot_od_payload.get("paired_vs_best_classical", {}) or {}
)
pinned_kalmannet_commit = read_text(
    root / "external/third_party/KalmanNet_TSP_COMMIT"
).strip()

loop_label_pattern = re.compile(r"tab:[A-Za-z0-9_:-]*_loop\d+")


def forbidden_hits(patterns: dict[str, str], source: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name, pattern in patterns.items():
        matches: list[str] = []
        for match in re.finditer(pattern, source, flags=re.IGNORECASE):
            start = max(0, match.start() - 48)
            end = min(len(source), match.end() + 48)
            snippet = re.sub(r"\s+", " ", source[start:end]).strip()
            matches.append(snippet)
        if matches:
            hits[name] = matches[:5]
    return hits


forbidden_public_archive_patterns = {
    "zenodo": r"\bzenodo\b",
    "doi_url": r"\b(?:doi\.org|dx\.doi)\b",
    "figshare": r"\bfigshare\b",
    "osf": r"\bosf\.io\b",
    "dryad": r"\bdryad\b",
    "public_doi": r"\bpublic\s+doi\b",
    "public_archive": r"\bpublic\s+(?:archive|deposition|deposit)\b",
    "public_repository": r"\bpublic(?:ly\s+available)?\s+repository\b",
    "public_code_release": r"\bpublic\s+code\s+release\b",
    "public_release": r"\bpublic\s+release\s+of\b",
    "publicly_released": r"\bpublicly\s+released\b",
    "open_source_release": r"\bopen-source\s+release\b",
}
forbidden_implementation_patterns = {
    "hardware": r"\bhardware\b",
    "cpu": r"\bcpu\b",
    "gpu": r"\bgpu\b",
    "cuda": r"\bcuda\b",
    "virtualenv": r"\bvirtualenv\b",
    "venv": r"\b\.venv\b|\bvenv\b",
    "conda": r"\bconda\b",
    "pythonpath": r"\bpythonpath\b",
    "scripts_path": r"\bscripts[\\/]",
    "results_path": r"\bresults[\\/]",
    "checkpoint": r"\bcheckpoint\b",
    "source_code": r"\bsource\s+code\b",
    "code_structure": r"\bcode[-\s]+structure\b",
    "repository_structure": r"\brepository[-\s]+structure\b",
    "model_weight_file": r"\.pt\b",
}

forbidden_public_archive_hits = forbidden_hits(
    forbidden_public_archive_patterns, expanded_text
)
forbidden_implementation_hits = forbidden_hits(
    forbidden_implementation_patterns, expanded_text
)

bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
cited_keys: list[str] = []
for match in re.finditer(r"\\cite\{([^}]+)\}", paper_source_text):
    cited_keys.extend([key.strip() for key in match.group(1).split(",") if key.strip()])
missing_cited_keys = sorted(set(cited_keys) - bib_keys)

required_bib_keys = [
    "celestrak_gp",
    "satnogs_network",
    "satnogs_observations",
    "satnogs_api",
    "hoots1980spacetrack",
    "vallado2006sgp4",
    "vallado2013",
    "montenbruck2000",
    "tapley2004statistical",
    "kalman1960",
    "jazwinski1970stochastic",
    "julier1997unscented",
    "wan2000unscented",
    "revach2022kalmannet",
    "mehra1970adaptive",
    "wright1981drag",
    "pearlman2002ilrs",
    "ilrs_npt_data",
    "ilrs_crd_format",
    "cddis_slr_npt",
    "holm1979",
    "benjamini1995",
    "efron1994",
    "morris2019simulation",
    "pineau2021reproducibility",
]
uncited_required_keys = sorted(set(required_bib_keys) - set(cited_keys))

artifact_paths = [
    *active_table_paths,
    *active_figure_paths,
    "paper/references.bib",
    "results/release_packet.json",
    "results/observed_step_preregistration/observed_step_preregistration.json",
    "results/observed_step_powered_stress_replication/observed_step_powered_stress_replication.json",
    "results/hifi_force_mismatch/hifi_force_mismatch.json",
    "results/hifi_force_mismatch_extended/hifi_force_mismatch_extended.json",
    kalmannet_spot_od_artifact,
    "results/kalmannet_spot_od_budget_adequacy_loop58/kalmannet_spot_od_budget_adequacy.json",
    "results/long_arc_hifi_force_mismatch/long_arc_hifi_force_mismatch.json",
    "release/SUPPLEMENTARY_MANIFEST.json",
    "release/CITATION.cff",
    "release/README.md",
]
artifact_existence = {rel: (root / rel).exists() for rel in artifact_paths}

checks = {
    "main_exists": main_path.exists(),
    "supplement_exists": supplement_path.exists(),
    "release_packet_exists": release_path.exists(),
    "active_table_inputs_resolve": all(
        (root / rel).exists() for rel in active_table_paths
    ),
    "active_figure_includes_resolve": all(
        (root / rel).exists() for rel in active_figure_paths
    ),
    "required_table_inputs_present": all(
        item in paper_source_text for item in required_table_inputs
    ),
    "required_sections_present": all(
        ("\\section{" + section + "}") in text for section in required_sections
    ),
    "abstract_present": "\\begin{abstract}" in text and "\\end{abstract}" in text,
    "contributions_present": "\\paragraph{Contributions.}" in text,
    "evidence_plan_not_input": "\\input{evidence_plan}" not in paper_source_text,
    "bibliography_file_used": (
        "\\bibliography{references}" in text
        and "\\bibliography{references}" in supplement_text
    ),
    "required_result_phrases_present": not missing_claim_phrases,
    "no_missing_bib_keys_for_citations": not missing_cited_keys,
    "required_citation_keys_are_cited": not uncited_required_keys,
    "unsupported_state_of_the_art_removed": "state-of-the-art" not in text_lower,
    "generic_replacement_title_removed": (
        "evidence-bounded graph neural state estimation under intermittent visibility"
        not in text_lower
    ),
    "explicit_evidence_boundary_present": all(
        phrase_present(phrase)
        for phrase in [
            "Conclusions are drawn strictly from the reported tables and figures",
            "the evidence is simulator-bound",
            "not operational POD",
            "not a broad learned-OD refutation",
        ]
    ),
    "data_availability_confidential_boundary_present": all(
        phrase_present(phrase)
        for phrase in [
            "confidential inspection materials supplied through the journal submission system",
            "versioned supplementary evidence package and manifest",
            "No DOI or public identifier is claimed at initial submission",
            "journal submission and confidential review channel only",
        ]
    ),
    "hifi_main_reference_matching_is_exact": not has_exact_latex_label_reference(
        r"S-Table~\ref{tab:hifi_force_mismatch_extended}",
        "tab:hifi_force_mismatch",
    ),
    "hifi_table_supplement_housed_and_main_referenced": (
        has_exact_latex_label_reference(text, "tab:hifi_force_mismatch")
        and "\\input{tables/hifi_force_mismatch.tex}" not in text
        and "\\input{tables/hifi_force_mismatch.tex}" in supplement_text
        and "\\label{tab:hifi_force_mismatch}" in hifi_table_text
    ),
    "active_kalmannet_spot_od_loop57_json_conforms": (
        bool(kalmannet_spot_od_payload)
        and kalmannet_spot_od_payload.get("scenario")
        == "kalmannet_spot_od_transposition"
        and kalmannet_spot_od_payload.get("vendor_commit")
        == pinned_kalmannet_commit
        and kalmannet_spot_od_config.get("m") == 6
        and kalmannet_spot_od_config.get("n") == 32
        and kalmannet_spot_od_config.get("n_train") == 160
        and kalmannet_spot_od_config.get("n_cv") == 24
        and kalmannet_spot_od_config.get("n_test") == 64
        and kalmannet_spot_od_config.get("n_steps") == 300
        and "KalmanNet-SPOT-OD" in kalmannet_spot_od_means
        and {"EKF", "UKF", "AUKF", "PUKF"}.issubset(
            set(kalmannet_spot_od_means)
        )
        and kalmannet_spot_od_paired.get("best_classical")
        in {"EKF", "UKF", "AUKF", "PUKF"}
    ),
    "kalmannet_labels_are_paper_facing_unsuffixed": (
        "tab:kalmannet_spot_od_transposition" in paper_source_text
        and "\\input{tables/kalmannet_spot_od_transposition.tex}" in supplement_text
        and "\\input{tables/kalmannet_spot_od_budget_adequacy.tex}" in supplement_text
        and "\\label{tab:kalmannet_spot_od_transposition}" in kalmannet_table_text
        and "\\label{tab:kalmannet_spot_od_budget_adequacy}" in kalmannet_budget_text
        and not loop_label_pattern.search(expanded_text)
    ),
    "observed_step_endpoint_fixation_boundary_present": (
        "Submitted observed-step endpoint-fixation support record"
        in observed_step_table_text
        and "lacks a created/finalized timestamp field" in observed_step_table_text
        and "$K{=}8$ endpoint-fixation support" in observed_step_table_text
        and "no confirmatory status" in observed_step_table_text
        and "predeclared" not in observed_step_table_text.lower()
    ),
    "no_false_public_archive_claim": not forbidden_public_archive_hits,
    "no_paper_facing_implementation_leak": not forbidden_implementation_hits,
    "no_stale_formal210_main_text": all(
        pattern not in text
        for pattern in [
            "formal 210-arc replay supersedes",
            "137/200",
            "605.68",
            "597.68",
            "630.65",
        ]
    ),
    "no_stale_formal210_supplement_heterogeneity": all(
        pattern not in supplement_text
        for pattern in [
            "Q1=37, median=58",
            "One-station arcs (n=36)",
            "4+ station arcs (n=25)",
            "highest quartile ($>$89",
        ]
    ),
}

result = {
    "verification": "manuscript_revision_static_checks",
    "method": (
        "Static verification of current paper/main.tex and paper/supplement.tex "
        "against active generated table/figure inputs, current claim-boundary "
        "phrasing, bibliography coverage, confidential-review data availability, "
        "and forbidden public-archive / implementation-leak wording."
    ),
    "all_static_checks_passed": all(checks.values()),
    "all_active_artifacts_exist": all(artifact_existence.values()),
    "checks": checks,
    "artifact_existence": artifact_existence,
    "active_table_inputs": active_table_paths,
    "active_figure_includes": active_figure_paths,
    "missing_claim_phrases": missing_claim_phrases,
    "forbidden_public_archive_hits": forbidden_public_archive_hits,
    "forbidden_implementation_hits": forbidden_implementation_hits,
    "cited_keys": sorted(set(cited_keys)),
    "missing_cited_keys": missing_cited_keys,
    "uncited_required_keys": uncited_required_keys,
    "changed_files": [
        "tests/test_loop42_hifi_kalmannet_artifacts.py",
        "scripts/verify_manuscript_revision.py",
        "results/manuscript_revision_verification.json",
    ],
    "active_evidence_sources": [
        *active_table_paths,
        *active_figure_paths,
        "results/release_packet.json",
        "release/SUPPLEMENTARY_MANIFEST.json",
    ],
}

out = root / "results" / "manuscript_revision_verification.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(
    0
    if result["all_static_checks_passed"] and result["all_active_artifacts_exist"]
    else 1
)
