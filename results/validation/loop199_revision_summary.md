# Loop 199 Revision Summary

## Overview
Implemented focused manuscript revisions addressing MC-1 through MC-4 from loop198 reviewer report. These are framing and clarity fixes only; no new experiments.

## Files Changed
- `paper/main.tex` - All four mandatory edits implemented

## Specific Edits

### MC-1: Operational Relevance (Added)
**Location**: Conclusion section, new paragraph after main conclusion paragraphs

**Change**: Added a "Campaign-planning workflow" paragraph that provides a concise operational workflow using existing evidence. The paragraph:
- Describes a 5-step compact-simulator gate workflow before precise-reference validation
- Makes Table 2 and Results evidence actionable with specific thresholds
- Includes worked readout using actual outcomes (RGR-GF fails gates, AUKF motivates guardrail)
- Frames workflow as resource-triage gate, not operational POD

**Word count**: ~250 words added to Conclusion

### MC-2: Contribution Hierarchy (Clarified)
**Location**: Introduction "Contributions" paragraph

**Change**: Restructured to explicitly clarify hierarchy:
- C1 remains "primary contribution" (unchanged)
- C2 explicitly labeled "Supporting methodology" 
- C3 now "secondary and illustrative, requires prospective external confirmation" with expanded disclosure that it's "an illustrative application of the evaluation discipline to the authors' own learned constructions"
- Preserved all required verifier phrases including "Exploratory bounded learned-negative"

**Effect**: Makes C1 primary and C3 secondary/illustrative explicit

### MC-3: Lim & Colombo Citation (Disambiguated)
**Location**: Introduction "Aerospace relevance" paragraph

**Change**: Separated public-orbit-data/operational context citations from learned/adaptive citations:
- **Before**: Mixed list with Lim & Colombo grouped with neural relative-navigation studies
- **After**: Two explicit groups: "operational and public-data OD contexts" (including Lim & Colombo) vs "learned and adaptive filter applications", with explicit note "SPOT-OD addresses learned-estimator evaluation rather than operational public-orbit maneuver reconstruction"

**Effect**: Removes ambiguity about Lim & Colombo being operational context, not learned/adaptive evidence

### MC-4: KalmanNet Framing (Signposted)
**Location**: Introduction "Estimator family evaluated" paragraph

**Change**: Added concise signpost at paragraph start:
- Opens with statement that "load-bearing learned-claim comparator is the in-house KalmanNet-style learned-gain comparator"
- Explicitly states this "anchors the learned-negative alongside the RGR-family constructions"
- Then introduces upstream KalmanNet reproduction/transposition as "Separately" - making subordinate relationship clear
- Emphasizes adapted transposition reports "167 km held-out result under documented adaptations but not used as evidence for the primary learned-negative claim"

**Effect**: Readers now understand load-bearing comparator upfront, before encountering transposition details

### Abstract Compression (Optional, Implemented)
**Location**: Abstract

**Change**: Compressed from ~250 words to ~220 words while preserving all claim boundaries:
- Expanded abbreviations on first use (AUKF, EKF, UKF) inline instead of in separate sentences
- Removed redundant phrase "after observing training-cohort data" (already implied by "post hoc")
- Trimmed repetitive wording
- Preserved all required disclosures: C1 primary, C3 exploratory/post-hoc/external-confirmation, operational gap

## Validation Results

All validation checks pass:

1. **Compilation**: `python scripts/compile_paper.py --with-supplement` ✓
2. **Pytest**: 24 tests passed in test_real_slr_sp3_od_expanded.py and test_paper_asset_caption_disclosures.py ✓
3. **Sync**: `python tools/sync_release_packet.py` ✓
4. **Submission validation**: Status "pass", page_count: 34 ✓
5. **Manuscript revision verification**: all_static_checks_passed: true ✓
6. **PDF text extraction**: pdftotext completed for both main and supplement ✓
7. **Forbidden term scan**: No hits for Claude, Codex, subagent, virtualenv, venv, public DOI, Zenodo, GitHub, package versions, CUDA, PyTorch ✓

## Page Count
Remained at **34 pages** (no change from loop198)

## Key Numbers Preserved
- All formal400 numbers unchanged
- All validation-clean claim boundaries preserved
- All required verifier phrases maintained

## Risks and Remaining Concerns

### Low Risk
- All four MC issues addressed
- Page count stable at 33
- All validation gates green
- Required verifier phrases intact
- Forbidden terms clean

### No New Risks Introduced
- Campaign-planning workflow uses only existing evidence
- No new claims added
- No weakening of post-hoc endpoint disclosure
- No removal of external-confirmation requirement

## Conceptual Changes Summary

1. **MC-1**: Added operational workflow paragraph making evaluation evidence actionable for practitioners
2. **MC-2**: Explicit C1 primary / C3 secondary-illustrative hierarchy in contributions
3. **MC-3**: Separated public-data/operational citations from learned/adaptive citations
4. **MC-4**: Signposted in-house KalmanNet-style comparator as load-bearing, upstream transposition as subordinate diagnostic

All changes are manuscript framing/clarity only. No new experiments, no new tables, no new numbers.
