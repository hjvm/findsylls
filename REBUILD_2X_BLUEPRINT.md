# findsylls 3.0.0 Full Rebuild Blueprint

Purpose: single source of truth for the one-pass architecture rebuild.

This document is written for maintainers and coding agents to resume work without context loss.

## Non-Negotiable Rules

1. No backward compatibility workarounds.
2. No aliases for removed names.
3. No superficial patches.
4. Delete stale/legacy paths before implementing new architecture.
5. Keep public API and docs consistent at every checkpoint.
6. Do not leave dead code or duplicate code paths.

## Correct Terminology

1. Syllable segmentation: produce `(start, peak, end)` spans from audio/features/envelopes.
2. Syllable discovery: cluster syllable tokens/embeddings into identity groups.

These are separate layers and must remain separate in architecture.

## 2026-04-07 Policy Clarification: Library-First Surface

1. Public Python APIs are the primary product surface for findsylls.
2. CLI and notebook utilities are convenience wrappers and MUST NOT contain unique behavior unavailable through Python APIs.
3. Discovery parameterization (method selection and method-specific kwargs) must be accepted by Python APIs first.
4. Orchestrator and CLI must pass configuration through to Python APIs without hidden rewrites.
5. Any feature that exists only in CLI glue is out-of-architecture and must be removed or migrated into canonical modules.

## 2026-04-07 Stale/Obsolete Patch Ledger (Initial)

Status legend: `P0` critical for release, `P1` high-priority cleanup, `P2` deferred cleanups.

### P0

1. `src/findsylls/pipeline/pipeline.py`
  - Problem: legacy alias and dual-routing paths keep old/new behavior alive simultaneously.
  - Impact: ambiguous API contract, hidden drift between public entrypoints.
  - Resolution: remove alias/backcompat paths, keep one canonical public API path.

2. `src/findsylls/evaluation/evaluator.py`
  - Problem: legacy compatibility wrappers duplicate canonical tiers-driven evaluation.
  - Impact: unnecessary branching and harder maintenance/testing.
  - Resolution: keep canonical `tiers`-based path and retire compatibility wrappers.

3. `src/findsylls/__init__.py`
  - Problem: optional embedding fallback shim masks import/configuration errors.
  - Impact: package appears partially available in invalid states.
  - Resolution: remove shim and keep embedding as first-class surface with explicit failures.

### P1

1. `src/findsylls/features/__init__.py`
  - Problem: non-canonical feature alias spellings.
  - Impact: name drift between docs, tests, and runtime behavior.
  - Resolution: enforce canonical feature names only.

2. Orchestrator/CLI-only behavior (cross-cutting)
  - Problem: wrappers may perform hidden rewrites not available in direct Python API usage.
  - Impact: script/notebook behavior diverges from library behavior.
  - Resolution: wrappers become pass-through composition layers only.

3. `src/findsylls/embedding/extractors.py` module-global model cache (`_SYLBER_SEGMENTER`, `_VG_HUBERT_SEGMENTER`)
  - Problem: process-lifetime singleton cache is stale relative to bounded workflow lifecycle policy.
  - Impact: potential unbounded retention in long runs and unclear ownership.
  - Resolution: retire module-global cache path and keep lifecycle ownership in canonical feature extractor/pipeline objects.

### Overnight Incident Classes to Resolve in Canonical Modules

1. Orchestrator return-contract mismatch causing `AttributeError: 'str' object has no attribute 'get'`.
2. Pseudo-envelope kwarg routing mismatch causing empty embedding outputs for peakdetect+neural envelope workflows.
3. Hardcoded discovery `n_clusters` causing `n_clusters > n_samples` failures on small subsets.

### 2026-04-07 Implementation Progress

1. `DONE` Orchestrator contract fix in `src/findsylls/pipeline/orchestrator.py`:
  - `segment_and_embed_audio` now returns `(embeddings, metadata)` tuple.
  - `segment_embed_and_discover` updated to consume tuple and return structured dict.

2. `DONE` Peakdetect pseudo-envelope kwarg routing fix in `src/findsylls/embedding/pipeline.py`:
  - Added envelope-kwargs flattening helper to merge nested `segmentation_kwargs['envelope_kwargs']` into top-level envelope kwargs.
  - Updated peakdetect routing to exclude envelope feature-context keys from segmenter kwargs.

3. `DONE` Discovery small-sample robustness fix in `scripts/findsylls_test_battery.py`:
  - Replaced fixed `n_clusters=20` with bounded policy `effective_clusters = max(2, min(target_clusters, n_samples))`.
  - Added reporting fields for target vs effective cluster count.

4. `DONE` Contract test alignment in `tests/test_orchestrator.py` for tuple-return semantics.

5. `TODO` Complete P0 stale-path removals in pipeline/evaluator/package init and update examples/docs accordingly.

6. `DONE` Memory lifecycle hardening (phase 1 implementation slice):
  - Added explicit extractor resource-release contract in `src/findsylls/features/base.py` (`FeatureExtractor.release`).
  - Implemented release hooks in `src/findsylls/features/hubert.py`, `src/findsylls/features/sylber.py`, and `src/findsylls/features/vg_hubert.py` to free model state after workflow boundaries.
  - Updated `src/findsylls/embedding/pipeline.py` to reuse a shared extractor across files in corpus workflows (`embed_corpus`, `embed_corpus_to_storage`) while ensuring cleanup after the run.
  - Added shared extractor injection into segmentation/peakdetect pseudo-envelope paths to avoid unnecessary per-file re-instantiation.
  - Added cleanup boundaries in `scripts/findsylls_test_battery.py` (extractor-cache teardown + explicit garbage collection points).

## 2026-04-07 Memory Safety Policy (Release-Critical)

1. Model instances may persist across files for the same active workflow configuration to avoid per-file reload overhead.
2. Model/extractor caches MUST be bounded by workflow scope and released at deterministic boundaries (end of corpus pass, end of battery stage, or process teardown).
3. Unbounded process-lifetime caches for heavy models are out-of-architecture for v3.
4. CLI/battery wrappers must not force repeated model reinitialization when canonical library pathways can reuse active instances.
5. Battery outputs should avoid carrying large per-row payloads in memory longer than needed; release temporary structures promptly.

## Current Gap Summary

1. Segmentation is mostly modularized (base classes + dispatch + presets).
2. Features are mostly modularized (extractor classes + factory).
3. Embedding stack is mostly migrated; remaining work is validation and documentation sync.
4. Discovery is implemented as a first-class module, with corpus manifests and label-mapping helpers in place.
5. Package-level orchestration rules are explicit, and the corpus discovery entrypoint is in place.
6. Current built-in discovery registry includes `agglomerative`, `kmeans`, and `minibatch_kmeans` (no built-in `dbscan` class in this repo at present).

## 2026-04-07 Memory Leak Investigation (Battery/Corpus Workflows)

### Symptoms Observed

1. Long battery runs showed process RSS growth to system-breaking levels (>80GB in user report).
2. Repeated model-loading warnings appeared across many combinations, suggesting lifecycle/caching mismatch.

### Root Cause Summary

1. Heavy extractor/model instances were reused inconsistently and lacked deterministic release boundaries.
2. Battery-stage caches (especially extractor cache) were not explicitly torn down between heavy workflow phases.
3. Discovery/evaluation loops kept large intermediate structures alive too long during long runs.
4. Some stale compatibility surfaces remain in canonical modules (see P0 ledger), increasing maintenance complexity and drift risk.

### Expected vs Unexpected Behavior

1. Expected: model instance persists across multiple files for a single active configuration to avoid per-file reload.
2. Unexpected/bug: model or cache persists indefinitely across unrelated combinations/phases without release.
3. Expected: wrappers call canonical library APIs.
4. Unexpected/bug: wrappers or helper scripts hold memory-heavy objects past workflow boundaries.

### Implemented in Current Slice

1. Added explicit extractor release contract and concrete release hooks.
2. Added shared extractor reuse in corpus embedding pathways so models are not reloaded per file.
3. Added deterministic cleanup boundaries after corpus/battery workflows (cache clear + GC points).

### Remaining Required Work (Release Gate)

1. Complete P0 stale-path removals in pipeline/evaluator/package init.
2. Add explicit memory telemetry checkpoints to battery outputs (RSS/VRAM snapshots per stage).
3. Validate full-battery memory trend is bounded and non-monotonic after cleanup boundaries.

## 2026-04-04 Compatibility Incident: Root Cause and Agreed Fixes

This section documents the segmentation/preset taxonomy bug discovered during notebook integration testing.

### Observed Problems

1. Segmentation taxonomy is mixed with end-to-end presets.
  - `list_segmenters()` currently exposes both algorithms and paper presets.
  - Users interpreted this list as pure segmentation algorithms, which produced invalid matrix construction in tests.
2. Preset names looked like segmentation algorithms in the notebook matrix.
  - `sylber`, `vg_hubert_mincut`, `vg_hubert_cls`, and `syllablelm` were treated as segmenter choices instead of recipe presets.
3. Invalid combination outcomes were difficult to interpret.
  - The notebook records many failures as `need at least one array to concatenate`, masking underlying per-file errors.
4. Some semantically invalid combinations passed.
  - The embedding pipeline currently allows broad segmentation/features mixing unless runtime crashes occur.
5. Runtime compatibility bugs amplified confusion.
  - `features='sylber'` with `device='auto'` can fail in model loading paths depending on environment.
  - `segmentation='sylber'` with `pooling='onc'` injects `detect_peaks` into segmenter kwargs, which is incompatible with the preset constructor.

### Agreed Structural Decisions

1. Keep segmentation method list algorithmic only.
  - Canonical segmentation methods: `peakdetect`, `cls_attention`, `mincut`, `greedy_cosine`.
2. Expose end-to-end paper configurations as presets in a dedicated namespace.
  - Implement a `findsylls.presets` module for recipe definitions and resolution.
  - Presets are configuration bundles, not segmentation algorithms.
3. Add strict compatibility validation at embedding entrypoints.
  - Reject ambiguous/incompatible combinations with explicit `ValueError` before expensive runtime.
4. Preserve user ergonomics with explicit presets.
  - `embed_audio` / `embed_corpus` accept a `preset` argument.
  - Preset resolution should configure segmentation/features and default kwargs consistently.
5. Update tests and notebook to enforce semantics.
  - Notebook matrix must separate segmentation algorithms from presets.
  - Compatibility checks must be executable assertions, not passive printouts.
  - Add negative tests that verify incompatible combinations are rejected.

### Implementation Checklist (Live)

Status legend: `TODO`, `IN_PROGRESS`, `DONE`.

1. `DONE` Blueprint incident contract captured (this section).
2. `DONE` Refactor segmentation dispatch taxonomy to algorithm-only canonical methods.
3. `DONE` Add dedicated `findsylls.presets` module with explicit preset registry/resolver.
4. `DONE` Add embedding compatibility validator and preset-aware parameter resolution.
5. `DONE` Fix known runtime compatibility paths (`device='auto'` handling, `detect_peaks` kwarg leakage).
6. `DONE` Add automated tests for:
  - algorithm-only `list_segmenters()` exposure,
  - preset listing/resolution,
  - explicit rejection of incompatible combinations.
7. `IN_PROGRESS` Update `findsylls_test.ipynb`:
  - new matrix structure (algorithms + presets as separate axes),
  - executable compatibility assertions,
  - explicit negative tests for rejected combinations.
8. `DONE` Append implemented outcomes back into this section as each item completes.

### Implemented Outcomes (2026-04-04)

1. Segmentation dispatch now exposes algorithmic methods only.
  - Canonical list reduced to `peakdetect`, `cls_attention`, `mincut`, `greedy_cosine`.
  - Generic algorithm wrappers now construct default feature extractors for `mincut` and `greedy_cosine`.
2. Preset recipes are now first-class and separate from segmentation discovery.
  - Added `src/findsylls/presets.py` with `list_presets`, `get_preset`, and `resolve_preset`.
  - Package exports include preset helpers.
3. Embedding pipeline now supports preset-aware configuration and strict compatibility rejection.
  - Added `preset` support to embedding pipeline entrypoints.
  - Explicitly rejects:
    - preset names passed as segmentation methods,
    - `cls_attention` with non-`vghubert` features,
    - `onc` pooling with non-`sylber` features.
4. Runtime compatibility bugs addressed.
  - Removed `detect_peaks` kwarg leakage into segmenter constructor path.
  - Normalized feature-extractor device handling by mapping `device='auto'` to `None` at extraction call sites.
5. Regression tests added and passing.
  - Added `tests/test_compatibility_contracts.py`.
  - Updated `tests/test_segmentation_normalization.py` for algorithm-only taxonomy.

### Follow-Up Clarifications (2026-04-04, Notebook Integration Pass)

This subsection captures the latest alignment decisions from integration debugging.

#### Issue 1: Segmentation and Peak Interoperability

Verified from implementation:
1. Canonical segmentation methods remain `peakdetect`, `cls_attention`, `mincut`, `greedy_cosine`.
2. All canonical segmenters currently return `(start, peak, end)` tuples:
  - `PeakdetectSegmenter`: peak is envelope maximum between valleys.
  - `CLSAttentionSegmenter`: peak is attention-peak/onset position.
  - `MinCutSegmenter`: peak is nucleus frame with highest within-segment average similarity.
  - `GreedyCosineSegmenter`: peak is nucleus frame with highest within-segment average cosine similarity.
3. Interoperability contract is therefore valid at output-shape level for pooling methods that consume nuclei.

Gaps to close:
1. Add explicit tests that assert `start <= peak <= end` for every canonical segmenter under dispatch, not only per-class assumptions.
2. Add a runtime validator in embedding path for segment tuple shape/invariants before pooling, so contract violations fail loudly.

#### Issue 2: CLS Attention Should Be Capability-Driven (Not Model-Hardcoded)

Current state:
1. `CLSAttentionSegmenter` currently instantiates `VGHuBERTFeatureExtractor` by default and effectively assumes that extractor type.
2. `VGHuBERTFeatureExtractor.extract(..., return_attention=True)` is implemented.
3. `HuBERTExtractor` and `SylberFeatureExtractor` currently expose only `extract(audio, sr)` and do not expose a CLS-attention retrieval path.
4. `FeatureExtractor` base contract has no attention-capability interface.
5. The current `CLSAttentionSegmenter` in `src/findsylls/segmentation/cls_attention.py` reduces attention to a 1-D trace and then applies peak-finding. That is architecturally acceptable as an auxiliary view, but it does not reproduce the original multi-head threshold/grouping logic.

Original vs current behavior:
1. Original script:
  - consumes raw attention weights with shape `[n_heads, tgt_len, src_len]` (or a no-CLS variant when required)
  - thresholds per head using a quantile
  - unions important indices across heads
  - groups contiguous indices into segments
  - pools segment features inside the segmentation routine
2. Current implementation:
  - collapses attention to a single 1-D score before segmentation
  - applies peak detection on that 1-D score
  - therefore matches the *envelope/peak-picking* family of methods, not the original raw-attention grouping behavior
3. Conclusion:
  - the original segmenter owns the reduction and grouping policy
  - the feature extractor should only expose attention capability
  - the 1-D envelope is useful as an auxiliary artifact, but it should not be the canonical CLS-attention segmentation path if we want identical results

Agreed direction:
1. CLS attention segmentation should request an attention-compatible signal through a feature-extractor capability interface.
2. Feature extractors should own responsibility for exposing attention when possible.
3. Segmenter should depend on capability checks, not a specific extractor class.
4. The raw-attention-to-segments algorithm should live in the CLS segmenter module, preserving the original quantile/groupby semantics.
5. A separate auxiliary helper should expose a 1-D CLS attention envelope for visualization and peakdetect experiments.

Required architecture changes:
1. Extend `FeatureExtractor` contract with an optional attention API (for example: `supports_attention` + `extract(..., return_attention=True)` or equivalent explicit method).
2. Implement attention-capable path for `HuBERTExtractor`.
3. Implement attention-capable path for `SylberFeatureExtractor` where model outputs make this possible; if not possible in a given backend/config, fail with explicit capability error.
4. Refactor `CLSAttentionSegmenter` to accept any extractor satisfying attention capability, and raise clear errors when requested capability is unavailable.
5. Move the original raw-attention segmentation logic into `CLSAttentionSegmenter` as the canonical implementation, rather than flattening to 1-D first.
6. Add an auxiliary CLS attention envelope helper for plotting and peakdetect trials, but keep it separate from the canonical segmenter.
7. Replace current compatibility check text to reference capability, not hardcoded `vghubert` naming.

### CLS Attention Parity Plan

This is the proposed architecture-compliant path to match the original implementation while keeping the package boundaries clean.

#### Canonical CLS-Attention Segmentation Path

**Location:** `src/findsylls/segmentation/cls_attention.py` → `CLSAttentionSegmenter.segment(envelope, times, **kwargs)`

**Responsibility:** Consume raw multi-head attention tensor `[n_heads, tgt_len, src_len]` (or variant without CLS token depending on model) and produce canonical `(start, peak, end)` segment tuples using original algorithm semantics.

**Algorithm Implementation:**
1. Accept raw attention tensor and per-head quantile threshold (e.g., `quantile=0.5`).
2. For each attention head:
  - Compute a threshold at the specified quantile of that head's values.
  - Identify important frames where attention >= threshold.
3. Union important indices across all heads (logical OR over binary masks).
4. Group contiguous important indices into segments.
5. For each segment, compute boundary frames (first, last) and peak frame (e.g., peak attention within segment or within-segment mean feature similarity if available).
6. Return list of `(start, peak, end)` tuples.

**Inputs:**
- `attention: np.ndarray` with shape `[n_heads, sequence_len, source_len]` or variant (multi-head attention tensor from extractor).
- `times: np.ndarray` with shape `[sequence_len]` (time positions in seconds, for boundary→time conversion).
- `quantile: float` (default `0.5`): per-head importance threshold.
- Optional: `merge_valley_tol: float` (small gap merging tolerance, e.g., 0.01s).

**Outputs:**
- `segments: List[(start, peak, end)]` where start/peak/end are times in seconds.
- `metadata: dict` containing:
  - `important_indices_union`: union of all per-head important indices (for diagnostics).
  - `num_heads_contributing`: count of heads with non-zero important indices.
  - `segments_source`: description of segmentation source (e.g., "cls_attention_raw_matrix").

**Architecture Rules:**
- Do NOT collapse attention to 1-D before segmentation (that is an auxiliary view, not canonical).
- Receive raw multi-head tensor from extractor via `extract_with_attention(...)`.
- At runtime, validate that extractor capability is available; fail explicitly if not.

#### Auxiliary CLS-Attention 1-D Envelope Path

**Location:** `src/findsylls/envelope/cls_attention.py` (new dedicated file, not in `feature_coherence.py`)

**Responsibility:** Reduce raw multi-head attention to a single 1-D envelope trace using the EXACT SAME aggregation logic as the canonical segmenter.

**Critical Architecture Principle:** 
- Feature extractors provide ONLY raw data (e.g., raw `[n_heads, tgt_len, src_len]` attention tensor)
- Both segmenter AND envelope computer independently consume that raw data
- Both apply identical aggregation transformations to remain semantically aligned
- The envelope 1-D trace represents the SAME reduction policy as the segmenter

**Public Class:**
```python
class CLSAttentionEnvelope(EnvelopeComputer):
    def compute(self, audio, sr) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 1-D envelope from raw multi-head CLS attention.
        
        Applies identical aggregation logic to CLSAttentionSegmenter:
        1. Extract raw [n_heads, tgt_len, src_len] attention
        2. Apply per-head quantile thresholding
        3. Union important indices across heads
        4. Aggregate to 1-D signal (binary mask or strength metric)
        5. Normalize to [0, 1]
        6. Return (envelope_1d, times)
        """
```

**Shared Aggregation Logic with Canonical Segmenter:**
1. Receive raw `[n_heads, tgt_len, src_len]` from `extract_with_attention(return_raw=True)`.
2. For each head: compute binary mask where attention >= quantile threshold.
3. Union masks across heads (logical OR).
4. Convert binary mask to 1-D signal by one of:
   - **Max-across-heads**: per-frame maximum of attention values where important
   - **Union-strength**: per-frame count of heads marking as important (normalized to [0,1])
   - **Mean-where-important**: per-frame mean of attention values in important regions
5. Normalize final 1-D vector to [0, 1].
6. Align temporally with audio and return `(envelope, times)`.

**Constructor Signature:**
```python
def __init__(
    self,
    feature_extractor: FeatureExtractor,
    layer: Optional[int] = None,
    quantile: float = 0.5,
    aggregation_method: str = "max",
    normalize: bool = True
):
    """
    Args:
        feature_extractor: Must provide raw attention via extract_with_attention(return_raw=True)
        layer: Transformer layer to extract attention from
        quantile: Per-head importance threshold (same as canonical segmenter default)
        aggregation_method: 'max', 'union_strength', 'mean_important' 
                          (MUST match segmenter's internal aggregation for semantic alignment)
        normalize: Whether to normalize envelope to [0, 1]
    """
```

**Architecture Rules:**
- DO NOT call any aggregation from the feature extractor; receive raw tensor only.
- Apply identical per-head quantile + union logic as canonical segmenter.
- Fail explicitly if extractor cannot provide raw multi-head attention.
- Keep implementation deterministic: same inputs → same envelope.
- Default quantile and aggregation method MUST match canonical segmenter defaults.

**Usage Contract:**
- This envelope represents the SAME reduction policy as the canonical segmenter
- Peaks in this envelope should align with boundaries found by the segmenter (both use per-head threshold + union)
- Use for visualization, diagnostics, and peakdetect-style experiments
- NOT a separate algorithm; a visualization of the SAME aggregation as segmentation

**Key Requirement for Parity:**
The envelope MUST apply the EXACT SAME aggregation steps as the segmenter so that:
- Envelope peaks align with segmenter boundaries
- Both use per-head quantile thresholding
- Both union important indices across heads
- The 1-D trace is a faithful representation of the segmenter's internal reduction

#### Feature Extractor Responsibility

1. Extractors must expose raw multi-head attention via `extract_with_attention(audio, sr, ...)` capability method.
2. Attention output should be shaped `[n_heads, sequence_len, source_len]` (or documented variant for models without CLS or with different attention layout).
3. Extractors must NOT pre-aggregate or collapse attention to 1-D; that is the responsibility of segmenter or envelope layer.
4. Time alignment: extractor must return `times` array corresponding to `sequence_len` dimension, in seconds.

#### Visualization and Experimentation

1. **Canonical segmentation plotting:** Call `CLSAttentionSegmenter.segment(...)` and plot resulting `(start, peak, end)` spans over waveform.
2. **Auxiliary envelope plotting:** Call `compute_cls_attention_envelope(...)` and overlay the 1-D trace on waveform for diagnostics.
3. **Peak-detection experiment:** Optionally call `peakdetect` segmenter on the 1-D envelope as a separate experiment, NOT as the main CLS-attention implementation.
4. Notebook should clearly label these paths and document that only canonical segmenter results are used for downstream evaluation/discovery.

#### Tests to Add (Issue 2 Closure)

1. **Regression fixture:** Deterministic toy attention tensor `[3, 10, 10]` with known per-head thresholds and expected groupings.  
  - Canonical segmenter should produce identical output to original reference implementation given same quantile and gap-merge thresholds.  
  - Regression test compares output tuples, boundary positions, and peak locations.

2. **Envelope alignment test:** Verify `compute_cls_attention_envelope(...)` returns 1-D trace `[sequence_len]` aligned with input `times` array.  
  - No time axis misalignment.  
  - Normalization methods produce expected [0, 1] bounds.

3. **Canonical segmenter raw-matrix test:** Verify `CLSAttentionSegmenter.segment(...)` does NOT collapse attention to 1-D before segmentation.  
  - Assert internal algorithm uses per-head thresholding union, not a pre-collapsed score.  
  - Confirm that using raw tensor vs pre-collapsed trace produces different segmentation results on test data.

4. **Envelope vs segmenter independence test:** Call both canonical segmenter and envelope helper on same attention input.  
  - Confirm envelope 1-D trace differs from canonical segment metadata (no accidental coupling).  
  - Ensure either path can be called independently without affecting the other.

#### Issue 3: Frequent "need at least one array to concatenate"

Observed behavior:
1. The frequent error arises when all files in a combo fail or yield zero embeddings, then notebook code calls `np.vstack([])`.
2. With `fail_on_error=False`, per-file exceptions are captured into result rows and not raised immediately.
3. The current notebook path can therefore mask the first real cause with a late concatenate failure.

Decision:
1. Keep this as a notebook/runtime diagnostics concern for current pass; user will re-run with diagnostics.
2. Add guardrails in notebook and library-level helper paths to summarize first-N per-file errors when no valid embeddings are present.

Immediate documentation contract:
1. Blueprint remains the single live incident document.
2. Do not create parallel ad-hoc analysis markdown files for this incident.

#### Issue 4: Pooling-dependent tuple validation policy

Observed gap:
1. The current embedding validator enforces `(start, peak, end)` for all pooling methods.
2. This over-constrains methods that only need span boundaries.

Agreed policy update:
1. Pooling methods that require a nucleus proxy must require explicit peaks (`onc`, `max` for current policy).
2. Pooling methods that only need interval spans (`mean`, `median`) should accept `(start, end)` and normalize internally.
3. Internal embedding metadata and pooling calls should continue to receive canonical `(start, peak, end)` tuples after normalization.
4. For `(start, end)` inputs, synthesize `peak` as midpoint `start + (end - start) / 2`.

Required implementation updates:
1. Replace strict tuple-shape validator with pooling-aware normalize+validate logic.
2. Run normalization immediately after segmentation, before pooling and metadata export.
3. Add targeted tests for accepted/rejected tuple shapes by pooling method.

#### Issue 5: Feature/segmentation compatibility with no hard exclusions

Observed gap:
1. Embedding config still contained hard gates that rejected valid cross-combinations:
  - `segmentation='cls_attention'` with non-VG-HuBERT features.
  - `pooling='onc'` with non-Sylber features.
2. These gates encoded policy restrictions rather than true interface constraints.

Agreed policy update:
1. All feature extractors must be allowed with all segmentation modules at config-validation time.
2. Segmentation and feature extraction are decoupled pipeline stages; no hard feature/segmenter exclusion checks.
3. Pooling requirements are enforced only by tuple-shape validation semantics (Issue 4), not by feature identity.

Required implementation updates:
1. Remove hard feature/segmentation and feature/pooling compatibility gates from embedding config resolver.
2. Add tests that assert these combinations are accepted.
3. Keep preset-name-as-segmentation rejection unchanged.

#### Issue 6: CLS attention raw-contract mismatch across feature extractors (2026-04-06)

Observed failure signature (notebook batch):
1. `hubert_cls_attention_*` and similar combos fail with:
  - `not enough values to unpack (expected 3, got 2)`
  - plus HuBERT warning about SDPA attention fallback when `output_attentions=True`.

Root cause confirmed from code audit:
1. Canonical CLS segmenter (`CLSAttentionSegmenter`) now expects raw multi-head attention shaped `[n_heads, seq_len, src_len]` and calls `segment_by_cls_attention_raw_matrix(...)`.
2. `HuBERTExtractor.extract_with_attention(return_raw=True)` currently returns a 2-D CLS slice `[n_heads, seq_len]` instead of 3-D raw attention.
3. `VGHuBERTFeatureExtractor.extract_with_attention(return_raw=True)` currently returns a 2-D CLS slice `[n_heads, seq_len]` instead of 3-D raw attention.
4. `SylberFeatureExtractor.extract_with_attention(return_raw=True)` currently returns a 2-D CLS slice `[n_heads, seq_len]` instead of 3-D raw attention.
5. That shape mismatch causes tuple-unpack failure in canonical CLS segmentation (`attention.shape` has 2 dims, not 3).

Architecture-compliant remediation plan:
1. **Unify raw attention contract in `FeatureExtractor` implementations**
  - `return_raw=True` MUST return 3-D attention tensors `[n_heads, seq_len, src_len]` aligned to feature frames.
  - `return_raw=False` may continue returning 1-D aggregated traces for diagnostics/envelope use.
2. **HuBERT extractor updates**
  - Return full per-head attention matrix from selected transformer layer (not pre-sliced CLS row) in raw mode.
  - Add robust frame alignment logic for feature length vs attention sequence length.
  - Load HuBERT with explicit eager attention implementation when attention outputs are requested (prepare for Transformers v5 requirement; silence fallback warning).
3. **VG-HuBERT extractor updates**
  - Return full per-head attention matrix in raw mode.
  - Keep 1-D aggregation only in non-raw mode.
  - Align layer indexing semantics with hidden states/attentions consistently.
4. **Sylber extractor updates**
  - Return full per-head attention matrix in raw mode.
  - Preserve explicit capability error when attention is genuinely unavailable.
5. **Embedding/segmentation binding updates**
  - Ensure selected `layer` from embedding entrypoints is propagated to feature-based segmenters (including `cls_attention`) so segmentation features match embedding features.
6. **Fail-fast compatibility diagnostics**
  - Add preflight check in embedding pipeline for `cls_attention`: verify extractor attention payload shape before corpus loop; raise clear contract error (`expected [n_heads, seq_len, src_len]`).
7. **Regression tests required**
  - Extractor-level tests for HuBERT/VG-HuBERT/Sylber asserting raw attention is 3-D and aligned.
  - End-to-end embedding tests for `features in {hubert, sylber, vg_hubert}` with `segmentation='cls_attention'` across poolers.
  - Negative test for non-attention extractors (`mfcc`, `melspec`) asserting explicit capability error for `cls_attention`.

Implementation constraints:
1. Do not reintroduce the deleted legacy 1-D CLS segmentation helper path.
2. Keep canonical segmentation output contract `(start, peak, end)` unchanged.
3. Keep pseudo-envelope classes separate from canonical CLS segmentation logic.

#### Issue 7: `peakdetect` must remain envelope-only (2026-04-06)

Observed failure mode from notebook testing:
1. The embedding pipeline can still route frame-level features from end-to-end extractors into `peakdetect`.
2. That causes `peakdetect` to operate on a multi-dimensional feature sequence, or on a surrogate derived directly from embeddings, instead of a true 1-D envelope.
3. The resulting syllable boundaries collapse toward the embedding frame rate, which is not a valid envelope-based segmentation result.

Architecture decision:
1. `peakdetect` is an envelope-based segmenter and must only consume a 1-D envelope.
2. If a neural extractor is used in a workflow that ends in `peakdetect`, the envelope must be produced explicitly by an envelope computer or pseudo-envelope computer first.
3. The embedding pipeline should not infer a peakdetect envelope from raw feature vectors.
4. No new default should be introduced for neural extractors in `peakdetect` workflows.

Enforcement plan:
1. Add a direct input-shape guard inside `segment_peakdetect(...)` and `PeakdetectSegmenter.segment(...)` so a 2-D feature matrix raises a clear `ValueError` immediately.
2. Keep the package-level orchestrator aware of the full feature/segmentation selection and make it fail fast when a neural feature extractor is paired with `peakdetect` but no envelope computer is explicitly provided.
3. Remove or gate any direct feature-vector-to-peakdetect fallback in the embedding pipeline.
4. Preserve discoverable valid envelope choices by name:
  - classical audio envelopes: `hilbert`, `rms`, `lowpass`, `sbs`, `theta`
  - pseudo-envelopes: `cls_attention`, `greedy_cosine`, `mincut`
5. Add regression tests that verify:
  - direct frame-feature inputs are rejected for `peakdetect`,
  - neural extractors can only reach `peakdetect` through an explicit envelope computer,
  - error messages name the accepted envelope methods and the canonical segmenter alternatives.

User notification policy:
1. Treat this as a configuration error, not a warning.
2. Raise one clear `ValueError` during preflight validation or pipeline construction, before any large corpus loop starts.
3. The message should state that `peakdetect` expects a 1-D envelope and should point the user to the envelope computer APIs or the canonical neural segmenters.

Default policy:
1. Do not add a new default for neural extractor + `peakdetect` workflows.
2. For classic envelope workflows, an explicit audio-envelope default such as `hilbert` is acceptable only when the user has already chosen envelope-based segmentation.
3. For neural extractors, there should be no implicit peakdetect default; the user must pick an envelope computer or a neural segmenter explicitly.

### Live TODO Execution Board (2026-04-04, Updated 2026-04-06)

Status legend: `TODO`, `IN_PROGRESS`, `DONE`, `BLOCKED`.

1. `DONE` Create live implementation TODO board in agent task tracker.
2. `DONE` Mirror live TODO board in this blueprint document.
3. `DONE` Add/expand tests that assert segment tuple invariants across all canonical segmenters.
4. `DONE` Add embedding-time validator for segment tuple shape/order before pooling.
5. `DONE` Improve empty-embedding diagnostics (`np.vstack([])` masking) with explicit root-cause summary.
6. `DONE` Introduce feature-extractor attention capability interface.
7. `DONE` Implement HuBERT attention extraction path under the capability interface.
8. `DONE` Implement Sylber attention extraction path under the capability interface (or explicit unsupported-capability error when unavailable).
9. `DONE` Refactor CLS attention segmenter to use capability checks instead of extractor hardcoding.
10. `DONE` Update compatibility-validation error policy to reference capabilities rather than specific model names where appropriate.
11. `DONE` Align notebook compatibility checks with capability-driven policy; remove stale "active rerun" blocker note.
12. `DONE` Implement canonical CLS-attention raw-matrix segmentation in `src/findsylls/segmentation/cls_attention.py` (per-head quantile/union/groupby strategy).
13. `DONE` Create dedicated auxiliary 1-D CLS-attention envelope module `src/findsylls/envelope/cls_attention.py` with internal aggregations.
14. `TODO` Implement regression test suite comparing canonical segmenter against original threshold/groupby semantics.
15. `DONE` Re-run-driven debugging for frequent concatenate failures (latest run outputs compared against prior run).
16. `DONE` Document pooling-aware tuple validation policy in blueprint and execution board.
17. `DONE` Implement pooling-aware tuple normalization/validation in embedding pipeline.
18. `DONE` Add tests for pooling-dependent tuple requirements (`onc`/`max` require peak, `mean`/`median` accept span-only tuples).
19. `DONE` Run targeted tests for pooling-aware validation changes.
20. `DONE` Remove hard feature/segmentation exclusion checks from embedding config.
21. `DONE` Add no-exception compatibility tests for `cls_attention` and `onc` across all feature extractors.
22. `DONE` Confirm `mean`/`median` accept both span-only and explicit `(start, peak, end)` tuples.
23. `DONE` Run targeted tests after no-exception compatibility updates.
24. `TODO` Update documentation and examples for CLS-attention canonical vs auxiliary path disambiguation.
25. `DONE` Implement canonical MinCut parity helpers in `src/findsylls/segmentation/mincut.py` (latest dynamic DP + quantile border search semantics).
26. `DONE` Align `MinCutSegmenter` wrapper defaults/behavior with latest reference parameters (`threshold`, `s`, `min_hop`, `delta`, `quantile`).
27. `DONE` Add deterministic MinCut parity regression tests against in-test reference implementation.
28. `DONE` Create dedicated MinCut pseudo-envelope module importing canonical MinCut helper functionality.
29. `DONE` Introduce shared `PseudoEnvelope` base class and refactor CLS/Greedy/MinCut pseudo-envelope classes to inherit from it.
30. `DONE` Update envelope exports/tests/docs for MinCut pseudo-envelope and base-class refactor; keep blueprint statuses synced.
31. `DONE` Normalize `return_raw=True` attention payload contract to 3-D `[n_heads, seq_len, src_len]` in HuBERT/VG-HuBERT/Sylber extractors.
32. `DONE` Add explicit eager-attention loading path for HuBERT when attention outputs are requested (Transformers v5-compatible behavior).
33. `TODO` Propagate embedding `layer` parameter into feature-based segmentation kwargs so segmentation and embedding use the same representation slice.
34. `TODO` Add embedding preflight validator for `cls_attention` raw-attention shape/capability before corpus batch loops.
35. `TODO` Add extractor contract tests + end-to-end regression tests for CLS attention combos (hubert/sylber/vg_hubert, all poolers), plus explicit failure tests for non-attention extractors.
36. `DONE` Remove direct feature-vector-to-peakdetect fallback from the embedding pipeline.
37. `TODO` Add an explicit envelope selection parameter to embedding and orchestrator entrypoints.
38. `DONE` Add preflight validation and tests for envelope-only `peakdetect` workflows.
39. `DONE` Add direct 2-D input guards to `segment_peakdetect(...)` and `PeakdetectSegmenter.segment(...)`.
40. `DONE` Add package-level orchestrator preflight for neural-extractor + `peakdetect` without explicit envelope computer.

Implemented in this pass:
1. Added segment output contract tests in `tests/test_segment_output_contracts.py` covering `peakdetect`, `cls_attention`, `mincut`, and `greedy_cosine` tuple invariants.
2. Added embedding-time segmentation output validator in `src/findsylls/embedding/pipeline.py` to enforce finite/non-negative `(start, peak, end)` with `start <= peak <= end`.
3. Added corpus-level diagnostics in `src/findsylls/embedding/pipeline.py` for the all-failed case when `verbose=True`, including first unique per-file errors.
4. Notebook loop hardening is pending final apply once active rerun writes stop; code-side diagnostics and validators are already in place.
5. Ran targeted tests successfully:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_segment_output_contracts.py tests/test_segmentation_normalization.py -q`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_compatibility_contracts.py -q`
6. Compared latest notebook run artifacts (`test_output/20260404_184423`) against prior artifacts (`test_output/20260404_163753`) and confirmed embedding failures are unchanged at combination level.
7. Confirmed hardcoded compatibility gates remain active in `src/findsylls/embedding/pipeline.py`:
  - `segmentation='cls_attention' requires features='vghubert'`
  - `pooling='onc' currently requires features='sylber'`
8. Replaced strict segment-shape validation with pooling-aware normalization in `src/findsylls/embedding/pipeline.py`:
  - `onc` and `max` require explicit `(start, peak, end)` tuples.
  - `mean` and `median` accept `(start, end)` tuples and synthesize midpoint peaks.
  - All downstream pooling + metadata paths now receive canonical `(start, peak, end)` tuples.
9. Added pooling-aware validation tests in `tests/test_embedding_tuple_validation.py` covering accepted/rejected tuple shapes and ordering validation.
10. Ran targeted tests for the new policy:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_embedding_tuple_validation.py tests/test_segment_output_contracts.py tests/test_compatibility_contracts.py -q`
11. Removed hard compatibility gates from `src/findsylls/embedding/pipeline.py` so config validation no longer rejects cross feature/segmentation combinations.
12. Added feature alias canonicalization in `src/findsylls/embedding/pipeline.py` (`vghubert` -> `vg_hubert`, `mel`/`melspectrogram` -> `melspec`).
13. Updated `tests/test_compatibility_contracts.py` to assert no-exception acceptance for:
  - `segmentation='cls_attention'` across all canonical feature extractors.
  - `pooling='onc'` across all canonical feature extractors.
14. Added explicit tuple-compatibility tests in `tests/test_embedding_tuple_validation.py` showing `mean` and `median` accept both `(start, end)` and `(start, peak, end)`.
15. Ran targeted tests for these updates:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_embedding_tuple_validation.py tests/test_compatibility_contracts.py -q`
16. Added optional attention capability contract to `src/findsylls/features/base.py`:
  - `supports_attention` property (default `False`).
  - `extract_with_attention(audio, sr, **kwargs)` method with explicit capability error by default.
17. Implemented unified `extract_with_attention(...)` API in neural extractors:
  - `src/findsylls/features/hubert.py`
  - `src/findsylls/features/sylber.py`
  - `src/findsylls/features/vg_hubert.py`
  All three now expose the same method signature and attention-head aggregation policy (`max` / `mean` / explicit head index).
18. Refactored `src/findsylls/segmentation/cls_attention.py` to capability-driven behavior:
  - segmenter now accepts any `FeatureExtractor` implementation.
  - runtime check gates only on `supports_attention` + `extract_with_attention(...)` availability.
  - removed extractor-class hardcoding dependency for attention retrieval.
19. Updated `src/findsylls/envelope/feature_coherence.py` CLS attention envelope path to use extractor capability interface instead of class-name branching.
20. Updated preset CLS segmentation path in `src/findsylls/segmentation/presets.py` to call `extract_with_attention(...)` for interface consistency.
21. Added and passed targeted capability contract tests in `tests/test_attention_capability_contracts.py`.
22. Ran targeted test suite after capability refactor:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_attention_capability_contracts.py tests/test_compatibility_contracts.py tests/test_segment_output_contracts.py -q`
23. Enforced representation-consistent segmentation in embedding flow:
  - `src/findsylls/embedding/pipeline.py` now binds feature-based segmenters (`mincut`, `greedy_cosine`, `cls_attention`) to the selected `features` value via segmentation kwargs (`feature_type`, `feature_kwargs`).
  - This removes cross-representation leakage where segmentation could be computed from a different feature family than embedding.
24. Updated `CLSAttentionSegmenter` constructor in `src/findsylls/segmentation/cls_attention.py` to accept `feature_type`/`feature_kwargs`, enabling the embedding pipeline to pass the selected representation directly into segmentation.
25. Added runtime contract tests in `tests/test_compatibility_contracts.py`:
  - `cls_attention` with `mfcc`/`melspec` now fails with capability error (no attention support).
  - `mincut`/`greedy_cosine` now execute while respecting selected features (`mfcc`, `melspec`) for segmentation and embedding consistency.
26. Ran targeted tests after representation-binding fix:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_compatibility_contracts.py tests/test_attention_capability_contracts.py tests/test_embedding_tuple_validation.py -q`
27. Extended representation-consistent segmentation contract to all canonical segmenters by binding `peakdetect` to the selected feature representation in embedding flow:
  - `src/findsylls/embedding/pipeline.py` now computes peakdetect segmentation from the selected `features` output (projected to 1-D envelope when needed) instead of a detached envelope path.
28. Added envelope feature-mode extraction support in embedding extractor adapter:
  - `src/findsylls/embedding/extractors.py` now supports `rms`, `hilbert`, `lowpass`, `sbs`, and `theta` in `extract_features`, including direct handling in the v3 adapter path.
29. Added contract coverage for `peakdetect` representation consistency:
  - `tests/test_compatibility_contracts.py` now verifies `peakdetect` works with multiple selected feature representations (`mfcc`, `theta`) and reports matching metadata.
30. Ran targeted tests after all-segmenter representation binding updates:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_compatibility_contracts.py tests/test_attention_capability_contracts.py tests/test_embedding_tuple_validation.py -q`
31. Replaced fragile class-identity checks in feature-based segmenters with capability-based extractor invocation:
  - Added shared `extract_frame_features(...)` contract in `src/findsylls/segmentation/base.py`.
  - Updated `MinCutSegmenter` and `GreedyCosineSegmenter` to use this contract instead of `isinstance(..., FeatureExtractor)` + callable fallback.
  - This fixes reload/session class-identity failures such as `'HuBERTExtractor' object is not callable` while preserving architecture (segmenters consume extractor capability, not concrete class identity).
32. Added regression coverage for extractor objects exposing `extract(audio, sr)` in `tests/test_segment_output_contracts.py` for both `mincut` and `greedy_cosine` segmenters.
33. Ran targeted tests after extractor-contract refactor:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_segment_output_contracts.py tests/test_compatibility_contracts.py tests/test_attention_capability_contracts.py -q`

### Planned: CLS Attention Parity Path Implementation (Issue 2 Closure)

34. `DONE` Extend feature extractor attention capability interface to support raw multi-head return:
  - Update `extract_with_attention(...)` signature to accept optional `return_raw: bool = False`.
  - When `return_raw=False` (default): return pre-aggregated 1-D signal (backward compatible).
  - When `return_raw=True`: return raw `[n_heads, tgt_len, src_len]` attention tensor for both segmenter and envelope.
  - Implement in all neural extractors: `HuBERTExtractor`, `SylberFeatureExtractor`, `VGHuBERTFeatureExtractor`.
  - Document clearly: raw mode is for downstream aggregation responsibility (segmenter/envelope consume raw data).
  - **Status**: Updated `src/findsylls/features/base.py` docstring; implemented return_raw in `vg_hubert.py`, `hubert.py`, `sylber.py` with identical shape validation logic.

35. `DONE` Implement canonical CLS-attention raw-matrix segmentation algorithm in `src/findsylls/segmentation/cls_attention.py`:
  - Call `extract_with_attention(audio, sr, layer=..., return_raw=True)` to get `[n_heads, tgt_len, src_len]`.
  - Implement identical per-head quantile thresholding and union logic to envelope:
    1. For each head: binarize attention where value >= quantile threshold
    2. Logical OR across all heads (union important indices)
    3. Group contiguous True frames into segments
    4. Compute segment (start, peak, end) tuples from groups
  - Return canonical `(start, peak, end)` tuples with metadata.
  - Default quantile: 0.5 (or original paper value).
  - **Status**: Added `segment_by_cls_attention_raw_matrix()` function; `CLSAttentionSegmenter.segment()` requests raw attention and uses canonical algorithm.

36. `DONE` Fix architectural inconsistency: Remove broken `CLSAttentionEnvelope` from `src/findsylls/envelope/feature_coherence.py`.
  - Current implementation receives pre-aggregated 1-D scores, breaking the contract.
  - Delete the class entirely; update module docstring to document only `SSMEnvelopeComputer` and `GreedyCosineEnvelope`.
  - Replace with new dedicated module (see item 37).
  - **Status**: Deleted lines 240-358 (entire CLSAttentionEnvelope class); updated module docstring and __all__ exports.

37. `DONE` Create dedicated CLS-attention envelope class `src/findsylls/envelope/cls_attention.py`:
  - New class `CLSAttentionEnvelope(EnvelopeComputer)` applying identical aggregation logic as canonical segmenter.
  - Constructor: `__init__(feature_extractor, layer=None, quantile=0.5, aggregation_method='max', normalize=True)`.
  - `compute(audio, sr)` method applies EXACT SAME per-head processing as segmenter:
    1. Call `extract_with_attention(audio, sr, layer=layer, return_raw=True)` → raw `[n_heads, tgt_len, src_len]`
    2. Apply per-head quantile thresholding: binary mask where attention >= quantile (IDENTICAL to segmenter)
    3. Union masks across heads: logical OR (IDENTICAL to segmenter)
    4. Aggregate to 1-D by specified method:
       - `'max'`: per-frame maximum attention in important regions
       - `'union_strength'`: per-frame count of heads marking as important (normalized to [0,1])
       - `'mean_important'`: per-frame mean attention where important
    5. Normalize to [0, 1]
    6. Return `(envelope_1d, times)` aligned to audio
  - **Default quantile and aggregation_method MUST match canonical segmenter** for semantic alignment.
  - Fail with explicit error if extractor cannot provide raw attention.
  - Envelope 1-D trace represents segmenter's internal reduction policy (peaks should align with boundaries).
  - **Status**: Implemented in `src/findsylls/envelope/cls_attention.py` (full class with docstrings, aggregation methods, error handling); updated `src/findsylls/envelope/__init__.py` to import from new module and removed import from feature_coherence.

38. `TODO` Add regression test suite `tests/test_cls_attention_parity_regression.py`:
  - **Test 38a:** Deterministic toy `[3, 10, 10]` attention tensor. Verify canonical segmenter produces expected (start, peak, end) tuples via quantile/union/groupby.
  - **Test 38b:** Same toy tensor to envelope. Verify 1-D envelope peaks align PRECISELY with segment boundaries (envelope[peak_frame] should be local maximum where important).
  - **Test 38c:** Verify envelope binary mask (per-head union) matches segmenter's internal reduction (both should be True at same frames).
  - **Test 38d:** Verify both segmenter and envelope are stateless and deterministic (same input → same output).
  - **Test 38e:** Verify both use identical quantile and aggregation logic (no divergence in thresholding, union, or reduction).
  - **Test 38f:** Integration test: extract raw attention from real audio, pass to both, verify envelope peaks correspond to segment boundaries (within 1-2 frame tolerance).

39. `TODO` Update feature extractor `extract_with_attention()` documentation:
  - Document `return_raw` parameter: raw mode provides `[n_heads, tgt_len, src_len]` for downstream aggregation consumers.
  - Add examples showing raw mode usage by both segmenter and envelope.
  - Clarify responsibility split: extractor provides data, segmenter/envelope perform aggregation.

40. `TODO` Update documentation for CLS-attention segmentation/envelope architecture:
  - Add comment in `src/findsylls/segmentation/cls_attention.py` explaining per-head quantile/union algorithm.
  - Add module docstring to `src/findsylls/envelope/cls_attention.py` with usage examples (plotting, peakdetect experiment).
  - Update `src/findsylls/envelope/feature_coherence.py` docstring reflecting `CLSAttentionEnvelope` removal.
  - Update README/docs: CLS-attention uses multi-head quantile/union strategy (not 1-D peak-finding).
  - **Document architectural principle:** Feature extractors provide raw data. Segmenters and envelopes independently apply identical aggregation logic to remain semantically aligned.

### Planned: GreedyCosine Parity Path Implementation

Audit date: 2026-04-06

Source of truth for parity:
1. Original Sylber `cossim` + `get_segment` implementation.
2. Current implementation in `src/findsylls/segmentation/greedy_cosine.py`.

Initial audit findings:
1. Core functional algorithm parity appears correct for boundaries.
  - `greedy_cosine_segment()` matches original `get_segment()` output in randomized equivalence testing (200 random cases) for `[start, end)` frame spans.
  - The two-phase logic (greedy pass + boundary refinement) is structurally equivalent.
2. Wrapper-level parity is not exact today.
  - `GreedyCosineSegmenter` class defaults diverge from canonical defaults:
    - current: `norm_threshold=0.3`, `merge_threshold=0.85`
    - canonical/original: `norm_threshold=2.6`, `merge_threshold=0.8`
  - `boundary_window` is currently exposed but not used by canonical algorithm.
  - Class-level nucleus computation is a local extension, not present in original boundary function.
  - Additional wrapper check against original `Segmenter.__call__` postprocessing (using identical hidden states):
    - Boundary parity: PASS when thresholds are matched.
    - Segment-feature parity: PASS (`np.stack([states[s:e].mean(0) ...])` matches `get_embeddings`).
  - Remaining output-contract differences vs original `Segmenter`:
    - Original returns dict with `segments`, `segment_features`, `hidden_states`.
    - Current segmenter API returns syllable tuples `(start, nucleus, end)` and exposes embeddings via `get_embeddings`.
    - Original supports list/batch input in a single call; current segmenter processes one waveform per `segment()` call.
    - Original segment times are hard-coded at 50 Hz (`segments / 50`); current wrapper uses extractor `frame_rate` (50 for Sylber, variable for others by design).
3. Envelope path is currently conceptually related but not canonical.
  - Current `GreedyCosineEnvelope` in `envelope/feature_coherence.py` uses local prototype similarity and does not reuse canonical greedy segmentation internals.

Implementation checklist (live):

41. `DONE` Freeze canonical GreedyCosine core and codify parity contract.
  - Explicitly define canonical contract in `segmentation/greedy_cosine.py`:
    - Input: frame states/features only
    - Output: boundary spans `[start, end)` identical to original algorithm
  - Add internal helper exports for canonical reusable pieces (cosine, norms/mask derivation, phase-1/phase-2 artifacts).
  - **Status**: Added canonical helper exports in `segmentation/greedy_cosine.py`:
    - `compute_greedy_cosine_norms`
    - `compute_greedy_cosine_mask`
    - `greedy_cosine_segments_to_envelope`

42. `DONE` Align `GreedyCosineSegmenter` wrapper behavior with canonical defaults and responsibilities.
  - Set wrapper defaults to canonical values (`norm_threshold=2.6`, `merge_threshold=0.8`).
  - Remove or replace unused `boundary_window` API (no dead parameters).
  - Keep architecture strict: feature extractor returns features/states; segmenter applies greedy algorithm.
  - Ensure no algorithmic behavior is hidden in extractor code paths.
  - **Status**: `GreedyCosineSegmenter` defaults now match canonical values; removed unused `boundary_window` parameter.

43. `DONE` Add deterministic parity regression tests against original reference logic.
  - New test file: `tests/test_greedy_cosine_parity_regression.py`.
  - Test functional parity of boundaries against in-test original reference implementation for:
    - random tensors,
    - edge cases (all low-norm, single-frame segments, immediate boundary flips),
    - externally supplied `norms`.
  - Test wrapper parity when using a fixed/stub feature extractor with controlled states.
  - Assert deterministic behavior (same input -> same output).
  - **Status**: Added `tests/test_greedy_cosine_parity_regression.py`; test run passed.

44. `DONE` Create dedicated GreedyCosine pseudo-envelope module mirroring CLS-attention pattern.
  - Add `src/findsylls/envelope/greedy_cosine.py` with `GreedyCosineEnvelope(EnvelopeComputer)`.
  - Import and reuse as much canonical segmentation functionality as possible from `segmentation/greedy_cosine.py`.
  - Compute a 1-D envelope that is an adaptation of canonical internals for visualization/peak-based experiments, not a separate algorithm.
  - Return `(envelope_1d, times)` aligned to frame rate.
  - **Status**: Implemented dedicated module with canonical phase-1 imports and direct merge-similarity trace adaptation (`compute_greedy_cosine_merge_similarity_trace`), avoiding segment-derived envelope construction.

45. `DONE` Resolve envelope ownership and remove non-canonical duplication.
  - Migrate envelope exports so GreedyCosine pseudo-envelope is sourced from dedicated module.
  - Update or retire current `GreedyCosineEnvelope` in `feature_coherence.py` if it conflicts with canonical parity objective.
  - Preserve one canonical implementation path for shared GreedyCosine computations.
  - **Status**: Removed legacy `GreedyCosineEnvelope` from `feature_coherence.py`; `envelope/__init__.py` now exports `GreedyCosineEnvelope` from dedicated module.

46. `DONE` Validate cross-path consistency between segmenter and pseudo-envelope.
  - Define and test alignment criterion:
    - envelope transitions/peaks correspond to canonical segment boundaries within fixed frame tolerance.
  - Ensure envelope derivation uses identical underlying states and thresholds as segmenter.
  - **Status**: Added regression assertion that pseudo-envelope output equals canonical phase-1 merge-similarity trace (`compute_greedy_cosine_merge_similarity_trace`) under the same thresholds.

47. `IN_PROGRESS` Documentation updates for GreedyCosine architecture.
  - Document canonical-vs-envelope split and shared helper usage.
  - Update README/docs and inline docstrings with extractor/segmenter responsibility boundary.
  - State explicitly: extractors output features/states; segmenter/envelope consume and transform.
  - **Status**: Inline docstrings/examples updated in segmentation/envelope modules; README/docs update still pending.

### Planned: MinCut Parity Path Implementation

Audit date: 2026-04-06

Source of truth for parity:
1. Latest SyllableLM MinCut dynamic-programming extraction path (`efficient_extraction_dp_helper`, `get_quantile_borders_helper`, `efficient_extraction`).
2. Current implementation in `src/findsylls/segmentation/mincut.py`.

Initial audit findings:
1. Current `MinCutSegmenter` is not parity-matched to the latest reference.
  - Current code uses fixed-K DP (`K ~= duration / sec_per_syllable`) over an SSM objective.
  - Latest reference uses dynamic unit search with:
    - threshold-derived max units (`m = int(threshold * n)`),
    - DP tables from chunk costs,
    - binary search over `mid_` with `delta` + `quantile` stopping criterion.
2. Current 1-D MinCut-related envelope path is not a canonical pseudo-envelope of MinCut boundary scoring.
  - `SSMEnvelopeComputer` visualizes global row-wise SSM coherence, which is useful but not the same boundary-driving signal used in MinCut DP.
3. Pseudo-envelope classes are duplicated structurally.
  - `CLSAttentionEnvelope` and `GreedyCosineEnvelope` each own repeated normalization/timing boilerplate.
  - MinCut pseudo-envelope should follow the same canonical pattern and share a base class.

Implementation checklist (live):

48. `DONE` Implement canonical MinCut parity helpers in `src/findsylls/segmentation/mincut.py`.
  - Add DP helper equivalent to latest reference (`efficient_extraction_dp_helper`) on frame embeddings.
  - Add quantile-border search helper equivalent to latest reference (`get_quantile_borders_helper`).
  - Add canonical wrapper equivalent to latest reference (`efficient_extraction`) and reuse from segmenter.
  - Preserve existing legacy fixed-K functions for backward compatibility where practical.

49. `DONE` Align `MinCutSegmenter` wrapper defaults/behavior with latest reference semantics.
  - Add canonical MinCut parameters (`threshold`, `s`, `min_hop`, `delta`, `quantile`) and defaults.
  - Ensure boundary extraction in `segment()` and `get_embeddings()` uses canonical helper path.
  - Keep extractor responsibility strict: extractors provide states, segmenter owns MinCut computation.

50. `DONE` Add MinCut parity regression tests against in-test reference implementation.
  - New tests for canonical helper parity (deterministic/randomized).
  - Wrapper parity checks for boundary times and segment embeddings.
  - Include edge cases for very short sequences and constrained hops.

51. `DONE` Create dedicated MinCut pseudo-envelope module importing canonical MinCut helpers.
  - Add `src/findsylls/envelope/mincut.py`.
  - Pseudo-envelope must visualize a pre-segmentation MinCut trace derived from canonical DP scoring, not segment outputs.
  - Return `(envelope_1d, times)` aligned to extractor frame rate.

52. `DONE` Introduce shared `PseudoEnvelope` base class and refactor CLS/Greedy/MinCut envelopes to inherit.
  - Add base class in `src/findsylls/envelope/base.py` (or adjacent canonical location in envelope module).
  - Centralize shared normalization/timing behavior.
  - Keep algorithm-specific aggregation in concrete subclasses.

53. `DONE` Update envelope exports/dispatch/tests/docs for MinCut pseudo-envelope + base-class refactor.
  - Update `envelope/__init__.py` exports.
  - Add/update tests that assert shared base-class contract and trace alignment.
  - Document canonical ownership rule: segmentation module defines algorithm helpers; pseudo-envelope imports those helpers.

54. `DONE` BLUEPRINT/live-board completion pass.
  - Update statuses for items 48-53 as implementation lands.
  - Mark final state with tested parity coverage and any residual documentation follow-ups.

## Target Architecture (End State)

### 1) Segmentation Layer (already mostly present)

- Package: `src/findsylls/segmentation/`
- Responsibility: audio to syllable spans only.
- Core pattern: base abstract classes + concrete methods + dispatch registry.

### 2) Features Layer (already mostly present)

- Package: `src/findsylls/features/`
- Responsibility: audio to frame features only.
- Core pattern: `FeatureExtractor` subclasses + factory/dispatch.

### 3) Embedding Layer (must be fully rebuilt)

- Package: `src/findsylls/embedding/`
- Responsibility: syllable spans + frame features to syllable embeddings.
- Core pattern: OOP pipeline + modular poolers.

Required structure:

- `src/findsylls/embedding/base.py`
  - abstract contracts for embedding pipeline components.
- `src/findsylls/embedding/poolers/base.py`
  - `BasePooler` abstract class.
- `src/findsylls/embedding/poolers/mean.py`
  - `MeanPooler`.
- `src/findsylls/embedding/poolers/onc.py`
  - `ONCPooler`.
- `src/findsylls/embedding/poolers/max.py`
  - `MaxPooler`.
- `src/findsylls/embedding/poolers/median.py`
  - `MedianPooler`.
- `src/findsylls/embedding/poolers/dispatch.py`
  - registry + `get_pooler` + `list_poolers`.
- `src/findsylls/embedding/pipeline.py`
  - class-based `EmbeddingPipeline`.
  - owns feature-extractor construction via `features.dispatch.get_extractor(...)`.
  - owns pooler construction via `embedding.poolers.dispatch.get_pooler(...)`.
  - canonical implementation; module-level functions are thin wrappers only.
  - thin functional wrappers only.
- `src/findsylls/embedding/storage.py`
  - keep focused on serialization formats.

Embedding responsibilities:

- `EmbeddingPipeline` orchestrates: audio/features inputs -> feature extraction (if needed) -> pooling.
- Concrete `Pooler` classes must be stateless aggregators over `(features, spans)` and must not call `features/` directly.
- `features/` owns all feature extraction algorithms; embedding must not duplicate extractor logic.

### 4) Discovery Layer (new first-class module)

- Package: `src/findsylls/discovery/`
- Responsibility: embedding matrix to cluster identities.
- Core pattern: abstract model + concrete discovery classes + dispatch + pipeline.

Required structure:

- `src/findsylls/discovery/base.py`
  - `BaseDiscoveryModel` with `fit`, `predict`, `fit_predict`.
- `src/findsylls/discovery/kmeans.py`
  - `KMeansDiscovery`.
- `src/findsylls/discovery/agglomerative.py`
  - `AgglomerativeDiscovery`.
- `src/findsylls/discovery/dispatch.py`
  - registry + `get_discovery_model` + `list_discovery_models`.
- `src/findsylls/discovery/pipeline.py`
  - class-based `DiscoveryPipeline` (corpus-level orchestration).
- `src/findsylls/discovery/types.py`
  - typed result containers.

### 5) Package-Level Orchestration Layer (explicit)

- Package: `src/findsylls/pipeline/` (extend existing package; do not create a parallel top-level orchestrator package).
- Core pattern: class-based canonical orchestrator + thin function wrappers.

Required structure updates:

- `src/findsylls/pipeline/orchestrator.py`
  - class-based `FindSyllsOrchestrator` as canonical end-to-end coordinator.
  - composes segmentation, embedding, and optional discovery pipelines.
- `src/findsylls/pipeline/pipeline.py`
  - keep public function API as thin wrappers that delegate to orchestrator methods.

Wrapper rule:

- Public function APIs remain for usability, but must contain no business logic.
- Any cross-layer logic belongs in `FindSyllsOrchestrator` methods.

## API Design Rules

1. OOP-first implementation.
2. Functional API allowed only as thin wrappers.
3. All modules follow same extension recipe:
   - add concrete class file
   - register in dispatch
   - export in `__init__.py`
   - add tests
4. No cross-layer leakage:
   - segmentation never clusters
   - discovery never computes envelopes
   - embedding never owns feature extraction algorithms
5. Orchestration is class-first:
   - canonical implementation in orchestrator classes
   - function-based APIs delegate only
6. Poolers are pure/stateless:
   - no feature extraction calls
   - no persistent model state
   - deterministic output for same inputs and parameters

## Orchestrator Placement Rules

Use orchestrator classes only where cross-component workflow logic exists.

Required orchestrators:

1. Embedding layer orchestrator:
  - `EmbeddingPipeline` in `src/findsylls/embedding/pipeline.py`.
  - Role: compose feature extraction and pooling.
2. Discovery layer orchestrator:
  - `DiscoveryPipeline` in `src/findsylls/discovery/pipeline.py`.
  - Role: compose model selection, fit/predict, and corpus-level discovery flow.
3. Package-level orchestrator:
  - `FindSyllsOrchestrator` in `src/findsylls/pipeline/orchestrator.py`.
  - Role: compose segmentation, embedding, and optional discovery into end-to-end workflows.

Modules that do not require orchestrator classes:

1. `features/`: extractor strategies and dispatch only.
2. `envelope/`: envelope computers and dispatch only.
3. `evaluation/`: metric/evaluator logic (function-first unless persistent evaluation state is added).
4. `storage/`: serialization helpers only.

Decision rule:

1. Add an orchestrator class when a module coordinates multiple components and owns workflow decisions.
2. Do not add an orchestrator class when a module only defines algorithms, factories, or utilities.

Boundary rule:

1. Orchestrators compose components; they do not re-implement underlying algorithms.
2. Algorithm ownership remains in concrete strategy classes and their dispatch registries.

## Completed Work (Already Executed)

Completed on 2026-04-01 in commit `7c32764`:

1. Stale docs removed:
  - `docs/dev/DEVELOPMENT_GUIDE.md`
  - `docs/dev/PHASE1_COMPLETE.md`
  - `docs/dev/PHASE2_SUMMARY.md`
  - `docs/dev/PHASE3_SUMMARY.md`
  - `docs/dev/PHASE4_COMPLETION_SUMMARY.md`
  - `docs/dev/PHASE5_COMPLETION_SUMMARY.md`
  - `docs/dev/TODO_INTERSPEECH.md`
  - `docs/dev/UNIFIED_ROADMAP.md`
  - `RELEASE_NOTES_v1.0.0.md`
2. Version bumped to 3.0.0 in package metadata.

Completed on 2026-04-01 (post-cleanup implementation progress):

3. Phase 2 core embedding architecture implemented:
  - Added `src/findsylls/embedding/base.py`.
  - Added `src/findsylls/embedding/poolers/` package with base + mean/max/median/onc + dispatch.
  - Added class-first `EmbeddingPipeline` in `src/findsylls/embedding/pipeline.py` with thin wrapper functions.
4. Phase 3 discovery architecture implemented:
  - Added `src/findsylls/discovery/base.py`.
  - Added `src/findsylls/discovery/types.py`.
  - Added `src/findsylls/discovery/kmeans.py` and `src/findsylls/discovery/agglomerative.py`.
  - Added `src/findsylls/discovery/dispatch.py` and `src/findsylls/discovery/pipeline.py`.
5. Phase 4 package orchestrator implemented:
  - Added `src/findsylls/pipeline/orchestrator.py` with `FindSyllsOrchestrator`.
  - Added orchestrator-facing wrappers in `src/findsylls/pipeline/pipeline.py`.
6. Phase 5 targeted tests added and passing:
  - `tests/test_poolers.py`
  - `tests/test_discovery.py`
  - `tests/test_orchestrator.py`
7. Embedding extraction refactored and routed through shared features module.
8. Discovery persistence and intrinsic fit metrics implemented:
  - Added `src/findsylls/discovery/storage.py`.
  - Added `DiscoveryPipeline.save(...)` / `DiscoveryPipeline.load(...)`.
  - Added intrinsic fit metric computation (`silhouette`, `davies_bouldin`, `calinski_harabasz`) and attached it to discovery results.
9. All references to deleted modules cleaned up.

✅ **Critical Path Complete**: Phases 2-5 fully implemented and passing validation.

## Identified Implementation Gaps (Current)

Core architecture and manifest chain are in place. The remaining work is now notebook-based end-to-end validation, docs sync, and release prep rather than more feature work.

1. The corpus-level discovery wrapper and normalized manifest chain are implemented.
  - `FindSyllsOrchestrator.discover_corpus(...)` composes embedding storage, discovery fitting/prediction, and normalized corpus manifest assembly.
  - Discovery, label, segmentation, and file manifests can be built and joined through shared helpers.

2. Discovery persistence and intrinsic metrics are implemented.
  - Discovery models can be saved and loaded.
  - Intrinsic clustering metrics are computed at fit time and stored with the discovery result.

3. The remaining pipeline work is validation.
  - The test notebook has been updated to exercise the manifest chain, module-level combo tests, and top-level orchestrator path.
  - We still need an end-to-end pass on a small real corpus subset to confirm the whole workflow behaves correctly together.

Remaining implementation cleanup/finalization:

1. ✅ Core embedding duplication cleanup — **COMPLETED**
2. ✅ Core streaming discovery tests — **COMPLETED**
3. ✅ Corpus orchestrator entrypoint — **COMPLETED**
4. ✅ Discovery corpus wrapper + label mapping artifacts — **COMPLETED**
5. ✅ Discovery model/artifact persistence helpers — **COMPLETED**
6. ⏳ Pipeline end-to-end testing — **TODO**
7. ⏳ Demo notebook and README sync — **TODO**
8. ⏳ Final release prep (build + twine check + tagging) — **TODO**

## Step-by-Step Implementation Plan

### Phase 2: Rebuild embedding as modular OOP

1. Implement `BasePooler` and concrete poolers.
2. Implement pooling dispatch registry.
3. Rewrite `EmbeddingPipeline` to compose:
  - `FeatureExtractor` (from features dispatch) for feature generation
  - `Pooler` (from embedding pooler dispatch) for aggregation
4. Explicitly enforce that poolers only consume `(features, spans)`.
5. Replace old embedding extraction internals entirely.
6. Keep wrappers:
   - `embed_audio(...)`
   - `embed_corpus(...)`
   as thin delegations to class methods.

### Phase 3: Implement discovery module

1. Add discovery base class and typed result objects.
2. Add concrete discovery models (kmeans and agglomerative first).
3. Add dispatch registry.
4. Add `DiscoveryPipeline` for corpus-level token identity assignment.
5. Expose discovery APIs from package root if desired.

### Phase 4: Implement package-level orchestrator (class-first)

1. Add `FindSyllsOrchestrator` to `src/findsylls/pipeline/orchestrator.py`.
2. Orchestrator responsibilities:
  - segment-only workflow
  - segment + embed workflow
  - segment + embed + discover workflow
3. Update function APIs in `src/findsylls/pipeline/pipeline.py` to be thin delegations.
4. Ensure function wrappers contain no business logic.

### Phase 5: Test and quality gates

Required tests:

1. Pooler unit tests (mean/onc/max/median output shapes and boundary edge cases).
2. Embedding pipeline integration tests:
   - `peakdetect + mfcc + mean`
   - `peakdetect + mfcc + onc`
3. Orchestrator tests:
  - wrapper functions delegate to class orchestrator
  - end-to-end segment + embed path returns expected shapes/metadata
4. Discovery tests:
   - deterministic labels for synthetic embeddings (seeded)
   - fit/predict contract compliance.
5. Serialization tests for embedding/discovery outputs.

### Phase 6: Documentation rewrite

1. Rewrite README around current architecture only.
2. Add one concise discovery usage section.
3. Add one architecture page (optional) generated from this blueprint.
4. Remove references to deleted docs.

### Phase 7: Release preparation

1. Confirm `src/findsylls/__init__.py` version reporting remains aligned with package metadata.
2. Update changelog with architectural rebuild notes.
3. Build and `twine check`.
4. Tag and release.

## Acceptance Criteria (Definition of Done)

1. No references to removed legacy names or deleted modules in tracked text files.
2. No duplicate implementations of the same feature extraction logic.
3. Embedding and discovery are class-based and registry-extensible.
4. Package orchestrator is class-based; function API is thin wrappers only.
5. Poolers are stateless and do not call `features/`.
6. Adding a new method in any module requires at most:
   - one new class file
   - one registry line
   - one export line
   - one test file
7. README examples execute against current APIs.
8. Build/test pipeline passes.

## Phase 6: Corpus Discovery Outputs and Persistence

**Objective**: Make discovery outputs auditable, corpus-aware, and evaluable without duplicating embedding or clustering logic.

### 6.0 Architecture Decisions (2026-04-02)

1. Streaming behavior is implemented as a **general pipeline utility** with model-specific capability checks.
  - `DiscoveryPipeline.fit_from_iterator(...)` / `fit_from_storage(...)` are general APIs.
  - A model must declare `supports_streaming=True` and implement `partial_fit(...)` to use streaming fit.
  - Non-streaming models (e.g., full-batch KMeans, Agglomerative) must use in-memory fit (global matrix/`vstack`).

2. Discovery orchestration boundary is explicit.
  - `DiscoveryPipeline` owns only **embedding -> cluster labels/metrics/artifacts** logic.
  - `FindSyllsOrchestrator` (top-level) owns **audio -> segmentation -> embedding -> discovery** workflows.
  - Do not move full corpus audio orchestration into discovery internals.

3. Manifest outputs are stage-specific and composable.
  - File-level embedding manifest (one row per source file) remains separate from syllable-level discovery manifest (one row per discovered token).
  - For large corpora, prefer normalized schema: separate `file_manifest` and `syllable_manifest` with `file_id` keys.
  - Add a segmentation manifest stage so each pipeline stage owns its own self-contained artifact, rather than storing nested lists inside a single row.

### 6.0.2 Canonical Corpus Artifact Chain

1. `segmentation_manifest.csv` (per source file)
  - One row per discovered syllable/token in a file.
  - Minimal columns: `file_id`, `segment_id`, `start`, `peak`, `end`, `segmentation_method`, `segmentation_kwargs_ref` (optional), `source_audio_path` or file lookup key.
  - Purpose: preserve the exact syllable spans produced by segmentation without coupling them to embeddings or labels.

2. `embedding_manifest.csv` (per source file)
  - One row per source file, storing `file_id`, `audio_path`, and `embedding_path` for the per-file embedding shard.
  - The manifest should reference the segmentation manifest via `file_id` and derive row alignment through `segment_id` ordering.

### 6.0.3 Key-Based Retrieval Requirement (No Positional Coupling)

This is a required invariant for corpus-scale usability and correctness:

1. Segment and embedding retrieval must be key-based, not positional.
  - It must be possible to retrieve `(file_id=A, segment_id=X)` and `(file_id=B, segment_id=Y)` without assuming row-order alignment.
  - No retrieval code may depend on "row i in embedding matrix equals segment_id i" as its only identity mechanism.

2. Per-file embedding shards must persist identity columns.
  - Required fields in each per-file NPZ/HDF5 shard:
    - `file_id` (scalar or per-row vector)
    - `segment_ids` (length `N`, where `N` is number of embedding rows)
  - Optional but recommended for integrity checks:
    - `starts`, `peaks`, `ends` arrays aligned with `segment_ids`

3. Manifest-level keys remain canonical.
  - `segmentation_manifest.csv` remains source-of-truth for segment identity (`file_id`, `segment_id`, spans).
  - `embedding_manifest.csv` remains file-level index into storage shards.
  - `discovery_manifest.csv` remains segment-level output and must continue to carry `file_id` + `segment_id`.

4. Direct retrieval helper is required.
  - Add a storage helper that can retrieve one or many embeddings by explicit keys:
    - Example API: `load_embedding_rows(manifest_path, requests=[(file_id, segment_id), ...])`
  - Behavior requirements:
    - Resolve `file_id -> embedding_path` via `embedding_manifest.csv`.
    - Load only the needed shard(s), not the full corpus.
    - Match rows by persisted `segment_ids`, not by index position.
    - Return an explicit table/object containing `file_id`, `segment_id`, and the embedding vector.
    - Raise a clear error when requested keys are missing.

5. Behavior for old shards.
  - If a shard does not include `segment_ids`/`file_id`, retrieval helper must fail closed with a clear migration message.
  - No positional fallback mode is allowed in this release.

3. `syllable_manifest.csv` (per syllable token, normalized)
  - One row per syllable token with `file_id`, `segment_id`, `embedding_id`, `start`, `peak`, `end`, `cluster_label`, `embedding_path`.
  - Can be materialized by joining segmentation + embedding + discovery outputs.

4. `label_manifest.csv` or post-hoc label columns
  - Attached after evaluation or at corpus-analysis time.
  - Keeps the same `file_id` + `segment_id` keys so labels can be joined without ambiguity.

5. Unified loader / join utility
  - Add a helper that can load and merge the stage artifacts by `file_id` / `segment_id`.
  - This helper should accept existing manifest paths or in-memory frames and return a joined corpus table for analysis, metric computation, and export.

### 6.0.1 Manifest Outputs by Stage

1. Embedding stage output: `embedding_manifest.csv` (file-level)
  - Columns: `file_id`, `audio_path`, `embedding_path`, `num_rows`, `embedding_dim`, `success`, `error`, `segmentation`, `features`, `pooling`.

2. Discovery stage output (normalized recommended):
  - `file_manifest.csv` (file-level lookup): `file_id`, `audio_path`.
  - `syllable_manifest.csv` (syllable-level): `file_id`, `embedding_id`, `start`, `peak`, `end`, `cluster_label`, `embedding_path`.
  - Optional denormalized export can include `audio_path` for convenience, but not required for canonical storage.

3. Label attachment output (post-hoc or early injection):
  - Extends syllable manifest with: `tg_labels`, `tier_labels_concat`, `primary_label`, `primary_label_peak`, `primary_label_max_overlap`, `textgrid_path`, `label_attached`, `label_source`.

4. Metric outputs (split by category):
  - `<out_base>_cluster_metrics.csv` (cluster-level precision/recall/F1, purity, support).
  - `<out_base>_label_metrics.csv` (label-level purity/support/best-cluster).
  - `<out_base>_global_metrics.json` (dataset-level summary: cluster purity, label purity, pNMI, macro/weighted F1 + intrinsic metrics when available).
  - `<out_base>_syllables.csv` (analysis-friendly export of row-level syllable assignments).

5. Model persistence outputs (high priority, pending implementation):
  - `<out_base>_model.pkl` or `<out_base>_model.joblib`.
  - `<out_base>_model_metadata.json` (method, params, fit-time, chunking config, intrinsic metrics snapshot).

### 6.1 Remaining Tasks

1. Add a thin corpus-level convenience wrapper only if the ergonomics justify it.
   - Prefer a wrapper that composes `EmbeddingPipeline` + `DiscoveryPipeline` + storage helpers.
   - The wrapper should not introduce new algorithms or hidden behavior.

2. Add syllable-level discovery tracking artifacts.
   - Persist a manifest that maps each discovered embedding back to source file identity and syllable span `(start, peak, end)`.
   - Keep this separate from the per-file embedding manifest.
   - Manifest schema should include (normalized form for large corpora):
    - `file_id`, `segment_id`, `embedding_id` (index into embedding matrix/storage)
     - `start`, `peak`, `end` (syllable boundaries)
     - `cluster_label` (discovered cluster assignment)
     - `embedding_path` (reference to source `.npz`)
   - Do not require repeating full `audio_path` in every syllable row for very large corpora.
   - Keep a separate file-level lookup table (`file_manifest`) mapping `file_id -> audio_path` and join when needed.

3. Add discovery model persistence helpers. ✅
  - Support saving/loading trained discovery models (`.pkl` or `joblib` format).
  - Store model hyperparameters and fit metadata alongside the model artifact.
  - Keep model artifacts separate from corpus label manifests.
  - This blocking item is now implemented.

4. Add discovery evaluation metrics and metadata.
   - Compute and store metrics in two groups:
     - **Intrinsic clustering metrics** (no TextGrid/labels required):
       - Silhouette score (global, and optional per-cluster summary)
       - Davies-Bouldin index
       - Calinski-Harabasz index
     - **Label-aware metrics** (require attached syllable labels from overlap logic, but no retraining):
       - Cluster purity
       - Label (syllable) purity
       - Label-normalized mutual information
       - Cluster-level precision, recall, F1 where cluster majority label is treated as cluster pseudo-ground-truth
       - Macro-averaged F1 across clusters
       - Weighted-averaged F1 across clusters
   - Persist metrics in explicit artifacts:
     - `<out_base>_global_metrics.json`: dataset-level summary metrics
     - `<out_base>_cluster_metrics.csv`: per-cluster rows (purity, precision, recall, F1, support)
     - `<out_base>_label_metrics.csv`: per-label rows (best cluster, purity, support)
     - `<out_base>_syllables.csv`: syllable-level mapping with `(start, peak, end)`, labels, and cluster ids
   - Compute timing:
     - Intrinsic metrics: compute immediately after `fit_predict` (or `fit` + `predict`) when embeddings and cluster labels are available
     - Label-aware metrics: compute after TextGrid labels are attached to syllable rows (post-fit, post-export, no retrain)
   - Do NOT require retraining or full corpus relabeling to compute any of the above once artifacts are saved.

5. Add corpus-level discovery export/load utilities.
   - Export discovered labels with file/span metadata to CSV or a similar tabular format.
   - Provide load helpers for downstream analysis and notebook use.
  - Export function should include computed evaluation metrics in summary and per-cluster/per-label files.
  - Provide a unified loader that can join segmentation, embedding, discovery, and label manifests into one analysis-ready corpus table.

6. Add corpus label attachment utilities (evaluation integration).
  - Implemented as `evaluation.attach_textgrid_labels_to_manifest(...)`.
  - Supports post-hoc attachment to existing discovery manifests/bundles and early-injection before saving manifests.
  - Accepts `wav_paths` and `textgrid_paths` as either lists or glob patterns.
  - Uses the same wav/TextGrid pairing logic as the evaluation pipeline.
  - Emits normalized label columns (`tg_labels`, `primary_label`, `primary_label_peak`, `primary_label_max_overlap`, `textgrid_path`) so downstream metric code can stay identical.

7. Add split discovery artifact export utilities.
  - Implemented as `evaluation.export_discovery_label_artifacts(...)`.
  - Writes separate files by category:
    - `<out_base>_syllables.csv`
    - `<out_base>_cluster_metrics.csv`
    - `<out_base>_label_metrics.csv`
    - `<out_base>_global_metrics.json`
  - Stores cluster-level precision/recall/F1, label purity, cluster purity, label-normalized MI, macro-F1, and weighted-F1 in category-specific outputs.

8. Validate the end-to-end corpus workflow on a small real corpus subset.
   - Embed to storage.
   - Fit discovery from storage.
   - Export labels, verify mapping back to input files.
   - Compute and log evaluation metrics.
   - Verify metrics can be read without re-embedding or retraining.
   - Validate both label pathways:
     - post-hoc label attachment on existing manifests
     - early label injection at corpus analysis time

9. Add corpus-scale orchestration path (Discovery module entrypoint).
   - Add a corpus entrypoint at top-level orchestrator that accepts audio files as list or glob and orchestrates segmentation -> embedding -> clustering.
   - For large corpora (e.g., LibriSpeech 100h), enforce low-memory mode by default:
     - incremental writing of embedding manifests and per-file embedding shards
     - discovery fit from storage iterators (chunked `partial_fit` when supported)
     - no global `vstack` unless algorithm requires non-streaming fit
   - Persist run metadata (chunk size, batch size, model type, timing) for reproducibility.
   - Clarify model lifecycle: load end-to-end feature/segmentation models only during embedding stage; release references after stage completion where possible.
  - The top-level orchestrator should create/consume segmentation manifests, not DiscoveryPipeline.

10. Implement key-addressable embedding retrieval and shard identity persistence.
  - Persist `file_id` and `segment_ids` in each per-file embedding shard.
  - Add direct retrieval helper for `(file_id, segment_id)` lookup without positional assumptions.
  - Add tests covering:
    - exact key retrieval,
    - missing key behavior,
    - multi-file targeted retrieval,
    - explicit failure on legacy shards lacking IDs.

### 6.2 Expected Outcomes

1. Discovery output is explicit, reproducible, and evaluable.
2. Corpus-level relabeling is inspectable per file and per syllable with full syllable metadata.
3. Trained models and corpus label artifacts can be saved and restored.
4. Evaluation metrics (intrinsic and label-aware) are computable from saved artifacts without model retraining or corpus relabeling.
5. The wrapper layer stays thin and does not own core algorithm logic.
6. Discovery output artifacts support downstream analysis without requiring access to original embeddings (except for optional ground-truth comparison).
7. Label attachment works in both modes (post-hoc and early injection) with consistent outputs from list/glob wav/TextGrid inputs.
8. Corpus-scale orchestrator path runs without requiring full in-memory embeddings.

---

## Critical Issues Raised (2026-04-01)

This section captures major package-level concerns raised during notebook and architecture review.
These are tracked as design + implementation tasks and should be resolved before release finalization.

### Issue 1: Segmentation API inconsistency (CLS attention path)

**Observed Problem**:
- `MinCutSegmenter` and `GreedyCosineSegmenter` are exposed as class-based segmenters.
- CLS attention is now exposed as the canonical `CLSAttentionSegmenter` class, and the VG-HuBERT preset delegates to it. The legacy 1-D helper path has been removed from the preset layer.
- Dispatch does not currently register a generic `'cls_attention'` segmenter, even though docs/test matrices list it.
- `billauer` alias still appears in dispatch/pipeline paths even though canonical method naming is `peakdetect`.
- `syllablelm` is registered as a method; implementation exists in presets, but must be kept only if dependency/runtime validation passes in tests.

**Evidence**:
- `findsylls_demo.ipynb` does use/test CLS attention paths (VG-HuBERT CLS envelopes and method configs).
- Package internals contain CLS segmentation logic, but registration/export pattern is inconsistent.

**Proposed Plan**:
1. Add `CLSAttentionSegmenter` class in `src/findsylls/segmentation/cls_attention.py` (feature-based wrapper, analogous to MinCut/GreedyCosine style).
2. Register `'cls_attention'` in `src/findsylls/segmentation/dispatch.py`.
3. Export class in `src/findsylls/segmentation/__init__.py` and document it in module docs.
4. Add tests ensuring `get_segmenter('cls_attention')` works and returns valid `(start, peak, end)` tuples.
5. Align naming in docs/notebooks to avoid mixed spellings (`vghubert_cls` vs `vg_hubert_cls`) unless intentionally aliased.
6. Remove `'billauer'` from public dispatch and wrapper checks; keep one canonical classical method name: `'peakdetect'`.
7. Validate `syllablelm` as a supported method via test gate:
  - keep only if preset class is importable and executable in optional-dependency test lane,
  - otherwise remove from advertised supported-method lists and keep it undocumented until fully validated.

### Issue 2: Corpus-scale discovery is currently in-memory

**Observed Problem**:
- Current discovery examples and notebook usage concatenate all embeddings with `np.vstack(...)` before fitting.
- This fails to scale to large corpora (e.g., LibriSpeech 100h), where embeddings may exceed RAM.

**Current Capability**:
- Embeddings can be saved to disk (NPZ/HDF5).
- Discovery pipeline currently accepts in-memory arrays only.
- Existing KMeans implementation uses full-batch sklearn `KMeans`, not incremental fitting.

**Proposed Plan**:
1. Add disk-backed embedding iterator utilities in `embedding/storage.py`:
  - chunked iteration over NPZ/HDF5 embeddings
  - optional metadata/manifest emission (`audio_path`, offsets, counts).
2. Introduce streaming discovery model option:
  - `MiniBatchKMeansDiscovery` with chunked `partial_fit`.
3. Extend `DiscoveryPipeline` with general storage-based APIs:
  - `fit_from_storage(...)`
  - `predict_from_storage(...)`
  - `discover_from_storage(...)`.
4. Implement model capability checks in discovery models:
  - streaming-capable models (e.g., MiniBatchKMeans) implement incremental fit path,
  - non-streaming models (e.g., agglomerative) raise clear `NotImplementedError` for streaming fit and document memory requirements.
5. Keep in-memory `fit/predict/discover` APIs unchanged for small/medium datasets.
6. Add large-corpus notebook/example using disk-backed clustering without global `vstack`.

### Issue 3: Manifest-first corpus embedding workflow

**Status**: implemented.

The embedding stack now includes a first-class storage path:
- `embed_corpus_to_storage(...)` exists on `EmbeddingPipeline` and as a thin module wrapper.
- `BaseEmbeddingPipeline` includes the storage contract.
- `embedding/storage.py` now exposes manifest write/load helpers and a chunked iterator for discovery.
- The remaining work is discovery-side corpus relabeling and artifact persistence, not embedding-side storage plumbing.

### Architecture Decision: General vs Model-Specific Streaming APIs

Decision:
1. Storage-iterator APIs in `DiscoveryPipeline` are **general** (pipeline-level contract).
2. Streaming execution is **model-capability-specific** (model-level implementation).

Rationale:
1. Keeps pipeline API stable as new streaming clusterers are added.
2. Avoids hard-coding MiniBatchKMeans logic into orchestration layer.
3. Preserves clean OOP separation: pipeline orchestrates, models implement algorithm-specific fit semantics.

### Approval Gate

Implementation of these three issues should proceed in this order:
1. Segmentation consistency + CLS registration/tests.
2. Manifest-first embedding API.
3. Disk-backed streaming discovery pipeline.

Each step should include tests and notebook/docs updates before moving to the next.

### Implementation TODO Checklist (Execution Tracker)

Use this as the live execution order for the next implementation pass.

1. Segmentation manifest and consistency hardening:
  - [x] Add a dedicated `segmentation_manifest.csv` stage with `file_id`, `segment_id`, `start`, `peak`, `end`, and segmentation metadata.
  - [x] Ensure segmentation output can be joined forward into embedding/discovery manifests by `file_id` and `segment_id`.
  - [x] Remove `billauer` alias from all public APIs and docs.
  - [x] Add `CLSAttentionSegmenter` class and register `cls_attention` in dispatch.
  - [x] Ensure segmentation method naming is canonical (`peakdetect`, `vg_hubert_cls`, `vg_hubert_ssm`), with notebook alias mapping for `featssm` and `greedy_cosine` reporting.
  - [x] Add/refresh segmentation tests for dispatch registration and output contracts.

2. SyllableLM support gate:
  - [ ] Validate `syllablelm` optional dependency path in tests.
  - [ ] If validation fails, remove from advertised supported method list (do not silently claim support).

3. Embedding OOP storage path:
  - [x] Extend `BaseEmbeddingPipeline` with `embed_corpus_to_storage(...)`.
  - [x] Implement `EmbeddingPipeline.embed_corpus_to_storage(...)` as canonical logic.
  - [x] Keep module-level `embed_corpus_to_storage(...)` as thin wrapper only.
  - [x] Add manifest schema + load helpers + chunk iterators in `embedding/storage.py`.
  - [x] Add a corpus manifest join utility that merges segmentation + embedding + discovery + labels into one analysis-ready table.

4. Discovery streaming architecture:
  - [x] Add `MiniBatchKMeansDiscovery` model.
  - [x] Extend `DiscoveryPipeline` with general storage-based APIs.
  - [x] Enforce model capability checks for streaming (`supports_streaming`, `partial_fit`).
  - [x] Keep non-streaming models explicit about memory constraints.
  - [x] Keep the full corpus audio orchestration in `FindSyllsOrchestrator`, not `DiscoveryPipeline`.

5. Notebook/documentation alignment:
  - [x] Update `findsylls_test.ipynb` to remove 2-method segmentation claim and align matrix methods/aliases with live dispatch availability.
  - [x] Fix module-level manifest setup in `findsylls_test.ipynb` to resolve `file_id` via stable identity mapping rather than positional `.loc` assumptions.
  - [ ] Replace in-memory-only discovery quick-start with storage-backed option.
  - [ ] Align matrix counts with representative vs exhaustive testing policy.

6. Validation and release guardrails:
  - [x] Run targeted tests for segmentation/embedding/discovery changes.
  - [ ] Run static/compile checks on touched modules.
  - [ ] Update README snippets to match new APIs.

7. Key-addressable embedding retrieval:
  - [x] Persist `file_id` and `segment_ids` in each per-file embedding shard.
  - [x] Add direct retrieval helper for `(file_id, segment_id)` without positional assumptions.
  - [x] Add tests for key retrieval, missing keys, and legacy shard fallback behavior.

### Execution Status Update (Current Pass)

Completed in code during this pass:
1. Removed `billauer` public alias paths in segmentation dispatch/wrappers; canonical method now `peakdetect`.
2. Added first-class `CLSAttentionSegmenter` and registered `cls_attention` in segmentation dispatch.
3. Added storage-first embedding API and manifest utilities:
  - `BaseEmbeddingPipeline.embed_corpus_to_storage(...)`
  - `EmbeddingPipeline.embed_corpus_to_storage(...)`
  - thin wrapper `embed_corpus_to_storage(...)`
  - `write_embedding_manifest(...)`
  - `load_embedding_manifest(...)`
  - `iter_embeddings_from_manifest(...)`.
4. Added streaming discovery model + pipeline APIs:
  - `MiniBatchKMeansDiscovery`
  - `DiscoveryPipeline.fit_from_storage(...)`
  - `DiscoveryPipeline.predict_from_storage(...)`
  - `DiscoveryPipeline.discover_from_storage(...)`.
5. Validated `tests/test_streaming_workflows.py`.
6. Validated `syllablelm` loads successfully as an optional dependency.
7. Updated `README.md` with corpus-scale storage and streaming examples.
8. Added key-addressable shard identity persistence in `embed_corpus_to_storage` output shards:
  - persisted `file_id`, `file_ids`, and `segment_ids` arrays in each per-file NPZ.
9. Added direct key-based retrieval helper in storage utilities:
  - `load_embedding_rows(manifest_path, requests=[(file_id, segment_id), ...])`
  - strict-by-default behavior for missing identity arrays (explicit opt-in positional fallback).
10. Added targeted tests for identity retrieval behavior:
  - `tests/test_embedding_storage_retrieval.py`.
11. Fixed notebook module-manifest assembly to avoid fragile positional indexing in `findsylls_test.ipynb`.
12. Ran targeted validation:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_embedding_storage_retrieval.py tests/test_manifests.py -xvs` (passed).

Still pending from the blueprint:
1. Pipeline end-to-end testing on a small real corpus subset.
2. Validate the updated `findsylls_test.ipynb` end-to-end and then sync `findsylls_demo.ipynb`.
3. README evaluation section documenting available metrics and interpretation.

---

## Phase 7: Documentation Refresh (Post-Validation)

**Objective**: Update README and docstrings to reflect current architecture, tested workflows, and evaluation capabilities.

### 7.1 Tasks

1. Update `findsylls_demo.ipynb` to show the final corpus workflow including evaluation step.
2. Keep `findsylls_test.ipynb` as the validation notebook for storage-backed discovery, label export, and evaluation metrics.
3. Refresh `README.md` with corpus embedding, storage, discovery, and evaluation examples.
4. Update `embed_corpus()` / discovery docstrings to match the final API shape.
5. Document the persistence format for manifests and saved discovery artifacts.
6. Add evaluation section to README explaining:
   - Available intrinsic clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).
   - How to load and interpret discovery evaluation results.
   - (Optional) How to compute downstream metrics (purity, NMI) if ground truth becomes available.

### 7.2 Acceptance Criteria

- Demo, test notebook, and README all reflect the same discovery workflow including evaluation.
- README contains working copy-paste examples for:
  - single file embedding
  - corpus embedding with storage
  - corpus discovery + relabeling
  - interpreting discovery evaluation metrics
- All docstrings reference current module paths and current persistence helpers.
- Links between modules remain correct.
- Evaluation section clearly explains which metrics are computed automatically and which require additional ground truth.

---

## Phase 8: Release Preparation

**Objective**: Finalize code, build, and prepare for release.

### 8.1 Tasks

1. Run the focused discovery/storage regression tests.
2. Run the full test suite: `pytest tests/ -v`.
3. Run compile checks: `python -m py_compile src/findsylls/**/*.py`.
4. Update CHANGELOG with discovery persistence and docs highlights.
5. Build distribution: `python -m build`.
6. Check build: `twine check dist/*`.
7. Tag release: `git tag -a v3.1.0`.
8. Upload to PyPI: `python -m twine upload dist/*` (if desired).

### 8.2 Acceptance Criteria

- All tests pass
- No compile errors
- Build produces valid wheels/tarballs
- `twine check` reports no errors

---

## Agent Execution Notes

When resuming from this blueprint:

1. Phase 1 cleanup is complete; begin from Phase 2.
2. Do not preserve backward compatibility aliases.
3. Keep phases in separate commits and independently buildable.
4. If conflicts appear between docs and code, contracts defined here win.
5. **Phases 2-5 complete as of latest session (2026-04-01)**; resume from Phase 6 for corpus discovery outputs and persistence.
6. Phase 6 is now a *corpus-output and persistence* phase, followed by notebook/README sync and release prep.
