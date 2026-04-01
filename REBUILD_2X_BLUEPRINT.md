# findsylls 2.x Full Rebuild Blueprint

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

## Current Gap Summary

1. Segmentation is mostly modularized (base classes + dispatch + presets).
2. Features are mostly modularized (extractor classes + factory).
3. Embedding stack is not fully migrated and still contains stale coupling and duplicated logic.
4. Discovery is not implemented as a first-class module.
5. Documentation set is outdated and internally inconsistent.

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
  - thin functional wrappers only.
- `src/findsylls/embedding/storage.py`
  - keep focused on serialization formats.

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

## Cleanup Plan (Execute BEFORE New Implementation)

Run this cleanup as one explicit phase and commit it separately.

### A) Delete stale docs and plans

Delete these files/directories:

- `docs/dev/DEVELOPMENT_GUIDE.md`
- `docs/dev/PHASE1_COMPLETE.md`
- `docs/dev/PHASE2_SUMMARY.md`
- `docs/dev/PHASE3_SUMMARY.md`
- `docs/dev/PHASE4_COMPLETION_SUMMARY.md`
- `docs/dev/PHASE5_COMPLETION_SUMMARY.md`
- `docs/dev/TODO_INTERSPEECH.md`
- `docs/dev/UNIFIED_ROADMAP.md`
- `RELEASE_NOTES_v1.0.0.md`

If still present and stale:

- `docs/EMBEDDING_PIPELINE.md`
- `docs/EMBEDDING_QUICKREF.md`

### B) Delete stale/duplicate code paths

1. Remove embedding-internal feature extraction duplication that bypasses `features/`.
2. Remove any imports in embedding that reference deleted module paths.
3. Remove compatibility aliases and legacy naming helpers that no longer belong in 2.x.
4. Remove any outdated test notebooks from `tests/` used as scratch artifacts.

### C) Keep only living docs during rebuild

Keep:

- `README.md`
- `CITATION.cff`
- `CHANGELOG.md`
- this file: `REBUILD_2X_BLUEPRINT.md`

Optional temporary keep if actively maintained:

- `docs/FINDSYLLS_USER_GUIDE.md`

## Step-by-Step Implementation Plan

### Phase 0: Stabilize repository state

1. Ensure branch is clean and synchronized.
2. Confirm package version strategy for rebuild release (target 2.1.0 or 3.0.0 depending on scope).

### Phase 1: Cleanup commit (mandatory gate)

1. Execute cleanup plan above.
2. Verify no dead imports remain.
3. Build package.
4. Commit as `cleanup: remove stale docs and legacy paths`.

### Phase 2: Rebuild embedding as modular OOP

1. Implement `BasePooler` and concrete poolers.
2. Implement pooling dispatch registry.
3. Rewrite `EmbeddingPipeline` to compose:
   - `Segmenter` (from segmentation dispatch)
   - `FeatureExtractor` (from features dispatch)
   - `Pooler` (from embedding pooler dispatch)
4. Replace old embedding extraction internals entirely.
5. Keep wrappers:
   - `embed_audio(...)`
   - `embed_corpus(...)`
   as thin delegations to class methods.

### Phase 3: Implement discovery module

1. Add discovery base class and typed result objects.
2. Add concrete discovery models (kmeans and agglomerative first).
3. Add dispatch registry.
4. Add `DiscoveryPipeline` for corpus-level token identity assignment.
5. Expose discovery APIs from package root if desired.

### Phase 4: Test and quality gates

Required tests:

1. Pooler unit tests (mean/onc/max/median output shapes and boundary edge cases).
2. Embedding pipeline integration tests:
   - `peakdetect + mfcc + mean`
   - `peakdetect + mfcc + onc`
3. Discovery tests:
   - deterministic labels for synthetic embeddings (seeded)
   - fit/predict contract compliance.
4. Serialization tests for embedding/discovery outputs.

### Phase 5: Documentation rewrite

1. Rewrite README around current architecture only.
2. Add one concise discovery usage section.
3. Add one architecture page (optional) generated from this blueprint.
4. Remove references to deleted docs.

### Phase 6: Release preparation

1. Align `src/findsylls/__init__.py` version reporting with package metadata.
2. Update changelog with architectural rebuild notes.
3. Build and `twine check`.
4. Tag and release.

## Acceptance Criteria (Definition of Done)

1. No references to removed legacy names or deleted modules in tracked text files.
2. No duplicate implementations of the same feature extraction logic.
3. Embedding and discovery are class-based and registry-extensible.
4. Adding a new method in any module requires at most:
   - one new class file
   - one registry line
   - one export line
   - one test file
5. README examples execute against current APIs.
6. Build/test pipeline passes.

## Agent Execution Notes

When resuming from this blueprint:

1. Start with Phase 1 cleanup commit before writing new architecture code.
2. Do not preserve backward compatibility aliases.
3. Do not merge cleanup and rebuild into one giant commit.
4. Keep each phase independently buildable.
5. If conflicts appear between docs and code, code contracts defined here win.
