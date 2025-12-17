# Copilot Project Instructions: `findsylls`

Concise guidance for AI coding agents working on this unsupervised syllable segmentation + evaluation toolkit.

## 1. Core Purpose
Pipeline to (a) load & normalize speech audio, (b) derive an amplitude (or modulation) envelope with pluggable methods, (c) segment syllable-like units (currently peak/valley based), and (d) evaluate predictions against TextGrid annotations at multiple granularities (nuclei, syllable boundaries/spans, word boundaries/spans) with aggregated metrics.

## 2. High-Level Architecture
- `audio/` – I/O + normalization. Key: `load_audio()` prefers torchaudio if installed; falls back to soundfile/librosa; always returns mono float32 at target samplerate. `match_wavs_to_textgrids()` performs fuzzy filename alignment (suffix stripping, alt index, prefix & substring heuristics). Changes here impact the entire pipeline.
- `envelope/` – Envelope computation dispatch (`dispatch.get_amplitude_envelope`). Supported methods: `rms`, `hilbert`, `lowpass`, `sbs` (spectral band subtraction), `gammatone`, `theta`. Each method returns `(envelope, times)` arrays aligned to audio duration. When adding a method: implement `compute_*` returning `(env, t)` and register in `dispatch.py`.
- `segmentation/` – `dispatch.segment_envelope` chooses algorithm (currently `peaks_and_valleys`; stubs for `mermelstein`, `theta` commented). Segmentation functions must accept `envelope`, `times`, **kwargs, and return a list of `(start, peak, end)` tuples (peak is the nucleus proxy). Downstream evaluation derives `peaks = [peak]` and `spans = [(start, end)]`.
- `evaluation/` – Orchestrated by `evaluation.evaluator.evaluate_segmentation`. Produces a dict with dynamically-generated keys based on specified tiers (e.g., `nuclei`, `syllable_boundaries`, `word_spans`). The `nuclei` evaluation uses the 'phone' tier from tiers dict for vocalic intervals. Boundary & span evaluators compare predicted spans against reference intervals extracted from any TextGrid tier. Each metric dict contains counts like `TP`, `Ins`, `Del`, `Sub`. Tier specification uses the `tiers` dict mapping tier names to indices (e.g., `tiers={'phone': 2, 'syllable': 1, 'word': 0}`). Legacy parameters (`phone_tier`, `syllable_tier`, `word_tier`) are still supported for backward compatibility.
- `pipeline/` – User-facing APIs. `pipeline.segment_audio()` = single-file end‑to‑end; `pipeline.run_evaluation()` batches matching wav/TextGrid pairs, executes segmentation + evaluation, tags results with chosen envelope/segmentation method, and flattens via `pipeline.results.flatten_results()` then optional aggregation with `aggregate_results()`.
- `parsing/` – TextGrid parsing: extracts phone / syllable / word intervals; filters vowels via `SYLLABIC` set; returns clean interval tuples. Note: `extract_syllable_intervals` returns dict with `intervals` & `deleted` – evaluation code expects raw list (handles None). Be consistent.
- `config/` – Static constants: evaluation method types (`EVAL_METHOD_TYPES` = `['nuclei', 'boundaries', 'spans']`), vowel sets (`SYLLABIC`), default tolerance. Note: actual evaluation keys are generated dynamically based on tier names, not hardcoded.
- `plotting/` – Visualization helpers (not yet referenced in pipeline here but likely expect flattened DataFrames with `audio_file`, `tg_file`, etc.).

## 3. Data & Object Flow (Typical)
`wav path` -> `audio.utils.load_audio` -> `(audio, sr)` -> `envelope.dispatch.get_amplitude_envelope` -> `(envelope, times)` -> `segmentation.dispatch.segment_envelope` -> `[(start, peak, end)...]` -> derive `peaks` & `spans` -> `evaluation.evaluator.evaluate_segmentation` + TextGrid tiers -> metrics dict -> `results.flatten_results` -> tidy `pd.DataFrame` keyed by `eval_method`.

## 4. Evaluation Conventions
- Evaluation keys are generated dynamically: `{tier_name}_boundaries` and `{tier_name}_spans` for each tier specified in the `evaluate_segmentation` call, plus `nuclei` if a phone tier is provided.
- Tolerance (boundary matching) default 0.05s – propagate via `tolerance` arg; do not hardcode elsewhere (except current segmentation onset heuristics).
- Substitution count is meaningful for span metrics; for nuclei/boundary F1 computation substitution should remain 0 (see note in `aggregate_results`).
- The `flatten_results` function dynamically detects all evaluation keys in results, so no constant maintenance is required when adding new tiers.

## 5. Adding a New Segmentation Method
1. Implement `segment_<name>(envelope, times, **kwargs) -> List[(start, peak, end)]` in `segmentation/`.
2. Register in `segmentation/dispatch.py` with a new `elif method == "<name>":` branch.
3. Ensure any parameters are passed via `segmentation_kwargs` in `pipeline.segment_audio` / `run_evaluation`.
4. Keep return shape consistent; evaluation logic expects ability to form `peaks` and `spans` exactly as in `run_evaluation`.

## 6. Adding a New Envelope Method
1. Implement function returning `(envelope: np.ndarray, times: np.ndarray)` (times in seconds aligned sample-wise) in `envelope/`.
2. Add a branch in `envelope/dispatch.get_amplitude_envelope`.
3. Avoid altering existing method signatures; downstream relies on `(env, times)` tuple.

## 7. Filename Matching Heuristics
`audio.utils.match_wavs_to_textgrids` performs multi-step fuzzy matching (exact base, stripped suffix, alt index removing short tokens, prefix, substring). When modifying, maintain deterministic pairing order (inputs sorted). Return parallel ordered lists of matched TextGrid and wav paths; downstream assumes zipped alignment.

## 8. Common Pitfalls
- Passing tier parameters: prefer using the `tiers` dict (`tiers={'phone': 2, 'syllable': 1, 'word': 0}`). Legacy parameters (`phone_tier=2, syllable_tier=1, word_tier=0`) still work for backward compatibility. The `tiers` dict allows arbitrary tier names and the 'phone' tier is used for nuclei evaluation.
- `extract_syllable_intervals` returns dict; evaluation code currently calls it for all tiers. The dict always has `intervals` and `deleted` keys; check `len(intervals)` to verify data exists.
- Ensure new envelope methods produce time arrays identical length to envelope to keep peak picking accurate.
- The `flatten_results` function now dynamically discovers evaluation keys, so adding new tiers doesn't require updating any constants.

## 9. Performance Notes
- Resampling uses `librosa.resample` if torchaudio path not taken; large batch processing may benefit from enabling torchaudio for GPU or using streaming, but current code assumes in-memory arrays.
- Peak detection uses external `findpeaks.peakdetect`; parameters: `delta`, `lookahead`, merging valleys by temporal gaps > `merge_valley_tol`.

## 10. Minimal Usage Examples
Single file segmentation:
```python
from findsylls.pipeline.pipeline import segment_audio
sylls, env, t = segment_audio('audio.wav', envelope_fn='hilbert', segment_fn='peaks_and_valleys', segmentation_kwargs={'delta':0.02})
```
Evaluation with flexible tier specification:
```python
from findsylls.evaluation.evaluator import evaluate_segmentation
result = evaluate_segmentation(
    peaks=peaks, spans=spans, textgrid_path='file.TextGrid',
    tiers={'phone': 2, 'syllable': 1, 'word': 0, 'phrase': 3}
)
```
Aggregate:
```python
from findsylls.pipeline.results import aggregate_results
summary = aggregate_results(results_df, dataset_name='MyCorpus')
```

## 11. Extensibility Checklist (Do when adding features)
- Register new method in proper dispatch file.
- Evaluation is now fully dynamic – just pass tier specifications via `phone_tier`, legacy params, or the `tiers` dict. No constant updates needed.
- Keep return signatures stable (see sections 5 & 6).
- Add any new dependencies explicitly (project currently implicit; consider adding requirements file if growing).

## 12. Open TODO / Gaps Not Implemented
- `generate_syllable_intervals` is placeholder (returns []). Supplying real generation logic would enable `syllable_tier=-1` path.
- Alternative segmentation algorithms (`mermelstein`, `theta`) are commented out; implement & register when ready.
- No explicit CLI / README; adding one would aid reproducibility.

---
Feedback welcome: Clarify tier indexing expectations? Need guidance on plotting or adding tests? Let me know what to refine.
