"""
End-to-end pipeline parity: findsylls vs reference implementations.

Tests three published pipelines against their reference implementations.
All tests use real audio from test_samples/ — no synthetic data.

Pipelines under test
--------------------
1. Sylber + Greedy Cosine  (Park et al., ICLR 2025)
2. VG-HuBERT (word ckpt, layer 9) + CLS attention  (Peng & Harwath, Interspeech 2022)
3. VG-HuBERT (syllable ckpt, layer 8) + MinCut DP  (Peng et al., Interspeech 2023)

Skip conditions
---------------
- All tests skip when test_samples/SP20_117.wav is absent.
- Sylber algorithm tests skip when reference_repos/sylber is absent.
- VG-HuBERT tests skip when models/vg-hubert_3/ checkpoints are absent.
- Sylber feature-parity test additionally requires the HuBERT hub checkpoint.
"""

import importlib.util
import os
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_THIS_DIR   = Path(__file__).resolve().parent
_REPO_ROOT  = _THIS_DIR.parent
_AUDIO_PATH = _REPO_ROOT / "test_samples" / "SP20_117.wav"
_VGHUBERT_DIR = _REPO_ROOT / "models" / "vg-hubert_3"
_REF_SYLBER   = (_REPO_ROOT.parent / "reference_repos" / "sylber").resolve()

_SYLLABLE_CKPT = _VGHUBERT_DIR / "vg-hubert-syllable.pth"
_WORD_CKPT     = _VGHUBERT_DIR / "vg-hubert-word.pth"

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

_skip_no_audio = pytest.mark.skipif(
    not _AUDIO_PATH.exists(),
    reason="test_samples/SP20_117.wav not found",
)
_skip_no_vghubert = pytest.mark.skipif(
    not _SYLLABLE_CKPT.exists() or not _WORD_CKPT.exists(),
    reason="VG-HuBERT checkpoints not found in models/vg-hubert_3/",
)
_skip_no_sylber_ref = pytest.mark.skipif(
    not _REF_SYLBER.is_dir(),
    reason="reference_repos/sylber not found",
)

# ---------------------------------------------------------------------------
# Audio fixture — shared by all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audio_16k():
    """Float32 mono audio array at 16 kHz from SP20_117.wav."""
    if not _AUDIO_PATH.exists():
        pytest.skip("test_samples/SP20_117.wav not found")
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(_AUDIO_PATH))
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        audio = wav[0].numpy()
    except Exception:
        import soundfile as sf
        audio, sr = sf.read(str(_AUDIO_PATH), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Inline reference: Sylber feature extraction
# Replicates sylber/model/sylber.py Sylber.__call__ without importing the
# broken top-level sylber package (which imports LightningModule at import time).
# ---------------------------------------------------------------------------

def _ref_sylber_extract(audio_np: np.ndarray) -> np.ndarray:
    """
    Reference Sylber feature extraction.

    Follows sylber.py exactly:
      config = HubertConfig.from_pretrained(..., num_hidden_layers=9)
      model  = HubertModel(config)
      model.load_state_dict(torch.load("sylber.ckpt"), strict=False)
      wav = (wav - wav.mean()) / wav.std()
      hidden = model(wav_tensor).last_hidden_state
    """
    from transformers import HubertModel, HubertConfig
    from huggingface_hub import hf_hub_download

    config = HubertConfig.from_pretrained(
        "facebook/hubert-base-ls960", num_hidden_layers=9
    )
    model = HubertModel(config)
    ckpt = hf_hub_download("cheoljun95/sylber", "sylber.ckpt")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()

    wav = audio_np.astype(np.float32)
    wav = (wav - wav.mean()) / wav.std()          # Reference: no +1e-8 epsilon
    with torch.no_grad():
        out = model(torch.from_numpy(wav).unsqueeze(0))
    return out.last_hidden_state.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Inline reference: CLS attention segmentation
# Verbatim from word-discovery/save_seg_feats.py (Peng & Harwath 2022)
# ---------------------------------------------------------------------------

def _ref_cls_attn_seg(cls_attn_np: np.ndarray, threshold: float, spf: float):
    """
    Reference CLS attention segmentation from word-discovery/save_seg_feats.py.

    Args:
        cls_attn_np: [n_heads, T] CLS row, audio frames only.
        threshold: per-head quantile threshold.
        spf: seconds per frame.

    Returns:
        (boundaries_in_sec, locations_sec)
          boundaries_in_sec: [[t_s_sec, t_e_sec], ...]  (t_e EXCLUSIVE)
          locations_sec: [peak_time_sec, ...]
    """
    cls_attn = torch.from_numpy(cls_attn_np.astype(np.float32))
    threshold_value = torch.quantile(cls_attn, threshold, dim=-1, keepdim=True)
    important_idx = torch.where(
        (cls_attn >= threshold_value).float().sum(0) > 0
    )[0].cpu().numpy()

    boundaries_all, boundaries_ex1 = [], []
    for _, g in groupby(enumerate(important_idx), lambda ix: ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s = seg[0]
        t_e = min(seg[-1] + 1, cls_attn.shape[-1])
        boundaries_all.append([t_s, t_e])
        if len(seg) > 1:
            boundaries_ex1.append([t_s, t_e])

    boundaries = boundaries_ex1 if len(boundaries_ex1) > 0 else boundaries_all

    cls_attn_sum = cls_attn.sum(0).cpu().numpy()
    boundaries_in_sec, locations_sec = [], []
    for t_s, t_e in boundaries:
        boundaries_in_sec.append([t_s * spf, t_e * spf])
        peak_local = int(np.argmax(cls_attn_sum[t_s:t_e]))
        locations_sec.append((t_s + peak_local) * spf)

    return boundaries_in_sec, locations_sec


# ---------------------------------------------------------------------------
# Inline reference: MinCut DP — verbatim from SyllableLM/extract_units.py
# (Baade et al., 2023). Inlined because the syllablelm package has a Python
# 3.11 dataclass incompatibility that prevents top-level import.
# ---------------------------------------------------------------------------

def _ref_dp_helper(x, threshold, s, min_hop):
    b, n, d = x.shape
    s = min(n, s)
    min_hop = min(s, min_hop)

    dists = x.new_full((b, s + 1, n + s), 16384)
    rolled = torch.stack(
        [torch.roll(x, shifts=-i, dims=-2) for i in range(s)]
    ).transpose(0, 1)
    rolled_prepend = x[:, :s].unsqueeze(2).repeat(1, 1, s - 1, 1)
    arranged = torch.cat([rolled_prepend, rolled], dim=2)

    len_indices = torch.arange(s, device=x.device) + 1
    dots   = arranged.pow(2).mean(dim=-1).cumsum(dim=-2)
    middle = (
        -1 / len_indices.view(1, -1, 1)
        * arranged.cumsum(dim=-3).pow(2).mean(dim=-1)
    )
    outs = dots + middle
    outs = torch.cat(
        [outs[:, i:i+1].roll(shifts=-(s-i-1), dims=2) for i in range(s)], dim=1
    )
    dists[:, 1:, s:] = outs[:, :, :-(s - 1)]
    dists += dists.new_full(dists.shape, 16384).tril(s - 2)
    dists = dists.clamp(max=16384)

    m = int(threshold * n)
    total_dists = x.new_full((b, n + 2), 16384)
    total_dists[:, 0] = 0
    back = x.new_zeros((b, n + 1, m + 1), dtype=torch.long)
    magic_mask = torch.tensor(
        [
            [(j + 1 - k if j + 1 >= k else n + 1) for j in range(n)]
            for k in range(min_hop, s + 1)
        ],
        device=x.device,
    ).unsqueeze(0).expand(b, s + 1 - min_hop, n)

    for j in range(1, m + 1):
        cur_min = torch.min(
            total_dists.unsqueeze(1)
            .expand(b, s + 1 - min_hop, n + 2)
            .gather(2, magic_mask)
            + dists[:, min_hop:, s : n + s],
            dim=1,
        )
        total_dists[:, 1:-1] = cur_min.values
        back[:, 1 : 1 + n, j] = cur_min.indices + min_hop

    return dists, back


def _ref_borders_helper(dists2d, back2d, n, s, num_units, delta, quantile):
    min_, max_ = num_units // 3, num_units
    best_m = min_

    while min_ <= max_:
        mid_ = (min_ + max_) // 2
        q, j = n, mid_
        costs = []
        while q > 0:
            step = int(back2d[q, j])
            costs.append(float(dists2d[step, q - 1 + s]) / step)
            q -= step
            j -= 1
        qcost = float(np.quantile(costs, quantile))
        if qcost > delta:
            min_ = mid_ + 1
            best_m = mid_
        else:
            max_ = mid_ - 1

    q, j = n, best_m
    borders = [q]
    while q > 0:
        q -= int(back2d[q, j])
        borders.append(q)
        j -= 1
    borders.reverse()
    return borders


def _ref_mincut(features_np, threshold, s, min_hop, delta, quantile):
    n = features_np.shape[0]
    s_eff = min(n, s)
    num_units = max(1, int(threshold * n))
    x = torch.from_numpy(features_np.astype(np.float32)).unsqueeze(0)
    dists_b, back_b = _ref_dp_helper(x, threshold=threshold, s=s_eff, min_hop=min_hop)
    return _ref_borders_helper(
        dists_b[0].cpu().numpy(),
        back_b[0].cpu().numpy(),
        n=n,
        s=s_eff,
        num_units=num_units,
        delta=delta,
        quantile=quantile,
    )


# ---------------------------------------------------------------------------
# Canonical MinCut hyperparameters (from findsylls/segmentation/mincut.py)
# ---------------------------------------------------------------------------

from findsylls.segmentation.mincut import MINCUT_THRESHOLD

_MC_S        = 35
_MC_MIN_HOP  = 3
_MC_DELTA    = 0.0033
_MC_QUANTILE = 0.75

SPF = 1.0 / 50.0   # 50 Hz frame rate


# ===========================================================================
# 1.  Sylber + Greedy Cosine
# ===========================================================================

class TestSylberGreedyCosineParity:
    """
    Parity for the Sylber + Greedy Cosine pipeline.

    Reference: Park et al. (ICLR 2025) — cheoljun95/sylber on HuggingFace Hub,
    get_segment() in reference_repos/sylber/sylber/utils/segment_utils.py.
    """

    # ------------------------------------------------------------------
    # Feature extraction parity
    # ------------------------------------------------------------------

    @_skip_no_audio
    def test_sylber_feature_parity_vs_reference(self, audio_16k):
        """
        SylberFeatureExtractor produces features within atol=1e-4 of the
        reference Sylber loading code on SP20_117.wav.

        Both paths:
          - HubertModel(HubertConfig.from_pretrained(..., num_hidden_layers=9))
          - load_state_dict(sylber.ckpt, strict=False)
          - normalize: (wav - mean) / std   [reference uses no +epsilon]
          - last_hidden_state
        The only difference is findsylls adds +1e-8 to std.  For real speech
        (std ≈ 0.01–0.2), this changes the input by < 1e-5 and propagates
        through the model to feature differences well below 1e-4.
        """
        pytest.importorskip("transformers")
        pytest.importorskip("huggingface_hub")

        from findsylls.features.sylber import SylberFeatureExtractor

        ref_features = _ref_sylber_extract(audio_16k)
        extractor = SylberFeatureExtractor()
        our_features = extractor.extract(audio_16k, sr=16000)

        assert ref_features.shape == our_features.shape, (
            f"Shape mismatch: ref={ref_features.shape} ours={our_features.shape}"
        )
        np.testing.assert_allclose(
            our_features, ref_features, atol=1e-4, rtol=0,
            err_msg=(
                f"Sylber feature parity failed on SP20_117.wav. "
                f"Max abs diff: {np.abs(our_features - ref_features).max():.2e}"
            ),
        )

    # ------------------------------------------------------------------
    # Algorithm parity on real features
    # ------------------------------------------------------------------

    @_skip_no_audio
    @_skip_no_sylber_ref
    def test_greedy_cosine_algorithm_parity_on_real_features(self, audio_16k):
        """
        greedy_cosine_segment() is bit-exact with reference get_segment() when
        given the same real features extracted from SP20_117.wav.

        Features are extracted once via findsylls SylberFeatureExtractor, then
        fed to both implementations.  This removes any feature-extraction
        variance and isolates the algorithm.
        """
        pytest.importorskip("transformers")
        pytest.importorskip("huggingface_hub")

        from findsylls.features.sylber import SylberFeatureExtractor
        from findsylls.segmentation.greedy_cosine import greedy_cosine_segment

        # Shared real features
        features = SylberFeatureExtractor().extract(audio_16k, sr=16000)

        # Reference algorithm
        seg_utils_path = _REF_SYLBER / "sylber" / "utils" / "segment_utils.py"
        spec = importlib.util.spec_from_file_location("_sylber_seg_utils", seg_utils_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_segment = mod.get_segment

        states = torch.from_numpy(features)
        ref_segs = np.array(get_segment(states, 2.6, 0.8))
        our_segs = greedy_cosine_segment(features, norm_threshold=2.6, merge_threshold=0.8)

        np.testing.assert_array_equal(
            ref_segs, our_segs,
            err_msg=(
                f"greedy_cosine_segment disagrees with reference get_segment on "
                f"real features from SP20_117.wav."
            ),
        )

    # ------------------------------------------------------------------
    # Full end-to-end parity
    # ------------------------------------------------------------------

    @_skip_no_audio
    @_skip_no_sylber_ref
    def test_sylber_segmenter_e2e_parity(self, audio_16k):
        """
        SylberSegmenter().segment() boundaries agree with the reference pipeline
        (reference Sylber model → reference get_segment) on SP20_117.wav.

        Boundary tolerance: 1 frame = 0.02 s (one-step rounding error at 50 Hz).
        """
        pytest.importorskip("transformers")
        pytest.importorskip("huggingface_hub")

        from findsylls.segmentation.presets import SylberSegmenter

        # Reference pipeline
        ref_features = _ref_sylber_extract(audio_16k)
        seg_utils_path = _REF_SYLBER / "sylber" / "utils" / "segment_utils.py"
        spec = importlib.util.spec_from_file_location("_sylber_seg_utils", seg_utils_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_segment = mod.get_segment

        states = torch.from_numpy(ref_features)
        ref_frame_segs = get_segment(states, 2.6, 0.8)
        ref_starts = [s * SPF for s, _ in ref_frame_segs]
        ref_ends   = [e * SPF for _, e in ref_frame_segs]

        # findsylls pipeline
        segmenter = SylberSegmenter()
        our_segs  = segmenter.segment(audio_16k, sr=16000)
        our_starts = [s for s, _, _ in our_segs]
        our_ends   = [e for _, _, e in our_segs]

        assert len(our_segs) == len(ref_frame_segs), (
            f"Segment count mismatch: findsylls={len(our_segs)} ref={len(ref_frame_segs)}"
        )
        for i, (rs, re, os, oe) in enumerate(zip(ref_starts, ref_ends, our_starts, our_ends)):
            assert abs(rs - os) <= SPF, (
                f"Segment {i} start mismatch: ref={rs:.4f} ours={os:.4f} (tol={SPF})"
            )
            assert abs(re - oe) <= SPF, (
                f"Segment {i} end mismatch: ref={re:.4f} ours={oe:.4f} (tol={SPF})"
            )


# ===========================================================================
# 2.  VG-HuBERT (word checkpoint, layer 9) + CLS attention
# ===========================================================================

class TestVGHubertCLSParity:
    """
    Parity for the VG-HuBERT + CLS attention pipeline.

    Reference: Peng & Harwath (Interspeech 2022) — word checkpoint, layer 9,
    cls_attn_seg_feats() from word-discovery/save_seg_feats.py (inlined above).
    """

    # ------------------------------------------------------------------
    # Algorithm parity on real attention
    # ------------------------------------------------------------------

    @_skip_no_audio
    @_skip_no_vghubert
    def test_cls_algorithm_parity_on_real_attention(self, audio_16k):
        """
        segment_by_cls_attention_raw_matrix() matches reference cls_attn_seg_feats()
        when given the same real CLS attention extracted from SP20_117.wav using
        the VG-HuBERT word checkpoint at layer 9.

        End-time convention: reference returns exclusive end (t_e * spf).
        findsylls returns inclusive end (times[last_frame]).
        Comparison converts findsylls end to exclusive by adding one SPF.
        """
        from findsylls.features.vg_hubert import VGHuBERTFeatureExtractor
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        extractor = VGHuBERTFeatureExtractor(mode="word")
        _, raw_attn = extractor.extract_with_attention(audio_16k, sr=16000, return_raw=True)

        # raw_attn: [n_heads, T+1, T+1]; CLS token is at position 0
        T = raw_attn.shape[1] - 1
        times = np.arange(T) * SPF
        cls_attn = raw_attn[:, 0, 1:]   # [n_heads, T]

        for quantile in (0.5, 0.9):
            ref_bounds, ref_peaks = _ref_cls_attn_seg(cls_attn, quantile, SPF)
            our_segs = segment_by_cls_attention_raw_matrix(raw_attn, times, quantile=quantile)

            assert len(our_segs) == len(ref_bounds), (
                f"Segment count mismatch at quantile={quantile}: "
                f"ref={len(ref_bounds)} ours={len(our_segs)}"
            )
            for i, ((ref_s, ref_e), (our_s, our_p, our_e)) in enumerate(
                zip(ref_bounds, our_segs)
            ):
                our_e_excl = our_e + SPF   # inclusive → exclusive
                assert abs(ref_s - our_s) < 1e-5, (
                    f"q={quantile} seg {i} start: ref={ref_s:.5f} ours={our_s:.5f}"
                )
                assert abs(ref_e - our_e_excl) < 1e-5, (
                    f"q={quantile} seg {i} end: ref={ref_e:.5f} ours_excl={our_e_excl:.5f}"
                )
            for i, (rp, (_, our_p, _)) in enumerate(zip(ref_peaks, our_segs)):
                assert abs(rp - our_p) < 1e-5, (
                    f"q={quantile} peak {i}: ref={rp:.5f} ours={our_p:.5f}"
                )

    # ------------------------------------------------------------------
    # Full end-to-end parity
    # ------------------------------------------------------------------

    @_skip_no_audio
    @_skip_no_vghubert
    def test_vghubert_cls_segmenter_default_is_word_mode(self):
        """
        VGHubertCLSSegmenter() defaults to mode='word' (word checkpoint, layer 9).
        """
        from findsylls.segmentation.presets import VGHubertCLSSegmenter
        seg = VGHubertCLSSegmenter()
        assert seg.mode == "word", (
            f"VGHubertCLSSegmenter default mode should be 'word', got '{seg.mode}'"
        )

    @_skip_no_audio
    @_skip_no_vghubert
    def test_vghubert_cls_preset_uses_word_mode(self):
        """
        The 'vg_hubert_cls' config-dict preset includes mode='word' in feature_kwargs.
        """
        from findsylls.presets import get_preset
        preset = get_preset("vg_hubert_cls")
        assert preset["feature_kwargs"].get("mode") == "word", (
            f"vg_hubert_cls preset must specify mode='word' in feature_kwargs, "
            f"got feature_kwargs={preset['feature_kwargs']}"
        )

    @_skip_no_audio
    @_skip_no_vghubert
    def test_vghubert_cls_segmenter_e2e_parity(self, audio_16k):
        """
        VGHubertCLSSegmenter().segment() boundaries agree with the reference
        cls_attn_seg_feats() on SP20_117.wav.

        The same raw attention extracted by findsylls is fed to both the
        findsylls segmenter and the reference algorithm, so this test
        isolates the algorithm from any feature-extraction variance.
        """
        from findsylls.features.vg_hubert import VGHuBERTFeatureExtractor
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        extractor = VGHuBERTFeatureExtractor(mode="word")
        _, raw_attn = extractor.extract_with_attention(audio_16k, sr=16000, return_raw=True)

        T = raw_attn.shape[1] - 1
        times = np.arange(T) * SPF
        cls_attn = raw_attn[:, 0, 1:]   # [n_heads, T]

        QUANTILE = 0.9   # default threshold used in word-discovery paper

        ref_bounds, _ = _ref_cls_attn_seg(cls_attn, QUANTILE, SPF)
        our_segs = segment_by_cls_attention_raw_matrix(raw_attn, times, quantile=QUANTILE)

        assert len(our_segs) == len(ref_bounds), (
            f"Segment count mismatch: findsylls={len(our_segs)} ref={len(ref_bounds)}"
        )
        for i, ((ref_s, ref_e), (our_s, our_p, our_e)) in enumerate(
            zip(ref_bounds, our_segs)
        ):
            our_e_excl = our_e + SPF
            assert abs(ref_s - our_s) < 1e-5, (
                f"Seg {i} start: ref={ref_s:.5f} ours={our_s:.5f}"
            )
            assert abs(ref_e - our_e_excl) < 1e-5, (
                f"Seg {i} end: ref_excl={ref_e:.5f} ours_excl={our_e_excl:.5f}"
            )


# ===========================================================================
# 3.  VG-HuBERT (syllable checkpoint, layer 8) + MinCut DP
# ===========================================================================

class TestVGHubertMinCutParity:
    """
    Parity for the VG-HuBERT + MinCut DP pipeline.

    Reference: Baade et al. SyllableLM (2023) — efficient_extraction_dp_helper +
    get_quantile_borders_helper inlined from SyllableLM/extract_units.py.

    Note: the default VGHubertMinCutSegmenter uses the SSM-based
    min_cut_optimized() path; this class tests the canonical DP path
    (use_reference=True) which has a known Python reference implementation.
    """

    # ------------------------------------------------------------------
    # DP algorithm parity on real features
    # ------------------------------------------------------------------

    @_skip_no_audio
    @_skip_no_vghubert
    def test_mincut_dp_tables_parity_on_real_features(self, audio_16k):
        """
        efficient_extraction_dp_helper() DP tables match the SyllableLM
        reference on real VG-HuBERT syllable-checkpoint features from SP20_117.wav.
        """
        from findsylls.features.vg_hubert import VGHuBERTFeatureExtractor
        from findsylls.segmentation.mincut import efficient_extraction_dp_helper

        features = VGHuBERTFeatureExtractor(mode="syllable").extract(audio_16k, sr=16000)
        n = features.shape[0]
        s_eff = min(n, _MC_S)

        # Reference DP tables
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
        ref_dists_b, ref_back_b = _ref_dp_helper(
            x, threshold=MINCUT_THRESHOLD, s=s_eff, min_hop=_MC_MIN_HOP
        )
        ref_dists = ref_dists_b[0].cpu().numpy()
        ref_back  = ref_back_b[0].cpu().numpy()

        # findsylls DP tables
        our_dists, our_back = efficient_extraction_dp_helper(
            features, threshold=MINCUT_THRESHOLD, s=_MC_S, min_hop=_MC_MIN_HOP
        )

        np.testing.assert_array_equal(ref_dists, our_dists, err_msg="dists table mismatch")
        np.testing.assert_array_equal(ref_back,  our_back,  err_msg="back table mismatch")

    @_skip_no_audio
    @_skip_no_vghubert
    def test_mincut_borders_parity_on_real_features(self, audio_16k):
        """
        get_quantile_borders_helper() returns identical borders to the SyllableLM
        reference on real VG-HuBERT syllable-checkpoint features from SP20_117.wav.
        """
        from findsylls.features.vg_hubert import VGHuBERTFeatureExtractor
        from findsylls.segmentation.mincut import (
            efficient_extraction_dp_helper,
            get_quantile_borders_helper,
        )

        features = VGHuBERTFeatureExtractor(mode="syllable").extract(audio_16k, sr=16000)

        ref_borders = _ref_mincut(
            features, MINCUT_THRESHOLD, _MC_S, _MC_MIN_HOP, _MC_DELTA, _MC_QUANTILE
        )

        n = features.shape[0]
        s_eff = min(n, _MC_S)
        num_units = max(1, int(MINCUT_THRESHOLD * n))
        our_dists, our_back = efficient_extraction_dp_helper(
            features, threshold=MINCUT_THRESHOLD, s=_MC_S, min_hop=_MC_MIN_HOP
        )
        our_borders = get_quantile_borders_helper(
            our_dists, our_back,
            n=n, s=s_eff, num_units=num_units,
            delta=_MC_DELTA, quantile=_MC_QUANTILE,
        )

        assert ref_borders == our_borders, (
            f"Border mismatch on SP20_117.wav:\n  ref={ref_borders}\n  ours={our_borders}"
        )

    # ------------------------------------------------------------------
    # Border validity on real features
    # ------------------------------------------------------------------

    @_skip_no_audio
    @_skip_no_vghubert
    def test_mincut_borders_valid_on_real_features(self, audio_16k):
        """
        MinCut borders on real features satisfy structural invariants:
        first=0, last=n, sorted, unique, all in [0, n].
        """
        from findsylls.features.vg_hubert import VGHuBERTFeatureExtractor
        from findsylls.segmentation.mincut import (
            efficient_extraction_dp_helper,
            get_quantile_borders_helper,
        )

        features = VGHuBERTFeatureExtractor(mode="syllable").extract(audio_16k, sr=16000)
        n = features.shape[0]
        s_eff = min(n, _MC_S)
        num_units = max(1, int(MINCUT_THRESHOLD * n))
        dists, back = efficient_extraction_dp_helper(
            features, threshold=MINCUT_THRESHOLD, s=_MC_S, min_hop=_MC_MIN_HOP
        )
        borders = get_quantile_borders_helper(
            dists, back, n=n, s=s_eff, num_units=num_units,
            delta=_MC_DELTA, quantile=_MC_QUANTILE,
        )

        assert borders[0] == 0,  "First border must be 0"
        assert borders[-1] == n, f"Last border must equal n={n}"
        assert borders == sorted(set(borders)), "Borders must be sorted and unique"
        assert all(0 <= b <= n for b in borders)

    # ------------------------------------------------------------------
    # Full end-to-end preset
    # ------------------------------------------------------------------

    @_skip_no_audio
    @_skip_no_vghubert
    def test_vghubert_mincut_segmenter_e2e(self, audio_16k):
        """
        VGHubertMinCutSegmenter().segment() returns valid (start, nucleus, end)
        tuples on SP20_117.wav: non-empty, ordered, within audio duration.
        """
        from findsylls.segmentation.presets import VGHubertMinCutSegmenter

        audio_duration = len(audio_16k) / 16000.0
        segmenter = VGHubertMinCutSegmenter(mode="syllable")
        segs = segmenter.segment(audio_16k, sr=16000)

        assert len(segs) > 0, "MinCut segmenter produced no segments"
        for i, (s, p, e) in enumerate(segs):
            assert s <= p <= e, f"Seg {i}: start={s} nucleus={p} end={e} not ordered"
            assert s >= 0.0, f"Seg {i}: start={s} < 0"
            assert e <= audio_duration + 0.05, f"Seg {i}: end={e} > audio duration"
        starts = [s for s, _, _ in segs]
        assert starts == sorted(starts), "Segment starts not non-decreasing"
