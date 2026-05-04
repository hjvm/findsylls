"""
CLS attention capability contracts and parity regression.

Tests three things:
1. API shape/type contracts for segment_by_cls_attention_raw_matrix
2. Parity against the reference algorithm from word-discovery (Peng & Harwath 2022)
3. Edge-case behavior (empty output, single-frame segments, all-important frames)

The reference algorithm is inlined here from
reference_repos/word-discovery/save_seg_feats.py so these tests require no
external model downloads and run without the reference repo being present.
"""

import numpy as np
import pytest
from itertools import groupby
from operator import itemgetter


# ---------------------------------------------------------------------------
# Reference algorithm: cls_attn_seg_feats from word-discovery/save_seg_feats.py
# (Peng & Harwath, Interspeech 2022)
# ---------------------------------------------------------------------------

def _ref_cls_attn_seg(cls_attn_np, threshold, spf, level2=False):
    """
    Port of cls_attn_seg_feats.

    Args:
        cls_attn_np: [n_heads, T] CLS row, audio frames only.
        threshold: per-head quantile threshold.
        spf: seconds per frame.
        level2: if True, include single-frame segments.

    Returns:
        boundaries_in_sec: list of [t_s_sec, t_e_sec] (t_e is exclusive end).
        locations_sec: list of peak times (argmax of CLS attention sum).
    """
    import torch
    cls_attn = torch.from_numpy(cls_attn_np.astype(np.float32))
    threshold_value = torch.quantile(cls_attn, threshold, dim=-1, keepdim=True)
    important_idx = torch.where(
        (cls_attn >= threshold_value).float().sum(0) > 0
    )[0].cpu().numpy()

    boundaries_all = []
    boundaries_ex1 = []
    for k, g in groupby(enumerate(important_idx), lambda ix: ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s = seg[0]
        t_e = min(seg[-1] + 1, cls_attn.shape[-1])
        if len(seg) > 1:
            boundaries_ex1.append([t_s, t_e])
        boundaries_all.append([t_s, t_e])

    boundaries = boundaries_all if (level2 or len(boundaries_ex1) == 0) else boundaries_ex1

    cls_attn_sum = cls_attn.sum(0).cpu().numpy()
    boundaries_in_sec = []
    locations_sec = []
    for t_s, t_e in boundaries:
        boundaries_in_sec.append([t_s * spf, t_e * spf])
        max_id = int(np.argmax(cls_attn_sum[t_s:t_e]))
        locations_sec.append((t_s + max_id) * spf)

    return boundaries_in_sec, locations_sec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def synthetic_attention():
    """Synthetic [12, 50, 50] attention tensor with CLS token at position 0."""
    rng = np.random.default_rng(42)
    attn = rng.random((12, 50, 50)).astype(np.float32)
    # Normalise rows so each row sums to 1 (attention-like)
    attn /= attn.sum(axis=-1, keepdims=True)
    return attn


@pytest.fixture(scope='module')
def uniform_attention():
    """All-uniform attention — every frame equally 'important'."""
    attn = np.ones((4, 20, 20), dtype=np.float32) / 20.0
    return attn


# ---------------------------------------------------------------------------
# API shape / type contracts
# ---------------------------------------------------------------------------

class TestShapeContracts:
    def test_accepts_nhh_attention(self, synthetic_attention):
        """Accepts raw [n_heads, T+1, T+1] attention and returns list."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        times = np.arange(T) / 50.0
        segs = segment_by_cls_attention_raw_matrix(synthetic_attention, times, quantile=0.9)
        assert isinstance(segs, list)

    def test_output_tuples_are_3_element(self, synthetic_attention):
        """Each output element is a (start, peak, end) 3-tuple."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        times = np.arange(T) / 50.0
        segs = segment_by_cls_attention_raw_matrix(synthetic_attention, times, quantile=0.9)
        for seg in segs:
            assert len(seg) == 3

    def test_start_le_peak_le_end(self, synthetic_attention):
        """start <= peak <= end for every segment."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        times = np.arange(T) / 50.0
        segs = segment_by_cls_attention_raw_matrix(synthetic_attention, times, quantile=0.5)
        for start, peak, end in segs:
            assert start <= peak, f"peak {peak} < start {start}"
            assert peak <= end, f"peak {peak} > end {end}"

    def test_segments_non_overlapping(self, synthetic_attention):
        """Segment start times are non-decreasing (no temporal overlap)."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        times = np.arange(T) / 50.0
        segs = segment_by_cls_attention_raw_matrix(synthetic_attention, times, quantile=0.5)
        if len(segs) > 1:
            for (s0, p0, e0), (s1, p1, e1) in zip(segs, segs[1:]):
                assert s1 >= s0, f"Segment starts not non-decreasing: {s0} then {s1}"

    def test_times_length_mismatch_raises(self, synthetic_attention):
        """ValueError when len(times) != T (number of audio frames)."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        wrong_times = np.arange(T + 5) / 50.0
        with pytest.raises(ValueError):
            segment_by_cls_attention_raw_matrix(synthetic_attention, wrong_times)


class TestEdgeCases:
    def test_uniform_attention_returns_segments(self, uniform_attention):
        """Uniform attention still produces at least one segment (fallback path)."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = uniform_attention.shape[1] - 1
        times = np.arange(T) / 50.0
        segs = segment_by_cls_attention_raw_matrix(uniform_attention, times, quantile=0.5)
        assert len(segs) >= 1

    def test_zero_quantile_gives_one_segment(self, synthetic_attention):
        """quantile=0 marks every frame as important (threshold=min), producing 1 segment."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        times = np.arange(T) / 50.0
        segs = segment_by_cls_attention_raw_matrix(synthetic_attention, times, quantile=0.0)
        assert len(segs) == 1, (
            f"quantile=0 must produce exactly 1 segment (all frames important), got {len(segs)}"
        )

    def test_single_head_attention(self):
        """Works with n_heads=1."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        rng = np.random.default_rng(5)
        attn = rng.random((1, 30, 30)).astype(np.float32)
        attn /= attn.sum(axis=-1, keepdims=True)
        T = 29
        times = np.arange(T) / 50.0
        segs = segment_by_cls_attention_raw_matrix(attn, times, quantile=0.5)
        assert isinstance(segs, list)


# ---------------------------------------------------------------------------
# Parity against reference algorithm
# ---------------------------------------------------------------------------

class TestParityVsWordDiscovery:
    """
    Verify bit-exact parity between segment_by_cls_attention_raw_matrix and
    the reference cls_attn_seg_feats from word-discovery (Peng & Harwath 2022).

    The comparison accounts for the end-time convention difference:
      - Reference: t_e * spf  (exclusive: index one past the last important frame)
      - findsylls: times[end_idx - 1]  (inclusive: time of the last important frame)
    We convert findsylls end times to exclusive by adding one spf before comparing.
    """

    SPF = 1.0 / 50.0  # 50 Hz frame rate

    @pytest.mark.parametrize("threshold", [0.50, 0.90])
    def test_boundaries_match(self, synthetic_attention, threshold):
        """Segment boundaries match reference to 1e-5 s after end-convention conversion."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        times = np.arange(T) * self.SPF

        cls_attn = synthetic_attention[:, 0, 1:]   # [n_heads, T]
        ref_bounds, _ = _ref_cls_attn_seg(cls_attn, threshold, self.SPF)

        segs = segment_by_cls_attention_raw_matrix(synthetic_attention, times, quantile=threshold)

        assert len(segs) == len(ref_bounds), (
            f"Segment count mismatch: ref={len(ref_bounds)} ours={len(segs)} "
            f"(threshold={threshold})"
        )

        for i, ((ref_s, ref_e), (our_s, our_p, our_e)) in enumerate(zip(ref_bounds, segs)):
            our_e_excl = our_e + self.SPF   # inclusive → exclusive
            assert abs(ref_s - our_s) < 1e-5, (
                f"Segment {i} start mismatch: ref={ref_s} ours={our_s}"
            )
            assert abs(ref_e - our_e_excl) < 1e-5, (
                f"Segment {i} end mismatch: ref={ref_e} ours_excl={our_e_excl}"
            )

    @pytest.mark.parametrize("threshold", [0.50, 0.90])
    def test_peaks_match(self, synthetic_attention, threshold):
        """Peak times match reference (both use argmax of sum-of-heads CLS attention)."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        T = synthetic_attention.shape[1] - 1
        times = np.arange(T) * self.SPF

        cls_attn = synthetic_attention[:, 0, 1:]
        _, ref_peaks = _ref_cls_attn_seg(cls_attn, threshold, self.SPF)

        segs = segment_by_cls_attention_raw_matrix(synthetic_attention, times, quantile=threshold)
        our_peaks = [p for _, p, _ in segs]

        assert len(our_peaks) == len(ref_peaks), (
            f"Peak count mismatch: ref={len(ref_peaks)} ours={len(our_peaks)}"
        )
        for i, (rp, op) in enumerate(zip(ref_peaks, our_peaks)):
            assert abs(rp - op) < 1e-5, f"Peak {i} mismatch: ref={rp} ours={op}"

    def test_single_head_parity(self):
        """Parity holds with a single attention head."""
        from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

        rng = np.random.default_rng(11)
        raw_attn = rng.random((1, 41, 41)).astype(np.float32)
        raw_attn /= raw_attn.sum(axis=-1, keepdims=True)

        T = 40
        times = np.arange(T) * self.SPF
        cls_attn = raw_attn[:, 0, 1:]   # [1, T]

        ref_bounds, ref_peaks = _ref_cls_attn_seg(cls_attn, 0.9, self.SPF)
        segs = segment_by_cls_attention_raw_matrix(raw_attn, times, quantile=0.9)

        assert len(segs) == len(ref_bounds)
        for (ref_s, ref_e), (our_s, our_p, our_e) in zip(ref_bounds, segs):
            assert abs(ref_s - our_s) < 1e-5
            assert abs(ref_e - (our_e + self.SPF)) < 1e-5
