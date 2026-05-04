"""
Replication parity test for neural segmentation methods.

For each method, applies BOTH the reference algorithm (ported directly from the
published repo) AND the findsylls implementation to the same extracted features.
Results must match exactly to claim scientific replication.

Methods tested:
  1. CLS Attention  — word-discovery (Peng & Harwath, Interspeech 2022)
  2. Greedy Cosine  — Sylber (Cho et al., ICLR 2025)
  3. MinCut DP      — SyllableLM (Baade et al., 2023)

Usage:
    python scripts/replication_test.py
"""

import sys
import os

# Ensure findsylls is importable from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import soundfile as sf
from itertools import groupby
from operator import itemgetter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TEST_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_samples')
REF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reference_repos')
SYLBER_REF = os.path.join(REF_DIR, 'sylber')

AUDIO_FILES = [
    os.path.join(TEST_SAMPLES_DIR, 'MMDB1_SI995.wav'),
    os.path.join(TEST_SAMPLES_DIR, 'SP20_117.wav'),
    os.path.join(TEST_SAMPLES_DIR, 'WKSP_M_0064_E1_0009.flac'),
]

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'


def load_audio(path):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


# ===========================================================================
# Reference algorithms (ported directly from published repos)
# ===========================================================================

def ref_cls_attn_seg(cls_attn_weights_np, threshold, spf, level2=False):
    """
    Port of cls_attn_seg_feats from word-discovery/save_seg_feats.py.

    Args:
        cls_attn_weights_np: np.ndarray [n_heads, T] — CLS row, audio frames only.
        threshold: quantile threshold (e.g. 0.90).
        spf: seconds per frame.
        level2: if True use all segments including single-frame ones.

    Returns:
        boundaries_in_sec: list of [t_s_sec, t_e_sec] (exclusive end).
        locations_sec: list of midpoint times in seconds.
    """
    import torch
    cls_attn_weights = torch.from_numpy(cls_attn_weights_np)
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True)
    important_idx = torch.where(
        (cls_attn_weights >= threshold_value).float().sum(0) > 0
    )[0].cpu().numpy()

    boundaries_all = []
    boundaries_ex1 = []
    for k, g in groupby(enumerate(important_idx), lambda ix: ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1] + 1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            boundaries_ex1.append([t_s, t_e])
        boundaries_all.append([t_s, t_e])

    if level2 or len(boundaries_ex1) == 0:
        boundaries = boundaries_all
    else:
        boundaries = boundaries_ex1

    cls_attn_sum = cls_attn_weights.sum(0).cpu().numpy()
    boundaries_in_sec = []
    locations_sec = []
    for t_s, t_e in boundaries:
        boundaries_in_sec.append([t_s * spf, t_e * spf])
        max_id = int(np.argmax(cls_attn_sum[t_s:t_e]))
        locations_sec.append((t_s + max_id) * spf)

    return boundaries_in_sec, locations_sec


def ref_get_segment(states_np, normthreshold, mergethreshold):
    """
    Load get_segment directly from sylber/utils/segment_utils.py via importlib.
    Avoids the broken top-level sylber package init (LightningModule import).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'segment_utils',
        os.path.join(SYLBER_REF, 'sylber', 'utils', 'segment_utils.py'),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import torch
    states = torch.from_numpy(states_np.astype(np.float32))
    return mod.get_segment(states, normthreshold, mergethreshold)


# ---------------------------------------------------------------------------
# Reference MinCut functions inlined from SyllableLM/extract_units.py.
# Cannot import extract_units directly (Python 3.11 dataclass incompatibility
# in the syllablelm package), so we inline the two pure-torch helper functions.
# ---------------------------------------------------------------------------

def _ref_efficient_extraction_dp_helper(x, threshold, s, min_hop):
    """Verbatim copy of efficient_extraction_dp_helper from extract_units.py."""
    import torch
    b, n, d = x.shape
    s = min(n, s)
    min_hop = min(s, min_hop)

    dists = x.new_full((b, s + 1, n + s), 16384)
    rolled = torch.stack([torch.roll(x, shifts=-i, dims=-2) for i in range(s)]).transpose(0, 1)
    rolled_prepend = x[:, :s].unsqueeze(2).repeat(1, 1, s - 1, 1)
    arranged = torch.cat([rolled_prepend, rolled], dim=2)

    len_indices = torch.arange(s, device=x.device) + 1
    dots = arranged.pow(2).mean(dim=-1).cumsum(dim=-2)
    middle = -1 / len_indices.view(1, -1, 1) * arranged.cumsum(dim=-3).pow(2).mean(dim=-1)
    outs = dots + middle
    outs = torch.cat([outs[:, i:i + 1].roll(shifts=-(s - i - 1), dims=2) for i in range(s)], dim=1)
    dists[:, 1:, s:] = outs[:, :, :-(s - 1)]
    dists += dists.new_full(dists.shape, 16384).tril(s - 2)
    dists = dists.clamp(max=16384)

    m = int(threshold * n)
    total_dists = x.new_full((b, n + 2), 16384)
    total_dists[:, 0] = 0
    back = x.new_zeros((b, n + 1, m + 1), dtype=torch.long)
    magic_mask = torch.tensor(
        [[(j + 1 - k if j + 1 >= k else n + 1) for j in range(n)] for k in range(min_hop, s + 1)],
        device=x.device,
    ).unsqueeze(0).expand(b, s + 1 - min_hop, n)

    for j in range(1, m + 1):
        cur_min = torch.min(
            total_dists.unsqueeze(1).expand(b, s + 1 - min_hop, n + 2).gather(2, magic_mask)
            + dists[:, min_hop:, s:n + s],
            dim=1,
        )
        total_dists[:, 1:-1] = cur_min.values
        back[:, 1:1 + n, j] = cur_min.indices + min_hop

    return dists, back  # still batched: [b, s+1, n+s], [b, n+1, m+1]


def _ref_get_quantile_borders_helper(dists2d, back2d, n, s, num_units, delta, quantile):
    """Verbatim copy of get_quantile_borders_helper from extract_units.py (unbatched)."""
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
        quantile_cost = float(np.quantile(costs, quantile))
        if quantile_cost > delta:
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


def ref_mincut_dp(features_np, threshold, s, min_hop, delta, quantile):
    """
    Run reference MinCut DP from inlined SyllableLM functions (extract_units.py).
    Returns list of frame boundary indices.
    """
    import torch
    n = features_np.shape[0]
    s_eff = min(n, s)
    num_units = max(1, int(threshold * n))

    x = torch.from_numpy(features_np.astype(np.float32)).unsqueeze(0)
    dists_b, back_b = _ref_efficient_extraction_dp_helper(x, threshold=threshold, s=s_eff, min_hop=min_hop)

    # Unbatch for the border search
    dists2d = dists_b[0].cpu().numpy()   # [s+1, n+s]
    back2d = back_b[0].cpu().numpy()     # [n+1, m+1]

    return _ref_get_quantile_borders_helper(dists2d, back2d, n=n, s=s_eff,
                                            num_units=num_units, delta=delta, quantile=quantile)


# ===========================================================================
# findsylls algorithms
# ===========================================================================

def findsylls_cls_attn_seg(cls_attn_np, threshold, spf, merge_valley_tol=0.0):
    """
    findsylls segment_by_cls_attention_raw_matrix called with pre-extracted CLS row.

    We wrap the CLS row into a fake full attention tensor so the function can
    extract it back via attention[:, 0, 1:] — this validates the extraction path.
    """
    from findsylls.segmentation.cls_attention import segment_by_cls_attention_raw_matrix

    n_heads, T = cls_attn_np.shape
    # Build fake [n_heads, T+1, T+1] with CLS row at row 0, cols 1..T
    fake_attn = np.zeros((n_heads, T + 1, T + 1), dtype=np.float32)
    fake_attn[:, 0, 1:] = cls_attn_np
    times = np.arange(T) * spf
    segs = segment_by_cls_attention_raw_matrix(
        fake_attn, times, quantile=threshold, merge_valley_tol=merge_valley_tol
    )
    return segs  # list of (start, peak, end) in seconds


def findsylls_greedy_cosine(features_np, normthreshold, mergethreshold):
    from findsylls.segmentation.greedy_cosine import greedy_cosine_segment
    return greedy_cosine_segment(features_np, norm_threshold=normthreshold,
                                 merge_threshold=mergethreshold)


def findsylls_mincut(features_np, threshold, s, min_hop, delta, quantile):
    from findsylls.segmentation.mincut import (
        efficient_extraction_dp_helper,
        get_quantile_borders_helper,
    )
    n = features_np.shape[0]
    s_eff = min(n, s)
    num_units = max(1, int(threshold * n))

    # findsylls efficient_extraction_dp_helper takes a 2D numpy array.
    dists2d, back2d = efficient_extraction_dp_helper(
        features_np, threshold=threshold, s=s, min_hop=min_hop
    )
    return get_quantile_borders_helper(dists2d, back2d, n=n, s=s_eff,
                                       num_units=num_units, delta=delta, quantile=quantile)


# ===========================================================================
# Comparison helpers
# ===========================================================================

def compare_boundaries(ref_bounds, our_bounds, tol=0.001, label=''):
    """Compare two lists of [start, end] boundary pairs (in seconds) within tol."""
    if len(ref_bounds) != len(our_bounds):
        print(f"  {FAIL} {label}: segment count mismatch ref={len(ref_bounds)} ours={len(our_bounds)}")
        print(f"    ref:  {ref_bounds[:5]}")
        print(f"    ours: {our_bounds[:5]}")
        return False
    max_err = 0.0
    for (rs, re), (os, oe) in zip(ref_bounds, our_bounds):
        max_err = max(max_err, abs(rs - os), abs(re - oe))
    ok = max_err <= tol
    status = PASS if ok else FAIL
    print(f"  {status} {label}: {len(ref_bounds)} segments, max_err={max_err:.6f}s")
    return ok


def compare_frame_segments(ref_segs, our_segs, label=''):
    """Compare two np.ndarray [N, 2] frame-index segment arrays."""
    if not np.array_equal(ref_segs, our_segs):
        print(f"  {FAIL} {label}: segment arrays differ")
        print(f"    ref shape {ref_segs.shape}, ours {our_segs.shape}")
        if ref_segs.shape == our_segs.shape:
            diff = np.abs(ref_segs.astype(int) - our_segs.astype(int))
            print(f"    max frame diff: {diff.max()}")
        return False
    print(f"  {PASS} {label}: {len(ref_segs)} segments, exact match")
    return True


def compare_border_lists(ref_borders, our_borders, label=''):
    """Compare two lists of frame boundary indices (MinCut output)."""
    ref_arr = np.array(ref_borders)
    our_arr = np.array(our_borders)
    if not np.array_equal(ref_arr, our_arr):
        print(f"  {FAIL} {label}: border arrays differ")
        print(f"    ref:  {ref_arr[:10]}")
        print(f"    ours: {our_arr[:10]}")
        return False
    print(f"  {PASS} {label}: {len(ref_arr)} borders, exact match")
    return True


# ===========================================================================
# Test runners
# ===========================================================================

def test_cls_attention():
    """
    CLS Attention parity test.

    Uses VGHuBERTFeatureExtractor to extract raw attention, then compares:
      - Reference: word-discovery cls_attn_seg_feats (threshold=0.90)
      - findsylls:  segment_by_cls_attention_raw_matrix (quantile=0.90)
    """
    print("\n=== CLS Attention Parity ===")
    from findsylls.features.vg_hubert import VGHuBERTFeatureExtractor

    extractor = VGHuBERTFeatureExtractor(mode='word', layer=9)
    all_pass = True

    for audio_path in AUDIO_FILES:
        name = os.path.basename(audio_path)
        print(f"\n  Audio: {name}")
        audio, sr = load_audio(audio_path)

        features, raw_attn = extractor.extract_with_attention(
            audio, sr, return_raw=True
        )
        print(f"    raw_attn shape: {raw_attn.shape}  features shape: {features.shape}")

        # Validate: raw_attn should be [n_heads, T+1, T+1]
        n_heads, seq_len, src_len = raw_attn.shape
        T = features.shape[0]
        if seq_len != T + 1:
            print(f"  {FAIL} Unexpected attention shape: seq_len={seq_len}, T={T}")
            all_pass = False
            continue

        cls_attn = raw_attn[:, 0, 1:]   # [n_heads, T]
        spf = 1.0 / extractor.frame_rate

        for threshold in [0.50, 0.90]:
            ref_bounds, ref_locs = ref_cls_attn_seg(cls_attn, threshold, spf)
            our_segs = findsylls_cls_attn_seg(cls_attn, threshold, spf, merge_valley_tol=0.0)

            our_bounds = [[s, e] for s, p, e in our_segs]
            # Reference end times are exclusive: t_e * spf.
            # findsylls uses inclusive last-frame time: times[end_idx - 1].
            # Convert our inclusive to exclusive for comparison: add one spf.
            our_bounds_excl = [[s, e + spf] for s, e in our_bounds]

            ok = compare_boundaries(
                ref_bounds, our_bounds_excl,
                tol=1e-5,
                label=f"threshold={threshold}"
            )
            if not ok:
                print(f"    ref peaks: {ref_locs[:5]}")
                print(f"    our peaks: {[p for s,p,e in our_segs[:5]]}")
            all_pass = all_pass and ok

    extractor.release()
    return all_pass


def test_greedy_cosine():
    """
    Greedy Cosine (Sylber) parity test.

    Uses SylberFeatureExtractor to extract features, then compares:
      - Reference: sylber.utils.segment_utils.get_segment
      - findsylls:  greedy_cosine_segment
    """
    print("\n=== Greedy Cosine (Sylber) Parity ===")
    from findsylls.features.sylber import SylberFeatureExtractor

    extractor = SylberFeatureExtractor()
    all_pass = True

    norm_threshold = 2.6
    merge_threshold = 0.8

    for audio_path in AUDIO_FILES:
        name = os.path.basename(audio_path)
        print(f"\n  Audio: {name}")
        audio, sr = load_audio(audio_path)

        features = extractor.extract(audio, sr)
        print(f"    features shape: {features.shape}")

        ref_segs = ref_get_segment(features, norm_threshold, merge_threshold)
        our_segs = findsylls_greedy_cosine(features, norm_threshold, merge_threshold)

        ok = compare_frame_segments(
            np.array(ref_segs), np.array(our_segs),
            label=f"norm={norm_threshold} merge={merge_threshold}"
        )
        all_pass = all_pass and ok

    extractor.release()
    return all_pass


def test_mincut():
    """
    MinCut DP parity test.

    Uses VGHuBERTFeatureExtractor to extract features, then compares:
      - Reference: SyllableLM efficient_extraction_dp_helper + get_quantile_borders_helper
      - findsylls:  same functions (ported)
    """
    print("\n=== MinCut DP Parity ===")
    from findsylls.features.vg_hubert import VGHuBERTFeatureExtractor

    extractor = VGHuBERTFeatureExtractor(mode='syllable', layer=8)
    all_pass = True

    from findsylls.segmentation.mincut import MINCUT_THRESHOLD
    threshold = MINCUT_THRESHOLD
    s = 35
    min_hop = 3
    delta = 0.0033
    quantile = 0.75

    for audio_path in AUDIO_FILES:
        name = os.path.basename(audio_path)
        print(f"\n  Audio: {name}")
        audio, sr = load_audio(audio_path)

        features = extractor.extract(audio, sr)
        print(f"    features shape: {features.shape}")

        try:
            ref_borders = ref_mincut_dp(features, threshold, s, min_hop, delta, quantile)
            our_borders = findsylls_mincut(features, threshold, s, min_hop, delta, quantile)
            ok = compare_border_lists(ref_borders, our_borders, label=f"delta={delta} q={quantile}")
        except Exception as exc:
            print(f"  {FAIL} MinCut error: {exc}")
            ok = False
        all_pass = all_pass and ok

    extractor.release()
    return all_pass


# ===========================================================================
# Main
# ===========================================================================

if __name__ == '__main__':
    results = {}

    try:
        results['cls_attention'] = test_cls_attention()
    except Exception as e:
        print(f"\nCLS Attention test CRASHED: {e}")
        import traceback; traceback.print_exc()
        results['cls_attention'] = False

    try:
        results['greedy_cosine'] = test_greedy_cosine()
    except Exception as e:
        print(f"\nGreedy Cosine test CRASHED: {e}")
        import traceback; traceback.print_exc()
        results['greedy_cosine'] = False

    try:
        results['mincut'] = test_mincut()
    except Exception as e:
        print(f"\nMinCut test CRASHED: {e}")
        import traceback; traceback.print_exc()
        results['mincut'] = False

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for method, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {method}")
        all_pass = all_pass and ok

    sys.exit(0 if all_pass else 1)
