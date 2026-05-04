"""
Greedy Cosine parity regression: findsylls vs Sylber reference.

Verifies that greedy_cosine_segment produces bit-for-bit identical output to
sylber/utils/segment_utils.py get_segment() from the reference repo at
reference_repos/sylber.

The reference is loaded via importlib to bypass the broken top-level sylber
package init (which imports LightningModule at import time and fails).
"""

import importlib.util
import os

import numpy as np
import pytest

REF_SYLBER = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'reference_repos', 'sylber')
)
_has_ref = os.path.isdir(REF_SYLBER)
requires_ref = pytest.mark.skipif(not _has_ref, reason="reference_repos/sylber not present")


def _load_ref_get_segment():
    path = os.path.join(REF_SYLBER, 'sylber', 'utils', 'segment_utils.py')
    spec = importlib.util.spec_from_file_location('_sylber_segment_utils', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_segment


@pytest.fixture(scope='module')
def ref_get_segment():
    if not _has_ref:
        pytest.skip("reference_repos/sylber not present")
    return _load_ref_get_segment()


@pytest.fixture(scope='module')
def features_120():
    rng = np.random.default_rng(42)
    return (rng.standard_normal((120, 768)) * 3.0).astype(np.float32)


@pytest.fixture(scope='module')
def features_20():
    rng = np.random.default_rng(7)
    return (rng.standard_normal((20, 768)) * 3.0).astype(np.float32)


@pytest.fixture(scope='module')
def features_sparse():
    """Features where most frames fall below norm threshold (mimics silence gaps)."""
    rng = np.random.default_rng(99)
    f = (rng.standard_normal((80, 768)) * 0.5).astype(np.float32)
    f[10:25] *= 8.0
    f[50:70] *= 8.0
    return f


@pytest.mark.parametrize("norm_threshold,merge_threshold", [
    (2.6, 0.8),
    (2.0, 0.7),
    (3.0, 0.9),
])
@requires_ref
def test_parity_canonical_params(ref_get_segment, features_120, norm_threshold, merge_threshold):
    """findsylls greedy_cosine_segment exactly matches Sylber get_segment."""
    import torch
    from findsylls.segmentation.greedy_cosine import greedy_cosine_segment

    states = torch.from_numpy(features_120)
    ref = ref_get_segment(states, norm_threshold, merge_threshold)
    ours = greedy_cosine_segment(features_120, norm_threshold=norm_threshold,
                                  merge_threshold=merge_threshold)
    np.testing.assert_array_equal(
        np.array(ref), np.array(ours),
        err_msg=f"Mismatch for norm={norm_threshold} merge={merge_threshold}",
    )


@requires_ref
def test_parity_short_sequence(ref_get_segment, features_20):
    """Parity holds on short (20-frame) sequences."""
    import torch
    from findsylls.segmentation.greedy_cosine import greedy_cosine_segment

    states = torch.from_numpy(features_20)
    ref = ref_get_segment(states, 2.6, 0.8)
    ours = greedy_cosine_segment(features_20, norm_threshold=2.6, merge_threshold=0.8)
    np.testing.assert_array_equal(np.array(ref), np.array(ours))


@requires_ref
def test_parity_sparse_features(ref_get_segment, features_sparse):
    """Parity holds when many frames fall below norm threshold (sparse activity)."""
    import torch
    from findsylls.segmentation.greedy_cosine import greedy_cosine_segment

    states = torch.from_numpy(features_sparse)
    ref = ref_get_segment(states, 2.6, 0.8)
    ours = greedy_cosine_segment(features_sparse, norm_threshold=2.6, merge_threshold=0.8)
    np.testing.assert_array_equal(np.array(ref), np.array(ours))


@requires_ref
def test_parity_all_silence(ref_get_segment):
    """Both return empty segments when all frames are below norm threshold.

    norm(0.05 * ones(768)) ≈ 1.39, which is below the default threshold of 2.6.
    """
    import torch
    from findsylls.segmentation.greedy_cosine import greedy_cosine_segment

    features = np.ones((30, 768), dtype=np.float32) * 0.05
    states = torch.from_numpy(features)
    ref = ref_get_segment(states, 2.6, 0.8)
    ours = greedy_cosine_segment(features, norm_threshold=2.6, merge_threshold=0.8)
    assert len(ref) == 0
    assert len(ours) == 0


@requires_ref
def test_parity_output_segments_are_valid(ref_get_segment, features_120):
    """Verified parity output satisfies segment validity (start < end, in bounds)."""
    import torch
    from findsylls.segmentation.greedy_cosine import greedy_cosine_segment

    states = torch.from_numpy(features_120)
    ref = ref_get_segment(states, 2.6, 0.8)
    ours = greedy_cosine_segment(features_120, norm_threshold=2.6, merge_threshold=0.8)
    np.testing.assert_array_equal(np.array(ref), np.array(ours))

    for start, end in ours:
        assert start < end, f"Invalid segment [{start}, {end}]"
        assert start >= 0
        assert end <= features_120.shape[0]
