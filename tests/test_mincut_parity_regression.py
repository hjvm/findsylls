"""
MinCut DP parity regression: findsylls vs SyllableLM reference.

Verifies that efficient_extraction_dp_helper + get_quantile_borders_helper
produce bit-for-bit identical output to the reference implementation from
SyllableLM (Baade et al., 2023).

The SyllableLM reference functions are inlined here because the syllablelm
package has a Python 3.11 dataclass incompatibility that prevents it from
importing (mutable default in dataclass in syllablelm/data2vec/models/modalities/base.py).
The two inlined functions are pure torch/numpy with no dependency on the rest of
the syllablelm package.
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Reference functions: verbatim copy from SyllableLM/extract_units.py
# (https://github.com/AlanBaade/SyllableLM/blob/main/extract_units.py)
# ---------------------------------------------------------------------------

def _ref_efficient_extraction_dp_helper(x, threshold, s, min_hop):
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

    return dists, back


def _ref_get_quantile_borders_helper(dists2d, back2d, n, s, num_units, delta, quantile):
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


def _ref_mincut(features_np, threshold, s, min_hop, delta, quantile):
    """Run reference MinCut DP from inlined SyllableLM functions."""
    n = features_np.shape[0]
    s_eff = min(n, s)
    num_units = max(1, int(threshold * n))

    x = torch.from_numpy(features_np.astype(np.float32)).unsqueeze(0)
    dists_b, back_b = _ref_efficient_extraction_dp_helper(
        x, threshold=threshold, s=s_eff, min_hop=min_hop
    )
    dists2d = dists_b[0].cpu().numpy()
    back2d = back_b[0].cpu().numpy()
    return _ref_get_quantile_borders_helper(
        dists2d, back2d, n=n, s=s_eff, num_units=num_units, delta=delta, quantile=quantile
    )


def _findsylls_mincut(features_np, threshold, s, min_hop, delta, quantile):
    from findsylls.segmentation.mincut import (
        efficient_extraction_dp_helper,
        get_quantile_borders_helper,
    )
    n = features_np.shape[0]
    s_eff = min(n, s)
    num_units = max(1, int(threshold * n))
    dists2d, back2d = efficient_extraction_dp_helper(
        features_np, threshold=threshold, s=s, min_hop=min_hop
    )
    return get_quantile_borders_helper(
        dists2d, back2d, n=n, s=s_eff, num_units=num_units, delta=delta, quantile=quantile
    )


# ---------------------------------------------------------------------------
# Canonical parameter set from SyllableLM / findsylls presets
# ---------------------------------------------------------------------------
THRESHOLD = 1.0 / 0.10 / 50.0   # MINCUT_THRESHOLD from mincut.py
S = 35
MIN_HOP = 3
DELTA = 0.0033
QUANTILE = 0.75


@pytest.fixture(scope='module')
def features_100():
    rng = np.random.default_rng(42)
    return (rng.standard_normal((100, 768)) * 1.0).astype(np.float32)


@pytest.fixture(scope='module')
def features_50():
    rng = np.random.default_rng(13)
    return (rng.standard_normal((50, 768)) * 1.0).astype(np.float32)


@pytest.fixture(scope='module')
def features_200():
    rng = np.random.default_rng(77)
    return (rng.standard_normal((200, 768)) * 1.0).astype(np.float32)


def test_dp_table_parity_100frames(features_100):
    """DP tables from findsylls match SyllableLM reference for 100-frame input."""
    n = features_100.shape[0]
    s_eff = min(n, S)

    x = torch.from_numpy(features_100).unsqueeze(0)
    ref_dists_b, ref_back_b = _ref_efficient_extraction_dp_helper(
        x, threshold=THRESHOLD, s=s_eff, min_hop=MIN_HOP
    )
    ref_dists = ref_dists_b[0].cpu().numpy()
    ref_back = ref_back_b[0].cpu().numpy()

    from findsylls.segmentation.mincut import efficient_extraction_dp_helper
    our_dists, our_back = efficient_extraction_dp_helper(
        features_100, threshold=THRESHOLD, s=S, min_hop=MIN_HOP
    )

    np.testing.assert_array_equal(ref_dists, our_dists, err_msg="dists table mismatch")
    np.testing.assert_array_equal(ref_back, our_back, err_msg="back table mismatch")


def test_border_parity_100frames(features_100):
    """Border lists from findsylls match SyllableLM reference for 100-frame input."""
    ref_borders = _ref_mincut(features_100, THRESHOLD, S, MIN_HOP, DELTA, QUANTILE)
    our_borders = _findsylls_mincut(features_100, THRESHOLD, S, MIN_HOP, DELTA, QUANTILE)
    assert ref_borders == our_borders, (
        f"Border mismatch:\n  ref={ref_borders}\n  ours={our_borders}"
    )


def test_border_parity_50frames(features_50):
    """Border parity on shorter (50-frame) sequence."""
    ref_borders = _ref_mincut(features_50, THRESHOLD, S, MIN_HOP, DELTA, QUANTILE)
    our_borders = _findsylls_mincut(features_50, THRESHOLD, S, MIN_HOP, DELTA, QUANTILE)
    assert ref_borders == our_borders


def test_border_parity_200frames(features_200):
    """Border parity on longer (200-frame) sequence."""
    ref_borders = _ref_mincut(features_200, THRESHOLD, S, MIN_HOP, DELTA, QUANTILE)
    our_borders = _findsylls_mincut(features_200, THRESHOLD, S, MIN_HOP, DELTA, QUANTILE)
    assert ref_borders == our_borders


@pytest.mark.parametrize("delta,quantile", [
    (0.0033, 0.75),
    (0.0028, 0.75),
    (0.0019, 0.75),
])
def test_border_parity_varied_hyperparams(features_100, delta, quantile):
    """Parity holds across all three SyllableLM model preset hyperparameters."""
    ref_borders = _ref_mincut(features_100, THRESHOLD, S, MIN_HOP, delta, quantile)
    our_borders = _findsylls_mincut(features_100, THRESHOLD, S, MIN_HOP, delta, quantile)
    assert ref_borders == our_borders, (
        f"Mismatch for delta={delta} q={quantile}"
    )


def test_borders_are_valid(features_100):
    """Borders produced by findsylls satisfy ordering invariants."""
    from findsylls.segmentation.mincut import (
        efficient_extraction_dp_helper,
        get_quantile_borders_helper,
    )
    n = features_100.shape[0]
    s_eff = min(n, S)
    num_units = max(1, int(THRESHOLD * n))
    dists2d, back2d = efficient_extraction_dp_helper(
        features_100, threshold=THRESHOLD, s=S, min_hop=MIN_HOP
    )
    borders = get_quantile_borders_helper(
        dists2d, back2d, n=n, s=s_eff, num_units=num_units, delta=DELTA, quantile=QUANTILE
    )

    assert borders[0] == 0, "First border must be 0"
    assert borders[-1] == n, f"Last border must equal n={n}"
    assert borders == sorted(set(borders)), "Borders must be sorted and unique"
    assert all(b >= 0 and b <= n for b in borders), "All borders must be in [0, n]"
