"""Speech-rhythm primitives (Zhang & Glass, ICASSP 2009).

These operate on peak trains, not audio -- they belong to the segmentation
layer. ``fit_rhythm_sinusoid`` places a sinusoid's crests on detected peaks
(Eq. 4); ``rhythm_crests`` returns the predicted nucleus times (the crests).
A rhythm-guided segmenter uses them to license sensitive peak detection where a
nucleus is predicted.

    Zhang, Y. & Glass, J. R. (2009). Speech rhythm guided syllable nuclei
    detection. ICASSP 2009. https://doi.org/10.1109/ICASSP.2009.4960454
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize


def fit_rhythm_sinusoid(
    peak_times,
    seed_period: float = 0.20,
    period_bounds: Tuple[float, float] = (0.06, 0.6),
) -> Tuple[float, float]:
    """Fit ``sin(k1*x + 2*pi*k2)`` to a peak train by the paper's LMS objective.

    Eq. (4): ``{k1,k2} = argmin mean_i (1 - sin(k1*p_i + 2*pi*k2))**2`` -- place
    crests (sin=1) on the peaks. The objective is multimodal (2 peaks are
    consistent with a period and all its harmonics), so this does a **seeded
    local** least-squares from the paper's 200 ms default rather than a global
    search -- which is what makes a 2-peak fit converge instead of railing.

    Args:
        peak_times: peak locations in seconds (>= 2).
        seed_period: starting period in seconds (Zhang's 200 ms default; also
            the warm-start when re-fitting).
        period_bounds: clamp the fitted period to this range in seconds.

    Returns:
        (k1, k2): angular frequency (rad/s) and phase offset in [0, 1).
    """
    p = np.asarray(sorted(peak_times), dtype=float)
    if p.size < 2:
        raise ValueError("Need at least two peaks to estimate rhythm.")

    def obj(theta):
        k1, k2 = theta
        return float(np.mean((1.0 - np.sin(k1 * p + 2 * np.pi * k2)) ** 2))

    k1_0 = 2 * np.pi / seed_period
    # seed the phase by a cheap 1-D scan at the seed frequency
    k2_grid = np.linspace(0.0, 1.0, 64, endpoint=False)
    k2_0 = float(k2_grid[int(np.argmin([obj((k1_0, k2)) for k2 in k2_grid]))])

    res = minimize(obj, x0=[k1_0, k2_0], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-7, maxiter=300))
    k1, k2 = res.x
    period = float(np.clip(2 * np.pi / abs(k1), period_bounds[0], period_bounds[1]))
    return 2 * np.pi / period, float(k2 % 1.0)


def rhythm_crests(k1: float, k2: float, t_start: float, t_end: float) -> np.ndarray:
    """Predicted nucleus times: where ``sin(k1*x + 2*pi*k2) == 1`` in the range.

    Crests satisfy ``k1*x + 2*pi*k2 = pi/2 + 2*pi*n``, spaced by one period.
    """
    if k1 <= 0:
        return np.array([])
    period = 2 * np.pi / k1
    # first integer n whose crest is >= t_start
    n0 = int(np.ceil((k1 * t_start + 2 * np.pi * k2 - np.pi / 2) / (2 * np.pi)))
    crests = []
    n = n0
    while True:
        x = (np.pi / 2 - 2 * np.pi * k2 + 2 * np.pi * n) / k1
        if x > t_end:
            break
        if x >= t_start:
            crests.append(x)
        n += 1
        if len(crests) > int((t_end - t_start) / period) + 5:  # safety
            break
    return np.asarray(crests, dtype=float)


__all__ = ["fit_rhythm_sinusoid", "rhythm_crests"]
