"""Max pooling strategy."""

from typing import List, Tuple

import numpy as np

from .base import BasePooler


class MaxPooler(BasePooler):
    """Take the max activation per dimension in each syllable span."""

    def pool(
        self,
        features: np.ndarray,
        syllables: List[Tuple[float, float, float]],
        fps: float,
        **kwargs,
    ) -> np.ndarray:
        if len(syllables) == 0:
            return np.empty((0, features.shape[1]))

        out = []
        max_idx = features.shape[0] - 1
        for start, _peak, end in syllables:
            s = int(max(0, min(round(start * fps), max_idx)))
            e = int(max(0, min(round(end * fps), max_idx + 1)))
            if e <= s:
                out.append(features[s])
            else:
                out.append(features[s:e].max(axis=0))
        return np.asarray(out)
