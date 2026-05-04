"""Onset-Nucleus-Coda pooling strategy."""

from typing import List, Tuple

import numpy as np

from .base import BasePooler


class ONCPooler(BasePooler):
    """Concatenate onset, peak, and coda frame vectors."""

    def pool(
        self,
        features: np.ndarray,
        syllables: List[Tuple[float, float, float]],
        fps: float,
        **kwargs,
    ) -> np.ndarray:
        if len(syllables) == 0:
            return np.empty((0, features.shape[1] * 3))

        out = []
        max_idx = features.shape[0] - 1
        frame_hop = 1.0 / fps

        for start, peak, end in syllables:
            t_on = start + 0.3 * max(0.0, peak - start)
            t_pk = peak
            t_cd = peak + 0.7 * max(0.0, end - peak)

            i_on = int(np.clip(round(t_on / frame_hop), 0, max_idx))
            i_pk = int(np.clip(round(t_pk / frame_hop), 0, max_idx))
            i_cd = int(np.clip(round(t_cd / frame_hop), 0, max_idx))

            out.append(np.concatenate([features[i_on], features[i_pk], features[i_cd]]))

        return np.asarray(out)
