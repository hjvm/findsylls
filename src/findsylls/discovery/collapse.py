"""Second-stage cluster collapsing for two-stage syllable discovery.

Fit K fine-grained clusters first, then collapse them into ``n_clusters`` coarser
classes by clustering the per-cluster centroids. The returned ``centroids`` and
``centroid_map`` are sufficient to assign new (held-out) embeddings at test time
via nearest-centroid lookup, which the agglomerative model itself cannot do
(scikit-learn ``AgglomerativeClustering`` has no out-of-sample prediction).

This two-stage pattern is used by Sylber (Cho et al., ICLR 2025) and VG-HuBERT
MinCut (Peng et al., Interspeech 2023).
"""

from __future__ import annotations

import numpy as np


def collapse_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    *,
    linkage: str = "ward",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse K fine-grained cluster labels into ``n_clusters`` coarse labels.

    Computes one centroid per fine-grained label, runs agglomerative clustering
    on those K centroids, and remaps every sample's fine label to its coarse id.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
        Per-sample embeddings.
    labels : np.ndarray, shape (N,)
        First-stage (fine-grained) cluster ids. Must be the contiguous 0-indexed
        integers ``0..K-1`` that ``DiscoveryPipeline.predict()`` produces.
    n_clusters : int
        Number of coarse output classes. Must be smaller than the number of
        unique fine labels.
    linkage : str, keyword-only
        Linkage criterion forwarded to ``sklearn.cluster.AgglomerativeClustering``
        (default: ``"ward"``).

    Returns
    -------
    new_labels : np.ndarray, shape (N,)
        Per-sample coarse assignments.
    centroids : np.ndarray, shape (n_clusters, D)
        Coarse cluster means; use for nearest-centroid test-time assignment.
    centroid_map : np.ndarray, shape (K,)
        Fine-to-coarse id mapping; index with a fine label to get its coarse
        label (``centroid_map[labels]`` reproduces ``new_labels``).

    Raises
    ------
    ValueError
        If ``n_clusters >= K`` (collapsing to the same count or more is
        meaningless), or if ``labels`` are not the contiguous integers ``0..K-1``.
    ImportError
        If scikit-learn is not installed.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError(
            "collapse_clusters requires scikit-learn. Install with: pip install scikit-learn"
        ) from exc

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    n_fine = len(unique_labels)
    if n_clusters >= n_fine:
        raise ValueError(
            f"n_clusters ({n_clusters}) must be smaller than the number of unique fine "
            f"labels ({n_fine}); collapsing to the same count or more is meaningless."
        )
    # The label-value -> coarse-id map (centroid_map) is indexed by label value,
    # so fine labels must be the contiguous 0..K-1 integers DiscoveryPipeline
    # produces. Fail loudly on gaps rather than IndexError / silently mis-map.
    if not np.array_equal(unique_labels, np.arange(n_fine)):
        raise ValueError(
            "collapse_clusters requires fine labels to be contiguous 0-indexed "
            f"integers (0..{n_fine - 1}); got {unique_labels.tolist()}."
        )

    # Stage-1 centroid per fine label (ordered by label value).
    fine_centroids = np.stack(
        [embeddings[labels == label].mean(axis=0) for label in unique_labels]
    )

    # Collapse the K centroids into n_clusters coarse classes.
    coarse_of_fine = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage
    ).fit_predict(fine_centroids)

    # Map fine label value -> coarse id. Labels are 0-indexed, so the unique
    # values index directly into a length-K array.
    centroid_map = np.empty(n_fine, dtype=np.int64)
    centroid_map[unique_labels] = coarse_of_fine

    new_labels = centroid_map[labels]

    # Recompute coarse centroids from the remapped membership.
    centroids = np.stack(
        [embeddings[new_labels == coarse].mean(axis=0) for coarse in range(n_clusters)]
    )

    return new_labels, centroids, centroid_map
