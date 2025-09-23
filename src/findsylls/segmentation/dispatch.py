import numpy as np
from .peaks_and_valleys import segment_peaks_and_valleys

def segment_envelope(envelope: np.ndarray, times: np.ndarray, method: str = "peaks_and_valleys", **kwargs) -> list:
    if method is None:
        return segment_envelope(envelope=envelope, times=times, **kwargs)
    if method == "peaks_and_valleys":
        return segment_peaks_and_valleys(envelope=envelope, times=times, **kwargs)
    raise ValueError(f"Unsupported segmentation method: {method}")
