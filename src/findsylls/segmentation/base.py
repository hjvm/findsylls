"""
Base classes for syllable segmentation methods.

Two concrete segmenter families:

1. EnvelopeBasedSegmenter — classical signal processing
   Accepts either raw audio (computes its own envelope) or a pre-computed
   (envelope, times) pair for the functional / backward-compat path.

2. End2EndSegmenter — neural end-to-end methods
   Accepts raw audio only; runs a learned representation internally.

Both families share the SAD-chunking logic and add_utterance_boundaries flag
provided by BaseSegmenter.  For envelope-based segmenters (PeakdetectSegmenter),
the flag passes add_boundary_valleys to segment_peakdetect so the in-algorithm
valley detection covers the full speech region.  For neural end-to-end segmenters
the flag is stored but is a documented no-op — those algorithms already produce
contiguous segmentation of the full chunk.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from ..vad import resolve_sad

if TYPE_CHECKING:
    from ..vad.base import BaseSAD


def extract_frame_features(
    feature_extractor: Any,
    audio: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Extract frame-level features from a segmentation feature extractor.

    Dispatch order:
    1. extractor.extract(audio, sr)
    2. extractor(audio, sr)
    """
    extract_fn = getattr(feature_extractor, "extract", None)
    if callable(extract_fn):
        return extract_fn(audio, sr)
    if callable(feature_extractor):
        return feature_extractor(audio, sr)
    raise TypeError(
        "feature_extractor must provide an extract(audio, sr) method "
        "or be callable as (audio, sr)."
    )


class BaseSegmenter(ABC):
    """
    Abstract base for all segmenters.

    Subclasses implement _segment(audio, sr).  The public segment(audio, sr)
    adds optional SAD chunking on top, running _segment per speech region and
    reassembling with global timestamps.

    Args:
        sample_rate: Target sample rate.
        sad: Optional SAD backend (BaseSAD).  When provided, segment() runs the
             core algorithm only on detected speech regions and reassembles with
             global timestamps.
        add_utterance_boundaries: Insert boundary markers at the onset and offset
             of each speech region so the algorithm can produce segments that cover
             the full region (default: True).  For envelope-based segmenters this
             triggers in-algorithm valley insertion.  For neural segmenters it is
             stored but has no effect (those algorithms already cover the full chunk).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        self.sample_rate = sample_rate
        self.sad = resolve_sad(sad)
        self.add_utterance_boundaries = add_utterance_boundaries

    @abstractmethod
    def _segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        """Core segmentation logic on a single audio chunk. No SAD."""
        pass

    def segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        """
        Segment audio into syllables.

        When sad is set, runs _segment per speech region and merges with global
        timestamps.
        """
        if self.sad is not None:
            regions = self.sad.get_speech_regions(audio, sr)
        else:
            regions = [(0.0, len(audio) / sr)]

        out: List[Tuple[float, float, float]] = []
        for start_s, end_s in regions:
            chunk = audio[int(start_s * sr): int(end_s * sr)]
            if len(chunk) == 0:
                continue
            segs = self._segment(chunk, sr)
            out.extend((s + start_s, p + start_s, e + start_s) for s, p, e in segs)
        return out

    def _validate_output(self, segments: List[Tuple[float, float, float]]) -> None:
        for i, (start, nucleus, end) in enumerate(segments):
            assert start <= nucleus <= end, (
                f"Invalid segment {i}: start={start}, nucleus={nucleus}, end={end}"
            )
            assert start >= 0, f"Negative start time in segment {i}: {start}"

    def cite(self) -> None:
        """Print the citation for this segmenter's source paper."""
        ref = getattr(self.__class__, "REFERENCE", None)
        if ref:
            print(ref)
        else:
            print(f"{self.__class__.__name__} has no associated paper reference.")

    def __call__(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        return self.segment(audio, sr)


class EnvelopeBasedSegmenter(BaseSegmenter):
    """
    Base for envelope-based segmenters.

    Adds a secondary entry point: segment(envelope=..., times=...) which bypasses
    SAD chunking and calls _segment_from_envelope directly.

    Subclasses must implement both _segment(audio, sr) and
    _segment_from_envelope(envelope, times).
    """

    @abstractmethod
    def _segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        pass

    @abstractmethod
    def _segment_from_envelope(
        self, envelope: np.ndarray, times: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        pass

    def segment(  # type: ignore[override]
        self,
        audio: Optional[np.ndarray] = None,
        sr: Optional[int] = None,
        envelope: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        **kwargs,
    ) -> List[Tuple[float, float, float]]:
        """
        Two entry points:

        1. segment(audio, sr) — goes through SAD chunking.
        2. segment(envelope=env, times=t) — bypasses SAD; calls _segment_from_envelope
           directly on the pre-computed envelope.
        """
        if envelope is not None:
            if times is None:
                raise ValueError("Must provide times when using pre-computed envelope")
            return self._segment_from_envelope(np.asarray(envelope), np.asarray(times))

        if audio is None or sr is None:
            raise ValueError("Must provide either (audio, sr) or (envelope, times)")
        return super().segment(audio, sr)


class End2EndSegmenter(BaseSegmenter):
    """
    Base for neural end-to-end segmenters.

    Args:
        sample_rate: Target sample rate.
        device: Torch device string (default: 'cpu').
        cache: Whether to cache the loaded model (default: True).
        sad: Optional SAD backend. Use sad='energy' or sad='silero' to restrict
             segmentation to detected speech regions.
        add_utterance_boundaries: Stored for API consistency; no-op for neural
             segmenters because the algorithms already cover the full chunk
             (default: True).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = 'cpu',
        cache: bool = True,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, sad=sad,
                         add_utterance_boundaries=add_utterance_boundaries)
        self.device = device
        self.cache = cache
        self._model = None

    @abstractmethod
    def _segment(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
        pass
