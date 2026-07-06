"""
Preset segmenter configurations from published papers and reference baselines.

These classes provide pre-configured segmenters that replicate the exact
configurations reported in the original papers. Each class IS-A algorithm class
configured with paper-canonical hyperparameters. SAD parameters from the parent
classes are available.

For flexibility and custom configurations, use the generic wrappers:
- MinCutSegmenter(feature_extractor, **params)
- GreedyCosineSegmenter(feature_extractor, **params)
- PeakdetectSegmenter(envelope_computer, **params)

Available presets:
- SBSPeakdetectSegmenter: SBS envelope + peakdetect (dissertation baseline)
- ThetaOscillatorSegmenter: Theta oscillator + peakdetect (Räsänen et al. 2018)
- SylberSegmenter: Sylber with greedy cosine (Cho et al. 2025)
- VGHubertMinCutSegmenter: VG-HuBERT with SSM + MinCut (Peng et al. 2023)
- VGHubertCLSSegmenter: VG-HuBERT with CLS attention (Peng et al. 2022)
"""

from typing import List, Optional, Tuple, TYPE_CHECKING

from .base import BaseSegmenter
from .cls_attention import CLSAttentionSegmenter
from .convexhull import ConvexHullSegmenter
from .gated import RegionGatedSegmenter
from .greedy_cosine import GreedyCosineSegmenter
from .mincut import MinCutSegmenter
from .peakdetect_segmenter import PeakdetectSegmenter
from ..features import SylberFeatureExtractor, VGHuBERTFeatureExtractor

if TYPE_CHECKING:
    from ..vad.base import BaseSAD


class SBSPeakdetectSegmenter(PeakdetectSegmenter):
    """
    Spectral Band Subtraction envelope + peak detection (findsylls baseline).

    A baseline configuration combining the SBS envelope (Liberman 2020) with
    Billauer peak detection (Billauer), as benchmarked in the findsylls
    toolkit paper. SBS is used here as a baseline; it is not a novel method.
    - Envelope: SBS (low-frequency minus high-frequency spectral energy, pivot at 3000 Hz,
                Hamming-smoothed at 70 ms / 7 samples at 100 Hz frame rate)
    - Segmentation: Billauer valley-picking with max syllable duration cap (400 ms)
                    and shallow-valley filter (merge valleys shallower than 40% of local max)

    Reference:
        SBS envelope: Liberman, M. (2020). "Syllables." Language Log.
        https://languagelog.ldc.upenn.edu/nll/?p=46144
        Peak detection: Billauer, E. peakdet: Peak detection using MATLAB.
        http://billauer.co.il/peakdet.html

    Args:
        pivot_freq: Frequency (Hz) dividing low- from high-energy bands (default: 3000)
        smoothing_window_samples: Hamming window length for envelope smoothing in frames
                                  (default: 7 = 70 ms at 100 Hz frame rate)
        delta: Billauer valley depth threshold (default: 0.01)
        max_syllable_dur: Maximum allowed syllable duration in seconds; syllables
                          whose valley-to-valley span exceeds this are dropped
                          (default: 0.4). Pass None to disable the cap.
        amplitude_ratio_tol: Shallow-valley merge threshold as fraction of local max
                             (default: 0.4 — merge valleys shallower than 40% of local peak).
                             Pass None to disable shallow-valley merging.
        min_syllable_dur: Minimum valley-to-valley span to emit a syllable, in seconds
                          (default: 0.05).
        merge_valley_tol: Time tolerance for merging nearby valleys, in seconds
                          (default: 0.05).
        min_amplitude_threshold: Minimum peak amplitude as a fraction of the max
                                 envelope amplitude (default: 0.0 = off). Suppresses
                                 low-amplitude spurious peaks such as sonorant-onset
                                 bumps (e.g. the [l] in "clever"); ~0.08–0.1 removes
                                 those without affecting real syllable nuclei.
        lookahead: Billauer look-ahead in samples (default: None = auto from
                   min_syllable_dur).
        sample_rate: Target sample rate (default: 16000)
        sad: Optional SAD backend for speech-region chunking (default: None)
        add_utterance_boundaries: Insert boundary valleys at region onset/offset so the
                                  algorithm can produce segments covering the full speech
                                  region (default: True)

    Example:
        >>> segmenter = SBSPeakdetectSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> segmenter.cite()
    """

    REFERENCE = (
        "findsylls SBS baseline. "
        'SBS envelope: Liberman, M. (2020, Feb 24). "Syllables." Language Log. '
        "https://languagelog.ldc.upenn.edu/nll/?p=46144 "
        "(sum of spectral energy below 3 kHz minus sum above 3 kHz, smoothed, "
        "peak-picked). "
        "Peak detection: Billauer, E. peakdet: Peak detection using MATLAB. "
        "http://billauer.co.il/peakdet.html. "
        "Configuration: SBS envelope (pivot_freq=3000 Hz, smoothing_window=70 ms at "
        "100 Hz) + Billauer peak detection (delta=0.01, max_syllable_dur=0.4 s, "
        "amplitude_ratio_tol=0.4)."
    )

    def __init__(
        self,
        pivot_freq: int = 3000,
        smoothing_window_samples: int = 7,
        delta: float = 0.01,
        max_syllable_dur: Optional[float] = 0.4,
        amplitude_ratio_tol: Optional[float] = 0.4,
        min_syllable_dur: float = 0.05,
        merge_valley_tol: float = 0.05,
        min_amplitude_threshold: float = 0.0,
        lookahead: Optional[int] = None,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        from ..envelope.sbs import SBSEnvelope
        super().__init__(
            envelope_computer=SBSEnvelope(
                pivot_freq=pivot_freq,
                smoothing_window_samples=smoothing_window_samples,
            ),
            delta=delta,
            lookahead=lookahead,
            min_syllable_dur=min_syllable_dur,
            max_syllable_dur=max_syllable_dur,
            merge_valley_tol=merge_valley_tol,
            amplitude_ratio_tol=amplitude_ratio_tol,
            min_amplitude_threshold=min_amplitude_threshold,
            sample_rate=sample_rate,
            sad=sad,
            add_utterance_boundaries=add_utterance_boundaries,
        )
        self.pivot_freq = pivot_freq
        self.smoothing_window_samples = smoothing_window_samples


class ThetaOscillatorSegmenter(PeakdetectSegmenter):
    """
    Theta oscillator syllable segmentation (Räsänen, Doyle & Frank 2018).

    Replicates the paper's canonical configuration from thetaOscillator.m:
    - Envelope: gammatone filterbank (20 bands, 50–7500 Hz, ERB-spaced, 1 kHz)
                → Hilbert envelope → harmonic damped oscillator bank → sonority
    - Parameters: f=5 Hz, Q=0.5, N=8 (top-8 bands)
    - Segmentation: Billauer valley-picking with depth threshold delta=0.025,
                    lookahead=1 (sample-by-sample, matching MATLAB peakdet.m)

    Known divergence from the original MATLAB implementation:
        The MATLAB code uses gammatone_c (a custom C gammatone filterbank) while
        this implementation uses the detly/gammatone ERB filterbank. The oscillator
        logic, delay table, and valley-picking algorithm are algebraically identical.
        Quantitative output parity against the Python port reference is confirmed at
        floating-point epsilon; parity against MATLAB is limited by the filterbank.

    Reference:
        Räsänen, O., Doyle, G., & Frank, M. C. (2018). "Pre-linguistic segmentation
        of speech into syllable-like units." Cognition, 171, 130–150.
        MATLAB code: github.com/orasanen/thetaOscillator

    Args:
        f: Oscillator center frequency in Hz (default: 5, paper default)
        Q: Q-factor / damping ratio (default: 0.5, paper default)
        N: Number of most-energetic bands to sum for sonority (default: 8, paper default)
        delta: Valley depth threshold for boundary detection (default: 0.025, paper default)
        sample_rate: Target sample rate (default: 16000)
        sad: Optional SAD backend for speech-region chunking (default: None)
        add_utterance_boundaries: Insert boundary valleys at region onset/offset so the
                                  algorithm can produce segments covering the full speech
                                  region (default: True)

    Example:
        >>> segmenter = ThetaOscillatorSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> segmenter.cite()
    """

    REFERENCE = (
        "Räsänen, O., Doyle, G., & Frank, M. C. (2018). "
        '"Pre-linguistic segmentation of speech into syllable-like units." '
        "Cognition, 171, 130–150. "
        "https://doi.org/10.1016/j.cognition.2017.11.003\n"
        "MATLAB implementation: https://github.com/orasanen/thetaOscillator"
    )

    def __init__(
        self,
        f: float = 5,
        Q: float = 0.5,
        N: int = 8,
        delta: float = 0.025,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        from ..envelope.theta import ThetaEnvelope
        super().__init__(
            envelope_computer=ThetaEnvelope(f=f, Q=Q, N=N),
            delta=delta,
            lookahead=1,
            sample_rate=sample_rate,
            sad=sad,
            add_utterance_boundaries=add_utterance_boundaries,
        )
        self.f = f
        self.Q = Q
        self.N = N


class SylberSegmenter(GreedyCosineSegmenter):
    """
    Sylber syllable segmentation (Cho et al. 2025).

    Replicates the paper's default configuration:
    - Feature extractor: Sylber's fine-tuned HuBERT (layer 9, 768-dim)
    - Algorithm: Greedy cosine similarity with boundary refinement
    - Hyperparameters: norm_threshold=2.6, merge_threshold=0.8

    Reference:
        Cho, C. J., Lee, N., Gupta, A., Agarwal, D., Chen, E., Black, A. W., &
        Anumanchipalli, G. K. (2025). "Sylber: Syllabic Embedding Representation
        of Speech from Raw Audio." ICLR 2025. arXiv:2410.07168.

    Args:
        norm_threshold: Energy threshold for silence detection (default: 2.6)
        merge_threshold: Cosine similarity threshold for merging (default: 0.8)
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)
        sad: Optional SAD backend for speech-region chunking (default: None)
        add_utterance_boundaries: No-op for this segmenter — the greedy cosine
                                  algorithm already covers the full chunk (default: True)

    Example:
        >>> segmenter = SylberSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> segmenter.cite()
    """

    REFERENCE = (
        "Cho, C. J., Lee, N., Gupta, A., Agarwal, D., Chen, E., Black, A. W., & "
        "Anumanchipalli, G. K. (2025). "
        '"Sylber: Syllabic Embedding Representation of Speech from Raw Audio." '
        "ICLR 2025. https://arxiv.org/abs/2410.07168"
    )

    def __init__(
        self,
        norm_threshold: float = 2.6,
        merge_threshold: float = 0.8,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(
            feature_extractor=SylberFeatureExtractor(device=device),
            norm_threshold=norm_threshold,
            merge_threshold=merge_threshold,
            sample_rate=sample_rate,
            sad=sad,
            add_utterance_boundaries=add_utterance_boundaries,
        )
        self.device = device


class VGHubertMinCutSegmenter(MinCutSegmenter):
    """
    VG-HuBERT with MinCut segmentation (Peng et al. 2023).

    Uses VG-HuBERT features with graph-based MinCut segmentation:
    - Feature extractor: VG-HuBERT (layer 8 for syllables, 9 for words)
    - Algorithm: SSM-based MinCut with K = ceil(duration / sec_per_syllable)
    - Post-processing: greedy cosine merge (minCutMerge-0.3, Peng et al. 2023)
    - Hyperparameters: sec_per_syllable=0.20, merge_threshold=0.3

    Note: Uses use_reference=False (SSM path) to match the original paper, which
    predates SyllableLM's DP algorithm. The DP's delta/quantile parameters are
    calibrated for Data2Vec2 features and over-segment VGHuBERT features.

    Algorithm note: defaults to use_optimized=False (the reference Cython min_cut
    algorithm) for exact segment-count parity with the paper. Set use_optimized=True
    for ~50x speedup with near-identical (but not bit-exact) boundaries.

    Reference:
        Peng, P., et al. (2023). "Syllable Discovery and Cross-Lingual Generalization
        in a Visually Grounded, Self-Supervised Speech Model." Interspeech 2023.

    Args:
        layer: VG-HuBERT layer (default: None = auto-select from mode)
        mode: Granularity - 'syllable' or 'word' (default: 'syllable')
        sec_per_syllable: Target syllable duration (default: 0.20)
        merge_threshold: Cosine similarity threshold for post-merge (default: 0.3).
                         Set to None to disable merging.
        use_optimized: Use optimized MinCut (default: False for reference parity;
                       set True for ~50x speedup with near-identical results).
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)
        sad: Optional SAD backend for speech-region chunking (default: None)
        add_utterance_boundaries: No-op for this segmenter — MinCut forces K
                                  contiguous segments covering the full chunk (default: True)

    Example:
        >>> segmenter = VGHubertMinCutSegmenter(mode='syllable')
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> segmenter.cite()
    """

    REFERENCE = (
        "Peng, P., Shang, Z., Harwath, D., & others (2023). "
        '"Syllable Discovery and Cross-Lingual Generalization in a Visually Grounded, '
        'Self-Supervised Speech Model." Interspeech 2023. '
        "https://doi.org/10.21437/Interspeech.2023-1430\n"
        "Code: https://github.com/jasonppy/syllable-discovery"
    )

    def __init__(
        self,
        layer: Optional[int] = None,
        mode: str = 'syllable',
        sec_per_syllable: float = 0.20,
        merge_threshold: Optional[float] = 0.3,
        use_optimized: bool = False,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        feature_extractor = VGHuBERTFeatureExtractor(layer=layer, mode=mode, device=device)
        super().__init__(
            feature_extractor=feature_extractor,
            sec_per_syllable=sec_per_syllable,
            use_reference=False,
            use_optimized=use_optimized,
            merge_threshold=merge_threshold,
            sample_rate=sample_rate,
            sad=sad,
            add_utterance_boundaries=add_utterance_boundaries,
        )
        self.layer = feature_extractor.layer
        self.mode = mode
        self.device = device


class VGHubertCLSSegmenter(CLSAttentionSegmenter):
    """
    VG-HuBERT with CLS attention segmentation (Peng & Harwath 2022).

    Replicates the word-discovery paper's canonical configuration:
    - Feature extractor: VG-HuBERT word checkpoint (layer 9)
    - Algorithm: CLS-token attention thresholding with per-head quantile union

    This is the configuration described in Peng & Harwath (Interspeech 2022), which
    introduced CLS attention segmentation using the word-level VG-HuBERT checkpoint.
    Use mode='syllable' only if you want layer-8 syllable-checkpoint features with
    CLS attention — that is not the canonical published configuration.

    Reference:
        Peng, P., & Harwath, D. (2022). "Self-Supervised Representation Learning for
        Speech Using Visual Grounding and Masked Language Modeling."
        Interspeech 2022. (word-discovery repo)

    Args:
        layer: VG-HuBERT layer (default: None = auto-select from mode)
        mode: Checkpoint and layer selection — 'word' (default, layer 9, word checkpoint)
              or 'syllable' (layer 8, syllable checkpoint).
        quantile: Per-head importance threshold (default: 0.9, matching the reference).
        min_distance: Optional gap tolerance in seconds for merging adjacent segments
                      (default: 0.0 = disabled, matching the reference).
        device: Device for model ('cuda', 'cpu', or None for auto-detect)
        sample_rate: Target sample rate (default: 16000)
        sad: Optional SAD backend for speech-region chunking (default: None)
        add_utterance_boundaries: No-op for this segmenter — CLS attention already
                                  marks salient frames across the full chunk (default: True)

    Example:
        >>> segmenter = VGHubertCLSSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> segmenter.cite()
    """

    REFERENCE = (
        "Peng, P., & Harwath, D. (2022). "
        '"Self-Supervised Representation Learning for Speech Using Visual Grounding '
        'and Masked Language Modeling." Interspeech 2022. '
        "https://doi.org/10.21437/Interspeech.2022-10631\n"
        "Code: https://github.com/jasonppy/word-discovery"
    )

    def __init__(
        self,
        layer: Optional[int] = None,
        mode: str = 'word',
        quantile: float = 0.9,
        min_distance: float = 0.0,
        device: Optional[str] = None,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(
            feature_extractor=VGHuBERTFeatureExtractor(layer=layer, mode=mode, device=device),
            layer=layer,
            mode=mode,
            quantile=quantile,
            min_distance=min_distance,
            device=device,
            sample_rate=sample_rate,
            sad=sad,
            add_utterance_boundaries=add_utterance_boundaries,
        )
        self.mode = mode
        self.device = device


class EnergyPeriodicitySegmenter(BaseSegmenter):
    """
    Energy + periodicity syllable-nucleus detection (Xie & Niyogi 2006).

    Two-stage: periodicity gates *where* voiced nuclei live (periodic regions),
    and energy picks the nucleus (the energy peak) within each region. The gate
    is ``periodicity > periodicity_threshold``; nuclei are convex-hull peaks of
    the energy.

    Two composition strategies (``composition=``) let you compare the faithful
    two-stage design against the multiplicative-gating reframe:

    - ``"slice"`` (default, faithful to Xie §2.3): periodicity gate proposes
      regions; the dB "relevant energy" is sliced per region and convex-hulled
      independently (peak-to-dip 4.5 dB). This is what the paper does.
    - ``"product"``: nuclei = convex-hull peaks of ``linear_energy * gate``, a
      single pass. Simpler, but flattens the per-region hulls and requires
      linear energy (a dB primary would invert under a 0/1 gate). ``weights``
      tunes each signal's influence (weighted geometric product; default equal).

    Tuned on TIMIT (60-file sweep, 200-file held-out validation): nuclei
    accuracy 84.8 / total error 31.0 vs the paper's 81.6 / 29.3 — at or above
    the paper's operating point.

    Reference:
        Xie, Z., & Niyogi, P. (2006). "Robust Acoustic-Based Syllable Detection."
        Interspeech 2006.

    Args:
        composition: "slice" (default) or "product".
        periodicity_threshold: absolute periodicity gate (default 0.3). The paper
            finds periodic regions by convex-hull on periodicity (peak-to-dip
            0.7); this class uses a simpler absolute threshold gate, whose
            equivalent operating point is lower (~0.3, tuned on TIMIT).
        energy_floor_db: drop gate frames whose relative energy is below this
            many dB (default -30.0; None disables). Plays the role of the
            paper's stop-closure energy floor (stated as 50 dB below max; our
            tighter floor compensates for the simpler threshold gate).
        energy_peak_to_dip: convex-hull dip threshold for picking energy peaks.
            Default 4.5 (dB, Xie Table 1) for "slice"; for "product" the energy
            is linear so pass a linear-domain value.
        weights: per-signal exponents for "product" composition,
            ``[w_energy, w_gate]`` (default equal weight 1).
        frame_length, hop_length: energy/periodicity framing (default 400/160 =
            25/10 ms at 16 kHz, Xie Table 1).
        min_syllable_dur: minimum nucleus-segment duration in seconds.
        sample_rate, sad, add_utterance_boundaries: as in BaseSegmenter.

    Example:
        >>> seg = EnergyPeriodicitySegmenter(composition="slice")
        >>> segments = seg.segment(audio, sr=16000)
        >>> seg.cite()
    """

    REFERENCE = (
        "Xie, Z., & Niyogi, P. (2006). "
        '"Robust Acoustic-Based Syllable Detection." '
        "Interspeech 2006. https://doi.org/10.21437/Interspeech.2006-440"
    )

    def __init__(
        self,
        composition: str = "slice",
        periodicity_threshold: float = 0.3,
        energy_floor_db: Optional[float] = -30.0,
        energy_peak_to_dip: float = 4.5,
        weights: Optional[list] = None,
        frame_length: int = 400,
        hop_length: int = 160,
        min_syllable_dur: float = 0.05,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        super().__init__(sample_rate=sample_rate, sad=sad,
                         add_utterance_boundaries=add_utterance_boundaries)
        from ..envelope import PeriodicityEnvelope, RMSEnvelope, ProductEnvelope, ThresholdGate

        if composition not in ("slice", "product"):
            raise ValueError(f"composition must be 'slice' or 'product', got {composition!r}")
        self.composition = composition
        self.periodicity_threshold = periodicity_threshold
        self.energy_floor_db = energy_floor_db
        self.energy_peak_to_dip = energy_peak_to_dip

        energy_db = RMSEnvelope(frame_length=frame_length, hop_length=hop_length,
                                db=True, reference="max")
        gate = ThresholdGate(PeriodicityEnvelope(), periodicity_threshold)
        if energy_floor_db is not None:
            # product of two 0/1 masks = logical AND (paper's stop-closure floor)
            gate = ProductEnvelope([gate, ThresholdGate(energy_db, energy_floor_db)])

        if composition == "slice":
            self._engine = RegionGatedSegmenter(
                primary_env=energy_db, gate_env=gate, gate_on=0.5,
                peak_to_dip=energy_peak_to_dip, min_syllable_dur=min_syllable_dur,
                sample_rate=sample_rate,
            )
        else:
            energy = RMSEnvelope(frame_length=frame_length, hop_length=hop_length, db=False)
            product = ProductEnvelope([energy, gate], weights=weights)
            self._engine = ConvexHullSegmenter(
                envelope_computer=product,
                peak_to_dip=energy_peak_to_dip, min_syllable_dur=min_syllable_dur,
                sample_rate=sample_rate,
            )

    def _segment(self, audio, sr) -> List[Tuple[float, float, float]]:
        # Outer BaseSegmenter.segment() applies SAD; engine runs SAD-free.
        return self._engine._segment(audio, sr)


class RhythmGuidedSegmenter(PeakdetectSegmenter):
    """
    Speech-rhythm guided syllable nuclei detection (Zhang & Glass 2009).

    Composition: ERB/gammatone Hilbert-sum envelope (the paper's E(t)) weighted
    by a batch-fitted rhythm sinusoid (RhythmEnvelope), gated by a periodicity
    voicing mask (the paper's pitch verification, §2.3, using findsylls's own
    PeriodicityEnvelope instead of an external pitch tracker), then peak-picked.

    Divergences from the paper (documented, all justified by the paper itself):
    - Whole-utterance batch rhythm fit instead of the iterative left-to-right
      loop (§3.1 reports "very similar results" for the batch fit).
    - Soft multiplicative rhythm weight instead of the hard one-peak-per-
      predicted-interval rule; the weight's floor keeps off-beat peaks damped
      rather than killed, mirroring the permissive 1.5-cycle search window.
    - Periodicity-threshold voicing instead of an ESPS pitch tracker.

    Performance (TIMIT, findsylls nuclei harness, 50 ms window): defaults give
    nuclei F1 ~85 (60-file tune 85.3, 200-file held-out 85.2), beating the SBS
    (84.1) and Theta (81.4) baselines on the same harness. The paper reports
    best-case F1 92.1 on its own vowel-center harness with per-corpus parameter
    selection; absolute numbers are not comparable across harnesses. Note: on
    this harness the rhythm weighting barely separates from the nRG ablation
    (~85.0), unlike the paper's +3.5 — likely because the batch fit + soft
    weight is gentler than the iterative hard-interval rule.

    Reference:
        Zhang, Y., & Glass, J. R. (2009). "Speech rhythm guided syllable nuclei
        detection." ICASSP 2009. https://doi.org/10.1109/ICASSP.2009.4960454

    Args:
        delta: Billauer peak/valley depth on the [0,1]-normalized weighted
            envelope (default 0.2, tuned on TIMIT).
        rhythm_floor: minimum rhythm weight in [0,1] (default 0.3; 1.0 disables
            rhythm weighting entirely — the paper's "nRG" ablation).
        voicing_threshold: periodicity gate for pitch verification (default 0.4;
            None disables).
        first_pass_delta: peakdetect delta for the first-pass peaks the rhythm
            sinusoid is fitted to (default 0.05).
        period_range: rhythm period bounds in seconds (default (0.1, 0.5)).
        min_syllable_dur: minimum segment duration in seconds (default 0.1,
            tuned on TIMIT).
        sample_rate, sad, add_utterance_boundaries: as in BaseSegmenter.

    Example:
        >>> segmenter = RhythmGuidedSegmenter()
        >>> segments = segmenter.segment(audio, sr=16000)
        >>> segmenter.cite()
    """

    REFERENCE = (
        "Zhang, Y., & Glass, J. R. (2009). "
        '"Speech rhythm guided syllable nuclei detection." '
        "ICASSP 2009, 3797-3800. https://doi.org/10.1109/ICASSP.2009.4960454"
    )

    def __init__(
        self,
        delta: float = 0.2,
        rhythm_floor: float = 0.3,
        voicing_threshold: Optional[float] = 0.4,
        first_pass_delta: float = 0.05,
        period_range: Tuple[float, float] = (0.1, 0.5),
        min_syllable_dur: float = 0.1,
        sample_rate: int = 16000,
        sad: Optional["BaseSAD"] = None,
        add_utterance_boundaries: bool = True,
    ):
        from ..envelope import (
            PeriodicityEnvelope,
            ProductEnvelope,
            RhythmEnvelope,
            ThresholdGate,
        )

        primary = RhythmEnvelope(delta=first_pass_delta, period_range=period_range,
                                 floor=rhythm_floor, output="weighted", normalize=True)
        if voicing_threshold is not None:
            envelope = ProductEnvelope(
                [primary, ThresholdGate(PeriodicityEnvelope(), voicing_threshold)]
            )
        else:
            envelope = primary
        super().__init__(
            envelope_computer=envelope,
            delta=delta,
            min_syllable_dur=min_syllable_dur,
            sample_rate=sample_rate,
            sad=sad,
            add_utterance_boundaries=add_utterance_boundaries,
        )
        self.rhythm_floor = rhythm_floor
        self.voicing_threshold = voicing_threshold


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

_SEGMENTER_PRESETS = {
    "sbs_peakdetect": SBSPeakdetectSegmenter,
    "theta_oscillator": ThetaOscillatorSegmenter,
    "energy_periodicity": EnergyPeriodicitySegmenter,
    "rhythm_guided": RhythmGuidedSegmenter,
    "sylber": SylberSegmenter,
    "vg_hubert_mincut": VGHubertMinCutSegmenter,
    "vg_hubert_cls": VGHubertCLSSegmenter,
}


def list_segmenter_presets() -> dict:
    """Return available preset segmenter names mapped to their classes.

    These are paper-replication configurations with fixed hyperparameters.
    Unlike ``list_presets()`` (which lists embedding pipeline configs), these
    are standalone segmenters that can be used directly or composed into a
    custom pipeline.

    Returns:
        Dict mapping preset name → segmenter class.

    Example:
        >>> from findsylls.segmentation.presets import list_segmenter_presets
        >>> list_segmenter_presets()
        {'theta_oscillator': ThetaOscillatorSegmenter, ...}
        >>> for name, cls in list_segmenter_presets().items():
        ...     print(name)
        ...     print(cls.REFERENCE)
    """
    return dict(_SEGMENTER_PRESETS)
