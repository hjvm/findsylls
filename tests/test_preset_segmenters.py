"""
Tests for envelope-based preset segmenters: SBSPeakdetectSegmenter and ThetaOscillatorSegmenter.

All segmentation tests run on real audio from test_samples/. The integration
tests verify that preset output (syllable spans) is accepted by the embedding
pipeline (MFCC features, mean pooling — no GPU required).
"""

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

SAMPLE_WAV = Path(__file__).parent.parent / "test_samples" / "MMDB1_SI995.wav"
SAMPLE_FLAC = Path(__file__).parent.parent / "test_samples" / "WKSP_M_0064_E1_0009.flac"

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_audio():
    """Load MMDB1_SI995.wav — the primary test sample for all preset tests."""
    audio, sr = sf.read(str(SAMPLE_WAV))
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), int(sr)


@pytest.fixture(scope="module")
def sample_audio_flac():
    """Load WKSP_M_0064_E1_0009.flac for secondary verification."""
    audio, sr = sf.read(str(SAMPLE_FLAC))
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32), int(sr)


def _assert_valid_segments(segments, label, expect_nonempty=True):
    assert isinstance(segments, list), f"{label}: expected list, got {type(segments)}"
    if expect_nonempty:
        assert len(segments) > 0, f"{label}: expected at least one segment on real speech"
    for i, seg in enumerate(segments):
        assert len(seg) == 3, f"{label}[{i}]: expected (start, peak, end), got {seg}"
        start, peak, end = seg
        assert start >= 0.0, f"{label}[{i}]: negative start {start}"
        assert start <= peak, f"{label}[{i}]: start {start} > peak {peak}"
        assert peak <= end, f"{label}[{i}]: peak {peak} > end {end}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_list_segmenter_presets_contains_envelope_presets():
    from findsylls.segmentation.presets import list_segmenter_presets
    presets = list_segmenter_presets()
    assert "sbs_peakdetect" in presets, "sbs_peakdetect missing from list_segmenter_presets()"
    assert "theta_oscillator" in presets, "theta_oscillator missing from list_segmenter_presets()"


def test_list_segmenter_presets_contains_all_expected_keys():
    from findsylls.segmentation.presets import list_segmenter_presets
    expected = {"sbs_peakdetect", "theta_oscillator", "sylber", "vg_hubert_mincut", "vg_hubert_cls"}
    assert set(list_segmenter_presets().keys()) == expected


def test_envelope_presets_not_in_algorithmic_segmenter_registry():
    """Presets must not pollute list_segmenters() (algorithmic methods only)."""
    from findsylls.segmentation import list_segmenters
    algos = set(list_segmenters())
    assert "sbs_peakdetect" not in algos
    assert "theta_oscillator" not in algos


# ---------------------------------------------------------------------------
# REFERENCE attribute and cite()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset_name", ["sbs_peakdetect", "theta_oscillator"])
def test_envelope_preset_has_nonempty_reference(preset_name):
    from findsylls.segmentation.presets import list_segmenter_presets
    cls = list_segmenter_presets()[preset_name]
    assert hasattr(cls, "REFERENCE"), f"{preset_name}: missing REFERENCE attribute"
    assert isinstance(cls.REFERENCE, str) and len(cls.REFERENCE) > 20


@pytest.mark.parametrize("preset_name", ["sbs_peakdetect", "theta_oscillator"])
def test_cite_prints_nonempty_text(preset_name, capsys):
    from findsylls.segmentation.presets import list_segmenter_presets
    instance = list_segmenter_presets()[preset_name]()
    instance.cite()
    captured = capsys.readouterr().out
    assert len(captured.strip()) > 0, f"{preset_name}.cite() printed nothing"


def test_all_presets_have_reference_attribute():
    from findsylls.segmentation.presets import list_segmenter_presets
    for name, cls in list_segmenter_presets().items():
        assert hasattr(cls, "REFERENCE"), f"Preset '{name}' missing REFERENCE attribute"
        assert cls.REFERENCE, f"Preset '{name}' has empty REFERENCE"


# ---------------------------------------------------------------------------
# Output contract on real audio
# ---------------------------------------------------------------------------

def test_sbs_peakdetect_segment_contract_on_real_audio(sample_audio):
    from findsylls.segmentation.presets import SBSPeakdetectSegmenter
    audio, sr = sample_audio
    segs = SBSPeakdetectSegmenter().segment(audio=audio, sr=sr)
    _assert_valid_segments(segs, "SBSPeakdetectSegmenter")


def test_theta_oscillator_segment_contract_on_real_audio(sample_audio):
    from findsylls.segmentation.presets import ThetaOscillatorSegmenter
    audio, sr = sample_audio
    segs = ThetaOscillatorSegmenter().segment(audio=audio, sr=sr)
    _assert_valid_segments(segs, "ThetaOscillatorSegmenter")


def test_sbs_peakdetect_segment_contract_on_flac(sample_audio_flac):
    from findsylls.segmentation.presets import SBSPeakdetectSegmenter
    audio, sr = sample_audio_flac
    segs = SBSPeakdetectSegmenter().segment(audio=audio, sr=sr)
    _assert_valid_segments(segs, "SBSPeakdetectSegmenter (flac)")


def test_theta_oscillator_segment_contract_on_flac(sample_audio_flac):
    from findsylls.segmentation.presets import ThetaOscillatorSegmenter
    audio, sr = sample_audio_flac
    segs = ThetaOscillatorSegmenter().segment(audio=audio, sr=sr)
    _assert_valid_segments(segs, "ThetaOscillatorSegmenter (flac)")


# ---------------------------------------------------------------------------
# Pre-computed envelope passthrough
# ---------------------------------------------------------------------------

def test_sbs_precomputed_envelope_matches_audio_path(sample_audio):
    from findsylls.segmentation.presets import SBSPeakdetectSegmenter
    from findsylls.envelope.sbs import spectral_band_subtraction
    audio, sr = sample_audio
    envelope, times = spectral_band_subtraction(audio, sr)
    seg = SBSPeakdetectSegmenter()
    segs_audio = seg.segment(audio=audio, sr=sr)
    segs_precomp = seg.segment(envelope=envelope, times=times)
    assert segs_precomp == segs_audio, (
        "SBSPeakdetectSegmenter: pre-computed envelope path and audio path must agree"
    )


def test_theta_precomputed_envelope_matches_audio_path(sample_audio):
    from findsylls.segmentation.presets import ThetaOscillatorSegmenter
    from findsylls.envelope.theta import theta_oscillator_envelope
    audio, sr = sample_audio
    # Use same params as ThetaOscillatorSegmenter defaults: f=5, Q=0.5, N=8
    envelope, times = theta_oscillator_envelope(audio, sr, f=5, Q=0.5, N=8)
    seg = ThetaOscillatorSegmenter()
    segs_audio = seg.segment(audio=audio, sr=sr)
    segs_precomp = seg.segment(envelope=envelope, times=times)
    assert segs_precomp == segs_audio, (
        "ThetaOscillatorSegmenter: pre-computed envelope path and audio path must agree"
    )


# ---------------------------------------------------------------------------
# Parameter behavior on real audio
# ---------------------------------------------------------------------------

def test_max_syllable_dur_caps_all_spans(sample_audio):
    """Every span returned with max_syllable_dur=0.15 must be ≤ 0.15 s."""
    from findsylls.segmentation.peakdetect_segmenter import PeakdetectSegmenter
    from findsylls.envelope.sbs import SBSEnvelope
    audio, sr = sample_audio
    cap = 0.15
    segs = PeakdetectSegmenter(SBSEnvelope(), max_syllable_dur=cap).segment(audio=audio, sr=sr)
    for start, _, end in segs:
        assert (end - start) <= cap + 1e-6, (
            f"Span {end - start:.4f}s exceeds max_syllable_dur={cap}"
        )


def test_max_syllable_dur_reduces_or_preserves_count(sample_audio):
    """A tight max_syllable_dur must not produce more syllables than no cap."""
    from findsylls.segmentation.peakdetect_segmenter import PeakdetectSegmenter
    from findsylls.envelope.sbs import SBSEnvelope
    audio, sr = sample_audio
    segs_uncapped = PeakdetectSegmenter(SBSEnvelope()).segment(audio=audio, sr=sr)
    segs_capped = PeakdetectSegmenter(SBSEnvelope(), max_syllable_dur=0.15).segment(audio=audio, sr=sr)
    assert len(segs_capped) <= len(segs_uncapped)


def test_amplitude_ratio_tol_reduces_or_preserves_count(sample_audio):
    """A strict amplitude_ratio_tol must not produce more syllables than no filter."""
    from findsylls.segmentation.peakdetect_segmenter import PeakdetectSegmenter
    from findsylls.envelope.sbs import SBSEnvelope
    audio, sr = sample_audio
    segs_unfiltered = PeakdetectSegmenter(SBSEnvelope()).segment(audio=audio, sr=sr)
    # 0.1 = keep only very deep valleys (strict); should prune some syllables
    segs_filtered = PeakdetectSegmenter(SBSEnvelope(), amplitude_ratio_tol=0.1).segment(audio=audio, sr=sr)
    assert len(segs_filtered) <= len(segs_unfiltered)


def test_sbs_smoothing_window_samples_affects_output(sample_audio):
    """Different smoothing window sizes should produce different envelopes and segment counts."""
    from findsylls.envelope.sbs import SBSEnvelope
    from findsylls.segmentation.peakdetect_segmenter import PeakdetectSegmenter
    audio, sr = sample_audio
    segs_narrow = PeakdetectSegmenter(SBSEnvelope(smoothing_window_samples=3)).segment(audio=audio, sr=sr)
    segs_wide = PeakdetectSegmenter(SBSEnvelope(smoothing_window_samples=21)).segment(audio=audio, sr=sr)
    # Wider smoothing suppresses fine detail — expect fewer or equal segments
    assert len(segs_wide) <= len(segs_narrow)


# ---------------------------------------------------------------------------
# Embedding pipeline integration (no GPU — uses MFCC + mean pooling)
# ---------------------------------------------------------------------------

def test_sbs_preset_segments_accepted_by_embedding_pipeline(sample_audio):
    """SBSPeakdetectSegmenter output is compatible with the MFCC + mean-pooling embedding path."""
    from findsylls.segmentation.presets import SBSPeakdetectSegmenter
    from findsylls.features import MFCCExtractor
    from findsylls.embedding.poolers.mean import MeanPooler

    audio, sr = sample_audio
    segments = SBSPeakdetectSegmenter().segment(audio=audio, sr=sr)
    assert len(segments) > 0, "Need at least one segment to test pooling"

    features = MFCCExtractor().extract(audio, sr)  # (n_frames, n_mfcc)
    frame_rate = MFCCExtractor().frame_rate
    pooler = MeanPooler()

    embeddings = pooler.pool(features, segments, fps=frame_rate)
    assert embeddings.shape == (len(segments), features.shape[1]), (
        f"Expected embeddings shape ({len(segments)}, {features.shape[1]}), got {embeddings.shape}"
    )


def test_theta_preset_segments_accepted_by_embedding_pipeline(sample_audio):
    """ThetaOscillatorSegmenter output is compatible with the MFCC + mean-pooling embedding path."""
    from findsylls.segmentation.presets import ThetaOscillatorSegmenter
    from findsylls.features import MFCCExtractor
    from findsylls.embedding.poolers.mean import MeanPooler

    audio, sr = sample_audio
    segments = ThetaOscillatorSegmenter().segment(audio=audio, sr=sr)
    assert len(segments) > 0

    features = MFCCExtractor().extract(audio, sr)
    frame_rate = MFCCExtractor().frame_rate
    pooler = MeanPooler()

    embeddings = pooler.pool(features, segments, fps=frame_rate)
    assert embeddings.shape == (len(segments), features.shape[1])
