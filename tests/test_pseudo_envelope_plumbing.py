import numpy as np

from findsylls.embedding.pipeline import embed_audio
from findsylls.envelope import dispatch as envelope_dispatch


def test_pseudo_envelope_dispatch_resolves_feature_extractor(monkeypatch):
    calls = {}

    class DummyExtractor:
        supports_attention = True

    class DummyEnvelope:
        def __init__(self, feature_extractor, **kwargs):
            calls["feature_extractor"] = feature_extractor
            calls["kwargs"] = kwargs

        def compute(self, waveform, sr):
            calls["waveform_shape"] = tuple(np.asarray(waveform).shape)
            calls["sr"] = sr
            return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.1], dtype=np.float32)

    def fake_get_extractor(feature_type, **feature_kwargs):
        calls["feature_type"] = feature_type
        calls["feature_kwargs"] = feature_kwargs
        return DummyExtractor()

    monkeypatch.setattr(envelope_dispatch, "get_extractor", fake_get_extractor)
    monkeypatch.setattr(envelope_dispatch, "CLSAttentionEnvelope", DummyEnvelope)

    envelope, times = envelope_dispatch.get_amplitude_envelope(
        np.zeros(8, dtype=np.float32),
        16000,
        method="cls_attention",
        feature_type="hubert",
        feature_kwargs={"alpha": 1},
        layer=9,
        quantile=0.25,
    )

    assert calls["feature_type"] == "hubert"
    assert calls["feature_kwargs"] == {"alpha": 1, "layer": 9}
    assert calls["feature_extractor"].__class__.__name__ == "DummyExtractor"
    assert calls["kwargs"] == {"quantile": 0.25}
    assert envelope.shape == times.shape == (2,)


def test_peakdetect_pseudo_envelope_forwards_feature_context(monkeypatch):
    calls = {}

    def fake_load_audio(audio_file, samplerate=16000):
        calls["audio_file"] = audio_file
        calls["samplerate"] = samplerate
        return np.zeros(16, dtype=np.float32), samplerate

    def fake_segment_audio_pipeline(**kwargs):
        calls["segment_audio_pipeline"] = kwargs
        return [(0.0, 0.1, 0.2)], None, None

    def fake_extract_features(audio, sr, method, layer=None, device="auto", return_times=False, **kwargs):
        calls["extract_features"] = {
            "method": method,
            "layer": layer,
            "device": device,
            "return_times": return_times,
            "kwargs": kwargs,
        }
        features = np.ones((4, 3), dtype=np.float32)
        times = np.arange(4, dtype=np.float32) * 0.05
        return (features, times) if return_times else features

    def fake_pool_syllables(frame_features, syllables, sr, method, hop_length, **kwargs):
        calls["pool_syllables"] = {
            "method": method,
            "hop_length": hop_length,
            "syllables": syllables,
        }
        return np.ones((len(syllables), frame_features.shape[1]), dtype=np.float32)

    monkeypatch.setattr("findsylls.embedding.pipeline.load_audio", fake_load_audio)
    monkeypatch.setattr("findsylls.embedding.pipeline.segment_audio_pipeline", fake_segment_audio_pipeline)
    monkeypatch.setattr("findsylls.embedding.pipeline.extract_features", fake_extract_features)
    monkeypatch.setattr("findsylls.embedding.pipeline.pool_syllables", fake_pool_syllables)

    embeddings, metadata = embed_audio(
        "dummy.wav",
        segmentation="peakdetect",
        features="hubert",
        pooling="mean",
        segmentation_kwargs={"envelope_method": "cls_attention"},
        feature_kwargs={"model_name": "facebook/hubert-base-ls960"},
        return_metadata=True,
    )

    assert calls["segment_audio_pipeline"]["method"] == "peakdetect"
    assert calls["segment_audio_pipeline"]["segmentation_kwargs"]["envelope_method"] == "cls_attention"
    envelope_kwargs = calls["segment_audio_pipeline"]["segmentation_kwargs"]["envelope_kwargs"]
    assert envelope_kwargs["feature_type"] == "hubert"
    assert envelope_kwargs["feature_kwargs"] == {"model_name": "facebook/hubert-base-ls960"}
    assert calls["extract_features"]["return_times"] is True
    assert calls["extract_features"]["method"] == "hubert"
    assert embeddings.shape == (1, 3)
    assert metadata["segmentation_method"] == "peakdetect"


def test_sylber_onc_uses_requested_segmentation(monkeypatch):
    calls = {}

    def fake_load_audio(audio_file, samplerate=16000):
        return np.zeros(16, dtype=np.float32), samplerate

    def fake_segment_audio_pipeline(**kwargs):
        calls["segment_audio_pipeline"] = kwargs
        return [(0.0, 0.1, 0.2)], None, None

    def fake_extract_features(audio, sr, method, layer=None, device="auto", return_times=False, return_segments=False, **kwargs):
        calls["extract_features"] = {
            "method": method,
            "layer": layer,
            "device": device,
            "return_times": return_times,
            "return_segments": return_segments,
            "kwargs": kwargs,
        }
        if return_segments:
            raise AssertionError("legacy Sylber shortcut should not be used")
        features = np.ones((4, 3), dtype=np.float32)
        times = np.arange(4, dtype=np.float32) * 0.05
        return (features, times) if return_times else features

    def fake_pool_syllables(frame_features, syllables, sr, method, hop_length, **kwargs):
        return np.ones((len(syllables), frame_features.shape[1]), dtype=np.float32)

    monkeypatch.setattr("findsylls.embedding.pipeline.load_audio", fake_load_audio)
    monkeypatch.setattr("findsylls.embedding.pipeline.segment_audio_pipeline", fake_segment_audio_pipeline)
    monkeypatch.setattr("findsylls.embedding.pipeline.extract_features", fake_extract_features)
    monkeypatch.setattr("findsylls.embedding.pipeline.pool_syllables", fake_pool_syllables)

    embeddings, metadata = embed_audio(
        "dummy.wav",
        segmentation="cls_attention",
        features="sylber",
        pooling="onc",
        feature_kwargs={"model_ckpt": "cheoljun95/sylber"},
        return_metadata=True,
    )

    assert calls["segment_audio_pipeline"]["method"] == "cls_attention"
    assert calls["extract_features"]["return_segments"] is False
    assert embeddings.shape == (1, 3)
    assert metadata["segmentation_method"] == "cls_attention"


def test_mincut_pseudo_envelope_defaults_to_inverted_trace(monkeypatch):
    calls = {}

    class DummyExtractor:
        supports_attention = False

    class DummyEnvelope:
        def __init__(self, feature_extractor, **kwargs):
            calls["feature_extractor"] = feature_extractor
            calls["kwargs"] = kwargs

        def compute(self, waveform, sr):
            return np.array([0.0, 1.0], dtype=np.float32), np.array([0.0, 0.1], dtype=np.float32)

    def fake_get_extractor(feature_type, **feature_kwargs):
        calls["feature_type"] = feature_type
        calls["feature_kwargs"] = feature_kwargs
        return DummyExtractor()

    monkeypatch.setattr(envelope_dispatch, "get_extractor", fake_get_extractor)
    monkeypatch.setattr(envelope_dispatch, "MinCutEnvelope", DummyEnvelope)

    envelope, times = envelope_dispatch.get_amplitude_envelope(
        np.zeros(8, dtype=np.float32),
        16000,
        method="mincut",
        feature_type="hubert",
        feature_kwargs={"alpha": 2},
        threshold=0.5,
    )

    assert calls["feature_type"] == "hubert"
    assert calls["feature_kwargs"] == {"alpha": 2}
    assert calls["kwargs"].get("invert", False) is False
    assert calls["kwargs"]["threshold"] == 0.5
    assert envelope.shape == times.shape == (2,)