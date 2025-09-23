import pathlib

import pytest

import findsylls


def test_version():
    assert hasattr(findsylls, "__version__")


def test_segment_audio_sample():
    sample = pathlib.Path("test_samples/SP20_117.wav")
    if not sample.exists():
        pytest.skip("sample audio not present")
    sylls, env, t = findsylls.segment_audio(str(sample), envelope_fn="hilbert")
    # Basic shape assertions
    assert len(env) == len(t)
    # Not asserting non-empty syllables (could be silent clip); ensure no exception
    assert isinstance(sylls, list)
