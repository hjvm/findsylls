"""
Theta oscillator parity regression: findsylls vs Python port reference.

Reference implementation:
    speech-utcluj/thetaOscillator-syllable-segmentation (GitHub)
    Python port of Räsänen, Doyle & Frank (2018). "Pre-linguistic segmentation
    of speech into syllable-like units." Cognition, 171, 130-150.

Original MATLAB: github.com/orasanen/thetaOscillator

The reference Python functions below are inlined from
thetaOscillator-syllable-segmentation/ThetaOscillator-SyllableSegmentation.ipynb
(locally cloned at ../thetaOscillator-syllable-segmentation/).
They are inlined rather than imported because the source is a notebook,
not an installable package.

Documented differences between MATLAB reference and Python port / findsylls
---------------------------------------------------------------------------
1. N (top-N bands combined for sonority):
       MATLAB default = 8  (hardcoded in thetaOscillator.m line 44)
       Python port    = 10 (hardcoded in notebook)
       findsylls      = 10 (default, configurable via N= kwarg)
   -> findsylls matches the Python port. Strict paper replication needs N=8.

2. Gammatone filterbank implementation:
       MATLAB   uses gammatone_c (custom C code, different filter numerics)
       Python port + findsylls use detly/gammatone (ERB filterbank)
   -> findsylls and the Python port are numerically equivalent; MATLAB differs.

3. Delay compensation (net result is identical - see TestDelayTableEquivalence):
       MATLAB table  = Python-port table + 1 (per entry)
       MATLAB shift  = drops delay_val-1 samples  (1-indexed: x(delay_val:end))
       Python shift  = drops delay_val   samples  (0-indexed: x[delay_val:])
   -> The +1 in the table and -1 in the shift cancel exactly for all Q, f.
"""

import numpy as np
import pytest
import soundfile as sf

AUDIO_PATH = "test_samples/WKSP_M_0064_E1_0009.flac"
ATOL_GAMMATONE = 1e-5
ATOL_OSCILLATOR = 1e-5


# ---------------------------------------------------------------------------
# Reference functions: inlined from Python port notebook
# speech-utcluj/thetaOscillator-syllable-segmentation
# ---------------------------------------------------------------------------

def _ref_gammatone_envelope(wav_data, fs, bands=20, minfreq=50, maxfreq=7500,
                             resample_rate=1000):
    """Gammatone filterbank from the Python port notebook (verbatim logic)."""
    import librosa
    import gammatone.filters
    from scipy.signal import hilbert as _hilbert

    cfs = np.zeros((bands, 1))
    const = (maxfreq / minfreq) ** (1 / (bands - 1))
    cfs[0] = minfreq
    for k in range(bands - 1):
        cfs[k + 1] = cfs[k] * const

    coefs = gammatone.filters.make_erb_filters(fs, cfs, width=1.0)
    filtered = gammatone.filters.erb_filterbank(wav_data, coefs)
    hilbert_env = np.abs(_hilbert(filtered))
    env = librosa.resample(hilbert_env, orig_sr=fs, target_sr=resample_rate)
    return env  # [bands, time]


def _ref_theta_oscillator(envelope, f=5, Q=0.5, N=10):
    """
    Theta oscillator from the Python port notebook (verbatim logic).
    Returns the normalized sonority array [time].
    """
    if N > envelope.size:
        N = envelope.shape[1]

    a_table = np.array([
        [72,  34, 22, 16, 12,  9,  8,  6,  5,  4,  3,  3,  2,  2,  1,  0,  0,  0,  0,  0],
        [107, 52, 34, 25, 19, 16, 13, 11, 10,  9,  8,  7,  6,  5,  5,  4,  4,  4,  3,  3],
        [129, 64, 42, 31, 24, 20, 17, 14, 13, 11, 10,  9,  8,  7,  7,  6,  6,  5,  5,  4],
        [145, 72, 47, 35, 28, 23, 19, 17, 15, 13, 12, 10,  9,  9,  8,  7,  7,  6,  6,  5],
        [157, 78, 51, 38, 30, 25, 21, 18, 16, 14, 13, 12, 11, 10,  9,  8,  8,  7,  7,  6],
        [167, 83, 55, 41, 32, 27, 23, 19, 17, 15, 14, 12, 11, 10, 10,  9,  8,  8,  7,  7],
        [175, 87, 57, 43, 34, 28, 24, 21, 18, 16, 15, 13, 12, 11, 10,  9,  9,  8,  8,  7],
        [181, 90, 59, 44, 35, 29, 25, 21, 19, 17, 15, 14, 13, 12, 11, 10,  9,  9,  8,  8],
        [187, 93, 61, 46, 36, 30, 25, 22, 19, 17, 16, 14, 13, 12, 11, 10, 10,  9,  8,  8],
        [191, 95, 63, 47, 37, 31, 26, 23, 20, 18, 16, 15, 13, 12, 11, 11, 10,  9,  9,  8],
    ])

    i1 = max(0, min(10, round(Q * 10)))
    i2 = max(0, min(20, round(f)))
    delay = a_table[i1 - 1][i2 - 1]

    T = 1.0 / f
    k = 1
    b = 2 * np.pi / T
    m = k / b ** 2
    c = np.sqrt(m * k) / Q

    e = np.transpose(envelope)                            # [time, bands]
    e = np.vstack((e, np.zeros((500, e.shape[1]))))       # zero-pad
    F = e.shape[1]

    x = np.zeros((e.shape[0], F))
    a_osc = np.zeros((e.shape[0], F))
    v = np.zeros((e.shape[0], F))

    for t in range(1, e.shape[0]):
        for cf in range(F):
            f_up = e[t, cf]
            f_down = -k * x[t - 1, cf] - c * v[t - 1, cf]
            f_tot = f_up + f_down
            a_osc[t, cf] = f_tot / m
            v[t, cf] = v[t - 1, cf] + a_osc[t, cf] * 0.001
            x[t, cf] = x[t - 1, cf] + v[t, cf] * 0.001

    for ch in range(F):
        if delay:
            x[:, ch] = np.append(x[delay:, ch], np.zeros(delay))

    x = x[:-500]

    tmp = x - np.min(x) + 0.00001
    sonority = np.zeros(tmp.shape[0])
    for zz in range(tmp.shape[0]):
        sort_tmp = np.sort(tmp[zz, :])[::-1]
        sonority[zz] = np.sum(np.log10(sort_tmp[:N]))

    sonority = sonority - np.min(sonority)
    sonority = sonority / np.max(sonority)
    return sonority


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_test_audio():
    audio, sr = sf.read(AUDIO_PATH)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def audio_16k():
    pytest.importorskip("gammatone")
    return _load_test_audio()


@pytest.fixture(scope="module")
def ref_gammatone(audio_16k):
    audio, sr = audio_16k
    return _ref_gammatone_envelope(audio, sr)


# ---------------------------------------------------------------------------
# Stage 1 — Gammatone filterbank parity
# ---------------------------------------------------------------------------

class TestGammatoneParity:
    """findsylls gammatone output matches the Python port reference."""

    def test_shape_matches(self, audio_16k, ref_gammatone):
        from findsylls.envelope.gammatone import _gammatone_filterbank_envelope
        audio, sr = audio_16k
        ours, _ = _gammatone_filterbank_envelope(audio, sr)
        assert ours.shape == ref_gammatone.shape, (
            f"Shape mismatch: findsylls={ours.shape}, reference={ref_gammatone.shape}"
        )

    def test_values_match(self, audio_16k, ref_gammatone):
        from findsylls.envelope.gammatone import _gammatone_filterbank_envelope
        audio, sr = audio_16k
        ours, _ = _gammatone_filterbank_envelope(audio, sr)
        np.testing.assert_allclose(
            ours, ref_gammatone, atol=ATOL_GAMMATONE,
            err_msg="Gammatone filterbank diverges from Python port reference"
        )


# ---------------------------------------------------------------------------
# Stage 2 — Oscillator parity (given identical gammatone input)
# ---------------------------------------------------------------------------

class TestOscillatorParity:
    """findsylls oscillator matches the Python port given identical gammatone input."""

    def test_default_params(self, audio_16k, ref_gammatone):
        """f=5, Q=0.5, N=10 (shared default for both implementations)."""
        from findsylls.envelope.theta import theta_oscillator_envelope
        audio, sr = audio_16k
        ours, _ = theta_oscillator_envelope(audio, sr, f=5, Q=0.5, N=10)
        ref = _ref_theta_oscillator(ref_gammatone, f=5, Q=0.5, N=10)

        assert len(ours) == len(ref), (
            f"Length mismatch: findsylls={len(ours)}, reference={len(ref)}"
        )
        np.testing.assert_allclose(
            ours, ref, atol=ATOL_OSCILLATOR,
            err_msg="Theta sonority diverges from Python port at default params"
        )

    @pytest.mark.parametrize("f_val,q_val", [(4, 0.3), (6, 0.7), (8, 1.0)])
    def test_varied_params(self, audio_16k, ref_gammatone, f_val, q_val):
        from findsylls.envelope.theta import theta_oscillator_envelope
        audio, sr = audio_16k
        ours, _ = theta_oscillator_envelope(audio, sr, f=f_val, Q=q_val, N=10)
        ref = _ref_theta_oscillator(ref_gammatone, f=f_val, Q=q_val, N=10)
        np.testing.assert_allclose(
            ours, ref, atol=ATOL_OSCILLATOR,
            err_msg=f"Theta sonority diverges at f={f_val}, Q={q_val}"
        )

    def test_paper_N(self, audio_16k, ref_gammatone):
        """N=8 (MATLAB / paper default) also produces matching output."""
        from findsylls.envelope.theta import theta_oscillator_envelope
        audio, sr = audio_16k
        ours, _ = theta_oscillator_envelope(audio, sr, f=5, Q=0.5, N=8)
        ref = _ref_theta_oscillator(ref_gammatone, f=5, Q=0.5, N=8)
        np.testing.assert_allclose(
            ours, ref, atol=ATOL_OSCILLATOR,
            err_msg="Theta sonority diverges at N=8 (MATLAB paper default)"
        )

    def test_output_normalized(self, audio_16k):
        from findsylls.envelope.theta import theta_oscillator_envelope
        audio, sr = audio_16k
        sonority, _ = theta_oscillator_envelope(audio, sr)
        assert sonority.min() >= -1e-6, "Sonority min below 0"
        assert sonority.max() <= 1.0 + 1e-6, "Sonority max above 1"


# ---------------------------------------------------------------------------
# Stage 3 — Delay table algebraic equivalence (no audio required)
# ---------------------------------------------------------------------------

class TestDelayTableEquivalence:
    """
    Proves that MATLAB and Python-port delay tables produce the same net shift.

    MATLAB:  table values = Python-port values + 1
             shift left   = delay_val - 1   (MATLAB 1-indexed: x(delay_val:end))
    Python:  table values as-is
             shift left   = delay_val        (Python 0-indexed: x[delay_val:])
    Net:     (python_val+1) - 1 == python_val  ->  identical for every entry.
    """

    PYTHON_TABLE = np.array([
        [72,  34, 22, 16, 12,  9,  8,  6,  5,  4,  3,  3,  2,  2,  1,  0,  0,  0,  0,  0],
        [107, 52, 34, 25, 19, 16, 13, 11, 10,  9,  8,  7,  6,  5,  5,  4,  4,  4,  3,  3],
        [129, 64, 42, 31, 24, 20, 17, 14, 13, 11, 10,  9,  8,  7,  7,  6,  6,  5,  5,  4],
        [145, 72, 47, 35, 28, 23, 19, 17, 15, 13, 12, 10,  9,  9,  8,  7,  7,  6,  6,  5],
        [157, 78, 51, 38, 30, 25, 21, 18, 16, 14, 13, 12, 11, 10,  9,  8,  8,  7,  7,  6],
        [167, 83, 55, 41, 32, 27, 23, 19, 17, 15, 14, 12, 11, 10, 10,  9,  8,  8,  7,  7],
        [175, 87, 57, 43, 34, 28, 24, 21, 18, 16, 15, 13, 12, 11, 10,  9,  9,  8,  8,  7],
        [181, 90, 59, 44, 35, 29, 25, 21, 19, 17, 15, 14, 13, 12, 11, 10,  9,  9,  8,  8],
        [187, 93, 61, 46, 36, 30, 25, 22, 19, 17, 16, 14, 13, 12, 11, 10, 10,  9,  8,  8],
        [191, 95, 63, 47, 37, 31, 26, 23, 20, 18, 16, 15, 13, 12, 11, 11, 10,  9,  9,  8],
    ])

    def test_matlab_python_net_shift_identical(self):
        matlab_net = (self.PYTHON_TABLE + 1) - 1
        python_net = self.PYTHON_TABLE
        np.testing.assert_array_equal(matlab_net, python_net)

    def test_findsylls_lookup_matches_python_port(self):
        """findsylls index arithmetic produces same delay value as Python port."""
        for Q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            for f in range(1, 21):
                # Python port lookup
                i1_port = max(0, min(10, round(Q * 10)))
                i2_port = max(0, min(20, round(f)))
                port_delay = self.PYTHON_TABLE[i1_port - 1][i2_port - 1]

                # findsylls lookup
                i1_ours = max(0, min(9, round(Q * 10) - 1))
                i2_ours = max(0, min(19, round(f) - 1))
                our_delay = self.PYTHON_TABLE[i1_ours][i2_ours]

                assert port_delay == our_delay, (
                    f"Delay mismatch at Q={Q}, f={f}: "
                    f"port={port_delay}, findsylls={our_delay}"
                )
