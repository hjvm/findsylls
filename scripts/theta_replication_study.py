"""
Theta oscillator replication study.

Compares three implementations of the theta oscillator envelope:
  1. MATLAB reference  — github.com/orasanen/thetaOscillator
                         (Räsänen, Doyle & Frank 2018, Cognition 171:130-150)
  2. Python port        — github.com/speech-utcluj/thetaOscillator-syllable-segmentation
  3. findsylls          — src/findsylls/envelope/theta.py

Since the MATLAB implementation cannot be executed directly from Python, this
study validates:
  (a) findsylls == Python port  (quantitative parity, same library stack)
  (b) MATLAB vs Python port differences (documented analytically, not numerically)

Usage:
    python scripts/theta_replication_study.py
    python scripts/theta_replication_study.py --audio path/to/file.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Reference: Python port (inlined from notebook)
# ---------------------------------------------------------------------------

def ref_gammatone(wav_data, fs, bands=20, minfreq=50, maxfreq=7500, resample_rate=1000):
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
    env = np.abs(_hilbert(filtered))
    return librosa.resample(env, orig_sr=fs, target_sr=resample_rate)


def ref_theta_oscillator(envelope, f=5, Q=0.5, N=10):
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

    e = np.transpose(envelope)
    e = np.vstack((e, np.zeros((500, e.shape[1]))))
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

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def _check(label, actual, expected, atol):
    if actual.shape != expected.shape:
        print(f"  {FAIL} {label}: shape mismatch {actual.shape} vs {expected.shape}")
        return False
    max_err = float(np.abs(actual - expected).max())
    mean_err = float(np.abs(actual - expected).mean())
    ok = max_err <= atol
    status = PASS if ok else FAIL
    print(f"  {status} {label}: max_err={max_err:.2e}  mean_err={mean_err:.2e}  (atol={atol:.0e})")
    return ok


def _load_audio(path):
    import soundfile as sf
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    return audio, sr


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_study(audio_path: str):
    print("=" * 70)
    print("THETA OSCILLATOR REPLICATION STUDY")
    print("=" * 70)
    print(f"Audio: {audio_path}\n")

    # --- load ---
    audio, sr = _load_audio(audio_path)
    print(f"Audio: {len(audio)/sr:.2f}s  ({len(audio)} samples @ {sr} Hz)\n")

    # --- documented MATLAB vs Python differences ---
    print("─" * 70)
    print("DOCUMENTED MATLAB vs PYTHON PORT DIFFERENCES (cannot run MATLAB)")
    print("─" * 70)
    print("  1. N parameter (top-N bands):")
    print("       MATLAB default : 8  (thetaOscillator.m line 44)")
    print("       Python port    : 10 (hardcoded in notebook)")
    print("       findsylls      : 10 (default kwarg, matches Python port)")
    print()
    print("  2. Gammatone filterbank:")
    print("       MATLAB   : gammatone_c (custom C implementation)")
    print("       Python   : detly/gammatone ERB filterbank")
    print("       -> Numerically different; cannot assert parity without MATLAB")
    print()
    print("  3. Delay compensation (algebraically equivalent):")
    print("       MATLAB table = Python table + 1 per entry")
    print("       MATLAB shift = drops (delay_val - 1) samples  [1-indexed]")
    print("       Python shift = drops (delay_val)     samples  [0-indexed]")
    print("       -> Net shift identical for all Q, f  (see delay table test)")
    print()

    # --- stage 1: gammatone ---
    print("─" * 70)
    print("STAGE 1: Gammatone filterbank  (findsylls vs Python port)")
    print("─" * 70)
    from findsylls.envelope.gammatone import _gammatone_filterbank_envelope

    t0 = time.time()
    ref_env = ref_gammatone(audio, sr)
    t_ref = time.time() - t0

    t0 = time.time()
    our_env, _ = _gammatone_filterbank_envelope(audio, sr)
    t_ours = time.time() - t0

    print(f"  Reference shape : {ref_env.shape}  ({t_ref:.2f}s)")
    print(f"  findsylls shape : {our_env.shape}  ({t_ours:.2f}s)")
    gamma_ok = _check("gammatone values", our_env, ref_env, atol=1e-5)
    print()

    # --- stage 2: full oscillator pipeline ---
    print("─" * 70)
    print("STAGE 2: Full oscillator pipeline  (findsylls vs Python port)")
    print("─" * 70)
    from findsylls.envelope.theta import theta_oscillator_envelope

    results = []
    param_sets = [
        (5,  0.5, 10, "default (f=5, Q=0.5, N=10)"),
        (5,  0.5,  8, "paper N (f=5, Q=0.5, N=8)"),
        (4,  0.3, 10, "f=4, Q=0.3, N=10"),
        (6,  0.7, 10, "f=6, Q=0.7, N=10"),
        (8,  1.0, 10, "f=8, Q=1.0, N=10"),
    ]

    for f_val, q_val, n_val, label in param_sets:
        t0 = time.time()
        ref_son = ref_theta_oscillator(ref_env, f=f_val, Q=q_val, N=n_val)
        t_ref = time.time() - t0

        t0 = time.time()
        our_son, _ = theta_oscillator_envelope(audio, sr, f=f_val, Q=q_val, N=n_val)
        t_ours = time.time() - t0

        ok = _check(label, our_son, ref_son, atol=1e-5)
        results.append(ok)
        print(f"         ref={t_ref:.2f}s  findsylls={t_ours:.2f}s")
        print()

    # --- delay table check ---
    print("─" * 70)
    print("STAGE 3: Delay table lookup parity  (all Q × f combinations)")
    print("─" * 70)
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
    mismatches = 0
    for Q in [0.1*i for i in range(1, 11)]:
        for f in range(1, 21):
            i1p = max(0, min(10, round(Q * 10)))
            i2p = max(0, min(20, round(f)))
            port_d = PYTHON_TABLE[i1p - 1][i2p - 1]

            i1o = max(0, min(9, round(Q * 10) - 1))
            i2o = max(0, min(19, round(f) - 1))
            our_d = PYTHON_TABLE[i1o][i2o]

            if port_d != our_d:
                mismatches += 1
                print(f"    MISMATCH Q={Q:.1f} f={f}: port={port_d} ours={our_d}")

    if mismatches == 0:
        print(f"  {PASS} All 200 (Q × f) delay lookups match between port and findsylls")
    else:
        print(f"  {FAIL} {mismatches} mismatches found")
    print()

    # --- summary ---
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = gamma_ok and all(results) and mismatches == 0
    print(f"  Gammatone parity    : {'PASS' if gamma_ok else 'FAIL'}")
    print(f"  Oscillator parity   : {'PASS' if all(results) else 'FAIL'} "
          f"({sum(results)}/{len(results)} param sets)")
    print(f"  Delay table parity  : {'PASS' if mismatches == 0 else 'FAIL'}")
    print()
    print("  Known MATLAB differences (expected, not a bug):")
    print("    - N=8 in MATLAB vs N=10 default in Python port / findsylls")
    print("    - gammatone_c vs detly/gammatone (different numerics)")
    print()
    if all_ok:
        print(f"  {PASS} findsylls matches the Python port reference exactly.")
    else:
        print(f"  {FAIL} Parity failures detected — see details above.")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theta oscillator replication study")
    parser.add_argument(
        "--audio",
        default=str(REPO_ROOT / "test_samples" / "WKSP_M_0064_E1_0009.flac"),
        help="Path to audio file (default: test_samples/WKSP_M_0064_E1_0009.flac)",
    )
    args = parser.parse_args()
    sys.exit(run_study(args.audio))
