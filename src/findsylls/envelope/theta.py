import numpy as np
from .gammatone import gammatone_filterbank_envelope

def theta_oscillator_envelope(waveform, sr, **kwargs):
    f = kwargs.get("f", 5)
    Q = kwargs.get("Q", 0.5)
    N = kwargs.get("N", 10)
    verbose = kwargs.get("verbose", 0)
    envelope, _ = gammatone_filterbank_envelope(waveform, sr, **kwargs)
    if N > envelope.shape[1]:
        N = envelope.shape[1]
    delays = np.array([[72,34,22,16,12,9,8,6,5,4,3,3,2,2,1,0,0,0,0,0],[107,52,34,25,19,16,13,11,10,9,8,7,6,5,5,4,4,4,3,3],[129,64,42,31,24,20,17,14,13,11,10,9,8,7,7,6,6,5,5,4],[145,72,47,35,28,23,19,17,15,13,12,10,9,9,8,7,7,6,6,5],[157,78,51,38,30,25,21,18,16,14,13,12,11,10,9,8,8,7,7,6],[167,83,55,41,32,27,23,19,17,15,14,12,11,10,10,9,8,8,7,7],[175,87,57,43,34,28,24,21,18,16,15,13,12,11,10,9,9,8,8,7],[181,90,59,44,35,29,25,21,19,17,15,14,13,12,11,10,9,9,8,8],[187,93,61,46,36,30,25,22,19,17,16,14,13,12,11,10,10,9,8,8],[191,95,63,47,37,31,26,23,20,18,16,15,13,12,11,11,10,9,9,8]])
    i1 = max(0, min(9, round(Q * 10) - 1))
    i2 = max(0, min(19, round(f) - 1))
    delay = delays[i1][i2]
    T = 1.0 / f
    k = 1
    b = 2 * np.pi / T
    m = k / b**2
    c = np.sqrt(m * k) / Q
    e = np.transpose(envelope)
    e = np.vstack((e, np.zeros((500, e.shape[1]))))
    x = np.zeros_like(e); v = np.zeros_like(e); a = np.zeros_like(e)
    dt = 0.001
    for t in range(1, e.shape[0]):
        f_up = e[t]
        f_down = -k * x[t-1] - c * v[t-1]
        f_tot = f_up + f_down
        a[t] = f_tot / m
        v[t] = v[t-1] + a[t] * dt
        x[t] = x[t-1] + v[t] * dt
    x = np.roll(x, -delay, axis=0)
    x[-delay:, :] = 0
    x = x[:-500]
    x = x - np.min(x) + 1e-5
    log_x = np.log10(x.clip(min=1e-5))
    idx = np.argpartition(log_x, -N, axis=1)
    topN = np.take_along_axis(log_x, idx[:, -N:], axis=1)
    sonority = topN.sum(axis=1)
    sonority = sonority - np.min(sonority)
    sonority = sonority / np.max(sonority)
    times = np.linspace(0, len(waveform)/sr, num=len(sonority))
    return sonority, times
