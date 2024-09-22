# LIBMUSIC
# Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com
#
# test_etsi
#
# Some tests of testing.
#
# date: August 2022


import numpy as np
import matplotlib.pyplot as plt
from libmusic_py import lm_spectral_method, lm_globals, lm_dtmf

o = lm_globals()
d = lm_dtmf()

FRACTION_LEN = 16

dtmfs = [1, 5, 6, 8, 9, 10]
M = 5
temp = np.arange(4000) + 1

Fs = 8000
method = lm_spectral_method("music", M, 4)

if True:
    for i in range(16):
        decision = 0
        f1 = 0
        f2 = 0
        y = o.g_dtmf_etsi(i, np.arrange(FRACTION_LEN) + 1)
        # figure
        # plot(y)
        _ = method.process(y)
        fs = method.eigenrooting(Fs, 0, 0)
        decision, f1, f2 = d.check_by_roots(fs, 0)
        print(f"By ROOTS i={i}, decision={decision}, f1={f1}, f2={f2}")

if True:
    plt.figure()
    for i in range(16):
        decision = 0
        f1 = 0
        f2 = 0
        y = o.g_dtmf_etsi(i, np.arrange(FRACTION_LEN) + 1)
        plt.subplot(4, 4, i + 1)
        plt.plot(y)
        c = o.dtmf_etsi_idx_2_symbol(i)
        plt.title(c)
        _ = method.process(y)
        peaks, pmu = method.peaks(temp, Fs, 0)
        decision, f1, f2 = d.check_by_peaks(peaks(1), peaks(2), 0)
        print(f"By PEAKS i={i} ({c}), decision={decision}, f1={f1}, f2={f2}")

Fs = 8000
M = 5
y = o.g_dtmf_etsi(1, np.arrange(FRACTION_LEN) + 1)

plt.figure()

PEAK_WIDTH = 10
method = lm_spectral_method("music", M, 4)
_ = method.process(y)
peaks, pmu = method.peaks(temp, Fs, PEAK_WIDTH)
X2, d2 = method.psd(method, temp, Fs)
plt.subplot(1, 3, 1)
plt.plot(X2)

PEAK_WIDTH = 20
method = lm_spectral_method("ev", M, 4)
_ = method.process(y)
peaks, pmu = method.peaks(temp, Fs, PEAK_WIDTH)
X2, d2 = method.psd(method, temp, Fs)
plt.subplot(1, 3, 2)
plt.plot(X2)

PEAK_WIDTH = 30
method = lm_spectral_method("mn", M, 4)
_ = method.process(y)
peaks, pmu = method.peaks(temp, Fs, PEAK_WIDTH)
X2, d2 = method.psd(method, temp, Fs)
plt.subplot(1, 3, 3)
plt.plot(X2)

plt.show()
