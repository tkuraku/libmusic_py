# LIBMUSIC
# Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com
#
# test_etsi_valid
#
# Test detection on fractions of valid DTMFs.
# By peaks and by eigenfilter.
#
# date: August 2022
# Translated to Python 2024


# %%

import numpy as np
import matplotlib.pyplot as plt
from libmusic_py import lm_spectral_method, lm_globals, lm_dtmf

# %%


o = lm_globals()
d = lm_dtmf()

DTMF_IDX = 1
FRACTION_LEN = 14
M = 5

i = 0
j = FRACTION_LEN - 1
s = np.squeeze(o.g_dtmf_v_valid[i, j, :])

Fs = 8000
method = lm_spectral_method("music", M, 4)


roots_f1_validation = [
    941,
    697,
    697,
    697,
    770,
    770,
    770,
    852,
    852,
    852,
    697,
    770,
    852,
    941,
    941,
    941,
]

roots_f2_validation = [
    1336,
    1209,
    1336,
    1477,
    1209,
    1336,
    1477,
    1209,
    1336,
    1477,
    1633,
    1633,
    1633,
    1633,
    1209,
    1477,
]

c_validation = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "*",
    "#",
]
# %%

for i in range(16):
    decision = 0
    f1 = 0
    f2 = 0
    y = np.squeeze(o.g_dtmf_v_valid[i, FRACTION_LEN, :])
    _ = method.process(y[o.DTMF_START : o.DTMF_START + FRACTION_LEN])
    fs = method.eigenrooting(Fs)
    decision, f1, f2 = d.check_by_roots(fs, 0)
    c = o.dtmf_etsi_idx_2_symbol(i)
    if (
        decision != 1
        or f1 != roots_f1_validation[i]
        or f2 != roots_f2_validation[i]
        or c != c_validation[i]
    ):
        raise ValueError("Test Failed")
    print(f"By ROOTS i={i} ({c}), decision={decision}, f1={f1}, f2={f2}")

# %%

for i in range(16):
    decision = 0
    f1 = 0
    f2 = 0
    y = np.squeeze(o.g_dtmf_v_valid[i, FRACTION_LEN, :])
    _ = method.process(y[o.DTMF_START : o.DTMF_START + FRACTION_LEN])
    PEAK_WIDTH = 10
    peaks, pmu = method.peaks(np.arange(4000) + 1, Fs, PEAK_WIDTH)
    decision, f1, f2 = d.check_by_peaks(peaks[0], peaks[1])
    c = o.dtmf_etsi_idx_2_symbol(i)
    if (
        decision != 1
        or f1 != roots_f1_validation[i]
        or f2 != roots_f2_validation[i]
        or c != c_validation[i]
    ):
        raise ValueError("Test Failed")
    print(f"By Peaks i={i} ({c}), decision={decision}, f1={f1}, f2={f2}")
