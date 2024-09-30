# LIBMUSIC
# Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com
#
# test_etsi_valid_2 - Test detection on fractions of valid DTMFs 2.
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

DETECT_BLOCK_LEN = [10, 11, 12]
# enum like sigma

N_START = 12
N_END = 16

Fs = 8000

for i in range(len(DETECT_BLOCK_LEN)):
    BLOCK_LEN = DETECT_BLOCK_LEN[i]
    # vectors = zeros(16,o.TEST_VECTOR_LEN);
    for N in range(N_START - 1, N_END):  # iterate fractions len
        print(f"== Block/Fraction: {BLOCK_LEN}/{N}")

        # Create 2d array of test vectors for given fraction length
        # from 3d array of test vectors
        for i in range(16):
            s = np.squeeze(o.g_dtmf_v_valid[i, N, :])
            if i == 0:
                vectors = s
            else:
                vectors = np.vstack((vectors, s))
        method = lm_spectral_method("music", 4, 4)
        byRoots = 1
        shouldDetect = 1
        shouldCheckSymbol = 1
        shouldCheckAmplitude = 0
        shouldDetectSampleStart = np.max([0, o.DTMF_START - BLOCK_LEN])
        shouldDetectSampleEnd = np.min([o.TEST_VECTOR_LEN, o.DTMF_START + N])
        success_rate = lm_dtmf.execute_on_test_vectors(
            d,
            o,
            vectors,
            method,
            BLOCK_LEN,
            byRoots,
            shouldDetect,
            shouldCheckSymbol,
            shouldCheckAmplitude,
            shouldDetectSampleStart,
            shouldDetectSampleEnd,
        )
        if success_rate < 1.0:
            raise ValueError("Test Failed")
        print(f"{success_rate=}")
