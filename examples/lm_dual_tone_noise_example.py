"""
Created on: 2024 - 09 - 06

@author: Trevor Clark

Description: Converted To python

https://github.com/dataandsignal/libmusic_m

LIBMUSIC
Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com

lm_dual_tone_noise
 
Run MUSIC on dual tone with noise.

date: August 2022

"""

# %%

import numpy as np
import matplotlib.pyplot as plt
from libmusic_py import lm_spectral_method

# %%


# Prepare input samples
Fs = 8000
t = 1 / Fs * np.arange(Fs)
f1 = 697
f2 = 1209
Amp = [3, 3]
s = Amp[0] * np.sin(2 * np.pi * f1 * t) + Amp[1] * np.sin(2 * np.pi * f2 * t)
x_start = 1
sigma = 0.0001
s = s + sigma * np.random.randn(Fs)
# Add white noise with standard deviation sigma

P = 2
# there are 2 real signal sources in stream
M = 7
# autocorrelation order
N = 24
# number of smaples to process

methods = ["pisarenko", "music", "ev", "mn"]

plt.figure(figsize=(16, 16))
j = 1
for i, m in enumerate(methods):
    print(f"method = {m}")
    # Create method
    lm = lm_spectral_method(m, M, 2 * P)

    # Process samples
    y = s[x_start:N]
    Vy, Vx, Ve, A, Ry = lm.process(y)

    # Plot PSD
    fs = np.linspace(1, 4000, 4000, endpoint=True)
    X2, d2 = lm.psd(np.linspace(1, 4000, 4000, endpoint=True), Fs)
    plt.subplot(4, 2, j)
    plt.plot(fs, X2, label=m)
    plt.xlabel("Hz")
    plt.ylabel("Pmu")
    plt.legend()
    j += 1

    # Get main frequency components by peak searching,
    # only frequencies at fs are checked
    peaks, pmu = lm.peaks(fs, Fs)

    plt.subplot(4, 2, j)
    plt.scatter(peaks, pmu)
    plt.xlabel("Hz")
    plt.ylabel("Pmu")
    j += 1

    # Get P main frequency components by eigenfilter method
    fs = lm.eigenrooting(Fs)
    print(f"Roots chosen by distance from unit circle:")
    for i in range(P):
        if i > fs.shape[0]:
            break
        z = fs[i, 1]
        f = np.real(fs[i, 2])
        print(
            f"z = {z.real:0.4f} {z.imag:0.4f}i, "
            f"|z| = {abs(z):0.4f}, "
            f"f = {f:0.4f} [Hz]"
        )

    f1_ = fs[0, 2].real
    f2_ = fs[1, 2].real

    # Get amplitude estimate by correlation method
    A = lm.dual_tone_amplitude(peaks[0], peaks[1], Fs)
    for i in range(P):
        print(f"Error: {(A[i]-Amp[i])*100/Amp[i]:0.4f} [%]")

    # Get amplitude estimate(s) by solving eigen equations
    A = lm.solve_for_amplitudes([peaks[0], peaks[1]], Fs)
    for i in range(P):
        print(f"Error: {(A[i]-Amp[i])*100/Amp[i]:0.4f} [%]")

    print()
plt.suptitle("Dual Tone Noise Example")
plt.show()
