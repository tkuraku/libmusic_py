"""
Created on: 2023 - 11 - 11

@author: Trevor Clark

Description: Lib Music Example

https://github.com/dataandsignal/libmusic_m

LIBMUSIC
Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com

lm_single_tone_noise

Run MUSIC on single tone with noise.

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
f1 = 1209
Amp = 3

s = Amp * np.sin(2 * np.pi * f1 * t)
x_start = 0
sigma = 0.1

# Add white noise with standard deviation sigma
s = s + sigma * np.random.randn(Fs)

P = 1  # there is single real signal source in stream
M = 5  # autocorrelation order
N = 24  # number of samples to process

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
    plt.scatter(peaks, pmu, label=m)
    plt.xlabel("Hz")
    plt.ylabel("Pmu")
    plt.legend()
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

    # Get amplitude estimate by correlation method
    A = lm.single_tone_amplitude()
    print(f"Amplitude Error: {(A-Amp)*100/Amp:0.4f} [%]")

    # Get amplitude estimate(s) by solving eigen equations
    A = lm.solve_for_amplitudes(f, Fs)
    print(f"Amplitude Error: {(A[0]-Amp)*100/Amp:0.4f} [%]")

    print()

plt.suptitle("Single Tone Noise Example")
plt.show()
