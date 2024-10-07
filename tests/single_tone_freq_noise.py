# LIBMUSIC
# Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com
#
# single_tone_freq_noise
#
# Test single tone frequency estimate in noise.
#
# date: August 2022
# Translated to Python 2024

# %%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from libmusic_py import lm_spectral_method

# %%

Fs = 8000
t = np.arange(0, 1, 1 / Fs)
f1 = 1209
Amp = [3]

# sigma = [0.0001,0.00025,0.0005,0.001,0.0025,0.005,0.01,0.025,0.5]
sigma = [10**-8, 10**-6, 10**-4, 10**-3]
# sigma = [0]
x_start = 0

P = 1
M_START = 3
M_END = M_START + 6
N_START = 12
N_END = N_START + 120
N_RUNS = 5

x = np.linspace(M_START, M_END, M_END - M_START + 1, endpoint=True)
M_SIZE = x.size
y = np.linspace(N_START, N_END, N_END - N_START + 1, endpoint=True)
N_SIZE = y.size

X, Y = np.meshgrid(x, y, indexing="ij")

# Z_SIZE = M_SIZE * N_SIZE
z1 = np.zeros((M_SIZE, N_SIZE))  # amp by correlation test
z2 = np.zeros((M_SIZE, N_SIZE))  # amp by eigenvectors

# Create a figure
fig = plt.figure()
# Create 2x2 subplots
axs = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

for k in range(len(sigma)):
    s = Amp[0] * np.sin(2 * np.pi * f1 * t)
    # Add white noise with standard deviation sigma
    s = s + sigma[k] * np.random.randn(Fs)
    for nr in range(N_RUNS):
        for M in range(M_START, M_END + 1):
            for N in range(N_START, N_END + 1):
                if M >= N:
                    print(f"Ignored, M={M}, N={N}")
                    continue
                err = 550
                signal_error = 0
                f1_found = 0
                y = s[x_start : x_start + N]

                try:
                    method = lm_spectral_method("music", M, 2 * P)
                    _ = method.process(y)
                    fs = method.eigenrooting(Fs)
                    # Alternatively
                    # peaks, pmu = method.peaks(fs, Fs)
                except:
                    print(f"Error, M={M}, N={N}")
                    continue

                if fs.shape[0] < 1:
                    print(f"Err, could not detect, M={M}, N={N}")
                    err = -1
                else:
                    z = fs[0, 1]
                    f = abs(fs[0, 2])
                    print(f"z = {z.real} + {z.imag}j, |z| = {np.abs(z)}, f = {f} [Hz]")
                    err = abs(f - f1)

                z1[M - M_START, N - N_START] = z1[M - M_START, N - N_START] + err

    z1 = z1 / nr

    # Plot in the k-th subplot
    s = axs[k].plot_surface(X, Y, z1, alpha=0.1, cmap="viridis")

    # Set labels
    axs[k].set_xlabel("N")
    axs[k].set_ylabel("M")
    axs[k].set_zlabel("Error [Hz]")
    axs[k].grid(True)

    # Set title
    axs[k].set_title(f"sigma = {sigma[k]}")

plt.show()
