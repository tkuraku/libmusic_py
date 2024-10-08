# LIBMUSIC
# Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com
#
# dual_tone_noise
#
# Test dual tone frequency estimates in noise.
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
f1 = 697
f2 = 1209
Amp = [3, 3]

sigma = [10**-8, 10**-6, 10**-4, 10**-3]
x_start = 0

P = 2
M_START = 4
M_END = M_START + 6
N_START = 8
N_END = N_START + 24
N_RUNS = 1

x = np.linspace(M_START, M_END, M_END - M_START + 1, endpoint=True)
M_SIZE = x.size
y = np.linspace(N_START, N_END, N_END - N_START + 1, endpoint=True)
N_SIZE = y.size

X, Y = np.meshgrid(x, y, indexing="ij")

z1 = np.zeros((M_SIZE, N_SIZE))

# Create a figure
fig = plt.figure()
# Create 2x2 subplots
axs = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

for k in range(len(sigma)):
    s = Amp[0] * np.sin(2 * np.pi * f1 * t) + Amp[1] * np.sin(2 * np.pi * f2 * t)
    # Add white noise with standard deviation sigma
    s = s + sigma[k] * np.random.randn(Fs)
    for nr in range(N_RUNS):
        for M in range(M_START, M_END + 1):
            for N in range(N_START, N_END + 1):
                if M >= N - 4:
                    print(f"Ignored, M={M}, N={N}")
                    continue
                err = 0
                signal_error = 0
                f1_found = 0
                f2_found = 0
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

                if fs.shape[0] < 2:
                    print(f"Err, M={M}, N={N}")
                    err = -1
                else:
                    f1_found = 0
                    f2_found = 0
                    for i in range(fs.shape[0]):
                        z = fs[i, 1]
                        f = abs(fs[i, 2])
                        print(
                            f"z = {z.real} + {z.imag}j, |z| = {np.abs(z)}, f = {f} [Hz]"
                        )
                        if abs(f - f1) < 0.5:
                            f1_found = f
                        else:
                            if abs(f - f2) < 0.5:
                                f2_found = f
                        if f1_found and f2_found:
                            break

                    if f1_found == 0 or f2_found == 0:
                        print(
                            f"?? Did not detect freqs, ({f1_found}/{f2_found}), M={M}, N={N}"
                        )
                        signal_error = 1
                        continue

                err1 = abs(f1 - f1_found)
                err2 = abs(f2 - f2_found)
                z1[M - M_START, N - N_START] += np.max([err1, err2])

    z1 = z1 / N_RUNS

    # Plot in the k-th subplot
    s = axs[k].plot_surface(X, Y, z1, alpha=0.5, cmap="viridis")

    # Set labels
    axs[k].set_xlabel("N")
    axs[k].set_ylabel("M")
    axs[k].set_zlabel("Error [Hz]")
    axs[k].grid(True)

    # Set title
    axs[k].set_title(f"sigma = {sigma[k]}")

plt.show()
