"""
Created on: 2023 - 11 - 11

@author: Trevor Clark

Description: libmusic_m reimplementation in python

https://github.com/dataandsignal/libmusic_m
"""

# %%

import numpy as np
import spectrum as spec
import matplotlib.pyplot as plt

# %%


class lm_spectral_method:
    """LIBMUSIC

    Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com

    Translated from matlab to python by Trevor Clark

    lm_spectral_method

    Spectral method implementation.

    This class provides support for MUSIC, Pisarenko, EV, Minimum Norm spectral
    methods which working principle is to decompose a noisy signal space Vy
    (spanned by eigenvalues of autocorrelation matrix Ry) into pure signal
    subspace Vx (spanned by selected P2 eigenvectors of Ry, when eigenvalues are
    sorted in decreasing order) and noise subspace Ve. This class provides then
    means to compute PSD and amplitude estimates. It has also support for
    finding maxima in PSD by a peak finding and by eigenfilter method
    (eigenrooting).

    Basic usage

    Create a spectral method ('music', 'pisarenko', 'ev', 'mn' for Minimum
    Norm method)
      method = lm_spectral_method('music', M, 2*P);

    Process input samples and optionally capture the intermediate results
    (all eigenvectors, signal eigenvectors, noise eigenvectors, eigenvalues,
    autocorrelation matrix)
      Vy,Vx,Ve,A,Ry = method.process(y);

    Compute full PSD for frequencies 1 - 4000 Hz (sampling rate is 8 kHz)
      X2,d2 = method.psd(method, np.linspace(1, 4000, 4000, endpoint=True), 8000);

    Get single tone amplitude
      A = method.single_tone_amplitude();

    Get 2 amplitudes, for dual tone with freqs f1, f2 and sampling rate Fs
      A = method.dual_tone_amplitude(f1, f2, Fs);

    Get all amplitudes (for each sinusoid given in fs)
      A = method.solve_for_amplitudes(fs, Fs);

    Get detected frequencies by peak searching (considering only these
    frequencies that are passed in fs). peakWidth is a width of a peak, use 0
    for default
      peaks, pmu = method.peaks(fs, Fs, peakWidth);

    Get detected frequencies by rooting noise eigenvectors
      fs = method.eigenrooting(Fs, plot, print);

    For detailed usage see examples and tests folders.

    date: September 2024
    """

    def __init__(self, kind, M, P2, verbose=False):
        self.verbose = verbose
        self.kind = None  # (in) method ('pisarenko', 'music', 'ev', 'mn')
        self.psd = None  # (out) pseudospectrum
        self.subspace_decomposer = None  # handle to core implementation
        self.M = None  # autocorrelation order (also max signal space dim)
        self.P2 = None  # pure signal space dim (must be < M+1)
        self.Vy = None  # (out) eigenvectors
        self.Vx = None  # (out) pure signal subspace eigenvectors
        self.Ve = None  # (out) noise subspace eigenvectors
        self.A = None  # (out) eigenvalues
        self.Ry = None  # (out) autocorrelation matrix
        self.y = None  # (in) samples
        self.peak_width = None  # when computing another peak, frequencies within
        # half this range from previous peaks are ignored
        self.signal_space_dim = None  # computed pure signal subspace dim
        self.noise_space_dim = None  # computed noise subspace dim

        self.kind = kind
        if self.kind == "pisarenko":
            self.psd = self.psd_pisarenko
        elif self.kind == "music":
            self.psd = self.psd_music
        elif self.kind == "ev":
            self.psd = self.psd_ev
        elif self.kind == "mn":
            self.psd = self.psd_minimum_norm
        else:
            self.kind = "music"
            self.psd = self.psd_music
        self.M = M
        self.P2 = P2
        self.peak_width = 10  # Default peak width in Hz
        self.signal_space_dim = 0
        self.noise_space_dim = 0

    def process(self, y):
        """Decompose noisy signal space into signal and noise subspaces

        After this returns method's signal_space_dim and noise_space_dim are set
        to result space dimensions. For Pisarenko noise space dim is truncated
        to 1.

        Parameters
        ----------
        y
            signal

        Returns
        -------
            Ry - correlation matrix
            Vy = [Vx Ve] - eigenvectors
            A - eigenvalues (sorted in descending order)
        """
        Vy, Vx, Ve, A, Ry = self.decompose(y)
        self.Vy = Vy
        self.Vx = Vx
        self.Ve = Ve
        self.A = A
        self.Ry = Ry
        self.y = y
        return (Vy, Vx, Ve, A, Ry)

    def psd_pisarenko(self, f, Fs):
        return self.psd_music(f, Fs)

    def psd_music(self, f, Fs):
        N = f.size
        X2 = np.zeros(N)
        d2 = np.zeros(N)
        pmu = np.zeros(N)

        n = self.A.shape[0]
        if n <= self.P2:
            raise ValueError(
                f"Number of eigenvectors {n}"
                f" too short for signal space dimension {self.P2}"
            )

        noise_space_dim = n - self.P2
        if self.kind == "pisarenko":
            noise_space_dim = 1

        e = np.squeeze(self.Ve[:, 0:noise_space_dim])

        if self.kind == "pisarenko":
            w = np.ones(1)
        elif self.kind == "music":
            x = np.ones(noise_space_dim)
            w = np.diag(x)
        elif self.kind == "ev":
            x = self.A  # A is already diagonalized
            x = x[self.P2 : self.P2 + noise_space_dim]
            x = 1 / x
            w = np.diag(x)
        elif self.kind == "mn":
            u1 = np.zeros(self.M + 1)
            u1[0] = 1
            a_nom = e @ e.T @ u1.T
            a_denom = u1 @ e @ e.T @ u1.T
            a = a_nom / a_denom
        else:
            raise ValueError(f"Unknown Type: {self.kind}")

        for i in range(N):
            #  Compute the distance from steering vector to signal supspace,
            #  i.e. projection of steering vector onto noise subspace
            t = np.arange(self.M + 1) * 1 / Fs
            # steering_vector = sin(2*pi*f(i)*t);
            steering_vector = np.exp(1j * 2 * np.pi * f[i] * t)
            if self.kind == "mn":
                d2[i] = np.abs(steering_vector @ (a * (a @ steering_vector)))
            else:
                temp = e.T @ steering_vector
                if w.size == 1:
                    temp *= w
                else:
                    temp = w @ temp
                if temp.size == 1:
                    temp = e * temp
                else:
                    temp = e @ temp
                temp = steering_vector @ temp
                d2[i] = np.abs(temp)
            pmu[i] = 1 / d2[i]
            X2[i] = pmu[i]
        return X2, d2

    def psd_ev(self, f, Fs):
        return self.psd_music(f, Fs)

    def psd_minimum_norm(self, f, Fs):
        return self.psd_music(f, Fs)

    def amplitude(self):
        A = 0
        if self.P2 == 2:
            A = self.single_component_amplitude()
        # if self.P2 == 4:
        #     A = self.double_components_amplitude(f1, f2, Fs)
        else:
            print("Oops, not implemented")
        return A

    def single_tone_amplitude(self):
        """Get single tone amplitude

        Returns
        -------
            A
        """
        corr_av = sum(np.diag(self.Ry)) / (self.M + 1)
        eigenvals = self.A  # A is already diagonalized
        noise_var_est = np.sum(eigenvals[self.P2 :]) / ((self.M + 1) - self.P2)
        A = np.sqrt(2 * (corr_av - noise_var_est))
        return A

    def dual_tone_amplitude(self, f1, f2, Fs):
        """Get 2 amplitudes, for dual tone with freqs f1, f2 and sampling rate Fs

        Parameters
        ----------
        f1
            freq1
        f2
            fre12
        Fs
            Sample rate

        Returns
        -------
            [A1, A2]]
        """
        r0av = np.sum(np.diag(self.Ry)) / (self.M + 1)
        r1av = np.sum(np.diag(self.Ry, 1)) / (self.M + 1 - 1)
        eigenvals = self.A  # A is already diagonalized
        noise_var_est = np.sum(eigenvals[self.P2 :]) / ((self.M + 1) - self.P2)
        c1 = np.cos(2 * np.pi * f1 * 1 / Fs)
        c2 = np.cos(2 * np.pi * f2 * 1 / Fs)
        P1 = (r1av - r0av * c2) / (c1 - c2)
        P2 = r0av - P1

        A1 = np.sqrt(2 * P1)
        A2 = np.sqrt(2 * P2)
        return [A1, A2]

    def solve_for_amplitudes(self, fs, Fs):
        """Find all amplitudes (for real sinusoids, this is P2/2 amplitudes).

        This methods finds amplitude estimates from pure signal space
        eigenvectors and steering vectors, for frequencies given in fs.

        Parameters
        ----------
        fs
            frequencies to evaluate
        Fs
            sampling rate

        Returns
        -------
            A
        """
        fs = np.array(fs)
        if fs.shape == ():
            N = 1
            f = np.array([fs, -fs])
        else:
            N = len(np.array(fs))
            f = np.concatenate((fs, -fs))
        eigenvals = self.A  # A is already diagonalized
        noise_var_est = np.sum(eigenvals[self.P2 :]) / ((self.M + 1) - self.P2)
        t = np.arange(1, self.M + 2) * 1 / Fs
        M = np.zeros((self.P2, self.P2), dtype=complex)
        for i in range(self.P2):
            steering_vector = np.exp(1j * 2 * np.pi * f[i] * t)
            for j in range(self.P2):
                v = self.Vx[:, j]
                v = v / (np.linalg.norm(v))
                M[j, i] = np.abs(steering_vector @ v) ** 2
        lambdas = eigenvals[: self.P2]
        B = lambdas - noise_var_est
        P = np.abs(np.linalg.pinv(M) @ B)
        A = 2 * np.sqrt(P[:N])
        return A

    def peak(self, f, Fs, f2ignore, bw2ignore):
        """Search for another peak.

        Parameters
        ----------
        f
            frequencies to evaluate
        Fs
            sampling rate
        f2ignore
            Frequencies of previous peaks that should be ignored in this search
        bw2ignore
            Frequency range that should be ignored around each of previous
            peaks, i.e. for each of previous peaks P ignored range is
            (P-bw2ignore, P+bw2ignore)

        Returns
        -------
            _description_
        """
        N = len(f)
        X2, d2 = self.psd(f, Fs)
        max_pmu = 0
        max_pmu_freq = 1

        for u in range(N):
            ignore = 0
            peaks_n = len(f2ignore)
            if peaks_n > 0:
                for k in range(peaks_n):
                    if np.abs(f2ignore[k] - f[u]) < bw2ignore:
                        ignore = 1
                        break
                if ignore:
                    continue
            pmu_u = X2[u]
            if max_pmu < pmu_u:
                max_pmu = np.roll(max_pmu, 1)
                max_pmu_freq = np.roll(max_pmu_freq, 1)
                max_pmu = pmu_u
                max_pmu_freq = f[u]
        peak = max_pmu_freq
        peak_pmu = max_pmu
        return peak, peak_pmu

    def peaks(self, freqs, Fs, peakWidth=0):
        """search for peaks

        Parameters
        ----------
        freqs
            frequencies to evaluate
        Fs
            sampling rate
        peakWidth
            Frequency range that should be ignored around each of previous
            peaks, i.e. for each of previous peaks P ignored range is
            (P-peakWidth/2, P+peakWidth/2), peakWidth=0 is default

        Returns
        -------
            peaks, peaks_pmu
        """
        peaks_n = int(self.P2 / 2)
        if peakWidth > 0:
            self.peak_width = peakWidth
        peaks = []
        peaks_pmu = []
        for i in range(peaks_n):
            p, pmu = self.peak(freqs, Fs, peaks, self.peak_width / 2)
            peaks.append(p)
            peaks_pmu.append(pmu)

        peaks = np.array(peaks)
        peaks_pmu = np.array(peaks_pmu)

        return peaks, peaks_pmu

    def plot_roots(self, p):
        if 1 > len(p):
            raise ValueError("Err: empty vector")
        r = np.roots(p)
        plt.figure()
        plt.plot(r.real, r.imag, ".", markersize=50, linewidth=10, label="Roots")
        plt.axis("equal")
        plt.grid(True)
        t = np.linspace(0, 2 * np.pi, 100, endpoint=True)
        plt.plot(np.cos(t), np.sin(t), "r-", label="Unit circle")
        plt.legend()
        return r

    def eigenrooting(self, Fs, show_plots=False):
        """Estimate freq by rooting noise eigenvectors

        Parameters
        ----------
        Fs
            sampling rate
        show_plots
            show eigen vector Roots

        Returns
        -------
            fs_
        """
        N = self.Ve.shape[1]

        for i in range(N):
            e = self.Ve[:, i]
            if self.verbose:
                print("Eigenvector:")
                print(f"{e=}")
            if show_plots:
                rts = self.plot_roots(e)
            else:
                rts = np.roots(e)

            sz = rts.shape[0]
            z = np.zeros(sz)
            fs = np.vstack((z, rts, z)).T
            for j in range(sz):
                r = fs[j, 1]
                fs[j, 0] = np.abs(1 - np.abs(r))
                fs[j, 2] = (np.angle(r) / np.pi) * Fs / 2
            if self.verbose:
                print("Roots (distance, z, f):")
                print(f"{fs[:,0]=}")
                print(f"{fs[:,1]=}")
                print(f"{fs[:,2]=}")
            if i == 0:
                fs_ = fs.copy()
            else:
                fs_ = np.vstack((fs_, fs))
        fs_ = fs_[fs_[:, 0].argsort()]
        if self.verbose:
            print("Wszystkie pierwiastki:")
            print(f"{fs_=}")
        indices = np.argwhere(fs_[:, 2] < 0)
        fs_ = np.delete(fs_, indices, axis=0)
        return fs_

    def decompose(self, y):
        """
        Decompose noisy signal space Vy into pure signal space Vx and noise
        signal space Ve.

        Parameters
        ----------
        y
            input signal

        Returns
        -------
            Vy, Vx, Ve, A, Ry
        """
        if len(y) <= self.M:
            raise ValueError("Err, N must be > M")
        X = spec.corrmtx(y, self.M, "covariance")

        # make it like matlab
        X = X / np.sqrt(y.size - self.M)  # normalize
        Ry = X.conj().T @ X  # compute R
        Ry = (Ry + Ry.conj().T) / 2  # hermitian symmetric

        _, A, Vy = np.linalg.svd(Ry, hermitian=True)

        Vy = Vy.conj().T  # make it same as matlab

        n = Vy.shape[1]
        if n <= self.P2:
            raise ValueError(
                f"Computed noisy signal space dimension {n} too short "
                f"for required signal space dimension P2 ({self.P2}). "
                f"Consider decreasing signal space dimension or increasing "
                f"autocorrelation order and/or samples count"
            )
        Vx = Vy[:, : self.P2]
        Ve = Vy[:, self.P2 :]
        return Vy, Vx, Ve, A, Ry


class lm_dtmf:
    """
    LIBMUSIC
    Copyright (C) 2022, Piotr Gregor piotr@dataandsignal.com

    lm_dtmf

    DTMF detection.

    date: August 2022
    Translated to python 2024
    """

    def __init__(self):
        self.dtmf_low_freqs = None
        self.dtmf_high_freqs = None

    def lm_dtmf(self):
        self.dtmf_low_freqs = np.array([697, 770, 852, 941], dtype=np.float64)
        self.dtmf_high_freqs = np.array([1209, 1336, 1477, 1633], dtype=np.float64)
        return self.dtmf_low_freqs, self.dtmf_high_freqs

    def check_by_roots(self, fs, verbose=False):
        decision = 0
        f1 = 0
        f2 = 0
        N = fs.shape[0]
        if N < 2:
            raise ValueError("Did not find roots")
        else:
            f1_found = 0
            f2_found = 0

            # Search low group freq
            for i in range(N):

                z = fs[i, 1]
                z_distance = fs[i, 0]
                z_f = np.abs(fs[i, 2])

                for j in range(4):

                    f = self.dtmf_low_freqs[j]

                    if verbose:
                        print(
                            f"LOW {f} Hz:  z = {z}, |z| = {np.abs(z)}, f = {z_f} [Hz]"
                        )
                    if abs(f - z_f) < 2:
                        f1_found = f
                        if verbose:
                            print(f"SELECTED z = {z} , |z| = {np.abs(z)}, f = {f} [Hz]")
                        break
                if f1_found:
                    break

            if f1_found:
                # Search high group freq
                for i in range(N):

                    z = fs(i, 2)
                    z_distance = fs(i, 1)
                    z_f = abs(fs(i, 3))

                    for j in range(4):

                        f = self.dtmf_high_freqs[j]

                        if verbose:
                            print(
                                f"HIGH {f} Hz: z = {z}, |z| = {np.abs(z)}, f = {z_f} [Hz]"
                            )
                        if np.abs(f - z_f) < 2:
                            f2_found = f
                            if verbose:
                                print(
                                    f"SELECTED z = {z}, |z| = {np.abs(z)}, f = {f} [Hz]"
                                )
                            break
                    if f2_found:
                        break
        if f1_found and f2_found:
            decision = 1
            f1 = f1_found
            f2 = f2_found
        return decision, f1, f2

    def check_by_peaks(self, peak1, peak2, verbose=False):
        decision = 0
        f1 = 0
        f2 = 0

        f1_found = 0
        f2_found = 0

        # Search low group freq
        for j in range(4):
            f = self.dtmf_low_freqs[j]
            if abs(f - peak1) <= 2:
                f1_found = f
                if verbose:
                    print(f"SELECTED f1 by peak1 f = {f} [Hz]")
            if abs(f - peak2) <= 2:
                f1_found = f
                if verbose:
                    print(f"SELECTED f1 by peak2 f = {f} [Hz]")

        if f1_found:
            # Search high group freq
            for j in range(4):
                f = self.dtmf_high_freqs[j]
                if np.abs(f - peak1) <= 2:
                    f2_found = f
                    if verbose:
                        print(f"SELECTED f2 by peak1 f = {f} [Hz]")
                    break
                if np.abs(f - peak2) <= 2:
                    f2_found = f
                    if verbose:
                        print(f"SELECTED f2 by peak2 f = {f} [Hz]")
                    break
        if f1_found and f2_found:
            decision = 1
            f1 = f1_found
            f2 = f2_found
        return decision, f1, f2

    def execute_on_test_vector(
        self, lm, vector, method, detectBlockLen, byRoots, checkAmplitude, verbose=False
    ):
        decision = 0
        detectSample = 0
        detectSymbol = 1500
        sample_n = vector.size
        sample_end = sample_n - detectBlockLen + 1
        if sample_end < 1:
            raise ValueError("Test vector too short")
        for i in range(sample_end):
            try:
                if verbose:
                    print(f"[%d/%d]->", detectBlockLen, i)
                    if i == 40:
                        print(f"o")
                f1 = 0
                f2 = 0
                y = vector[i : i + detectBlockLen]
                # Energy check
                ENERGY_THRESHOLD = (
                    0.5 * (lm.g_dtmf_req_etsi_f2_amp_min_v ^ 2) * detectBlockLen
                )
                energy = np.sum(y**2)
                if energy < ENERGY_THRESHOLD:
                    continue
                _ = method.process(vector[i : i + detectBlockLen])

                if byRoots:
                    fs = method.eigenrooting(lm.g_Fs, 0, 0)
                    decision, f1, f2 = self.check_by_roots(fs, 0)
                    if verbose:
                        print(
                            f"[%d/%d]By ROOTS, decision=%d, f1=%d, f2=%d",
                            detectBlockLen,
                            i,
                            decision,
                            f1,
                            f2,
                        )
                else:
                    peaks, peaks_pmu = method.peaks(np.arange(4000 + 1), lm.g_Fs, 0)
                    decision, f1, f2 = self.check_by_peaks(peaks[0], peaks[1], verbose)
                if decision:
                    if checkAmplitude:
                        A = method.dual_tone_amplitude(f1, f2, lm.g_Fs)
                        if A.shape[1] < 2:
                            continue
                        if (
                            A[0] < lm.g_dtmf_req_etsi_f1_amp_min_v
                            or A[0] > lm.g_dtmf_req_etsi_f1_amp_max_v
                        ):
                            continue
                        if (
                            A[1] < lm.g_dtmf_req_etsi_f1_amp_min_v
                            or A[1] > lm.g_dtmf_req_etsi_f1_amp_max_v
                        ):
                            continue
                    if verbose:
                        print(f"DETECTED")
                    detectSample = i
                    detectSymbol = lm.dtmf_etsi_f1_f2_2_symbol(f1, f2)
                    return
            except:
                if verbose:
                    print(f"Detection failed")
                continue
        return decision, detectSample, detectSymbol

    def execute_on_test_vectors(
        self,
        lm,
        vectors,
        method,
        detectBlockLen,
        byRoots,
        shouldDetect,
        shouldCheckSymbol,
        shouldCheckAmplitude,
        shouldDetectSampleStart,
        shouldDetectSampleEnd,
        verbose=False,
    ):
        tests_n = vectors.shape[0]
        score = 0
        for i in range(tests_n):
            if verbose:
                print(f"=== [Exec vector {i}]")
            try:
                decision, detectSample, detectSymbol = self.execute_on_test_vector(
                    lm,
                    vectors[i, :],
                    method,
                    detectBlockLen,
                    byRoots,
                    shouldCheckAmplitude,
                    verbose,
                )
                if decision:
                    if shouldDetect:
                        # check symbol index
                        if shouldCheckSymbol:
                            trueSymbol = lm.dtmf_etsi_idx_2_symbol(i)
                            if trueSymbol != detectSymbol:
                                if verbose:
                                    print(f"Wrong symbol detected")
                                continue
                            if verbose:
                                print(f"Right symbol detected [%c]", trueSymbol)

                        # check sample number is valid
                        if (detectSample < shouldDetectSampleStart) or (
                            detectSample > shouldDetectSampleEnd
                        ):
                            if verbose:
                                print(f"False detection")
                        else:
                            score = score + 1
                    else:
                        continue
                if not decision:
                    if not shouldDetect:
                        score = score + 1
                    else:
                        continue
            except:
                continue
        success_rate = score / tests_n
        return success_rate
