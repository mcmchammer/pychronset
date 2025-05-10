"""
Utility functions for signal processing.

This module provides functions for spectral analysis, including:
- multitaper_spectrogram_approx: an approximation of multitaper spectral analysis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram, stft
from scipy.signal.windows import dpss


def multitaper_spectrogram_approx(
    signal: np.ndarray, fs: int, window_length_ms: int = 10, step_size_ms: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximation of multitaper spectral analysis using scipy.signal.spectrogram.

    NOTE: This is a simplified approximation. A full multitaper implementation with
    Slepian tapers for variance reduction would require specialized libraries
    (e.g., nitime, MNE-Python) or a complex custom implementation.
    This uses a Hann window for simplicity.
    This function is kept for reference or comparison.
    """
    nperseg = int(fs * window_length_ms / 1000)  # samples per segment
    noverlap = nperseg - int(fs * step_size_ms / 1000)  # overlap samples

    # Ensure nperseg is at least 1 and noverlap is not negative
    if nperseg < 1:
        nperseg = 1
        noverlap = 0
    noverlap = max(noverlap, 0)

    f, t, Sxx = spectrogram(  # noqa: N806
        signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, mode="psd"
    )

    return f, t, Sxx


def exact_multitaper_spectrogram(
    signal: np.ndarray,
    fs: int,
    window_length_ms: int = 10,
    step_size_ms: int = 1,
    num_tapers: int = 8,
    time_bandwidth_product: float = 5.0,  # NW for DPSS, paper implies NW=5 for W=0.5kHz, L=10ms
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute multitaper spectrogram using Discrete Prolate Spheroidal Sequences (DPSS).

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    fs : int
        Sampling frequency.
    window_length_ms : int, optional
        Length of each segment in milliseconds. Defaults to 10.
    step_size_ms : int, optional
        Step size between segments in milliseconds. Defaults to 1.
    num_tapers : int, optional
        Number of Slepian tapers to use. Defaults to 8, as per Roux et al. (2016).
    time_bandwidth_product : float, optional
        Time-bandwidth product (NW) for DPSS tapers. Defaults to 5.0.
        Roux et al. (2016) state k=8 tapers and a half-bandwidth of ±0.5 kHz for a 10ms window.
        W = 0.5 kHz, L = 10 ms = 0.01 s. NW = L * W = 0.01 * 500 = 5.
        Typically, K_max_tapers = 2*NW - 1. For NW=5, K_max=9. Using num_tapers=8 means we take the first 8.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        f: Array of sample frequencies.
        t: Array of segment times.
        Sxx_mean: Mean Power Spectral Density across tapers.
    """
    nperseg = int(fs * window_length_ms / 1000)  # samples per segment
    noverlap = nperseg - int(fs * step_size_ms / 1000)  # overlap samples

    if nperseg < 1:
        # This case should ideally not happen with typical parameters
        # but handle defensively.
        # If nperseg is too small, DPSS might fail or be meaningless.
        # Fallback to a simple STFT with Hann if too small for DPSS.
        # Or, raise an error. For now, let's assume valid nperseg for DPSS.
        nperseg = max(1, nperseg)  # Ensure positive
        noverlap = 0  # No overlap if segment is tiny

    noverlap = max(0, min(noverlap, nperseg - 1))  # Ensure valid overlap

    # Generate DPSS tapers
    # Kmax in dpss is the number of tapers to return.
    tapers = dpss(M=nperseg, NW=time_bandwidth_product, Kmax=num_tapers, sym=False)

    all_sxx_taper = []
    # Initialize f_axis and t_axis as empty ndarrays with float dtype to satisfy type hints
    f_axis: np.ndarray = np.array([], dtype=float)
    t_axis: np.ndarray = np.array([], dtype=float)
    axes_initialized = False

    for taper_idx in range(tapers.shape[0]):
        current_taper = tapers[taper_idx, :]
        # Use scaling='psd' for STFT to get Zxx such that |Zxx|^2 is PSD
        f_raw, t_raw, Zxx_taper = stft(
            signal,
            fs=fs,
            window=current_taper,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="psd",
        )
        if not axes_initialized:  # Store frequency and time axes from the first STFT
            f_axis = np.asarray(f_raw, dtype=float)
            t_axis = np.asarray(t_raw, dtype=float)
            axes_initialized = True

        Sxx_one_taper = np.abs(Zxx_taper) ** 2
        all_sxx_taper.append(Sxx_one_taper)

    if not all_sxx_taper:
        # Should not happen if num_tapers > 0 and signal is valid
        # Return empty arrays with float dtype or handle error appropriately
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Average the PSDs from all tapers
    Sxx_mean = np.mean(np.array(all_sxx_taper), axis=0)

    return f_axis, t_axis, Sxx_mean


def apply_gaussian_smoothing(
    feature_series: np.ndarray, smoothing_window_ms: int = 10
) -> np.ndarray:
    """Apply Gaussian smoothing to a feature series."""
    # Assuming feature_series corresponds to 1ms step size from spectrogram
    # Sigma is calculated based on the window length (in samples).
    # A common heuristic for sigma is window_length / 4 or
    # window_length / (2 * sqrt(2 * log(2))) for FWHM.
    # For a 10ms window with 1ms steps, this means 10 samples.
    sigma = smoothing_window_ms / 4.0  # Example: 10ms window -> sigma = 2.5 samples

    if len(feature_series) < sigma * 3:  # Heuristic: need enough points for Gaussian tail
        # print(f"Warning: Feature series too short ({len(feature_series)} samples) for
        # {smoothing_window_ms}ms Gaussian smoothing. Skipping.")
        return feature_series

    smoothed_feature = gaussian_filter1d(feature_series, sigma=sigma)
    return smoothed_feature


def normalize_feature(feature_series: np.ndarray) -> np.ndarray:
    """Normalize a feature series to range from 0 to 1."""
    min_val = np.min(feature_series)
    max_val = np.max(feature_series)
    if max_val == min_val:  # Avoid division by zero if all values are the same
        return np.zeros_like(feature_series)
    return (feature_series - min_val) / (max_val - min_val)
