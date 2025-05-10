"""
Utility functions for signal processing.

This module provides functions for spectral analysis, including:
- multitaper_spectrogram_approx: an approximation of multitaper spectral analysis.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram


def multitaper_spectrogram_approx(
    signal: np.ndarray, fs: int, window_length_ms: int = 10, step_size_ms: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximation of multitaper spectral analysis using scipy.signal.spectrogram.

    NOTE: This is a simplified approximation. A full multitaper implementation with
    Slepian tapers for variance reduction would require specialized libraries
    (e.g., nitime, MNE-Python) or a complex custom implementation.
    This uses a Hann window for simplicity.
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
