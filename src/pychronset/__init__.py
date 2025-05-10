"""pychronset."""

import logging

import numpy as np
from scipy.stats import gmean

from pychronset.utils.io import extract_thresholds_from_mat, load_audio
from pychronset.utils.signal import (
    apply_gaussian_smoothing,
    multitaper_spectrogram_approx,
    normalize_feature,
)

logger = logging.getLogger(__name__)
MIN_FEATURES_CRITERION = (
    4  # Minimum number of features to meet criterion for speech onset detection
)


def extract_features(fs: int, f: np.ndarray, Sxx: np.ndarray) -> dict:  # noqa: N803
    """
    Extract the six acoustic features from the spectrogram.

    Sxx is assumed to be power spectral density (PSD) or power.
    """
    features = {}

    # Define frequency range for Amplitude integration (0.15 kHz to 22.05 kHz)
    # Find indices closest to these frequencies
    freq_min_idx = np.argmin(np.abs(f - 150))
    freq_max_idx = np.argmin(np.abs(f - 22050))

    # Ensure valid frequency range
    if freq_min_idx > freq_max_idx or freq_min_idx >= len(f) or freq_max_idx >= len(f):
        # Fallback to full range or adjust if specified range is out of bounds
        logger.warning(
            "Specified frequency range for Amplitude is outside spectrogram range. Adjusting."
        )
        freq_min_idx = np.int_(0)
        freq_max_idx = np.int_(len(f) - 1)

    # a. Amplitude (Amp)
    # "logarithm of spectral power integrated over frequencies between 0.15 and 22.05 kHz"
    amplitude = np.log(
        np.sum(Sxx[freq_min_idx : freq_max_idx + 1, :], axis=0) + 1e-9
    )  # Add epsilon to avoid log(0)
    features["Amplitude"] = amplitude

    # b. Wiener Entropy (WE)
    # "ratio of the geometric mean and the arithmetic mean of the spectrum"
    wiener_entropy = []
    for i in range(Sxx.shape[1]):
        spectrum = Sxx[:, i]
        spectrum_positive = spectrum + 1e-9  # Ensure non-negative for gmean
        if np.all(
            spectrum_positive == spectrum_positive[0]
        ):  # Handle flat spectrum (e.g., silence)
            we = 0.0
        else:
            we = gmean(spectrum_positive) / np.mean(spectrum_positive)
        wiener_entropy.append(we)
    features["WienerEntropy"] = np.array(wiener_entropy)

    # c. Spectral Change (SC)
    # "combined measure of how power changes simultaneously in both time and frequency"
    # Approximated as sum of absolute differences across frequency bins within each time frame
    spectral_change = np.sum(np.abs(np.diff(Sxx, axis=0)), axis=0)
    features["SpectralChange"] = spectral_change

    # d. Amplitude Modulation (AM)
    # "overall change in power across all frequencies... reflects the magnitude of the change in
    # energy of a speech sound over time"
    # Calculated as the absolute time derivative of the Amplitude feature
    amplitude_diff = np.abs(np.diff(amplitude, prepend=amplitude[0], append=amplitude[-1]))
    features["AmplitudeModulation"] = amplitude_diff

    # e. Frequency Modulation (FM)
    # "assesses how much the concentration of power changes across spectral bands over time"
    # Approximated using time derivative of spectral centroid
    spectral_centroids = np.array(
        [np.sum(f * Sxx[:, i]) / (np.sum(Sxx[:, i]) + 1e-9) for i in range(Sxx.shape[1])]
    )
    fm_raw = np.abs(
        np.diff(spectral_centroids, prepend=spectral_centroids[0], append=spectral_centroids[-1])
    )
    features["FrequencyModulation"] = fm_raw

    # f. Harmonic Pitch (HP)
    # "computed a second spectrum of the power spectrum, which measures the periodicity of the
    # peaks in the power spectrum"
    # This refers to the power cepstrum.
    harmonic_pitch = []
    for i in range(Sxx.shape[1]):
        spectrum_frame = Sxx[:, i]

        if len(spectrum_frame) > 1:
            log_spectrum = np.log(spectrum_frame + 1e-9)

            # Pad to next power of 2 for IFFT for better resolution
            n_fft_ceps = int(2 ** np.ceil(np.log2(len(log_spectrum))))

            # Compute inverse real FFT (cepstrum)
            cepstrum_temp = np.fft.irfft(log_spectrum, n=n_fft_ceps)

            # Define quefrency axis in seconds.
            # df is the frequency resolution of the original spectrum.
            df_val = (
                f[1] - f[0] if len(f) > 1 else (fs / 2) / len(spectrum_frame)
            )  # Fallback if f is too short
            q_axis_sec = np.fft.rfftfreq(
                n_fft_ceps, d=df_val
            )  # Quefrency axis in "seconds" (period)

            # Define quefrency range for typical human pitch
            # (e.g., 50 Hz to 400 Hz -> 2.5 ms to 20 ms)
            q_min_idx = np.argmin(np.abs(q_axis_sec - 0.0025))  # 2.5 ms
            q_max_idx = np.argmin(np.abs(q_axis_sec - 0.020))  # 20 ms

            # Ensure indices are in correct order and within bounds
            if q_min_idx > q_max_idx:
                q_min_idx, q_max_idx = q_max_idx, q_min_idx

            # Ensure there's a valid range to search in the cepstrum
            if (
                q_max_idx <= q_min_idx
                or q_max_idx >= len(cepstrum_temp)
                or q_min_idx >= len(cepstrum_temp)
            ):
                hp_val = 0.0  # No valid peak found in range
            else:
                # Find the maximum value (peak) in the relevant quefrency range
                hp_val = np.max(np.abs(cepstrum_temp[q_min_idx:q_max_idx]))
            harmonic_pitch.append(hp_val)
        else:
            harmonic_pitch.append(0.0)  # Not enough data for cepstrum calculation

    features["HarmonicPitch"] = np.array(harmonic_pitch)

    # Apply normalization and smoothing to all features
    for feature_name, data in features.items():
        smoothed_data = apply_gaussian_smoothing(data, fs)
        features[feature_name] = normalize_feature(smoothed_data)

    return features


def detect_speech_onset(  # noqa: C901, PLR0912
    features: dict,
    optimized_thresholds_dict: dict,
    feature_comparison_types: dict,
    duration_threshold_ms: int = 35,
) -> float | None:
    """
    Detect speech onset based on multiple feature thresholds and duration.

    Use specific comparison types for each feature.

    Parameters
    ----------
    features : dict
        Dictionary of normalized and smoothed feature time series.
    fs : int
        Sampling rate (used to convert duration_threshold_ms to samples,
        assuming 1ms per time step from spectrogram).
    optimized_thresholds_dict : dict
        Dictionary of optimized thresholds for each feature.
    feature_comparison_types : dict
        Dictionary specifying if a feature should be '>' or '<' its threshold.
    duration_threshold_ms : int
        Minimum duration (in ms) for features to be sustained above the threshold.

    Returns
    -------
    float
        Speech onset time in milliseconds, or None if not found.
    """
    time_step_ms = 1  # Spectrogram step size is 1ms
    duration_threshold_frames = int(duration_threshold_ms / time_step_ms)

    num_frames = len(features["Amplitude"])

    if num_frames < duration_threshold_frames:
        return None

    candidate_onsets = []
    feature_names_ordered = [
        "Amplitude",
        "WienerEntropy",
        "SpectralChange",
        "AmplitudeModulation",
        "FrequencyModulation",
        "HarmonicPitch",
    ]

    def is_feature_above_criterion(feature_name, frame_idx):
        """Check if a single feature meets its criterion at a given frame."""
        value = features[feature_name][frame_idx]
        threshold = optimized_thresholds_dict.get(feature_name)
        comparison_type = feature_comparison_types.get(feature_name)

        if threshold is None or comparison_type is None:
            # Fallback for missing threshold/type - consider it not met
            return False

        if comparison_type == ">":
            return value > threshold
        if comparison_type == "<":
            return value < threshold
        return False  # Unknown comparison type

    for i in range(num_frames):
        # Check if at least MIN_FEATURES_CRITERION of 6
        # features meet their criterion at current frame i
        features_meeting_criterion_count = 0
        for name in feature_names_ordered:
            if is_feature_above_criterion(name, i):
                features_meeting_criterion_count += 1

        if features_meeting_criterion_count >= MIN_FEATURES_CRITERION:
            # Check if this condition holds for the next 'duration_threshold_frames'
            is_sustained = True
            for j in range(1, duration_threshold_frames):  # Check from next frame onwards
                if i + j >= num_frames:  # Reached end of signal
                    is_sustained = False
                    break

                current_features_meeting_count = 0
                for name in feature_names_ordered:
                    if is_feature_above_criterion(name, i + j):
                        current_features_meeting_count += 1

                if current_features_meeting_count < MIN_FEATURES_CRITERION:
                    is_sustained = False
                    break

            if is_sustained:
                candidate_onsets.append(i)

    if not candidate_onsets:
        return None  # No sustained onset region found

    first_candidate_frame = candidate_onsets[0]

    # "quantified speech onset as the first point in time at which the amplitude was
    # elevated above threshold within the time window defined by the four-feature criterion."
    # This means we specifically check the amplitude feature within the identified sustained window.

    onset_frame = None
    # The Amplitude feature uses the '>' comparison type.
    amplitude_threshold = optimized_thresholds_dict.get("Amplitude")

    for k in range(first_candidate_frame, first_candidate_frame + duration_threshold_frames):
        if k < num_frames and features["Amplitude"][k] > amplitude_threshold:
            onset_frame = k
            break

    if onset_frame is None:
        # Fallback if no amplitude elevation is found within the window
        # (should be rare if Amplitude was one of the 4 features)
        onset_frame = first_candidate_frame

    return onset_frame * time_step_ms  # Convert frame index to milliseconds


def run_chronset(
    wav_file_path: str,
    mat_file_path: str = "thresholds/greedy_optim_thresholds_BCN_final.mat",
    optimized_thresholds_dict: dict | None = None,
    feature_comparison_types: dict | None = None,
    duration_threshold_ms: int = 35,  # This parameter from detect_speech_onset
) -> float | None:
    """
    Run the Chronset algorithm to detect speech onset using optimized thresholds.

    Parameters
    ----------
    wav_file_path : str
        Path to the input WAV file.
    mat_file_path : str, optional
        Path to the MAT file containing optimized thresholds.
    optimized_thresholds_dict : dict, optional
        Dictionary of optimized thresholds for each feature. If None, uses
        arbitrary defaults.
    feature_comparison_types : dict, optional
        Dictionary specifying if a feature should be '>' or '<' its threshold.
        If None, assumes all are '>'.
    duration_threshold_ms : int
        Minimum duration (in ms) for features to be above threshold.
    plot_features : bool
        If True, plots the normalized features and detected onset. Requires
        matplotlib.

    Returns
    -------
    float
        Detected speech onset time in milliseconds, or None.
    """
    if optimized_thresholds_dict is None or feature_comparison_types is None:
        # Load optimized thresholds from the provided .mat file
        optimized_thresholds_dict, feature_comparison_types = extract_thresholds_from_mat(
            mat_file_path
        )
        if optimized_thresholds_dict is None or feature_comparison_types is None:
            logger.error("Failed to load optimized thresholds from .mat file.")
            return None

    # Load the audio file
    fs, signal = load_audio(wav_file_path)
    if signal is None or fs is None:  # Explicitly check fs as well
        logger.error(f"Failed to load audio or get sampling rate from {wav_file_path}.")
        return None

    # Define parameters based on the paper
    window_length_ms = 10
    step_size_ms = 1

    # Step 1 & 2: Multitaper Spectral Analysis (Approximated)
    f, _, Sxx = multitaper_spectrogram_approx(  # noqa: N806
        signal, fs, window_length_ms, step_size_ms
    )

    # Handle cases where Sxx is empty (e.g., very short or silent signal)
    if Sxx.shape[1] == 0:
        logger.warning("Spectrogram is empty. Cannot extract features.")
        return None

    # Step 3: Extract and Process Features
    features = extract_features(fs, f, Sxx)

    # Step 4: Automatic Onset Detection
    onset_time_ms = detect_speech_onset(
        features, optimized_thresholds_dict, feature_comparison_types, duration_threshold_ms
    )

    logger.info(f"Detected speech onset time: {onset_time_ms} ms")

    return onset_time_ms
