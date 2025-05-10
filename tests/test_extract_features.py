"""
Tests for the feature extraction functionality in the pychronset package.

This module contains unit tests to ensure the correct behavior of the
`extract_features` function and its integration with the `load_audio` utility.
"""

import logging

from pychronset import extract_features
from pychronset.utils.io import load_audio
from pychronset.utils.signal import multitaper_spectrogram_approx

logger = logging.getLogger(__name__)


def test_extract_features():
    """
    Test the feature extraction functionality in the pychronset package.

    This function tests the feature extraction on a sample WAV file.
    """
    # Load the audio file
    wav_file_path = "tests/data/1P8_91235211.WAV"
    fs, signal = load_audio(wav_file_path)

    # Define parameters based on the paper
    window_length_ms = 10
    step_size_ms = 1

    # Step 1 & 2: Multitaper Spectral Analysis (Approximated)
    f, _, Sxx = multitaper_spectrogram_approx(  # noqa: N806
        signal, fs, window_length_ms, step_size_ms
    )

    # Step 3: Extract and Process Features
    features = extract_features(fs, f, Sxx)

    # Check the output dimensions
    assert len(features) == 6  # noqa: PLR2004

    logger.info(f"Extracted features: {features}")
