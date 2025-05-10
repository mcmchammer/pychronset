"""
Test module for multitaper spectrogram approximation.

This module contains tests for the multitaper spectrogram approximation
functionality provided by the pychronset library.
"""

import logging

from pychronset.utils.io import load_audio
from pychronset.utils.signal import multitaper_spectrogram_approx

logger = logging.getLogger(__name__)


def test_multitaper_spectrogram_approx():
    """
    Test the multitaper spectrogram approximation function.

    This function tests the multitaper spectrogram approximation on a sample WAV file.
    """
    # Load the audio file
    wav_file_path = "tests/data/1P8_91235211.WAV"
    fs, signal = load_audio(wav_file_path)
    assert signal is not None

    # Define parameters based on the paper
    window_length_ms = 10
    step_size_ms = 1

    f, t_spectrogram, Sxx = multitaper_spectrogram_approx(  # noqa: N806
        signal, fs, window_length_ms, step_size_ms
    )

    # Check the output dimensions
    assert len(f) > 0
    assert len(t_spectrogram) > 0
    assert Sxx.shape == (len(f), len(t_spectrogram))

    logger.info(f"Frequency bins: {len(f)}")
    logger.info(f"Time bins: {len(t_spectrogram)}")
    logger.info(f"Spectrogram shape: {Sxx.shape}")
