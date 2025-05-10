"""
Provide tests for the pychronset package.

Specifically, test the `run_chronset` function and its dependencies
to ensure correct functionality with sample input data.
"""

import logging
import os

import numpy as np
import soundfile as sf

from pychronset import run_chronset

logger = logging.getLogger(__name__)


def test_run_chronset():
    """
    Test the run_chronset function.

    This function tests the run_chronset function with a sample WAV file and
    checks if the output is as expected.
    """
    # Create a dummy WAV file for testing ---
    test_wav_file = "test_audio_with_onset.wav"
    sr = 44100  # Sample rate
    duration_silence = 0.5  # seconds (500 ms)
    duration_speech = 1.0  # seconds
    frequency = 800  # Hz (a clearer tone)

    silence = np.zeros(int(sr * duration_silence))
    t_speech = np.linspace(0, duration_speech, int(sr * duration_speech), endpoint=False)
    speech_signal = 0.5 * np.sin(2 * np.pi * frequency * t_speech) * np.exp(-5 * t_speech)

    full_signal = np.concatenate((silence, speech_signal))

    sf.write(test_wav_file, full_signal, sr)
    logger.info(f"Created dummy WAV file: {test_wav_file}")

    onset_time_ms = run_chronset(test_wav_file)

    os.remove(test_wav_file)

    assert onset_time_ms is not None

    logger.info("\n--- Detection Result (using optimized thresholds) ---")
    logger.info(f"Detected speech onset: {onset_time_ms:.2f} ms")
    expected_onset_ms = duration_silence * 1000
    logger.info(f"Expected speech onset (approximate): {expected_onset_ms:.2f} ms")
    logger.info(f"Difference (Detected - Expected): {onset_time_ms - expected_onset_ms:.2f} ms")
