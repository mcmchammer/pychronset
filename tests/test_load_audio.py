"""
Tests for the read_wav function in the pychronset.utils module.

This module contains unit tests to verify the functionality of the read_wav
function, ensuring it correctly reads WAV files and returns the expected data
and sample rate.
"""

from pychronset.utils.io import load_audio


def test_load_audio():
    """Test the read_wav function with a valid WAV file."""
    wav_file = "tests/data/1P8_91235211.WAV"
    fs, data = load_audio(wav_file)

    assert data is not None
    assert fs > 0
    assert len(data) > 0
