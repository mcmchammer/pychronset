"""I/O utility functions."""

import logging
import os

import numpy as np
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from scipy.io import loadmat

logger = logging.getLogger(__name__)


def load_audio(audio_file_path: str) -> tuple[int | None, np.ndarray | None]:
    """Load an audio file and return sample rate and audio data."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    try:
        audio = AudioSegment.from_file(audio_file_path)
        # Convert to mono
        audio = audio.set_channels(1)
        # Get raw audio data as numpy array
        # AudioSegment.get_array_of_samples() returns an array of samples.
        # Each sample is an integer representing the amplitude.
        # The data needs to be scaled to float32, typically in [-1.0, 1.0] range for processing.
        # The maximum value for a signed 16-bit integer is 32767.
        data = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Normalize to [-1.0, 1.0] if it's not already (depends on source format and pydub behavior)
        # Pydub samples are integers based on sample_width. For 16-bit (2 bytes), max is 2**(8*2-1)-1 = 32767
        # For 24-bit (3 bytes), max is 2**(8*3-1)-1 = 8388607
        # For 32-bit (4 bytes), max is 2**(8*4-1)-1 = 2147483647
        # We should normalize based on the actual max possible value for the sample width.
        if audio.sample_width == 2 or audio.sample_width == 3:  # 16-bit
            data = data / (2 ** (8 * audio.sample_width - 1))
        elif audio.sample_width == 4:  # 32-bit float or int
            # If it's already float, it might be in [-1, 1]. If int, needs scaling.
            # Assuming it's int for now, consistent with other cases.
            data = data / (2 ** (8 * audio.sample_width - 1))
        # else: # 8-bit or other cases, might need different handling or pydub handles it.
        # For simplicity, we assume common cases. If data is already float and in [-1,1], this might be an issue.
        # However, get_array_of_samples usually gives integers.

        # Ensure float32 for processing
        data = data.astype(np.float32)

        return audio.frame_rate, data
    except CouldntDecodeError:
        logger.error(
            f"Could not decode audio file: {audio_file_path}. Ensure ffmpeg is installed and supports the format."
        )
        return None, None
    except FileNotFoundError as e:
        if e.filename in ("ffmpeg", "ffprobe"):
            logger.error(
                f"Error loading audio file {audio_file_path}: {e}. "
                "This indicates that ffmpeg (which includes ffprobe) is not installed or not in your system's PATH. "
                "Please install ffmpeg to support this audio format."
            )
        else:
            logger.error(f"File not found error while loading audio file {audio_file_path}: {e}")
        return None, None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading audio file {audio_file_path}: {e}"
        )
        return None, None


def extract_thresholds_from_mat(mat_file_path: str) -> tuple[dict | None, dict | None]:
    """
    Extract optimized thresholds from the provided MATLAB .mat file.

    Args:
        mat_file_path (str): Path to the MATLAB .mat file containing optim_data.

    Returns
    -------
        tuple
            A tuple containing:
            - dict: A dictionary of extracted feature names and their corresponding thresholds.
            - dict: A dictionary indicating whether each feature uses a '>' or '<' threshold.
    """
    try:
        mat_data = loadmat(mat_file_path)
    except FileNotFoundError:
        logger.error(f".mat file not found at {mat_file_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error loading .mat file: {e}")
        return None, None

    # The 'optim_data' structure in MATLAB becomes a structured NumPy array/dict in Python
    # We expect it to be accessible via a key in the loaded mat_data
    if "optim_data" not in mat_data:
        logger.error("'optim_data' not found in the .mat file.")
        return None, None

    optim_data = mat_data["optim_data"]

    # In MATLAB, optim_data.hist_e is directly accessed. In Python, for structured arrays,
    # elements are accessed by field names and then by index (e.g., [0,0] for the first element).
    hist_e = optim_data["hist_e"][0, 0]
    hist_t = optim_data["hist_t"][0, 0]

    # Find the index of the minimum training error
    min_val = np.min(hist_e)
    i1_coords, i2_coords = np.where(hist_e == min_val)

    # MATLAB code extracts the first unique occurrence:
    # i1 = unique(i1); i2 = unique(i2); i1 = i1(1); i2 = i2(1);
    # So we take the first element from the found coordinates.
    i1 = i1_coords[0]
    i2 = i2_coords[0]

    # Print training and testing error as in MATLAB code for verification
    training_error = hist_e[i1, i2]
    test_e = optim_data["test_e"][0, 0]  # Assuming test_e is structured similarly to hist_e
    testing_error = test_e[i1, i2]

    logger.info(f"The selected thresholds have a training error of: {training_error:.2f} ms")
    logger.info(f"The selected thresholds have a testing error of: {testing_error:.2f} ms")

    # Extract thresholds
    # hist_t is (rows, cols, features) in MATLAB.
    # We access hist_t[i1, i2, it] to get the specific threshold.
    num_features = hist_t.shape[2]
    extracted_thresholds_list = []
    for it in range(num_features):
        threshold_val = hist_t[i1, i2, it]
        extracted_thresholds_list.append(
            threshold_val.item()
        )  # .item() converts 0-dim array to scalar

    logger.info("\nExtracted feature thresholds (ordered as per paper's feature list):")
    for thresh_val in extracted_thresholds_list:
        logger.info(f"{thresh_val:.6f}")

    # Define how each threshold is used based on the MATLAB example:
    # if ismember(it,[1 3 4 6]) then features{it}>tresh{it}
    # elseif ismember(it,[2 5]) then features{it}<tresh{it}
    # Mapping to paper's feature order (1-indexed in MATLAB, 0-indexed here):
    # 1: Amplitude, 2: WienerEntropy, 3: SpectralChange, 4: AmplitudeModulation,
    # 5: FrequencyModulation, 6: HarmonicPitch
    feature_names_ordered = [
        "Amplitude",
        "WienerEntropy",
        "SpectralChange",
        "AmplitudeModulation",
        "FrequencyModulation",
        "HarmonicPitch",
    ]

    feature_comparison_types = {
        feature_names_ordered[0]: ">",  # Amplitude (Feature 1)
        feature_names_ordered[1]: "<",  # WienerEntropy (Feature 2)
        feature_names_ordered[2]: ">",  # SpectralChange (Feature 3)
        feature_names_ordered[3]: ">",  # AmplitudeModulation (Feature 4)
        feature_names_ordered[4]: "<",  # FrequencyModulation (Feature 5)
        feature_names_ordered[5]: ">",  # HarmonicPitch (Feature 6)
    }

    # Create a dictionary of feature names to their extracted threshold values
    optimized_thresholds_dict = {}
    for i, name in enumerate(feature_names_ordered):
        if i < len(extracted_thresholds_list):
            optimized_thresholds_dict[name] = extracted_thresholds_list[i]
        else:
            logger.warning(
                f"Not enough optimized thresholds for {name}. Expected {len(feature_names_ordered)} but got {len(extracted_thresholds_list)}."
            )

    return optimized_thresholds_dict, feature_comparison_types
