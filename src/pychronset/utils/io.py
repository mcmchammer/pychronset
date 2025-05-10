"""I/O utility functions."""

import logging
import os

import numpy as np
from scipy.io import loadmat, wavfile

logger = logging.getLogger(__name__)


def load_audio(wav_file_path: str) -> tuple[int | None, np.ndarray | None]:
    """Load a WAV file and return sample rate and audio data."""
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"Audio file not found: {wav_file_path}")
    try:
        sample_rate, data = wavfile.read(wav_file_path)
        if data.ndim > 1:
            data = data.mean(axis=1)  # Convert stereo to mono by averaging channels
        return sample_rate, data.astype(np.float32)  # Ensure float32 for processing
    except Exception as e:
        logger.info(f"Error loading audio file: {e}")
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
