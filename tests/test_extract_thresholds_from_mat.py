"""
Provide tests for extracting thresholds and comparison types from a .mat file.

Include a function to read optimization data from a .mat file and display the extracted
thresholds and their corresponding comparison types in a formatted manner.
"""

import logging

from pychronset.utils.io import extract_thresholds_from_mat

logger = logging.getLogger(__name__)


def test_extract_thresholds_from_mat():
    """
    Extract and display optimized thresholds and their comparison types from a .mat file.

    This function reads a .mat file containing optimization data, extracts the thresholds
    and their corresponding comparison types, and prints them in a formatted manner.
    """
    mat_file_path = "thresholds/greedy_optim_thresholds_BCN_final.mat"

    optimized_thresholds_dict, feature_comparison_types = extract_thresholds_from_mat(mat_file_path)

    assert optimized_thresholds_dict is not None

    logger.info("\n--- Final Optimized Thresholds and Their Usage ---")
    for name, value in optimized_thresholds_dict.items():
        logger.info(
            f"{name}: {value:.6f} (Comparison: {feature_comparison_types.get(name, 'Unknown')})"
        )
