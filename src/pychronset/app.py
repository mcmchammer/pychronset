"""
Provide a Gradio-based interface for speech onset detection using Pychronset.

Features:
- Upload and process audio files (e.g., .wav, .weba) for onset detection.
- Visualize signals and detected onsets.
- Download results as a CSV file.
"""

import csv
import io
import logging
import os
import tempfile
import uuid  # Added uuid module
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend BEFORE importing pyplot
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# Assuming run_chronset is in the same package, adjust import if necessary
from pychronset import run_chronset
from pychronset.utils.io import load_audio  # For loading audio data for plotting

logger = logging.getLogger(__name__)

# Define the mat file path relative to the project root
# The run_chronset function expects it relative to where it's called from,
# or an absolute path. Since this app.py might be run from different locations,
# it's safer to resolve its absolute path.
# The default in run_chronset is 'thresholds/greedy_optim_thresholds_BCN_final.mat'
# We need to make sure this path is correct relative to the CWD when the app runs.
# For simplicity, we'll assume the app is run from the project root or that
# the 'thresholds' directory is accessible.
# A more robust way would be to determine the project root and build the path from there.
# For now, we rely on the default path in run_chronset or ensure CWD is project root.
# Alternatively, construct an absolute path if app.py's location is fixed relative to thresholds.
# Assuming app.py is in src/pychronset and thresholds is at the project root:
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAT_FILE_PATH = str(PROJECT_ROOT / "thresholds" / "greedy_optim_thresholds_BCN_final.mat")


def plot_signal_with_onset(selected_filename: str, all_data: list) -> plt.Figure:
    """
    Plot the signal and detected onset for a selected file.

    Parameters
    ----------
    selected_filename : str
        The name of the file to inspect.
    all_data : list of dict
        A list of dictionaries containing signal data, sample rate, and onset information.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure object displaying the signal and onset information.
    """
    if not selected_filename or not all_data:
        # Return a blank figure or a message
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "Please select a file to inspect.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        plt.tight_layout()
        return fig

    data_item = next((item for item in all_data if item["filename"] == selected_filename), None)

    if not data_item or data_item.get("signal") is None or data_item.get("sr") is None:
        fig, ax = plt.subplots()
        error_message = f'Data for "{selected_filename}" not available or incomplete.'
        if data_item and data_item.get("error_loading"):
            error_message += f"\nReason: {data_item['error_loading']}"
        ax.text(
            0.5,
            0.5,
            error_message,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            wrap=True,
            bbox={"boxstyle": "round,pad=0.5", "fc": "lightcoral", "alpha": 0.7},
        )
        plt.tight_layout()
        return fig

    signal = data_item["signal"]
    sr = data_item["sr"]
    onset_time_ms_val = data_item["onset_time_ms"]

    fig, ax = plt.subplots(figsize=(10, 4))  # Adjusted figure size

    # Add type check for signal
    if not isinstance(signal, np.ndarray):
        error_message = f'Signal data for "{selected_filename}" is not in the expected format (should be a NumPy array, but got {type(signal).__name__}).'
        if data_item.get("error_loading"):
            error_message += f"\nAudio load error: {data_item['error_loading']}"
        ax.text(
            0.5,
            0.5,
            error_message,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            wrap=True,
            bbox={"boxstyle": "round,pad=0.5", "fc": "lightcoral", "alpha": 0.7},
        )
        plt.tight_layout()
        return fig

    time_samples = np.arange(len(signal))
    ax.plot(time_samples, signal, lw=0.8)  # Thinner line for signal
    ax.set_xlabel("Samples")
    ax.set_ylabel("Signal Amplitude")
    ax.set_title(f"Signal: {selected_filename}", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.7)

    if isinstance(onset_time_ms_val, int | float):
        onset_sample = (onset_time_ms_val / 1000.0) * sr
        ax.axvline(
            x=onset_sample, color="r", linestyle="--", label=f"Onset: {onset_time_ms_val:.2f} ms"
        )
        ax.legend(fontsize=8)
    elif isinstance(onset_time_ms_val, str) and (
        "Error" in onset_time_ms_val or "failed" in onset_time_ms_val.lower()
    ):
        ax.text(
            0.02,
            0.95,
            f"Onset detection issue: {onset_time_ms_val}",
            transform=ax.transAxes,
            fontsize=8,
            color="red",
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "yellow", "alpha": 0.6},
        )
    elif onset_time_ms_val == "N/A":
        ax.text(
            0.02,
            0.95,
            "Onset: N/A (not detected)",
            transform=ax.transAxes,
            fontsize=8,
            color="darkorange",
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "fc": "lightgray", "alpha": 0.6},
        )

    plt.tight_layout()
    return fig


# Helper function to be run in parallel by ProcessPoolExecutor
def _process_single_file_for_inspection_worker(
    file_path, original_filename, mat_file_path_for_worker
):
    onset_time_ms_val: float | str | None = None
    signal_data_val = None
    sr_data_val = None
    onset_error_val = None  # Specific error from onset detection
    load_error_val = None  # Specific error from audio loading

    try:
        onset_time_ms_val = run_chronset(file_path, mat_file_path=mat_file_path_for_worker)
        if onset_time_ms_val is None:  # run_chronset might return None if no onset found
            onset_time_ms_val = "N/A"
    except Exception as e:
        onset_error_val = f"Error in run_chronset: {str(e)[:150]}"
        onset_time_ms_val = onset_error_val  # This value will be used in CSV and for display if plotting fails due to this
        logger.error(f"Error during onset detection for {original_filename}: {e}")

    # Attempt to load audio regardless of onset detection outcome,
    # as user might still want to see the signal if onset detection failed.
    try:
        # Corrected unpacking order: load_audio returns (sample_rate, signal_data)
        potential_sr, potential_signal = load_audio(file_path)
        if isinstance(potential_signal, np.ndarray) and isinstance(potential_sr, int):
            signal_data_val = potential_signal
            sr_data_val = potential_sr
        else:
            # load_audio did not return expected types or did not raise an exception for failure
            load_error_val = (
                f"Audio load error: `load_audio` returned unexpected types "
                f"(signal: {type(potential_signal).__name__}, sr: {type(potential_sr).__name__}). "
                f"Expected (numpy.ndarray, int for signal and sr respectively after unpacking)."  # Clarified expected types post-unpacking
            )
            # Ensure signal_data_val and sr_data_val are None if types are wrong
            signal_data_val = None
            sr_data_val = None
            logger.warning(f"Warning for {original_filename}: {load_error_val}")

    except Exception as e:
        load_error_val = f"Audio load error during call: {str(e)[:150]}"
        # signal_data_val and sr_data_val remain None if an exception occurs during the call
        signal_data_val = None
        sr_data_val = None
        logger.error(f"Error during audio loading for {original_filename}: {e}")

    return {
        "original_filename": original_filename,
        "onset_time_ms": onset_time_ms_val,  # This can be time, "N/A", or an error string from run_chronset
        "signal": signal_data_val,
        "sr": sr_data_val,
        "filepath": file_path,
        "error_loading_audio": load_error_val,  # Specific to audio loading failure
        # "error_onset_detection": onset_error_val # This info is embedded in onset_time_ms if error occurred
    }


def process_wav_files_and_prepare_inspection(files: list) -> tuple:
    """
    Process uploaded audio files for onset detection and prepare data for inspection.

    Parameters
    ----------
    files : list
        List of uploaded audio files.

    Returns
    -------
    tuple
        A tuple containing:
        - Path to the generated CSV file (or None if no file was created).
        - Gradio update object for the file selector dropdown.
        - Gradio update object for the visibility of the inspection tab.
        - List of detailed results for each processed file.
        - None (placeholder for clearing plot output).
    """
    if not files:
        return None, gr.update(choices=[], value=None), gr.update(visible=False), [], None

    detailed_results = []
    csv_rows = []

    file_paths = [file.name for file in files]
    original_filenames = [os.path.basename(getattr(file, "orig_name", file.name)) for file in files]

    with ProcessPoolExecutor() as executor:
        # Pass MAT_FILE_PATH to each worker
        futures = [
            executor.submit(
                _process_single_file_for_inspection_worker, fp, original_filenames[i], MAT_FILE_PATH
            )
            for i, fp in enumerate(file_paths)
        ]
        for future in futures:
            try:
                # Result is a dictionary from the worker
                processed_data_item = future.result()

                # For CSV: filename and onset_time_ms (which includes onset errors)
                csv_rows.append(
                    {
                        "filename": processed_data_item["original_filename"],
                        "onset_time_ms": processed_data_item["onset_time_ms"],
                    }
                )

                # For detailed_results (used by plot_signal_with_onset state):
                detailed_results.append(
                    {
                        "filename": processed_data_item["original_filename"],
                        "onset_time_ms": processed_data_item["onset_time_ms"],
                        "signal": processed_data_item["signal"],
                        "sr": processed_data_item["sr"],
                        "filepath": processed_data_item["filepath"],
                        "error_loading": processed_data_item[
                            "error_loading_audio"
                        ],  # Key expected by plot_signal_with_onset
                    }
                )

            except Exception as e:  # noqa: PERF203
                # This is a fallback for errors in future.result() itself, not errors from the
                # worker function. Worker errors should be handled and returned within
                # processed_data_item. To associate this error with a file is hard here,
                # so we log it. A more robust solution might involve passing original_filename
                # to the future and catching it here.
                logger.error(f"Critical error retrieving result from worker: {e}")
                # We could add a placeholder error to detailed_results if
                # we knew which file failed at this stage.
                # For now, this error means a result for one file might be missing.

    # Create CSV file for download
    csv_file_to_return = None
    if csv_rows:
        output = io.StringIO()
        # Ensure fieldnames match keys in csv_rows
        writer = csv.DictWriter(output, fieldnames=["filename", "onset_time_ms"])
        writer.writeheader()
        writer.writerows(csv_rows)
        csv_data = output.getvalue()
        output.close()

        csv_filename_prefix = f"pychronset_results_{uuid.uuid4()}"
        try:
            # delete=False is important for Gradio to be able to serve it
            with tempfile.NamedTemporaryFile(
                mode="w+",
                delete=False,
                prefix=csv_filename_prefix,
                suffix=".csv",
                encoding="utf-8",
                newline="",
            ) as tmp_csv:
                tmp_csv.write(csv_data)
                csv_file_to_return = tmp_csv.name
        except Exception as e:
            logger.error(f"Critical error creating temporary CSV file: {e}")
            csv_file_to_return = None

    # Prepare list of filenames for the dropdown, only if signal data is available
    filenames_for_dropdown = [
        d["filename"] for d in detailed_results if d["signal"] is not None and d["sr"] is not None
    ]

    # If no files could be processed successfully for inspection, hide the inspection tab
    if not filenames_for_dropdown:
        return (
            csv_file_to_return,
            gr.update(choices=[], value=None),
            gr.update(visible=False),
            detailed_results,
            None,
        )

    return (
        csv_file_to_return,
        gr.update(choices=filenames_for_dropdown, value=None),  # Update dropdown, clear selection
        gr.update(visible=True),  # Make inspect_tab visible
        detailed_results,  # New state for processed_data_state
        None,  # Clear plot_output in inspect tab
    )


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as demo:
    processed_data_state = gr.State([])  # To store signal data, sr, and onset for each file

    gr.Markdown(
        "# PyChronset Speech Onset Detection\n\n[![View on GitHub](https://img.shields.io/badge/GitHub-View_Source-blue?logo=github)](https://github.com/mcmchammer/pychronset)"
    )

    with gr.Tabs() as tabs:
        with gr.TabItem("Process Files", id=0):
            gr.Markdown(
                "Upload one or more audio files (e.g., .wav, .weba). Results will be available as a downloadable CSV, "
                "and visualizations will be available in the 'Inspect Results' tab."
            )
            with gr.Row():
                wav_input = gr.File(
                    file_count="multiple",
                    file_types=[".wav", ".WAV", ".weba", ".WEBA"],
                    label="Upload audio files",
                )
                csv_output = gr.File(label="Download Results (CSV)")
            process_button = gr.Button("Process Files and Prepare Inspection")

        with gr.TabItem("Inspect Results", id=1, visible=False) as inspect_tab:
            gr.Markdown("Select a processed file to visualize its signal and detected onset.")
            with gr.Row():
                file_selector_dropdown = gr.Dropdown(
                    label="Select File to Inspect", choices=[], interactive=True
                )
            with gr.Row():
                plot_output = gr.Plot(label="Signal with Onset")

            # Clear button for the plot
            clear_plot_button = gr.Button("Clear Plot and Selection")

    # Define click actions
    process_button.click(
        fn=process_wav_files_and_prepare_inspection,
        inputs=wav_input,
        outputs=[
            csv_output,
            file_selector_dropdown,
            inspect_tab,  # This will update the 'visible' property of the tab
            processed_data_state,
            plot_output,  # Clear previous plot when new files are processed
        ],
    )

    file_selector_dropdown.change(
        fn=plot_signal_with_onset,
        inputs=[file_selector_dropdown, processed_data_state],
        outputs=plot_output,
    )

    def clear_plot_and_selection_action(current_data_state: list):
        """
        Clear the selected file in the dropdown and the plot output,
        but retain the available choices in the dropdown.

        Parameters
        ----------
        current_data_state : list
            The current state containing data for all processed files.

        Returns
        -------
        tuple
            A tuple containing updates for the dropdown (retained choices, cleared value)
            and the plot (cleared output).
        """
        choices_to_retain = []
        if current_data_state:  # Check if current_data_state is not None or empty
            choices_to_retain = [
                d["filename"]
                for d in current_data_state
                if d.get("signal") is not None and d.get("sr") is not None
            ]
        # Update dropdown: keep choices, clear selected value. Clear plot.
        return gr.update(choices=choices_to_retain, value=None), None

    clear_plot_button.click(
        fn=clear_plot_and_selection_action,
        inputs=[processed_data_state],  # Pass the state as input
        outputs=[file_selector_dropdown, plot_output],
    )


if __name__ == "__main__":
    # Check if the MAT file exists before launching
    if not os.path.exists(MAT_FILE_PATH):
        logger.error(f"MAT file not found at {MAT_FILE_PATH}")
        logger.info(
            "Please ensure the 'thresholds' directory and its contents are correctly placed relative to the project root."
        )
    else:
        logger.info(f"Using MAT file: {MAT_FILE_PATH}")
        demo.launch()
