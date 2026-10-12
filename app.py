"""Streamlit interface for the Signal Processing Lab."""

# Generate example signals and time labels.
import numpy as np

# Build the web interface.
import streamlit as st

# Plot waveforms and frequency spectra.
import matplotlib.pyplot as plt

# Build downloadable CSV text in memory.
from io import StringIO

# Prepare uploaded and exported files in temporary folders.
from tempfile import TemporaryDirectory

# Build paths and read or write file bytes.
from pathlib import Path

# Reuse the processing functions from our shared package.
from signal_lab.validation import validate_signal
from signal_lab.spectral import calculate_fft, detect_peaks
from signal_lab.recommendation import recommend_filter
from signal_lab.filters import apply_filter
from signal_lab.metrics import compare_rms
from signal_lab.data_io import load_wav, load_csv, save_processed_wav


# Configure the browser title and page width.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Define the workflow order.
stages = ["Upload", "Inspect", "Configure", "Compare", "Export"]

# Select Upload when a new session starts.
if "current_stage" not in st.session_state:
    st.session_state["current_stage"] = "Upload"


def select_stage(stage_name):
    """Change the selected workflow stage."""

    st.session_state["current_stage"] = stage_name


def clear_processed_result():
    """Remove results belonging to an old input or configuration."""

    # Missing keys are harmless because the default is None.
    st.session_state.pop("filtered_samples", None)
    st.session_state.pop("applied_filter", None)
    st.session_state.pop("applied_band", None)

    # Reset the CSV-to-audio choice for a new result.
    st.session_state.pop("csv_audio_export", None)


def clear_input():
    """Remove an input after a failed loading attempt."""

    # Prevent a failed upload from leaving an old signal active.
    for key in (
        "samples",
        "fs_hz",
        "signal_source",
        "is_demo",
        "input_kind",
    ):
        st.session_state.pop(key, None)

    clear_processed_result()


def store_input(samples, fs_hz, source_name, is_demo, input_kind):
    """Store a validated input and reset its previous settings."""

    # Store samples, sampling rate, and source information.
    st.session_state["samples"] = samples
    st.session_state["fs_hz"] = fs_hz
    st.session_state["signal_source"] = source_name
    st.session_state["is_demo"] = is_demo
    st.session_state["input_kind"] = input_kind

    # Previous processed samples belong to the previous input.
    clear_processed_result()

    # Use known useful-band settings for the generated example.
    if is_demo:
        useful_band = (80.0, 250.0)

    # Start other inputs with a broad band below Nyquist.
    # This is an initial setting, not a noise diagnosis.
    else:
        useful_band = (0.0, float(0.9 * (fs_hz / 2)))

    st.session_state["useful_band"] = useful_band

    # Clear old widget values so the new defaults appear.
    st.session_state.pop("useful_low_hz", None)
    st.session_state.pop("useful_high_hz", None)


def amplitude_unit():
    """Return an amplitude label appropriate to the current input."""

    # CSV values retain their units, which are not yet entered in the app.
    if st.session_state.get("input_kind") == "csv":
        return "original CSV units"

    # The demo and WAV loader use digital amplitude values.
    return "dimensionless"


def style_plot(axis, x_label, y_label):
    """Apply consistent labels and styling."""

    # Label both axes, including units.
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    # Add a faint grid and hide unnecessary plot borders.
    axis.grid(alpha=0.15)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def display_plot(figure):
    """Display a figure and release its plotting resources."""

    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


# Build the application header.
brand_column, status_column = st.columns([4, 1])

with brand_column:
    st.title("Signal Lab")
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

with status_column:
    st.info("Signal workspace")

st.divider()


# Build sidebar navigation.
with st.sidebar:
    st.title("Signal Lab")
    st.caption("YOUR WORKSPACE")

    for number, stage_name in enumerate(stages, start=1):
        # Highlight the selected stage.
        is_selected = (
            st.session_state["current_stage"] == stage_name
        )

        st.button(
            label=f"{number:02d} · {stage_name}",
            key=f"navigation_{stage_name}",
            type="primary" if is_selected else "secondary",
            width="stretch",
            on_click=select_stage,
            args=(stage_name,),
        )

    st.divider()
    st.caption("WAV, CSV, and guided example available.")
    st.caption("Processed CSV and WAV downloads available.")


# Retrieve the selected stage and its position.
stage = st.session_state["current_stage"]
stage_index = stages.index(stage)

st.caption(f"STEP {stage_index + 1:02d} OF {len(stages):02d}")

# Show the active signal's name.
if "samples" in st.session_state:
    st.caption(
        "Current signal: "
        + st.session_state.get("signal_source", "Built-in example")
    )


# ------------------------------------------------------------------
# STAGE 1: UPLOAD
# ------------------------------------------------------------------
if stage == "Upload":
    st.header("Start with a signal")
    st.caption("Upload WAV or CSV data, or try the guided example.")

    upload_column, example_column = st.columns(2)

    with upload_column:
        with st.container(border=True):
            st.subheader("Your recording")
            st.write("Analyse one channel from a WAV recording.")

            # Select a WAV file.
            uploaded_file = st.file_uploader(
                "Choose a WAV file",
                type=["wav"],
                key="wav_upload",
            )

            # Channel numbering starts at zero.
            channel = st.number_input(
                "Audio channel",
                min_value=0,
                value=0,
                step=1,
                key="wav_channel",
            )

            st.caption(
                "Mono: use channel 0. Stereo: 0 is left, 1 is right."
            )

            # Load only when the user clicks the button.
            if st.button(
                "Load recording",
                key="load_recording",
                type="primary",
                width="stretch",
                disabled=uploaded_file is None,
            ):
                try:
                    # Give the upload an isolated temporary folder.
                    with TemporaryDirectory() as temporary_folder:
                        input_path = (
                            Path(temporary_folder)
                            / "uploaded_recording.wav"
                        )

                        # Copy the uploaded bytes into a temporary WAV.
                        input_path.write_bytes(
                            uploaded_file.getvalue()
                        )

                        # Read and validate the selected channel.
                        # The sampling rate comes from the WAV itself.
                        samples, fs_hz = load_wav(
                            input_path,
                            channel=int(channel),
                        )

                except (ValueError, OSError, EOFError) as error:
                    clear_input()
                    st.error(
                        f"Could not load the recording: {error}"
                    )

                else:
                    # Store the new recording and reset previous results.
                    store_input(
                        samples,
                        fs_hz,
                        source_name=uploaded_file.name,
                        is_demo=False,
                        input_kind="wav",
                    )

                    st.success(
                        f"Loaded {uploaded_file.name} · "
                        f"{fs_hz:g} Hz · "
                        f"{len(samples) / fs_hz:.2f} seconds"
                    )

            st.divider()
            st.subheader("CSV signal data")

            # Select a CSV containing an amplitude column.
            uploaded_csv = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                key="csv_upload",
            )

            # Choose how the sampling rate is obtained.
            rate_source = st.radio(
                "Sampling rate source",
                ["Infer from time_s", "Enter sampling rate"],
                key="csv_rate_source",
            )

            # None asks the loader to infer the rate from timestamps.
            csv_fs_hz = None

            if rate_source == "Enter sampling rate":
                # Sampling rate is measured in samples per second, or Hz.
                csv_fs_hz = st.number_input(
                    "CSV sampling rate (Hz)",
                    min_value=0.001,
                    value=2000.0,
                    step=1.0,
                    format="%.3f",
                    key="csv_fs_hz",
                )

            st.caption(
                "Required column: amplitude. For automatic rate detection, "
                "also include uniformly spaced timestamps in time_s."
            )

            st.caption(
                "If time_s is present and you enter a rate, "
                "the loader checks that they agree."
            )

            if st.button(
                "Load CSV",
                key="load_csv_signal",
                type="primary",
                width="stretch",
                disabled=uploaded_csv is None,
            ):
                try:
                    # Give this CSV its own temporary folder.
                    with TemporaryDirectory() as temporary_folder:
                        input_path = (
                            Path(temporary_folder)
                            / "uploaded_signal.csv"
                        )

                        # Save the selected file without changing its bytes.
                        input_path.write_bytes(
                            uploaded_csv.getvalue()
                        )

                        # Reuse the shared CSV loader and validation.
                        samples, fs_hz = load_csv(
                            input_path,
                            fs_hz=csv_fs_hz,
                        )

                except (ValueError, OSError) as error:
                    clear_input()
                    st.error(f"Could not load the CSV: {error}")

                else:
                    # Store CSV data and clear previous processing results.
                    store_input(
                        samples,
                        fs_hz,
                        source_name=uploaded_csv.name,
                        is_demo=False,
                        input_kind="csv",
                    )

                    st.success(
                        f"Loaded {uploaded_csv.name} · "
                        f"{fs_hz:g} Hz · "
                        f"{len(samples):,} samples"
                    )

                    st.info(
                        "CSV amplitudes retain their original units. "
                        "Plots label them as original CSV units."
                    )

    with example_column:
        with st.container(border=True):
            st.subheader("Try the guided example")
            st.write(
                "A useful 100 Hz tone mixed with an unwanted 400 Hz tone."
            )
            st.caption(
                "5 seconds · 2,000 Hz sampling rate · "
                "dimensionless amplitude"
            )

            if st.button(
                "Load example",
                key="load_example",
                type="primary",
                width="stretch",
            ):
                # Define sampling rate in Hz and duration in seconds.
                fs_hz = 2000.0
                duration_s = 5.0

                # Calculate the sample count and time in seconds.
                sample_count = int(fs_hz * duration_s)
                time_s = np.arange(sample_count) / fs_hz

                # Generate the two known tones.
                useful_signal = np.sin(
                    2 * np.pi * 100.0 * time_s
                )
                unwanted_signal = 0.5 * np.sin(
                    2 * np.pi * 400.0 * time_s
                )

                # Validate their sum.
                samples = validate_signal(
                    useful_signal + unwanted_signal,
                    fs_hz,
                )

                # Store the demo and reset settings from previous inputs.
                store_input(
                    samples,
                    fs_hz,
                    source_name="Built-in example",
                    is_demo=True,
                    input_kind="demo",
                )

    if "samples" in st.session_state:
        st.success("Signal ready. Continue to Inspect below.")


# ------------------------------------------------------------------
# STAGE 2: INSPECT
# ------------------------------------------------------------------
elif stage == "Inspect":
    st.header("Understand your signal")
    st.caption("Inspect the waveform and its detected frequencies.")

    if "samples" not in st.session_state:
        st.warning("Go to Upload and load a signal first.")

    else:
        # Retrieve the input and amplitude label.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]
        is_demo = st.session_state.get("is_demo", True)
        units = amplitude_unit()

        rate_column, duration_column, nyquist_column = st.columns(3)

        with rate_column:
            with st.container(border=True):
                st.metric("Sampling rate", f"{fs_hz:g} Hz")

        with duration_column:
            with st.container(border=True):
                st.metric(
                    "Duration",
                    f"{len(samples) / fs_hz:.2f} s",
                )

        with nyquist_column:
            with st.container(border=True):
                st.metric(
                    "Nyquist frequency",
                    f"{fs_hz / 2:g} Hz",
                )

        with st.container(border=True):
            st.subheader("Signal preview")

            # Select a short preview, with at least two samples.
            preview_count = min(
                len(samples),
                max(2, int(0.05 * fs_hz)),
            )

            # Calculate time labels in milliseconds.
            time_ms = np.arange(preview_count) / fs_hz * 1000.0

            st.caption(
                f"Original signal · first {preview_count:,} samples · "
                f"time shown through {time_ms[-1]:g} ms"
            )

            figure, axis = plt.subplots(figsize=(8, 3.8))

            axis.plot(
                time_ms,
                samples[:preview_count],
                color="#0F766E",
                linewidth=1.8,
            )

            style_plot(
                axis,
                "Time (ms)",
                f"Amplitude ({units})",
            )

            axis.set_xlim(0, max(50.0, float(time_ms[-1])))

            plot_column, details_column = st.columns([3, 1])

            with plot_column:
                display_plot(figure)

            with details_column:
                if is_demo:
                    st.markdown("**Known demo tones**")
                    st.metric("Useful tone", "100 Hz")
                    st.metric("Unwanted tone", "400 Hz")
                    st.caption(
                        "These roles are known because we generated "
                        "the example."
                    )
                else:
                    st.markdown("**Uploaded signal**")
                    st.write(
                        st.session_state.get(
                            "signal_source",
                            "Uploaded signal",
                        )
                    )
                    st.caption(
                        "Review the spectrum before choosing "
                        "which frequencies to preserve."
                    )

        with st.container(border=True):
            st.subheader("Frequency analysis")
            st.caption("FFT of the complete signal")

            try:
                # Run the shared FFT and peak detector.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

            except ValueError as error:
                st.error(f"Frequency analysis failed: {error}")

            else:
                # Select detected frequencies and amplitudes.
                detected_hz = frequency_hz[peak_indices]
                detected_amplitude = amplitude[peak_indices]

                figure, axis = plt.subplots(figsize=(8, 3.8))

                axis.plot(
                    frequency_hz,
                    amplitude,
                    color="#0F766E",
                    linewidth=1.5,
                )

                axis.scatter(
                    detected_hz,
                    detected_amplitude,
                    color="#0F172A",
                    s=35,
                    zorder=3,
                    label="Detected peaks",
                )

                style_plot(
                    axis,
                    "Frequency (Hz)",
                    f"Estimated amplitude ({units})",
                )

                axis.set_xlim(0, fs_hz / 2)

                if len(peak_indices) > 0:
                    axis.legend(frameon=False)

                display_plot(figure)

                if len(peak_indices) == 0:
                    st.info(
                        "No prominent peaks found with these settings."
                    )
                else:
                    st.dataframe(
                        {
                            "Frequency (Hz)": detected_hz,
                            f"Estimated amplitude ({units})": (
                                detected_amplitude
                            ),
                        },
                        hide_index=True,
                        width="stretch",
                    )

                st.caption(
                    "Minimum height: 10% of the largest FFT amplitude. "
                    "Minimum prominence: 5% of that amplitude. "
                    "Minimum peak separation: 5 Hz."
                )

        st.info(
            "A detected frequency is not automatically noise. "
            "Decide which components matter for your signal."
        )

        with st.expander("How do I read this signal?"):
            st.write(
                "The waveform shows amplitude against time. "
                "The spectrum shows estimated amplitudes against frequency."
            )

            if st.session_state.get("input_kind") == "csv":
                st.write(
                    "CSV amplitudes retain their original units. "
                    "For example, voltage samples remain in volts "
                    "if the CSV values were measured in volts."
                )
            else:
                st.write(
                    "WAV integer samples are converted to dimensionless "
                    "digital amplitudes. Floating-point WAV values "
                    "are retained. These values are not calibrated "
                    "sound-pressure measurements."
                )

            st.write(
                f"This signal is sampled at {fs_hz:g} samples per second. "
                f"Its Nyquist frequency is {fs_hz / 2:g} Hz."
            )

            st.write(
                "Preventing aliasing requires appropriate filtering "
                "before sampling. Digital filtering cannot undo "
                "aliasing already present in the data."
            )

            st.write(
                "Our FFT function removes the mean for analysis, "
                "applies a Hann window, and compensates for its "
                "amplitude scaling."
            )

            if is_demo:
                st.write(
                    "The useful 100 Hz demo tone completes five cycles "
                    "in 50 milliseconds. The unwanted 400 Hz tone "
                    "adds faster changes."
                )


# ------------------------------------------------------------------
# STAGE 3: CONFIGURE
# ------------------------------------------------------------------
elif stage == "Configure":
    st.header("Choose what to preserve")
    st.caption("Set the useful band, review the suggestion, and apply it.")

    if "samples" not in st.session_state:
        st.warning("Go to Upload and load a signal first.")

    else:
        # Retrieve the input and calculate Nyquist.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]
        nyquist_hz = fs_hz / 2
        is_demo = st.session_state.get("is_demo", True)

        # Supply defaults if none have been stored.
        if "useful_band" not in st.session_state:
            st.session_state["useful_band"] = (
                0.0,
                float(0.9 * nyquist_hz),
            )

        saved_low_hz, saved_high_hz = st.session_state["useful_band"]

        with st.container(border=True):
            st.subheader("Useful frequency band")
            st.write(
                "Choose the frequencies you want to keep. "
                "Peaks outside this band become candidate interference."
            )

            lower_column, upper_column = st.columns(2)

            with lower_column:
                useful_low_hz = st.number_input(
                    "Lower frequency (Hz)",
                    min_value=0.0,
                    max_value=float(nyquist_hz),
                    value=float(saved_low_hz),
                    step=5.0,
                    key="useful_low_hz",
                )

            with upper_column:
                useful_high_hz = st.number_input(
                    "Upper frequency (Hz)",
                    min_value=0.0,
                    max_value=float(nyquist_hz),
                    value=float(saved_high_hz),
                    step=5.0,
                    key="useful_high_hz",
                )

            if is_demo:
                st.caption(
                    "For the guided example, use 80–250 Hz "
                    "to preserve the 100 Hz tone."
                )
            else:
                st.caption(
                    "The initial band is a starting point. "
                    "Use your signal's purpose and spectrum "
                    "to choose what to preserve."
                )

        # Store the current band.
        current_band = (useful_low_hz, useful_high_hz)
        st.session_state["useful_band"] = current_band

        # Remove results produced using different settings.
        if st.session_state.get("applied_band") != current_band:
            clear_processed_result()

        if useful_low_hz >= useful_high_hz:
            st.error(
                "The lower frequency must be below the upper frequency."
            )

        else:
            try:
                # Detect peaks using the same settings as Inspect.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

                detected_hz = frequency_hz[peak_indices]

                # Reuse the shared recommendation function.
                recommendation = recommend_filter(
                    detected_hz,
                    fs_hz,
                    useful_low_hz,
                    useful_high_hz,
                )

            except ValueError as error:
                clear_processed_result()
                st.error(f"Could not recommend a filter: {error}")

            else:
                # Convert internal names into readable labels.
                filter_labels = {
                    "none": "Keep signal unchanged",
                    "notch": "Notch filter",
                    "lowpass": "Low-pass filter",
                    "highpass": "High-pass filter",
                    "bandpass": "Band-pass filter",
                }

                filter_type = recommendation["filter_type"]
                cutoff_hz = recommendation["cutoff_hz"]

                with st.container(border=True):
                    st.subheader("Suggested action")
                    st.markdown(f"**{filter_labels[filter_type]}**")
                    st.write(recommendation["reason"])

                    if filter_type == "notch":
                        st.metric(
                            "Notch centre",
                            f"{cutoff_hz:g} Hz",
                        )

                    elif filter_type == "bandpass":
                        st.write(
                            f"Band edges: {cutoff_hz[0]:g} Hz "
                            f"to {cutoff_hz[1]:g} Hz"
                        )

                    elif cutoff_hz is not None:
                        st.metric(
                            "Cutoff frequency",
                            f"{cutoff_hz:g} Hz",
                        )

                    if recommendation["candidate_hz"]:
                        st.dataframe(
                            {
                                "Candidate interference (Hz)": (
                                    recommendation["candidate_hz"]
                                )
                            },
                            hide_index=True,
                            width="stretch",
                        )

                    st.caption(
                        "This is a rule-based recommendation using your "
                        "declared useful band. It does not prove that "
                        "a detected component is noise."
                    )

                    st.caption(
                        "Defaults: Butterworth design order 4; "
                        "notch Q = 30, dimensionless. "
                        "The notch design is second order."
                    )

                    st.caption(
                        "Filtering uses offline forward-backward processing. "
                        "The original samples remain available."
                    )

                    action_label = (
                        "Keep signal unchanged"
                        if filter_type == "none"
                        else "Apply suggested filter"
                    )

                    if st.button(
                        action_label,
                        key="apply_suggestion",
                        type="primary",
                        width="stretch",
                    ):
                        clear_processed_result()

                        try:
                            # Reuse the shared filtering implementation.
                            processed = apply_filter(
                                samples,
                                fs_hz,
                                filter_type,
                                cutoff_hz=cutoff_hz,
                                order=4,
                                notch_q=30.0,
                            )

                        except ValueError as error:
                            st.error(f"Filtering failed: {error}")

                        else:
                            # Store the result and the settings that made it.
                            st.session_state["filtered_samples"] = processed
                            st.session_state["applied_filter"] = (
                                recommendation.copy()
                            )
                            st.session_state["applied_band"] = current_band

                if "filtered_samples" in st.session_state:
                    applied = st.session_state["applied_filter"]

                    if applied["filter_type"] == "none":
                        st.success(
                            "Signal kept unchanged. "
                            "Result saved for comparison."
                        )
                    else:
                        st.success(
                            "Filter applied. "
                            "Processed signal saved for comparison."
                        )

                    st.caption(
                        "Processed samples: "
                        f"{len(st.session_state['filtered_samples']):,}"
                    )
                else:
                    st.info(
                        "Review the suggestion and apply it "
                        "before continuing to Compare."
                    )


# ------------------------------------------------------------------
# STAGE 4: COMPARE
# ------------------------------------------------------------------
elif stage == "Compare":
    st.header("See what changed")
    st.caption("Compare the original signal with the processed result.")

    if "filtered_samples" not in st.session_state:
        st.warning("Go to Configure and apply a filter first.")

    else:
        # Retrieve the signals, sampling rate, and amplitude label.
        original = st.session_state["samples"]
        processed = st.session_state["filtered_samples"]
        fs_hz = st.session_state["fs_hz"]
        units = amplitude_unit()

        # Retrieve the settings associated with this result.
        applied = st.session_state["applied_filter"]
        applied_low_hz, applied_high_hz = (
            st.session_state["applied_band"]
        )

        st.caption(
            f"Applied operation: {applied['filter_type']} · "
            f"Declared useful band: "
            f"{applied_low_hz:g}–{applied_high_hz:g} Hz"
        )

        try:
            # Compare RMS through the shared metrics module.
            measurements = compare_rms(original, processed)

            # Apply identical FFT processing to both signals.
            original_frequency_hz, original_amplitude = calculate_fft(
                original,
                fs_hz,
            )
            processed_frequency_hz, processed_amplitude = calculate_fft(
                processed,
                fs_hz,
            )

        except ValueError as error:
            st.error(f"Comparison failed: {error}")

        else:
            original_column, processed_column = st.columns(2)

            with original_column:
                with st.container(border=True):
                    st.metric(
                        f"Original RMS ({units})",
                        f"{measurements['rms_before']:.4f}",
                    )

            with processed_column:
                with st.container(border=True):
                    st.metric(
                        f"Processed RMS ({units})",
                        f"{measurements['rms_after']:.4f}",
                    )

            st.caption(
                "RMS measures overall amplitude. A lower value can reflect "
                "removed interference or lost useful signal; "
                "inspect the spectra too."
            )

            with st.container(border=True):
                st.subheader("Waveform comparison")

                # Choose the same short preview for both signals.
                preview_count = min(
                    len(original),
                    max(2, int(0.05 * fs_hz)),
                )
                time_ms = np.arange(preview_count) / fs_hz * 1000.0

                st.caption(
                    f"First {preview_count:,} samples · "
                    "edge transients may be visible."
                )

                figure, axis = plt.subplots(figsize=(8, 3.8))

                axis.plot(
                    time_ms,
                    original[:preview_count],
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.4,
                    label="Original",
                )

                axis.plot(
                    time_ms,
                    processed[:preview_count],
                    color="#0F766E",
                    linewidth=1.8,
                    label="Processed",
                )

                style_plot(
                    axis,
                    "Time (ms)",
                    f"Amplitude ({units})",
                )

                axis.set_xlim(0, max(50.0, float(time_ms[-1])))
                axis.legend(frameon=False)
                display_plot(figure)

            with st.container(border=True):
                st.subheader("Spectrum comparison")
                st.caption(
                    "Complete signal · identical FFT scaling "
                    "for both signals."
                )

                figure, axis = plt.subplots(figsize=(8, 3.8))

                axis.plot(
                    original_frequency_hz,
                    original_amplitude,
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.4,
                    label="Original",
                )

                axis.plot(
                    processed_frequency_hz,
                    processed_amplitude,
                    color="#0F766E",
                    linewidth=1.8,
                    label="Processed",
                )

                # Highlight the declared band, not the filter response.
                axis.axvspan(
                    applied_low_hz,
                    applied_high_hz,
                    color="#0F766E",
                    alpha=0.08,
                    label="Declared useful band",
                )

                style_plot(
                    axis,
                    "Frequency (Hz)",
                    f"Estimated amplitude ({units})",
                )

                axis.set_xlim(0, fs_hz / 2)
                axis.legend(frameon=False)
                display_plot(figure)

            if applied["filter_type"] == "notch":
                centre_hz = applied["cutoff_hz"]

                st.info(
                    f"Check that the component near {centre_hz:g} Hz "
                    "decreases while useful components remain "
                    "near their original amplitudes."
                )

            elif applied["filter_type"] == "none":
                st.info(
                    "No filtering was applied. The curves should overlap "
                    "and both RMS values should match."
                )

            else:
                st.info(
                    "Check that candidate interference decreases "
                    "and useful components remain sufficiently preserved."
                )


# ------------------------------------------------------------------
# STAGE 5: EXPORT
# ------------------------------------------------------------------
elif stage == "Export":
    st.header("Download your processed signal")
    st.caption("Save numerical data or prepare an audio download.")

    if "filtered_samples" not in st.session_state:
        st.warning("Go to Configure and apply a filter first.")

    else:
        # Retrieve the processed signal and sampling rate.
        processed = st.session_state["filtered_samples"]
        fs_hz = st.session_state["fs_hz"]
        input_kind = st.session_state.get("input_kind", "wav")

        with st.container(border=True):
            st.subheader("Processed signal data")
            st.caption(
                "Download timestamps and processed amplitudes "
                "without audio scaling."
            )

            # Create sample times in seconds, starting at zero.
            export_time_s = np.arange(len(processed)) / fs_hz

            # Arrange time and amplitude into two columns.
            export_data = np.column_stack(
                (export_time_s, processed)
            )

            # Create an in-memory text buffer for the CSV.
            csv_buffer = StringIO()

            # Use the column names expected by our shared CSV loader.
            # Seventeen significant digits preserve float64 precision.
            np.savetxt(
                csv_buffer,
                export_data,
                delimiter=",",
                header="time_s,amplitude",
                comments="",
                fmt="%.17g",
            )

            # Offer the completed CSV text for download.
            st.download_button(
                label="Download processed CSV",
                data=csv_buffer.getvalue(),
                file_name="processed_signal.csv",
                mime="text/csv",
                key="download_processed_csv",
            )

            st.caption(
                "Amplitudes retain the processing units. "
                "Timestamps start at zero; original timestamp offsets "
                "and amplitude-unit metadata are not included."
            )

        # CSV data may represent physical measurements rather than audio.
        # Ask the user to choose audio conversion explicitly.
        if input_kind == "csv":
            st.info(
                "WAV export converts CSV values to digital audio. "
                "Use the CSV download to retain numerical amplitudes "
                "without audio scaling or 16-bit quantization."
            )

            prepare_audio = st.checkbox(
                "Convert this CSV signal to audio for playback and WAV export",
                key="csv_audio_export",
            )

        else:
            prepare_audio = True

        if prepare_audio:
            try:
                # Prepare the WAV in an isolated temporary folder.
                with TemporaryDirectory() as temporary_folder:
                    output_path = (
                        Path(temporary_folder) / "processed_signal.wav"
                    )

                    # Reuse the shared WAV export algorithm.
                    amplitude_divisor = save_processed_wav(
                        output_path,
                        processed,
                        fs_hz,
                    )

                    # Keep the bytes before the folder is removed.
                    wav_bytes = output_path.read_bytes()

            except (ValueError, OSError) as error:
                st.error(f"Could not prepare the audio: {error}")

            else:
                with st.container(border=True):
                    st.subheader("Your processed audio")
                    st.caption("Mono WAV · 16-bit PCM")

                    rate_column, duration_column = st.columns(2)

                    with rate_column:
                        st.metric("Sampling rate", f"{fs_hz:g} Hz")

                    with duration_column:
                        st.metric(
                            "Duration",
                            f"{len(processed) / fs_hz:.2f} s",
                        )

                    # Play the same WAV offered for download.
                    st.audio(wav_bytes, format="audio/wav")

                    st.download_button(
                        label="Download processed WAV",
                        data=wav_bytes,
                        file_name="processed_signal.wav",
                        mime="audio/wav",
                        key="download_processed_wav",
                    )

                if amplitude_divisor > 1.0:
                    st.info(
                        "For WAV export, sample values were divided by "
                        f"{amplitude_divisor:.4f} to fit the audio range. "
                        "Stored samples, plots, and CSV export are unchanged."
                    )

                else:
                    st.caption(
                        "No amplitude reduction was needed for WAV export."
                    )

                st.caption(
                    "Conversion to 16-bit audio introduces small "
                    "quantization differences. WAV export requires "
                    "a positive whole-number sampling rate in Hz."
                )


# ------------------------------------------------------------------
# WORKFLOW NAVIGATION
# ------------------------------------------------------------------
st.divider()

back_column, continue_column = st.columns(2)

with back_column:
    # Display Back except on the first stage.
    if stage_index > 0:
        previous_stage = stages[stage_index - 1]

        st.button(
            label=f"← Back to {previous_stage}",
            key=f"back_{stage}",
            on_click=select_stage,
            args=(previous_stage,),
        )

with continue_column:
    # Display Continue except on the final stage.
    if stage_index < len(stages) - 1:
        next_stage = stages[stage_index + 1]

        # Check which data is available.
        signal_loaded = "samples" in st.session_state
        result_available = "filtered_samples" in st.session_state

        # Early stages require input; later stages require a result.
        can_continue = (
            signal_loaded and stage in ("Upload", "Inspect")
        ) or (
            result_available and stage in ("Configure", "Compare")
        )

        st.button(
            label=f"Continue to {next_stage} →",
            key=f"next_{stage}",
            type="primary",
            width="stretch",
            disabled=not can_continue,
            on_click=select_stage,
            args=(next_stage,),
        )

        if stage == "Configure" and not can_continue:
            st.caption("Apply the suggested action before continuing.")
