"""Streamlit interface for the Signal Processing Lab."""

# Import NumPy for generating the guided example and time labels.
import numpy as np

# Import Streamlit for building the web interface.
import streamlit as st

# Import Matplotlib for plotting signals and spectra.
import matplotlib.pyplot as plt

# Create temporary folders for preparing WAV downloads.
from tempfile import TemporaryDirectory

# Build file paths and read the completed WAV file.
from pathlib import Path

# Import the same processing functions used by our teaching notebook.
from signal_lab.validation import validate_signal
from signal_lab.spectral import calculate_fft, detect_peaks
from signal_lab.recommendation import recommend_filter
from signal_lab.filters import apply_filter
from signal_lab.metrics import compare_rms

# Load WAV recordings and export processed WAV audio.
from signal_lab.data_io import load_wav, save_processed_wav


# Configure the browser title and use the available page width.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Define the stages in their intended order.
stages = ["Upload", "Inspect", "Configure", "Compare", "Export"]

# Start at Upload when this browser session has no selected stage.
if "current_stage" not in st.session_state:
    st.session_state["current_stage"] = "Upload"


def select_stage(stage_name):
    """Remember the stage selected by a navigation button."""

    # Streamlit reruns the page after this callback updates the stage.
    st.session_state["current_stage"] = stage_name


def clear_processed_result():
    """Remove a result when its input or configuration changes."""

    # Remove stored processed samples, if present.
    st.session_state.pop("filtered_samples", None)

    # Remove the settings associated with the previous result.
    st.session_state.pop("applied_filter", None)

    # Remove the useful band associated with the previous result.
    st.session_state.pop("applied_band", None)


def style_plot(axis, x_label, y_label):
    """Apply consistent presentation settings to each plot."""

    # Label the horizontal axis, including its units.
    axis.set_xlabel(x_label)

    # Label the vertical axis, including its units.
    axis.set_ylabel(y_label)

    # Add a faint grid to make values easier to read.
    axis.grid(alpha=0.15)

    # Hide unnecessary borders around the plot.
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


# Divide the page header into brand and status columns.
brand_column, status_column = st.columns([4, 1])

with brand_column:
    st.title("Signal Lab")
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

with status_column:
    st.info("Demo workspace")

st.divider()


# Build the sidebar navigation.
with st.sidebar:
    st.title("Signal Lab")
    st.caption("YOUR WORKSPACE")

    # Create a navigation button for each stage.
    for number, stage_name in enumerate(stages, start=1):
        # Identify the selected stage so its button can be highlighted.
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
    st.caption("Built-in example available.")
    st.caption("Processed WAV playback and download available.")


# Retrieve the current stage and its position in the workflow.
stage = st.session_state["current_stage"]
stage_index = stages.index(stage)

st.caption(f"STEP {stage_index + 1:02d} OF {len(stages):02d}")


# ------------------------------------------------------------------
# STAGE 1: UPLOAD
# ------------------------------------------------------------------
if stage == "Upload":
    st.header("Start with a signal")
    st.caption("Use the guided example to explore the complete workflow.")

    upload_column, example_column = st.columns(2)

    with upload_column:
        with st.container(border=True):
            st.subheader("Your recording")
            st.write("Upload a WAV recording or CSV signal.")

            # Keep uploads disabled until we connect the loading workflow.
            st.file_uploader(
                "Choose a signal file",
                type=["wav", "csv"],
                disabled=True,
                key="planned_upload",
            )

            st.caption("File uploads will be connected in a later step.")

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
                type="primary",
                width="stretch",
            ):
                # Set the sampling rate in samples per second, or Hz.
                fs_hz = 2000.0

                # Set the recording duration in seconds.
                duration_s = 5.0

                # Calculate the total number of samples.
                sample_count = int(fs_hz * duration_s)

                # Calculate each sample's time in seconds.
                time_s = np.arange(sample_count) / fs_hz

                # Generate the useful tone with amplitude 1.
                useful_signal = np.sin(
                    2 * np.pi * 100.0 * time_s
                )

                # Generate the unwanted tone with amplitude 0.5.
                unwanted_signal = 0.5 * np.sin(
                    2 * np.pi * 400.0 * time_s
                )

                # Validate the combined signal with our shared function.
                samples = validate_signal(
                    useful_signal + unwanted_signal,
                    fs_hz,
                )

                # Store the input so it survives navigation.
                st.session_state["samples"] = samples
                st.session_state["fs_hz"] = fs_hz

                # A new input requires a new processed result.
                clear_processed_result()

    if "samples" in st.session_state:
        st.success("Example ready. Continue to Inspect below.")


# ------------------------------------------------------------------
# STAGE 2: INSPECT
# ------------------------------------------------------------------
elif stage == "Inspect":
    st.header("Understand your signal")
    st.caption("Inspect the waveform and its detected frequencies.")

    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the example first.")

    else:
        # Retrieve the original samples and sampling rate.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]

        # Display three measurements side by side.
        rate_column, duration_column, nyquist_column = st.columns(3)

        with rate_column:
            with st.container(border=True):
                st.metric("Sampling rate", f"{fs_hz:,.0f} Hz")

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
                    f"{fs_hz / 2:,.0f} Hz",
                )

        with st.container(border=True):
            st.subheader("Signal preview")
            st.caption("Original signal · first 50 milliseconds")

            # Limit the preview to the first 0.05 seconds.
            preview_count = min(
                len(samples),
                int(0.05 * fs_hz),
            )

            # Convert sample positions into time in milliseconds.
            time_ms = (
                np.arange(preview_count) / fs_hz
            ) * 1000.0

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
                "Amplitude (dimensionless)",
            )

            axis.set_xlim(0, 50)
            figure.tight_layout()

            plot_column, details_column = st.columns([3, 1])

            with plot_column:
                st.pyplot(figure)

            with details_column:
                st.markdown("**Known demo tones**")
                st.metric("Useful tone", "100 Hz")
                st.metric("Unwanted tone", "400 Hz")
                st.caption(
                    "These roles are known because we generated the example."
                )

            # Release the figure after displaying it.
            plt.close(figure)

        with st.container(border=True):
            st.subheader("Frequency analysis")
            st.caption("FFT of the complete recording")

            try:
                # Use our shared peak detector and FFT calculation.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

            except ValueError as error:
                st.error(f"Frequency analysis failed: {error}")

            else:
                # Select the detected frequencies and their amplitudes.
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
                    "Estimated amplitude (dimensionless)",
                )

                axis.set_xlim(0, fs_hz / 2)

                if len(peak_indices) > 0:
                    axis.legend(frameon=False)

                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

                if len(peak_indices) == 0:
                    st.info(
                        "No prominent peaks found with these settings."
                    )
                else:
                    st.dataframe(
                        {
                            "Frequency (Hz)": detected_hz,
                            "Estimated amplitude (dimensionless)": (
                                detected_amplitude
                            ),
                        },
                        hide_index=True,
                        width="stretch",
                    )

                st.caption(
                    "Minimum height: 10% of the largest FFT amplitude. "
                    "Minimum prominence: 5%. "
                    "Minimum peak separation: 5 Hz."
                )

        st.info(
            "A detected frequency is not automatically noise. "
            "In this example, we know that 100 Hz is useful "
            "and 400 Hz is unwanted."
        )

        with st.expander("How do I read this signal?"):
            st.write(
                "The waveform shows how amplitude changes over time. "
                "This preview displays only the first 50 milliseconds."
            )

            st.write(
                "Amplitude is dimensionless because this is a generated "
                "digital signal."
            )

            st.write(
                "The 100 Hz tone completes five cycles in 50 milliseconds. "
                "The 400 Hz tone adds faster changes to the waveform."
            )

            st.write(
                "The sampling rate is 2,000 samples per second. "
                "Its Nyquist frequency is 1,000 Hz. Frequencies above "
                "that limit require appropriate filtering before sampling "
                "to prevent aliasing."
            )

            st.write(
                "Our FFT function removes the mean for analysis, "
                "applies a Hann window, and compensates for its "
                "amplitude scaling."
            )


# ------------------------------------------------------------------
# STAGE 3: CONFIGURE
# ------------------------------------------------------------------
elif stage == "Configure":
    st.header("Choose what to preserve")
    st.caption("Set the useful band, review the suggestion, and apply it.")

    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the example first.")

    else:
        # Retrieve the input and calculate its Nyquist frequency.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]
        nyquist_hz = fs_hz / 2

        # Remember the useful band separately from the input widgets.
        if "useful_band" not in st.session_state:
            st.session_state["useful_band"] = (80.0, 250.0)

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

            st.caption(
                "For the guided example, use 80–250 Hz "
                "to preserve the 100 Hz tone."
            )

        # Store the currently declared useful band.
        current_band = (useful_low_hz, useful_high_hz)
        st.session_state["useful_band"] = current_band

        # Remove a processed result if its useful band has changed.
        if st.session_state.get("applied_band") != current_band:
            clear_processed_result()

        if useful_low_hz >= useful_high_hz:
            st.error(
                "The lower frequency must be below the upper frequency."
            )

        else:
            try:
                # Detect prominent frequencies with the shared detector.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

                # Extract the frequencies of the detected peaks.
                detected_hz = frequency_hz[peak_indices]

                # Ask the shared recommendation function for a candidate.
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
                # Convert internal names into readable interface labels.
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
                        # Clear any previous result before applying settings.
                        clear_processed_result()

                        try:
                            # Apply the shared processing function.
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
        # Retrieve the original, result, and sampling rate.
        original = st.session_state["samples"]
        processed = st.session_state["filtered_samples"]
        fs_hz = st.session_state["fs_hz"]

        # Retrieve the settings associated with this processed result.
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
            # Calculate RMS using our shared comparison function.
            measurements = compare_rms(original, processed)

            # Use the same FFT function and scaling for both signals.
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
                        "Original RMS (dimensionless)",
                        f"{measurements['rms_before']:.4f}",
                    )

            with processed_column:
                with st.container(border=True):
                    st.metric(
                        "Processed RMS (dimensionless)",
                        f"{measurements['rms_after']:.4f}",
                    )

            st.caption(
                "RMS measures overall amplitude. A lower value can reflect "
                "removed interference or lost useful signal; "
                "inspect the spectra too."
            )

            with st.container(border=True):
                st.subheader("Waveform comparison")
                st.caption(
                    "First 50 milliseconds · edge transients may be visible."
                )

                # Select the first 50 milliseconds of both signals.
                preview_count = min(
                    len(original),
                    int(0.05 * fs_hz),
                )

                # Calculate the preview's time labels in milliseconds.
                time_ms = (
                    np.arange(preview_count) / fs_hz
                ) * 1000.0

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
                    "Amplitude (dimensionless)",
                )

                axis.set_xlim(0, 50)
                axis.legend(frameon=False)
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

            with st.container(border=True):
                st.subheader("Spectrum comparison")
                st.caption(
                    "Complete recording · identical FFT scaling "
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

                # Highlight the user's useful band, not the filter response.
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
                    "Estimated amplitude (dimensionless)",
                )

                axis.set_xlim(0, fs_hz / 2)
                axis.legend(frameon=False)
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

            if applied["filter_type"] == "notch":
                # Display the actual applied notch centre.
                centre_hz = applied["cutoff_hz"]

                st.info(
                    f"Check that the component near {centre_hz:g} Hz "
                    "decreases while the frequencies you want to keep "
                    "remain near their original amplitudes."
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
    st.caption("Listen to the result, then save it as a WAV file.")

    if "filtered_samples" not in st.session_state:
        st.warning("Go to Configure and apply a filter first.")

    else:
        # Retrieve the processed samples without modifying them.
        processed = st.session_state["filtered_samples"]

        # Retrieve the sampling rate in hertz.
        fs_hz = st.session_state["fs_hz"]

        try:
            # Create an isolated temporary folder for this export.
            # Python removes the folder when the block finishes.
            with TemporaryDirectory() as temporary_folder:
                # Build the path for the temporary WAV file.
                output_path = (
                    Path(temporary_folder) / "processed_signal.wav"
                )

                # Save audio using the shared export function.
                # Its return value reports any amplitude reduction.
                amplitude_divisor = save_processed_wav(
                    output_path,
                    processed,
                    fs_hz,
                )

                # Read the WAV into memory before deleting its folder.
                wav_bytes = output_path.read_bytes()

        except (ValueError, OSError) as error:
            st.error(f"Could not prepare the audio: {error}")

        else:
            with st.container(border=True):
                st.subheader("Your processed audio")
                st.caption("Mono WAV · 16-bit PCM")

                # Display the exported recording's rate and duration.
                rate_column, duration_column = st.columns(2)

                with rate_column:
                    st.metric("Sampling rate", f"{fs_hz:,.0f} Hz")

                with duration_column:
                    st.metric(
                        "Duration",
                        f"{len(processed) / fs_hz:.2f} s",
                    )

                # Play the same WAV bytes used by the download button.
                st.audio(wav_bytes, format="audio/wav")

                # Download the prepared WAV to the user's computer.
                st.download_button(
                    label="Download processed WAV",
                    data=wav_bytes,
                    file_name="processed_signal.wav",
                    mime="audio/wav",
                    key="download_processed_wav",
                )

            # Explain whether the export function reduced amplitudes.
            if amplitude_divisor > 1.0:
                st.info(
                    "For export, amplitudes were divided by "
                    f"{amplitude_divisor:.4f} to fit the audio range. "
                    "Your stored samples and comparison plots are unchanged."
                )
            else:
                st.caption(
                    "No amplitude reduction was needed for export."
                )

            st.caption(
                "The download converts floating-point samples to "
                "16-bit audio, which introduces small quantization "
                "differences."
            )


# ------------------------------------------------------------------
# WORKFLOW NAVIGATION
# ------------------------------------------------------------------
st.divider()

back_column, continue_column = st.columns(2)

with back_column:
    # The first stage has no previous stage.
    if stage_index > 0:
        previous_stage = stages[stage_index - 1]

        st.button(
            label=f"← Back to {previous_stage}",
            key=f"back_{stage}",
            on_click=select_stage,
            args=(previous_stage,),
        )

with continue_column:
    # Export is the final stage, so it needs no Continue button.
    if stage_index < len(stages) - 1:
        next_stage = stages[stage_index + 1]

        # Check whether this session has an input and processed result.
        signal_loaded = "samples" in st.session_state
        result_available = "filtered_samples" in st.session_state

        # Upload and Inspect need an input.
        # Configure and Compare need a processed result.
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
