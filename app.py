# Import NumPy for example generation and sample times.
import numpy as np

# Import Streamlit for the web interface.
import streamlit as st

# Import Matplotlib for waveform and spectrum plots.
import matplotlib.pyplot as plt

# Import the shared processing functions.
from signal_lab.validation import validate_signal
from signal_lab.spectral import detect_peaks
from signal_lab.recommendation import recommend_filter
from signal_lab.filters import apply_filter

# Configure the browser-tab title and page width.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Define the workflow stages in order.
stages = ["Upload", "Inspect", "Configure", "Compare", "Export"]

# Initialize navigation for a new session.
if "current_stage" not in st.session_state:
    st.session_state["current_stage"] = "Upload"


# Define the callback for navigation buttons.
def select_stage(stage_name):
    # Remember the selected stage before the page redraws.
    st.session_state["current_stage"] = stage_name


# Define how to remove an outdated processing result.
def clear_processed_result():
    # Remove processed samples if they exist.
    st.session_state.pop("filtered_samples", None)

    # Remove the settings that produced those samples.
    st.session_state.pop("applied_filter", None)

    # Remove the useful band associated with the result.
    st.session_state.pop("applied_band", None)


# Arrange the brand and preview status.
brand_column, status_column = st.columns([4, 1])

# Display the application name and description.
with brand_column:
    st.title("Signal Lab")
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

# Identify this version as a demo workspace.
with status_column:
    st.info("Demo workspace")

# Separate the header from the page content.
st.divider()

# Display workflow navigation in the sidebar.
with st.sidebar:
    st.subheader("Signal Lab")
    st.caption("YOUR WORKSPACE")

    # Create one numbered navigation button per stage.
    for number, stage_name in enumerate(stages, start=1):
        # Determine which button to highlight.
        is_selected = st.session_state["current_stage"] == stage_name

        # Connect each button to the navigation callback.
        st.button(
            f"{number:02d}  ·  {stage_name}",
            key=f"navigation_{stage_name}",
            type="primary" if is_selected else "secondary",
            width="stretch",
            on_click=select_stage,
            args=(stage_name,),
        )

    # Separate navigation from availability information.
    st.divider()
    st.caption("Built-in example available")
    st.caption("Compare and Export are coming next.")

# Retrieve the selected stage.
stage = st.session_state["current_stage"]

# Find its zero-based position in the workflow.
stage_index = stages.index(stage)

# Display the step number using one-based numbering.
st.caption(f"STEP {stage_index + 1:02d} OF {len(stages):02d}")

# Display signal selection.
if stage == "Upload":
    st.header("Start with a signal")
    st.caption("Explore the guided example to get familiar with the lab.")

    # Arrange the recording and example panels side by side.
    upload_column, example_column = st.columns(2)

    # Display the planned upload panel.
    with upload_column:
        with st.container(border=True):
            st.subheader("Your recording")
            st.write("WAV audio or CSV signal measurements.")

            # Keep uploads disabled until loading is connected.
            st.file_uploader(
                "Choose a signal file",
                type=["wav", "csv"],
                disabled=True,
                key="planned_upload",
            )
            st.caption("File uploads will be connected in a later step.")

    # Display the working example panel.
    with example_column:
        with st.container(border=True):
            st.subheader("Try the guided example")
            st.write(
                "A useful 100 Hz tone mixed with an unwanted 400 Hz tone."
            )
            st.caption(
                "5 seconds · 2,000 Hz sampling rate · Dimensionless amplitude"
            )

            # Generate the example when the button is clicked.
            if st.button("Load example", type="primary", width="stretch"):
                # Set sampling rate in hertz and duration in seconds.
                fs_hz = 2000.0
                duration_s = 5.0

                # Calculate the total number of samples.
                sample_count = int(fs_hz * duration_s)

                # Calculate each sample's time in seconds.
                time_s = np.arange(sample_count) / fs_hz

                # Generate the useful tone with dimensionless amplitude 1.
                useful_signal = np.sin(2 * np.pi * 100.0 * time_s)

                # Generate the unwanted tone with dimensionless amplitude 0.5.
                unwanted_signal = 0.5 * np.sin(2 * np.pi * 400.0 * time_s)

                # Validate the combined signal using the shared module.
                samples = validate_signal(
                    useful_signal + unwanted_signal,
                    fs_hz,
                )

                # Preserve the samples and sampling rate between stages.
                st.session_state["samples"] = samples
                st.session_state["fs_hz"] = fs_hz

                # Clear old processing results when the example is reloaded.
                clear_processed_result()

            # Confirm that the example is ready.
            if "samples" in st.session_state:
                st.success("Example ready. Continue to Inspect below.")

# Display signal inspection.
elif stage == "Inspect":
    st.header("Understand your signal")
    st.caption("Inspect the waveform and its detected frequency components.")

    # Handle inspection before loading a signal.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the built-in example first.")

    else:
        # Retrieve the samples and sampling rate in hertz.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]

        # Create three equally sized measurement columns.
        rate_card, duration_card, nyquist_card = st.columns(3)

        # Display the sampling rate.
        with rate_card:
            with st.container(border=True):
                st.metric("Sampling rate", f"{fs_hz:,.0f} Hz")

        # Display the recording duration in seconds.
        with duration_card:
            with st.container(border=True):
                st.metric("Duration", f"{len(samples) / fs_hz:.2f} s")

        # Display half the sampling rate as the Nyquist frequency.
        with nyquist_card:
            with st.container(border=True):
                st.metric("Nyquist frequency", f"{fs_hz / 2:,.0f} Hz")

        # Group the waveform and known demo information.
        with st.container(border=True):
            st.subheader("Signal preview")
            st.caption("Original signal · first 50 milliseconds")

            # Select up to 50 milliseconds of samples.
            preview_count = min(len(samples), int(0.05 * fs_hz))

            # Convert sample positions into milliseconds.
            preview_time_ms = np.arange(preview_count) / fs_hz * 1000

            # Create the taller waveform figure.
            figure, axis = plt.subplots(figsize=(8, 3.8))

            # Plot the original samples in teal.
            axis.plot(
                preview_time_ms,
                samples[:preview_count],
                color="#0F766E",
                linewidth=1.8,
            )

            # Label the waveform axes and their units.
            axis.set_xlabel("Time (ms)")
            axis.set_ylabel("Amplitude (dimensionless)")

            # Display the 0–50 ms interval with a subtle grid.
            axis.set_xlim(0, 50)
            axis.grid(True, alpha=0.15)

            # Remove unnecessary plot borders.
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

            # Fit the labels inside the figure.
            figure.tight_layout()

            # Give the waveform more width than its details.
            plot_column, details_column = st.columns([3, 1])

            # Display the waveform.
            with plot_column:
                st.pyplot(figure)

            # Display the known components used to generate the example.
            with details_column:
                st.markdown("**Known demo tones**")
                st.metric("Useful tone", "100 Hz")
                st.caption("The component we want to preserve.")
                st.metric("Unwanted tone", "400 Hz")
                st.caption("The interference we will aim to reduce.")

            # Release the figure after rendering.
            plt.close(figure)

        # Display automatic frequency analysis.
        with st.container(border=True):
            st.subheader("Frequency analysis")
            st.caption("Calculated from the complete 5-second signal.")

            # Catch expected validation errors.
            try:
                # Call the packaged detector with explicit settings.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

            # Display a readable analysis error.
            except ValueError as error:
                st.error(f"Frequency analysis could not run: {error}")

            else:
                # Select detected frequencies in hertz.
                detected_hz = frequency_hz[peak_indices]

                # Select their estimated dimensionless amplitudes.
                detected_amplitude = amplitude[peak_indices]

                # Create the spectrum figure.
                spectrum_figure, spectrum_axis = plt.subplots(
                    figsize=(8, 3.8)
                )

                # Plot the spectrum returned by the shared detector.
                spectrum_axis.plot(
                    frequency_hz,
                    amplitude,
                    color="#0F766E",
                    linewidth=1.5,
                )

                # Mark the detected peaks.
                spectrum_axis.scatter(
                    detected_hz,
                    detected_amplitude,
                    color="#0F172A",
                    marker="o",
                    s=35,
                    zorder=3,
                    label="Detected peaks",
                )

                # Display the complete one-sided frequency range.
                spectrum_axis.set_xlim(0, fs_hz / 2)

                # Label the spectrum axes with their units.
                spectrum_axis.set_xlabel("Frequency (Hz)")
                spectrum_axis.set_ylabel(
                    "Estimated amplitude (dimensionless)"
                )

                # Add a grid and remove unnecessary borders.
                spectrum_axis.grid(True, alpha=0.15)
                spectrum_axis.spines["top"].set_visible(False)
                spectrum_axis.spines["right"].set_visible(False)

                # Display a legend only when peaks exist.
                if len(peak_indices) > 0:
                    spectrum_axis.legend(frameon=False)

                # Fit the labels and display the spectrum.
                spectrum_figure.tight_layout()
                st.pyplot(spectrum_figure)

                # Release the spectrum figure's memory.
                plt.close(spectrum_figure)

                # Explain when no peaks meet the settings.
                if len(peak_indices) == 0:
                    st.info("No peaks met the current detection settings.")

                else:
                    # Display detected peaks ranked by amplitude.
                    st.dataframe(
                        {
                            "Frequency (Hz)": detected_hz,
                            "Estimated amplitude (dimensionless)":
                                detected_amplitude,
                        },
                        hide_index=True,
                        width="stretch",
                    )

                # Explain the detection settings.
                st.caption(
                    "Minimum height: 10% of the largest spectral amplitude. "
                    "Minimum prominence: 5% of that amplitude. "
                    "Minimum peak separation: 5 Hz."
                )

                # Distinguish frequency detection from identifying noise.
                st.info(
                    "A detected frequency is not automatically noise. "
                    "For this example, 100 Hz is useful and 400 Hz is unwanted."
                )

        # Keep optional learning guidance in a collapsible section.
        with st.expander("How do I read this signal?"):
            st.markdown(
                "**Time (ms):** shows when each sample was recorded. "
                "The waveform preview displays the first 50 milliseconds."
            )
            st.markdown(
                "**Amplitude:** shows the sample value at each instant. "
                "This generated example uses dimensionless values."
            )
            st.markdown(
                "**The shape:** the 100 Hz tone produces five main cycles "
                "in 50 milliseconds. The 400 Hz tone adds faster changes."
            )
            st.markdown(
                "**Sampling rate:** 2,000 Hz means 2,000 samples per second."
            )
            st.markdown(
                "**Nyquist frequency:** half the sampling rate is 1,000 Hz. "
                "The input must be limited to frequencies below this value "
                "before sampling to avoid aliasing."
            )
            st.markdown(
                "**Frequency spectrum:** shows estimated frequency-component "
                "amplitudes. The module removes the mean and applies a Hann "
                "window before calculating the FFT."
            )
            st.caption(
                "The known tone labels describe the generated example. "
                "The frequency table contains calculated detection results."
            )

# Display filter configuration and application.
elif stage == "Configure":
    st.header("Choose what to preserve")
    st.caption("Set the useful band, review the suggestion, and apply it.")

    # Require a loaded signal.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the example first.")

    else:
        # Retrieve the original samples and sampling rate in hertz.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]

        # Calculate the Nyquist boundary in hertz.
        nyquist_hz = fs_hz / 2

        # Initialize persistent useful-band settings for the demo.
        if "useful_band" not in st.session_state:
            st.session_state["useful_band"] = (80.0, 250.0)

        # Retrieve saved boundaries so navigation does not reset them.
        saved_low_hz, saved_high_hz = st.session_state["useful_band"]

        # Group the useful-band controls.
        with st.container(border=True):
            st.subheader("Useful frequency band")
            st.write(
                "Choose the frequency range containing the signal you want "
                "to keep. Peaks outside it become candidates for removal."
            )

            # Arrange the boundaries side by side.
            low_column, high_column = st.columns(2)

            # Read the lower boundary in hertz.
            with low_column:
                useful_low_hz = st.number_input(
                    "Lower frequency (Hz)",
                    min_value=0.0,
                    max_value=float(nyquist_hz),
                    value=float(saved_low_hz),
                    step=5.0,
                    key="useful_low_hz",
                )

            # Read the upper boundary in hertz.
            with high_column:
                useful_high_hz = st.number_input(
                    "Upper frequency (Hz)",
                    min_value=0.0,
                    max_value=float(nyquist_hz),
                    value=float(saved_high_hz),
                    step=5.0,
                    key="useful_high_hz",
                )

            # Explain the demo's starting assumption.
            st.caption(
                "Demo starting point: 80–250 Hz preserves the 100 Hz tone "
                "and places the 400 Hz tone outside the useful band."
            )

        # Record the current boundaries as a tuple.
        current_band = (useful_low_hz, useful_high_hz)

        # Preserve boundaries even when their widgets leave the screen.
        st.session_state["useful_band"] = current_band

        # Clear processed samples if they belong to different settings.
        if st.session_state.get("applied_band") != current_band:
            clear_processed_result()

        # Reject reversed or equal boundaries.
        if useful_low_hz >= useful_high_hz:
            st.error("The lower frequency must be less than the upper frequency.")

        else:
            # Catch validation errors from detection or recommendation.
            try:
                # Detect peaks using the same settings as Inspect.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

                # Select detected frequencies in hertz.
                detected_hz = frequency_hz[peak_indices]

                # Ask the shared module for its recommendation.
                recommendation = recommend_filter(
                    detected_hz,
                    fs_hz,
                    useful_low_hz,
                    useful_high_hz,
                )

            # Display a readable recommendation error.
            except ValueError as error:
                clear_processed_result()
                st.error(f"Could not recommend a filter: {error}")

            else:
                # Translate internal filter names into readable labels.
                filter_labels = {
                    "none": "No filter suggested",
                    "notch": "Notch filter",
                    "lowpass": "Low-pass filter",
                    "highpass": "High-pass filter",
                    "bandpass": "Band-pass filter",
                }

                # Group the suggestion and application controls.
                with st.container(border=True):
                    st.subheader("Suggested filter")

                    # Display the readable filter name.
                    st.markdown(
                        f"**{filter_labels[recommendation['filter_type']]}**"
                    )

                    # Display the explanation from the shared module.
                    st.write(recommendation["reason"])

                    # Retrieve the proposed frequency setting in hertz.
                    cutoff_hz = recommendation["cutoff_hz"]

                    # Display a notch's center frequency.
                    if recommendation["filter_type"] == "notch":
                        st.metric(
                            "Notch center frequency",
                            f"{cutoff_hz:g} Hz",
                        )

                    # Display both edges for a band-pass suggestion.
                    elif recommendation["filter_type"] == "bandpass":
                        st.write(
                            f"Suggested band edges: "
                            f"**{cutoff_hz[0]:g}–{cutoff_hz[1]:g} Hz**"
                        )

                    # Display the cutoff for low-pass or high-pass.
                    elif cutoff_hz is not None:
                        st.metric("Suggested cutoff", f"{cutoff_hz:g} Hz")

                    # Retrieve candidate interference frequencies.
                    candidates = recommendation["candidate_hz"]

                    # Display candidates when any exist.
                    if candidates:
                        st.dataframe(
                            {"Candidate interference frequency (Hz)": candidates},
                            hide_index=True,
                            width="stretch",
                        )

                    # Explain the limits of the suggestion.
                    st.caption(
                        "This suggestion depends on your useful-band choice. "
                        "It does not prove that a component is noise."
                    )

                    # Explain the fixed settings used for this step.
                    st.caption(
                        "Processing defaults: Butterworth design order 4 "
                        "for low-pass, high-pass, or band-pass; "
                        "Q = 30 for a second-order notch."
                    )

                    # Identify the processing method and retained input.
                    st.caption(
                        "Offline forward-backward filtering uses the complete "
                        "recording. The original samples remain available."
                    )

                    # Choose the appropriate action label.
                    action_label = (
                        "Keep signal unchanged"
                        if recommendation["filter_type"] == "none"
                        else "Apply suggested filter"
                    )

                    # Process only when the user clicks the button.
                    if st.button(
                        action_label,
                        key="apply_suggestion",
                        type="primary",
                        width="stretch",
                    ):
                        # Remove any previous result before a new attempt.
                        clear_processed_result()

                        # Catch errors from the packaged filter.
                        try:
                            # Always process the original samples.
                            filtered_samples = apply_filter(
                                samples,
                                fs_hz,
                                filter_type=recommendation["filter_type"],
                                cutoff_hz=recommendation["cutoff_hz"],
                                order=4,
                                notch_q=30.0,
                            )

                        # Display a readable filtering error.
                        except ValueError as error:
                            st.error(f"Could not process the signal: {error}")

                        else:
                            # Store processed samples for later comparison.
                            st.session_state["filtered_samples"] = (
                                filtered_samples
                            )

                            # Store the settings that produced the result.
                            st.session_state["applied_filter"] = (
                                recommendation.copy()
                            )

                            # Store the useful band used for this result.
                            st.session_state["applied_band"] = current_band

                    # Display confirmation when a processed result exists.
                    if "filtered_samples" in st.session_state:
                        # Retrieve the completed operation's settings.
                        applied_filter = st.session_state["applied_filter"]

                        # Describe an unchanged copy accurately.
                        if applied_filter["filter_type"] == "none":
                            st.success(
                                "Signal kept unchanged. "
                                "Result saved for comparison."
                            )

                        # Confirm filtering without claiming improved quality.
                        else:
                            st.success(
                                "Filter applied. "
                                "Processed signal saved for comparison."
                            )

                        # Display the number of processed samples.
                        st.caption(
                            f"Processed samples: "
                            f"{len(st.session_state['filtered_samples']):,}"
                        )

                    # Explain what to do before processing.
                    else:
                        st.info("Review the suggestion, then apply it above.")

# Display placeholders for Compare and Export.
else:
    st.header(stage)
    st.info(
        "This stage is coming next. Use Upload, Inspect, and Configure for now."
    )

# Separate page content from workflow navigation.
st.divider()

# Create columns for Back and Continue.
back_column, next_column = st.columns(2)

# Show backward navigation when an earlier stage exists.
with back_column:
    if stage_index > 0:
        # Retrieve the previous stage.
        previous_stage = stages[stage_index - 1]

        # Connect the Back button to the navigation callback.
        st.button(
            f"← Back to {previous_stage}",
            key="back_stage",
            on_click=select_stage,
            args=(previous_stage,),
        )

# Show forward navigation when another stage exists.
with next_column:
    if stage_index < len(stages) - 1:
        # Retrieve the next stage.
        next_stage = stages[stage_index + 1]

        # Check whether a signal has been loaded.
        signal_loaded = "samples" in st.session_state

        # Enable forward navigation through Upload and Inspect.
        can_continue = signal_loaded and stage in ("Upload", "Inspect")

        # Keep later Continue buttons disabled until their pages are connected.
        st.button(
            f"Continue to {next_stage} →",
            key="next_stage",
            type="primary",
            width="stretch",
            disabled=not can_continue,
            on_click=select_stage,
            args=(next_stage,),
        )

        # Explain why later Continue buttons are disabled.
        if stage not in ("Upload", "Inspect"):
            st.caption("The next processing stage is still being built.")
