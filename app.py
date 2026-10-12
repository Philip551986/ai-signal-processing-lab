# Import NumPy for example generation and sample times.
import numpy as np

# Import Streamlit for the interface.
import streamlit as st

# Import Matplotlib for signal plots.
import matplotlib.pyplot as plt

# Import shared processing functions instead of duplicating their algorithms.
from signal_lab.validation import validate_signal
from signal_lab.spectral import detect_peaks, calculate_fft
from signal_lab.recommendation import recommend_filter
from signal_lab.filters import apply_filter
from signal_lab.metrics import compare_rms

# Configure the browser-tab title and page width.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Define the workflow order.
stages = ["Upload", "Inspect", "Configure", "Compare", "Export"]

# Initialize navigation for a new session.
if "current_stage" not in st.session_state:
    st.session_state["current_stage"] = "Upload"


# Define the navigation callback.
def select_stage(stage_name):
    # Remember the selected stage before the page redraws.
    st.session_state["current_stage"] = stage_name


# Define how to clear outdated processing results.
def clear_processed_result():
    # Safely remove processed samples and their associated settings.
    st.session_state.pop("filtered_samples", None)
    st.session_state.pop("applied_filter", None)
    st.session_state.pop("applied_band", None)


# Define a presentation helper for consistent chart formatting.
def style_plot(axis, x_label, y_label):
    # Label the plotted quantities and their units.
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)

    # Add a subtle grid and remove unnecessary borders.
    axis.grid(True, alpha=0.15)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


# Arrange the application brand and preview status.
brand_column, status_column = st.columns([4, 1])

# Display the brand.
with brand_column:
    st.title("Signal Lab")
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

# Display the workspace status.
with status_column:
    st.info("Demo workspace")

# Separate the header from the current page.
st.divider()

# Place workflow navigation in the sidebar.
with st.sidebar:
    st.subheader("Signal Lab")
    st.caption("YOUR WORKSPACE")

    # Number the stages starting from one.
    for number, stage_name in enumerate(stages, start=1):
        # Determine which stage to highlight.
        is_selected = st.session_state["current_stage"] == stage_name

        # Create a navigation button with a unique key.
        st.button(
            f"{number:02d}  ·  {stage_name}",
            key=f"navigation_{stage_name}",
            type="primary" if is_selected else "secondary",
            width="stretch",
            on_click=select_stage,
            args=(stage_name,),
        )

    # Display current feature availability.
    st.divider()
    st.caption("Built-in example available")
    st.caption("Export is coming next.")

# Retrieve the current stage and its zero-based position.
stage = st.session_state["current_stage"]
stage_index = stages.index(stage)

# Display the human-readable step number.
st.caption(f"STEP {stage_index + 1:02d} OF {len(stages):02d}")

# Display signal selection.
if stage == "Upload":
    st.header("Start with a signal")
    st.caption("Explore the guided example to get familiar with the lab.")

    # Create side-by-side recording and example panels.
    upload_column, example_column = st.columns(2)

    # Display the planned upload panel.
    with upload_column:
        with st.container(border=True):
            st.subheader("Your recording")
            st.write("WAV audio or CSV signal measurements.")

            # Keep uploads disabled until their loading logic is connected.
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

                # Calculate sample count and sample times in seconds.
                sample_count = int(fs_hz * duration_s)
                time_s = np.arange(sample_count) / fs_hz

                # Generate the useful tone with dimensionless amplitude 1.
                useful_signal = np.sin(2 * np.pi * 100.0 * time_s)

                # Generate the unwanted tone with dimensionless amplitude 0.5.
                unwanted_signal = 0.5 * np.sin(2 * np.pi * 400.0 * time_s)

                # Validate the combined signal.
                samples = validate_signal(
                    useful_signal + unwanted_signal,
                    fs_hz,
                )

                # Store the original samples and sampling rate.
                st.session_state["samples"] = samples
                st.session_state["fs_hz"] = fs_hz

                # Clear results belonging to a previously loaded signal.
                clear_processed_result()

            # Confirm that an example is loaded.
            if "samples" in st.session_state:
                st.success("Example ready. Continue to Inspect below.")

# Display inspection of the original signal.
elif stage == "Inspect":
    st.header("Understand your signal")
    st.caption("Inspect the waveform and its detected frequency components.")

    # Handle an empty session.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the built-in example first.")

    else:
        # Retrieve the original samples and sampling rate in hertz.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]

        # Arrange measurement cards in three equal columns.
        rate_card, duration_card, nyquist_card = st.columns(3)

        # Display the sampling rate.
        with rate_card:
            with st.container(border=True):
                st.metric("Sampling rate", f"{fs_hz:,.0f} Hz")

        # Display duration in seconds.
        with duration_card:
            with st.container(border=True):
                st.metric("Duration", f"{len(samples) / fs_hz:.2f} s")

        # Display the Nyquist frequency in hertz.
        with nyquist_card:
            with st.container(border=True):
                st.metric("Nyquist frequency", f"{fs_hz / 2:,.0f} Hz")

        # Display the waveform and known example details.
        with st.container(border=True):
            st.subheader("Signal preview")
            st.caption("Original signal · first 50 milliseconds")

            # Select up to 50 milliseconds of samples.
            preview_count = min(len(samples), int(0.05 * fs_hz))

            # Convert sample positions into milliseconds.
            preview_time_ms = np.arange(preview_count) / fs_hz * 1000

            # Create the waveform figure.
            figure, axis = plt.subplots(figsize=(8, 3.8))

            # Plot actual samples without smoothing.
            axis.plot(
                preview_time_ms,
                samples[:preview_count],
                color="#0F766E",
                linewidth=1.8,
            )

            # Apply consistent axis labels and styling.
            style_plot(axis, "Time (ms)", "Amplitude (dimensionless)")
            axis.set_xlim(0, 50)
            figure.tight_layout()

            # Give the plot more width than the explanatory panel.
            plot_column, details_column = st.columns([3, 1])

            # Render the waveform.
            with plot_column:
                st.pyplot(figure)

            # Identify the known tones used to generate this example.
            with details_column:
                st.markdown("**Known demo tones**")
                st.metric("Useful tone", "100 Hz")
                st.caption("The component we want to preserve.")
                st.metric("Unwanted tone", "400 Hz")
                st.caption("The interference we will aim to reduce.")

            # Release the figure after rendering.
            plt.close(figure)

        # Display calculated frequency analysis.
        with st.container(border=True):
            st.subheader("Frequency analysis")
            st.caption("Calculated from the complete 5-second signal.")

            # Catch expected validation errors.
            try:
                # Detect peaks using the packaged function.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

            # Show a readable error if analysis fails.
            except ValueError as error:
                st.error(f"Frequency analysis could not run: {error}")

            else:
                # Select detected frequencies and dimensionless amplitudes.
                detected_hz = frequency_hz[peak_indices]
                detected_amplitude = amplitude[peak_indices]

                # Create the spectrum figure.
                figure, axis = plt.subplots(figsize=(8, 3.8))

                # Plot the spectrum returned by the shared detector.
                axis.plot(
                    frequency_hz,
                    amplitude,
                    color="#0F766E",
                    linewidth=1.5,
                )

                # Mark detected peaks with circles.
                axis.scatter(
                    detected_hz,
                    detected_amplitude,
                    color="#0F172A",
                    s=35,
                    zorder=3,
                    label="Detected peaks",
                )

                # Label and style the spectrum.
                style_plot(
                    axis,
                    "Frequency (Hz)",
                    "Estimated amplitude (dimensionless)",
                )
                axis.set_xlim(0, fs_hz / 2)

                # Show the legend only when peaks exist.
                if len(peak_indices) > 0:
                    axis.legend(frameon=False)

                # Render and release the spectrum figure.
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

                # Display either an empty-state message or the detected table.
                if len(peak_indices) == 0:
                    st.info("No peaks met the current detection settings.")

                else:
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

                # Distinguish detected frequencies from proven noise.
                st.info(
                    "A detected frequency is not automatically noise. "
                    "For this example, 100 Hz is useful and 400 Hz is unwanted."
                )

        # Provide optional learning guidance.
        with st.expander("How do I read this signal?"):
            st.markdown(
                "**Time (ms):** shows when each sample was recorded. "
                "The waveform preview displays the first 50 milliseconds."
            )
            st.markdown(
                "**Amplitude:** shows the sample value at each instant. "
                "This example uses dimensionless values."
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
                "Known tone labels describe the generated example. "
                "The frequency table contains calculated detection results."
            )

# Display filter configuration and application.
elif stage == "Configure":
    st.header("Choose what to preserve")
    st.caption("Set the useful band, review the suggestion, and apply it.")

    # Require original samples before configuring a filter.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the example first.")

    else:
        # Retrieve the original samples and sampling rate.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]

        # Calculate the Nyquist boundary in hertz.
        nyquist_hz = fs_hz / 2

        # Initialize persistent useful-band settings.
        if "useful_band" not in st.session_state:
            st.session_state["useful_band"] = (80.0, 250.0)

        # Retrieve saved settings so page navigation does not reset them.
        saved_low_hz, saved_high_hz = st.session_state["useful_band"]

        # Display useful-band controls.
        with st.container(border=True):
            st.subheader("Useful frequency band")
            st.write(
                "Choose the range containing the signal you want to keep. "
                "Peaks outside it become candidates for removal."
            )

            # Arrange the lower and upper controls side by side.
            low_column, high_column = st.columns(2)

            # Read the lower frequency boundary in hertz.
            with low_column:
                useful_low_hz = st.number_input(
                    "Lower frequency (Hz)",
                    min_value=0.0,
                    max_value=float(nyquist_hz),
                    value=float(saved_low_hz),
                    step=5.0,
                    key="useful_low_hz",
                )

            # Read the upper frequency boundary in hertz.
            with high_column:
                useful_high_hz = st.number_input(
                    "Upper frequency (Hz)",
                    min_value=0.0,
                    max_value=float(nyquist_hz),
                    value=float(saved_high_hz),
                    step=5.0,
                    key="useful_high_hz",
                )

            # Explain the demo's starting settings.
            st.caption(
                "Demo starting point: 80–250 Hz preserves the 100 Hz tone "
                "and places the 400 Hz tone outside the useful band."
            )

        # Save current settings separately from temporary widget state.
        current_band = (useful_low_hz, useful_high_hz)
        st.session_state["useful_band"] = current_band

        # Clear results produced with different useful-band settings.
        if st.session_state.get("applied_band") != current_band:
            clear_processed_result()

        # Reject equal or reversed boundaries.
        if useful_low_hz >= useful_high_hz:
            st.error("The lower frequency must be less than the upper frequency.")

        else:
            # Catch validation errors from analysis and recommendation.
            try:
                # Use the same peak-detection settings as Inspect.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

                # Select detected peak frequencies in hertz.
                detected_hz = frequency_hz[peak_indices]

                # Request a recommendation from the shared module.
                recommendation = recommend_filter(
                    detected_hz,
                    fs_hz,
                    useful_low_hz,
                    useful_high_hz,
                )

            # Clear obsolete results and report the error.
            except ValueError as error:
                clear_processed_result()
                st.error(f"Could not recommend a filter: {error}")

            else:
                # Translate internal names into readable filter labels.
                filter_labels = {
                    "none": "No filter suggested",
                    "notch": "Notch filter",
                    "lowpass": "Low-pass filter",
                    "highpass": "High-pass filter",
                    "bandpass": "Band-pass filter",
                }

                # Display the suggestion and processing controls.
                with st.container(border=True):
                    st.subheader("Suggested filter")

                    # Display the suggested filter name and decision explanation.
                    st.markdown(
                        f"**{filter_labels[recommendation['filter_type']]}**"
                    )
                    st.write(recommendation["reason"])

                    # Retrieve the suggested frequency setting in hertz.
                    cutoff_hz = recommendation["cutoff_hz"]

                    # Display the notch center when a notch is suggested.
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

                    # Display the single low-pass or high-pass cutoff.
                    elif cutoff_hz is not None:
                        st.metric("Suggested cutoff", f"{cutoff_hz:g} Hz")

                    # Retrieve candidate interference frequencies.
                    candidates = recommendation["candidate_hz"]

                    # Display candidates when they exist.
                    if candidates:
                        st.dataframe(
                            {"Candidate interference frequency (Hz)": candidates},
                            hide_index=True,
                            width="stretch",
                        )

                    # Explain the recommendation's assumption.
                    st.caption(
                        "The suggestion depends on your useful-band choice. "
                        "It does not prove that a component is noise."
                    )

                    # Describe the fixed processing settings.
                    st.caption(
                        "Defaults: Butterworth design order 4 for low-pass, "
                        "high-pass, or band-pass; Q = 30 for a second-order notch."
                    )
                    st.caption(
                        "Offline forward-backward filtering uses the complete "
                        "recording. The original samples remain available."
                    )

                    # Choose a label appropriate to the suggested operation.
                    action_label = (
                        "Keep signal unchanged"
                        if recommendation["filter_type"] == "none"
                        else "Apply suggested filter"
                    )

                    # Process only after an explicit button click.
                    if st.button(
                        action_label,
                        key="apply_suggestion",
                        type="primary",
                        width="stretch",
                    ):
                        # Clear the previous result before attempting processing.
                        clear_processed_result()

                        # Catch errors reported by the packaged filter.
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

                        # Display a readable processing error.
                        except ValueError as error:
                            st.error(f"Could not process the signal: {error}")

                        else:
                            # Store the processed samples and applied settings.
                            st.session_state["filtered_samples"] = (
                                filtered_samples
                            )
                            st.session_state["applied_filter"] = (
                                recommendation.copy()
                            )
                            st.session_state["applied_band"] = current_band

                    # Confirm a completed operation.
                    if "filtered_samples" in st.session_state:
                        # Retrieve the operation associated with the saved result.
                        applied_filter = st.session_state["applied_filter"]

                        # Distinguish an unchanged copy from filtering.
                        if applied_filter["filter_type"] == "none":
                            st.success(
                                "Signal kept unchanged. "
                                "Result saved for comparison."
                            )
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

                    # Give guidance when no processed result exists.
                    else:
                        st.info("Review the suggestion, then apply it above.")

# Display before-and-after comparison.
elif stage == "Compare":
    st.header("See what changed")
    st.caption("Compare the original signal with the processed result.")

    # Require a completed processing operation.
    if "filtered_samples" not in st.session_state:
        st.warning("Go to Configure and apply the suggestion first.")

    else:
        # Retrieve original and processed dimensionless samples.
        original = st.session_state["samples"]
        filtered = st.session_state["filtered_samples"]

        # Retrieve sampling rate and the settings used for this result.
        fs_hz = st.session_state["fs_hz"]
        applied_filter = st.session_state["applied_filter"]
        applied_low_hz, applied_high_hz = st.session_state["applied_band"]

        # Describe the completed operation.
        st.caption(
            f"Applied operation: {applied_filter['filter_type']} · "
            f"Declared useful band: {applied_low_hz:g}–{applied_high_hz:g} Hz"
        )

        # Catch validation errors from the shared comparison functions.
        try:
            # Measure RMS using the packaged function.
            measurements = compare_rms(original, filtered)

            # Calculate both spectra using identical FFT processing.
            before_frequency_hz, before_amplitude = calculate_fft(
                original, fs_hz
            )
            after_frequency_hz, after_amplitude = calculate_fft(
                filtered, fs_hz
            )

        # Display a readable error if comparison fails.
        except ValueError as error:
            st.error(f"Could not compare the signals: {error}")

        else:
            # Create side-by-side RMS cards.
            before_card, after_card = st.columns(2)

            # Display original RMS in the signal's amplitude units.
            with before_card:
                with st.container(border=True):
                    st.metric(
                        "Original RMS (dimensionless)",
                        f"{measurements['rms_before']:.4f}",
                    )

            # Display processed RMS in the same units.
            with after_card:
                with st.container(border=True):
                    st.metric(
                        "Processed RMS (dimensionless)",
                        f"{measurements['rms_after']:.4f}",
                    )

            # Explain why lower RMS alone does not establish better quality.
            st.caption(
                "RMS measures overall amplitude. A lower value can reflect "
                "removed interference or lost useful signal; inspect the spectra too."
            )

            # Display a waveform overlay using shared axes.
            with st.container(border=True):
                st.subheader("Waveform comparison")
                st.caption(
                    "First 50 milliseconds · edge transients may be visible."
                )

                # Select matching samples from both signals.
                preview_count = min(len(original), int(0.05 * fs_hz))

                # Calculate their times in milliseconds.
                preview_time_ms = np.arange(preview_count) / fs_hz * 1000

                # Create the waveform figure.
                figure, axis = plt.subplots(figsize=(8, 3.8))

                # Draw the original waveform as a dashed grey line.
                axis.plot(
                    preview_time_ms,
                    original[:preview_count],
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.4,
                    label="Original",
                )

                # Draw the processed waveform as a solid teal line.
                axis.plot(
                    preview_time_ms,
                    filtered[:preview_count],
                    color="#0F766E",
                    linewidth=1.8,
                    label="Processed",
                )

                # Label and style the shared axes.
                style_plot(axis, "Time (ms)", "Amplitude (dimensionless)")
                axis.set_xlim(0, 50)
                axis.legend(frameon=False)

                # Render the overlay and release its figure.
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

            # Display both spectra using shared axes and amplitude scaling.
            with st.container(border=True):
                st.subheader("Spectrum comparison")
                st.caption(
                    "Complete recording · identical FFT scaling for both signals."
                )

                # Create the spectrum figure.
                figure, axis = plt.subplots(figsize=(8, 3.8))

                # Draw the original spectrum as a dashed grey line.
                axis.plot(
                    before_frequency_hz,
                    before_amplitude,
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.4,
                    label="Original",
                )

                # Draw the processed spectrum as a solid teal line.
                axis.plot(
                    after_frequency_hz,
                    after_amplitude,
                    color="#0F766E",
                    linewidth=1.8,
                    label="Processed",
                )

                # Shade the declared useful band, not the filter response.
                axis.axvspan(
                    applied_low_hz,
                    applied_high_hz,
                    color="#0F766E",
                    alpha=0.08,
                    label="Declared useful band",
                )

                # Label and style the shared frequency axes.
                style_plot(
                    axis,
                    "Frequency (Hz)",
                    "Estimated amplitude (dimensionless)",
                )
                axis.set_xlim(0, fs_hz / 2)
                axis.legend(frameon=False)

                # Render the spectrum overlay and release its figure.
                figure.tight_layout()
                st.pyplot(figure)
                plt.close(figure)

                # Explain the expected result specifically for the notch example.
                if applied_filter["filter_type"] == "notch":
                    st.info(
                        "For the 400 Hz notch example, check that the 400 Hz "
                        "component decreases while the 100 Hz component remains "
                        "close to its original amplitude."
                    )

                # Explain the result when no filtering was requested.
                elif applied_filter["filter_type"] == "none":
                    st.info(
                        "The signal was kept unchanged. Both waveforms and "
                        "spectra should overlap, and the RMS values should match."
                    )

                # Give general guidance for other filter suggestions.
                else:
                    st.info(
                        "Check whether candidate interference decreased and "
                        "the components you wanted to preserve remain."
                    )

# Display the remaining Export placeholder.
else:
    st.header("Export")
    st.info("Export is coming next. Review your result in Compare for now.")

# Separate page content from navigation controls.
st.divider()

# Arrange Back and Continue buttons.
back_column, next_column = st.columns(2)

# Display Back when an earlier stage exists.
with back_column:
    if stage_index > 0:
        # Retrieve the previous stage.
        previous_stage = stages[stage_index - 1]

        # Connect backward navigation to the callback.
        st.button(
            f"← Back to {previous_stage}",
            key="back_stage",
            on_click=select_stage,
            args=(previous_stage,),
        )

# Display Continue when another stage exists.
with next_column:
    if stage_index < len(stages) - 1:
        # Retrieve the next stage.
        next_stage = stages[stage_index + 1]

        # Determine which required data is available.
        signal_loaded = "samples" in st.session_state
        result_available = "filtered_samples" in st.session_state

        # Enable navigation through Inspect and into Compare.
        can_continue = (
            signal_loaded and stage in ("Upload", "Inspect")
        ) or (
            stage == "Configure" and result_available
        )

        # Disable Continue when its required result is unavailable.
        st.button(
            f"Continue to {next_stage} →",
            key="next_stage",
            type="primary",
            width="stretch",
            disabled=not can_continue,
            on_click=select_stage,
            args=(next_stage,),
        )

        # Explain the remaining unfinished stage.
        if stage == "Compare":
            st.caption("Export is still being built.")

        # Explain the requirement for moving into Compare.
        elif stage == "Configure" and not can_continue:
            st.caption("Apply the suggestion before continuing to Compare.")
