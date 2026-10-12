# Import NumPy for sample positions and the generated example.
import numpy as np

# Import Streamlit for the web interface.
import streamlit as st

# Import Matplotlib for waveform and spectrum plots.
import matplotlib.pyplot as plt

# Import the shared signal validator.
from signal_lab.validation import validate_signal

# Import the packaged detector rather than rewriting its calculations.
from signal_lab.spectral import detect_peaks

# Set the browser-tab title and wide layout.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Define the workflow order.
stages = ["Upload", "Inspect", "Configure", "Compare", "Export"]

# Initialize navigation for a new browser session.
if "current_stage" not in st.session_state:
    st.session_state["current_stage"] = "Upload"


# Define the callback used by navigation buttons.
def select_stage(stage_name):
    # Remember the selected stage before Streamlit redraws the page.
    st.session_state["current_stage"] = stage_name


# Create columns for the brand and workspace status.
brand_column, status_column = st.columns([4, 1])

# Display the brand in the wider column.
with brand_column:
    st.title("Signal Lab")
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

# Display the preview status in the smaller column.
with status_column:
    st.info("Demo workspace")

# Separate the header from the stage content.
st.divider()

# Put workflow navigation in the sidebar.
with st.sidebar:
    st.subheader("Signal Lab")
    st.caption("YOUR WORKSPACE")

    # Loop through the stages, numbering them from one.
    for number, stage_name in enumerate(stages, start=1):
        # Determine which navigation button to highlight.
        is_selected = st.session_state["current_stage"] == stage_name

        # Create a full-width navigation button with a unique key.
        st.button(
            f"{number:02d}  ·  {stage_name}",
            key=f"navigation_{stage_name}",
            type="primary" if is_selected else "secondary",
            width="stretch",
            on_click=select_stage,
            args=(stage_name,),
        )

    # Separate the navigation from the availability notes.
    st.divider()
    st.caption("Built-in example available")
    st.caption("Configure, Compare, and Export are coming next.")

# Retrieve the current stage.
stage = st.session_state["current_stage"]

# Find its zero-based position in the workflow.
stage_index = stages.index(stage)

# Display a step number that counts from one.
st.caption(f"STEP {stage_index + 1:02d} OF {len(stages):02d}")

# Show the signal-selection page.
if stage == "Upload":
    st.header("Start with a signal")
    st.caption("Explore the guided example to get familiar with the lab.")

    # Arrange the recording and example panels side by side.
    upload_column, example_column = st.columns(2)

    # Display the planned file-upload panel.
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

            # Generate the example only when the button is clicked.
            if st.button("Load example", type="primary", width="stretch"):
                # Set the sampling rate in samples per second, expressed in Hz.
                fs_hz = 2000.0

                # Set the recording duration in seconds.
                duration_s = 5.0

                # Calculate the number of samples.
                sample_count = int(fs_hz * duration_s)

                # Calculate sample times in seconds.
                time_s = np.arange(sample_count) / fs_hz

                # Generate the useful tone with dimensionless amplitude 1.
                useful_signal = np.sin(2 * np.pi * 100.0 * time_s)

                # Generate the unwanted tone with dimensionless amplitude 0.5.
                unwanted_signal = 0.5 * np.sin(2 * np.pi * 400.0 * time_s)

                # Validate the combined signal using our shared module.
                samples = validate_signal(
                    useful_signal + unwanted_signal,
                    fs_hz,
                )

                # Remember the signal and sampling rate between pages.
                st.session_state["samples"] = samples
                st.session_state["fs_hz"] = fs_hz

            # Confirm that the example is available.
            if "samples" in st.session_state:
                st.success("Example ready. Continue to Inspect below.")

# Show the signal-inspection page.
elif stage == "Inspect":
    st.header("Understand your signal")
    st.caption("Inspect the waveform and its detected frequency components.")

    # Handle navigation to Inspect before loading a signal.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the built-in example first.")

    else:
        # Retrieve the dimensionless signal samples and sampling rate in Hz.
        samples = st.session_state["samples"]
        fs_hz = st.session_state["fs_hz"]

        # Arrange the three measurement cards in equal columns.
        rate_card, duration_card, nyquist_card = st.columns(3)

        # Display the sampling rate.
        with rate_card:
            with st.container(border=True):
                st.metric("Sampling rate", f"{fs_hz:,.0f} Hz")

        # Display sample count divided by sampling rate as seconds.
        with duration_card:
            with st.container(border=True):
                st.metric("Duration", f"{len(samples) / fs_hz:.2f} s")

        # Display half the sampling rate as the Nyquist frequency.
        with nyquist_card:
            with st.container(border=True):
                st.metric("Nyquist frequency", f"{fs_hz / 2:,.0f} Hz")

        # Group the waveform and known example details.
        with st.container(border=True):
            st.subheader("Signal preview")
            st.caption("Original signal · first 50 milliseconds")

            # Select the first 50 ms of samples.
            preview_count = min(len(samples), int(0.05 * fs_hz))

            # Convert sample positions to milliseconds.
            preview_time_ms = np.arange(preview_count) / fs_hz * 1000

            # Create a taller waveform figure.
            figure, axis = plt.subplots(figsize=(8, 3.8))

            # Plot the actual selected samples without smoothing.
            axis.plot(
                preview_time_ms,
                samples[:preview_count],
                color="#0F766E",
                linewidth=1.8,
            )

            # Label the axes with their quantities and units.
            axis.set_xlabel("Time (ms)")
            axis.set_ylabel("Amplitude (dimensionless)")

            # Set the displayed time interval and subtle grid.
            axis.set_xlim(0, 50)
            axis.grid(True, alpha=0.15)

            # Remove unnecessary plot borders.
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

            # Fit the axis labels inside the figure.
            figure.tight_layout()

            # Give the waveform more width than the explanation.
            plot_column, details_column = st.columns([3, 1])

            # Display the waveform.
            with plot_column:
                st.pyplot(figure)

            # Identify the known components used to generate this example.
            with details_column:
                st.markdown("**Known demo tones**")
                st.metric("Useful tone", "100 Hz")
                st.caption("The component we want to preserve.")
                st.metric("Unwanted tone", "400 Hz")
                st.caption("The interference we will aim to reduce.")

            # Release the figure after rendering it.
            plt.close(figure)

        # Group automatic analysis in its own panel.
        with st.container(border=True):
            st.subheader("Frequency analysis")
            st.caption("Calculated from the complete 5-second signal.")

            # Handle expected input or detection-setting errors clearly.
            try:
                # Call the packaged detector with explicit settings.
                frequency_hz, amplitude, peak_indices = detect_peaks(
                    samples,
                    fs_hz,
                    relative_threshold=0.10,
                    minimum_separation_hz=5.0,
                )

            # Display the shared function's error message if analysis fails.
            except ValueError as error:
                st.error(f"Frequency analysis could not run: {error}")

            else:
                # Extract detected frequencies in Hz.
                detected_hz = frequency_hz[peak_indices]

                # Extract their estimated dimensionless sinusoidal amplitudes.
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

                # Mark detected peaks using a distinct shape.
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

                # Label the spectrum axes and amplitude units.
                spectrum_axis.set_xlabel("Frequency (Hz)")
                spectrum_axis.set_ylabel(
                    "Estimated amplitude (dimensionless)"
                )

                # Add a subtle grid and remove unnecessary borders.
                spectrum_axis.grid(True, alpha=0.15)
                spectrum_axis.spines["top"].set_visible(False)
                spectrum_axis.spines["right"].set_visible(False)

                # Display the marker legend when peaks exist.
                if len(peak_indices) > 0:
                    spectrum_axis.legend(frameon=False)

                # Fit the labels and render the spectrum.
                spectrum_figure.tight_layout()
                st.pyplot(spectrum_figure)

                # Release the spectrum figure's memory.
                plt.close(spectrum_figure)

                # Explain when the detector finds no qualifying peaks.
                if len(peak_indices) == 0:
                    st.info("No peaks met the current detection settings.")

                else:
                    # Display peaks in the amplitude ranking returned by the module.
                    st.dataframe(
                        {
                            "Frequency (Hz)": detected_hz,
                            "Estimated amplitude (dimensionless)":
                                detected_amplitude,
                        },
                        hide_index=True,
                        width="stretch",
                    )

                # Explain the detection settings in plain language.
                st.caption(
                    "Detection settings: minimum height 10% of the largest "
                    "spectral amplitude; minimum prominence 5% of that "
                    "amplitude; minimum peak separation 5 Hz."
                )

                # Avoid confusing a detected component with identified noise.
                st.info(
                    "A detected frequency is not automatically noise. "
                    "For this example, we already know that 100 Hz is useful "
                    "and 400 Hz is unwanted."
                )

        # Provide optional learning guidance below the results.
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
                "**Frequency spectrum:** shows the estimated amplitudes "
                "of frequency components. The module removes the mean and "
                "applies a Hann window before calculating the FFT."
            )
            st.caption(
                "The known tone labels describe the generated example. "
                "The frequency table contains calculated detection results."
            )

# Clearly identify stages that are not connected yet.
else:
    st.header(stage)
    st.info("This stage is coming next. Use Upload and Inspect for now.")

# Separate results from workflow navigation.
st.divider()

# Create columns for Back and Continue.
back_column, next_column = st.columns(2)

# Display backward navigation when an earlier stage exists.
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

# Display forward navigation when another stage exists.
with next_column:
    if stage_index < len(stages) - 1:
        # Retrieve the next stage.
        next_stage = stages[stage_index + 1]

        # Check whether signal samples are available.
        signal_loaded = "samples" in st.session_state

        # Enable Continue from Upload once the example is loaded.
        can_continue = stage == "Upload" and signal_loaded

        # Keep later Continue buttons disabled until their stages are connected.
        st.button(
            f"Continue to {next_stage} →",
            key="next_stage",
            type="primary",
            width="stretch",
            disabled=not can_continue,
            on_click=select_stage,
            args=(next_stage,),
        )

        # Explain why forward navigation is unavailable on later stages.
        if stage != "Upload":
            st.caption("The next processing stage is still being built.")
