# Import NumPy to generate and handle signal samples.
import numpy as np

# Import Streamlit to build the web interface.
import streamlit as st

# Import Matplotlib to plot the waveform.
import matplotlib.pyplot as plt

# Import the shared signal-validation function.
from signal_lab.validation import validate_signal

# Set the browser-tab title and use a wide layout.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Divide the header into columns with a 4-to-1 width ratio.
brand_column, status_column = st.columns([4, 1])

# Place the application name in the wider column.
with brand_column:
    # Display the brand.
    st.title("Signal Lab")

    # Describe the workspace.
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

# Place the preview status in the smaller column.
with status_column:
    # Identify this version as a demo.
    st.info("Demo workspace")

# Separate the header from the page content.
st.divider()

# Create navigation and retrieve the selected stage.
stage = st.sidebar.radio(
    "Choose a stage",
    ["Upload", "Inspect", "Configure", "Compare", "Export"],
)

# Display the Upload stage.
if stage == "Upload":
    # Display the stage heading.
    st.header("1. Choose your signal")

    # Explain which input is currently available.
    st.caption("Start with the guided example. File uploads are coming later.")

    # Describe the example's known frequency components.
    st.write(
        "Example: a useful 100 Hz tone plus an unwanted 400 Hz tone, "
        "sampled at 2,000 samples per second for 5 seconds."
    )

    # Generate the example when the user clicks the button.
    if st.button("Load built-in example", type="primary"):
        # Set the sampling rate in hertz (samples per second).
        fs_hz = 2000.0

        # Set the duration in seconds.
        duration_s = 5.0

        # Calculate the total number of samples.
        sample_count = int(fs_hz * duration_s)

        # Calculate the time of each sample in seconds.
        time_s = np.arange(sample_count) / fs_hz

        # Generate the useful 100 Hz tone with dimensionless amplitude 1.
        useful_signal = np.sin(2 * np.pi * 100.0 * time_s)

        # Generate the unwanted 400 Hz tone with dimensionless amplitude 0.5.
        unwanted_signal = 0.5 * np.sin(2 * np.pi * 400.0 * time_s)

        # Combine the tones and validate the resulting signal.
        samples = validate_signal(useful_signal + unwanted_signal, fs_hz)

        # Preserve the samples when the user switches stages.
        st.session_state["samples"] = samples

        # Preserve the sampling rate alongside the samples.
        st.session_state["fs_hz"] = fs_hz

    # Confirm that the example is available.
    if "samples" in st.session_state:
        st.success("Example loaded. Select Inspect in the sidebar.")

# Display the Inspect stage.
elif stage == "Inspect":
    # Display the stage heading.
    st.header("Understand your signal")

    # Explain what the user should check here.
    st.caption("Check your signal's key properties before choosing a filter.")

    # Give instructions if no signal is available.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the built-in example first.")

    # Display the measurements and waveform when a signal is available.
    else:
        # Retrieve the stored signal samples.
        samples = st.session_state["samples"]

        # Retrieve the sampling rate in hertz.
        fs_hz = st.session_state["fs_hz"]

        # Create three columns of equal width.
        rate_card, duration_card, nyquist_card = st.columns(3)

        # Place the sampling-rate card in the first column.
        with rate_card:
            # Draw a border around the card.
            with st.container(border=True):
                # Display the sampling rate with a thousands separator.
                st.metric("Sampling rate", f"{fs_hz:,.0f} Hz")

        # Place the duration card in the second column.
        with duration_card:
            # Draw a matching card border.
            with st.container(border=True):
                # Divide sample count by sampling rate to get seconds.
                st.metric("Duration", f"{len(samples) / fs_hz:.2f} s")

        # Place the Nyquist-frequency card in the third column.
        with nyquist_card:
            # Draw a matching card border.
            with st.container(border=True):
                # Display half the sampling rate in hertz.
                st.metric("Nyquist frequency", f"{fs_hz / 2:,.0f} Hz")

        # Create a waveform panel below the cards.
        with st.container(border=True):
            # Display the panel heading.
            st.subheader("Signal preview")

            # Explain the time interval being plotted.
            st.caption("Original signal · first 50 milliseconds")

            # Select up to 50 milliseconds of signal samples.
            preview_count = min(len(samples), int(0.05 * fs_hz))

            # Convert sample positions to time in milliseconds.
            preview_time_ms = np.arange(preview_count) / fs_hz * 1000

            # Create a figure and its plotting area.
            figure, axis = plt.subplots(figsize=(10, 3))

            # Plot the original samples in teal.
            axis.plot(
                preview_time_ms,
                samples[:preview_count],
                color="#0F766E",
                linewidth=1.8,
            )

            # Label the horizontal axis with its units.
            axis.set_xlabel("Time (ms)")

            # Label the vertical axis for this demo's dimensionless samples.
            axis.set_ylabel("Amplitude (dimensionless)")

            # Show the complete 0–50 millisecond viewing interval.
            axis.set_xlim(0, 50)

            # Add a faint grid for reading the waveform.
            axis.grid(True, alpha=0.15)

            # Remove the upper plot border.
            axis.spines["top"].set_visible(False)

            # Remove the right plot border.
            axis.spines["right"].set_visible(False)

            # Adjust spacing so the axis labels fit.
            figure.tight_layout()

            # Display the plot in the application.
            st.pyplot(figure)

            # Release the figure's memory after displaying it.
            plt.close(figure)

            # Explain the known components of the built-in example.
            st.caption(
                "Known demo tones: 100 Hz useful signal · "
                "400 Hz unwanted signal."
            )

# Display placeholders for the remaining stages.
else:
    # Display the selected stage's name.
    st.header(stage)

    # Explain which features are currently available.
    st.info("This stage is coming next. Use Upload and Inspect for now.")
