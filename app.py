# Import NumPy to generate our example signal.
import numpy as np

# Import Streamlit to build the web interface.
import streamlit as st

# Import the shared validation function.
from signal_lab.validation import validate_signal

# Set the browser-tab title and use a wide page layout.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Divide the header into two columns with a 4-to-1 width ratio.
brand_column, status_column = st.columns([4, 1])

# Place the brand in the wider column.
with brand_column:
    # Display the application's name.
    st.title("Signal Lab")

    # Explain what the workspace does.
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

# Place the preview status in the smaller column.
with status_column:
    # Identify this version as a demo workspace.
    st.info("Demo workspace")

# Separate the header from the stage content.
st.divider()

# Create sidebar navigation and store the selected stage.
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

    # Describe the known useful and unwanted tones.
    st.write(
        "Example: a useful 100 Hz tone plus an unwanted 400 Hz tone, "
        "sampled at 2,000 samples per second for 5 seconds."
    )

    # Generate the example when this button is clicked.
    if st.button("Load built-in example"):
        # Set the sampling rate in hertz.
        fs_hz = 2000.0

        # Set the duration in seconds.
        duration_s = 5.0

        # Calculate the number of samples.
        sample_count = int(fs_hz * duration_s)

        # Calculate each sample's time in seconds.
        time_s = np.arange(sample_count) / fs_hz

        # Generate the useful 100 Hz tone with dimensionless amplitude 1.
        useful_signal = np.sin(2 * np.pi * 100.0 * time_s)

        # Generate the unwanted 400 Hz tone with dimensionless amplitude 0.5.
        unwanted_signal = 0.5 * np.sin(2 * np.pi * 400.0 * time_s)

        # Combine the tones and validate the result.
        samples = validate_signal(useful_signal + unwanted_signal, fs_hz)

        # Remember the samples when the user changes stages.
        st.session_state["samples"] = samples

        # Remember the sampling rate alongside the samples.
        st.session_state["fs_hz"] = fs_hz

    # Confirm that a signal is available.
    if "samples" in st.session_state:
        st.success("Example loaded. Select Inspect in the sidebar.")

# Display the Inspect stage.
elif stage == "Inspect":
    # Display the stage heading.
    st.header("2. Inspect your signal")

    # Give instructions if no signal has been loaded.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the built-in example first.")

    # Display measurements when a signal is available.
    else:
        # Retrieve the stored samples.
        samples = st.session_state["samples"]

        # Retrieve the sampling rate in hertz.
        fs_hz = st.session_state["fs_hz"]

        # Display the sampling rate.
        st.metric("Sampling rate", f"{fs_hz:g} Hz")

        # Calculate and display the duration in seconds.
        st.metric("Duration", f"{len(samples) / fs_hz:.2f} s")

        # Calculate and display the Nyquist frequency in hertz.
        st.metric("Nyquist frequency", f"{fs_hz / 2:g} Hz")

# Display placeholders for the remaining stages.
else:
    # Display the selected stage's name.
    st.header(stage)

    # Explain that this feature is still being developed.
    st.info("This stage is coming next. Use Upload and Inspect for now.")
