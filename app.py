# Import NumPy to generate our example signal.
import numpy as np

# Import Streamlit to build the web interface.
import streamlit as st

# Import the same validation function used by our teaching notebook.
from signal_lab.validation import validate_signal

# Set the browser-tab title and use a wide page layout.
st.set_page_config(page_title="AI Signal Processing Lab", layout="wide")

# Display the application's main title.
st.title("AI Signal Processing Lab")

# Explain the intended workflow.
st.caption("Upload → Inspect → Configure → Compare → Export")

# Explain the scope of this first preview.
st.info("Start with the built-in example. File uploads are coming later.")

# Create navigation and remember the selected stage.
stage = st.sidebar.radio(
    "Choose a stage",
    ["Upload", "Inspect", "Configure", "Compare", "Export"],
)

# Show the first stage when Upload is selected.
if stage == "Upload":
    # Display a heading for this stage.
    st.header("1. Choose your signal")

    # Describe the example and identify its known unwanted component.
    st.write(
        "Example: a useful 100 Hz tone plus an unwanted 400 Hz tone, "
        "sampled at 2,000 samples per second for 5 seconds."
    )

    # Run the following block when the user clicks this button.
    if st.button("Load built-in example"):
        # Set the sampling rate in hertz (samples per second).
        fs_hz = 2000.0

        # Set the signal duration in seconds.
        duration_s = 5.0

        # Calculate the number of samples.
        sample_count = int(fs_hz * duration_s)

        # Calculate each sample's time in seconds.
        time_s = np.arange(sample_count) / fs_hz

        # Generate a 100 Hz sine wave with dimensionless amplitude 1.
        useful_signal = np.sin(2 * np.pi * 100.0 * time_s)

        # Generate a 400 Hz sine wave with dimensionless amplitude 0.5.
        unwanted_signal = 0.5 * np.sin(2 * np.pi * 400.0 * time_s)

        # Combine the tones and validate them using our shared module.
        samples = validate_signal(useful_signal + unwanted_signal, fs_hz)

        # Store the signal so it survives navigation between stages.
        st.session_state["samples"] = samples

        # Store its sampling rate alongside it.
        st.session_state["fs_hz"] = fs_hz

    # Show confirmation whenever an example is stored.
    if "samples" in st.session_state:
        st.success("Example loaded. Select Inspect in the sidebar.")

# Show basic signal information when Inspect is selected.
elif stage == "Inspect":
    # Display this stage's heading.
    st.header("2. Inspect your signal")

    # Check whether the user has loaded the example.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the built-in example first.")
    else:
        # Retrieve the stored samples.
        samples = st.session_state["samples"]

        # Retrieve the sampling rate in hertz.
        fs_hz = st.session_state["fs_hz"]

        # Display the sampling rate.
        st.metric("Sampling rate", f"{fs_hz:g} Hz")

        # Display duration: sample count divided by samples per second.
        st.metric("Duration", f"{len(samples) / fs_hz:.2f} s")

        # Display the Nyquist frequency in hertz.
        st.metric("Nyquist frequency", f"{fs_hz / 2:g} Hz")

# Display a placeholder for each remaining stage.
else:
    # Show the selected stage's name.
    st.header(stage)

    # Explain that this stage will be connected in a later step.
    st.info("This stage will be connected after the preview is running.")
