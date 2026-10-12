# Import NumPy to generate and handle signal samples.
import numpy as np

# Import Streamlit to build the interface.
import streamlit as st

# Import Matplotlib to draw the waveform.
import matplotlib.pyplot as plt

# Import our shared signal-validation function.
from signal_lab.validation import validate_signal

# Configure the browser-tab title and page width.
st.set_page_config(page_title="Signal Lab", layout="wide")

# Define the workflow stages in order.
stages = ["Upload", "Inspect", "Configure", "Compare", "Export"]

# Start on Upload when no stage has been selected.
if "current_stage" not in st.session_state:
    st.session_state["current_stage"] = "Upload"


# Define the callback used by navigation buttons.
def select_stage(stage_name):
    # Remember which stage the user selected.
    st.session_state["current_stage"] = stage_name


# Create a wide brand column and a narrower status column.
brand_column, status_column = st.columns([4, 1])

# Place the brand in the wider column.
with brand_column:
    # Display the application name.
    st.title("Signal Lab")

    # Describe the workspace.
    st.caption("AI Signal Processing Lab · Explore. Filter. Compare.")

# Place the status in the narrower column.
with status_column:
    # Identify the current version as a demo.
    st.info("Demo workspace")

# Separate the header from the stage content.
st.divider()

# Group navigation inside the sidebar.
with st.sidebar:
    # Display the workspace name.
    st.subheader("Signal Lab")

    # Label the navigation area.
    st.caption("YOUR WORKSPACE")

    # Create one numbered button for each stage.
    for number, stage_name in enumerate(stages, start=1):
        # Check whether this is the selected stage.
        is_selected = st.session_state["current_stage"] == stage_name

        # Highlight the selected button and connect its callback.
        st.button(
            f"{number:02d}  ·  {stage_name}",
            key=f"navigation_{stage_name}",
            type="primary" if is_selected else "secondary",
            width="stretch",
            on_click=select_stage,
            args=(stage_name,),
        )

    # Separate navigation from supporting information.
    st.divider()

    # Identify the available input.
    st.caption("Built-in example available")

    # Clearly label the unfinished stages.
    st.caption("Configure, Compare, and Export are coming next.")

# Retrieve the selected stage.
stage = st.session_state["current_stage"]

# Display the Upload stage.
if stage == "Upload":
    # Display the main heading.
    st.header("Start with a signal")

    # Explain the available starting point.
    st.caption("Explore the guided example to get familiar with the lab.")

    # Create two equally sized columns.
    upload_column, example_column = st.columns(2)

    # Place the future upload area in the left column.
    with upload_column:
        # Group the upload content inside a bordered panel.
        with st.container(border=True):
            # Display the panel heading.
            st.subheader("Your recording")

            # Explain which file formats are planned.
            st.write("WAV audio or CSV signal measurements.")

            # Display the future upload control without enabling it.
            st.file_uploader(
                "Choose a signal file",
                type=["wav", "csv"],
                disabled=True,
                key="planned_upload",
            )

            # Explain why uploads are disabled.
            st.caption("File uploads will be connected in a later step.")

    # Place the working example in the right column.
    with example_column:
        # Group the example content inside a matching panel.
        with st.container(border=True):
            # Display the panel heading.
            st.subheader("Try the guided example")

            # Describe the known signal components.
            st.write(
                "A useful 100 Hz tone mixed with an unwanted 400 Hz tone."
            )

            # Display the example's recording details.
            st.caption(
                "5 seconds · 2,000 Hz sampling rate · Dimensionless amplitude"
            )

            # Generate the example when this button is clicked.
            if st.button("Load example", type="primary", width="stretch"):
                # Set the sampling rate in hertz (samples per second).
                fs_hz = 2000.0

                # Set the recording duration in seconds.
                duration_s = 5.0

                # Calculate the total sample count.
                sample_count = int(fs_hz * duration_s)

                # Calculate each sample's time in seconds.
                time_s = np.arange(sample_count) / fs_hz

                # Generate the useful tone with dimensionless amplitude 1.
                useful_signal = np.sin(2 * np.pi * 100.0 * time_s)

                # Generate the unwanted tone with dimensionless amplitude 0.5.
                unwanted_signal = 0.5 * np.sin(2 * np.pi * 400.0 * time_s)

                # Validate the combined signal using our shared function.
                samples = validate_signal(
                    useful_signal + unwanted_signal,
                    fs_hz,
                )

                # Remember the samples between stages.
                st.session_state["samples"] = samples

                # Remember the sampling rate alongside the samples.
                st.session_state["fs_hz"] = fs_hz

            # Confirm when the example is ready.
            if "samples" in st.session_state:
                st.success("Example ready. Continue to Inspect below.")

# Display the Inspect stage.
elif stage == "Inspect":
    # Display the stage heading.
    st.header("Understand your signal")

    # Explain the purpose of this stage.
    st.caption("Check your signal's key properties before choosing a filter.")

    # Provide guidance when no signal is loaded.
    if "samples" not in st.session_state:
        st.warning("Go to Upload and load the built-in example first.")

    # Display measurements and waveform when samples are available.
    else:
        # Retrieve the stored samples.
        samples = st.session_state["samples"]

        # Retrieve the sampling rate in hertz.
        fs_hz = st.session_state["fs_hz"]

        # Create three equally sized measurement columns.
        rate_card, duration_card, nyquist_card = st.columns(3)

        # Place the sampling-rate card in the first column.
        with rate_card:
            # Draw a border around the card.
            with st.container(border=True):
                # Display the sampling rate with a thousands separator.
                st.metric("Sampling rate", f"{fs_hz:,.0f} Hz")

        # Place the duration card in the second column.
        with duration_card:
            # Draw a matching border.
            with st.container(border=True):
                # Calculate and display duration in seconds.
                st.metric("Duration", f"{len(samples) / fs_hz:.2f} s")

        # Place the Nyquist card in the third column.
        with nyquist_card:
            # Draw a matching border.
            with st.container(border=True):
                # Display half the sampling rate in hertz.
                st.metric("Nyquist frequency", f"{fs_hz / 2:,.0f} Hz")

        # Create the waveform panel below the cards.
        with st.container(border=True):
            # Display the panel heading.
            st.subheader("Signal preview")

            # Explain the displayed time interval.
            st.caption("Original signal · first 50 milliseconds")

            # Select up to 50 milliseconds of samples.
            preview_count = min(len(samples), int(0.05 * fs_hz))

            # Convert sample positions to time in milliseconds.
            preview_time_ms = np.arange(preview_count) / fs_hz * 1000

            # Create a figure and plotting area.
            figure, axis = plt.subplots(figsize=(10, 3))

            # Plot the original signal in teal.
            axis.plot(
                preview_time_ms,
                samples[:preview_count],
                color="#0F766E",
                linewidth=1.8,
            )

            # Label the horizontal axis.
            axis.set_xlabel("Time (ms)")

            # Label the vertical axis for the dimensionless demo signal.
            axis.set_ylabel("Amplitude (dimensionless)")

            # Show the 0–50 millisecond viewing interval.
            axis.set_xlim(0, 50)

            # Add a faint grid.
            axis.grid(True, alpha=0.15)

            # Hide the upper plot border.
            axis.spines["top"].set_visible(False)

            # Hide the right plot border.
            axis.spines["right"].set_visible(False)

            # Adjust spacing to fit the labels.
            figure.tight_layout()

            # Give the waveform more width than the tone details.
            plot_column, details_column = st.columns([3, 1])

            # Display the waveform in the wider column.
            with plot_column:
                # Render the actual signal plot.
                st.pyplot(figure)

            # Display known example information beside the waveform.
            with details_column:
                # Clearly identify these as known demo components.
                st.markdown("**Known demo tones**")

                # Display the useful component's frequency in hertz.
                st.metric("Useful tone", "100 Hz")

                # Explain its role in the example.
                st.caption("The component we want to preserve.")

                # Display the unwanted component's frequency in hertz.
                st.metric("Unwanted tone", "400 Hz")

                # Explain its role in the example.
                st.caption("The interference we will aim to reduce.")

            # Release the figure's memory after displaying it.
            plt.close(figure)

# Display placeholders for the remaining stages.
else:
    # Display the selected stage's name.
    st.header(stage)

    # Explain which stages are currently available.
    st.info("This stage is coming next. Use Upload and Inspect for now.")

# Separate the stage content from the navigation controls.
st.divider()

# Find the current stage's position in the workflow.
stage_index = stages.index(stage)

# Create columns for backward and forward navigation.
back_column, next_column = st.columns(2)

# Place the Back button in the left column.
with back_column:
    # Show Back only when an earlier stage exists.
    if stage_index > 0:
        # Retrieve the previous stage's name.
        previous_stage = stages[stage_index - 1]

        # Navigate backward using the existing callback.
        st.button(
            f"← Back to {previous_stage}",
            key="back_stage",
            on_click=select_stage,
            args=(previous_stage,),
        )

# Place the Continue button in the right column.
with next_column:
    # Show Continue only when another stage exists.
    if stage_index < len(stages) - 1:
        # Retrieve the next stage's name.
        next_stage = stages[stage_index + 1]

        # Check whether a signal has been loaded.
        signal_loaded = "samples" in st.session_state

        # Allow forward navigation from Upload after loading a signal.
        can_continue = stage == "Upload" and signal_loaded

        # Display the forward navigation button.
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
        if stage != "Upload":
            st.caption("The next processing stage is still being built.")
