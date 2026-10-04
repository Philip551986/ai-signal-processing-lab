# AI Signal Processing Lab

Load WAV or CSV signals, detect frequency peaks, suggest and apply
filters, and compare the results using shared Python modules.

## Project files

- pyproject.toml: package settings and dependencies.
- signal_lab/__init__.py: identifies the Python package.
- signal_lab/validation.py: checks signal inputs and sampling rate.
- signal_lab/spectral.py: FFT calculation and peak detection.
- signal_lab/recommendation.py: filter selection using a useful band.
- signal_lab/filters.py: filter design and application.
- signal_lab/metrics.py: RMS measurements and comparison.
- signal_lab/data_io.py: WAV/CSV loading and processed WAV export.
- tests/: automated tests against the packaged functions.
- notebooks/Signal_Lab_Teaching.ipynb: maintained teaching notebook.
- outputs/: generated results, created when needed.

## Restore in Colab

Upload ai_signal_lab_handoff.zip through the Files sidebar.
Extract it under /content using these Python statements:

    from zipfile import ZipFile
    with ZipFile("/content/ai_signal_lab_handoff.zip") as archive:
        archive.extractall("/content")

The restored project must contain:
    /content/ai_signal_lab/pyproject.toml

Keep the downloaded archive: Colab runtime files are temporary.

## Install and test in Colab

Run these lines in a Colab code cell:

    %pip install -e "/content/ai_signal_lab[test]"
    !python -m pytest /content/ai_signal_lab/tests -q --import-mode=importlib

The current test collection contains 53 tests.

## Teaching notebook

Extract the archive on your computer.
Choose File > Upload notebook in Colab and select:
    ai_signal_lab/notebooks/Signal_Lab_Teaching.ipynb

Restore the project archive in that notebook's runtime too.
Install plotting support if needed:
    %pip install matplotlib

Run notebook cells from top to bottom.

## Use your own file

Upload your WAV or CSV through the Colab Files sidebar.
Set input_path to its path in the file-selection cell.
Leave input_path = None to use the demonstration.

For WAV, wav_channel = 0 selects the first channel.
The sampling rate comes from the WAV file.
WAV amplitudes are dimensionless digital values.

CSV must contain an amplitude column.
An optional time_s column contains timestamps in seconds.
Timestamps must be finite, increasing, and uniformly spaced.
Without timestamps, supply csv_fs_hz in Hz.
Declare csv_amplitude_unit correctly, such as V.

Adjust useful_low_hz and useful_high_hz to your recording.
Rerun all following cells after changing the input or settings.

## Download audio

Digital audio is saved to outputs/processed_audio.wav.
Set download_audio = True in the export cell to download it in Colab.
The same output filename is reused on later exports.

Export produces mono 16-bit WAV and reports its amplitude divisor.
Peaks above 1 are scaled down; smaller signals are not amplified.
Measurement data in physical units requires an explicit conversion
before it can be treated as digital audio.

## Current assumptions and limits

- One real-valued, uniformly sampled channel is processed.
- Processing requires at least 64 finite, nonconstant samples.
- The processing validator rejects silence and constants.
  RMS and WAV export can handle silence.
- The user declares the useful frequency band.
- Detected peaks outside that band are unwanted candidates.
  This is a rule-based recommendation, not general AI noise recognition.
- Forward/backward filtering is for offline recordings.
  Edge transients may occur.
- Lower RMS alone does not establish improved signal quality.
- WAV export requires a positive whole-number sampling rate.

## Maintain the shared code

Edit calculations in signal_lab modules.
The teaching notebook and future app must import those modules.
Do not maintain separate copies of the processing algorithms.

Keep the original development notebook as historical material.
Use Signal_Lab_Teaching.ipynb as the maintained teaching notebook.

## Verification performed

53 packaged-function tests passed.
The teaching notebook ran in a fresh Python kernel.
Demonstration, generated WAV, and generated CSV scenarios passed.
These checks do not establish noise recognition for every recording.
