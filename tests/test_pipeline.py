
import numpy as np

from signal_lab.spectral import calculate_fft, detect_peaks
from signal_lab.recommendation import recommend_filter
from signal_lab.filters import apply_filter


def test_two_tone_processing():
    # Sampling rate: 2000 samples per second (Hz).
    fs_hz = 2000.0

    # Generate 5 seconds of sample times, measured in seconds.
    time_s = np.arange(10000) / fs_hz

    # The useful tone is 100 Hz with amplitude 1.
    useful = np.sin(2 * np.pi * 100 * time_s)

    # The unwanted tone is 400 Hz with amplitude 0.5.
    unwanted = 0.5 * np.sin(2 * np.pi * 400 * time_s)

    # Combine both tones into the input signal.
    original = useful + unwanted

    # Detect the prominent frequency peaks.
    frequencies_hz, amplitudes, peak_indices = detect_peaks(
        original, fs_hz
    )
    detected_hz = frequencies_hz[peak_indices]

    # Allow one FFT-bin spacing when checking peak frequencies.
    bin_width_hz = fs_hz / len(original)

    # Confirm that both known tones were detected.
    assert np.any(np.abs(detected_hz - 100) <= bin_width_hz)
    assert np.any(np.abs(detected_hz - 400) <= bin_width_hz)

    # Declare 80–250 Hz as the useful frequency band.
    choice = recommend_filter(
        detected_hz,
        fs_hz,
        useful_low_hz=80.0,
        useful_high_hz=250.0,
    )

    # A single detected tone outside that band should select a notch.
    assert choice["filter_type"] == "notch"
    assert abs(choice["cutoff_hz"] - 400.0) <= bin_width_hz

    # Apply the recommended filter.
    filtered = apply_filter(
        original,
        fs_hz,
        filter_type=choice["filter_type"],
        cutoff_hz=choice["cutoff_hz"],
    )

    # Filtering must preserve sample count and return finite values.
    assert filtered.shape == original.shape
    assert np.all(np.isfinite(filtered))

    # Calculate spectra before and after filtering.
    frequencies_hz, before = calculate_fft(original, fs_hz)
    _, after = calculate_fft(filtered, fs_hz)

    # Find the FFT bins nearest the two known frequencies.
    useful_bin = np.argmin(np.abs(frequencies_hz - 100.0))
    unwanted_bin = np.argmin(np.abs(frequencies_hz - 400.0))

    # Require at least 90% reduction of the unwanted tone's amplitude.
    assert after[unwanted_bin] < 0.10 * before[unwanted_bin]

    # Require the useful tone's amplitude to stay within 5% of its input.
    useful_ratio = after[useful_bin] / before[useful_bin]
    assert 0.95 <= useful_ratio <= 1.05
