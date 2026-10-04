"""Frequency analysis for the signal-processing lab."""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from .validation import validate_signal


def calculate_fft(x, fs_hz):
    """Return frequencies in Hz and estimated sinusoidal amplitudes."""

    # Use the shared validator before performing calculations.
    x = validate_signal(x, fs_hz)
    sample_count = len(x)

    # Remove DC for analysis without changing the original samples.
    centered = x - np.mean(x)

    # Reduce spectral leakage with a periodic Hann window.
    window = signal.windows.hann(sample_count, sym=False)

    # Calculate the one-sided transform and its frequency labels.
    transformed = rfft(centered * window)
    frequency_hz = rfftfreq(sample_count, d=1 / fs_hz)

    # Compensate for the window's amplitude scaling.
    amplitude = np.abs(transformed) / np.sum(window)

    # Double positive-frequency magnitudes, excluding DC and Nyquist.
    if sample_count % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2

    return frequency_hz, amplitude


def detect_peaks(x, fs_hz, relative_threshold=0.10,
                 minimum_separation_hz=5.0):
    """Return frequencies, amplitudes and peak indices ranked by amplitude."""

    # Require sensible detection settings.
    if not np.isfinite(relative_threshold) or not 0 < relative_threshold <= 1:
        raise ValueError("Relative threshold must be greater than 0 and at most 1.")

    if not np.isfinite(minimum_separation_hz) or minimum_separation_hz <= 0:
        raise ValueError("Minimum peak separation must be finite and positive.")

    # Use the shared FFT function; it also validates the samples.
    frequency_hz, amplitude = calculate_fft(x, fs_hz)

    # Convert the desired separation from Hz into FFT bins.
    bin_spacing_hz = fs_hz / len(x)
    distance_bins = max(
        1, int(np.ceil(minimum_separation_hz / bin_spacing_hz))
    )

    # Set the minimum peak height relative to the largest amplitude.
    threshold = relative_threshold * np.max(amplitude)

    # Detect peaks using the same rules as the original notebook.
    peak_indices, _ = signal.find_peaks(
        amplitude,
        height=threshold,
        prominence=threshold / 2,
        distance=distance_bins,
    )

    # Rank detected peaks from strongest to weakest.
    ranking = np.argsort(amplitude[peak_indices])[::-1]
    peak_indices = peak_indices[ranking]

    return frequency_hz, amplitude, peak_indices
