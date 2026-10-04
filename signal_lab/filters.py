"""Offline filter application for the signal-processing lab."""

import numpy as np
from scipy import signal
from .validation import validate_signal


def apply_filter(x, fs_hz, filter_type, cutoff_hz=None,
                 order=4, notch_q=30.0):
    """Return filtered samples in the same amplitude units as x."""

    # Use the shared input validator.
    x = validate_signal(x, fs_hz)

    # Return an independent copy when no filtering is requested.
    if filter_type == "none":
        return x.copy()

    if filter_type in ("lowpass", "highpass", "bandpass", "bandstop"):
        # Require a positive integer Butterworth design order.
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
            raise ValueError("Filter order must be a positive integer.")
        if order < 1:
            raise ValueError("Filter order must be a positive integer.")

        # Convert cutoff settings into an array for checking.
        edges_hz = np.atleast_1d(cutoff_hz).astype(float)

        if edges_hz.ndim != 1 or not np.all(np.isfinite(edges_hz)):
            raise ValueError("Cutoffs must be a finite one-dimensional array.")

        if np.any(edges_hz <= 0) or np.any(edges_hz >= fs_hz / 2):
            raise ValueError("Cutoffs must be above zero and below Nyquist.")

        expected_edges = 2 if filter_type in ("bandpass", "bandstop") else 1

        if len(edges_hz) != expected_edges:
            raise ValueError("Incorrect number of cutoff frequencies.")

        if expected_edges == 2 and edges_hz[0] >= edges_hz[1]:
            raise ValueError("The lower cutoff must be below the upper cutoff.")

        # Design the Butterworth filter in second-order sections.
        sos = signal.butter(
            order,
            edges_hz,
            btype=filter_type,
            fs=fs_hz,
            output="sos",
        )

    elif filter_type == "notch":
        # A notch needs exactly one centre frequency.
        centers_hz = np.atleast_1d(cutoff_hz).astype(float)

        if centers_hz.ndim != 1 or centers_hz.size != 1:
            raise ValueError("A notch requires one centre frequency.")

        center_hz = float(centers_hz[0])

        if not np.isfinite(center_hz) or not 0 < center_hz < fs_hz / 2:
            raise ValueError("Notch centre must be above zero and below Nyquist.")

        if not np.isfinite(notch_q) or notch_q <= 0:
            raise ValueError("Notch Q must be finite and positive.")

        # Design the second-order notch and convert it to SOS.
        b, a = signal.iirnotch(center_hz, Q=notch_q, fs=fs_hz)
        sos = signal.tf2sos(b, a)

    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    try:
        # Apply forward-backward filtering for offline zero-phase output.
        y = signal.sosfiltfilt(sos, x)
    except ValueError as error:
        raise ValueError(
            "Filtering failed; the recording may be too short for padding."
        ) from error

    if not np.all(np.isfinite(y)):
        raise ValueError("Filtering produced nonfinite samples.")

    return y
