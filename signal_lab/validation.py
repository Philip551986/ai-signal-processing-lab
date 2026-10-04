"""Input validation for the signal-processing lab."""

import numpy as np  # This module imports its own dependency.


def validate_signal(x, fs_hz):
    """Check the samples and sampling rate before analysis."""

    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Select one signal channel.")

    if len(x) < 64:
        raise ValueError("Use at least 64 samples for this lab.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains missing or infinite values.")

    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("Sampling rate must be finite and positive.")

    if np.ptp(x) == 0:
        raise ValueError("Signal is constant.")

    return x
