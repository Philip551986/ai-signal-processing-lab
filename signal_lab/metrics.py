"""Signal comparison measurements."""

import numpy as np


def rms(x):
    """Return root-mean-square amplitude in the same units as x."""

    # Reject complex samples before converting them to real numbers.
    if np.iscomplexobj(x):
        raise ValueError("RMS expects real-valued samples.")

    samples = np.asarray(x, dtype=float)

    # Require a nonempty, single-channel signal.
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("RMS requires a nonempty one-dimensional signal.")

    if not np.all(np.isfinite(samples)):
        raise ValueError("RMS samples must be finite.")

    # Scaling avoids overflow when squaring large sample values.
    scale = float(np.max(np.abs(samples)))

    # Silence has zero RMS; avoid dividing by zero.
    if scale == 0:
        return 0.0

    return float(scale * np.sqrt(np.mean((samples / scale) ** 2)))


def compare_rms(original, filtered):
    """Return input and output RMS values for equal-length signals."""

    if np.asarray(original).shape != np.asarray(filtered).shape:
        raise ValueError("Original and filtered signal shapes must match.")

    return {
        "rms_before": rms(original),
        "rms_after": rms(filtered),
    }
