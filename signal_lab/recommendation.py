"""Explainable filter recommendations based on detected peaks."""

import numpy as np


def recommend_filter(detected_hz, fs_hz, useful_low_hz, useful_high_hz):
    """Return a candidate filter, cutoff in Hz and decision explanation."""

    # Validate the sampling rate and useful-band boundaries.
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("Sampling rate must be finite and positive.")

    if not 0 <= useful_low_hz < useful_high_hz <= fs_hz / 2:
        raise ValueError("Useful band must lie between zero and Nyquist.")

    # Require a one-dimensional array of valid peak frequencies.
    detected_hz = np.asarray(detected_hz, dtype=float)

    if detected_hz.ndim != 1 or not np.all(np.isfinite(detected_hz)):
        raise ValueError("Detected frequencies must be a finite 1D array.")

    if np.any(detected_hz < 0) or np.any(detected_hz > fs_hz / 2):
        raise ValueError("Detected frequencies must lie between zero and Nyquist.")

    # Identify peaks outside the declared useful band.
    below_hz = detected_hz[detected_hz < useful_low_hz]
    above_hz = detected_hz[detected_hz > useful_high_hz]
    candidate_hz = np.concatenate((below_hz, above_hz))

    # Start with no filtering, then update the recommendation if needed.
    filter_type = "none"
    cutoff_hz = None
    reason = "No prominent peaks detected outside the useful band."

    if len(candidate_hz) == 1:
        center_hz = float(candidate_hz[0])

        # A notch cannot be designed exactly at DC or Nyquist.
        if 0 < center_hz < fs_hz / 2:
            filter_type = "notch"
            cutoff_hz = center_hz
            reason = "One isolated candidate interference tone was detected."
        else:
            reason = "Candidate lies at DC or Nyquist; review before filtering."

    elif len(below_hz) > 0 and len(above_hz) > 0:
        filter_type = "bandpass"
        cutoff_hz = [float(useful_low_hz), float(useful_high_hz)]
        reason = "Candidate interference peaks occur on both sides of the band."

    elif len(above_hz) > 0:
        filter_type = "lowpass"
        cutoff_hz = float(useful_high_hz)
        reason = "Multiple candidate interference peaks occur above the band."

    elif len(below_hz) > 0:
        filter_type = "highpass"
        cutoff_hz = float(useful_low_hz)
        reason = "Multiple candidate interference peaks occur below the band."

    # Return the settings and their explanation together.
    return {
        "filter_type": filter_type,
        "cutoff_hz": cutoff_hz,
        "candidate_hz": candidate_hz.tolist(),
        "reason": reason,
    }
