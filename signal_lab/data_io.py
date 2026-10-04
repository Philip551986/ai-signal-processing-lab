
"""Functions for loading signal files."""

import numpy as np
from scipy.io import wavfile

# Reuse the existing validator.
from .validation import validate_signal


def load_wav(path, channel=0):
    """Return one audio channel and its sampling rate in Hz."""

    # Read the sampling rate and original sample array.
    fs_hz, raw = wavfile.read(path)

    # Mono has one channel; multichannel audio has columns.
    if raw.ndim == 1:
        channel_count = 1
    elif raw.ndim == 2:
        channel_count = raw.shape[1]
    else:
        raise ValueError("Expected mono or multichannel WAV data.")

    # Channel numbers must be integers, starting from zero.
    if isinstance(channel, (bool, np.bool_)) or not isinstance(
        channel, (int, np.integer)
    ):
        raise ValueError("Channel must be an integer.")

    # Reject a channel that does not exist.
    if not 0 <= channel < channel_count:
        raise ValueError(
            f"Channel must be between 0 and {channel_count - 1}."
        )

    # Select the requested channel without averaging channels.
    selected = raw if raw.ndim == 1 else raw[:, channel]

    # Convert unsigned 8-bit audio: its zero level is 128.
    if selected.dtype == np.uint8:
        samples = (selected.astype(float) - 128.0) / 128.0

    # Convert signed integer audio using its storage bit width.
    elif np.issubdtype(selected.dtype, np.signedinteger):
        bits = np.iinfo(selected.dtype).bits
        samples = selected.astype(float) / (2 ** (bits - 1))

    # Floating-point audio already contains amplitude values.
    elif np.issubdtype(selected.dtype, np.floating):
        samples = selected.astype(float)

    # Reject any sample format we have not accounted for.
    else:
        raise ValueError("Unsupported WAV sample data type.")

    # Check the selected signal using our shared validation rules.
    samples = validate_signal(samples, fs_hz)

    # Return amplitude samples and sampling rate in Hz.
    return samples, float(fs_hz)



def load_csv(path, fs_hz=None):
    """Return amplitude samples and sampling rate in Hz."""
    import csv

    # Read named columns using Python's built-in CSV reader.
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        # Require the column name used by our lab.
        if not reader.fieldnames or "amplitude" not in reader.fieldnames:
            raise ValueError("CSV must contain an amplitude column.")

        # Store rows before closing the file.
        rows = list(reader)
        has_time = "time_s" in reader.fieldnames

    # Reject files that contain headers but no samples.
    if not rows:
        raise ValueError("CSV contains no signal samples.")

    # Convert amplitude text into floating-point numbers.
    try:
        samples = np.array(
            [float(row["amplitude"]) for row in rows], dtype=float
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Amplitude values must be numeric.") from error

    # Validate an explicitly supplied sampling rate.
    if fs_hz is not None:
        fs_hz = float(fs_hz)
        if not np.isfinite(fs_hz) or fs_hz <= 0:
            raise ValueError("Sampling rate must be finite and positive.")

    # Infer the rate when timestamps are present.
    if has_time:
        try:
            time_s = np.array(
                [float(row["time_s"]) for row in rows], dtype=float
            )
        except (TypeError, ValueError) as error:
            raise ValueError("time_s values must be numeric.") from error

        # At least two finite timestamps are needed for a time interval.
        if len(time_s) < 2 or not np.all(np.isfinite(time_s)):
            raise ValueError("Use at least two finite timestamps.")

        # Calculate intervals between consecutive samples, in seconds.
        intervals_s = np.diff(time_s)

        # Timestamps must move forward without duplicates.
        if np.any(intervals_s <= 0):
            raise ValueError("Timestamps must be strictly increasing.")

        # Use the median interval as the representative sample interval.
        step_s = float(np.median(intervals_s))

        # Permit small rounding differences, but reject uneven sampling.
        if not np.allclose(
            intervals_s, step_s, rtol=1e-3, atol=step_s * 1e-6
        ):
            raise ValueError("Timestamps must be uniformly spaced.")

        # Sampling rate in Hz is the reciprocal of the interval in seconds.
        inferred_fs_hz = 1.0 / step_s

        # Reject a supplied rate that disagrees with the timestamps.
        if fs_hz is not None and not np.isclose(
            fs_hz, inferred_fs_hz, rtol=1e-3, atol=0.0
        ):
            raise ValueError("Sampling rate conflicts with time_s.")

        fs_hz = inferred_fs_hz

    # Without timestamps, the caller must provide the sampling rate.
    elif fs_hz is None:
        raise ValueError("Provide fs_hz when CSV has no time_s column.")

    # Reuse our existing signal validation.
    samples = validate_signal(samples, fs_hz)

    # Keep amplitudes in their original units and return the rate in Hz.
    return samples, float(fs_hz)



def save_processed_wav(path, samples, fs_hz):
    """Save mono digital audio; return the amplitude divisor used."""
    from pathlib import Path

    # Reject complex samples before converting to real numbers.
    if np.iscomplexobj(samples):
        raise ValueError("Audio samples must be real-valued.")

    # Convert samples into a floating-point array.
    audio = np.asarray(samples, dtype=float)

    # Export requires a nonempty, one-dimensional signal.
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Export requires nonempty mono audio.")

    # Missing and infinite values cannot be exported.
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio samples must be finite.")

    # WAV sampling rates must be positive whole numbers, in Hz.
    rate_hz = float(fs_hz)
    if (
        not np.isfinite(rate_hz)
        or rate_hz <= 0
        or not rate_hz.is_integer()
        or rate_hz > 4294967295
    ):
        raise ValueError("WAV sampling rate must be a valid positive integer.")

    # Require a WAV filename.
    output_path = Path(path)
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output filename must end in .wav.")

    # Find the largest absolute amplitude.
    peak = float(np.max(np.abs(audio)))

    # Reduce amplitudes only when their peak exceeds 1.
    amplitude_divisor = max(1.0, peak)

    # Map the resulting amplitudes into signed 16-bit integers.
    pcm = np.round(
        (audio / amplitude_divisor) * 32767.0
    ).astype(np.int16)

    # Save the samples with the supplied sampling rate.
    wavfile.write(output_path, int(rate_hz), pcm)

    # Report scaling so the caller can explain the exported amplitude.
    return amplitude_divisor
