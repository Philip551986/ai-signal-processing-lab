
import numpy as np
import pytest

# Use the actual packaged validation and filtering functions.
from signal_lab.validation import validate_signal
from signal_lab.filters import apply_filter


@pytest.mark.parametrize("bad_samples", [
    np.arange(10, dtype=float),       # Too few samples.
    np.ones(100),                     # Constant signal.
    np.zeros(100),                    # Silence.
    np.ones((100, 2)),                # Two channels instead of one.
    np.concatenate([np.arange(99), [np.nan]]),  # Missing value.
    np.concatenate([np.arange(99), [np.inf]]),  # Infinite value.
])
def test_invalid_signals(bad_samples):
    # Invalid signals must raise ValueError.
    with pytest.raises(ValueError):
        validate_signal(bad_samples, fs_hz=2000.0)


@pytest.mark.parametrize("bad_fs_hz", [
    0.0,       # Zero sampling rate.
    -2000.0,   # Negative sampling rate.
    np.nan,    # Missing sampling rate.
    np.inf,    # Infinite sampling rate.
])
def test_invalid_sampling_rates(bad_fs_hz):
    # Use a varying signal so only the sampling rate is invalid.
    samples = np.arange(100, dtype=float)

    with pytest.raises(ValueError):
        validate_signal(samples, fs_hz=bad_fs_hz)


@pytest.mark.parametrize("bad_cutoff_hz", [
    0.0,       # Cutoff must be above zero.
    -100.0,    # Negative cutoff is invalid.
    1000.0,    # Exactly Nyquist for a 2000 Hz sampling rate.
    1200.0,    # Above Nyquist.
    np.nan,    # Missing cutoff.
])
def test_invalid_lowpass_cutoffs(bad_cutoff_hz):
    samples = np.arange(100, dtype=float)

    with pytest.raises(ValueError):
        apply_filter(
            samples,
            fs_hz=2000.0,
            filter_type="lowpass",
            cutoff_hz=bad_cutoff_hz,
        )


def test_reversed_bandpass_edges():
    samples = np.arange(100, dtype=float)

    # Band edges must be supplied from lower to higher frequency.
    with pytest.raises(ValueError):
        apply_filter(
            samples,
            fs_hz=2000.0,
            filter_type="bandpass",
            cutoff_hz=[500.0, 100.0],
        )


def test_unknown_filter_type():
    samples = np.arange(100, dtype=float)

    # An unsupported filter name must be rejected.
    with pytest.raises(ValueError):
        apply_filter(
            samples,
            fs_hz=2000.0,
            filter_type="unknown",
        )
