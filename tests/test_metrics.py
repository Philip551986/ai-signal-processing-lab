
import numpy as np
import pytest

# Import the packaged functions instead of redefining them here.
from signal_lab.metrics import rms, compare_rms


def test_constant_rms():
    # A constant signal with amplitude 2 must have RMS 2.
    samples = np.full(100, 2.0)
    assert rms(samples) == pytest.approx(2.0)


def test_sine_rms():
    # Sample a 100 Hz unit-amplitude sine for one second at 2000 Hz.
    fs_hz = 2000
    time_s = np.arange(fs_hz) / fs_hz
    samples = np.sin(2 * np.pi * 100 * time_s)

    # A unit sine spanning whole cycles has RMS 1/sqrt(2).
    assert rms(samples) == pytest.approx(1 / np.sqrt(2))


def test_silence_rms():
    # Silence has zero RMS and should not cause division by zero.
    assert rms(np.zeros(100)) == 0.0


@pytest.mark.parametrize("bad_samples", [
    [],                    # Empty input.
    [[1.0, 2.0]],          # Two-dimensional input.
    [1.0, np.nan],         # A missing numerical value.
    [1.0, np.inf],         # An infinite value.
    [1.0 + 1.0j],          # Complex samples.
])
def test_invalid_rms_input(bad_samples):
    # Each invalid input must raise ValueError.
    with pytest.raises(ValueError):
        rms(bad_samples)


def test_comparison_requires_matching_shapes():
    # Comparing signals of different lengths must raise ValueError.
    with pytest.raises(ValueError, match="shapes must match"):
        compare_rms(np.ones(100), np.ones(80))
