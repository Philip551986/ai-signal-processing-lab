
import numpy as np
import pytest
from scipy.io import wavfile

from signal_lab.data_io import save_processed_wav


@pytest.mark.parametrize("peak, expected_divisor", [
    (0.5, 1.0),   # Audio within range should not be amplified.
    (1.5, 1.5),   # Audio above range should be scaled down.
    (0.0, 1.0),   # Silence should export without division by zero.
])
def test_exported_waveform(tmp_path, peak, expected_divisor):
    # Use a known waveform with the specified peak amplitude.
    samples = np.tile([-peak, 0.0, peak], 40)
    path = tmp_path / "processed.wav"

    # Export using the packaged function.
    divisor = save_processed_wav(path, samples, fs_hz=2000)

    # Read the saved file independently.
    saved_fs_hz, saved_samples = wavfile.read(path)

    # Check the reported scaling, sampling rate, and audio format.
    assert divisor == pytest.approx(expected_divisor)
    assert saved_fs_hz == 2000
    assert saved_samples.dtype == np.int16
    assert saved_samples.shape == samples.shape

    # Compare the stored waveform with the expected scaled input.
    # Allow one integer step of quantisation error.
    np.testing.assert_allclose(
        saved_samples.astype(float) / 32767.0,
        samples / expected_divisor,
        atol=1.0 / 32767.0,
    )


@pytest.mark.parametrize("bad_samples", [
    [],                  # Empty audio.
    [[0.0, 0.5]],        # Two-dimensional audio.
    [0.0, np.nan],       # Missing value.
    [0.0, np.inf],       # Infinite value.
    [1.0 + 1.0j],        # Complex value.
])
def test_invalid_export_samples(tmp_path, bad_samples):
    path = tmp_path / "invalid.wav"

    # Invalid samples must be rejected before a file is written.
    with pytest.raises(ValueError):
        save_processed_wav(path, bad_samples, fs_hz=2000)

    assert not path.exists()


@pytest.mark.parametrize("bad_fs_hz", [
    0.0,       # Zero rate.
    -2000.0,   # Negative rate.
    2000.5,    # WAV export requires a whole-number rate.
    np.nan,    # Missing rate.
    np.inf,    # Infinite rate.
])
def test_invalid_export_rates(tmp_path, bad_fs_hz):
    path = tmp_path / "invalid_rate.wav"

    with pytest.raises(ValueError):
        save_processed_wav(path, [0.0, 0.5], fs_hz=bad_fs_hz)

    assert not path.exists()


def test_wrong_output_extension(tmp_path):
    path = tmp_path / "processed.mp3"

    # Our function writes WAV audio, so reject an MP3 filename.
    with pytest.raises(ValueError, match="end in .wav"):
        save_processed_wav(path, [0.0, 0.5], fs_hz=2000)

    assert not path.exists()
