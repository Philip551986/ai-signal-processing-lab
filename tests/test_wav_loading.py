
import numpy as np
import pytest
from scipy.io import wavfile

from signal_lab.data_io import load_wav


@pytest.mark.parametrize("dtype, stored, expected", [
    # Unsigned 8-bit audio is centred on 128.
    (np.uint8, [0, 128, 255], [-1.0, 0.0, 127 / 128]),

    # Signed 16-bit audio is divided by 32768.
    (np.int16, [-32768, 0, 16384], [-1.0, 0.0, 0.5]),

    # Signed 32-bit audio is divided by 2147483648.
    (np.int32, [-2147483648, 0, 1073741824], [-1.0, 0.0, 0.5]),

    # Floating-point amplitudes should remain unchanged.
    (np.float32, [-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5]),
])
def test_wav_amplitude_conversion(tmp_path, dtype, stored, expected):
    # Repeat the pattern to exceed our minimum signal length.
    raw_samples = np.tile(np.array(stored, dtype=dtype), 32)

    # pytest supplies a temporary folder through tmp_path.
    path = tmp_path / "format_check.wav"

    # Create a WAV with a known sampling rate and stored amplitudes.
    wavfile.write(path, 2000, raw_samples)

    # Load it using our packaged function.
    samples, fs_hz = load_wav(path)

    # Check the rate, sample count, and converted amplitudes.
    assert fs_hz == 2000.0
    assert samples.shape == raw_samples.shape
    np.testing.assert_allclose(samples, np.tile(expected, 32))


def test_stereo_channel_selection(tmp_path):
    # Create two distinct channels with dimensionless amplitudes.
    left = np.tile([-0.5, 0.5], 50)
    right = np.tile([-0.25, 0.25], 50)

    # Each column represents one audio channel.
    stereo = np.column_stack([left, right]).astype(np.float32)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 2000, stereo)

    # Channel 1 is the second channel because numbering starts at 0.
    samples, fs_hz = load_wav(path, channel=1)

    # Confirm that the loader selected the right channel.
    np.testing.assert_allclose(samples, right)
    assert fs_hz == 2000.0


def test_missing_channel_is_rejected(tmp_path):
    # Create mono audio, which has only channel 0.
    mono = np.tile([-0.5, 0.5], 50).astype(np.float32)
    path = tmp_path / "mono.wav"
    wavfile.write(path, 2000, mono)

    # Requesting channel 1 must produce a clear error.
    with pytest.raises(ValueError, match="Channel must be between"):
        load_wav(path, channel=1)
