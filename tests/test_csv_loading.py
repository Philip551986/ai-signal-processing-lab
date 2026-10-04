
import numpy as np
import pytest

from signal_lab.data_io import load_csv


def test_csv_with_timestamps(tmp_path):
    # Create 100 sample times in seconds at 2000 Hz.
    time_s = np.arange(100) / 2000.0

    # Use varying amplitudes to satisfy our validation rules.
    amplitudes = np.arange(100, dtype=float)
    path = tmp_path / "with_time.csv"

    # Write named time and amplitude columns.
    np.savetxt(
        path,
        np.column_stack([time_s, amplitudes]),
        delimiter=",",
        header="time_s,amplitude",
        comments="",
    )

    # The loader should infer the rate from timestamps.
    samples, fs_hz = load_csv(path)

    # Check amplitude preservation and the inferred rate.
    np.testing.assert_allclose(samples, amplitudes)
    assert fs_hz == pytest.approx(2000.0)


def test_csv_without_timestamps(tmp_path):
    amplitudes = np.arange(100, dtype=float)
    path = tmp_path / "amplitude_only.csv"

    # This file contains amplitude values only.
    np.savetxt(
        path,
        amplitudes,
        delimiter=",",
        header="amplitude",
        comments="",
    )

    # Supply the sampling rate because the file has no timestamps.
    samples, fs_hz = load_csv(path, fs_hz=2000.0)

    np.testing.assert_allclose(samples, amplitudes)
    assert fs_hz == 2000.0


def test_missing_sampling_rate(tmp_path):
    path = tmp_path / "missing_rate.csv"

    # There are enough samples, but no timing information.
    np.savetxt(
        path,
        np.arange(100),
        header="amplitude",
        comments="",
    )

    # The loader must not guess a sampling rate.
    with pytest.raises(ValueError, match="Provide fs_hz"):
        load_csv(path)


def test_conflicting_sampling_rate(tmp_path):
    time_s = np.arange(100) / 2000.0
    path = tmp_path / "conflicting_rate.csv"

    np.savetxt(
        path,
        np.column_stack([time_s, np.arange(100)]),
        delimiter=",",
        header="time_s,amplitude",
        comments="",
    )

    # The timestamps indicate 2000 Hz, so 1000 Hz must be rejected.
    with pytest.raises(ValueError, match="conflicts"):
        load_csv(path, fs_hz=1000.0)


def test_uneven_timestamps(tmp_path):
    time_s = np.arange(100) / 2000.0

    # Shift later timestamps, creating one unusually large interval.
    time_s[50:] += 0.001
    path = tmp_path / "uneven_time.csv"

    np.savetxt(
        path,
        np.column_stack([time_s, np.arange(100)]),
        delimiter=",",
        header="time_s,amplitude",
        comments="",
    )

    # Our current processing requires uniformly sampled data.
    with pytest.raises(ValueError, match="uniformly spaced"):
        load_csv(path)


def test_missing_amplitude_column(tmp_path):
    path = tmp_path / "wrong_header.csv"

    # Use a column name that our loader does not accept.
    path.write_text("voltage\n1\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="amplitude column"):
        load_csv(path, fs_hz=2000.0)
