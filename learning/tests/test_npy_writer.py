"""Files written by NpyWriter (controllers/common/npy_writer.h) load with
np.load, holding exactly the dtype, shape and values the C++ side wrote.

Runs the C++ test binary to write them, so build it first (make in
controllers/build).
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO   = Path(__file__).resolve().parents[2]
BINARY = REPO / "controllers/build/test_npy_writer"


@pytest.mark.skipif(not BINARY.exists(), reason=f"{BINARY} not built (make in controllers/build)")
def test_files_load_with_written_values(tmp_path):
    result = subprocess.run([str(BINARY), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    # obs_value() and plan_value() in controllers/tests/test_npy_writer.cpp.
    r, c = np.meshgrid(np.arange(4), np.arange(3), indexing="ij")
    obs = (r * 10.0 + c * 0.5).astype(np.float32)
    r, i, j = np.meshgrid(np.arange(3), np.arange(2), np.arange(3), indexing="ij")
    plan = (r * 100.0 + i * 10.0 + j * 0.25).astype(np.float32)

    for name, expected in [("obs", obs), ("plan", plan)]:
        loaded = np.load(tmp_path / f"{name}.npy")
        assert loaded.dtype == np.float32
        assert loaded.shape == expected.shape
        np.testing.assert_array_equal(loaded, expected)

    empty = np.load(tmp_path / "empty.npy")
    assert empty.dtype == np.float32
    assert empty.shape == (0, 3)


@pytest.mark.skipif(not BINARY.exists(), reason=f"{BINARY} not built (make in controllers/build)")
def test_killed_run_still_loads(tmp_path):
    # A run killed after writing a row but before the header counts it leaves
    # one uncounted row at the end; np.load must return the counted rows.
    subprocess.run([str(BINARY), str(tmp_path)], check=True, capture_output=True)
    path = tmp_path / "obs.npy"
    counted = np.load(path)

    with open(path, "ab") as f:
        f.write(np.ones(3, np.float32).tobytes())

    np.testing.assert_array_equal(np.load(path), counted)
