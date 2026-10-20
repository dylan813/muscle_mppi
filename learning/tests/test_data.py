"""Loading a round of mppi_sim --dataset episodes (learning/data.py)."""

import numpy as np
import pytest
import torch

from learning.data import OBS_SUFFIX, PLAN_SUFFIX, load_round
from learning.observation import NUM_MUSCLES, OBS_DIM

HORIZON = 40


def write_episode(prefix, rows, value, plan_rows=None):
    np.save(f"{prefix}{OBS_SUFFIX}", np.full((rows, OBS_DIM), value, np.float32))
    np.save(f"{prefix}{PLAN_SUFFIX}", np.full((plan_rows or rows, HORIZON, NUM_MUSCLES), value, np.float32))


def test_episodes_join_in_order(tmp_path):
    write_episode(tmp_path / "ep0", 5, 0.25)
    write_episode(tmp_path / "ep1", 3, 0.75)
    obs, plans, skipped = load_round([tmp_path / "ep0", tmp_path / "ep1"])

    assert obs.shape == (8, OBS_DIM) and plans.shape == (8, HORIZON, NUM_MUSCLES)
    assert obs.dtype == plans.dtype == torch.float32
    assert torch.all(obs[:5] == 0.25) and torch.all(obs[5:] == 0.75)
    assert torch.all(plans[:5] == 0.25) and torch.all(plans[5:] == 0.75)
    assert skipped == []


def test_episode_without_files_is_skipped(tmp_path):
    write_episode(tmp_path / "ep0", 5, 0.5)
    _, _, skipped = load_round([tmp_path / "ep0", tmp_path / "fell_early"])
    assert skipped == [str(tmp_path / "fell_early")]


def test_no_data_at_all_raises(tmp_path):
    with pytest.raises(ValueError, match="no episode data"):
        load_round([tmp_path / "missing"])


def test_episode_with_one_file_raises(tmp_path):
    write_episode(tmp_path / "ep0", 5, 0.5)
    (tmp_path / f"ep0{PLAN_SUFFIX}").unlink()
    with pytest.raises(FileNotFoundError):
        load_round([tmp_path / "ep0"])


def test_row_count_mismatch_raises(tmp_path):
    write_episode(tmp_path / "ep0", 5, 0.5, plan_rows=4)
    with pytest.raises(ValueError, match="5 observations but 4 plans"):
        load_round([tmp_path / "ep0"])
