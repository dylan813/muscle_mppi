"""Settings (learning/training.py): the repo's config.yaml loads, and an unknown
or missing key is an error rather than silently ignored."""

import pytest
import yaml

from learning.training import DEFAULT_PATH, load_config


def write_modified(tmp_path, change):
    node = yaml.safe_load(DEFAULT_PATH.read_text())
    change(node)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(node))
    return path


def test_repo_config_loads():
    cfg = load_config()
    assert cfg.seed == 0
    assert cfg.mppi_task.name == "walk"
    assert cfg.mppi_task.tasks_yaml == "controllers/muscle/utils/tasks.yaml"
    assert cfg.model.feature_dims == [128, 128, 128]
    assert cfg.model.timestep_embedding_dim == 64
    assert cfg.fit.learning_rate == 1e-3 and isinstance(cfg.fit.learning_rate, float)
    assert cfg.fit.batch_size == 128 and cfg.fit.num_epochs == 10
    assert cfg.checkpoint_every == 1
    assert cfg.wandb.project == "muscle-mppi-gpc"


def test_unknown_key_raises(tmp_path):
    path = write_modified(tmp_path, lambda n: n.update(extra=1))
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_unknown_nested_key_raises(tmp_path):
    path = write_modified(tmp_path, lambda n: n["fit"].update(lerning_rate=1e-3))
    with pytest.raises(ValueError, match="fit.*unknown key"):
        load_config(path)


def test_missing_key_raises(tmp_path):
    path = write_modified(tmp_path, lambda n: n.pop("seed"))
    with pytest.raises(ValueError, match="missing key"):
        load_config(path)
