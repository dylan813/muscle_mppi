"""Training settings, read from learning/config.yaml (see it for each value's source)."""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).with_name("config.yaml")
REPO_ROOT    = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    """A repo-relative path made absolute (absolute paths pass through), as the
    C++ side's repo_path() does (controllers/common/task_config.h)."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class MPPITaskConfig:
    name: str
    tasks_yaml: str


@dataclass
class ModelConfig:
    feature_dims: list[int]
    timestep_embedding_dim: int
    kernel_size: int


@dataclass
class FitConfig:
    learning_rate: float
    batch_size: int
    num_epochs: int


@dataclass
class WandbConfig:
    project: str
    run_name: str | None
    mode: str


@dataclass
class Config:
    seed: int
    mppi_task: MPPITaskConfig
    model: ModelConfig
    fit: FitConfig
    checkpoint_every: int
    wandb: WandbConfig


def load_config(path: Path = DEFAULT_PATH) -> Config:
    return _build(Config, yaml.safe_load(Path(path).read_text()), str(path))


def _build(cls, node, where: str):
    """cls from a YAML mapping whose keys are exactly cls's fields."""
    if not isinstance(node, dict):
        raise ValueError(f"{where}: expected a mapping, got {node!r}")

    names = {f.name for f in fields(cls)}
    if unknown := node.keys() - names:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")
    if missing := names - node.keys():
        raise ValueError(f"{where}: missing key(s) {sorted(missing)}")

    values = {}
    for f in fields(cls):
        value = node[f.name]
        if is_dataclass(f.type):
            value = _build(f.type, value, f"{where}: {f.name}")
        elif f.type in (int, float):
            value = f.type(value)
        values[f.name] = value
    return cls(**values)
