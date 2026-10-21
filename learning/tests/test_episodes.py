"""The training loop's mppi_sim side (learning/mppi.py, episodes): the task YAML
each round writes, how an episode's end is read from its console output,
gpc's per-round costs from the episodes' CSVs, and removing fitted data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from learning.mppi import (FELL, FELL_LINE, GOAL, GOAL_LINE, OBS_SUFFIX, PLAN_SUFFIX, REPO_ROOT, TIME,
                           delete_dataset_files, episode_outcome, round_costs, write_task_yaml)

TASKS = REPO_ROOT / "controllers/muscle/utils/tasks.yaml"


def test_task_yaml_is_the_task_with_a_policy(tmp_path):
    policy = {"path": "/run/round_000/policy.onnx", "num_samples": 30, "warm_start": 0.0}
    write_task_yaml(tmp_path / "tasks.yaml", "walk", TASKS, policy, threads=6)

    written = yaml.safe_load((tmp_path / "tasks.yaml").read_text())
    assert list(written) == ["walk"]
    walk = written["walk"]
    assert walk.pop("policy") == policy
    assert walk.pop("num_threads") == 6

    source = yaml.safe_load(TASKS.read_text())["walk"]
    source.pop("num_threads")
    assert walk == source


@pytest.mark.parametrize("line, outcome", [(None, TIME),
                                           (f"{FELL_LINE}3.20 s — stopping.", FELL),
                                           (f"{GOAL_LINE}12.73 s — stopping.", GOAL)])
def test_outcome_from_console(line, outcome):
    console = ["Stand-up complete. Body height: 0.312 m", "Converged (avg solve 5.0 ms)."]
    if line:
        console.append(line)
    console.append("Done. Logged to a.csv and a_qpos.csv")
    assert episode_outcome("\n".join(console)) == outcome


def test_console_lines_match_harness():
    harness = (REPO_ROOT / "controllers/common/harness.h").read_text()
    for line in (FELL_LINE, GOAL_LINE):
        assert f'printf("{line}' in harness


def test_round_costs_follow_gpc(tmp_path):
    """Means over every step of every episode; an episode logging no step
    (one that fell before logging) adds nothing."""
    episodes = {"a": {"t": [0.01, 0.02, 0.03], "mppi_min": [10.0, 20.0, 30.0], "policy_min": [5.0, 25.0, 40.0]},
                "b": {"t": [0.01],             "mppi_min": [40.0],             "policy_min": [30.0]},
                "c": {"t": [],                 "mppi_min": [],                 "policy_min": []}}
    for name, columns in episodes.items():
        pd.DataFrame(columns).to_csv(tmp_path / f"{name}.csv", index=False)

    costs = round_costs([tmp_path / name for name in episodes])
    assert costs["mppi_cost"] == pytest.approx(25.0)      # (10 + 20 + 30 + 40) / 4
    assert costs["policy_cost"] == pytest.approx(25.0)    # (5 + 25 + 40 + 30) / 4
    assert costs["policy_best_frac"] == pytest.approx(0.5)


def test_only_dataset_files_are_deleted(tmp_path):
    prefix = tmp_path / "episode_000"
    kept = [Path(f"{prefix}{suffix}") for suffix in (".csv", "_qpos.csv", ".log", ".gif")]
    deleted = [Path(f"{prefix}{suffix}") for suffix in (OBS_SUFFIX, PLAN_SUFFIX)]
    for path in kept + deleted:
        path.write_bytes(np.zeros(1).tobytes())

    delete_dataset_files([prefix, tmp_path / "episode_001"])   # 001 has no files
    assert all(path.exists() for path in kept)
    assert not any(path.exists() for path in deleted)
