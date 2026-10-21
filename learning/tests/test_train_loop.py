"""gpc's training loop, `python -m learning.training train`, end to end on a
small setting: 2 rounds of 2 walk episodes, each 1 s long. Each round collects
with that round's exported policy, fits, deletes the fitted data, and logs;
checkpoints follow gpc's rule.

Runs mppi_sim, so build it first (make in controllers/build). Takes about a
minute.
"""

import subprocess
import sys
from pathlib import Path

import onnxruntime as ort
import pytest
import yaml

from learning.mppi import MPPI_SIM, OBS_SUFFIX, PLAN_SUFFIX, REPO_ROOT
from learning.policy import Policy
from learning.training import DEFAULT_PATH

ROUNDS, EPISODES = 2, 2

pytestmark = pytest.mark.skipif(not MPPI_SIM.exists(),
                                reason=f"{MPPI_SIM} not built (make in controllers/build)")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The run's output folder and console output."""
    tmp = tmp_path_factory.mktemp("loop")
    tasks = yaml.safe_load((REPO_ROOT / "controllers/muscle/utils/tasks.yaml").read_text())
    tasks["walk"]["sim_duration"] = 1.0
    (tmp / "tasks.yaml").write_text(yaml.safe_dump(tasks))

    node = yaml.safe_load(DEFAULT_PATH.read_text())
    node["mppi_task"]["tasks_yaml"] = str(tmp / "tasks.yaml")
    node["loop"].update(num_rounds=ROUNDS, episodes_per_round=EPISODES, parallel_episodes=2,
                        threads_per_episode=4, gifs_per_round=1)
    node["checkpoint_every"] = 1
    node["wandb"]["mode"] = "disabled"
    (tmp / "config.yaml").write_text(yaml.safe_dump(node))

    out = tmp / "run"
    result = subprocess.run([sys.executable, "-m", "learning.training", "train",
                             "--config", str(tmp / "config.yaml"), "--out", str(out)],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr
    return out, result.stdout


@pytest.mark.parametrize("i", range(ROUNDS))
def test_round_collects_with_its_policy(run, i):
    out, _ = run
    round_dir = out / f"round_{i:03d}"
    ort.InferenceSession(str(round_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
    policy = yaml.safe_load((round_dir / "tasks.yaml").read_text())["walk"]["policy"]
    assert Path(policy["path"]) == (round_dir / "policy.onnx").resolve()

    for j in range(EPISODES):
        episode = round_dir / f"episode_{j:03d}"
        assert Path(f"{episode}.csv").exists() and Path(f"{episode}.log").exists()
        assert not Path(f"{episode}{OBS_SUFFIX}").exists()     # fitted, then deleted
        assert not Path(f"{episode}{PLAN_SUFFIX}").exists()
    assert (round_dir / "episode_000.gif").exists()           # gifs_per_round: 1
    assert not (round_dir / "episode_001.gif").exists()


@pytest.mark.parametrize("i", range(ROUNDS))
def test_round_is_logged(run, i):
    _, stdout = run
    line = next(line for line in stdout.splitlines() if line.startswith(f"round {i} |"))
    for metric in ["sim/policy_cost", "sim/mppi_cost", "sim/policy_best_frac", "sim/time",
                   "sim/fell", "sim/reached_goal", "sim/rows", "fit/loss", "fit/time"]:
        assert metric in line


def test_checkpoints_and_final_policy(run):
    out, _ = run
    assert not (out / "policy_ckpt_0.pt").exists()   # gpc: never after round 0
    Policy.load(out / "policy_ckpt_1.pt")
    Policy.load(out / "policy.pt")
    ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])


def test_existing_output_folder_is_refused(run):
    out, _ = run
    result = subprocess.run([sys.executable, "-m", "learning.training", "train", "--out", str(out)],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode != 0
    assert "exists" in result.stderr
