"""Training on the stage-3 mppi_sim run (see test_dataset_run.py for how to make
it): fitting one round with config.yaml's settings lowers the loss, and
`python -m learning.training` runs end to end and saves a loadable policy and
ONNX file.
A smoke test of the pipeline on real data, not a claim about policy quality."""

import subprocess
import sys
from pathlib import Path

import onnxruntime as ort
import pytest
import torch
import yaml

from learning.mppi import OBS_SUFFIX, PLAN_SUFFIX, KnotSpline, load_round, repo_path
from learning.policy import Policy
from learning.training import DEFAULT_PATH, fit_round, load_config, make_optimizer

REPO = Path(__file__).resolve().parents[2]
RUN  = REPO / "analysis/data/mppi_sim/mppi_sim"

pytestmark = pytest.mark.skipif(
    not (Path(f"{RUN}{OBS_SUFFIX}").exists() and Path(f"{RUN}{PLAN_SUFFIX}").exists()),
    reason=f"no --dataset run at {RUN} (see test_dataset_run.py)")


def test_fit_lowers_loss():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = Policy(cfg.model).to(device)
    spline = KnotSpline.from_task(cfg.mppi_task.name, repo_path(cfg.mppi_task.tasks_yaml))
    obs, plans, _ = load_round([RUN])

    losses = fit_round(policy, make_optimizer(policy, cfg.fit.learning_rate), obs,
                       spline.from_plan(plans), cfg.fit)
    assert losses[-10:].mean() < losses[:10].mean()


def test_train_command_saves_a_loadable_policy(tmp_path):
    node = yaml.safe_load(DEFAULT_PATH.read_text())
    node["wandb"]["mode"] = "disabled"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(node))

    out = tmp_path / "run"
    result = subprocess.run([sys.executable, "-m", "learning.training", "--config", str(config),
                             "--data", str(RUN), "--out", str(out)],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr
    assert "round 0 | fit/loss" in result.stdout
    Policy.load(out / "policy.pt")
    ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
