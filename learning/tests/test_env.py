"""Checks that the `muscle` conda env runs everything Python in muscle_mppi
outside analysis/: the run_*.sh YAML helpers, mppi_sim's GIF rendering
(analysis/render_gif.py), and the planned learning/ stack (torch → ONNX →
ONNX Runtime).

    conda activate muscle
    python -m pytest -v learning/tests      # from the repo root
"""

import os

# MuJoCo picks its GL backend at import; mppi_sim renders GIFs headless with EGL.
os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import yaml
from PIL import Image

REPO       = Path(__file__).resolve().parents[2]   # learning/tests/ -> repo root
TASKS_YAML = REPO / "controllers/muscle/utils/tasks.yaml"
GO2_SCENE  = REPO / "unitree_mujoco/unitree_robots/go2/scene.xml"
QPOS_LOG   = REPO / "analysis/data/mppi_sim/mppi_sim_qpos.csv"   # an existing mppi_sim run


# ── env isolation ─────────────────────────────────────────────────────────────

def test_env_is_active_and_isolated():
    assert Path(sys.prefix).name == "muscle", f"active env is {sys.prefix}"
    assert os.environ.get("PYTHONNOUSERSITE") == "1"
    assert os.environ.get("LD_LIBRARY_PATH", None) == ""
    assert os.environ.get("PYTHONPATH", None) == ""

    # Every import path is the env's, except the working and test directories
    # Python and pytest add (catches ~/.local and ROS's /opt/ros/... leaking in).
    local = {"", os.getcwd(), str(Path(__file__).parent)}
    foreign = [p for p in sys.path if p not in local and not p.startswith(sys.prefix)]
    assert not foreign, f"import paths outside the env: {foreign}"


def test_python3_resolves_to_env():
    # mppi_sim and the run_*.sh scripts call plain `python3`.
    assert Path(shutil.which("python3")).resolve().is_relative_to(Path(sys.prefix).resolve())


def test_pinned_versions():
    assert torch.__version__ == "2.7.1+cu128"
    assert mujoco.__version__ == "3.3.6"         # C++ libmujoco.so.3.3.6
    assert ort.__version__.startswith("1.22.")   # C++ ONNX Runtime 1.22.2


# ── run_*.sh YAML helpers ─────────────────────────────────────────────────────

def test_tasks_yaml_round_trip(tmp_path):
    # run_breakdown.sh / run_bandwidth.sh: safe_load the task file, safe_dump a copy.
    cfg = yaml.safe_load(TASKS_YAML.read_text())
    out = tmp_path / "tasks.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    assert yaml.safe_load(out.read_text()) == cfg
    assert len(cfg["walk"]["muscle"]["peak_force"]) == 12   # anchor *default_muscle_quad expanded


# ── MuJoCo and mppi_sim's GIF rendering ──────────────────────────────────────

def test_mujoco_steps_go2():
    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
    data  = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_mujoco_egl_render():
    model = mujoco.MjModel.from_xml_path(str(GO2_SCENE))
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=120, width=160) as renderer:
        renderer.update_scene(data)
        pixels = renderer.render()
    assert pixels.shape == (120, 160, 3) and pixels.any()


@pytest.mark.skipif(not QPOS_LOG.exists(), reason=f"no qpos log at {QPOS_LOG}")
def test_render_gif_script(tmp_path):
    # The exact command mppi_sim runs after a rollout (render_rollout_gif, common/log.h).
    gif = tmp_path / "env_test.gif"
    cmd = ["python3", str(REPO / "analysis/render_gif.py"),
           str(QPOS_LOG), str(gif), "walk", str(TASKS_YAML)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    with Image.open(gif) as im:
        assert im.n_frames > 1


# ── learning/ stack: torch on the GPU, export to ONNX, run in ONNX Runtime ──

class TinyDenoiser(torch.nn.Module):
    """Same layer types as the planned DenoisingCNN: conv over time, BatchNorm,
    SiLU (swish), and an observation-conditioned bias."""

    def __init__(self, act=24, obs=63, feat=32):
        super().__init__()
        self.conv = torch.nn.Conv1d(act, feat, 3, padding=1)
        self.bn   = torch.nn.BatchNorm1d(feat)
        self.cond = torch.nn.Linear(obs + 1, feat)
        self.out  = torch.nn.Conv1d(feat, act, 3, padding=1)

    def forward(self, u, y, t):                 # u [B,H,A], y [B,O], t [B,1]
        z = self.bn(self.conv(u.transpose(1, 2)))
        z = torch.nn.functional.silu(z + self.cond(torch.cat([y, t], -1))[..., None])
        return self.out(z).transpose(1, 2)


def test_torch_cuda_trains_on_gpu():
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (12, 0)   # RTX 5080, sm_120
    net = TinyDenoiser().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    u, y, t = torch.randn(16, 40, 24, device="cuda"), torch.randn(16, 63, device="cuda"), torch.rand(16, 1, device="cuda")
    loss = net(u, y, t).square().mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_onnx_export_matches_torch(tmp_path):
    net = TinyDenoiser().eval()
    path = tmp_path / "tiny.onnx"
    example = (torch.randn(2, 40, 24), torch.randn(2, 63), torch.rand(2, 1))
    torch.onnx.export(net, example, str(path), opset_version=17, dynamo=False,
                      input_names=["u", "y", "t"], output_names=["v"],
                      dynamic_axes={n: {0: "batch"} for n in ["u", "y", "t", "v"]})
    onnx.checker.check_model(onnx.load(str(path)))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 7):
        u, y, t = torch.randn(batch, 40, 24), torch.randn(batch, 63), torch.rand(batch, 1)
        (v,) = session.run(None, {"u": u.numpy(), "y": y.numpy(), "t": t.numpy()})
        with torch.no_grad():
            np.testing.assert_allclose(v, net(u, y, t).numpy(), atol=1e-5)
