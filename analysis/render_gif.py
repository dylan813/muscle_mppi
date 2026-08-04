"""
Render a GIF from a qpos trajectory saved by mppi_sim.

mppi_sim also writes a sibling <name>.csv (default mppi_sim.csv) with per-joint
velocity and per-muscle activation columns for FR-leg plotting via
analysis/log/plot_walk_leg.py — this script only reads the qpos companion and
ignores that file.

Usage (from muscle_mppi/muscle_mppi/):
    python3 ../../analysis/render_gif.py [qpos_csv] [output.gif] [task_name] [tasks_yaml]

Defaults:
    qpos_csv   = mppi_sim_qpos.csv
    output     = mppi_sim.gif
    task_name  = none -> renders against the flat go2/scene.xml (old behavior)
    tasks_yaml = ../muscle_mppi/utils/tasks.yaml (relative to this script)

task_name must match whichever task the qpos_csv was actually generated from
(the same name passed to mppi_sim) — its model_path is looked up in
tasks_yaml so the render uses the scene the trajectory was actually simulated
on. Without it, this always renders against the flat scene regardless of the
data's actual source scene, which silently produces a wrong-looking render
for any task using a non-default model_path (e.g. walk_straight_rough).
"""

import sys
import os
import numpy as np
import mujoco
import yaml
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
_DIR                = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH  = os.path.join(_DIR, "../unitree_mujoco/unitree_robots/go2/scene.xml")
DEFAULT_TASKS_YAML  = os.path.join(_DIR, "../muscle_mppi/utils/tasks.yaml")
# model_path values inside tasks.yaml are relative to muscle_mppi/muscle_mppi/build/
# (mppi_sim's own working directory) -- resolve against that, not this script's dir.
TASKS_YAML_BASE_DIR = os.path.join(_DIR, "../muscle_mppi/build")

qpos_path  = sys.argv[1] if len(sys.argv) > 1 else "mppi_sim_qpos.csv"
gif_path   = sys.argv[2] if len(sys.argv) > 2 else "mppi_sim.gif"
task_name  = sys.argv[3] if len(sys.argv) > 3 else None
tasks_yaml = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_TASKS_YAML

if task_name:
    with open(tasks_yaml) as f:
        tasks = yaml.safe_load(f)
    if task_name not in tasks:
        raise ValueError(f"Task '{task_name}' not found in {tasks_yaml}")
    MODEL_PATH = os.path.normpath(os.path.join(TASKS_YAML_BASE_DIR, tasks[task_name]["model_path"]))
else:
    MODEL_PATH = DEFAULT_MODEL_PATH

# ── load ───────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

print(f"Loading qpos: {qpos_path}")
qpos_log = np.loadtxt(qpos_path, delimiter=",")
print(f"  {len(qpos_log)} frames  |  nq={model.nq}")

# ── render ─────────────────────────────────────────────────────────────────────
HEIGHT, WIDTH = 480, 640
FPS           = 25
SKIP          = max(1, int(round(50 / FPS)))  # sim runs at 50 Hz → downsample to 25 fps

cam           = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 2.5
cam.elevation = -15.0
cam.azimuth   = 90.0

frames = []
with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    for qpos in qpos_log[::SKIP]:
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = 0.3
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

# ── save ───────────────────────────────────────────────────────────────────────
print(f"Saving GIF ({len(frames)} frames @ {FPS} fps): {gif_path}")
duration_ms = int(1000 / FPS)
pil_frames = [Image.fromarray(f) for f in frames]
pil_frames[0].save(
    gif_path,
    save_all=True,
    append_images=pil_frames[1:],
    duration=duration_ms,
    loop=0,
)
print("Done.")
