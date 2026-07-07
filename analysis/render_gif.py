"""
Render a GIF from a qpos trajectory saved by mppi_sim.

Usage (from muscle_mppi/muscle_mppi/):
    python3 ../../analysis/render_gif.py [qpos_csv] [output.gif]

Defaults:
    qpos_csv  = mppi_sim_qpos.csv
    output    = mppi_sim.gif
"""

import sys
import os
import numpy as np
import mujoco
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "../unitree_mujoco/unitree_robots/go2/scene.xml")

qpos_path = sys.argv[1] if len(sys.argv) > 1 else "mppi_sim_qpos.csv"
gif_path  = sys.argv[2] if len(sys.argv) > 2 else "mppi_sim.gif"

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
