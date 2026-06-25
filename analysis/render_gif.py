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
_DIR      = os.path.dirname(os.path.abspath(__file__))
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
FPS           = 25         # GIF playback speed
SKIP          = max(1, int(round(50 / FPS)))  # sim runs at 50Hz → downsample

renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

# camera: side view slightly behind and above
cam           = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 2.5
cam.elevation = -15.0
cam.azimuth   = 90.0

frames = []
for i, qpos in enumerate(qpos_log[::SKIP]):
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    frame = renderer.render()
    frames.append(frame)

# follow robot in x — re-render with tracking camera
renderer2 = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
frames_tracked = []
for i, qpos in enumerate(qpos_log[::SKIP]):
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    cam.lookat[0] = data.qpos[0]   # track x position
    cam.lookat[1] = data.qpos[1]
    cam.lookat[2] = 0.3
    renderer2.update_scene(data, camera=cam)
    frame = renderer2.render()
    frames_tracked.append(frame)

# save
print(f"Saving GIF ({len(frames_tracked)} frames @ {FPS} fps): {gif_path}")
duration_ms = int(1000 / FPS)
pil_frames = [Image.fromarray(f) for f in frames_tracked]
pil_frames[0].save(
    gif_path,
    save_all=True,
    append_images=pil_frames[1:],
    duration=duration_ms,
    loop=0
)
print("Done.")
