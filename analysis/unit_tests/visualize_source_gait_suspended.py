"""
Kinematic scrub of the raw joint-space source gait (position + velocity, the
input to generate_activation_gaits.py) rendered on the suspended Go2 model
(scene_suspended.xml) — same model, camera, and mj_forward-teleport scrub as
visualize_activation_gait_suspended.py, but with no Hill model / activation
overlay involved. Meant to be played side by side with that script's output
for a direct before/after comparison: this is exactly what get_bias_torques()
saw when generate_activation_gaits.py computed tau_req at each phase.

Usage (run from anywhere; paths are relative to this file):
    python3 visualize_source_gait_suspended.py [gait_key] [output.gif]

    gait_key = "{TIER}_{vel}_{height}", e.g. FAST_0_1_10cm (default)
               Must have a matching source gait TSV under RTWholeBodyMPPI's
               gait_scheduler/gaits/.

Defaults:
    gait_key = FAST_0_1_10cm
    output   = ../videos/source_gait_suspended_<gait_key>.gif
"""

import sys
import os
import numpy as np
import mujoco
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "../../unitree_mujoco/unitree_robots/go2/scene_suspended.xml")
SRC_GAIT_DIR = os.path.join(_DIR, "../../../RTWholeBodyMPPI/legged_mppi/"
                                   "whole_body_mppi/control/gait_scheduler/gaits")

gait_key = sys.argv[1] if len(sys.argv) > 1 else "FAST_0_1_10cm"
gif_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    _DIR, "../videos", f"source_gait_suspended_{gait_key}.gif")

parts  = gait_key.split("_")
tier   = parts[0]
height = parts[-1]
vel    = "_".join(parts[1:-1])

src_gait_path = os.path.join(SRC_GAIT_DIR, tier,
                              f"walking_gait_raibert_{tier}_{vel}_{height}_100hz.tsv")

if not os.path.exists(src_gait_path):
    raise FileNotFoundError(
        f"{src_gait_path} not found — check gait_key '{gait_key}' matches an "
        f"existing tier/vel/height combo.")

NUM_JOINTS = 12
GAIT_HZ    = 100.0   # source gait file phase rate
N_CYCLES   = 3
FPS        = 25

print(f"Loading model: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

print(f"Loading source gait: {src_gait_path}")
src_gait = np.loadtxt(src_gait_path, delimiter="\t")
q_traj  = src_gait[:12, :]
dq_traj = src_gait[12:24, :]
n_phases = q_traj.shape[1]
print(f"  {n_phases} phases")

# joint order matches the gait file's row order: FR/FL/RR/RL x hip/thigh/calf
qa = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(NUM_JOINTS)]
qv = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(NUM_JOINTS)]

HEIGHT, WIDTH = 480, 640
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
cam = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 1.6
cam.elevation = -15.0
cam.azimuth   = 90.0
cam.lookat[:] = [0, 0, 0.6]   # base is fixed at z=0.8; frame the legs below it

frames = []
render_every = max(1, round(GAIT_HZ / FPS))

mujoco.mj_resetData(model, data)
tick = 0
for _cycle in range(N_CYCLES):
    for phase in range(n_phases):
        for j in range(NUM_JOINTS):
            data.qpos[qa[j]] = q_traj[j, phase]
            data.qvel[qv[j]] = dq_traj[j, phase]
        mujoco.mj_forward(model, data)

        if tick % render_every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        tick += 1

print(f"Saving GIF ({len(frames)} frames @ {FPS} fps): {gif_path}")
os.makedirs(os.path.dirname(gif_path), exist_ok=True)
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
