"""
High-frequency PD playback of the source joint-space gait (position AND
velocity) on the grounded, free-floating Go2 model (scene.xml) — real contact
dynamics via mj_step, unlike visualize_source_gait_suspended.py's fixed-base
kinematic scrub. This is step 1 toward deriving activation gaits from forces
that actually include ground reaction: first confirm the PD-tracked gait looks
right on the ground before logging the joint torques it takes to hold it.

PD target tracks both q_des and dq_des from the source gait (not just q_des
with an implicit zero-velocity target), since the downstream force-derivation
step needs the robot's actual (q, dq) at each phase to match the pair the
activation gait was solved against as closely as possible:

    ctrl[j] = KP * (q_des[j] - q[j]) + KD * (dq_des[j] - dq[j])

500 Hz control loop (DT=0.002), gait phase advances at the source file's
100 Hz rate — same convention as visualize_raibert_gait.py / test_gait.cpp.

Usage (run from anywhere; paths are relative to this file):
    python3 visualize_source_gait_grounded.py [gait_key] [output.gif]

    gait_key = "{TIER}_{vel}_{height}", e.g. FAST_0_1_10cm (default)

Defaults:
    gait_key = FAST_0_1_10cm
    output   = ../videos/source_gait_grounded_<gait_key>.gif
"""

import sys
import os
import numpy as np
import mujoco
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "../../unitree_mujoco/unitree_robots/go2/scene.xml")
SRC_GAIT_DIR = os.path.join(_DIR, "../../../RTWholeBodyMPPI/legged_mppi/"
                                   "whole_body_mppi/control/gait_scheduler/gaits")

gait_key = sys.argv[1] if len(sys.argv) > 1 else "FAST_0_1_10cm"
gif_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    _DIR, "../videos", f"source_gait_grounded_{gait_key}.gif")

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
DT              = 0.002    # physics timestep (500 Hz)
GAIT_HZ         = 100.0    # gait file phase rate
TICKS_PER_PHASE = round(1.0 / (GAIT_HZ * DT))
KP, KD          = 50.0, 3.5
N_CYCLES        = 4
FPS             = 25

print(f"Loading model: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
model.opt.timestep = DT
data = mujoco.MjData(model)

print(f"Loading source gait: {src_gait_path}")
gait = np.loadtxt(src_gait_path, delimiter="\t")
q_traj  = gait[:12, :]
dq_traj = gait[12:24, :]
n_phases = q_traj.shape[1]
print(f"  {n_phases} phases")

# joint order matches the gait file's row order: FR/FL/RR/RL x hip/thigh/calf
qa = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(NUM_JOINTS)]
qv = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(NUM_JOINTS)]

# place the robot on the ground in the gait's own phase-0 leg configuration
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.5
data.qpos[3] = 1.0
for j in range(NUM_JOINTS):
    data.qpos[qa[j]] = q_traj[j, 0]
mujoco.mj_forward(model, data)
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_names]
min_z = min(data.xpos[b, 2] for b in foot_ids if b >= 0)
data.qpos[2] -= min_z
mujoco.mj_forward(model, data)

HEIGHT, WIDTH = 480, 640
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
cam = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 2.5
cam.elevation = -15.0
cam.azimuth   = 90.0

frames = []
total_cycle_ticks = n_phases * TICKS_PER_PHASE
total_ticks = N_CYCLES * total_cycle_ticks
render_every = max(1, round(1.0 / (FPS * DT)))

phase = 0
for tick in range(total_ticks):
    q_des  = q_traj[:, phase]
    dq_des = dq_traj[:, phase]
    for j in range(NUM_JOINTS):
        q  = data.qpos[qa[j]]
        dq = data.qvel[qv[j]]
        data.ctrl[j] = KP * (q_des[j] - q) + KD * (dq_des[j] - dq)
    mujoco.mj_step(model, data)

    if tick % TICKS_PER_PHASE == TICKS_PER_PHASE - 1:
        phase = (phase + 1) % n_phases

    if tick % render_every == 0:
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = 0.3
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

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
