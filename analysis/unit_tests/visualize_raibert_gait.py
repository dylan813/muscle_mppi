"""
Dynamic playback of a raw RTWholeBodyMPPI joint-space gait reference
(walking_gait_raibert_*.tsv), rendered on the Go2 model for a direct visual
comparison against test_gait_pos's Hill-model activation gait output
(generated from this same source file by generate_activation_gaits.py).

No Hill model — instead the gait file's joint-position rows (0-11) drive a
plain PD torque controller (kp/kd, same structure as test_gait_pos.cpp's
stand-up PD), and mj_step actually integrates physics: gravity acts on the
free-floating base, and ground contact determines whether the robot stays
upright. This is different from a pure kinematic scrub (mj_forward only),
which ignores gravity/contact and just teleports the pose.

Usage (run from anywhere; paths are relative to this file):
    python3 visualize_raibert_gait.py [gait.tsv] [output.gif]

Defaults:
    gait.tsv = RTWholeBodyMPPI's 'walk' gait (GAIT_WALK_PATH in mppi_locomotion.py)
    output   = ../videos/raibert_gait_reference.gif
"""

import sys
import os
import numpy as np
import mujoco
import yaml
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "../../unitree_mujoco/unitree_robots/go2/scene.xml")
YAML_PATH = os.path.join(_DIR, "../../muscle_mppi/utils/tasks.yaml")
DEFAULT_GAIT = os.path.join(
    _DIR, "../../../RTWholeBodyMPPI/legged_mppi/whole_body_mppi/control/"
          "gait_scheduler/gaits/MED/walking_gait_raibert_MED_0_1_10cm_100hz.tsv")

gait_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GAIT
gif_path  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_DIR, "../videos/raibert_gait_reference.gif")

DT = 0.002            # physics timestep (500 Hz), matches test_gait_pos.cpp
GAIT_HZ = 100.0        # gait file phase rate
TICKS_PER_PHASE = round(1.0 / (GAIT_HZ * DT))
KP, KD = 50.0, 3.5     # position-tracking gains, same as test_gait_pos.cpp's stand-up PD
N_CYCLES = 3
FPS = 25

print(f"Loading model: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
model.opt.timestep = DT
data = mujoco.MjData(model)

print(f"Loading gait:  {gait_path}")
gait = np.loadtxt(gait_path, delimiter="\t")
joint_pos_ref = gait[:12]   # rows 12-23 are joint velocities, unused here
n_phases = joint_pos_ref.shape[1]
print(f"  {n_phases} phases")

# joint order matches the gait file's row order: FR/FL/RR/RL x hip/thigh/calf
qa = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(12)]
qv = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(12)]

# joint damping, matching muscle_mppi's tasks.yaml convention
with open(YAML_PATH) as f:
    kd_sim = yaml.safe_load(f)["default_muscle_quad"]["kd_sim"]
for j in range(12):
    model.dof_damping[qv[j]] = kd_sim[j]

# place the robot on the ground in the gait's own phase-0 leg configuration
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.5
data.qpos[3] = 1.0
for j in range(12):
    data.qpos[qa[j]] = joint_pos_ref[j, 0]
mujoco.mj_forward(model, data)
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in foot_names]
min_z = min(data.xpos[b, 2] for b in foot_ids if b >= 0)
data.qpos[2] -= min_z
mujoco.mj_forward(model, data)

HEIGHT, WIDTH = 480, 640
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 2.5
cam.elevation = -15.0
cam.azimuth = 90.0
cam.lookat[:] = [0, 0, 0.3]

frames = []
total_cycle_ticks = n_phases * TICKS_PER_PHASE
total_ticks = N_CYCLES * total_cycle_ticks
render_every = max(1, round(1.0 / (FPS * DT)))

phase = 0
for tick in range(total_ticks):
    q_des = joint_pos_ref[:, phase]
    for j in range(12):
        q = data.qpos[qa[j]]
        dq = data.qvel[qv[j]]
        data.ctrl[j] = KP * (q_des[j] - q) + KD * (-dq)
    mujoco.mj_step(model, data)

    if tick % TICKS_PER_PHASE == TICKS_PER_PHASE - 1:
        phase = (phase + 1) % n_phases

    if tick % render_every == 0:
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

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
