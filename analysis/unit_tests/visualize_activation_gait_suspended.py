"""
Kinematic scrub of a generated activation gait, rendered on the suspended Go2
model (scene_suspended.xml) — i.e. the exact same setup generate_activation_gaits.py
used to compute tau_req (base rigidly fixed at z=0.8, gravity + Coriolis/
centrifugal only, zero contact). Replays q_traj/dq_traj via direct qpos/qvel
assignment + mj_forward (teleport, no mj_step — there is no dynamics to
integrate since the base isn't free and nothing contacts the floor), same as
get_bias_torques() in generate_activation_gaits.py.

Each leg link (hip/thigh/calf) is additionally tinted by that joint's
co-contraction level (a1+a2)/2, taken from the corresponding row pair of the
already-generated activation_gait_*.tsv — blue (slack) to red (near-max
co-contraction) — so the render doubles as a visual check of both the
suspended kinematics and the activation output for that gait.

Usage (run from anywhere; paths are relative to this file):
    python3 visualize_activation_gait_suspended.py [gait_key] [output.gif]

    gait_key = "{TIER}_{vel}_{height}", e.g. FAST_0_1_10cm (default)
               Must have a matching source gait TSV under RTWholeBodyMPPI's
               gait_scheduler/gaits/ and a generated activation_gait_*.tsv
               under muscle_mppi/gaits/ (run generate_activation_gaits.py first).

Defaults:
    gait_key = FAST_0_1_10cm
    output   = ../videos/activation_gait_suspended_<gait_key>.gif
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
ACT_GAIT_DIR = os.path.join(_DIR, "../../muscle_mppi/gaits")

gait_key = sys.argv[1] if len(sys.argv) > 1 else "FAST_0_1_10cm"
gif_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    _DIR, "../videos", f"activation_gait_suspended_{gait_key}.gif")

parts  = gait_key.split("_")
tier   = parts[0]
height = parts[-1]
vel    = "_".join(parts[1:-1])

src_gait_path = os.path.join(SRC_GAIT_DIR, tier,
                              f"walking_gait_raibert_{tier}_{vel}_{height}_100hz.tsv")
act_gait_path = os.path.join(ACT_GAIT_DIR, tier, f"activation_gait_{gait_key}.tsv")

for p in (src_gait_path, act_gait_path):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} not found — run generate_activation_gaits.py first "
            f"(or check gait_key '{gait_key}' matches an existing tier/vel/height combo).")

NUM_JOINTS  = 12
JOINT_NAMES = ["FR_hip", "FR_thigh", "FR_calf",
               "FL_hip", "FL_thigh", "FL_calf",
               "RR_hip", "RR_thigh", "RR_calf",
               "RL_hip", "RL_thigh", "RL_calf"]

GAIT_HZ  = 100.0   # source gait file phase rate
N_CYCLES = 3
FPS      = 25

print(f"Loading model: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

print(f"Loading source gait: {src_gait_path}")
src_gait = np.loadtxt(src_gait_path, delimiter="\t")
q_traj  = src_gait[:12, :]
dq_traj = src_gait[12:24, :]
n_phases = q_traj.shape[1]
print(f"  {n_phases} phases")

print(f"Loading activation gait: {act_gait_path}")
act_gait = np.loadtxt(act_gait_path, delimiter="\t")
a1 = act_gait[:12, :]
a2 = act_gait[12:24, :]
assert a1.shape[1] == n_phases, "activation gait and source gait phase counts differ"
cocontraction = np.clip((a1 + a2) / 2.0, 0.0, 1.0)   # (12, N)

# joint order matches JOINT_NAMES / actuator order in the model
qa = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(NUM_JOINTS)]
qv = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(NUM_JOINTS)]

# geoms owned by each joint's link body — recolour all of them (visual mesh +
# collision primitive) to show that joint's co-contraction level
body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in JOINT_NAMES]
link_geom_ids = [
    list(range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]))
    for b in body_ids
]

COLD = np.array([0.29, 0.56, 0.89])   # slack — steel blue
HOT  = np.array([0.85, 0.10, 0.10])   # near-max co-contraction — red

def apply_cocontraction_color(t):
    for j in range(NUM_JOINTS):
        c = cocontraction[j, t]
        rgb = (1.0 - c) * COLD + c * HOT
        for gid in link_geom_ids[j]:
            model.geom_rgba[gid, :3] = rgb
            model.geom_rgba[gid, 3]  = 1.0

HEIGHT, WIDTH = 480, 640
renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
cam = mujoco.MjvCamera()
cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance = 1.6
cam.elevation = -15.0
cam.azimuth  = 90.0
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
        apply_cocontraction_color(phase)

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
