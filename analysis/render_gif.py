"""
Render a GIF from a qpos trajectory saved by mppi_sim.

mppi_sim also writes a sibling <name>.csv (default mppi_sim.csv), one row per
qpos row. This script reads only its push columns (Pusher in
controllers/common/harness.h), if present, and draws each push as a red arrow:
  horizontal_kick  arrow pointing at the trunk along the kick velocity,
                   kept on screen KICK_SHOW_S after the kick
  down_force       arrow pressing down on the push point while the force is held

Usage (from anywhere; the sims call it automatically after each run):
    python3 analysis/render_gif.py [qpos_csv] [output.gif] [task_name] [tasks_yaml]

Defaults:
    qpos_csv   = analysis/data/mppi_sim/mppi_sim_qpos.csv (mppi_sim's working log)
    output     = analysis/data/mppi_sim/mppi_sim.gif
    task_name  = none -> renders against the flat go2/scene.xml (old behavior)
    tasks_yaml = ../controllers/muscle/utils/tasks.yaml (relative to this script)

task_name must match whichever task the qpos_csv was actually generated from
(the same name passed to mppi_sim) — its model_path is looked up in
tasks_yaml so the render uses the scene the trajectory was actually simulated
on. Without it, this always renders against the flat scene regardless of the
data's actual source scene, which silently produces a wrong-looking render
for any task using a non-default model_path (e.g. walk_rough).
"""

import sys
import os
import numpy as np
import mujoco
import yaml
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
_DIR                = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH  = os.path.join(_DIR, "../controllers/common/go2/scene.xml")
DEFAULT_TASKS_YAML  = os.path.join(_DIR, "../controllers/muscle/utils/tasks.yaml")
# model_path values inside tasks.yaml are relative to the repo root (the same
# rule mppi_sim applies, see controllers/common/task_config.h) -- resolve against that,
# not this script's dir.
TASKS_YAML_BASE_DIR = os.path.join(_DIR, "..")

DEFAULT_OUT_DIR     = os.path.join(_DIR, "data", "mppi_sim")
qpos_path  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DEFAULT_OUT_DIR, "mppi_sim_qpos.csv")
gif_path   = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DEFAULT_OUT_DIR, "mppi_sim.gif")
task_name  = sys.argv[3] if len(sys.argv) > 3 else None
tasks_yaml = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_TASKS_YAML

if task_name:
    with open(tasks_yaml) as f:
        tasks = yaml.safe_load(f)
    if task_name not in tasks:
        raise ValueError(f"Task '{task_name}' not found in {tasks_yaml}")
    MODEL_PATH = os.path.normpath(os.path.join(TASKS_YAML_BASE_DIR, tasks[task_name]["model_path"]))
    SIM_HZ = 1.0 / tasks[task_name]["dt"]
else:
    MODEL_PATH = DEFAULT_MODEL_PATH
    SIM_HZ = 50.0  # unknown task's actual dt; keep the old assumed rate

# ── load ───────────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

print(f"Loading qpos: {qpos_path}")
qpos_log = np.loadtxt(qpos_path, delimiter=",")
print(f"  {len(qpos_log)} frames  |  nq={model.nq}")

# ── push arrows ────────────────────────────────────────────────────────────────
KICK_SHOW_S     = 0.5                  # s a kick's arrow stays on screen
KICK_M_PER_MPS  = 0.5                  # kick arrow length per m/s of kick velocity
FORCE_M_PER_N   = 0.003                # down-force arrow length per N (150 N -> 0.45 m)
ARROW_GAP       = 0.25                 # m between the trunk centre and a kick arrow's head
TRUNK_TOP       = 0.06                 # m above the trunk centre where a down-force arrow's head sits
ARROW_WIDTH     = 0.03                 # m
ARROW_RGBA      = np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32)

# Base body, resolved as find_base_body() in controllers/common/control_utils.h.
BASE_BID = next((b for n in ("trunk", "base", "base_link")
                 if (b := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)) >= 0), 1)


def load_pushes(csv_path, n_rows):
    """Per-row push arrows from the run CSV: (kick, down) arrays of shape (n_rows, 2|3), or None."""
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    if "push_vx" not in header and "push_fz" not in header:
        return None, None

    cols = [c for c in ("push_vx", "push_vy", "push_fz", "push_dx", "push_dy") if c in header]
    vals = np.loadtxt(csv_path, delimiter=",", skiprows=1, usecols=[header.index(c) for c in cols], ndmin=2)
    if len(vals) != n_rows:
        print(f"  {csv_path}: {len(vals)} rows, qpos has {n_rows}; skipping push arrows")
        return None, None

    if "push_fz" in cols:
        return None, vals   # fz, dx, dy: non-zero only while the force is held

    # A kick sets push_vx/vy for its own control step only; hold it KICK_SHOW_S.
    kick      = np.zeros_like(vals)
    show_rows = max(1, int(round(KICK_SHOW_S * SIM_HZ)))
    for i in np.flatnonzero(np.any(vals != 0.0, axis=1)):
        kick[i:i + show_rows] = vals[i]
    return kick, None


def add_arrow(scene, tail, head):
    """Arrow from tail to head. MuJoCo's renderer (render_gl3.c, mjGEOM_ARROW) draws
    an arrow geom over only the first half of its connector, so the connector ends
    at tail + 2 (head - tail)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), ARROW_RGBA)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, ARROW_WIDTH, tail, tail + 2.0 * (head - tail))
    scene.ngeom += 1


def draw_push(scene, kick_row, down_row):
    trunk = data.xipos[BASE_BID]

    if kick_row is not None and np.any(kick_row != 0.0):
        v   = np.array([kick_row[0], kick_row[1], 0.0])
        u   = v / np.linalg.norm(v)
        head = trunk - ARROW_GAP * u
        add_arrow(scene, head - KICK_M_PER_MPS * np.linalg.norm(v) * u, head)

    if down_row is not None and down_row[0] != 0.0:
        fz, dx, dy = down_row
        com  = data.subtree_com[BASE_BID]   # push point = CoM + (dx, dy), as Pusher::press() picks it
        head = np.array([com[0] + dx, com[1] + dy, trunk[2] + TRUNK_TOP])
        add_arrow(scene, head + np.array([0.0, 0.0, FORCE_M_PER_N * abs(fz)]), head)


stem = qpos_path[:-len("_qpos.csv")] if qpos_path.endswith("_qpos.csv") else None   # log.h qpos_path_for()
kick_log, down_log = load_pushes(stem + ".csv", len(qpos_log)) if stem else (None, None)

# ── render ─────────────────────────────────────────────────────────────────────
HEIGHT, WIDTH = 480, 640
FPS           = 25
SKIP          = max(1, int(round(SIM_HZ / FPS)))  # downsample the task's actual sim rate to 25 fps

cam           = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 2.5
cam.elevation = -15.0
cam.azimuth   = 90.0

frames = []
with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
    for i in range(0, len(qpos_log), SKIP):
        qpos = qpos_log[i]
        data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(model, data)
        cam.lookat[0] = data.qpos[0]
        cam.lookat[1] = data.qpos[1]
        cam.lookat[2] = 0.3
        renderer.update_scene(data, camera=cam)
        draw_push(renderer.scene,
                  kick_log[i] if kick_log is not None else None,
                  down_log[i] if down_log is not None else None)
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
