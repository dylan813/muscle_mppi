"""
Objective function for CMA-ES optimization of Hill muscle parameters.

Each call to evaluate(x):
  1. Computes the closed-form active/passive force-length curve area for
     each joint type from x (no simulation involved) and combines it with
     the locomotion cost below via AREA_WEIGHT (env var, default 0 = off).
  2. Expands 12D x=[lce_min×3, lce_max×3, pFLmax×3, FVmax×3] to 12-element
     arrays (hip/thigh/calf values shared across all 4 legs).
  3. Writes a temp tasks.yaml with absolute paths
  4. Regenerates the single gait file used by the walk task
  5. Runs mppi_sim as a subprocess
  6. Parses the output CSV and computes the mean per-step cost from tasks.yaml weights

The cost formula mirrors mppi_locomotion.cpp::step_cost() for terms computable
from the logged CSV (pos_x/y/z, orientation, vel_x/y/z — px/py/pz are the
whole-robot subtree CoM and vx/vy/vz are body-frame velocity, both computed
in mppi_sim.cpp to match step_cost() exactly). ang_vel and gait_ref
terms are omitted as those quantities are not logged.
"""

import os

# Headless (no X11 DISPLAY) EGL rendering — must be set before mujoco is
# imported, since the GL backend is selected at import time. Only defaults
# it; an existing MUJOCO_GL (e.g. a real DISPLAY, or osmesa) is left alone.
os.environ.setdefault("MUJOCO_GL", "egl")

# mppi_sim parallelizes its rollout sampling internally via OpenMP
# (#pragma omp parallel for in base_mppi.cpp / mppi_locomotion.cpp), which by
# default spawns one thread per core. Each CMA-ES generation already runs
# many mppi_sim processes concurrently (one per ProcessPoolExecutor worker),
# so without this, every worker process independently tries to claim all
# cores and they oversubscribe each other. subprocess.run() below inherits
# this env, so setting it here covers every entry point (sweep script,
# direct cmaes_walk.py invocation, etc). Only defaults it; an explicit
# OMP_NUM_THREADS in the environment is left alone.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import copy
import subprocess
import tempfile
import mujoco
import numpy as np
import pandas as pd
import yaml

from gait_generator import generate_gait

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(_HERE, "..", ".."))
MPPI_SIM   = os.path.join(REPO_ROOT, "muscle_mppi", "build", "mppi_sim")
MODEL_PATH = os.path.join(REPO_ROOT, "unitree_mujoco", "unitree_robots", "go2", "scene.xml")
BASE_YAML  = os.path.join(REPO_ROOT, "muscle_mppi", "utils", "tasks.yaml")

SIM_TIMEOUT = 120  # seconds before killing a runaway mppi_sim

# Per infeasible phase penalty — must dominate any feasible fitness value so
# CMA-ES always prefers feasible candidates. mppi_sim is skipped entirely.
INFEASIBLE_PENALTY_PER_PHASE = 5e4

# Number of phases in the gait trajectory (rows of the source gait TSV).
# Candidates that never produce a scorable rollout at all (gait generation
# error, sim timeout, unreadable/empty CSV) are scored as maximally
# infeasible — every phase failed — rather than as a separate cost bucket.
N_PHASES = 100
MAX_INFEASIBLE_PENALTY = INFEASIBLE_PENALTY_PER_PHASE * N_PHASES

# Discard this many seconds of transient at the start of the log
TRANSIENT_SECS = 2.0

# ── rollout rendering (for gif logging) ──────────────────────────────────────
RENDER_HEIGHT, RENDER_WIDTH = 480, 640
RENDER_FPS = 25


# Joint order: FR_hip(0), FR_thigh(1), FR_calf(2), FL_hip(3), ...
# Type index repeats: [0,1,2, 0,1,2, 0,1,2, 0,1,2]
_JOINT_TYPE = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

# ── closed-form active/passive force-length curve area (no simulation) ──────
# Reimplements muscle.h's active/passive FL curves exactly as in
# unit_tests/plot_force_length.py, so the area between them can be scored
# for every candidate before (and regardless of) running mppi_sim.
#
# Integration domain is fixed and wide enough to contain both curves'
# crossing points for any lce_min/lce_max/pFLmax within cmaes_walk.py's
# LO/HI bounds — active_fl is 0 outside [lce_min, lce_max], so the domain
# only needs to bracket that interval plus passive's rise past lce=1.0.
_AREA_DOMAIN = np.linspace(0.3, 1.9, 400)


def _active_force_length(length, A, mid, B):
    left  = 0.5 * (A + mid)
    right = 0.5 * (mid + B)
    if length <= A or length >= B:
        return 0.0
    if length < left:
        temp = (length - A) / (left - A)
        return 0.5 * temp * temp
    elif length < mid:
        temp = (mid - length) / (mid - left)
        return 1.0 - 0.5 * temp * temp
    elif length < right:
        temp = (length - mid) / (right - mid)
        return 1.0 - 0.5 * temp * temp
    else:
        temp = (B - length) / (B - right)
        return 0.5 * temp * temp


def _active_fl(lce, lmin, lmax):
    return (_active_force_length(lce, lmin, 1.0, lmax)
            + 0.15 * _active_force_length(lce, lmin, 0.5 * (lmin + 0.95), 0.95))


def _passive_force_length(lce, fpmax, b):
    if lce <= 1.0:
        return 0.0
    elif lce <= b:
        temp = (lce - 1.0) / (b - 1.0)
        return 0.25 * fpmax * temp ** 3
    else:
        temp = (lce - b) / (b - 1.0)
        return 0.25 * fpmax * (1.0 + 3.0 * temp)


def _joint_curve_area(lmin, lmax, fpmax):
    """Area of the region where the active curve exceeds the passive curve,
    i.e. ∫ max(active - passive, 0) dl over _AREA_DOMAIN. The max(...,0)
    clamp means passive's unbounded growth past its crossing with active
    never contributes negatively — larger pFLmax can only shrink this area,
    never make it negative."""
    b = 0.5 * (lmax + 1.0)
    active  = np.array([_active_fl(l, lmin, lmax) for l in _AREA_DOMAIN])
    passive = np.array([_passive_force_length(l, fpmax, b) for l in _AREA_DOMAIN])
    gap = np.maximum(active - passive, 0.0)
    return float(np.trapz(gap, _AREA_DOMAIN))


def curve_area_total(x):
    """Sum of the active/passive FL-curve area across hip/thigh/calf, given
    a 12D (or 9D) candidate x whose first 9 entries are
    [lce_min_hip/thigh/calf, lce_max_hip/thigh/calf, pFLmax_hip/thigh/calf]."""
    lce_min, lce_max, pFLmax = x[0:3], x[3:6], x[6:9]
    return sum(_joint_curve_area(lce_min[j], lce_max[j], pFLmax[j]) for j in range(3))


def _broadcast_per_type(vals_by_type):
    """[hip_val, thigh_val, calf_val] → 12-element list following joint order."""
    return [float(vals_by_type[t]) for t in _JOINT_TYPE]


def _build_muscle_params(x, base_quad):
    """
    x = [lce_min_hip, lce_min_thigh, lce_min_calf,
         lce_max_hip, lce_max_thigh, lce_max_calf,
         pFLmax_hip,  pFLmax_thigh,  pFLmax_calf,
         FVmax_hip,   FVmax_thigh,   FVmax_calf]
    All 4 legs share the same value per joint type.
    Fixed params (vmax, phi_min, phi_max, peak_force) taken from base_quad.
    """
    p = copy.deepcopy(base_quad)
    p["lce_min"] = _broadcast_per_type(x[0:3])
    p["lce_max"] = _broadcast_per_type(x[3:6])
    p["pFLmax"]  = _broadcast_per_type(x[6:9])
    p["FVmax"]   = _broadcast_per_type(x[9:12])
    return p


def _write_temp_yaml(base_cfg, muscle_params, gait_path_abs, tmp_dir):
    """
    Write a modified tasks.yaml to tmp_dir, with:
    - walk.muscle fields updated to muscle_params
    - walk.model_path  set to absolute MODEL_PATH
    - walk.gait_path   set to gait_path_abs
    Returns the path to the written yaml.
    """
    cfg = copy.deepcopy(base_cfg)

    # Update muscle params in both the anchor block and the resolved walk.muscle dict
    for key in ("lce_min", "lce_max", "FVmax", "pFLmax"):
        cfg["default_muscle_quad"][key] = muscle_params[key]
        cfg["walk"]["muscle"][key]      = muscle_params[key]

    # Absolute paths so mppi_sim works from any CWD
    cfg["walk"]["model_path"] = MODEL_PATH
    cfg["walk"]["gait_path"]  = gait_path_abs

    yaml_path = os.path.join(tmp_dir, "tasks.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None)
    return yaml_path


def _compute_fitness(csv_path, cost_weights, goal_pos, cmd_vel):
    """
    Parse mppi_sim CSV and compute mean per-step cost using the same weights
    as mppi_locomotion.cpp::step_cost() (for terms available in the CSV).
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return MAX_INFEASIBLE_PENALTY

    if df.empty:
        return MAX_INFEASIBLE_PENALTY

    # Discard transient
    df = df[df["t"] > TRANSIENT_SECS]
    if df.empty:
        return MAX_INFEASIBLE_PENALTY

    w   = cost_weights
    gx, gy, gz = goal_pos
    cvx, cvy   = cmd_vel

    cost = (
        w["pos_x"]      * np.abs(df["px"] - gx)
      + w["pos_y"]      * np.abs(df["py"] - gy)
      + w["pos_z"]      * np.abs(df["pz"] - gz)
      + w["orientation"] * (1.0 - np.abs(df["qw"])) ** 2
      + w["vel_x"]      * (df["vx"] - cvx) ** 2
      + w["vel_y"]      * (df["vy"] - cvy) ** 2
      + w["vel_z"]      * df["vz"] ** 2
    )

    return float(cost.mean())


def _locomotion_cost(x, worker_id=0, verbose=False):
    """
    Evaluate one candidate x = [lce_min, lce_max, pFLmax, FVmax] (12D) by
    running mppi_sim. Returns scalar locomotion cost (lower is better);
    does not include the curve-area term (see evaluate()).
    """
    # Load base config each call (safe for multiprocessing; file is read-only)
    with open(BASE_YAML) as f:
        base_cfg = yaml.safe_load(f)

    quad_base = base_cfg["default_muscle_quad"]
    walk_cfg  = base_cfg["walk"]

    muscle_params = _build_muscle_params(x, quad_base)
    # height_target needed by gait_generator for bias torque computation
    muscle_params["height_target"] = walk_cfg["height_target"]

    with tempfile.TemporaryDirectory(prefix=f"cmaes_w{worker_id}_") as tmp_dir:
        # Generate single gait file
        gait_out = os.path.join(tmp_dir, "gaits", "FAST",
                                "activation_gait_FAST_0_1_10cm.tsv")
        try:
            n_inf = generate_gait("FAST", "0_1", "10cm", gait_out, muscle_params,
                                  stiffness=walk_cfg["cost"]["gait_stiffness"])
        except Exception as e:
            if verbose:
                print(f"  [w{worker_id}] gait generation failed: {e}")
            return MAX_INFEASIBLE_PENALTY

        # Feasibility pre-check — skip mppi_sim entirely if any phases are infeasible
        if n_inf > 0:
            penalty = INFEASIBLE_PENALTY_PER_PHASE * n_inf
            if verbose:
                print(f"  [w{worker_id}] INFEASIBLE ({n_inf} phases) → penalty={penalty:.0f}, skipping sim")
            return penalty

        # Write temp yaml
        yaml_path = _write_temp_yaml(base_cfg, muscle_params, gait_out, tmp_dir)
        csv_path  = os.path.join(tmp_dir, "sim.csv")

        # Run mppi_sim
        cmd = [MPPI_SIM, "walk", yaml_path, csv_path]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=SIM_TIMEOUT
            )
            if result.returncode != 0 and verbose:
                print(f"  [w{worker_id}] mppi_sim stderr: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"  [w{worker_id}] mppi_sim timed out")
            return MAX_INFEASIBLE_PENALTY

        # Parse fitness
        cost_weights = walk_cfg["cost"]
        goal_pos     = walk_cfg["cost"]["goal_pos"]
        cmd_vel      = walk_cfg["cmd_vel"]

        cost = _compute_fitness(csv_path, cost_weights, goal_pos, cmd_vel)

    if verbose:
        hip, thigh, calf = (
            f"lce=[{x[0]:.3f},{x[3]:.3f}] pFL={x[6]:.3f} FV={x[9]:.3f}",
            f"lce=[{x[1]:.3f},{x[4]:.3f}] pFL={x[7]:.3f} FV={x[10]:.3f}",
            f"lce=[{x[2]:.3f},{x[5]:.3f}] pFL={x[8]:.3f} FV={x[11]:.3f}",
        )
        print(f"  [w{worker_id}] hip:{hip}  thigh:{thigh}  calf:{calf}"
              f"  → cost={cost:.1f}  inf_phases={n_inf}")

    return cost


def evaluate(x, worker_id=0, verbose=False):
    """
    Evaluate one candidate x = [lce_min, lce_max, pFLmax, FVmax] (12D).
    Returns scalar fitness (lower is better) = locomotion_cost -
    AREA_WEIGHT * curve_area_total(x). AREA_WEIGHT is read fresh from the
    environment on every call (not cached at import time) so cmaes_walk.py
    can set it via os.environ after argparse, before spawning workers.
    Set AREA_WEIGHT=0 (the default) to recover plain locomotion-cost fitness.
    """
    area_total  = curve_area_total(x)
    area_weight = float(os.environ.get("AREA_WEIGHT", 0.0))

    cost    = _locomotion_cost(x, worker_id=worker_id, verbose=verbose)
    fitness = cost - area_weight * area_total

    if verbose and area_weight:
        print(f"  [w{worker_id}] area_total={area_total:.4f} "
              f"(weight={area_weight:g})  cost={cost:.1f}  → fitness={fitness:.1f}")

    return fitness


def render_rollout(x, fps=RENDER_FPS):
    """
    Re-run the sim for candidate x and render its rollout to frames.
    Returns a (T, C, H, W) uint8 numpy array (wandb.Video layout), or None
    if the candidate is infeasible / the sim fails.
    """
    with open(BASE_YAML) as f:
        base_cfg = yaml.safe_load(f)

    quad_base = base_cfg["default_muscle_quad"]
    walk_cfg  = base_cfg["walk"]

    muscle_params = _build_muscle_params(x, quad_base)
    muscle_params["height_target"] = walk_cfg["height_target"]

    with tempfile.TemporaryDirectory(prefix="cmaes_render_") as tmp_dir:
        gait_out = os.path.join(tmp_dir, "gaits", "FAST",
                                "activation_gait_FAST_0_1_10cm.tsv")
        try:
            n_inf = generate_gait("FAST", "0_1", "10cm", gait_out, muscle_params,
                                  stiffness=walk_cfg["cost"]["gait_stiffness"])
        except Exception:
            return None
        if n_inf > 0:
            return None

        yaml_path = _write_temp_yaml(base_cfg, muscle_params, gait_out, tmp_dir)
        csv_path  = os.path.join(tmp_dir, "sim.csv")
        qpos_path = os.path.join(tmp_dir, "sim_qpos.csv")

        try:
            result = subprocess.run([MPPI_SIM, "walk", yaml_path, csv_path],
                                    capture_output=True, text=True, timeout=SIM_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0 or not os.path.exists(qpos_path):
            return None

        qpos_log = np.loadtxt(qpos_path, delimiter=",")
        if qpos_log.ndim == 1:
            qpos_log = qpos_log.reshape(1, -1)
        if len(qpos_log) == 0:
            return None

        sim_hz = round(1.0 / walk_cfg["dt"])
        skip   = max(1, round(sim_hz / fps))

        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data  = mujoco.MjData(model)

        cam           = mujoco.MjvCamera()
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance  = 2.5
        cam.elevation = -15.0
        cam.azimuth   = 90.0

        frames = []
        with mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH) as renderer:
            for qpos in qpos_log[::skip]:
                data.qpos[:len(qpos)] = qpos
                mujoco.mj_forward(model, data)
                cam.lookat[0] = data.qpos[0]
                cam.lookat[1] = data.qpos[1]
                cam.lookat[2] = 0.3
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

    return np.stack(frames).transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
