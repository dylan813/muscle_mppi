"""
Objective function for CMA-ES optimization of Hill muscle parameters.

Each call to evaluate(x):
  1. Expands 13D x=[lce_min×3, lce_max×3, pFLmax×3, FVmax×3, stiffness] to
     12-element arrays (hip/thigh/calf values shared across all 4 legs) and
     the co-contraction level stiffness.
  2. Generates the activation gait(s) the walk task's phases use into a temp
     dir (gait_generator.py) and checks them for infeasible phases
  3. Writes a temp tasks.yaml with the candidate's muscle params and the
     CMA-ES task settings (GOAL_POS, CMD_VEL, SIM_DURATION), pointing each
     walk phase's gait_path at its generated gait
  4. Runs mppi_sim as a subprocess (it computes the warm start from these
     muscle params at startup)
  5. Parses the output CSV and computes the mean per-step cost (FITNESS_TERMS)

The cost is the controller's own: mppi_sim logs each executed step's
step_cost() terms (cost_* columns, mppi_locomotion.cpp), so the fitness
scores exactly what MPPI optimizes, not a reimplementation of it.
"""

import os
import sys

# mppi_sim renders its rollout GIF by running analysis/render_gif.py with the
# MUSCLE_MPPI_PYTHON interpreter (controllers/common/log.h; python3 if unset).
# Default it to the Python running this script, which has render_gif.py's
# packages (muscle env).
os.environ.setdefault("MUSCLE_MPPI_PYTHON", sys.executable)

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
import io
import subprocess
import tempfile
import pandas as pd
import yaml

from gait_generator import NAMED_GAIT_SOURCES, generate_gait

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(_HERE, ".."))
MPPI_SIM   = os.path.join(REPO_ROOT, "controllers", "build", "mppi_sim")
MODEL_PATH = os.path.join(REPO_ROOT, "unitree_mujoco", "unitree_robots", "go2", "scene.xml")
BASE_YAML  = os.path.join(REPO_ROOT, "controllers", "muscle", "utils", "tasks.yaml")

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

# step_cost() terms scored, as mppi_sim's CSV columns (weights already applied).
# Left out: cost_ang_vel and cost_gait.
FITNESS_TERMS = ["cost_pos", "cost_orient", "cost_vel"]

# Task settings applied to walk in each candidate's YAML, so tasks.yaml's walk
# keeps its own. Goal x and SIM_DURATION are those of the July 2026 FVmax-sweep
# runs and the Aug 4 2026 run that produced tasks.yaml's muscle values (commits
# 7e977c6, b0c904a); runs end on duration, not on reaching the goal, so every
# run lasts SIM_DURATION unless the robot falls. Goal height 0.27 m is
# RTWholeBodyMPPI's walking height (utils/tasks.py), within 3 mm of the whole-
# robot CoM height walk_fast's joint angles imply on Go2 (0.273 m).
GOAL_POS     = [5.0, 0.0, 0.27]
SIM_DURATION = 10.0   # s

# Commanded body-frame velocity [vx, vy] (m/s): RTWholeBodyMPPI's walking command
# with walk_fast (legged_mppi/whole_body_mppi/utils/tasks.py: walk_straight,
# walk_octagon), copied so results compare directly against it. walk_fast's
# own stride is built for about 0.1 m/s.
CMD_VEL = [0.2, 0.0]


# Joint order: FR_hip(0), FR_thigh(1), FR_calf(2), FL_hip(3), ...
# Type index repeats: [0,1,2, 0,1,2, 0,1,2, 0,1,2]
_JOINT_TYPE = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]


def _broadcast_per_type(vals_by_type):
    """[hip_val, thigh_val, calf_val] → 12-element list following joint order."""
    return [float(vals_by_type[t]) for t in _JOINT_TYPE]


def _build_muscle_params(x, base_quad):
    """
    x = [lce_min_hip, lce_min_thigh, lce_min_calf,
         lce_max_hip, lce_max_thigh, lce_max_calf,
         pFLmax_hip,  pFLmax_thigh,  pFLmax_calf,
         FVmax_hip,   FVmax_thigh,   FVmax_calf,
         stiffness]
    All 4 legs share the same value per joint type; stiffness is one value
    for all joints. Fixed params (vmax, phi_min, phi_max, peak_force) taken
    from base_quad.
    """
    p = copy.deepcopy(base_quad)
    p["lce_min"]   = _broadcast_per_type(x[0:3])
    p["lce_max"]   = _broadcast_per_type(x[3:6])
    p["pFLmax"]    = _broadcast_per_type(x[6:9])
    p["FVmax"]     = _broadcast_per_type(x[9:12])
    p["stiffness"] = float(x[12])
    return p


def _write_temp_yaml(base_cfg, muscle_params, phase_gait_paths, tmp_dir):
    """
    Write a modified tasks.yaml to tmp_dir, with:
    - walk.muscle fields updated to muscle_params
    - walk.model_path  set to absolute MODEL_PATH
    - walk.phases[i].gait_path set to phase_gait_paths[i] (absolute), so
      mppi_sim uses the candidate's gaits instead of generating its own into
      the shared controllers/muscle/gaits/
    - walk run with GOAL_POS, CMD_VEL and SIM_DURATION, ending early only on a fall
    Returns the path to the written yaml.
    """
    cfg = copy.deepcopy(base_cfg)

    # Update muscle params in both the anchor block and the resolved walk.muscle
    # dict. stiffness too: the controller's gait-tracking cost and warm start
    # must use the same value the candidate's gait was generated with.
    for key in ("lce_min", "lce_max", "FVmax", "pFLmax", "stiffness"):
        cfg["default_muscle_quad"][key] = muscle_params[key]
        cfg["walk"]["muscle"][key]      = muscle_params[key]

    # Absolute paths so mppi_sim works from any CWD
    cfg["walk"]["model_path"] = MODEL_PATH
    for phase, gait_path in zip(cfg["walk"]["phases"], phase_gait_paths):
        phase["gait_path"] = gait_path

    walk = cfg["walk"]
    (phase,) = walk["phases"]   # raises unless walk has exactly one phase
    phase["goal_pos"]    = GOAL_POS
    phase["cmd_vel"]     = CMD_VEL
    walk["sim_duration"] = SIM_DURATION
    walk["end_on"]       = "duration"

    yaml_path = os.path.join(tmp_dir, "tasks.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None)
    return yaml_path


def _prepare_candidate(x, tmp_dir):
    """
    Build candidate x's muscle params and generate the activation gait for every
    source gait the walk task's phases use (by desired_gait) into tmp_dir.

    Returns (base_cfg, muscle_params, phase_gait_paths, n_infeasible).
    Raises if a phase has no desired_gait or gait generation fails.
    """
    # Load base config each call (safe for multiprocessing; file is read-only)
    with open(BASE_YAML) as f:
        base_cfg = yaml.safe_load(f)

    muscle_params = _build_muscle_params(x, base_cfg["default_muscle_quad"])

    generated = {}          # source key -> generated gait path
    phase_gait_paths = []
    n_inf = 0
    for phase in base_cfg["walk"]["phases"]:
        key = NAMED_GAIT_SOURCES[phase["desired_gait"]]
        if key not in generated:
            out = os.path.join(tmp_dir, "gaits", f"activation_gait_{key}.tsv")
            n_inf += generate_gait(key, out, muscle_params,
                                   stiffness=muscle_params["stiffness"])
            generated[key] = out
        phase_gait_paths.append(generated[key])

    return base_cfg, muscle_params, phase_gait_paths, n_inf


def _compute_fitness(csv_path):
    """
    Parse mppi_sim CSV and compute the mean per-step sum of FITNESS_TERMS.
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

    return float(df[FITNESS_TERMS].sum(axis=1).mean())


def evaluate(x, worker_id=0, verbose=False):
    """
    Evaluate one candidate x = [lce_min, lce_max, pFLmax, FVmax, stiffness]
    (13D) by running mppi_sim. Returns scalar locomotion cost (lower is better).
    """
    with tempfile.TemporaryDirectory(prefix=f"cmaes_w{worker_id}_") as tmp_dir:
        # Generate the walk task's gait file(s)
        try:
            base_cfg, muscle_params, phase_gait_paths, n_inf = _prepare_candidate(x, tmp_dir)
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
        yaml_path = _write_temp_yaml(base_cfg, muscle_params, phase_gait_paths, tmp_dir)
        csv_path  = os.path.join(tmp_dir, "sim.csv")

        # Run mppi_sim
        cmd = [MPPI_SIM, "walk", yaml_path, csv_path, "--no-gif"]   # no per-candidate GIF
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

        cost = _compute_fitness(csv_path)

    if verbose:
        hip, thigh, calf = (
            f"lce=[{x[0]:.3f},{x[3]:.3f}] pFL={x[6]:.3f} FV={x[9]:.3f}",
            f"lce=[{x[1]:.3f},{x[4]:.3f}] pFL={x[7]:.3f} FV={x[10]:.3f}",
            f"lce=[{x[2]:.3f},{x[5]:.3f}] pFL={x[8]:.3f} FV={x[11]:.3f}",
        )
        print(f"  [w{worker_id}] hip:{hip}  thigh:{thigh}  calf:{calf}  stiffness={x[12]:.3f}"
              f"  → cost={cost:.1f}  inf_phases={n_inf}")

    return cost


def render_rollout(x):
    """
    Re-run the sim for candidate x and return the rollout GIF mppi_sim renders
    (analysis/render_gif.py), or None if the candidate is infeasible / the sim
    or render fails. Returned in memory, since tmp_dir is deleted on return.
    """
    with tempfile.TemporaryDirectory(prefix="cmaes_render_") as tmp_dir:
        try:
            base_cfg, muscle_params, phase_gait_paths, n_inf = _prepare_candidate(x, tmp_dir)
        except Exception:
            return None
        if n_inf > 0:
            return None

        yaml_path = _write_temp_yaml(base_cfg, muscle_params, phase_gait_paths, tmp_dir)
        csv_path  = os.path.join(tmp_dir, "sim.csv")
        gif_path  = os.path.join(tmp_dir, "sim.gif")

        try:
            result = subprocess.run([MPPI_SIM, "walk", yaml_path, csv_path],
                                    capture_output=True, text=True, timeout=SIM_TIMEOUT)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0 or not os.path.exists(gif_path):
            return None

        with open(gif_path, "rb") as f:
            return io.BytesIO(f.read())
