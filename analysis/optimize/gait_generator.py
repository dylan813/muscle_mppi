"""
Importable gait generation — mirrors controllers/muscle/control/activation_gait.cpp
but accepts muscle params directly instead of reading from tasks.yaml.

Used by the CMA-ES optimizer to generate each candidate's activation gaits into
its own temp directory (also counting infeasible phases, which the C++ generator
doesn't report), without touching the canonical tasks.yaml or
controllers/muscle/gaits/, which concurrent candidates would otherwise race on.
"""

import numpy as np
import os
import mujoco

REPO_ROOT  = os.path.join(os.path.dirname(__file__), "..", "..")
SUSP_MODEL = os.path.join(REPO_ROOT, "unitree_mujoco", "unitree_robots", "go2", "scene_suspended.xml")
GAIT_DIR   = os.path.join(REPO_ROOT, "controllers", "pd", "gaits")   # joint-space source gaits

NUM_JOINTS = 12

# desired_gait name -> source gait key. Must match kNamedGaitSources in
# controllers/muscle/control/mppi_locomotion.cpp.
NAMED_GAIT_SOURCES = {
    "in_place":  "FAST_0_0_10cm",
    "walk":      "MED_0_1_10cm",
    "walk_fast": "FAST_0_1_10cm",
    "trot":      "MED_0_5_15cm",
}


def source_gait_path(key):
    """"FAST_0_1_10cm" -> controllers/pd/gaits/FAST/gait_FAST_0_1_10cm.tsv"""
    return os.path.join(GAIT_DIR, key.split("_", 1)[0], f"gait_{key}.tsv")

# ── Hill model (param-explicit versions) ────────────────────────────────────

def _afl_bump(lce, A, mid, B):
    left  = 0.5 * (A + mid)
    right = 0.5 * (mid + B)
    if lce <= A or lce >= B:
        return 0.0
    if lce < left:
        t = (lce - A) / (left - A);     return 0.5 * t * t
    elif lce < mid:
        t = (mid - lce) / (mid - left); return 1.0 - 0.5 * t * t
    elif lce < right:
        t = (lce - mid) / (right - mid); return 1.0 - 0.5 * t * t
    else:
        t = (B - lce) / (B - right);    return 0.5 * t * t

def _active_fl(lce, lce_min_j, lce_max_j):
    return (_afl_bump(lce, lce_min_j, 1.0, lce_max_j)
            + 0.15 * _afl_bump(lce, lce_min_j, 0.5 * (lce_min_j + 0.95), 0.95))

def _passive_fl(lce, lce_max_j, pFLmax_j):
    b = 0.5 * (lce_max_j + 1.0)
    if lce <= 1.0: return 0.0
    if lce <= b:
        t = (lce - 1.0) / (b - 1.0); return 0.25 * pFLmax_j * t ** 3
    t = (lce - b) / (b - 1.0); return 0.25 * pFLmax_j * (1.0 + 3.0 * t)

def _force_vel(velocity, vmax_j, FVmax_j):
    c   = FVmax_j - 1.0
    eff = velocity / vmax_j
    if eff < -1.0: return 0.0
    if eff <= 0.0: return (eff + 1.0) ** 2
    if eff <= c:   return FVmax_j - (c - eff) ** 2 / c
    return FVmax_j

def _moment_arm(lce_min_j, lce_max_j, phi_min_j, phi_max_j):
    eps = 1e-6
    return (lce_max_j - lce_min_j + eps) / (phi_max_j - phi_min_j + eps)

def _lce_pair(q, r, lce_min_j, phi_min_j, phi_max_j):
    lce1 = q *  r + (lce_min_j - r * phi_min_j)
    lce2 = q * -r + (lce_min_j + r * phi_max_j)
    return lce1, lce2

def _constraint_midpoint(q, dq, tau_req, j, p, stiffness):
    """(a1, a2, infeasible) for one joint — same as hill_invert_torque() in
    muscle.h. infeasible is True when no activation pair in [0, 1] produces
    tau_req (the band collapses and a2 is clamped to its midpoint)."""
    lce_min_j  = p["lce_min"][j]
    lce_max_j  = p["lce_max"][j]
    phi_min_j  = p["phi_min"][j]
    phi_max_j  = p["phi_max"][j]
    vmax_j     = p["vmax"][j]
    FVmax_j    = p["FVmax"][j]
    pFLmax_j   = p["pFLmax"][j]
    peak_j     = p["peak_force"][j]

    r = _moment_arm(lce_min_j, lce_max_j, phi_min_j, phi_max_j)
    lce1, lce2 = _lce_pair(q, r, lce_min_j, phi_min_j, phi_max_j)

    FL1 = _active_fl(lce1, lce_min_j, lce_max_j)
    FL2 = _active_fl(lce2, lce_min_j, lce_max_j)
    P1  = _passive_fl(lce1, lce_max_j, pFLmax_j)
    P2  = _passive_fl(lce2, lce_max_j, pFLmax_j)
    FV1 = _force_vel( r * dq, vmax_j, FVmax_j)
    FV2 = _force_vel(-r * dq, vmax_j, FVmax_j)

    eff1 = FL1 * FV1
    eff2 = FL2 * FV2

    C      = tau_req / (-r * peak_j) - (P1 - P2)
    denom2 = eff2 if eff2 > 1e-8 else 1e-8
    a2_lo  = max(0.0, -C / denom2)
    a2_hi  = min(1.0, (eff1 - C) / denom2)
    infeasible = a2_lo > a2_hi   # counted even when eff1 ~ 0 below

    if eff1 < 1e-8:
        return 0.0, 0.0, infeasible

    if infeasible:
        a2_out = np.clip(0.5 * (a2_lo + a2_hi), 0.0, 1.0)
    else:
        a2_out = a2_lo + stiffness * (a2_hi - a2_lo)

    a1_out = np.clip((C + eff2 * a2_out) / eff1, 0.0, 1.0)
    return float(a1_out), float(a2_out), infeasible


def generate_gait(key, output_path, muscle_params, stiffness=0.75):
    """
    Generate the activation gait for source gait `key` (e.g. "FAST_0_1_10cm",
    see source_gait_path()) using the given muscle_params dict.

    muscle_params must contain keys (each a list of 12 floats):
        lce_min, lce_max, FVmax, pFLmax, vmax, phi_min, phi_max, peak_force

    Writes a 24×N TSV to output_path (no header line — it's loaded through an
    explicit gait_path, which the controller never regenerates).
    Returns the number of infeasible (phase, joint) pairs clamped.
    """
    gait_path = source_gait_path(key)
    if not os.path.exists(gait_path):
        raise FileNotFoundError(f"Source gait not found: {gait_path}")

    gait    = np.loadtxt(gait_path, delimiter='\t')
    N       = gait.shape[1]
    q_traj  = gait[:12, :]
    dq_traj = gait[12:24, :]

    model = mujoco.MjModel.from_xml_path(SUSP_MODEL)
    data  = mujoco.MjData(model)
    _jid     = [model.actuator_trnid[i, 0] for i in range(NUM_JOINTS)]
    _qa_adr  = [model.jnt_qposadr[j]       for j in _jid]
    _dof_adr = [model.jnt_dofadr[j]        for j in _jid]

    def get_bias_torques(q_joints, dq_joints):
        # Fixed-base model: only the joints are set (no free joint to place).
        mujoco.mj_resetData(model, data)
        for i in range(NUM_JOINTS):
            data.qpos[_qa_adr[i]]  = q_joints[i]
            data.qvel[_dof_adr[i]] = dq_joints[i]
        mujoco.mj_forward(model, data)
        return np.array([data.qfrc_bias[_dof_adr[i]] for i in range(NUM_JOINTS)])

    a1_out       = np.zeros((NUM_JOINTS, N))
    a2_out       = np.zeros((NUM_JOINTS, N))
    n_infeasible = 0

    for t in range(N):
        tau = get_bias_torques(q_traj[:, t], dq_traj[:, t])
        for j in range(NUM_JOINTS):
            a1_out[j, t], a2_out[j, t], infeasible = _constraint_midpoint(
                q_traj[j, t], dq_traj[j, t], tau[j], j, muscle_params, stiffness)
            n_infeasible += infeasible

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_array = np.vstack([a1_out, a2_out])
    np.savetxt(output_path, out_array, delimiter='\t', fmt='%.8f')

    return n_infeasible
