"""
Posture-constraint generation — mirrors plot_stiffness_cocontraction.py's
torque-balance derivation but accepts muscle params directly instead of
reading from tasks.yaml.

Used by the CMA-ES optimizer to recompute posture_bias / posture_FL1 /
posture_FL2 per candidate, since those fields depend on lce_min, lce_max,
pFLmax (all optimized by CMA-ES) at the standing nominal pose.

Reuses the Hill-model helpers from gait_generator.py so the force-length
math stays numerically identical to muscle.h and to plot_stiffness_
cocontraction.py.
"""

import os
import numpy as np
import mujoco

from gait_generator import _active_fl, _passive_fl, _moment_arm, _lce_pair

REPO_ROOT  = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_PATH = os.path.join(REPO_ROOT, "unitree_mujoco", "unitree_robots", "go2", "scene.xml")

NUM_JOINTS = 12

# Stand-up PD procedure — identical to stand_go2.cpp / plot_stiffness_cocontraction.py.
# Ground contact forces need to build up naturally (not a single mj_forward at
# the nominal pose) for the equilibrium torques to be correct.
STAND_DOWN_POS = [ 0.0473455,  1.22187, -2.44375,
                  -0.0473455,  1.22187, -2.44375,
                   0.0473455,  1.22187, -2.44375,
                  -0.0473455,  1.22187, -2.44375]
DT_SIM   = 0.002
KD_PD    = 3.5
RAMP_SEC = 3.0
HOLD_SEC = 2.0


def _gravity_torques(nominal_pose, model_path):
    """
    Replay the stand_go2.cpp stand-up (tanh kp ramp 20->50, kd=3.5) in MuJoCo
    so ground contact forces develop naturally, then read steady-state joint
    torques (qfrc_bias - qfrc_constraint) in ctrl order (FR/FL/RR/RL).
    """
    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)

    jid     = [model.actuator_trnid[i, 0] for i in range(NUM_JOINTS)]
    qa_adr  = [model.jnt_qposadr[j] for j in jid]
    dof_adr = [model.jnt_dofadr[j]  for j in jid]

    data.qpos[:] = 0.0
    data.qpos[2] = 0.5
    data.qpos[3] = 1.0
    for i in range(NUM_JOINTS):
        data.qpos[qa_adr[i]] = STAND_DOWN_POS[i]
    mujoco.mj_forward(model, data)

    foot_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                 for n in ('FR_foot', 'FL_foot', 'RR_foot', 'RL_foot')]
    foot_bids = [b for b in foot_bids if b >= 0]
    if foot_bids:
        data.qpos[2] -= min(data.xpos[b, 2] for b in foot_bids)
        mujoco.mj_forward(model, data)

    n_steps = int((RAMP_SEC + HOLD_SEC) / DT_SIM)
    t = 0.0
    for _ in range(n_steps):
        t += DT_SIM
        if t < RAMP_SEC:
            phase = np.tanh(t / 1.2)
            kp    = phase * 50.0 + (1.0 - phase) * 20.0
        else:
            phase = 1.0
            kp    = 50.0
        for i in range(NUM_JOINTS):
            q_des = phase * nominal_pose[i] + (1.0 - phase) * STAND_DOWN_POS[i]
            data.ctrl[i] = kp * (q_des - data.qpos[qa_adr[i]]) + KD_PD * (-data.qvel[dof_adr[i]])
        mujoco.mj_step(model, data)

    return np.array([data.qfrc_bias[dof_adr[i]] - data.qfrc_constraint[dof_adr[i]]
                      for i in range(NUM_JOINTS)])


def compute_posture(muscle_params, nominal_pose, model_path=MODEL_PATH):
    """
    Compute posture_bias / posture_FL1 / posture_FL2 (each 12 floats, ctrl
    order FR->FL->RR->RL x hip/thigh/calf) for the given muscle_params at
    nominal_pose. Matches the constraint-line seed used in
    mppi_locomotion.cpp's constructor:

        posture_FL1[j] * a1 - posture_FL2[j] * a2 = posture_bias[j]

    muscle_params must contain keys (each a list of 12 floats):
        lce_min, lce_max, pFLmax, phi_min, phi_max, peak_force
    """
    tau_grav = _gravity_torques(nominal_pose, model_path)

    posture_bias = [0.0] * NUM_JOINTS
    posture_FL1  = [0.0] * NUM_JOINTS
    posture_FL2  = [0.0] * NUM_JOINTS

    for j in range(NUM_JOINTS):
        lce_min_j = muscle_params["lce_min"][j]
        lce_max_j = muscle_params["lce_max"][j]
        phi_min_j = muscle_params["phi_min"][j]
        phi_max_j = muscle_params["phi_max"][j]
        pFLmax_j  = muscle_params["pFLmax"][j]
        peak_j    = muscle_params["peak_force"][j]

        r = _moment_arm(lce_min_j, lce_max_j, phi_min_j, phi_max_j)
        lce1, lce2 = _lce_pair(nominal_pose[j], r, lce_min_j, phi_min_j, phi_max_j)

        FL1 = _active_fl(lce1, lce_min_j, lce_max_j)
        FL2 = _active_fl(lce2, lce_min_j, lce_max_j)
        P1  = _passive_fl(lce1, lce_max_j, pFLmax_j)
        P2  = _passive_fl(lce2, lce_max_j, pFLmax_j)

        posture_bias[j] = round(float(tau_grav[j] / (-r * peak_j) - (P1 - P2)), 6)
        posture_FL1[j]  = round(float(FL1), 6)
        posture_FL2[j]  = round(float(FL2), 6)

    return posture_bias, posture_FL1, posture_FL2
