"""
Reference-Free MPPI controller — entry point.

Usage:
  # MuJoCo viewer (simulation):
  python run_mppi.py --task stand --sim

  # Real robot (Unitree SDK2):
  python run_mppi.py --task walk_forward --interface eth0
"""

import sys
import os
import time
import argparse
import threading
import numpy as np

# Allow running from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control.tasks import get_task, NUM_JOINTS, STAND_POSE
from control.ref_free_mppi import RefFreeMPPI, RobotState


# -------------------------------------------------------------------------
# Standup helper
# -------------------------------------------------------------------------

def interpolate_pose(q_from: np.ndarray, q_to: np.ndarray, alpha: float) -> np.ndarray:
    return q_from + alpha * (q_to - q_from)


# -------------------------------------------------------------------------
# Simulation mode (MuJoCo viewer)
# -------------------------------------------------------------------------

def run_sim(task_name: str):
    import mujoco
    import mujoco.viewer

    task = get_task(task_name)
    mppi = RefFreeMPPI(task)

    mj_model = mujoco.MjModel.from_xml_path(task.model_path)
    mj_data  = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    # State shared between physics thread and MPPI thread
    lock            = threading.Lock()
    latest_state    = RobotState()
    latest_q_tgt    = np.array(task.nominal_pose)
    latest_dq_tgt   = np.zeros(NUM_JOINTS)
    mppi_ready      = threading.Event()
    stop_flag       = threading.Event()

    # Per-actuator qpos/qvel addresses (same logic as MPPI ctor)
    act_qpos_adr = np.zeros(NUM_JOINTS, dtype=int)
    act_qvel_adr = np.zeros(NUM_JOINTS, dtype=int)
    for j in range(NUM_JOINTS):
        jid = int(mj_model.actuator_trnid[j, 0])
        act_qpos_adr[j] = mj_model.jnt_qposadr[jid]
        act_qvel_adr[j] = mj_model.jnt_dofadr[jid]

    def read_state() -> RobotState:
        s = RobotState()
        s.pos  = mj_data.xpos[mppi.base_bid].copy()
        s.vel  = mj_data.cvel[mppi.base_bid, 3:6].copy()
        s.quat = mj_data.xquat[mppi.base_bid].copy()
        s.gyro = mj_data.cvel[mppi.base_bid, 0:3].copy()
        for j in range(NUM_JOINTS):
            s.q[j]  = mj_data.qpos[act_qpos_adr[j]]
            s.dq[j] = mj_data.qvel[act_qvel_adr[j]]
        s.valid = True
        return s

    def mppi_thread_fn():
        nonlocal latest_q_tgt, latest_dq_tgt
        while not stop_flag.is_set():
            t0 = time.time()
            with lock:
                state = latest_state

            if state.valid:
                q, dq = mppi.update(state)
                solve_ms = (time.time() - t0) * 1000.0
                mppi.set_predict_delay(solve_ms / 1000.0)
                with lock:
                    latest_q_tgt  = q
                    latest_dq_tgt = dq
                mppi_ready.set()
                print(f"\rMPPI solve: {solve_ms:.1f} ms", end="", flush=True)

            # 50 Hz target
            elapsed = time.time() - t0
            time.sleep(max(0.0, task.dt_ctrl - elapsed))

    # ---- Standup poses (matching Unitree go2w_stand_example.cpp) ----
    # Phase 1: fold to low crouch first (more stable than going directly to stand)
    q_crouch = np.array(task.nominal_pose)
    q_crouch[:12] = [0.0, 1.36, -2.65,   # FR
                     0.0, 1.36, -2.65,   # FL
                     0.0, 1.36, -2.65,   # RR
                     0.0, 1.36, -2.65]   # RL
    q_stand = np.array(task.nominal_pose)

    phase1_steps = int(1.0 / task.dt)   # 1 s: current → crouch
    phase2_steps = int(1.0 / task.dt)   # 1 s: crouch → stand
    settle_steps = int(0.5 / task.dt)   # 0.5 s: hold stand, let dynamics settle

    # ---- Main loop with viewer ----
    with mujoco.viewer.launch_passive(
        mj_model, mj_data,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        viewer.cam.type        = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mppi.base_bid

        def pd_step(q_ref):
            for j in range(NUM_JOINTS):
                mj_data.ctrl[j] = (
                    task.kp * (q_ref[j] - mj_data.qpos[act_qpos_adr[j]])
                    - task.kd * mj_data.qvel[act_qvel_adr[j]]
                )
            mujoco.mj_step(mj_model, mj_data)

        print("Standup: phase 1 — fold to crouch...")
        q_init = np.array([mj_data.qpos[act_qpos_adr[j]] for j in range(NUM_JOINTS)])
        for step_i in range(phase1_steps):
            alpha = step_i / phase1_steps
            pd_step(interpolate_pose(q_init, q_crouch, alpha))
            if step_i % 50 == 0:
                viewer.sync()

        print("Standup: phase 2 — rise to stand...")
        for step_i in range(phase2_steps):
            alpha = step_i / phase2_steps
            pd_step(interpolate_pose(q_crouch, q_stand, alpha))
            if step_i % 50 == 0:
                viewer.sync()

        print("Standup: settling...")
        for step_i in range(settle_steps):
            pd_step(q_stand)
            if step_i % 50 == 0:
                viewer.sync()
        print("Standup done.")

        # Seed MPPI with the post-standup state, then start the thread
        with lock:
            latest_state    = read_state()
            latest_q_tgt    = np.array(task.nominal_pose)
            latest_dq_tgt   = np.zeros(NUM_JOINTS)

        t_mppi = threading.Thread(target=mppi_thread_fn, daemon=True)
        t_mppi.start()

        while viewer.is_running():
            step_start = time.time()

            # Apply current MPPI targets as PD control
            with lock:
                q_tgt  = latest_q_tgt.copy()
                dq_tgt = latest_dq_tgt.copy()

            for j in range(NUM_JOINTS):
                q_err  = q_tgt[j]  - mj_data.qpos[act_qpos_adr[j]]
                dq_err = dq_tgt[j] - mj_data.qvel[act_qvel_adr[j]]
                if j < 12:
                    mj_data.ctrl[j] = task.kp * q_err + task.kd * dq_err
                else:
                    mj_data.ctrl[j] = 0.5 * dq_err

            mujoco.mj_step(mj_model, mj_data)

            # Update shared state
            with lock:
                latest_state = read_state()

            viewer.sync()

            # Pace to real-time
            dt_remaining = task.dt - (time.time() - step_start)
            if dt_remaining > 0:
                time.sleep(dt_remaining)

    stop_flag.set()


# -------------------------------------------------------------------------
# Real-robot mode (Unitree SDK2)
# -------------------------------------------------------------------------

def run_real(task_name: str, interface: str):
    from unitree_sdk2py.core.channel import (
        ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize,
    )
    from unitree_sdk2py.idl.default import (
        unitree_go_msg_dds__LowState_,
        unitree_go_msg_dds__LowCmd_,
    )
    from unitree_sdk2py.utils.crc import CRC

    task = get_task(task_name)
    mppi = RefFreeMPPI(task)

    ChannelFactoryInitialize(0, interface)

    # ---- Shared state ----
    lock          = threading.Lock()
    robot_state   = RobotState()
    latest_q_tgt  = np.array(task.nominal_pose)
    latest_dq_tgt = np.zeros(NUM_JOINTS)
    stop_flag     = threading.Event()

    def low_state_cb(msg: unitree_go_msg_dds__LowState_):
        with lock:
            for j in range(NUM_JOINTS):
                robot_state.q[j]  = msg.motor_state[j].q
                robot_state.dq[j] = msg.motor_state[j].dq
            robot_state.gyro[0] = msg.imu_state.gyroscope[0]
            robot_state.gyro[1] = msg.imu_state.gyroscope[1]
            robot_state.gyro[2] = msg.imu_state.gyroscope[2]
            robot_state.quat[0] = msg.imu_state.quaternion[0]  # w
            robot_state.quat[1] = msg.imu_state.quaternion[1]  # x
            robot_state.quat[2] = msg.imu_state.quaternion[2]  # y
            robot_state.quat[3] = msg.imu_state.quaternion[3]  # z
            robot_state.valid   = True

    state_sub = ChannelSubscriber("rt/lowstate", unitree_go_msg_dds__LowState_)
    state_sub.Init(low_state_cb, 10)

    cmd_pub = ChannelPublisher("rt/lowcmd", unitree_go_msg_dds__LowCmd_)
    cmd_pub.Init()
    crc = CRC()

    def publish_cmd(q_tgt: np.ndarray, dq_tgt: np.ndarray, kp_scale: float = 1.0):
        cmd = unitree_go_msg_dds__LowCmd_()
        for j in range(NUM_JOINTS):
            cmd.motor_cmd[j].mode = 0x01
            cmd.motor_cmd[j].q    = float(q_tgt[j])
            cmd.motor_cmd[j].dq   = float(dq_tgt[j])
            if j < 12:
                cmd.motor_cmd[j].kp  = task.kp * kp_scale
                cmd.motor_cmd[j].kd  = task.kd
            else:
                cmd.motor_cmd[j].kp  = 0.0
                cmd.motor_cmd[j].kd  = 0.5
            cmd.motor_cmd[j].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        cmd_pub.Write(cmd)

    # ---- Wait for valid state ----
    print("Waiting for robot state...")
    while not robot_state.valid:
        time.sleep(0.01)
    print("Robot state received.")

    # ---- Standup: interpolate to stand pose over 3 s at 500 Hz ----
    print("Standup sequence (3 s)...")
    with lock:
        q_init = robot_state.q.copy()
    q_stand = np.array(task.nominal_pose)
    standup_secs = 3.0
    t_start = time.time()
    while (elapsed := time.time() - t_start) < standup_secs:
        alpha = elapsed / standup_secs
        q_ref = interpolate_pose(q_init, q_stand, alpha)
        publish_cmd(q_ref, np.zeros(NUM_JOINTS), kp_scale=alpha)
        time.sleep(0.002)
    print("Standup done.")

    # ---- MPPI thread ----
    def mppi_thread_fn():
        nonlocal latest_q_tgt, latest_dq_tgt
        while not stop_flag.is_set():
            t0 = time.time()
            with lock:
                state = RobotState(
                    q=robot_state.q.copy(),
                    dq=robot_state.dq.copy(),
                    gyro=robot_state.gyro.copy(),
                    quat=robot_state.quat.copy(),
                    valid=robot_state.valid,
                )
            if state.valid:
                q, dq = mppi.update(state)
                solve_ms = (time.time() - t0) * 1000.0
                mppi.set_predict_delay(solve_ms / 1000.0)
                with lock:
                    latest_q_tgt  = q
                    latest_dq_tgt = dq
                print(f"\rMPPI solve: {solve_ms:.1f} ms", end="", flush=True)
            elapsed = time.time() - t0
            time.sleep(max(0.0, task.dt_ctrl - elapsed))

    t_mppi = threading.Thread(target=mppi_thread_fn, daemon=True)
    t_mppi.start()

    # ---- 50 Hz servo loop ----
    print("Running. Ctrl-C to stop.")
    try:
        while True:
            t0 = time.time()
            with lock:
                q_tgt  = latest_q_tgt.copy()
                dq_tgt = latest_dq_tgt.copy()
            publish_cmd(q_tgt, dq_tgt)
            elapsed = time.time() - t0
            time.sleep(max(0.0, task.dt_ctrl - elapsed))
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stop_flag.set()
        # Send damping-only command
        publish_cmd(np.array(task.nominal_pose), np.zeros(NUM_JOINTS), kp_scale=0.0)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reference-Free MPPI for Go2W")
    parser.add_argument("--task",      type=str, default="stand",
                        choices=["stand", "walk_forward"])
    parser.add_argument("--sim",       action="store_true",
                        help="Run in MuJoCo simulation (default: real robot)")
    parser.add_argument("--interface", type=str, default="eth0",
                        help="Network interface for Unitree SDK2 (real-robot only)")
    args = parser.parse_args()

    if args.sim:
        run_sim(args.task)
    else:
        run_real(args.task, args.interface)


if __name__ == "__main__":
    main()
