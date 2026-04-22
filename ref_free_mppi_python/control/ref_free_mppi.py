"""
Reference-Free Sampling-Based MPC — Schramm et al., arXiv:2511.19204
GPU-accelerated via JAX + MJX (MuJoCo XLA backend).

Algorithm differences vs vanilla MPPI (same as C++ reference implementation):
  1. Cubic Hermite spline parameterisation of joint position AND velocity nodes.
  2. Diffusion-inspired noise annealing (Eq. 8): larger noise for later nodes
     and earlier inner iterations.
  3. Best-trajectory tracking (tau_best): executed actions come from the best
     fully-simulated trajectory, not an untested weighted mixture.
  4. State prediction: simulate ahead to compensate computation delay.
  5. Reference-free cost: no gait priors.
"""

import os
import time
import functools
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from .tasks import TaskConfig, NUM_JOINTS, NUM_LEG_JOINTS, NUM_WHEELS

# Enable Triton GEMM for ~30% throughput improvement on Ampere/Ada GPUs
_xla = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = _xla + " --xla_gpu_triton_gemm_any=True"


@dataclass
class RobotState:
    pos:   np.ndarray = None   # [3] base position in world frame
    vel:   np.ndarray = None   # [3] base linear velocity in world frame
    quat:  np.ndarray = None   # [4] base orientation [w, x, y, z]
    gyro:  np.ndarray = None   # [3] base angular velocity
    q:     np.ndarray = None   # [NUM_JOINTS] joint positions
    dq:    np.ndarray = None   # [NUM_JOINTS] joint velocities
    valid: bool = False

    def __post_init__(self):
        if self.pos  is None: self.pos  = np.zeros(3)
        if self.vel  is None: self.vel  = np.zeros(3)
        if self.quat is None: self.quat = np.array([1., 0., 0., 0.])
        if self.gyro is None: self.gyro = np.zeros(3)
        if self.q    is None: self.q    = np.zeros(NUM_JOINTS)
        if self.dq   is None: self.dq   = np.zeros(NUM_JOINTS)


class RefFreeMPPI:
    """
    JAX/MJX implementation of reference-free MPPI.
    GPU-parallel rollouts via jax.vmap over N samples.
    """

    def __init__(self, task: TaskConfig):
        self.task = task
        K = task.rf.K
        I = task.rf.I_iter
        N = task.n_samples

        # ---- Load model ----
        self.mj_model = mujoco.MjModel.from_xml_path(task.model_path)
        self.mj_model.opt.timestep = task.dt

        self.mx = mjx.put_model(self.mj_model)

        # ---- Actuator → qpos/qvel address mapping ----
        # Corrects LowCmd actuator order (FR/FL/RR/RL) ≠ MuJoCo body-def order.
        act_qpos_adr = np.zeros(NUM_JOINTS, dtype=np.int32)
        act_qvel_adr = np.zeros(NUM_JOINTS, dtype=np.int32)
        for j in range(NUM_JOINTS):
            jid = int(self.mj_model.actuator_trnid[j, 0])
            act_qpos_adr[j] = self.mj_model.jnt_qposadr[jid]
            act_qvel_adr[j] = self.mj_model.jnt_dofadr[jid]
        self.act_qpos_adr = jnp.array(act_qpos_adr)
        self.act_qvel_adr = jnp.array(act_qvel_adr)

        # ---- Base body ID ----
        for candidate in ("trunk", "base", "base_link"):
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if bid >= 0:
                self.base_bid = bid
                print(f"Base body: '{candidate}' (id={bid})")
                break
        else:
            raise RuntimeError("Could not find base body (tried trunk/base/base_link)")

        # ---- Wheel body IDs and axle directions ----
        wheel_names = ["FR_wheel_link", "FL_wheel_link",
                       "RR_wheel_link", "RL_wheel_link"]
        wheel_body_ids = np.zeros(4, dtype=np.int32)
        wheel_axle     = np.zeros((4, 3))
        for i, name in enumerate(wheel_names):
            wid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if wid < 0:
                raise RuntimeError(f"Wheel body '{name}' not found in model")
            wheel_body_ids[i] = wid
            jid = int(self.mj_model.actuator_trnid[NUM_LEG_JOINTS + i, 0])
            wheel_axle[i] = self.mj_model.jnt_axis[jid]  # in body frame
        self.wheel_body_ids = jnp.array(wheel_body_ids)
        self.wheel_axle     = jnp.array(wheel_axle)

        # ---- Per-joint gains and noise scales ----
        kp_arr = np.array([task.kp if j < NUM_LEG_JOINTS else 0.0  for j in range(NUM_JOINTS)])
        kd_arr = np.array([task.kd if j < NUM_LEG_JOINTS else 0.5  for j in range(NUM_JOINTS)])
        self.kp_arr = jnp.array(kp_arr)
        self.kd_arr = jnp.array(kd_arr)

        scale_q = np.array([
            task.rf.scale_q_leg if j < NUM_LEG_JOINTS else task.rf.scale_q_wheel
            for j in range(NUM_JOINTS)
        ])
        scale_v = np.array([
            task.rf.scale_v_leg if j < NUM_LEG_JOINTS else task.rf.scale_v_wheel
            for j in range(NUM_JOINTS)
        ])
        self.scale_q = jnp.array(scale_q)
        self.scale_v = jnp.array(scale_v)

        # ---- Spline dimensions ----
        self.K        = K
        self.dt_spline = task.rf.H_time / (K - 1)
        self.n_ctrl   = int(round(task.rf.H_time / task.dt_ctrl))

        # ---- Noise schedule [I_iter × K] — std dev (covariance = Eq. 8 then sqrt) ----
        noise_sched = np.zeros((I, K))
        for i in range(I):
            for k in range(K):
                noise_sched[i, k] = np.exp(
                    -0.5 * i / (task.rf.beta1 * I)
                    -0.5 * (K - k) / (task.rf.beta2 * K)
                )
        self.noise_sched = jnp.array(noise_sched)

        # ---- Static arrays for cost/limits ----
        self.q_min        = jnp.array(task.q_min)
        self.q_max        = jnp.array(task.q_max)
        self.nominal_pose = jnp.array(task.nominal_pose)

        # ---- Trajectory state: nominal τ₀ and best τ_best ----
        nom_q = jnp.tile(jnp.array(task.nominal_pose)[None, :], (K, 1))
        nom_v = jnp.zeros((K, NUM_JOINTS))
        self.theta_q_nom  = nom_q
        self.theta_v_nom  = nom_v
        self.theta_q_best = nom_q
        self.theta_v_best = nom_v
        self.best_cost    = 1e12

        # ---- RNG ----
        self.rng = jax.random.PRNGKey(42)

        # ---- Delay compensation ----
        self.predict_steps = 1

        # ---- JIT-compile main rollout fn ----
        self._batched_rollout = self._build_batched_rollout()
        self._warmup(N)

    # ------------------------------------------------------------------
    # JIT construction
    # ------------------------------------------------------------------

    def _build_batched_rollout(self):
        """Build the JIT+vmap'd batched rollout closure capturing static config."""
        mx            = self.mx
        kp_arr        = self.kp_arr
        kd_arr        = self.kd_arr
        act_qpos_adr  = self.act_qpos_adr
        act_qvel_adr  = self.act_qvel_adr
        base_bid      = self.base_bid
        wheel_ids     = self.wheel_body_ids
        wheel_axle    = self.wheel_axle
        nominal_pose  = self.nominal_pose
        q_min         = self.q_min
        q_max         = self.q_max
        K             = self.K
        dt_sp         = self.dt_spline
        n_ctrl        = self.n_ctrl
        substeps      = self.task.substeps
        dt_ctrl       = self.task.dt_ctrl
        w_h           = self.task.cost.height
        w_o           = self.task.cost.orientation
        w_qr          = self.task.cost.joint_reg
        w_cv          = self.task.cost.contact_vel
        w_cf          = self.task.cost.contact_frc
        w_term        = self.task.cost.terminal
        w_vel         = self.task.cost.vel_cmd
        vel_des       = jnp.array(self.task.cost.vel_des)
        h_tgt         = self.task.height_target

        def spline_eval(tq, tv, t):
            """Cubic Hermite spline. tq, tv: [K, NJ]."""
            k = jnp.clip(jnp.floor(t / dt_sp).astype(jnp.int32), 0, K - 2)
            s = jnp.clip((t - k * dt_sp) / dt_sp, 0.0, 1.0)
            h00 = 2*s**3 - 3*s**2 + 1
            h10 = s**3 - 2*s**2 + s
            h01 = -2*s**3 + 3*s**2
            h11 = s**3 - s**2
            q_k  = tq[k];   q_k1 = tq[k + 1]
            v_k  = tv[k];   v_k1 = tv[k + 1]
            q_out = h00*q_k + h10*dt_sp*v_k + h01*q_k1 + h11*dt_sp*v_k1
            dh00 = 6*s**2 - 6*s
            dh10 = 3*s**2 - 4*s + 1
            dh01 = -6*s**2 + 6*s
            dh11 = 3*s**2 - 2*s
            v_out = (dh00*q_k + dh10*dt_sp*v_k + dh01*q_k1 + dh11*dt_sp*v_k1) / dt_sp
            return q_out, v_out

        def step_cost(data):
            """Running cost (Eq. 17)."""
            cost = jnp.zeros(())

            # Height
            z = data.xpos[base_bid, 2]
            cost += w_h * jnp.abs(z - h_tgt)

            # Orientation — SO(3) log angle²
            quat = data.xquat[base_bid]          # [w, x, y, z]
            theta = 2.0 * jnp.arccos(jnp.clip(jnp.abs(quat[0]), 0.0, 1.0))
            cost += w_o * theta**2

            # Joint regularisation (legs only)
            q_leg = data.qpos[act_qpos_adr[:NUM_LEG_JOINTS]]
            cost += w_qr * jnp.sum((q_leg - nominal_pose[:NUM_LEG_JOINTS])**2)

            # Wheel lateral slip + contact force
            for i in range(NUM_WHEELS):
                wid = wheel_ids[i]
                # cvel layout: [ang(3), lin(3)] in world frame
                vel_world = data.cvel[wid, 3:6]
                # Rotate axle from body frame to world frame via xmat [nbody, 9]
                R = data.xmat[wid].reshape(3, 3)
                axle_world = R @ wheel_axle[i]
                slip = jnp.dot(axle_world, vel_world)
                cost += w_cv * jnp.abs(slip)
                # cfrc_ext: [torque(3), force(3)] in world frame — index 5 = Fz
                fz = jnp.abs(data.cfrc_ext[wid, 5])
                cost += w_cf * fz

            # Velocity command tracking (body-frame linear vel)
            base_vel = data.cvel[base_bid, 3:6]
            cost += w_vel * (jnp.abs(base_vel[0] - vel_des[0]) +
                             jnp.abs(base_vel[1] - vel_des[1]))
            return cost

        def terminal_cost(data, pos0):
            """Terminal cost (Eq. 18): L1 horizontal displacement from start."""
            pos = data.xpos[base_bid]
            return w_term * (jnp.abs(pos[0] - pos0[0]) + jnp.abs(pos[1] - pos0[1]))

        def single_rollout(d_init, tq, tv):
            """Full horizon rollout for one sample. Returns scalar cost."""
            pos0 = d_init.xpos[base_bid]

            def ctrl_step(data, t_idx):
                t = t_idx.astype(jnp.float32) * dt_ctrl
                q_tgt, v_tgt = spline_eval(tq, tv, t)
                q_cur = data.qpos[act_qpos_adr]
                v_cur = data.qvel[act_qvel_adr]
                tau   = kp_arr * (q_tgt - q_cur) + kd_arr * (v_tgt - v_cur)
                data  = data.replace(ctrl=tau)

                def substep(d, _):
                    return mjx.step(mx, d), None

                data, _ = jax.lax.scan(substep, data, None, length=substeps)
                return data, step_cost(data)

            t_indices = jnp.arange(n_ctrl)
            final_data, costs = jax.lax.scan(ctrl_step, d_init, t_indices)
            return jnp.sum(costs) + terminal_cost(final_data, pos0)

        # vmap over N samples; d_init shared across all (in_axes=None)
        batched = jax.jit(jax.vmap(single_rollout, in_axes=(None, 0, 0)))
        return batched

    def _warmup(self, N: int):
        """Trigger JIT compilation before the first real solve."""
        print("JIT compiling GPU rollouts (first call takes ~30–120 s)...")
        t0 = time.time()
        dummy = RobotState()
        d_init = self._set_mjx_state(dummy)
        tq_b = jnp.tile(self.theta_q_nom[None], (N, 1, 1))
        tv_b = jnp.zeros((N, self.K, NUM_JOINTS))
        self._batched_rollout(d_init, tq_b, tv_b).block_until_ready()
        print(f"JIT warmup complete in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_mjx_state(self, state: RobotState) -> mjx.Data:
        """Convert RobotState to MJX data (runs on CPU, called once per solve)."""
        mj_d = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, mj_d)

        mj_d.qpos[0:3] = state.pos
        mj_d.qpos[3:7] = state.quat   # [w, x, y, z]
        mj_d.qvel[0:3] = state.vel
        mj_d.qvel[3:6] = state.gyro

        adr  = np.array(self.act_qpos_adr)
        dadr = np.array(self.act_qvel_adr)
        for j in range(NUM_JOINTS):
            mj_d.qpos[adr[j]]  = state.q[j]
            mj_d.qvel[dadr[j]] = state.dq[j]

        mujoco.mj_forward(self.mj_model, mj_d)
        return mjx.put_data(self.mj_model, mj_d)

    # ------------------------------------------------------------------
    # Spline helpers (NumPy, for warm-start shifting)
    # ------------------------------------------------------------------

    def _np_spline_eval(self, tq: np.ndarray, tv: np.ndarray, t: float):
        """NumPy cubic Hermite eval for shift_best."""
        dt = self.dt_spline
        k  = int(np.clip(np.floor(t / dt), 0, self.K - 2))
        s  = np.clip((t - k * dt) / dt, 0.0, 1.0)
        h00 = 2*s**3 - 3*s**2 + 1
        h10 = s**3 - 2*s**2 + s
        h01 = -2*s**3 + 3*s**2
        h11 = s**3 - s**2
        q_k = tq[k]; q_k1 = tq[k + 1]
        v_k = tv[k]; v_k1 = tv[k + 1]
        q_out = h00*q_k + h10*dt*v_k + h01*q_k1 + h11*dt*v_k1
        dh00 = 6*s**2 - 6*s
        dh10 = 3*s**2 - 4*s + 1
        dh01 = -6*s**2 + 6*s
        dh11 = 3*s**2 - 2*s
        v_out = (dh00*q_k + dh10*dt*v_k + dh01*q_k1 + dh11*dt*v_k1) / dt
        return q_out, v_out

    def _shift_best(self, steps: int):
        """Shift tau_best forward by `steps` control steps (warm-start)."""
        tq = np.array(self.theta_q_best)
        tv = np.array(self.theta_v_best)
        new_tq = np.zeros_like(tq)
        new_tv = np.zeros_like(tv)
        for k in range(self.K):
            t = min(k * self.dt_spline + steps * self.task.dt_ctrl, self.task.rf.H_time)
            new_tq[k], new_tv[k] = self._np_spline_eval(tq, tv, t)
        self.theta_q_best = jnp.array(new_tq)
        self.theta_v_best = jnp.array(new_tv)

    # ------------------------------------------------------------------
    # Node-velocity clamping (Eq. 5) — JAX-compatible
    # ------------------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=(0,))
    def _clamp_velocities(self, tq: jnp.ndarray, tv: jnp.ndarray) -> jnp.ndarray:
        """Clamp node velocities to keep spline within joint limits (legs only)."""
        dt   = self.dt_spline
        v_lo = 2.0 * (self.q_min - tq) / dt
        v_hi = 2.0 * (self.q_max - tq) / dt
        is_leg = jnp.arange(NUM_JOINTS)[None, :] < NUM_LEG_JOINTS  # [1, NJ]
        return jnp.where(is_leg, jnp.clip(tv, v_lo, v_hi), tv)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_height_target(self, z: float):
        self.task.height_target = z
        # Rebuild JIT to pick up the new height target
        self._batched_rollout = self._build_batched_rollout()

    def set_predict_delay(self, delay_s: float):
        self.predict_steps = max(1, round(delay_s / self.task.dt_ctrl))

    def update(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """
        One MPC step.

        Returns:
            q_out  [NUM_JOINTS]  joint position targets
            dq_out [NUM_JOINTS]  joint velocity targets
        """
        K = self.K
        N = self.task.n_samples
        I = self.task.rf.I_iter

        # Shift warm-start by predict_steps and use as nominal
        self._shift_best(self.predict_steps)
        self.theta_q_nom = self.theta_q_best
        self.theta_v_nom = self.theta_v_best
        self.best_cost   = 1e12

        # Build initial MJX state (includes predict_steps compensation)
        d_init = self._set_mjx_state(state)

        for iter_i in range(I):
            sigma = self.noise_sched[iter_i]  # [K]

            # Sample perturbations: ε ~ N(0, σ²·scale²)
            self.rng, k1, k2 = jax.random.split(self.rng, 3)
            eps_q = jax.random.normal(k1, (N, K, NUM_JOINTS)) * (
                sigma[None, :, None] * self.scale_q[None, None, :])
            eps_v = jax.random.normal(k2, (N, K, NUM_JOINTS)) * (
                sigma[None, :, None] * self.scale_v[None, None, :])

            tq_b = self.theta_q_nom[None] + eps_q   # [N, K, NJ]
            tv_b = self.theta_v_nom[None] + eps_v

            # Clamp node velocities per sample
            tv_b = jax.vmap(
                lambda tq, tv: self._clamp_velocities(tq, tv),
                in_axes=(0, 0)
            )(tq_b, tv_b)

            # GPU-parallel rollouts
            costs = self._batched_rollout(d_init, tq_b, tv_b)  # [N]

            # Track best fully-simulated trajectory
            best_idx = int(jnp.argmin(costs))
            if float(costs[best_idx]) < self.best_cost:
                self.best_cost    = float(costs[best_idx])
                self.theta_q_best = tq_b[best_idx]
                self.theta_v_best = tv_b[best_idx]

            # MPPI weight update (Algorithm 1)
            c_min  = jnp.min(costs)
            c_max  = jnp.max(costs)
            c_norm = (costs - c_min) / jnp.maximum(c_max - c_min, 1e-8)
            log_w  = -c_norm / self.task.lambda_
            weights = jax.nn.softmax(log_w)          # [N]

            # Weighted average → new nominal
            self.theta_q_nom = jnp.einsum("n,nkj->kj", weights, tq_b)
            self.theta_v_nom = jnp.einsum("n,nkj->kj", weights, tv_b)

        # Extract targets from best trajectory at t=0
        tq = np.array(self.theta_q_best)
        tv = np.array(self.theta_v_best)
        q_out, dq_out = self._np_spline_eval(tq, tv, 0.0)
        return q_out, dq_out
