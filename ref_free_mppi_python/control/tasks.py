from dataclasses import dataclass, field
from typing import List

NUM_JOINTS     = 16
NUM_LEG_JOINTS = 12
NUM_WHEELS     = 4

# Actuator order (matches go2w.xml):
#  0-2:  FR hip/thigh/calf
#  3-5:  FL hip/thigh/calf
#  6-8:  RR hip/thigh/calf
#  9-11: RL hip/thigh/calf
#  12-15: FR/FL/RR/RL wheel

STAND_POSE: List[float] = [
     0.0,  0.67, -1.3,   # FR leg
     0.0,  0.67, -1.3,   # FL leg
     0.0,  0.67, -1.3,   # RR leg
     0.0,  0.67, -1.3,   # RL leg
    0.0, 0.0, 0.0, 0.0,  # wheels
]


@dataclass
class CostWeights:
    height:      float = 100.0
    orientation: float = 10.0
    joint_reg:   float = 0.0
    contact_vel: float = 0.5
    contact_frc: float = 5e-2
    terminal:    float = 2.5e3
    vel_cmd:     float = 0.0
    vel_des:     List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class RefFreeParams:
    K:            int   = 6
    I_iter:       int   = 3
    H_time:       float = 0.9
    beta1:        float = 3.0
    beta2:        float = 3.0
    scale_q_leg:   float = 0.05
    scale_q_wheel: float = 0.15
    scale_v_leg:   float = 0.3
    scale_v_wheel: float = 1.0


@dataclass
class TaskConfig:
    model_path: str = "../../unitree_mujoco/unitree_robots/go2w/scene_mjx.xml"

    height_target: float = 0.45
    nominal_pose:  List[float] = field(default_factory=lambda: list(STAND_POSE))

    cost: CostWeights  = field(default_factory=CostWeights)
    rf:   RefFreeParams = field(default_factory=RefFreeParams)

    n_samples: int   = 50     # tunable; increase if VRAM allows
    lambda_:   float = 0.5

    dt:       float = 0.002
    dt_ctrl:  float = 0.02
    substeps: int   = 10

    kp: float = 50.0
    kd: float =  3.5

    q_min: List[float] = field(default_factory=lambda: [
        -1.0472, -1.5708, -2.7227,
        -1.0472, -1.5708, -2.7227,
        -1.0472, -0.5236, -2.7227,
        -1.0472, -0.5236, -2.7227,
        -1e9, -1e9, -1e9, -1e9,
    ])
    q_max: List[float] = field(default_factory=lambda: [
         1.0472,  3.4907, -0.83776,
         1.0472,  3.4907, -0.83776,
         1.0472,  4.5379, -0.83776,
         1.0472,  4.5379, -0.83776,
         1e9,  1e9,  1e9,  1e9,
    ])


def get_task(name: str) -> TaskConfig:
    cfg = TaskConfig()

    if name == "stand":
        cfg.nominal_pose     = list(STAND_POSE)
        cfg.height_target    = 0.464
        cfg.cost.vel_cmd     = 0.0
        cfg.cost.vel_des     = [0.0, 0.0, 0.0]

    elif name == "walk_forward":
        cfg.nominal_pose    = list(STAND_POSE)
        cfg.height_target   = 0.464
        cfg.cost.height     = 1e2
        cfg.cost.orientation = 10.0
        cfg.cost.joint_reg  = 0.0
        cfg.cost.contact_vel = 0.5
        cfg.cost.contact_frc = 5e-2
        cfg.cost.terminal   = 2.5e3
        cfg.cost.vel_cmd    = 1.0
        cfg.cost.vel_des    = [0.5, 0.0, 0.0]
        cfg.rf.H_time       = 0.9
        cfg.n_samples       = 50

    else:
        raise ValueError(f"Unknown task: {name}")

    return cfg
