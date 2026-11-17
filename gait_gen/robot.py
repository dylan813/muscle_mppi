"""
Robot description, read from the MuJoCo model so the simulator's model stays
the only source of leg lengths and joint limits.

Expects unitree_mujoco's Go2 naming: bodies <leg>_hip, <leg>_thigh, <leg>_calf,
<leg>_foot, joints <leg>_hip_joint, <leg>_thigh_joint, <leg>_calf_joint, and
sphere foot geoms named <leg>.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from gait import LEGS
from leg_ik import LegGeometry

JOINTS = ("hip", "thigh", "calf")   # joint order within a leg, as in the gait tables


@dataclass(frozen=True)
class Leg:
    geometry: LegGeometry
    limits: np.ndarray   # (3, 2) rad: lower and upper limit of each joint, in JOINTS order


class Robot:
    def __init__(self, model_path):
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)

        self.legs = {leg: self._read_leg(leg) for leg in LEGS}

    def _read_leg(self, leg):
        # Each body's position is given in its parent's frame: the thigh sits
        # sideways of the hip, the calf below the thigh, the foot below the calf.
        geometry = LegGeometry(
            lateral=self._model.body(f"{leg}_thigh").pos[1],
            thigh=-self._model.body(f"{leg}_calf").pos[2],
            calf=-self._model.body(f"{leg}_foot").pos[2],
        )
        limits = np.array([self._model.joint(f"{leg}_{joint}_joint").range for joint in JOINTS])

        return Leg(geometry, limits)

    def foot_position(self, leg, angles):
        """
        Foot position relative to the hip joint (body frame, m) with the leg's
        joints at `angles`, by the model's forward kinematics.
        """
        mujoco.mj_resetData(self._model, self._data)   # level body, all joints at zero

        for joint, angle in zip(JOINTS, angles):
            address = self._model.joint(f"{leg}_{joint}_joint").qposadr[0]
            self._data.qpos[address] = angle

        mujoco.mj_kinematics(self._model, self._data)

        foot = self._data.xpos[self._model.body(f"{leg}_foot").id]
        hip = self._data.xpos[self._model.body(f"{leg}_hip").id]

        return foot - hip

    @property
    def weight(self):
        """Gravity's force on the whole robot (N, world frame): (3,)."""
        mass = self._model.body_subtreemass[0]   # the world body's subtree: every body

        return mass * self._model.opt.gravity

    def friction(self, leg):
        """
        Sliding friction coefficient of the leg's foot contact. The foot geoms
        have priority 1, so MuJoCo uses their friction over the floor's.
        """
        return self._model.geom(leg).friction[0]

    def com_and_contacts(self, angles):
        """
        Whole-robot centre of mass and each foot's contact point, the bottom of
        its sphere (world frame, m), with the body level and all 12 joints at
        `angles` (table order: LEGS x JOINTS). Returns (com, {leg: point}).
        """
        mujoco.mj_resetData(self._model, self._data)   # level body

        for leg_index, leg in enumerate(LEGS):
            for joint_index, joint in enumerate(JOINTS):
                address = self._model.joint(f"{leg}_{joint}_joint").qposadr[0]
                self._data.qpos[address] = angles[leg_index * len(JOINTS) + joint_index]

        mujoco.mj_kinematics(self._model, self._data)
        mujoco.mj_comPos(self._model, self._data)

        com = self._data.subtree_com[0].copy()   # the world body's subtree: every body
        contacts = {}

        for leg in LEGS:
            foot = self._model.geom(leg)
            if foot.type[0] != mujoco.mjtGeom.mjGEOM_SPHERE:
                raise ValueError(f"foot geom {leg} is not a sphere")

            contacts[leg] = self._data.geom_xpos[foot.id] - np.array([0.0, 0.0, foot.size[0]])

        return com, contacts
