"""
Leg inverse kinematics: foot position -> hip, thigh and calf angles.

Leg model (Go2, unitree_mujoco go2.xml): the hip joint turns about the body's
x axis. The thigh joint sits `lateral` to its side and turns about y, as does
the calf joint (the knee). At zero angles the leg hangs straight down.

Closed form, exact. Own derivation, checked against MuJoCo forward kinematics.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegGeometry:
    lateral: float   # m, thigh joint's sideways offset from the hip joint: + left legs, - right legs
    thigh: float     # m, thigh joint to knee
    calf: float      # m, knee to foot centre


def leg_angles(geometry, foot):
    """
    Angles (hip, thigh, calf), rad, that put the foot centre at `foot`.

    foot: position relative to the hip joint, body frame (x forward, y left, z up), m.
    The calf angle comes out negative: the knee bends backward.
    """
    x, y, z = foot

    # Front view: the foot's distance from the hip axis splits into the
    # sideways offset and the leg's drop within its own plane.
    drop_squared = y**2 + z**2 - geometry.lateral**2
    if drop_squared <= 0.0:
        raise ValueError(f"foot {foot} is nearer the hip axis than the lateral offset allows")

    drop = np.sqrt(drop_squared)
    hip = np.arctan2(z, y) - np.arctan2(-drop, geometry.lateral)

    # Side view, in the leg's plane: thigh, calf and the line from thigh joint
    # to foot form a triangle. The law of cosines gives the knee.
    cos_calf = (x**2 + drop**2 - geometry.thigh**2 - geometry.calf**2) / (2.0 * geometry.thigh * geometry.calf)
    if not -1.0 <= cos_calf <= 1.0:
        raise ValueError(f"foot {foot} is out of the leg's reach")

    calf = -np.arccos(cos_calf)

    # The thigh points at the foot, less the part of that direction the bent knee supplies.
    knee_part = np.arctan2(geometry.calf * np.sin(calf), geometry.thigh + geometry.calf * np.cos(calf))
    thigh = np.arctan2(-x, drop) - knee_part

    return np.array([hip, thigh, calf])
