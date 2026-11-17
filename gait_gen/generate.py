"""
Gait table generator.

Writes one joint-space gait table per entry under `gaits` in config.yaml, in
the form the controllers track: 24 rows, one column per controller tick over
one gait cycle. Rows 0-11 are joint angles (rad), rows 12-23 joint velocities
(rad/s), in FR/FL/RR/RL x hip/thigh/calf order (controllers/common/gait.h,
GaitTable).

Beside each table, <label>_forces.tsv holds the ground's planned force on
each foot (grf.py): 12 rows, FR/FL/RR/RL x world x/y/z (N), zero
in swing, one column per table column.

Each file's first line, starting with '#', records the inputs the table was
generated from. Table readers skip it.

Usage, from this folder:
    python generate.py
"""

import json
import os

import numpy as np

from config import REPO_ROOT, forces_path, load_config, table_path
from gait import Gait, LEGS, foot_offset, leg_phase
from grf import foot_forces
from leg_ik import leg_angles
from robot import JOINTS, Robot

# Largest gap allowed between an intended foot position and the one the
# model's forward kinematics gives for the generated angles.
FOOT_TOLERANCE = 1e-6   # m


def column_count(period, dt):
    """Columns in one gait cycle. The period must be a whole number of ticks."""
    columns = round(period / dt)

    if abs(columns * dt - period) > 1e-9:
        raise ValueError(f"period {period} s is not a whole number of {dt} s ticks")

    return columns


def leg_rows(leg):
    """Table rows holding `leg`'s joint angles."""
    start = LEGS.index(leg) * len(JOINTS)

    return slice(start, start + len(JOINTS))


def foot_targets(gait, stance_height, robot, columns, dt):
    """Each leg's foot position in every column, relative to its hip joint (m): {leg: (columns, 3)}."""
    targets = {}

    for leg in LEGS:
        # Standing, the foot rests straight below the thigh joint.
        nominal = np.array([0.0, robot.legs[leg].geometry.lateral, -stance_height])

        targets[leg] = np.array([
            nominal + foot_offset(gait, leg_phase(gait, leg, column * dt))
            for column in range(columns)
        ])

    return targets


def joint_angles(robot, targets):
    """Rows 0-11 of the table (rad): (12, columns)."""
    columns = len(targets[LEGS[0]])
    angles = np.zeros((len(LEGS) * len(JOINTS), columns))

    for leg in LEGS:
        angles[leg_rows(leg)] = np.array([leg_angles(robot.legs[leg].geometry, foot) for foot in targets[leg]]).T

    return angles


def joint_velocities(angles, dt):
    """Rows 12-23 of the table (rad/s): central difference of the angles, wrapping around the cycle."""
    return (np.roll(angles, -1, axis=1) - np.roll(angles, 1, axis=1)) / (2.0 * dt)


def check_limits(robot, angles):
    """Raise if any joint leaves its range."""
    for leg in LEGS:
        for joint, row, (lower, upper) in zip(JOINTS, angles[leg_rows(leg)], robot.legs[leg].limits):
            if row.min() < lower or row.max() > upper:
                raise ValueError(f"{leg} {joint} spans [{row.min():.3f}, {row.max():.3f}] rad, "
                                 f"outside its limits [{lower:.3f}, {upper:.3f}]")


def check_against_model(robot, angles, targets):
    """Raise if the model's forward kinematics does not reproduce the intended foot positions."""
    for leg in LEGS:
        for column, target in enumerate(targets[leg]):
            gap = np.abs(robot.foot_position(leg, angles[leg_rows(leg), column]) - target).max()

            if gap > FOOT_TOLERANCE:
                raise ValueError(f"{leg} column {column}: the model puts the foot {gap:.2e} m from its target")


def gait_table(gait, stance_height, robot, dt):
    """The 24-row table of one gait."""
    columns = column_count(gait.period, dt)

    targets = foot_targets(gait, stance_height, robot, columns, dt)
    angles = joint_angles(robot, targets)

    check_limits(robot, angles)
    check_against_model(robot, angles, targets)

    return np.vstack([angles, joint_velocities(angles, dt)])


def main():
    config = load_config()
    shared = {key: value for key, value in config.items() if key != "gaits"}
    robot = Robot(os.path.join(REPO_ROOT, config["model_path"]))

    for label, inputs in config["gaits"].items():
        gait = Gait(**inputs)

        try:
            table = gait_table(gait, config["stance_height"], robot, config["dt"])
            forces = foot_forces(gait, robot, table[:len(LEGS) * len(JOINTS)], config["dt"])
        except ValueError as error:
            raise ValueError(f"gait '{label}': {error}") from error

        header = "gait_gen " + json.dumps(shared | inputs)
        force_rows = np.vstack([forces[leg].T for leg in LEGS])

        for path, rows in ((table_path(label), table), (forces_path(label), force_rows)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savetxt(path, rows, delimiter="\t", header=header)

            print(f"wrote {os.path.relpath(path)}: {rows.shape[0]} x {rows.shape[1]}, one {inputs['period']} s cycle")


if __name__ == "__main__":
    main()
