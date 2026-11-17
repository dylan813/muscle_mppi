"""
Planned ground reaction forces: the ground's force on each stance foot, in
every column of a gait table.

The table assumes a level body at constant velocity, so nothing accelerates
and the stance feet only hold the body's weight: their forces must add up to
the weight, upward, with no moment about the centre of mass. Per column, a QP
picks forces that do so:

    minimize    (A F - b)^T S (A F - b) + REGULARIZATION |F|^2
    subject to  |F_x|, |F_y| <= mu F_z   (friction pyramid)
                F_z >= 0                 (the ground only pushes)

over the stance feet's forces F (swing feet carry none). A F stacks the total
force and the total moment about the centre of mass; b is the weight, upward,
and zero moment. S weights the balance error, which stays nonzero only when
the stance feet cannot balance the body exactly (two diagonal feet with the
centre of mass off their line).

Forces are in the world frame, which the level body shares (x forward, y left, z up).

Sources:
    MIT Cheetah-Software (Controllers/BalanceController/BalanceController.cpp;
        Focchi et al. 2016, "High-slope terrain locomotion for torque-controlled
        quadruped robots"): the cost, its weights (FSM_State.cpp,
        runBalanceController) and the friction pyramid. Its smoothing term
        alpha |F - F_prev|^2, for real-time use, is left out.
    MIT Cheetah-Software (Controllers/convexMPC/SolverMPC.cpp): F_z >= 0, the
        lower bound its walking controller uses.
"""

import numpy as np
import quadprog

from gait import LEGS, Contact, leg_phase

FORCE_WEIGHTS = (1.0, 1.0, 10.0)     # S, total force error x, y, z (Cheetah's COM_weights_stance)
MOMENT_WEIGHTS = (20.0, 10.0, 10.0)  # S, moment error about x, y, z (Cheetah's Base_weights_stance)
REGULARIZATION = 1e-3                # Cheetah's (alpha + 1e-3) with alpha = 0: smallest forces among equals


def _skew(r):
    """Matrix of r x (.): skew(r) @ f == np.cross(r, f)."""
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


def balance_problem(robot, angles, legs):
    """
    (A, b) of one column: A @ F, with F the `legs`' forces stacked (3 per leg),
    is the total force and the total moment about the centre of mass; b is what
    holds the body, the weight upward and zero moment. angles: (12,) joint angles.
    """
    com, contacts = robot.com_and_contacts(angles)

    force_rows = np.hstack([np.eye(3) for _ in legs])
    moment_rows = np.hstack([_skew(contacts[leg] - com) for leg in legs])

    A = np.vstack([force_rows, moment_rows])
    b = np.concatenate([-robot.weight, np.zeros(3)])

    return A, b


def _friction_rows(mu):
    """Rows c of one foot's constraints c @ f >= 0: the friction pyramid and F_z >= 0."""
    return np.array([[-1.0, 0.0, mu],
                     [1.0, 0.0, mu],
                     [0.0, -1.0, mu],
                     [0.0, 1.0, mu],
                     [0.0, 0.0, 1.0]])


def _solve(robot, angles, legs):
    """The QP of one column: the `legs`' forces, (len(legs), 3)."""
    A, b = balance_problem(robot, angles, legs)
    S = np.diag(FORCE_WEIGHTS + MOMENT_WEIGHTS)

    # quadprog minimizes 1/2 x^T G x - a^T x subject to C^T x >= c.
    G = 2.0 * (A.T @ S @ A + REGULARIZATION * np.eye(A.shape[1]))
    a = 2.0 * A.T @ S @ b

    constraints = np.zeros((5 * len(legs), 3 * len(legs)))
    for i, leg in enumerate(legs):
        constraints[5 * i:5 * i + 5, 3 * i:3 * i + 3] = _friction_rows(robot.friction(leg))

    forces = quadprog.solve_qp(G, a, constraints.T, np.zeros(len(constraints)))[0]

    return forces.reshape(len(legs), 3)


def stance_legs(gait, t):
    """Legs on the ground at time t (s), in LEGS order."""
    return [leg for leg in LEGS if leg_phase(gait, leg, t).contact is Contact.STANCE]


def foot_forces(gait, robot, angles, dt):
    """
    The ground's planned force on each foot (N, world frame) in every column:
    {leg: (columns, 3)}, zero in swing.
    angles: the table's joint angles, (12, columns).
    """
    columns = angles.shape[1]
    forces = {leg: np.zeros((columns, 3)) for leg in LEGS}

    for column in range(columns):
        legs = stance_legs(gait, column * dt)
        if not legs:
            raise ValueError(f"column {column}: no foot on the ground to hold the body")

        for leg, force in zip(legs, _solve(robot, angles[:, column], legs)):
            forces[leg][column] = force

    return forces
