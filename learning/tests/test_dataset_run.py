"""Training data from one mppi_sim run (--dataset) agrees with that run's CSV.

Run first, from controllers/build:

    ./mppi_sim walk --dataset ../../analysis/data/mppi_sim/mppi_sim --no-gif

Dataset row k comes from the update() whose control step CSV row k logs. Its
observation y is the state that update() planned from, which is the state CSV
row k-1 logged after the previous step; its plan's step 0 is the command the
step ran. The CSV prints 6 significant digits, so values agree to that.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO  = Path(__file__).resolve().parents[2]
RUN   = REPO / "analysis/data/mppi_sim/mppi_sim"
CSV   = RUN.parent / f"{RUN.name}.csv"
OBS   = RUN.parent / f"{RUN.name}_obs.npy"
PLAN  = RUN.parent / f"{RUN.name}_plan.npy"
TASKS = REPO / "controllers/muscle/utils/tasks.yaml"

NUM_JOINTS, NUM_MUSCLES = 12, 24

# Start index of each term in y (controllers/muscle/control/observation.h).
OBS_GOAL, OBS_GRAVITY, OBS_LIN_VEL, OBS_ANG_VEL = 0, 3, 6, 9
OBS_Q   = 12
OBS_DQ  = OBS_Q + NUM_JOINTS
OBS_ACT = OBS_DQ + NUM_JOINTS
OBS_DIM = OBS_ACT + NUM_MUSCLES

CSV_DIGITS = dict(rtol=1e-5, atol=1e-6)   # 6 significant digits, stored as float32

pytestmark = pytest.mark.skipif(
    not (CSV.exists() and OBS.exists() and PLAN.exists()),
    reason=f"no --dataset run at {RUN} (see this file's docstring)")


@pytest.fixture(scope="module")
def run():
    return pd.read_csv(CSV), np.load(OBS), np.load(PLAN)


@pytest.fixture(scope="module")
def walk():
    return yaml.safe_load(TASKS.read_text())["walk"]


def test_one_row_per_csv_row(run, walk):
    csv, obs, plan = run
    stale = "dataset and CSV differ in length: a later run without --dataset replaced the CSV?"
    assert obs.shape == (len(csv), OBS_DIM), stale
    assert plan.shape == (len(csv), walk["horizon"], NUM_MUSCLES), stale


def test_plan_step_0_is_the_command_run(run):
    csv, _, plan = run
    act_cmd = csv[[f"act_cmd_m{m}" for m in range(NUM_MUSCLES)]].to_numpy()
    np.testing.assert_allclose(plan[:, 0, :], act_cmd, **CSV_DIGITS)


def test_observation_is_previous_rows_state(run):
    csv, obs, _ = run
    prev, y = csv.iloc[:-1], obs[1:]
    for start, cols in [(OBS_LIN_VEL, ["vx", "vy", "vz"]),
                        (OBS_ANG_VEL, ["wx", "wy", "wz"]),
                        (OBS_DQ,  [f"dq_j{j}" for j in range(NUM_JOINTS)]),
                        (OBS_ACT, [f"act_m{m}" for m in range(NUM_MUSCLES)])]:
        np.testing.assert_allclose(y[:, start:start + len(cols)], prev[cols].to_numpy(),
                                   **CSV_DIGITS, err_msg=f"{cols[0]}..")


def test_goal_offset_is_from_previous_rows_com(run, walk):
    # px/py/pz are the whole-robot CoM. The heading rotation keeps horizontal
    # distance, so distance and height error can be checked without the yaw.
    csv, obs, _ = run
    prev, y = csv.iloc[:-1], obs[1:]
    goal = np.array(walk["phases"][0]["goal_pos"])

    np.testing.assert_allclose(y[:, OBS_GOAL + 2], goal[2] - prev["pz"].to_numpy(), atol=1e-5)
    distance = np.hypot(goal[0] - prev["px"].to_numpy(), goal[1] - prev["py"].to_numpy())
    np.testing.assert_allclose(np.hypot(y[:, OBS_GOAL], y[:, OBS_GOAL + 1]), distance, atol=1e-5)
