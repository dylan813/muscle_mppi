"""Layout of the policy observation y, mirroring
controllers/muscle/control/observation.h (see it for what each term is)."""

NUM_JOINTS  = 12
NUM_MUSCLES = 2 * NUM_JOINTS   # antagonistic pair per joint

# Start index of each term in y.
OBS_GOAL    = 0
OBS_GRAVITY = OBS_GOAL + 3
OBS_LIN_VEL = OBS_GRAVITY + 3
OBS_ANG_VEL = OBS_LIN_VEL + 3
OBS_Q       = OBS_ANG_VEL + 3
OBS_DQ      = OBS_Q + NUM_JOINTS
OBS_ACT     = OBS_DQ + NUM_JOINTS
OBS_DIM     = OBS_ACT + NUM_MUSCLES
