"""
The gait: when each foot is on the ground (clock) and where it is (foot path).

Clock. One gait cycle lasts `period` seconds. A leg's phase runs 0 -> 1 over
the cycle, starting at its touchdown: stance while phase < duty, swing after.

Foot path. Assumes the body moves level at constant velocity and stance feet
do not slip. Seen from the body, a stance foot then moves backward at exactly
the body's speed. Positions are in the body frame (x forward, y left, z up),
relative to the foot's nominal point: where it rests when standing. The
stance is centered on it.

Sources:
    go2-convex-mpc (convex_mpc/gait.py): continuous-time phase; swing curve
        (make_swing_trajectory).
    MIT Cheetah-Software (convexMPC/Gait.cpp, OffsetDurationGait): per-leg
        touchdown offsets and the progress output.
    MIT Cheetah-Software (convexMPC/ConvexMPCLocomotion.cpp): stance centered
        on the nominal point, from its Raibert term v * 0.5 * stance_time.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

LEGS = ("FR", "FL", "RR", "RL")   # leg order of the gait tables


# ── Clock ────────────────────────────────────────────────────────────────────

class Contact(Enum):
    STANCE = "stance"   # foot on the ground
    SWING = "swing"     # foot in the air


@dataclass(frozen=True)
class Gait:
    period: float               # s, one full cycle
    duty: float                 # fraction of the cycle each foot is on the ground
    offsets: dict[str, float]   # per leg: fraction of the cycle at which it touches down

    def __post_init__(self):
        if self.period <= 0.0:
            raise ValueError(f"period must be positive, got {self.period}")

        if not 0.0 < self.duty < 1.0:
            raise ValueError(f"duty must be between 0 and 1, got {self.duty}")

        if set(self.offsets) != set(LEGS):
            raise ValueError(f"offsets needs exactly the legs {LEGS}, got {tuple(self.offsets)}")

    @property
    def stance_time(self):
        return self.duty * self.period

    @property
    def swing_time(self):
        return (1.0 - self.duty) * self.period


@dataclass(frozen=True)
class LegPhase:
    contact: Contact
    progress: float   # 0 at the start of the stance or swing, 1 at its end


def leg_phase(gait, leg, t):
    """Where `leg` is in its cycle at time t (s)."""
    phase = (t / gait.period - gait.offsets[leg]) % 1.0

    if phase < gait.duty:
        return LegPhase(Contact.STANCE, phase / gait.duty)

    return LegPhase(Contact.SWING, (phase - gait.duty) / (1.0 - gait.duty))


# ── Foot path ────────────────────────────────────────────────────────────────

def _min_jerk(s):
    """Rises 0 -> 1 over s in [0, 1]; zero velocity and acceleration at both ends."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _bump(s):
    """Rises 0 -> 1 at s = 0.5, falls back to 0 at s = 1; zero velocity and acceleration at both ends."""
    return 64 * s**3 * (1 - s)**3


def foot_offset(gait, velocity, swing_height, phase):
    """
    Foot position relative to its nominal point (m).

    velocity:     body velocity (forward, left), m/s
    swing_height: apex of the swing above the ground, m
    phase:        the leg's LegPhase
    """
    velocity = np.array([velocity[0], velocity[1], 0.0])
    excursion = velocity * gait.stance_time   # how far the foot moves back under the body in stance

    if phase.contact is Contact.STANCE:
        return excursion * (0.5 - phase.progress)

    # Swing: built over the ground, then seen from the moving body. Over the
    # ground the foot leaves and lands at rest, and travels one stride.
    stride = velocity * gait.period
    over_ground = -0.5 * excursion + stride * _min_jerk(phase.progress)
    body_advance = velocity * gait.swing_time * phase.progress

    offset = over_ground - body_advance
    offset[2] = swing_height * _bump(phase.progress)

    return offset
