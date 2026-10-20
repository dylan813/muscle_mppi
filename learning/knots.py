"""The policy's plans as cubic knots. The policy outputs n_knots values per
muscle, and a plan's horizon steps are the spline through them, built exactly
as the controller builds its cubic noise (controllers/muscle/control/base_mppi.cpp:
the knot steps in BaseMPPI's constructor, and not_a_knot_cubic_spline()).

Knots replace whole-horizon plans because gpc's network sees only a short
window of a plan. gpc's own plans are 3-4 steps, which the window covers. On
40-step plans the network mixed two modes within one plan; on 4 knots it
doesn't.
"""

from pathlib import Path

import numpy as np
import torch
import yaml


def knot_steps(horizon: int, n_knots: int) -> list[int]:
    """BaseMPPI's knot steps: round(k · (horizon - 1) / (n_knots - 1)), with
    halves rounded up as std::round rounds them."""
    steps = [int(np.floor(k * (horizon - 1) / (n_knots - 1) + 0.5)) for k in range(n_knots)]
    if len(set(steps)) != n_knots:
        raise ValueError(f"{n_knots} knots don't fit distinct steps of a {horizon}-step horizon")
    return steps


def not_a_knot_cubic_spline(xk: np.ndarray, yk: np.ndarray, horizon: int) -> np.ndarray:
    """base_mppi.cpp's not_a_knot_cubic_spline(), evaluated at steps 0..horizon-1."""
    n = len(xk)
    t = np.arange(horizon, dtype=float)
    if n == 2:
        span = xk[1] - xk[0]
        frac = (t - xk[0]) / span if span > 0 else np.zeros(horizon)
        return yk[0] + frac * (yk[1] - yk[0])

    h = np.diff(xk)
    A, rhs = np.zeros((n, n)), np.zeros(n)
    if n == 3:
        # One interior knot: the two not-a-knot conditions are the same equation,
        # so the C++ falls back to a natural spline (S'' = 0 at both ends).
        A[0, 0] = 1.0
        A[1] = [h[0], 2 * (h[0] + h[1]), h[1]]
        rhs[1] = 3 * ((yk[2] - yk[1]) / h[1] - (yk[1] - yk[0]) / h[0])
        A[2, 2] = 1.0
    else:
        A[0, :3] = [-h[1], h[0] + h[1], -h[0]]                         # not-a-knot at x1
        for i in range(1, n - 1):
            A[i, i - 1:i + 2] = [h[i - 1], 2 * (h[i - 1] + h[i]), h[i]]
            rhs[i] = 3 * ((yk[i + 1] - yk[i]) / h[i] - (yk[i] - yk[i - 1]) / h[i - 1])
        A[n - 1, n - 3:] = [-h[n - 2], h[n - 3] + h[n - 2], -h[n - 3]]   # not-a-knot at x(n-2)

    c = np.linalg.solve(A, rhs)
    b = (yk[1:] - yk[:-1]) / h - h * (c[1:] + 2 * c[:-1]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    out = np.empty(horizon)
    seg = 0
    for i, x in enumerate(t):
        while seg < n - 2 and x > xk[seg + 1]:
            seg += 1
        dx = x - xk[seg]
        out[i] = yk[seg] + b[seg] * dx + c[seg] * dx**2 + d[seg] * dx**3
    return out


class KnotSpline:
    """Maps between knots [N, n_knots, A] and plans [N, horizon, A]."""

    def __init__(self, horizon: int, n_knots: int):
        n_knots = min(max(n_knots, 2), horizon)   # BaseMPPI clamps it the same way
        self.horizon = horizon
        self.steps = knot_steps(horizon, n_knots)

        # The spline is linear in its knot values: column j is the spline of knot j alone.
        xk = np.array(self.steps, dtype=float)
        columns = [not_a_knot_cubic_spline(xk, np.eye(n_knots)[j], horizon) for j in range(n_knots)]
        self.basis = torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)   # [horizon, n_knots]

    @classmethod
    def from_task(cls, name: str, tasks_yaml: str | Path) -> "KnotSpline":
        """The knots of an mppi_sim task (controllers/muscle/utils/tasks.yaml):
        its horizon and n_knots, which it must sample with (sample_type: cubic)."""
        task = yaml.safe_load(Path(tasks_yaml).read_text())[name]
        sample_type = task.get("sample_type", "normal")   # BaseMPPI's defaults
        if sample_type != "cubic":
            raise ValueError(f"task {name!r} samples {sample_type!r} noise, not cubic knots")
        return cls(task["horizon"], task.get("n_knots", 4))

    @property
    def n_knots(self) -> int:
        return len(self.steps)

    def to_plan(self, knots: torch.Tensor) -> torch.Tensor:
        """The spline through each muscle's knots, at every horizon step."""
        return torch.einsum("hk,nka->nha", self.basis.to(knots), knots)

    def from_plan(self, plans: torch.Tensor) -> torch.Tensor:
        """A plan's knots: its values at the knot steps, as the controller defines
        a knot spline (and as RTWholeBodyMPPI's perturb_action takes them).

        On the stage-3 run this keeps 85% of the plans' variation along the
        horizon, and step 0 exactly. A least-squares fit keeps 93% but misses
        step 0 by about 0.07; it may replace this later.
        """
        if plans.shape[1] != self.horizon:
            raise ValueError(f"plans have {plans.shape[1]} steps, the task's horizon is {self.horizon}")
        return plans[:, self.steps, :]
