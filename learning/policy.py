"""gpc's policy.py: the Policy (network + observation normalizer) that samples
plans from raw observations, saved and loaded as one file, with what it uses:
the observation normalizer, the flow sampler (gpc's Policy.apply), and the
mapping between activations and network units. Also its export to ONNX for
the C++ controller (PlanSampler, export_onnx).

The network works on plans in network units, [-1, 1]. Activations stay in
[ACT_MIN, ACT_MAX] = [0, 1] everywhere else, and are mapped at the network's
boundary as gpc maps any action range (gpc/training.py:412-416, policy.py:101-104).
"""

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx
import torch
from torch import nn

from learning.architectures import DenoisingCNN
from learning.mppi import NUM_MUSCLES, OBS_DIM, KnotSpline

ACT_MIN, ACT_MAX = 0.0, 1.0   # muscle activation range (hill_compute_torques clamps to it)
FLOW_STEPS = 10               # gpc's Policy.dt = 0.1


@dataclass
class ModelConfig:
    """The network's size: config.yaml's model block."""
    feature_dims: list[int]
    timestep_embedding_dim: int
    kernel_size: int


def to_network(act: torch.Tensor) -> torch.Tensor:
    return (act - (ACT_MAX + ACT_MIN) / 2) / ((ACT_MAX - ACT_MIN) / 2)


def from_network(u: torch.Tensor) -> torch.Tensor:
    return u * ((ACT_MAX - ACT_MIN) / 2) + (ACT_MAX + ACT_MIN) / 2


# ── Observation normalizer ───────────────────────────────────────────────────
#
# gpc's, ported as is. gpc uses a Flax BatchNorm with no learned scale or bias
# (gpc/training.py:324-331), which standardizes each observation feature,
# (y - mean) / sqrt(var + eps):
#   - training: each round's observations are standardized with that round's
#     own mean and variance (training.py:420, use_running_average=False)
#   - runtime: with running statistics (policy.py:80), which start at mean 0
#     and variance 1 and after each round become 0.1 · old + 0.9 · that round's
# So in early rounds the runtime statistics still carry the starting values;
# their share shrinks tenfold per round.

NORM_MOMENTUM = 0.1    # gpc: share of the old running statistics kept each round
NORM_EPS      = 1e-5   # Flax BatchNorm default


class ObsNormalizer(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))

    @torch.no_grad()
    def fit_round(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize one round's observations [N, F] with their own statistics,
        and blend those into the running statistics."""
        mean = y.mean(0)
        var  = (y - mean).square().mean(0)   # biased, as Flax computes it
        self.mean.mul_(NORM_MOMENTUM).add_((1 - NORM_MOMENTUM) * mean)
        self.var.mul_(NORM_MOMENTUM).add_((1 - NORM_MOMENTUM) * var)
        return (y - mean) / torch.sqrt(var + NORM_EPS)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Standardize with the running statistics, as at runtime."""
        return (y - self.mean) / torch.sqrt(self.var + NORM_EPS)


# ── Sampler (gpc's Policy.apply, gpc/policy.py:55-106) ───────────────────────

@torch.no_grad()
def sample(model: nn.Module, y: torch.Tensor, prev_plan: torch.Tensor,
           warm_start: float | torch.Tensor = 0.0, steps: int = FLOW_STEPS,
           noise: torch.Tensor | None = None) -> torch.Tensor:
    """Flow a plan from noise, conditioned on y, on a batch.

    y: normalized observations [B, O]. prev_plan: the previous plan [B, H, A],
    in activations. The flow starts at warm_start · prev_plan + (1 - warm_start)
    · noise, with prev_plan first mapped to network units (gpc blends it
    unmapped). Each of the Euler steps is clipped to [-1, 1]. Returns
    activations. noise [B, H, A] is drawn unless given. The model must be in
    eval mode, as gpc samples (model.eval()).

    Written with tensor operations only, so export_onnx() records the whole
    flow, warm start included, rather than freezing it at export.
    """
    if model.training:
        raise ValueError("sample() needs the model in eval mode (model.eval())")
    if noise is None:
        noise = torch.randn_like(prev_plan)

    if not isinstance(warm_start, torch.Tensor):
        warm_start = torch.tensor(warm_start, dtype=prev_plan.dtype, device=prev_plan.device)
    warm_start = warm_start.clamp(0.0, 1.0)
    u = warm_start * to_network(prev_plan) + (1 - warm_start) * noise

    dt = 1.0 / steps
    for k in range(steps):
        t = torch.full_like(u[:, :1, 0], k * dt)   # [B, 1]
        u = torch.clamp(u + dt * model(u, y, t), -1.0, 1.0)
    return from_network(u)


# ── Policy (gpc's Policy) ────────────────────────────────────────────────────

class Policy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.normalizer = ObsNormalizer(OBS_DIM)
        self.model = DenoisingCNN(NUM_MUSCLES, OBS_DIM, cfg.feature_dims,
                                  cfg.kernel_size, cfg.timestep_embedding_dim)

    def sample(self, y: torch.Tensor, prev_plan: torch.Tensor, warm_start: float | torch.Tensor = 0.0,
               noise: torch.Tensor | None = None) -> torch.Tensor:
        """Plans [B, H, A] in activations for raw observations y [B, OBS_DIM],
        normalized with the running statistics as gpc's Policy.apply does."""
        return sample(self.model, self.normalizer(y), prev_plan, warm_start, noise=noise)

    def save(self, path: Path) -> None:
        torch.save({"model_config": asdict(self.cfg), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path, map_location: str = "cpu") -> "Policy":
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        policy = Policy(ModelConfig(**ckpt["model_config"]))
        policy.load_state_dict(ckpt["state"])
        return policy


# ── Export for the C++ controller (no gpc counterpart: gpc runs in JAX) ──────

ONNX_OPSET = 17


class PlanSampler(nn.Module):
    """What the C++ controller runs, as one ONNX graph: the policy's sampler on
    whole plans, in activations. The previous plan [B, H, A] becomes knots, the
    flow starts at warm_start · those knots + (1 - warm_start) · noise
    [B, K, A], and the sampled knots become a plan [B, H, A] again. noise is an
    input, drawn by the caller, so the same inputs always give the same plan."""

    def __init__(self, policy: Policy, spline: KnotSpline):
        super().__init__()
        self.policy = policy
        self.spline = spline

    def forward(self, y: torch.Tensor, prev_plan: torch.Tensor, noise: torch.Tensor,
                warm_start: torch.Tensor) -> torch.Tensor:
        knots = self.policy.sample(y, self.spline.from_plan(prev_plan), warm_start, noise=noise)
        return self.spline.to_plan(knots)


def export_onnx(policy: Policy, spline: KnotSpline, path: Path, task: str) -> None:
    """Write a PlanSampler to an ONNX file, any batch size (the number of policy
    samples), labeled in its metadata with what it was built for, so the C++
    side can refuse a policy that doesn't match its task.

      inputs   y [B, OBS_DIM] raw, prev_plan [B, H, A], noise [B, K, A], warm_start [1]
      output   plan [B, H, A]

    Exporting prints one expected TracerWarning, from KnotSpline.from_plan's
    horizon check: it runs once at export, and the file's fixed H in
    prev_plan's shape enforces it at runtime.
    """
    sampler = PlanSampler(copy.deepcopy(policy).cpu(), spline).eval()
    example = (torch.zeros(2, OBS_DIM), torch.full((2, spline.horizon, NUM_MUSCLES), 0.5),
               torch.zeros(2, spline.n_knots, NUM_MUSCLES), torch.zeros(1))
    batched = ["y", "prev_plan", "noise", "plan"]
    torch.onnx.export(sampler, example, str(path), opset_version=ONNX_OPSET, dynamo=False,
                      input_names=["y", "prev_plan", "noise", "warm_start"], output_names=["plan"],
                      dynamic_axes={name: {0: "batch"} for name in batched})

    model = onnx.load(str(path))
    labels = {"task": task, "horizon": spline.horizon, "knot_steps": ",".join(map(str, spline.steps)),
              "obs_dim": OBS_DIM, "num_muscles": NUM_MUSCLES, "flow_steps": FLOW_STEPS}
    for key, value in labels.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(model, str(path))
