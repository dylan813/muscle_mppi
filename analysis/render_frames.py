"""
Render matched still frames from two saved trials, at identical sim timesteps.

Companion to analysis/log/plot_torque_spike.py: that figure shows the PD
controller commanding ~46 N.m (1.93x its +/-23.7 N.m limit) on RL hip flex-ext
at t = 12100 ms, and these frames show what each robot is doing at that moment.

Two choices worth knowing about:

  * FIXED camera on the gap, not the robot-tracking camera render_gif.py uses.
    Tracking would centre both robots and hide the thing that matters -- that
    PD is still at the gap while the muscle run is a body-length past it.

  * Platform colour is set per controller (orange = PD, green = muscle) by
    overwriting the "platform" material's rgba at runtime. scene_guinea_fowl.xml
    is not modified, so anything else rendering that scene is unaffected.

Frame indexing: mppi_sim starts logging once MPPI has converged, so qpos row k
is sim time t = 0.10 + k*0.01 s. Verified against the CSV's own t column.

Usage:
  python3 render_frames.py [--ms 10000 11500 12100 12600 13500] [--outdir ...]
"""

import argparse
import os

import numpy as np
import matplotlib.colors as mcolors
import mujoco
import yaml
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_DIR, ".."))

TASK = "guinea_fowl"
TASKS_YAML = os.path.join(_REPO, "controllers", "muscle", "utils", "tasks.yaml")
MODEL_BASE = os.path.join(_REPO, "controllers", "build")

TRIALS = {
    "pd": (os.path.join(_DIR, "log", "trials", "workshop", "7pd", "trial_070",
                        "pd_mppi_sim_qpos.csv"), "#d1620a"),
    "muscle": (os.path.join(_DIR, "log", "trials", "workshop", "7muscle", "trial_017",
                            "mppi_sim_qpos.csv"), "#0f8a5f"),
}

DEFAULT_MS = [10000, 11500, 12100, 12600, 13500]
W, H = 1280, 800
T0, DT = 0.10, 0.01          # first logged sample, and the sim timestep

# Fixed view of the gap (platform A ends x=0.80, B starts x=1.05).
CAM = dict(distance=2.1, elevation=-10.0, azimuth=90.0, lookat=[1.05, 0.0, 0.22])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS)
    ap.add_argument("--outdir", default=os.path.join(
        _REPO, "analysis", "data", "mppi_sim", "summer_results", "workshop", "frames"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with open(TASKS_YAML) as f:
        model_path = os.path.normpath(os.path.join(
            MODEL_BASE, yaml.safe_load(f)[TASK]["model_path"]))

    for kind, (qpos_csv, colour) in TRIALS.items():
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        # The scene ships a 640x480 offscreen buffer; raise it for print-res stills.
        model.vis.global_.offwidth, model.vis.global_.offheight = W, H
        mat = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "platform")
        model.mat_rgba[mat] = list(mcolors.to_rgb(colour)) + [1.0]

        q = np.loadtxt(qpos_csv, delimiter=",")
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = (
            CAM["distance"], CAM["elevation"], CAM["azimuth"])
        cam.lookat[:] = CAM["lookat"]

        with mujoco.Renderer(model, height=H, width=W) as r:
            for ms in args.ms:
                row = int(round((ms / 1e3 - T0) / DT))
                if not 0 <= row < len(q):
                    print(f"  {kind} {ms}ms: outside the logged range, skipped")
                    continue
                data.qpos[:q.shape[1]] = q[row]
                mujoco.mj_forward(model, data)
                r.update_scene(data, camera=cam)
                out = os.path.join(args.outdir, f"frame_{ms:05d}ms_{kind}.png")
                Image.fromarray(r.render().copy()).save(out)
                print(f"  {kind:6} {ms:>6} ms  row {row:>4}  "
                      f"x={q[row,0]:.3f}  z={q[row,2]:.3f}  -> {os.path.basename(out)}")

    print(f"\nwrote {len(args.ms)*len(TRIALS)} frames to {args.outdir}")


if __name__ == "__main__":
    main()
