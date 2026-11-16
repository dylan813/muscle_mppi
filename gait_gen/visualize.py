"""
Gait table viewer: plays generated tables on the suspended Go2 and saves a GIF of each.

The body hangs fixed in the air (scene_suspended.xml) and the joints are set
straight to each column's angles. No physics runs, so the GIF shows the table
itself, not a controller tracking it. Same model, camera and method as
analysis/unit_tests/visualize_source_gait_suspended.py.

Usage, from this folder, after generate.py (MUJOCO_GL=egl renders without a
display, as for analysis/render_gif.py). With no label, every gait in
config.yaml is rendered:
    MUJOCO_GL=egl python visualize.py [label ...]
"""

import math
import os
import sys

import mujoco
import numpy as np
from PIL import Image

from config import REPO_ROOT, load_config, table_path, video_path
from gait import LEGS
from robot import JOINTS

SUSPENDED_MODEL = "unitree_mujoco/unitree_robots/go2/scene_suspended.xml"   # relative to the repo root

SPEED = 1.0       # playback speed: 1 is real time, 0.25 is four times slower
FRAME_RATE = 25   # GIF frames per second, approximately
WIDTH, HEIGHT = 640, 480


def side_camera():
    """Side view of the legs. The suspended body sits 0.8 m above the floor."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.6
    camera.elevation = -15.0
    camera.azimuth = 90.0
    camera.lookat[:] = [0.0, 0.0, 0.6]

    return camera


def render_frames(model, angles, columns_per_frame):
    """Images of the robot posed at every `columns_per_frame`-th column of the table."""
    data = mujoco.MjData(model)
    camera = side_camera()
    addresses = [model.joint(f"{leg}_{joint}_joint").qposadr[0] for leg in LEGS for joint in JOINTS]

    # Show the fewest whole cycles that end on a frame, so the GIF loops without a jump.
    columns = angles.shape[1]
    loop_columns = math.lcm(columns, columns_per_frame)

    frames = []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for column in range(0, loop_columns, columns_per_frame):
            data.qpos[addresses] = angles[:, column % columns]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=camera)
            frames.append(Image.fromarray(renderer.render()))

    return frames


def main():
    config = load_config()
    dt = config["dt"]
    labels = sys.argv[1:] or list(config["gaits"])

    model = mujoco.MjModel.from_xml_path(os.path.join(REPO_ROOT, SUSPENDED_MODEL))

    columns_per_frame = max(1, round(SPEED / (FRAME_RATE * dt)))
    frame_ms = 1000.0 * columns_per_frame * dt / SPEED

    for label in labels:
        table = np.loadtxt(table_path(label), delimiter="\t")
        angles = table[: len(LEGS) * len(JOINTS)]

        frames = render_frames(model, angles, columns_per_frame)

        path = video_path(label)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)

        print(f"wrote {os.path.relpath(path)}: {len(frames)} frames, {SPEED}x real time")


if __name__ == "__main__":
    main()
