"""
The generator's inputs (config.yaml) and where its files live.
"""

import os

import yaml

GAIT_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(GAIT_GEN_DIR, "..")


def load_config():
    with open(os.path.join(GAIT_GEN_DIR, "config.yaml")) as f:
        return yaml.safe_load(f)


def table_path(label):
    """Where generate.py writes the gait table labelled `label`."""
    return os.path.join(GAIT_GEN_DIR, "gaits", f"{label}.tsv")


def video_path(label):
    """Where visualize.py writes that gait's GIF."""
    return os.path.join(GAIT_GEN_DIR, "videos", f"{label}.gif")
