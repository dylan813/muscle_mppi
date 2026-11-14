"""
The generator's inputs (config.yaml) and where its files live.
"""

import os
import sys

import yaml

GAIT_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(GAIT_GEN_DIR, "..")


def load_config():
    """The config named on the command line, or config.yaml in this folder."""
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(GAIT_GEN_DIR, "config.yaml")

    with open(path) as f:
        return yaml.safe_load(f)


def table_path(config):
    """Where generate.py writes the gait table."""
    return os.path.join(GAIT_GEN_DIR, "gaits", f"{config['name']}.tsv")


def video_path(config):
    """Where visualize.py writes the GIF."""
    return os.path.join(GAIT_GEN_DIR, "videos", f"{config['name']}.gif")
