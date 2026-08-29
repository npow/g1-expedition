#!/usr/bin/env python3
"""Register this task, then hand off to Isaac Lab's maintained skrl trainer."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cooperative_beam_isaaclab  # noqa: F401
from cooperative_beam_isaaclab.tasks.parameter_sharing import install_parameter_shared_runner


def find_isaaclab_root() -> Path:
    candidates = [
        Path(os.environ["ISAACLAB_ROOT"]) if "ISAACLAB_ROOT" in os.environ else None,
        Path("/home/npow/isaac-validation/IsaacLab"),
        PROJECT_ROOT.parent / "IsaacLab",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "scripts/reinforcement_learning/skrl/train.py").is_file():
            return candidate
    raise RuntimeError("Set ISAACLAB_ROOT to an Isaac Lab checkout containing the skrl training script")


if __name__ == "__main__":
    algorithm = ""
    if "--algorithm" in sys.argv:
        algorithm_index = sys.argv.index("--algorithm") + 1
        if algorithm_index < len(sys.argv):
            algorithm = sys.argv[algorithm_index].upper()
    if algorithm == "MAPPO":
        install_parameter_shared_runner(share_parameters=os.environ.get("COOP_PARAMETER_SHARING", "1") != "0")
    train_script = find_isaaclab_root() / "scripts/reinforcement_learning/skrl/train.py"
    runpy.run_path(str(train_script), run_name="__main__")
