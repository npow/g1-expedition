#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_ROOT:-/home/npow/isaac-validation/IsaacLab}"
ISAACLAB_VENV="${ISAACLAB_VENV:-/home/npow/isaac-validation/.isaac-venv}"

source "${ISAACLAB_VENV}/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export ISAACLAB_ROOT="${ISAACLAB_DIR}"

cd "${PROJECT_DIR}"
python scripts/train.py \
  --task Isaac-Cooperative-G1-Timber-Direct-v0 \
  --algorithm MAPPO \
  --num_envs "${NUM_ENVS:-256}" \
  --viz none \
  "$@"

