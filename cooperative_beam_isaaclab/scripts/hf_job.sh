#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CLI_PACKAGE="huggingface_hub>=0.35"
HF_NAMESPACE="${HF_NAMESPACE:-iteratehack}"
HF_FLAVOR="${HF_FLAVOR:-rtx-pro-6000}"
# Hugging Face Jobs always has a platform timeout. Keep it well beyond a
# normal run so max_iterations, rather than a budget-derived clock, stops training.
HF_TIMEOUT="${HF_TIMEOUT:-30d}"
HF_NUM_ENVS="${HF_NUM_ENVS:-512}"
HF_ITERATIONS="${HF_ITERATIONS:-10000}"
HF_JOB_NAME="${HF_JOB_NAME:-cooperative-g1-mappo}"
HF_TASK="${HF_TASK:-Isaac-Cooperative-G1-Timber-Direct-v0}"

# The source tree is synced to a temporary Hub bucket by the Jobs CLI. The
# NVIDIA image runs as uid 1000 while bucket FUSE mounts are nobody:755, so it
# uploads one artifact tarball with a scoped token instead of writing via FUSE.
uv tool run --from "${HF_CLI_PACKAGE}" hf jobs run \
  --detach \
  --namespace "${HF_NAMESPACE}" \
  --name "${HF_JOB_NAME}" \
  --flavor "${HF_FLAVOR}" \
  --timeout "${HF_TIMEOUT}" \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --env ISAACLAB_ROOT=/workspace/isaaclab \
  --env COOP_HF_NAMESPACE="${HF_NAMESPACE}" \
  --env COOP_JOB_NAME="${HF_JOB_NAME}" \
  --secrets HF_TOKEN \
  --volume "${PROJECT_DIR}/src:/workspace/project/src:ro" \
  --volume "${PROJECT_DIR}/scripts:/workspace/project/scripts:ro" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
  /bin/bash /workspace/project/scripts/hf_train_entrypoint.sh \
  "${HF_NUM_ENVS}" "${HF_ITERATIONS}" "${HF_TASK}"
