#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CLI_PACKAGE="huggingface_hub>=0.35"
HF_NAMESPACE="${HF_NAMESPACE:-iteratehack}"
HF_FLAVOR="${HF_FLAVOR:-rtx-pro-6000}"
HF_START_INDEX="${HF_START_INDEX:-0}"
HF_END_INDEX="${HF_END_INDEX:-2}"
HF_NAME_SUFFIX="${HF_NAME_SUFFIX:-}"
HF_SMOKE_ACTION="${HF_SMOKE_ACTION:---scripted_lift}"
HF_SMOKE_STEPS="${HF_SMOKE_STEPS:-120}"
HF_SMOKE_NUM_ENVS="${HF_SMOKE_NUM_ENVS:-2}"

tasks=(
  "Isaac-Cooperative-G1-Rescue-Crate-Direct-v0"
  "Isaac-Cooperative-G1-Timber-Direct-v0"
  "Isaac-Cooperative-G1-Footbridge-Girder-Direct-v0"
)
names=(
  "cooperative-g1-crate-smoke"
  "cooperative-g1-timber-smoke"
  "cooperative-g1-girder-smoke"
)

for index in "${!tasks[@]}"; do
  if ((index < HF_START_INDEX || index > HF_END_INDEX)); then
    continue
  fi
  uv tool run --from "${HF_CLI_PACKAGE}" hf jobs run \
    --detach \
    --namespace "${HF_NAMESPACE}" \
    --name "${names[index]}${HF_NAME_SUFFIX}" \
    --flavor "${HF_FLAVOR}" \
    --timeout 20m \
    --env ACCEPT_EULA=Y \
    --env PRIVACY_CONSENT=Y \
    --volume "${PROJECT_DIR}/src:/workspace/project/src:ro" \
    --volume "${PROJECT_DIR}/scripts:/workspace/project/scripts:ro" \
    nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
    /workspace/isaaclab/isaaclab.sh -p /workspace/project/scripts/smoke.py \
    --task "${tasks[index]}" --num_envs "${HF_SMOKE_NUM_ENVS}" \
    --steps "${HF_SMOKE_STEPS}" "${HF_SMOKE_ACTION}" --viz none
done
