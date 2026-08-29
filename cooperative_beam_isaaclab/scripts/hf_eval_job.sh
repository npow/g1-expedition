#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CLI_PACKAGE="huggingface_hub>=0.35"
HF_NAMESPACE="${HF_NAMESPACE:-iteratehack}"
HF_FLAVOR="${HF_FLAVOR:-rtx-pro-6000}"
HF_TIMEOUT="${HF_TIMEOUT:-2h}"
HF_JOB_NAME="${HF_JOB_NAME:-cooperative-g1-eval}"
HF_TASK="${HF_TASK:-Isaac-Cooperative-G1-Timber-Direct-v0}"
HF_CHECKPOINT="${HF_CHECKPOINT:?set HF_CHECKPOINT to a local .pt file}"
HF_TEAM_SIZE="${HF_TEAM_SIZE:-3}"
HF_PAYLOAD_MASS="${HF_PAYLOAD_MASS:-8}"
HF_TRANSPORT_SCALE="${HF_TRANSPORT_SCALE:-1.0}"
HF_NUM_ENVS="${HF_NUM_ENVS:-64}"
HF_STEPS="${HF_STEPS:-1200}"
HF_LOAD_MODE="${HF_LOAD_MODE:-full}"

if [[ ! -f "${HF_CHECKPOINT}" ]]; then
  echo "checkpoint does not exist: ${HF_CHECKPOINT}" >&2
  exit 2
fi
checkpoint_dir="$(cd "$(dirname "${HF_CHECKPOINT}")" && pwd)"
checkpoint_name="$(basename "${HF_CHECKPOINT}")"

uv tool run --from "${HF_CLI_PACKAGE}" hf jobs run \
  --detach \
  --namespace "${HF_NAMESPACE}" \
  --name "${HF_JOB_NAME}" \
  --flavor "${HF_FLAVOR}" \
  --timeout "${HF_TIMEOUT}" \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --env ISAACLAB_ROOT=/workspace/isaaclab \
  --volume "${PROJECT_DIR}/src:/workspace/project/src:ro" \
  --volume "${PROJECT_DIR}/scripts:/workspace/project/scripts:ro" \
  --volume "${checkpoint_dir}:/workspace/checkpoint:ro" \
  nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1 \
  /bin/bash /workspace/project/scripts/hf_eval_entrypoint.sh \
  "${HF_TASK}" "/workspace/checkpoint/${checkpoint_name}" "${HF_TEAM_SIZE}" \
  "${HF_PAYLOAD_MASS}" "${HF_TRANSPORT_SCALE}" "${HF_NUM_ENVS}" "${HF_STEPS}" "${HF_LOAD_MODE}"
