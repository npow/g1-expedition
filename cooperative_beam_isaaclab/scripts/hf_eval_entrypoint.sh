#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task is required}"
CHECKPOINT="${2:?checkpoint path is required}"
TEAM_SIZE="${3:?team size is required}"
PAYLOAD_MASS="${4:?payload mass is required}"
TRANSPORT_SCALE="${5:-1.0}"
NUM_ENVS="${6:-64}"
STEPS="${7:-1200}"
LOAD_MODE="${8:-full}"

extra_args=()
if [[ "${LOAD_MODE}" == "actor_only" ]]; then
  extra_args+=(--actor_only)
fi

/workspace/isaaclab/isaaclab.sh -p /workspace/project/scripts/evaluate.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --team_size "${TEAM_SIZE}" \
  --payload_mass "${PAYLOAD_MASS}" \
  --transport_scale "${TRANSPORT_SCALE}" \
  --num_envs "${NUM_ENVS}" \
  --steps "${STEPS}" \
  --viz none \
  "${extra_args[@]}"
