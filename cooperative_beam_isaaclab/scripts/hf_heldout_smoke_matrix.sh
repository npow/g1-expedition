#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CLI_PACKAGE="huggingface_hub>=0.35"
HF_NAMESPACE="${HF_NAMESPACE:-iteratehack}"
HF_FLAVOR="${HF_FLAVOR:-rtx-pro-6000}"
HF_START_INDEX="${HF_START_INDEX:-0}"
HF_END_INDEX="${HF_END_INDEX:-4}"
HF_NAME_SUFFIX="${HF_NAME_SUFFIX:-}"

# One-factor probes isolate the source of a failure. The final row combines
# all three held-out factors and is deliberately expected to be difficult.
tasks=(
  "Isaac-Cooperative-G1-Rescue-Crate-Direct-v0"
  "Isaac-Cooperative-G1-Timber-Direct-v0"
  "Isaac-Cooperative-G1-Rescue-Crate-Direct-v0"
  "Isaac-Cooperative-G1-Footbridge-Girder-Direct-v0"
  "Isaac-Cooperative-G1-Footbridge-Girder-Direct-v0"
)
team_sizes=(4 6 3 5 6)
payload_masses=(16 24 27 20 54)
names=(
  "cooperative-g1-heldout-team-interpolation"
  "cooperative-g1-heldout-team-extrapolation"
  "cooperative-g1-heldout-mass"
  "cooperative-g1-heldout-geometry"
  "cooperative-g1-heldout-compound"
)

for index in "${!tasks[@]}"; do
  if ((index < HF_START_INDEX || index > HF_END_INDEX)); then
    continue
  fi
  for attempt in 1 2 3; do
    if uv tool run --from "${HF_CLI_PACKAGE}" hf jobs run \
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
      --task "${tasks[index]}" \
      --team_size "${team_sizes[index]}" \
      --payload_mass "${payload_masses[index]}" \
      --num_envs 2 --steps 180 --scripted_lift --viz none; then
      break
    fi
    if ((attempt == 3)); then
      echo "Failed to submit ${names[index]} after ${attempt} attempts" >&2
      exit 1
    fi
    sleep 3
  done
done
