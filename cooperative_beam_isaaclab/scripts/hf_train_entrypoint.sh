#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${1:-512}"
MAX_ITERATIONS="${2:-10000}"
TASK="${3:-Isaac-Cooperative-G1-Timber-Direct-v0}"
RUN_DIR="$(mktemp -d)"

copy_outputs() {
  if [[ -d "${RUN_DIR}/logs/skrl/cooperative_g1_payload" ]]; then
    upload_dir="/tmp/cooperative-g1-upload"
    artifact="${upload_dir}/artifacts.tar.gz"
    mkdir -p "${upload_dir}"
    tar -czf "${artifact}" -C "${RUN_DIR}" logs
    /workspace/isaaclab/_isaac_sim/python.sh -m pip install \
      --quiet --target /tmp/hf-client "huggingface_hub>=1.1"
    PYTHONPATH=/tmp/hf-client /workspace/isaaclab/_isaac_sim/python.sh -c \
      'import os, sys; from huggingface_hub import sync_bucket; sync_bucket(sys.argv[1], sys.argv[2], token=os.environ["HF_TOKEN"])' \
      "${upload_dir}" \
      "hf://buckets/${COOP_HF_NAMESPACE:-iteratehack}/jobs-artifacts/training-results/${COOP_JOB_NAME:-cooperative-g1}"
  fi
}
trap copy_outputs EXIT

cd "${RUN_DIR}"
/workspace/isaaclab/isaaclab.sh -p /workspace/project/scripts/train.py \
  --task "${TASK}" \
  --algorithm MAPPO \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --viz none
