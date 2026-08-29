#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOB_NAME" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_NAMESPACE="${HF_NAMESPACE:-iteratehack}"
JOB_NAME="$1"
OUTPUT_DIR="${PROJECT_DIR}/hf_job_outputs/${JOB_NAME}"
ARCHIVE="${OUTPUT_DIR}/artifacts.tar.gz"

mkdir -p "${OUTPUT_DIR}"
uv tool run --from "huggingface_hub>=1.1" hf buckets cp \
  "hf://buckets/${HF_NAMESPACE}/jobs-artifacts/training-results/${JOB_NAME}/artifacts.tar.gz" \
  "${ARCHIVE}"
tar -xzf "${ARCHIVE}" -C "${OUTPUT_DIR}"
echo "Artifacts downloaded to ${OUTPUT_DIR}"
