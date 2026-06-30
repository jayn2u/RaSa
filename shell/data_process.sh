#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/env/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -z "${DATASET_ROOT:-}" ]]; then
  echo "DATASET_ROOT is not set. Define it in ${ENV_FILE} or export it before running." >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "DATASET_ROOT does not exist: ${DATASET_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

uv run python data_process.py --dataset_name "CUHK-PEDES" --dataset_root_dir "${DATASET_ROOT}/CUHK-PEDES"
uv run python data_process.py --dataset_name "ICFG-PEDES" --dataset_root_dir "${DATASET_ROOT}/ICFG-PEDES"
uv run python data_process.py --dataset_name "RSTPReid" --dataset_root_dir "${DATASET_ROOT}/RSTPReid"
