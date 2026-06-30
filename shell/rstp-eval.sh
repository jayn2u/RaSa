#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

NPROC_PER_NODE="$(uv run python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [ "$NPROC_PER_NODE" -lt 1 ]; then
  echo "No CUDA GPUs are available for evaluation." >&2
  exit 1
fi

uv run python -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --rdzv_endpoint=127.0.0.1:29501 \
Retrieval.py \
--config configs/PS_rstp_reid.yaml \
--output_dir output/rstp-reid/evaluation/ \
--checkpoint checkpoints/rasa_rstp_checkpoint.pth \
--eval_mAP \
--evaluate
