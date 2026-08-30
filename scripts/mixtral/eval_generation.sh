#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):${PYTHONPATH:-}"

# Static HC-SMoE example:
# bash scripts/mixtral/eval_generation.sh \
#   --model_path=results/static/model.pth \
#   --output_path=results/static \
#   --eval_only=True
#
# Residual HC-SMoE example:
# bash scripts/mixtral/eval_generation.sh \
#   --model_path=results/residual/model.pth \
#   --output_path=results/residual \
#   --eval_only=True \
#   --residual_eval_only=True

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python hcsmoe/merging-mixtral.py \
  --task=arc_challenge \
  --num_average_groups=4 \
  --eval_only=True \
  --eval_generation=True \
  --eval_batch_size=1 \
  --result_path=results/generation_eval.txt \
  "$@"
