#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-your-huggingface-home-path}"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):${PYTHONPATH:-}"

# Static same-base eval: pass --eval_only=True --lora_eval_only=False.
# LoRA eval: pass --eval_only=True --lora_eval_only=True --lora_path=results/lora/lora.pth.
accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 hcsmoe/merging-mixtral.py \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --model_name="mistralai/Mixtral-8x7B-v0.1" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hierarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=4 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --start_layer=0 \
  --lora_rank=56 \
  --lora_alpha=56 \
  --lora_data_limit=4096 \
  --lora_epochs=3 \
  --lora_lr=0.0001 \
  --lora_batch_size=64 \
  --lora_val_ratio=0.1 \
  --lora_patience=2 \
  --seed=0 \
  --result_path="results/result_mixtral_lora.txt" \
  --output_path="results/lora" \
  "$@" |& tee results/log_mixtral_lora
