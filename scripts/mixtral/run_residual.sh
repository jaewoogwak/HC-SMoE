#!/usr/bin/env bash
set -euo pipefail

export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-your-huggingface-home-path}"

# Reload example:
# bash scripts/mixtral/run_residual.sh --eval_only=True --residual_eval_only=True \
#   --model_path=results/residual/model.pth --output_path=results/residual
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
  --residual_width=256 \
  --residual_data_limit=4096 \
  --residual_epochs=3 \
  --residual_lr=0.001 \
  --residual_batch_size=64 \
  --residual_val_ratio=0.1 \
  --residual_patience=2 \
  --seed=0 \
  --result_path="results/result_mixtral_residual.txt" \
  --output_path="results/residual" \
  "$@" |& tee results/log_mixtral_residual
