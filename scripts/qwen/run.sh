export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-/home/jaewoo/.cache/huggingface}"

# HC-SMoE baseline:
# bash scripts/qwen/run.sh --grouping_method=hcsmoe --calib_seed=42 \
#   --output_path=results/qwen_60to30/hcsmoe \
#   --result_path=results/qwen_60to30/hcsmoe/lm_eval.txt
#
# Pure Routing-Aware:
# bash scripts/qwen/run.sh --grouping_method=routing_aware --alpha=1.0 --calib_seed=42 \
#   --output_path=results/qwen_60to30/routing_aware_a100 \
#   --result_path=results/qwen_60to30/routing_aware_a100/lm_eval.txt

accelerate launch --config_file static/finetune_config.yaml --main_process_port 29512 \
  hcsmoe/merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --similarity_base="expert-output" \
  --cluster="hierarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --gpu_memory="14GiB" \
  --cpu_memory="900GiB" \
  "$@"
