export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM="false"
# Keep a caller-provided cache location; the old placeholder literally created
# a directory named "your-huggingface-home-path" under the repository.
export HF_HOME="${HF_HOME:-/home/jaewoo/.cache/huggingface}"

accelerate launch --config_file static/finetune_config.yaml \
  --main_process_port 29512 hcsmoe/merging-qwen.py \
  --model_name="Qwen/Qwen1.5-MoE-A2.7B-Chat" \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --dominant="no" \
  --similarity_base="expert-output" \
  --cluster="hierarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=30 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --gpu_memory="${GPU_MEMORY:-60GiB}" \
  --cpu_memory="${CPU_MEMORY:-900GiB}" \
  --result_path="results/qwen_60to30/hcsmoe/lm_eval.txt" \
  --output_path="results/qwen_60to30/hcsmoe" \
  "$@"
