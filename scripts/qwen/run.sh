export NCCL_P2P_DISABLE=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
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
  --num_average_groups=45 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  --gpu_memory="14GiB" \
  --cpu_memory="900GiB" \
  --result_path="results/results_qwen_test.txt" \
  --output_path="results/qwen/merge-45e/test" \
  "$@" |& tee results/log_45e_test
