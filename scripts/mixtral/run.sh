export NCCL_P2P_DISABLE=0
export TOKENIZERS_PARALLELISM="false"

# HC-SMoE baseline:
# bash scripts/mixtral/run.sh --grouping_method=hcsmoe --calib_seed=42 \
#   --output_path=results/mixtral_8to4/hcsmoe \
#   --result_path=results/mixtral_8to4/hcsmoe/lm_eval.txt
#
# Routing-aware grouping:
# bash scripts/mixtral/run.sh --grouping_method=routing_aware --alpha=1.0 --calib_seed=42 \
#   --output_path=results/mixtral_8to4/routing_aware_a100 \
#   --result_path=results/mixtral_8to4/routing_aware_a100/lm_eval.txt
#
# Eval-only:
# bash scripts/mixtral/run.sh --eval_only=True --model_path=... --group_state_path=...

accelerate launch --config_file static/finetune_config.yaml --main_process_port 29512 \
  hcsmoe/merging-mixtral.py \
  --task="winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte" \
  --model_name="mistralai/Mixtral-8x7B-v0.1" \
  --similarity_base="expert-output" \
  --cluster="hierarchical" \
  --linkage="average" \
  --merge="freq" \
  --num_average_groups=4 \
  --n_sentences=32 \
  --train_batch_size=2 \
  --eval_batch_size=16 \
  "$@"
