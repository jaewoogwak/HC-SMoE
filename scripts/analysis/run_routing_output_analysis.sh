#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

analysis_seed="${ANALYSIS_SEED:-123}"
analysis_blocks="${ANALYSIS_BLOCKS:-8}"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" hcsmoe/analyze_routing_output_tradeoff.py \
  --model mixtral \
  --results_dir results/mixtral_8to4 \
  --analysis_seed "$analysis_seed" \
  --analysis_blocks "$analysis_blocks"

"$python_bin" hcsmoe/analyze_routing_output_tradeoff.py \
  --model qwen \
  --results_dir results/qwen_60to30 \
  --analysis_seed "$analysis_seed" \
  --analysis_blocks "$analysis_blocks"
