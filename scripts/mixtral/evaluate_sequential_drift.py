#!/usr/bin/env python3
"""Evaluation-only end-to-end sequential drift diagnostic for saved Mixtral HC-SMoE."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from accelerate import cpu_offload
from transformers import AutoTokenizer, MixtralForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcsmoe.merging.mixtral_checkpoint import load_compressed_model_for_evaluation
from hcsmoe.merging.sequential_drift_mixtral import (
    collect_original_trajectory,
    combine_metrics,
    evaluate_candidate_trajectory,
    format_sequential_drift_summary,
    make_global_sample_plan,
    save_sequential_drift_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--static-model-path", required=True)
    parser.add_argument("--group-state-path", required=True)
    parser.add_argument("--residual-path", required=True)
    parser.add_argument("--calibration-blocks", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--sample-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="results/residual/sequential_drift.json")
    return parser.parse_args()


def calibration_loader(*args, **kwargs):
    """Reuse the C4 calibration pipeline without importing MC-evaluation extras."""
    path = ROOT / "hcsmoe" / "evaluation" / "minipile.py"
    spec = importlib.util.spec_from_file_location("hcsmoe_minipile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_calib_dataloder(*args, **kwargs)


def load_cpu_offloaded_teacher(model_name: str) -> MixtralForCausalLM:
    """Use leaf-module CPU offload so any A100 retains execution workspace."""
    if not torch.cuda.is_available():
        raise RuntimeError("Sequential Mixtral drift evaluation requires CUDA")
    model = MixtralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    cpu_offload(model, execution_device=torch.device("cuda:0"))
    print("[Sequential drift] Original teacher uses Accelerate CPU offload on cuda:0")
    return model


def main() -> None:
    args = parse_args()
    if args.calibration_blocks <= 0 or args.block_size <= 0 or args.batch_size <= 0:
        raise ValueError("calibration blocks, block size, and batch size must be positive")
    if args.sample_tokens <= 0:
        raise ValueError("--sample-tokens must be positive")

    # Avoid tokenizer worker-fork warnings; this does not affect model routing.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    loader = calibration_loader(
        "c4",
        tokenizer,
        args.block_size,
        args.calibration_blocks,
        args.batch_size,
        args.num_workers,
        seed=args.seed,
    )
    effective_block_size = min(tokenizer.model_max_length, args.block_size)
    sample_plan = make_global_sample_plan(
        total_tokens=len(loader.dataset) * effective_block_size,
        sample_tokens=args.sample_tokens,
        seed=args.seed,
    )
    print(
        "[Sequential drift] calibration_tokens={} sampled_tokens={} sampling_seed={}".format(
            sample_plan.total_tokens,
            sample_plan.token_count,
            sample_plan.seed,
        )
    )

    print(f"[Sequential drift] Loading original teacher: {args.model_name}")
    teacher = load_cpu_offloaded_teacher(args.model_name)
    original = collect_original_trajectory(teacher, loader, sample_plan)
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[Sequential drift] Loading static HC-SMoE checkpoint")
    static_model, _ = load_compressed_model_for_evaluation(
        args.model_name,
        args.static_model_path,
        args.group_state_path,
        False,
        None,
    )
    static_hidden, static_router = evaluate_candidate_trajectory(
        static_model,
        loader,
        sample_plan,
        original,
        "[Sequential drift] static C4",
    )
    del static_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[Sequential drift] Loading Static+Residual HC-SMoE checkpoint")
    residual_model, _ = load_compressed_model_for_evaluation(
        args.model_name,
        args.static_model_path,
        args.group_state_path,
        True,
        args.residual_path,
    )
    residual_hidden, residual_router = evaluate_candidate_trajectory(
        residual_model,
        loader,
        sample_plan,
        original,
        "[Sequential drift] residual C4",
    )
    del residual_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = combine_metrics(static_hidden, static_router, residual_hidden, residual_router)
    result["config"] = {
        "model_name": args.model_name,
        "static_model_path": args.static_model_path,
        "group_state_path": args.group_state_path,
        "residual_path": args.residual_path,
        "calibration_blocks": args.calibration_blocks,
        "block_size": effective_block_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "sample_tokens": sample_plan.token_count,
        "calibration_tokens": sample_plan.total_tokens,
        "sampling_seed": args.seed,
        "forward_definition": "independent sequential end-to-end forwards; each model uses its own hidden states and router top-2",
        "forward_dtype": "bfloat16",
        "metric_accumulation_dtype": "float32",
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plots = save_sequential_drift_plots(result, output_path)
    summary = format_sequential_drift_summary(result)
    print(summary)
    print(f"[Sequential drift] JSON: {output_path}")
    for plot in plots:
        print(f"[Sequential drift] Plot: {plot}")


if __name__ == "__main__":
    main()
