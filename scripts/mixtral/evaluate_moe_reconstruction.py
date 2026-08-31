#!/usr/bin/env python3
"""Evaluation-only frozen-routing MoE reconstruction for saved Mixtral HC-SMoE."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, MixtralForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hcsmoe.merging.mixtral_checkpoint import load_compressed_model_for_evaluation
from hcsmoe.merging.moe_reconstruction_mixtral import collect_frozen_moe_tokens, evaluate_frozen_moe_reconstruction, format_moe_reconstruction_summary, materialize_original_outputs


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="mistralai/Mixtral-8x7B-v0.1")
    p.add_argument("--static-model-path", required=True)
    p.add_argument("--group-state-path", required=True)
    p.add_argument("--residual-path")
    p.add_argument("--use-residual", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--calibration-blocks", type=int, default=32)
    p.add_argument("--max-block-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--moe-reconstruction-limit", type=int, default=4096, help="whole top-2 tokens/layer")
    p.add_argument("--sanity-tokens", type=int, default=8)
    p.add_argument("--output-path", required=True)
    return p.parse_args()


def calibration_loader(*args, **kwargs):
    # Avoid importing hcsmoe.evaluation.__init__, whose MC-eval extras are not
    # needed for this C4-only reconstruction experiment.
    path = ROOT / "hcsmoe" / "evaluation" / "minipile.py"
    spec = importlib.util.spec_from_file_location("hcsmoe_minipile", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.get_calib_dataloder(*args, **kwargs)


def paths(output):
    path = Path(output)
    return (path, path.with_suffix(".txt")) if path.suffix == ".json" else (path / "moe_reconstruction.json", path / "moe_reconstruction.txt")


def main():
    a = args()
    if a.use_residual and not a.residual_path: raise ValueError("--use-residual requires --residual-path")
    if a.batch_size <= 0 or a.moe_reconstruction_limit <= 0: raise ValueError("batch size and token limit must be positive")
    torch.manual_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model_name); tokenizer.pad_token_id = tokenizer.eos_token_id
    loader = calibration_loader("c4", tokenizer, a.max_block_size, a.calibration_blocks, a.batch_size, a.num_workers, seed=a.seed)
    print(f"[MoE reconstruction] Loading original teacher: {a.model_name}")
    teacher = MixtralForCausalLM.from_pretrained(a.model_name, torch_dtype=torch.bfloat16, device_map="auto"); teacher.eval()
    frozen = collect_frozen_moe_tokens(teacher, loader, a.moe_reconstruction_limit)
    materialize_original_outputs(teacher, frozen)
    del teacher; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    model, groups = load_compressed_model_for_evaluation(a.model_name, a.static_model_path, a.group_state_path, a.use_residual, a.residual_path)
    model.eval()
    metrics = evaluate_frozen_moe_reconstruction(model, groups, frozen, a.use_residual, a.sanity_tokens)
    metrics["config"] = {"model_name": a.model_name, "static_model_path": a.static_model_path, "group_state_path": a.group_state_path, "residual_path": a.residual_path, "residual_enabled": a.use_residual, "calibration_blocks": a.calibration_blocks, "max_block_size": a.max_block_size, "batch_size": a.batch_size, "seed": a.seed, "moe_reconstruction_limit": a.moe_reconstruction_limit}
    summary = format_moe_reconstruction_summary(metrics)
    json_path, text_path = paths(a.output_path); json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True)); text_path.write_text(summary + "\n")
    print(summary); print(f"[MoE reconstruction] JSON: {json_path}"); print(f"[MoE reconstruction] Summary: {text_path}")


if __name__ == "__main__":
    main()
