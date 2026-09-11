#!/usr/bin/env python3
"""Analyze vanilla HC-SMoE-induced hidden-state and original-router drift."""
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
    assert_router_weights_unchanged,
    autoregressive_routing_trace,
    collect_fixed_routing_trace,
    compare_decode_traces,
    compare_fixed_routing_traces,
    forced_decode_summary,
    free_generation_summary,
    freeze_for_analysis,
    make_global_sample_plan,
    router_weight_snapshot,
    save_routing_drift_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--model-path", required=True, help="Saved vanilla HC-SMoE model.pth")
    parser.add_argument("--group-state-path", required=True, help="Saved vanilla HC-SMoE group_state_dict.pt")
    parser.add_argument("--sample-tokens", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--calibration-blocks", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/routing_drift/mixtral_hcsmoe_8to4")
    return parser.parse_args()


def calibration_loader(*args, **kwargs):
    """Reuse the repository C4 loader without importing evaluation extras."""
    path = ROOT / "hcsmoe" / "evaluation" / "minipile.py"
    spec = importlib.util.spec_from_file_location("hcsmoe_minipile", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_calib_dataloder(*args, **kwargs)


def _freeze_and_cpu_offload(model: torch.nn.Module) -> torch.nn.Module:
    freeze_for_analysis(model)
    cpu_offload(model, execution_device=torch.device("cuda:0"))
    return model


def load_original_for_analysis(model_name: str) -> tuple[MixtralForCausalLM, list[torch.Tensor]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Mixtral routing-drift analysis requires CUDA")
    model = MixtralForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map={"": "cpu"})
    routers = router_weight_snapshot(model)
    print("[Routing drift] Original model: CPU offload with cuda:0 execution")
    return _freeze_and_cpu_offload(model), routers


def load_merged_for_analysis(
    model_name: str,
    model_path: str,
    group_state_path: str,
    original_routers: list[torch.Tensor],
) -> MixtralForCausalLM:
    model, _ = load_compressed_model_for_evaluation(
        model_name,
        model_path,
        group_state_path,
        False,
        None,
        cpu_offload_for_analysis=True,
    )
    assert_router_weights_unchanged(original_routers, model)
    print("[Routing drift] Vanilla HC-SMoE model: router weights identical; CPU offload with cuda:0 execution")
    return _freeze_and_cpu_offload(model)


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    for name in ("sample_tokens", "block_size", "calibration_blocks", "decode_steps", "prompt_tokens", "batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not Path(args.model_path).is_file() or not Path(args.group_state_path).is_file():
        raise FileNotFoundError("--model-path and --group-state-path must point to saved vanilla HC-SMoE artifacts")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    loader = calibration_loader(
        "c4", tokenizer, args.block_size, args.calibration_blocks,
        args.batch_size, args.num_workers, seed=args.seed,
    )
    effective_block_size = min(tokenizer.model_max_length, args.block_size)
    sample_plan = make_global_sample_plan(
        len(loader.dataset) * effective_block_size, args.sample_tokens, args.seed,
    )

    original_model, original_routers = load_original_for_analysis(args.model_name)
    original_fixed = collect_fixed_routing_trace(
        original_model, loader, sample_plan, description="[Routing drift] Original fixed-input C4",
    )
    prompt_ids = original_fixed.input_id_batches[0][:1, : min(args.prompt_tokens, effective_block_size)].clone()
    original_free = autoregressive_routing_trace(
        original_model, prompt_ids, args.decode_steps, description="Original free generation",
    )
    reference_tokens = original_free.token_ids.clone()
    original_forced = autoregressive_routing_trace(
        original_model, prompt_ids, args.decode_steps, forced_tokens=reference_tokens,
        description="Original forced replay",
    )
    if not torch.equal(original_forced.token_ids, reference_tokens):
        raise AssertionError("original forced replay changed the saved continuation IDs")
    del original_model
    _clear_cuda()

    merged_model = load_merged_for_analysis(
        args.model_name, args.model_path, args.group_state_path, original_routers,
    )
    merged_fixed = collect_fixed_routing_trace(
        merged_model, loader, sample_plan,
        expected_input_batches=original_fixed.input_id_batches,
        description="[Routing drift] Vanilla HC-SMoE fixed-input C4",
    )
    merged_forced = autoregressive_routing_trace(
        merged_model, prompt_ids, args.decode_steps, forced_tokens=reference_tokens,
        description="Vanilla HC-SMoE forced replay",
    )
    merged_free = autoregressive_routing_trace(
        merged_model, prompt_ids, args.decode_steps, description="Vanilla HC-SMoE free generation",
    )
    del merged_model
    _clear_cuda()

    layer_metrics, token_metrics = compare_fixed_routing_traces(
        original_fixed, merged_fixed, sample_plan.positions,
    )
    forced_metrics = compare_decode_traces(original_forced, merged_forced, require_identical_tokens=True)
    forced_summary = forced_decode_summary(forced_metrics)
    free_metrics = compare_decode_traces(original_free, merged_free, require_identical_tokens=False)
    free_summary = free_generation_summary(free_metrics)

    (output_dir / "layer_metrics.json").write_text(json.dumps(layer_metrics, indent=2) + "\n")
    torch.save(token_metrics, output_dir / "token_metrics.pt")
    torch.save(forced_metrics, output_dir / "forced_decode_metrics.pt")
    torch.save(free_metrics, output_dir / "free_generation_metrics.pt")
    summary = {
        "config": {
            "model_name": args.model_name,
            "model_path": str(Path(args.model_path).resolve()),
            "group_state_path": str(Path(args.group_state_path).resolve()),
            "sample_tokens": sample_plan.token_count,
            "block_size": effective_block_size,
            "calibration_blocks": args.calibration_blocks,
            "decode_steps": args.decode_steps,
            "prompt_tokens": prompt_ids.shape[1],
            "seed": args.seed,
            "comparison": "Original vs vanilla HC-SMoE 8-to-4",
            "router_comparison": "unchanged original expert IDs; no projection to merged groups",
            "fixed_forward": "independent sequential forwards with model-owned hidden states and routing",
            "decode_forward": "identical forced token IDs with separate model-owned KV caches",
        },
        "prompt_token_ids": prompt_ids.flatten().tolist(),
        "reference_continuation_token_ids": reference_tokens.tolist(),
        "forced_decode": forced_summary,
        "free_generation": free_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_paths = save_routing_drift_plots(layer_metrics, token_metrics, forced_metrics, output_dir)

    print(f"[Routing drift] Layer-0 exact routing match: {layer_metrics[0]['exact_top2_match_rate']:.6f}")
    print(f"[Routing drift] First clean routing divergence step: {free_summary['first_routing_divergence_step']}")
    print(f"[Routing drift] First token divergence step: {free_summary['first_token_divergence_step']}")
    print(f"[Routing drift] Output directory: {output_dir}")
    for path in plot_paths:
        print(f"[Routing drift] Plot: {path}")


if __name__ == "__main__":
    main()
