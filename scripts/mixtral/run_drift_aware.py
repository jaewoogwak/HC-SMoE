#!/usr/bin/env python3
"""Sequential output- and next-routing-drift-aware Mixtral expert grouping."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, MixtralForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcsmoe.evaluation import evaluate_fewshot, get_calib_dataloder
from hcsmoe.merging.sequential_drift_aware_mixtral import run_sequential_drift_aware_merging
from hcsmoe.merging.sequential_drift_mixtral import (
    assert_router_weights_unchanged,
    router_weight_snapshot,
)
from hcsmoe.models.mixtral.utils import expand_shared_expert_state_dict


DEFAULT_TASKS = "winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte"


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", "--model-name", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--num_average_groups", "--num-average-groups", type=int, default=4)
    parser.add_argument("--n_sentences", "--n-sentences", type=int, default=8)
    parser.add_argument("--block_size", "--block-size", type=int, default=2048)
    parser.add_argument("--train_batch_size", "--train-batch-size", type=int, default=1)
    parser.add_argument("--calib_seed", "--calib-seed", type=int, default=42)
    parser.add_argument("--lambda_route", "--lambda-route", type=float, default=0.5)
    parser.add_argument("--start_layer", "--start-layer", type=int, default=0)
    parser.add_argument("--sequential", type=_parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--output_path", "--output-path", required=True)
    parser.add_argument("--result_path", "--result-path", default=None)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", type=int, default=16)
    parser.add_argument("--num_fewshot", "--num-fewshot", type=int, default=0)
    parser.add_argument("--task", default=DEFAULT_TASKS)
    parser.add_argument(
        "--max_layers",
        "--max-layers",
        type=int,
        default=None,
        help="Validation-only layer limit; omit for the full model.",
    )
    parser.add_argument(
        "--execution_device",
        "--execution-device",
        default="cuda:0",
        help="Device used for the current and next complete decoder layers.",
    )
    parser.add_argument(
        "--skip_model_save",
        "--skip-model-save",
        action="store_true",
        help="Validation-only: save JSON/group mapping but omit the large model.pth.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("num_average_groups", "n_sentences", "block_size", "train_batch_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if not 0.0 <= args.lambda_route <= 1.0:
        raise ValueError("--lambda_route must be in [0, 1]")
    if args.num_average_groups != 4:
        raise ValueError("this exact first-version Mixtral experiment requires 8 -> 4 grouping")
    if args.max_layers is not None and args.max_layers <= 0:
        raise ValueError("--max_layers must be positive")
    if args.skip_model_save and args.result_path:
        raise ValueError("--skip_model_save cannot be combined with --result_path")
    if torch.device(args.execution_device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA execution was requested but CUDA is unavailable")


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _save_checkpoint(
    model: MixtralForCausalLM,
    group_state: dict[str, torch.Tensor],
    output_dir: Path,
) -> None:
    non_cpu = [
        name for name, parameter in model.named_parameters()
        if parameter.is_meta or parameter.device.type != "cpu"
    ]
    if non_cpu:
        raise AssertionError(f"checkpoint model must be fully materialized on CPU: {non_cpu[:5]}")
    state_dict = expand_shared_expert_state_dict(model.state_dict(), group_state)
    torch.save(state_dict, output_dir / "model.pth")


def _evaluate_if_requested(
    model: MixtralForCausalLM,
    tokenizer: Any,
    args: argparse.Namespace,
) -> None:
    if not args.result_path:
        return
    model.to(torch.device(args.execution_device), dtype=torch.bfloat16)
    tasks = args.task.split(",") if isinstance(args.task, str) else list(args.task)
    for task in tasks:
        evaluate_fewshot(
            model,
            tokenizer=tokenizer,
            task=task,
            num_fewshot=args.num_fewshot,
            eval_batch_size=args.eval_batch_size,
            output_path=args.result_path,
            log=True,
        )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.calib_seed)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("[Drift-aware] Loading pristine Mixtral on CPU; decoder layers are staged on demand.")
    model = MixtralForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    if int(model.config.num_local_experts) != 8:
        raise ValueError(
            f"this experiment requires Mixtral with 8 local experts, got {model.config.num_local_experts}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    routers = router_weight_snapshot(model)

    dataloader = get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=args.block_size,
        n_blocks_for_stat=args.n_sentences,
        batch_size=args.train_batch_size,
        num_workers=4,
        seed=args.calib_seed,
    )
    artifacts = run_sequential_drift_aware_merging(
        model=model,
        dataloader=dataloader,
        num_average_groups=args.num_average_groups,
        lambda_route=args.lambda_route,
        start_layer=args.start_layer,
        sequential=args.sequential,
        max_layers=args.max_layers,
        calibration_seed=args.calib_seed,
        execution_device=args.execution_device,
    )
    assert_router_weights_unchanged(routers, model)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("drift-aware calibration must remain eval-only and gradient-free")
    print("[Drift-aware] Router weights are bitwise unchanged.")
    print(f"[Drift-aware] Exact calibration tokens processed: {artifacts['calibration_token_count']}")

    group_state = artifacts["group_state_dict"]
    torch.save(group_state, output_dir / "group_state_dict.pt")
    metadata = {
        "model_name": args.model_name,
        "method": "sequential_output_and_next_routing_drift_aware",
        "num_average_groups": args.num_average_groups,
        "n_sentences": args.n_sentences,
        "block_size": args.block_size,
        "train_batch_size": args.train_batch_size,
        "calibration_token_count": artifacts["calibration_token_count"],
        "calibration_seed": args.calib_seed,
        "lambda_route": args.lambda_route,
        "start_layer": args.start_layer,
        "sequential": args.sequential,
        "processed_layers": artifacts["processed_layers"],
        "router_weights_unchanged": True,
        "group_mapping_file": "group_state_dict.pt",
        "lambda_zero_interpretation": "output-drift-only greedy; not vanilla HC-SMoE",
    }
    _json_dump(output_dir / "group_mapping_metadata.json", metadata)
    _json_dump(output_dir / "merge_trace.json", artifacts["merge_trace"])
    _json_dump(output_dir / "per_layer_summary.json", artifacts["per_layer_summary"])
    if not args.skip_model_save:
        _save_checkpoint(model, group_state, output_dir)
    print(f"[Drift-aware] Saved artifacts in {output_dir}")

    gc.collect()
    torch.cuda.empty_cache()
    _evaluate_if_requested(model, tokenizer, args)


if __name__ == "__main__":
    main()
