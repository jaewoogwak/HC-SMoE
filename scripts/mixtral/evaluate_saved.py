#!/venv/mixtral/bin/python3.11
"""Evaluate a merged Mixtral checkpoint without running grouping or merging.

Example:
    /venv/mixtral/bin/python3.11 scripts/mixtral/evaluate_saved.py
"""

import argparse
import sys
from pathlib import Path

import torch
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoTokenizer, MixtralForCausalLM

# Make direct execution from scripts/mixtral independent of PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from hcsmoe.evaluation import evaluate_fewshot


DEFAULT_TASKS = (
    "winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="results/model.pth",
        help="Path to the state dict saved by merging-mixtral.py.",
    )
    parser.add_argument(
        "--model-name",
        default="mistralai/Mixtral-8x7B-v0.1",
        help="The original base model used to create the checkpoint.",
    )
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS,
        help="Comma-separated lm-eval task names.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument(
        "--result-path",
        default="results/result_mixtral_test.txt",
        help="Where to write evaluation tables.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to --result-path instead of replacing it.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Restore and validate the checkpoint, then exit before evaluation.",
    )
    return parser.parse_args()


def load_state_dict(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def storage_signature(tensor: torch.Tensor) -> tuple:
    """Identify tensors which were saved as views of the same parameter."""
    return (
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def restore_shared_experts(model: MixtralForCausalLM, state_dict: dict) -> None:
    """Recreate the expert-module aliases made by frequency merging."""
    for layer_idx, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts"
        representatives = {}
        for expert_idx in range(len(layer.block_sparse_moe.experts)):
            signature = tuple(
                storage_signature(state_dict[f"{prefix}.{expert_idx}.{weight}.weight"])
                for weight in ("w1", "w2", "w3")
            )
            if signature in representatives:
                layer.block_sparse_moe.experts[expert_idx] = (
                    layer.block_sparse_moe.experts[representatives[signature]]
                )
            else:
                representatives[signature] = expert_idx


def materialize_rotary_buffers(model: MixtralForCausalLM, device: str) -> None:
    """Initialize the non-persistent RoPE buffers omitted from state_dict."""
    for layer in model.model.layers:
        rotary = layer.self_attn.rotary_emb
        inv_freq = 1.0 / (
            rotary.base
            ** (torch.arange(0, rotary.dim, 2, dtype=torch.int64).float() / rotary.dim)
        )
        rotary.register_buffer("inv_freq", inv_freq, persistent=False)
        rotary._set_cos_sin_cache(
            seq_len=rotary.max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )


def find_meta_tensors(model: MixtralForCausalLM) -> list[str]:
    names = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.is_meta:
            names.append(f"parameter: {name}")
    for module_name, module in model.named_modules(remove_duplicate=False):
        for name, buffer in module._buffers.items():
            if buffer is not None and buffer.is_meta:
                names.append(f"buffer: {module_name}.{name}".strip("."))
    return names


def place_model(model: MixtralForCausalLM) -> MixtralForCausalLM:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this evaluation.")
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
        **{f"model.layers.{i}": "cpu" for i in range(len(model.model.layers))},
    }
    print("Dispatching decoder layers with CPU offload and cuda:0 execution")
    return dispatch_model(model, device_map=device_map, offload_buffers=True)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    result_path = Path(args.result_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Merged checkpoint not found: {checkpoint_path}")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise ValueError("At least one task is required.")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        result_path.write_text("")

    # The merged checkpoint is complete. Only tokenizer/config metadata comes
    # from Hugging Face; no original model-weight shard is downloaded.
    print(f"Loading tokenizer and config only: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Restoring merged checkpoint: {checkpoint_path}")
    checkpoint = load_state_dict(checkpoint_path)
    config = AutoConfig.from_pretrained(args.model_name)
    with init_empty_weights():
        model = MixtralForCausalLM(config)
    materialize_rotary_buffers(model, device="cpu")
    restore_shared_experts(model, checkpoint)
    # Keep CPU tensors and their shared expert aliases intact for offloading.
    model.load_state_dict(checkpoint, strict=True, assign=True)
    del checkpoint
    model = place_model(model)
    if args.check_only:
        print("Checkpoint restored and CPU offload hooks attached.")
        return
    model.eval()

    for task in tasks:
        print(f"\n[Evaluation] {task}")
        evaluate_fewshot(
            model=model,
            tokenizer=tokenizer,
            task=task,
            num_fewshot=args.num_fewshot,
            eval_batch_size=args.eval_batch_size,
            output_path=str(result_path),
            log=True,
        )


if __name__ == "__main__":
    main()
