"""Shared saved Mixtral HC-SMoE checkpoint restoration."""

import os

import torch
from accelerate import init_empty_weights
from transformers import MixtralConfig, MixtralForCausalLM

from hcsmoe.models.mixtral.utils import (
    bind_shared_experts_from_group_state,
    expand_shared_expert_state_dict,
    load_lora_state_dict,
    load_residual_state_dict,
    validate_shared_expert_topology,
)


def load_compressed_model_for_evaluation(
    model_name: str,
    model_path: str,
    group_state_path: str,
    residual_eval_only: bool,
    residual_path: str | None,
    lora_eval_only: bool = False,
    lora_path: str | None = None,
):
    """Restore aliases, static weights, and optionally residual weights."""
    if residual_eval_only and lora_eval_only:
        raise ValueError("--residual_eval_only and --lora_eval_only are mutually exclusive")
    if not torch.cuda.is_available():
        raise RuntimeError("Compressed Mixtral eval-only mode requires a CUDA device.")
    if not os.path.exists(group_state_path):
        raise FileNotFoundError(f"--eval_only=True requires group mapping: {group_state_path}")
    group_state = torch.load(group_state_path, map_location="cpu")
    state_dict = torch.load(model_path, map_location="cpu")
    meta_keys = [name for name, value in state_dict.items() if isinstance(value, torch.Tensor) and value.is_meta]
    if meta_keys:
        raise RuntimeError(
            f"Checkpoint {model_path} contains {len(meta_keys)} meta tensors (for example {meta_keys[0]}), "
            "which have no saved data. Regenerate the static checkpoint with the current HC-SMoE code."
        )
    expand_shared_expert_state_dict(state_dict, group_state)
    config = MixtralConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = MixtralForCausalLM(config)
    bind_shared_experts_from_group_state(model, group_state)
    load_result = model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    print(f"[HC-SMoE] Static checkpoint loaded: missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")
    if residual_eval_only:
        if not residual_path or not os.path.exists(residual_path):
            raise FileNotFoundError(f"--residual_eval_only=True requires residual checkpoint: {residual_path}")
        payload = torch.load(residual_path, map_location="cpu")
        residual_width = load_residual_state_dict(model, payload, group_state)
        print(f"[Residual] Reloaded width={residual_width} from {residual_path}")
    if lora_eval_only:
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(f"--lora_eval_only=True requires LoRA checkpoint: {lora_path}")
        payload = torch.load(lora_path, map_location="cpu")
        lora_rank, lora_alpha = load_lora_state_dict(model, payload, group_state)
        print(f"[LoRA] Reloaded rank={lora_rank} alpha={lora_alpha:g} from {lora_path}")
    model.to(device=torch.device("cuda:0"), dtype=torch.bfloat16)
    group_counts = validate_shared_expert_topology(model, group_state)
    for name, group_count in group_counts.items():
        print(f"[HC-SMoE] {name}: unique expert count={group_count}, group count={group_count}, shared_identity=True")
    invalid = [name for name, parameter in model.named_parameters() if parameter.is_meta or parameter.device.type != "cuda"]
    if invalid:
        raise AssertionError(f"Evaluation model has CPU/meta parameters: {invalid[:5]}")
    unique_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    vram_gib = torch.cuda.memory_allocated("cuda:0") / (1024 ** 3)
    print(f"[HC-SMoE] Evaluation placement: unique_parameters={unique_parameter_count:,}, vram_allocated_gib={vram_gib:.2f}, cpu_meta_parameters=0")
    return model, group_state
