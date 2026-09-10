"""Static shared-expert Mixtral checkpoint restoration for eval-only runs."""

import os

import torch
from accelerate import init_empty_weights
from transformers import MixtralConfig, MixtralForCausalLM

from hcsmoe.models.mixtral.utils import (
    bind_shared_experts_from_group_state,
    expand_shared_expert_state_dict,
    validate_shared_expert_topology,
)


def load_compressed_model_for_evaluation(model_name: str, model_path: str, group_state_path: str):
    """Restore static expert aliases before assigning the saved checkpoint."""
    if not torch.cuda.is_available():
        raise RuntimeError("Compressed Mixtral eval-only mode requires CUDA")
    if not os.path.exists(group_state_path):
        raise FileNotFoundError(f"--eval_only=True requires group mapping: {group_state_path}")
    group_state = torch.load(group_state_path, map_location="cpu")
    state_dict = torch.load(model_path, map_location="cpu")
    meta_keys = [name for name, value in state_dict.items() if isinstance(value, torch.Tensor) and value.is_meta]
    if meta_keys:
        raise RuntimeError(f"Checkpoint contains meta tensors (for example {meta_keys[0]})")
    expand_shared_expert_state_dict(state_dict, group_state)
    with init_empty_weights():
        model = MixtralForCausalLM(MixtralConfig.from_pretrained(model_name))
    bind_shared_experts_from_group_state(model, group_state)
    load_result = model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    print(f"[HC-SMoE] Static checkpoint loaded: missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}")
    model.to(device=torch.device("cuda:0"), dtype=torch.bfloat16)
    counts = validate_shared_expert_topology(model, group_state)
    for name, count in counts.items():
        print(f"[HC-SMoE] {name}: group count={count}, shared_identity=True")
    invalid = [name for name, parameter in model.named_parameters() if parameter.is_meta or parameter.device.type != "cuda"]
    if invalid:
        raise AssertionError(f"Evaluation model has CPU/meta parameters: {invalid[:5]}")
    return model, group_state
