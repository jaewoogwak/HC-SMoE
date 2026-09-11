"""Restore a saved vanilla Qwen HC-SMoE checkpoint without rerunning merging."""

import os

import torch
from accelerate import init_empty_weights
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM


def _layer_index(name: str) -> int:
    prefix = "model.layers."
    suffix = ".mlp"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"unexpected Qwen group-state key: {name}")
    return int(name[len(prefix) : -len(suffix)])


def bind_shared_experts_from_group_state(
    model: Qwen2MoeForCausalLM,
    group_state: dict[str, torch.Tensor],
) -> None:
    """Recreate the ModuleList aliases produced by Qwen's vanilla merge."""
    expected_layers = {
        f"model.layers.{index}.mlp"
        for index, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "experts")
    }
    if set(group_state) != expected_layers:
        missing = sorted(expected_layers - set(group_state))
        extra = sorted(set(group_state) - expected_layers)
        raise AssertionError(f"Qwen group-state layer mismatch: missing={missing[:3]}, extra={extra[:3]}")

    for name, labels in group_state.items():
        layer_index = _layer_index(name)
        mlp = model.model.layers[layer_index].mlp
        labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        if labels.ndim != 1 or labels.numel() != len(mlp.experts):
            raise AssertionError(
                f"{name}: expected {len(mlp.experts)} expert labels, got shape {tuple(labels.shape)}"
            )
        representatives: dict[int, int] = {}
        for expert_index, label_tensor in enumerate(labels):
            label = int(label_tensor)
            representative = representatives.setdefault(label, expert_index)
            if expert_index != representative:
                mlp.experts[expert_index] = mlp.experts[representative]


def validate_shared_expert_topology(
    model: Qwen2MoeForCausalLM,
    group_state: dict[str, torch.Tensor],
    expected_num_groups: int | None = None,
) -> dict[str, int]:
    group_counts = {}
    expected_experts = int(model.config.num_experts)
    for name, labels in group_state.items():
        layer_index = _layer_index(name)
        mlp = model.model.layers[layer_index].mlp
        labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        group_count = int(labels.unique().numel())
        if expected_num_groups is not None and group_count != expected_num_groups:
            raise AssertionError(f"{name}: expected {expected_num_groups} groups, found {group_count}")
        if len(mlp.experts) != expected_experts or mlp.gate.out_features != expected_experts:
            raise AssertionError(f"{name}: sparse router/expert dimension is no longer {expected_experts}")
        for label in labels.unique():
            members = torch.where(labels == label)[0].tolist()
            if len({id(mlp.experts[index]) for index in members}) != 1:
                raise AssertionError(f"{name}: group {int(label)} does not share one expert module")
        if len({id(expert) for expert in mlp.experts}) != group_count:
            raise AssertionError(f"{name}: expert alias count does not match group count")
        group_counts[name] = group_count
    return group_counts


def load_compressed_model_for_analysis(
    model_name: str,
    model_path: str,
    group_state_path: str,
    expected_num_groups: int = 30,
) -> tuple[Qwen2MoeForCausalLM, dict[str, torch.Tensor]]:
    """Load Qwen on CPU with the exact expert aliases used by HC-SMoE."""
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen routing-drift analysis requires a CUDA execution device")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"saved Qwen HC-SMoE model not found: {model_path}")
    if not os.path.isfile(group_state_path):
        raise FileNotFoundError(f"saved Qwen HC-SMoE group mapping not found: {group_state_path}")

    group_state = torch.load(group_state_path, map_location="cpu")
    state_dict = torch.load(model_path, map_location="cpu")
    meta_keys = [
        name for name, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.is_meta
    ]
    if meta_keys:
        raise RuntimeError(
            f"checkpoint contains {len(meta_keys)} meta tensors (for example {meta_keys[0]}); regenerate it"
        )

    config = Qwen2MoeConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = Qwen2MoeForCausalLM(config)
    bind_shared_experts_from_group_state(model, group_state)
    load_result = model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError(
            f"Qwen checkpoint mismatch: missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    model.to(device=torch.device("cpu"), dtype=torch.bfloat16)
    group_counts = validate_shared_expert_topology(model, group_state, expected_num_groups)
    invalid = [
        name for name, parameter in model.named_parameters()
        if parameter.is_meta or parameter.device.type != "cpu"
    ]
    if invalid:
        raise AssertionError(f"Qwen analysis model has parameters outside CPU: {invalid[:5]}")
    print(
        "[Qwen routing drift] Restored HC-SMoE topology: "
        f"layers={len(group_counts)}, groups_per_layer={expected_num_groups}, "
        f"unique_parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    return model, group_state


__all__ = [
    "bind_shared_experts_from_group_state",
    "load_compressed_model_for_analysis",
    "validate_shared_expert_topology",
]
