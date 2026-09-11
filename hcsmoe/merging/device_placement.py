"""Automatic A100-aware placement helpers for routing-drift analysis models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from accelerate import cpu_offload, dispatch_model, infer_auto_device_map
from accelerate.utils import convert_file_size_to_int

MIXTRAL_NO_SPLIT_MODULE_CLASSES = ["MixtralDecoderLayer"]
PLACEMENT_CHOICES = ("auto", "cpu-offload", "gpu")
GIB = 1024**3


@dataclass(frozen=True)
class PlacementSettings:
    mode: str
    gpu_name: str
    gpu_total_bytes: int
    gpu_budget_bytes: Optional[int]
    gpu_reserve_bytes: Optional[int]
    cpu_budget_bytes: int

    @property
    def max_memory(self) -> Optional[dict[int | str, int]]:
        if self.mode != "auto" or self.gpu_budget_bytes is None:
            return None
        return {0: self.gpu_budget_bytes, "cpu": self.cpu_budget_bytes}


def _default_reserve_bytes(total_bytes: int) -> int:
    """Leave useful runtime space on both 40GB and 80GB A100 variants."""
    total_gib = total_bytes / GIB
    if total_gib >= 64:
        return 10 * GIB
    if total_gib >= 32:
        return 8 * GIB
    return max(4 * GIB, int(total_bytes * 0.20))


def resolve_placement_settings(
    mode: str,
    gpu_memory: Optional[str],
    cpu_memory: str,
) -> PlacementSettings:
    if mode not in PLACEMENT_CHOICES:
        raise ValueError(f"placement must be one of {PLACEMENT_CHOICES}, got {mode!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("Mixtral routing-drift placement requires a CUDA device")
    properties = torch.cuda.get_device_properties(0)
    total_bytes = int(properties.total_memory)
    cpu_bytes = int(convert_file_size_to_int(cpu_memory))
    if cpu_bytes <= 0:
        raise ValueError("--cpu-memory must be positive")
    budget: Optional[int] = None
    reserve: Optional[int] = None
    if mode == "auto":
        if gpu_memory is None:
            reserve = _default_reserve_bytes(total_bytes)
            budget = total_bytes - reserve
        else:
            budget = int(convert_file_size_to_int(gpu_memory))
            reserve = total_bytes - budget
        if budget <= 0:
            raise ValueError("auto placement GPU budget must be positive")
        if reserve < GIB:
            raise ValueError(
                "auto placement must leave at least 1 GiB outside --gpu-memory for KV cache and activations"
            )
    return PlacementSettings(
        mode=mode,
        gpu_name=properties.name,
        gpu_total_bytes=total_bytes,
        gpu_budget_bytes=budget,
        gpu_reserve_bytes=reserve,
        cpu_budget_bytes=cpu_bytes,
    )


def pretrained_cpu_load_kwargs() -> dict[str, Any]:
    """Load once on CPU so router snapshots can be taken before dispatch."""
    return {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": "cpu"},
    }


def _target_for_parameter(parameter_name: str, device_map: dict[str, Any]) -> Any:
    module_name = parameter_name.rsplit(".", 1)[0] if "." in parameter_name else ""
    while module_name not in device_map and module_name:
        module_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    if module_name not in device_map:
        raise AssertionError(f"device map has no target for parameter {parameter_name}")
    return device_map[module_name]


def _normalized_device(target: Any) -> str:
    if isinstance(target, int):
        return "cuda"
    device = torch.device(target)
    return "cuda" if device.type == "cuda" else device.type


def _assert_whole_decoder_layers(
    model: torch.nn.Module,
    device_map: dict[str, Any],
    decoder_layer_name: str,
) -> None:
    """Guard against expert-level maps that recreate per-token transfer pathology."""
    for layer_index, layer in enumerate(model.model.layers):
        prefix = f"model.layers.{layer_index}."
        targets = {
            _normalized_device(_target_for_parameter(prefix + name, device_map))
            for name, _parameter in layer.named_parameters()
        }
        if targets and len(targets) != 1:
            raise AssertionError(
                f"auto placement split {decoder_layer_name} {layer_index} across {sorted(targets)}"
            )


def place_model_for_analysis(
    model: torch.nn.Module,
    settings: PlacementSettings,
    no_split_module_classes: Optional[list[str]] = None,
    decoder_layer_name: Optional[str] = None,
) -> torch.nn.Module:
    """Place a fully materialized CPU model without changing its weights."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if settings.mode == "cpu-offload":
        cpu_offload(model, execution_device=torch.device("cuda:0"))
        model.hf_device_map = {"": "cpu-offload"}
        return model
    if settings.mode == "gpu":
        try:
            model.to(device=torch.device("cuda:0"), dtype=torch.bfloat16)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "--placement gpu could not fit the complete model on cuda:0; use --placement auto"
            ) from error
        model.hf_device_map = {"": 0}
        return model

    if settings.max_memory is None:
        raise AssertionError("auto placement is missing max_memory")
    tie_weights = getattr(model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    no_split = no_split_module_classes or MIXTRAL_NO_SPLIT_MODULE_CLASSES
    decoder_name = decoder_layer_name or no_split[0]
    device_map = infer_auto_device_map(
        model,
        max_memory=settings.max_memory,
        no_split_module_classes=no_split,
        dtype=torch.bfloat16,
        offload_buffers=True,
        clean_result=True,
        fallback_allocation=True,
    )
    _assert_whole_decoder_layers(model, device_map, decoder_name)
    model = dispatch_model(
        model,
        device_map=device_map,
        main_device=torch.device("cuda:0"),
        offload_buffers=True,
    )
    return model


def _parameter_placement_counts(model: torch.nn.Module) -> dict[str, tuple[int, int]]:
    device_map = getattr(model, "hf_device_map", None)
    totals: dict[str, list[int]] = {}
    if device_map == {"": "cpu-offload"}:
        count = sum(parameter.numel() for parameter in model.parameters())
        return {"cpu/offloaded": (count, len(list(model.parameters())))}
    for name, parameter in model.named_parameters():
        if device_map:
            device = _normalized_device(_target_for_parameter(name, device_map))
        else:
            device = parameter.device.type
        values = totals.setdefault(device, [0, 0])
        values[0] += parameter.numel()
        values[1] += 1
    return {key: (values[0], values[1]) for key, values in totals.items()}


def print_placement_environment(settings: PlacementSettings) -> None:
    total_gib = settings.gpu_total_bytes / GIB
    print(f"[Placement] GPU: {settings.gpu_name} ({total_gib:.2f} GiB)")
    print(f"[Placement] mode={settings.mode}")
    if settings.mode == "auto":
        print(
            f"[Placement] max_memory cuda:0={settings.gpu_budget_bytes / GIB:.2f} GiB, "
            f"reserved_for_runtime={settings.gpu_reserve_bytes / GIB:.2f} GiB, "
            f"cpu={settings.cpu_budget_bytes / GIB:.2f} GiB"
        )


def print_model_placement(model: torch.nn.Module, label: str) -> None:
    counts = _parameter_placement_counts(model)
    summary = ", ".join(
        f"{device}_params={parameters:,} ({tensors} tensors)"
        for device, (parameters, tensors) in sorted(counts.items())
    )
    device_map = getattr(model, "hf_device_map", {})
    module_counts: dict[str, int] = {}
    for target in device_map.values():
        device = "cpu/offloaded" if target == "cpu-offload" else _normalized_device(target)
        module_counts[device] = module_counts.get(device, 0) + 1
    allocated = torch.cuda.memory_allocated(0) / GIB
    reserved = torch.cuda.memory_reserved(0) / GIB
    print(f"[Placement] {label}: {summary}")
    if module_counts:
        print(f"[Placement] {label}: mapped_modules={module_counts}")
    print(f"[Placement] {label}: cuda_allocated={allocated:.2f} GiB, cuda_reserved={reserved:.2f} GiB")


__all__ = [
    "MIXTRAL_NO_SPLIT_MODULE_CLASSES",
    "PLACEMENT_CHOICES",
    "PlacementSettings",
    "place_model_for_analysis",
    "pretrained_cpu_load_kwargs",
    "print_model_placement",
    "print_placement_environment",
    "resolve_placement_settings",
]
