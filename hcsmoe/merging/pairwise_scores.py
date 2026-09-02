"""Memory-conscious expert pairwise score helpers.

The functions in this module are deliberately independent from HC-SMoE grouping:
they collect calibration statistics only and never alter expert assignments.
"""

import os
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


def module_execution_device(module: nn.Module) -> torch.device:
    """Return the device used by an Accelerate-dispatched module.

    ``device_map='auto'`` can leave parameters on ``meta`` while Accelerate
    materializes them through a hook, so parameter.device alone is not enough.
    """
    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Cannot infer an execution device for an all-meta module without an Accelerate hook.")


def accumulate_corouting(
    routing_count: torch.Tensor,
    usage_count: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate token-level top-k expert co-activation counts in-place.

    ``routing_count`` stores only distinct expert pairs; expert activation
    totals are kept separately in ``usage_count``.
    """
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [num_tokens, num_experts]")
    num_experts = router_logits.shape[-1]
    if routing_count.shape != (num_experts, num_experts):
        raise ValueError("routing_count shape does not match router_logits")
    if usage_count.shape != (num_experts,):
        raise ValueError("usage_count shape does not match router_logits")
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

    selected = torch.topk(router_logits, k=top_k, dim=-1).indices.to("cpu")
    usage_count += torch.bincount(selected.reshape(-1), minlength=num_experts).to(torch.int64)
    for left in range(top_k):
        for right in range(left + 1, top_k):
            pair = selected[:, (left, right)]
            flat_indices = pair[:, 0] * num_experts + pair[:, 1]
            counts = torch.bincount(flat_indices, minlength=num_experts * num_experts)
            counts = counts.reshape(num_experts, num_experts).to(torch.int64)
            routing_count += counts + counts.T
    routing_count.fill_diagonal_(0)
    return routing_count, usage_count


def compute_output_fingerprint(
    expert: nn.Module,
    layer_input: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute an expert's FP32 mean output without retaining token outputs."""
    if layer_input.ndim != 2 or layer_input.shape[0] == 0:
        raise ValueError("layer_input must be a non-empty [num_tokens, hidden_size] tensor")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    device = module_execution_device(expert)
    output_sum = torch.zeros(layer_input.shape[-1], dtype=torch.float32, device="cpu")
    num_tokens = 0
    with torch.no_grad():
        for start in range(0, layer_input.shape[0], chunk_size):
            inputs = layer_input[start:start + chunk_size].to(device, non_blocking=True)
            outputs = expert(inputs)
            output_sum += outputs.detach().float().sum(dim=0).cpu()
            num_tokens += outputs.shape[0]
            del inputs, outputs
    return output_sum / num_tokens


def build_output_score_matrices(output_fingerprint: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Build CPU FP32 Euclidean-distance and cosine matrices from fingerprints."""
    if output_fingerprint.ndim != 2:
        raise ValueError("output_fingerprint must have shape [num_experts, hidden_size]")
    fingerprint = output_fingerprint.detach().to(device="cpu", dtype=torch.float32)
    output_distance = torch.cdist(fingerprint, fingerprint, p=2)
    output_distance = (output_distance + output_distance.T) / 2
    output_distance.fill_diagonal_(0)

    normalized = F.normalize(fingerprint, p=2, dim=-1, eps=torch.finfo(torch.float32).eps)
    output_cosine = normalized @ normalized.T
    output_cosine = (output_cosine + output_cosine.T) / 2
    output_cosine.fill_diagonal_(1)
    return {
        "output_fingerprint": fingerprint,
        "output_distance": output_distance,
        "output_cosine": output_cosine,
    }


def validate_pairwise_scores(layers: Mapping[str, Mapping[str, object]]) -> None:
    """Raise ValueError when a score payload violates its storage contract."""
    for layer_name, scores in layers.items():
        required = {
            "output_fingerprint", "output_distance", "output_cosine", "routing_count",
            "routing_rate", "usage_count", "num_tokens",
        }
        missing = required.difference(scores)
        if missing:
            raise ValueError(f"{layer_name} is missing score fields: {sorted(missing)}")
        fingerprint = scores["output_fingerprint"]
        distance = scores["output_distance"]
        cosine = scores["output_cosine"]
        routing_count = scores["routing_count"]
        routing_rate = scores["routing_rate"]
        usage_count = scores["usage_count"]
        num_tokens = scores["num_tokens"]
        num_experts = fingerprint.shape[0]
        matrix_shape = (num_experts, num_experts)
        if fingerprint.ndim != 2 or any(matrix.shape != matrix_shape for matrix in (distance, cosine, routing_count, routing_rate)):
            raise ValueError(f"{layer_name} has inconsistent expert matrix shapes")
        if usage_count.shape != (num_experts,) or not isinstance(num_tokens, int) or num_tokens <= 0:
            raise ValueError(f"{layer_name} has invalid usage_count or num_tokens")
        if fingerprint.device.type != "cpu" or distance.device.type != "cpu" or cosine.device.type != "cpu":
            raise ValueError(f"{layer_name} output scores must be CPU tensors")
        if fingerprint.dtype != torch.float32 or distance.dtype != torch.float32 or cosine.dtype != torch.float32:
            raise ValueError(f"{layer_name} output scores must be FP32")
        if routing_count.dtype != torch.int64 or usage_count.dtype != torch.int64:
            raise ValueError(f"{layer_name} routing and usage counts must be int64")
        if not all(torch.isfinite(tensor).all() for tensor in (fingerprint, distance, cosine, routing_rate)):
            raise ValueError(f"{layer_name} contains NaN or Inf")
        if not torch.allclose(distance, distance.T) or not torch.allclose(cosine, cosine.T):
            raise ValueError(f"{layer_name} output matrices must be symmetric")
        if not torch.equal(routing_count, routing_count.T) or torch.any(routing_count.diag() != 0):
            raise ValueError(f"{layer_name} routing_count must be symmetric with zero diagonal")
        if not torch.allclose(distance.diag(), torch.zeros(num_experts)):
            raise ValueError(f"{layer_name} output_distance diagonal must be zero")
        if not torch.allclose(cosine.diag(), torch.ones(num_experts)):
            raise ValueError(f"{layer_name} output_cosine diagonal must be one")
        if torch.any(routing_count > torch.minimum(usage_count[:, None], usage_count[None, :])):
            raise ValueError(f"{layer_name} co-routing count exceeds usage count")


def save_pairwise_scores(
    output_path: str,
    metadata: Mapping[str, object],
    layers: Mapping[str, Mapping[str, object]],
) -> None:
    """Validate and save one complete, self-describing score artifact."""
    validate_pairwise_scores(layers)
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save({"version": 1, "metadata": dict(metadata), "layers": dict(layers)}, output_path)
