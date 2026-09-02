"""Memory-conscious expert pairwise score helpers.

The functions in this module are deliberately independent from HC-SMoE grouping:
they collect calibration statistics only and never alter expert assignments.
"""

import os
import tempfile
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

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


def minmax_offdiagonal_score(
    matrix: torch.Tensor,
    smaller_is_better: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Min-max normalize an expert matrix without using its diagonal.

    A constant off-diagonal matrix is intentionally neutral: every distinct
    pair receives 0.5 rather than producing a NaN or an arbitrary preference.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    values = matrix.detach().to(device="cpu", dtype=torch.float32)
    if not torch.allclose(values, values.T):
        raise ValueError("matrix must be symmetric")
    num_experts = values.shape[0]
    mask = ~torch.eye(num_experts, dtype=torch.bool)
    off_diagonal = values[mask]
    if off_diagonal.numel() == 0 or not torch.isfinite(off_diagonal).all():
        raise ValueError("matrix must contain finite off-diagonal values")
    minimum = off_diagonal.min()
    maximum = off_diagonal.max()
    score = torch.zeros_like(values, dtype=torch.float32)
    if torch.isclose(maximum, minimum):
        score[mask] = 0.5
    elif smaller_is_better:
        score = (maximum - values) / (maximum - minimum)
    else:
        score = (values - minimum) / (maximum - minimum)
    score = (score + score.T) / 2
    score.fill_diagonal_(0)
    return score, {"min": float(minimum), "max": float(maximum)}


def build_hybrid_score_matrices(
    output_distance: torch.Tensor,
    routing_rate: torch.Tensor,
    alpha: float,
) -> Dict[str, object]:
    """Build CPU FP32 output/routing similarities and their hybrid distance."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    output_score, output_range = minmax_offdiagonal_score(output_distance, smaller_is_better=True)
    routing_score, routing_range = minmax_offdiagonal_score(routing_rate, smaller_is_better=False)
    if output_score.shape != routing_score.shape:
        raise ValueError("output_distance and routing_rate must have the same shape")
    hybrid_score = alpha * output_score + (1.0 - alpha) * routing_score
    hybrid_score = (hybrid_score + hybrid_score.T) / 2
    hybrid_score.fill_diagonal_(1)
    hybrid_distance = (1.0 - hybrid_score).to(dtype=torch.float32, device="cpu")
    hybrid_distance = (hybrid_distance + hybrid_distance.T) / 2
    hybrid_distance.fill_diagonal_(0)
    if not torch.isfinite(hybrid_distance).all():
        raise ValueError("hybrid score computation produced NaN or Inf")
    return {
        "output_score": output_score,
        "routing_score": routing_score,
        "hybrid_score": hybrid_score,
        "hybrid_distance": hybrid_distance,
        "normalization": {
            "output_min": output_range["min"],
            "output_max": output_range["max"],
            "routing_min": routing_range["min"],
            "routing_max": routing_range["max"],
        },
    }


def canonical_groups(labels: Sequence[int] | torch.Tensor) -> list[list[int]]:
    """Return a label-ID-invariant, deterministic group representation."""
    labels = [int(label) for label in labels]
    groups: Dict[int, list[int]] = {}
    for expert, label in enumerate(labels):
        groups.setdefault(label, []).append(expert)
    return sorted((sorted(members) for members in groups.values()), key=lambda members: members[0])


def partitions_equal(left: Sequence[int] | torch.Tensor, right: Sequence[int] | torch.Tensor) -> bool:
    return canonical_groups(left) == canonical_groups(right)


def changed_expert_count(reference: Sequence[int] | torch.Tensor, candidate: Sequence[int] | torch.Tensor) -> int:
    """Count experts whose group member set differs, ignoring label IDs."""
    if len(reference) != len(candidate):
        raise ValueError("partitions must have the same number of experts")
    def memberships(labels):
        groups = canonical_groups(labels)
        result = {}
        for group in groups:
            members = tuple(group)
            for expert in group:
                result[expert] = members
        return result
    left, right = memberships(reference), memberships(candidate)
    return sum(left[expert] != right[expert] for expert in left)


def grouping_metrics(
    labels: Sequence[int] | torch.Tensor,
    output_distance: torch.Tensor,
    routing_rate: torch.Tensor,
) -> Dict[str, Optional[float]]:
    """Calculate Mixtral top-2 locality and raw output-distance proxies."""
    groups = canonical_groups(labels)
    routing_rate = routing_rate.detach().to(device="cpu", dtype=torch.float32)
    output_distance = output_distance.detach().to(device="cpu", dtype=torch.float32)
    same_group_routing_rate = 0.0
    distances = []
    for group in groups:
        for offset, i in enumerate(group):
            for j in group[offset + 1:]:
                same_group_routing_rate += float(routing_rate[i, j])
                distances.append(float(output_distance[i, j]))
    return {
        "same_group_routing_rate": same_group_routing_rate,
        "expected_unique_groups_per_token": 2.0 - same_group_routing_rate,
        "mean_intragroup_output_distance": sum(distances) / len(distances) if distances else None,
    }


def validate_pairwise_score_payload(
    payload: Mapping[str, object],
    expected_metadata: Mapping[str, object],
    expected_layer_keys: Iterable[str],
    num_experts: int,
    hidden_size: int,
) -> None:
    """Validate the cache metadata and the Mixtral pairwise tensor contract."""
    errors = []
    if payload.get("version") != 1:
        errors.append(f"cached version: {payload.get('version')}; requested version: 1")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        errors.append("cache metadata is missing")
        metadata = {}
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            errors.append(f"cached {key}: {metadata.get(key)!r}; requested {key}: {value!r}")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        errors.append("cache layers are missing")
        layers = {}
    expected_layer_keys = list(expected_layer_keys)
    missing_layers = [key for key in expected_layer_keys if key not in layers]
    if missing_layers:
        errors.append(f"missing layer keys: {missing_layers}")
    if not errors:
        try:
            validate_pairwise_scores({key: layers[key] for key in expected_layer_keys})
            for key in expected_layer_keys:
                scores = layers[key]
                if scores["output_fingerprint"].shape != (num_experts, hidden_size):
                    raise ValueError(
                        f"{key} output_fingerprint shape {tuple(scores['output_fingerprint'].shape)} "
                        f"does not match {(num_experts, hidden_size)}"
                    )
                if scores["output_distance"].shape != (num_experts, num_experts):
                    raise ValueError(f"{key} output_distance has the wrong shape")
                if scores["routing_rate"].shape != (num_experts, num_experts):
                    raise ValueError(f"{key} routing_rate has the wrong shape")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    if errors:
        raise ValueError(
            "Pairwise score cache is incompatible with the current run:\n- "
            + "\n- ".join(errors)
            + "\nUse --recompute_pairwise_scores=True to regenerate it."
        )


def load_or_compute_pairwise_scores(
    path: str,
    model,
    grouper,
    dataloader,
    metadata: Mapping[str, object],
    chunk_size: int,
    recompute: bool = False,
) -> Dict[str, object]:
    """Load a valid cache or atomically replace it with freshly computed scores."""
    expected_layers = [f"model.layers.{idx}.block_sparse_moe" for idx in grouper.sparse_layer_indices]
    expected_metadata = {
        key: metadata[key] for key in (
            "model_name", "dataset", "num_blocks", "block_size",
            "num_calibration_tokens", "num_experts", "top_k", "seed",
        )
    }
    if os.path.exists(path) and not recompute:
        print(f"[Hybrid] Loading pairwise score cache: {path}")
        payload = torch.load(path, map_location="cpu")
        validate_pairwise_score_payload(
            payload, expected_metadata, expected_layers, grouper.num_experts, grouper.d_model
        )
        cached_chunk_size = payload["metadata"].get("chunk_size")
        if cached_chunk_size != chunk_size:
            print(f"[Hybrid] Cache chunk_size={cached_chunk_size}; requested chunk_size={chunk_size}; reusing cache")
        print("[Hybrid] Pairwise score cache validation passed")
        return payload

    if os.path.exists(path):
        print(f"[Hybrid] Recomputing pairwise score cache: {path}")
    else:
        print(f"[Hybrid] Pairwise score cache not found: {path}")
    print("[Hybrid] Computing Mixtral C4 pairwise scores")
    layers = grouper.compute_pairwise_score_matrices(model, dataloader, chunk_size=chunk_size)
    payload = {"version": 1, "metadata": dict(metadata), "layers": layers}
    validate_pairwise_score_payload(
        payload, expected_metadata, expected_layers, grouper.num_experts, grouper.d_model
    )
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".pairwise_scores_", suffix=".pt", dir=directory, delete=False) as handle:
            temporary_path = handle.name
        save_pairwise_scores(temporary_path, metadata, layers)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    print(f"[Hybrid] Saved pairwise score cache: {path}")
    return payload


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
