"""Frozen-routing top-2 MoE reconstruction evaluation for Mixtral."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hcsmoe.merging.residual_mixtral import _expert_device, _input_device
from hcsmoe.models.mixtral.utils import group_members_from_labels

TOP_K = 2
EPSILON = 1e-12
TEACHER_SANITY_MAX_RELATIVE_L2 = 1e-3
TEACHER_SANITY_MIN_COSINE = 0.99999


@dataclass
class FrozenMoETokens:
    """Teacher data for complete routed tokens in one MoE layer."""

    hidden_states: torch.Tensor
    expert_indices: torch.Tensor
    routing_weights: torch.Tensor
    teacher_moe_outputs: Optional[torch.Tensor] = None
    original_outputs: Optional[torch.Tensor] = None

    @property
    def token_count(self) -> int:
        return self.hidden_states.shape[0]


def _layer_name(index: int) -> str:
    return f"model.layers.{index}.block_sparse_moe"


def _assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _assert_normalized_weights(weights: torch.Tensor, name: str) -> None:
    _assert_finite(f"{name} routing weights", weights)
    expected = torch.ones(weights.shape[0], dtype=weights.dtype, device=weights.device)
    if not torch.allclose(weights.sum(dim=-1), expected, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"{name}: normalized top-2 routing weights must sum to one")


def _validate_frozen_tokens(data: FrozenMoETokens, name: str) -> None:
    if data.hidden_states.device.type != "cpu":
        raise AssertionError("frozen hidden states must be CPU-resident")
    if data.expert_indices.shape != (data.token_count, TOP_K):
        raise AssertionError(f"{name}: expert indices must be [tokens, {TOP_K}]")
    if data.routing_weights.shape != data.expert_indices.shape:
        raise AssertionError(f"{name}: routing weights must align with expert indices")
    if data.teacher_moe_outputs is not None and data.teacher_moe_outputs.shape != data.hidden_states.shape:
        raise AssertionError(f"{name}: teacher MoE outputs must align with hidden states")
    _assert_finite(f"{name} hidden states", data.hidden_states)
    _assert_normalized_weights(data.routing_weights, name)


class _TokenPairReservoir:
    """Bounded deterministic priority reservoir for complete routed tokens."""

    def __init__(self, limit: int, seed: int) -> None:
        self.limit = limit
        self.seen = 0
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.priorities = torch.empty(0)
        self.data = {}

    def add(self, **candidate_tensors: torch.Tensor) -> None:
        candidate_count = next(iter(candidate_tensors.values())).shape[0]
        if any(value.shape[0] != candidate_count for value in candidate_tensors.values()):
            raise AssertionError("all fields of a routed token pair must have the same length")
        priorities = torch.rand(candidate_count, generator=self.generator)
        self.seen += candidate_count

        if not self.data:
            combined = candidate_tensors
            combined_priorities = priorities
        else:
            combined = {
                name: torch.cat([self.data[name], value.cpu()])
                for name, value in candidate_tensors.items()
            }
            combined_priorities = torch.cat([self.priorities, priorities])
        selected_count = min(self.limit, combined_priorities.numel())
        selected = torch.topk(combined_priorities, selected_count, largest=False).indices
        self.priorities = combined_priorities[selected]
        self.data = {name: value[selected] for name, value in combined.items()}


def _capture_hook(name, reservoir: _TokenPairReservoir):
    """Capture h, top-2 routes, and actual teacher output as one token record."""
    def hook(_module, inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError(f"{name}: expected MoE output with router logits")
        hidden_states = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        teacher_moe_outputs = output[0].detach().reshape_as(hidden_states)
        router_logits = output[1].detach().reshape(hidden_states.shape[0], -1)
        probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float)
        weights, indices = torch.topk(probabilities, TOP_K, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        reservoir.add(
            hidden_states=hidden_states.cpu(),
            expert_indices=indices.cpu(),
            routing_weights=weights.cpu(),
            teacher_moe_outputs=teacher_moe_outputs.cpu(),
        )
    return hook


@torch.no_grad()
def collect_frozen_moe_tokens(teacher, dataloader, token_limit: int, sampling_seed: int):
    """Run all calibration blocks, then sample complete teacher token pairs."""
    if token_limit <= 0:
        raise ValueError("--moe-reconstruction-limit must be positive")
    reservoirs, names, handles = {}, [], []
    for index, layer in enumerate(teacher.model.layers):
        moe = layer.block_sparse_moe
        if moe.top_k != TOP_K:
            raise ValueError(f"Layer {index}: expected top_k={TOP_K}, got {moe.top_k}")
        name = _layer_name(index)
        names.append(name)
        reservoirs[name] = _TokenPairReservoir(token_limit, sampling_seed + index)
        handles.append(moe.register_forward_hook(_capture_hook(name, reservoirs[name])))
    try:
        teacher.eval()
        for batch in tqdm(dataloader, desc="[MoE reconstruction] teacher C4 pairs"):
            inputs = {key: value.to(_input_device(teacher)) for key, value in batch.items() if key != "labels"}
            teacher.model(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for name in names:
        reservoir = reservoirs[name]
        if not reservoir.data:
            raise RuntimeError(f"{name}: calibration produced no tokens")
        data = FrozenMoETokens(**reservoir.data)
        _validate_frozen_tokens(data, name)
        result[name] = data
    print("[MoE reconstruction] Calibration tokens seen per layer: " + ", ".join(
        f"{index}={reservoirs[name].seen}" for index, name in enumerate(names)
    ))
    print("[MoE reconstruction] Selected frozen tokens per layer: " + ", ".join(
        f"{index}={result[name].token_count}" for index, name in enumerate(names)
    ))
    print(f"[MoE reconstruction] Sampling seed: {sampling_seed}")
    return result


@torch.no_grad()
def _run_runtime_module(module, hidden_states: torch.Tensor, hidden_dtype: torch.dtype) -> torch.Tensor:
    """Run a module in the same dtype used by the loaded compressed model."""
    parameter = next(module.parameters())
    if parameter.dtype != hidden_dtype:
        raise AssertionError(
            f"Expected runtime module dtype {hidden_dtype}, found {parameter.dtype}"
        )
    inputs = hidden_states.to(device=_expert_device(module), dtype=hidden_dtype)
    return module(inputs)


@torch.no_grad()
def _compute_outputs(moe, data: FrozenMoETokens, apply_residual: bool):
    """Reconstruct static/residual MoE outputs with runtime-faithful BF16 arithmetic."""
    _validate_frozen_tokens(data, "frozen")
    hidden_dtype = data.hidden_states.dtype
    output_device = _expert_device(moe.experts[0])
    output_shape = (data.token_count, data.hidden_states.shape[-1])
    static = torch.zeros(output_shape, device=output_device, dtype=hidden_dtype)
    residual = torch.zeros_like(static)
    residual_modules = getattr(moe, "residual_experts", {})
    for expert_index in range(moe.num_experts):
        token_indices, route_indices = torch.where(data.expert_indices == expert_index)
        if token_indices.numel() == 0:
            continue
        expert = moe.experts[expert_index]
        expert_values = _run_runtime_module(
            expert, data.hidden_states[token_indices], hidden_dtype
        )
        gates = data.routing_weights[token_indices, route_indices].to(
            device=expert_values.device, dtype=hidden_dtype
        ).unsqueeze(-1)
        static.index_add_(
            0,
            token_indices.to(output_device),
            (expert_values * gates).to(device=output_device, dtype=hidden_dtype),
        )
        values = expert_values
        module = residual_modules[str(expert_index)] if str(expert_index) in residual_modules else None
        if apply_residual and module is not None:
            values = values + _run_runtime_module(
                module, data.hidden_states[token_indices], hidden_dtype
            )
        residual.index_add_(
            0,
            token_indices.to(output_device),
            (values * gates).to(device=output_device, dtype=hidden_dtype),
        )
    static, residual = static.cpu(), residual.cpu()
    _assert_finite("static routed MoE output", static)
    _assert_finite("residual routed MoE output", residual)
    return static, residual


@torch.no_grad()
def materialize_original_outputs(teacher, frozen_by_layer):
    """Validate the manual BF16 path and keep actual teacher output as target."""
    for index, layer in enumerate(teacher.model.layers):
        data = frozen_by_layer[_layer_name(index)]
        if data.teacher_moe_outputs is None:
            raise RuntimeError("teacher MoE forward outputs were not captured")
        manual_output, _ = _compute_outputs(
            layer.block_sparse_moe, data, apply_residual=False
        )
        sanity = _comparison_metrics(manual_output, data.teacher_moe_outputs)
        print(
            "[Teacher sanity] layer={layer} rel_l2={relative_l2:.8e} "
            "cosine={cosine:.8f} max_abs_diff={max_abs_diff:.8e}".format(
                layer=index, **sanity
            )
        )
        if (
            sanity["relative_l2"] > TEACHER_SANITY_MAX_RELATIVE_L2
            or sanity["cosine"] < TEACHER_SANITY_MIN_COSINE
        ):
            raise AssertionError(
                f"Teacher sanity failed at layer {index}: "
                f"rel_l2={sanity['relative_l2']:.3e} "
                f"(max={TEACHER_SANITY_MAX_RELATIVE_L2:.3e}), "
                f"cosine={sanity['cosine']:.8f} "
                f"(min={TEACHER_SANITY_MIN_COSINE:.8f})"
            )
        # The metric target is actual Mixtral teacher MoE output, while the
        # manual result above exists only to validate frozen routing arithmetic.
        data.original_outputs = data.teacher_moe_outputs


def _assert_zero_residual_matches_static(moe, data: FrozenMoETokens, sanity_tokens: int) -> None:
    """Temporarily replace residual forward outputs with zeros without mutation."""
    count = min(sanity_tokens, data.token_count)
    if count == 0:
        return
    subset = FrozenMoETokens(
        hidden_states=data.hidden_states[:count],
        expert_indices=data.expert_indices[:count],
        routing_weights=data.routing_weights[:count],
    )
    static_output, _ = _compute_outputs(moe, subset, apply_residual=False)
    hooks = []
    for residual in getattr(moe, "residual_experts", {}).values():
        hooks.append(residual.register_forward_hook(
            lambda _module, _inputs, output: torch.zeros_like(output)
        ))
    try:
        _, zero_residual_output = _compute_outputs(moe, subset, apply_residual=True)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(static_output, zero_residual_output, rtol=0.0, atol=0.0)


def _metric_totals(target, static, residual):
    """Accumulate relative-L2/cosine terms in FP32 after BF16 MoE forward."""
    target = target.float()
    static = static.float()
    residual = residual.float()
    _assert_finite("original routed MoE output", target)
    return {
        "tokens": target.shape[0],
        "target_norm": target.square().sum().item(),
        "static_error": (static - target).square().sum().item(),
        "residual_error": (residual - target).square().sum().item(),
        "static_dot": (static * target).sum().item(),
        "residual_dot": (residual * target).sum().item(),
        "static_norm": static.square().sum().item(),
        "residual_norm": residual.square().sum().item(),
    }


def _comparison_metrics(reference: torch.Tensor, candidate: torch.Tensor):
    """FP32 diagnostics for numerical agreement between two BF16 outputs."""
    reference = reference.float()
    candidate = candidate.float()
    difference = candidate - reference
    reference_norm = max(reference.square().sum().item(), EPSILON)
    return {
        "relative_l2": math.sqrt(difference.square().sum().item() / reference_norm),
        "cosine": (reference * candidate).sum().item() / max(
            math.sqrt(reference.square().sum().item() * candidate.square().sum().item()),
            EPSILON,
        ),
        "max_abs_diff": difference.abs().max().item(),
    }


def _metrics(totals):
    target_norm = max(totals["target_norm"], EPSILON)
    cosine = lambda dot, norm: dot / max(math.sqrt(norm * totals["target_norm"]), EPSILON)
    return {
        "tokens": int(totals["tokens"]),
        "static_relative_l2": math.sqrt(totals["static_error"] / target_norm),
        "static_cosine": cosine(totals["static_dot"], totals["static_norm"]),
        "residual_relative_l2": math.sqrt(totals["residual_error"] / target_norm),
        "residual_cosine": cosine(totals["residual_dot"], totals["residual_norm"]),
    }


@torch.no_grad()
def evaluate_frozen_moe_reconstruction(model, group_state, frozen_by_layer, use_residual, sanity_tokens=8):
    """Measure static/residual reconstruction against y_orig, layer by layer."""
    aggregate = defaultdict(float)
    layers = {}
    for index, layer in enumerate(model.model.layers):
        name, data = _layer_name(index), frozen_by_layer[_layer_name(index)]
        for members in group_members_from_labels(group_state[name]).values():
            if len(members) == 1 and str(members[0]) in getattr(layer.block_sparse_moe, "residual_experts", {}):
                raise AssertionError(f"{name}: singleton expert {members[0]} has a residual")
        if data.original_outputs is None:
            raise RuntimeError(f"{name}: missing original output")
        _assert_zero_residual_matches_static(layer.block_sparse_moe, data, sanity_tokens)
        static, residual = _compute_outputs(layer.block_sparse_moe, data, apply_residual=use_residual)
        if not use_residual:
            torch.testing.assert_close(static, residual, rtol=0.0, atol=0.0)
        totals = _metric_totals(data.original_outputs, static, residual)
        layers[str(index)] = _metrics(totals)
        for key, value in totals.items():
            aggregate[key] += value
    result = _metrics(aggregate)
    result["residual_enabled"] = bool(use_residual)
    result["relative_l2_improvement_percent"] = 100 * (result["static_relative_l2"] - result["residual_relative_l2"]) / max(result["static_relative_l2"], EPSILON)
    return {"layers": layers, "aggregate": result}


def format_moe_reconstruction_summary(metrics):
    lines = []
    for index, values in metrics["layers"].items():
        lines.extend([f"Layer {index} (tokens={values['tokens']})", f"Static   rel_L2={values['static_relative_l2']:.6f} cosine={values['static_cosine']:.6f}", f"Residual rel_L2={values['residual_relative_l2']:.6f} cosine={values['residual_cosine']:.6f}", ""])
    values = metrics["aggregate"]
    lines.extend(["Aggregate", f"Static   rel_L2={values['static_relative_l2']:.6f} cosine={values['static_cosine']:.6f}", f"Residual rel_L2={values['residual_relative_l2']:.6f} cosine={values['residual_cosine']:.6f}", "Improvement:", f"  relative_L2: {values['relative_l2_improvement_percent']:.2f}%"])
    return "\n".join(lines)
