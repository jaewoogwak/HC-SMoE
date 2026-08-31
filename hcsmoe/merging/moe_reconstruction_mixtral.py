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
    teacher_expert_outputs: Optional[torch.Tensor] = None
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
def _compute_outputs(moe, data: FrozenMoETokens, apply_residual: bool, return_expert_outputs: bool = False):
    """Reconstruct static/residual MoE outputs with runtime-faithful BF16 arithmetic."""
    _validate_frozen_tokens(data, "frozen")
    hidden_dtype = data.hidden_states.dtype
    output_device = _expert_device(moe.experts[0])
    output_shape = (data.token_count, data.hidden_states.shape[-1])
    static = torch.zeros(output_shape, device=output_device, dtype=hidden_dtype)
    residual = torch.zeros_like(static)
    static_expert_outputs = (
        torch.zeros((data.token_count, TOP_K, data.hidden_states.shape[-1]), dtype=hidden_dtype)
        if return_expert_outputs else None
    )
    residual_expert_outputs = torch.zeros_like(static_expert_outputs) if return_expert_outputs else None
    residual_modules = getattr(moe, "residual_experts", {})
    for expert_index in range(moe.num_experts):
        token_indices, route_indices = torch.where(data.expert_indices == expert_index)
        if token_indices.numel() == 0:
            continue
        expert = moe.experts[expert_index]
        expert_values = _run_runtime_module(
            expert, data.hidden_states[token_indices], hidden_dtype
        )
        if return_expert_outputs:
            static_expert_outputs[token_indices, route_indices] = expert_values.cpu()
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
        if return_expert_outputs:
            residual_expert_outputs[token_indices, route_indices] = values.cpu()
        residual.index_add_(
            0,
            token_indices.to(output_device),
            (values * gates).to(device=output_device, dtype=hidden_dtype),
        )
    static, residual = static.cpu(), residual.cpu()
    _assert_finite("static routed MoE output", static)
    _assert_finite("residual routed MoE output", residual)
    if return_expert_outputs:
        return static, residual, static_expert_outputs, residual_expert_outputs
    return static, residual


@torch.no_grad()
def materialize_original_outputs(teacher, frozen_by_layer):
    """Report manual-vs-teacher agreement without gating the A experiment.

    This is diagnostic only: the A-experiment target remains the captured actual
    teacher_moe_outputs.  Manual reconstruction validates frozen-routing logic.
    """
    sanity_results = []
    for index, layer in enumerate(teacher.model.layers):
        data = frozen_by_layer[_layer_name(index)]
        if data.teacher_moe_outputs is None:
            raise RuntimeError("teacher MoE forward outputs were not captured")
        manual_output, _, manual_expert_outputs, _ = _compute_outputs(
            layer.block_sparse_moe,
            data,
            apply_residual=False,
            return_expert_outputs=True,
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
            print(
                "[Teacher sanity][WARNING] layer={layer} exceeds diagnostic "
                "threshold (rel_l2={relative_l2:.3e}, cosine={cosine:.8f})".format(
                    layer=index,
                    **sanity,
                )
            )
        sanity_results.append((index, sanity))
        # The metric target is actual Mixtral teacher MoE output; the manual
        # result above is only a frozen-routing implementation diagnostic.
        data.original_outputs = data.teacher_moe_outputs
        data.teacher_expert_outputs = manual_expert_outputs
    _print_teacher_sanity_summary(sanity_results)


def _print_teacher_sanity_summary(sanity_results) -> None:
    """Print numerical-agreement extrema across all teacher MoE layers."""
    if not sanity_results:
        return
    max_relative_l2 = max(sanity_results, key=lambda item: item[1]["relative_l2"])
    min_cosine = min(sanity_results, key=lambda item: item[1]["cosine"])
    max_absolute_difference = max(sanity_results, key=lambda item: item[1]["max_abs_diff"])
    print("[Teacher sanity] summary:")
    print(
        "  max_rel_l2={:.8e} at layer={}".format(
            max_relative_l2[1]["relative_l2"],
            max_relative_l2[0],
        )
    )
    print(
        "  min_cosine={:.8f} at layer={}".format(
            min_cosine[1]["cosine"],
            min_cosine[0],
        )
    )
    print(
        "  max_abs_diff={:.8e} at layer={}".format(
            max_absolute_difference[1]["max_abs_diff"],
            max_absolute_difference[0],
        )
    )


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


def _decomposition_totals(data, static_experts, residual_experts, token_mask):
    """Aggregate top-2 self/cross error terms for one token subset in FP32."""
    if data.teacher_expert_outputs is None:
        raise RuntimeError("missing frozen original expert outputs for decomposition")
    original = data.teacher_expert_outputs[token_mask].float()
    static = static_experts[token_mask].float()
    residual = residual_experts[token_mask].float()
    gates = data.routing_weights[token_mask].float()
    target = data.original_outputs[token_mask].float()

    def terms(approximation):
        error = approximation - original
        self_term = (gates.square() * error.square().sum(dim=-1)).sum(dim=-1)
        cross_term = 2 * gates[:, 0] * gates[:, 1] * (error[:, 0] * error[:, 1]).sum(dim=-1)
        total_term = self_term + cross_term
        direct_total = (gates.unsqueeze(-1) * error).sum(dim=1).square().sum(dim=-1)
        if not torch.allclose(total_term, direct_total, rtol=1e-4, atol=1e-5):
            raise AssertionError("top-2 decomposition does not match direct routed error")
        return self_term.sum().item(), cross_term.sum().item(), total_term.sum().item()

    static_self, static_cross, static_total = terms(static)
    residual_self, residual_cross, residual_total = terms(residual)
    residual_value = residual - static
    return {
        "tokens": int(token_mask.sum().item()),
        "target_energy": target.square().sum().item(),
        "static_self_raw": static_self,
        "static_cross_raw": static_cross,
        "static_total_raw": static_total,
        "residual_self_raw": residual_self,
        "residual_cross_raw": residual_cross,
        "residual_total_raw": residual_total,
        "residual_energy": residual_value.square().sum().item(),
        "original_expert_energy": original.square().sum().item(),
        "static_expert_energy": static.square().sum().item(),
    }


def _normalized_decomposition(totals):
    denominator = max(totals["target_energy"], EPSILON)
    return {
        "tokens": int(totals["tokens"]),
        "static_self": totals["static_self_raw"] / denominator,
        "static_cross": totals["static_cross_raw"] / denominator,
        "static_total": totals["static_total_raw"] / denominator,
        "residual_self": totals["residual_self_raw"] / denominator,
        "residual_cross": totals["residual_cross_raw"] / denominator,
        "residual_total": totals["residual_total_raw"] / denominator,
    }


def _add_totals(destination, source):
    for key, value in source.items():
        destination[key] += value


def _residual_magnitude(totals):
    return {
        "relative_to_original": math.sqrt(
            totals["residual_energy"] / max(totals["original_expert_energy"], EPSILON)
        ),
        "relative_to_static": math.sqrt(
            totals["residual_energy"] / max(totals["static_expert_energy"], EPSILON)
        ),
    }


def _expert_diagnostics(data, static_experts, residual_experts, num_experts):
    """Compute route-occurrence diagnostics without additional expert forwards."""
    if data.teacher_expert_outputs is None:
        raise RuntimeError("missing frozen original expert outputs for diagnostics")
    diagnostics = {}
    static_weighted_error = 0.0
    residual_weighted_error = 0.0

    for expert_index in range(num_experts):
        token_indices, route_indices = torch.where(data.expert_indices == expert_index)
        route_count = int(token_indices.numel())
        if route_count == 0:
            diagnostics[str(expert_index)] = {"routes": 0}
            continue

        original = data.teacher_expert_outputs[token_indices, route_indices].float()
        static = static_experts[token_indices, route_indices].float()
        residual = residual_experts[token_indices, route_indices].float()
        residual_value = residual - static
        gates = data.routing_weights[token_indices, route_indices].float()

        original_energy = original.square().sum().item()
        static_energy = static.square().sum().item()
        static_error = (static - original).square().sum().item()
        residual_error = (residual - original).square().sum().item()
        residual_energy = residual_value.square().sum().item()
        weighted_original_energy = (gates.square().unsqueeze(-1) * original.square()).sum().item()
        weighted_static_error = (gates.square().unsqueeze(-1) * (static - original).square()).sum().item()
        weighted_residual_error = (gates.square().unsqueeze(-1) * (residual - original).square()).sum().item()
        static_relative_l2 = math.sqrt(static_error / max(original_energy, EPSILON))
        residual_relative_l2 = math.sqrt(residual_error / max(original_energy, EPSILON))

        diagnostics[str(expert_index)] = {
            "routes": route_count,
            "static_relative_l2": static_relative_l2,
            "residual_relative_l2": residual_relative_l2,
            "relative_l2_improvement_percent": (
                100.0 * (static_relative_l2 - residual_relative_l2)
                / max(static_relative_l2, EPSILON)
            ),
            "residual_relative_to_original": math.sqrt(
                residual_energy / max(original_energy, EPSILON)
            ),
            "residual_relative_to_static": math.sqrt(
                residual_energy / max(static_energy, EPSILON)
            ),
            "weighted_static_relative_l2": math.sqrt(
                weighted_static_error / max(weighted_original_energy, EPSILON)
            ),
            "weighted_residual_relative_l2": math.sqrt(
                weighted_residual_error / max(weighted_original_energy, EPSILON)
            ),
        }
        static_weighted_error += weighted_static_error
        residual_weighted_error += weighted_residual_error
    return diagnostics, static_weighted_error, residual_weighted_error


def _assert_expert_self_matches_decomposition(static_error, residual_error, decomposition):
    """Self terms equal the sum of gate-squared routed expert errors."""
    if not math.isclose(static_error, decomposition["static_self_raw"], rel_tol=1e-5, abs_tol=1e-5):
        raise AssertionError("static expert diagnostics do not sum to decomposition self term")
    if not math.isclose(residual_error, decomposition["residual_self_raw"], rel_tol=1e-5, abs_tol=1e-5):
        raise AssertionError("residual expert diagnostics do not sum to decomposition self term")


def _expert_diagnostic_summary(diagnostics):
    active = [(int(index), value) for index, value in diagnostics.items() if value["routes"] > 0]
    if not active:
        return {"improved_experts": 0, "degraded_experts": 0}
    worst = max(active, key=lambda item: item[1]["residual_relative_l2"])
    largest_original = max(active, key=lambda item: item[1]["residual_relative_to_original"])
    largest_static = max(active, key=lambda item: item[1]["residual_relative_to_static"])
    return {
        "worst_residual_relative_l2_expert": worst[0],
        "worst_residual_relative_l2": worst[1]["residual_relative_l2"],
        "largest_residual_relative_to_original_expert": largest_original[0],
        "largest_residual_relative_to_original": largest_original[1]["residual_relative_to_original"],
        "largest_residual_relative_to_static_expert": largest_static[0],
        "largest_residual_relative_to_static": largest_static[1]["residual_relative_to_static"],
        "improved_experts": sum(value["relative_l2_improvement_percent"] > 0 for _, value in active),
        "degraded_experts": sum(value["relative_l2_improvement_percent"] < 0 for _, value in active),
    }


@torch.no_grad()
def evaluate_frozen_moe_reconstruction(model, group_state, frozen_by_layer, use_residual, sanity_tokens=8):
    """Measure static/residual reconstruction against y_orig, layer by layer."""
    aggregate = defaultdict(float)
    aggregate_decomposition = {
        "all": defaultdict(float),
        "same_group": defaultdict(float),
        "different_group": defaultdict(float),
    }
    layers = {}
    for index, layer in enumerate(model.model.layers):
        name, data = _layer_name(index), frozen_by_layer[_layer_name(index)]
        for members in group_members_from_labels(group_state[name]).values():
            if len(members) == 1 and str(members[0]) in getattr(layer.block_sparse_moe, "residual_experts", {}):
                raise AssertionError(f"{name}: singleton expert {members[0]} has a residual")
        if data.original_outputs is None:
            raise RuntimeError(f"{name}: missing original output")
        _assert_zero_residual_matches_static(layer.block_sparse_moe, data, sanity_tokens)
        static, residual, static_experts, residual_experts = _compute_outputs(
            layer.block_sparse_moe,
            data,
            apply_residual=use_residual,
            return_expert_outputs=True,
        )
        if not use_residual:
            torch.testing.assert_close(static, residual, rtol=0.0, atol=0.0)
        totals = _metric_totals(data.original_outputs, static, residual)
        layer_metrics = _metrics(totals)
        group_ids = group_state[name][data.expert_indices]
        subset_masks = {
            "all": torch.ones(data.token_count, dtype=torch.bool),
            "same_group": group_ids[:, 0] == group_ids[:, 1],
            "different_group": group_ids[:, 0] != group_ids[:, 1],
        }
        decomposition = {}
        decomposition_totals = {}
        for subset_name, mask in subset_masks.items():
            subset_totals = _decomposition_totals(
                data,
                static_experts,
                residual_experts,
                mask,
            )
            decomposition_totals[subset_name] = subset_totals
            decomposition[subset_name] = _normalized_decomposition(subset_totals)
            _add_totals(aggregate_decomposition[subset_name], subset_totals)
        expert_diagnostics, static_self, residual_self = _expert_diagnostics(
            data,
            static_experts,
            residual_experts,
            layer.block_sparse_moe.num_experts,
        )
        _assert_expert_self_matches_decomposition(
            static_self,
            residual_self,
            decomposition_totals["all"],
        )
        layer_metrics["decomposition"] = decomposition
        layer_metrics["residual_magnitude"] = _residual_magnitude(decomposition_totals["all"])
        layer_metrics["expert_diagnostics"] = expert_diagnostics
        layer_metrics["expert_diagnostic_summary"] = _expert_diagnostic_summary(expert_diagnostics)
        layers[str(index)] = layer_metrics
        for key, value in totals.items():
            aggregate[key] += value
    result = _metrics(aggregate)
    result["residual_enabled"] = bool(use_residual)
    result["relative_l2_improvement_percent"] = 100 * (result["static_relative_l2"] - result["residual_relative_l2"]) / max(result["static_relative_l2"], EPSILON)
    result["decomposition"] = {
        subset_name: _normalized_decomposition(totals)
        for subset_name, totals in aggregate_decomposition.items()
    }
    result["residual_magnitude"] = _residual_magnitude(aggregate_decomposition["all"])
    return {"layers": layers, "aggregate": result}


def _format_expert_value(value, precision=".4f"):
    return "N/A" if value is None else format(value, precision)


def _append_expert_diagnostics(lines, layer_index, values):
    """Append all experts, including zero-route experts, as a compact table."""
    lines.append(f"Expert diagnostics [Layer {layer_index}]")
    lines.append("Expert Routes StaticRelL2 ResidualRelL2 Improve% R/E R/M WStatic WResidual")
    for expert_index in sorted(values["expert_diagnostics"], key=int):
        diagnostic = values["expert_diagnostics"][expert_index]
        if diagnostic["routes"] == 0:
            lines.append(f"{expert_index:<6} {0:<6} N/A N/A N/A N/A N/A N/A N/A")
            continue
        lines.append(
            "{expert:<6} {routes:<6} {static:<11} {residual:<13} {improvement:<9} "
            "{relative_original:<8} {relative_static:<8} {weighted_static:<8} {weighted_residual:<8}".format(
                expert=expert_index,
                routes=diagnostic["routes"],
                static=_format_expert_value(diagnostic["static_relative_l2"]),
                residual=_format_expert_value(diagnostic["residual_relative_l2"]),
                improvement=_format_expert_value(diagnostic["relative_l2_improvement_percent"], "+.1f"),
                relative_original=_format_expert_value(diagnostic["residual_relative_to_original"]),
                relative_static=_format_expert_value(diagnostic["residual_relative_to_static"]),
                weighted_static=_format_expert_value(diagnostic["weighted_static_relative_l2"]),
                weighted_residual=_format_expert_value(diagnostic["weighted_residual_relative_l2"]),
            )
        )
    summary = values["expert_diagnostic_summary"]
    lines.extend([
        f"Expert diagnostic summary [Layer {layer_index}]",
        "  worst residual rel_L2: expert {expert} = {value:.6f}".format(
            expert=summary["worst_residual_relative_l2_expert"],
            value=summary["worst_residual_relative_l2"],
        ),
        "  largest R/E:           expert {expert} = {value:.6f}".format(
            expert=summary["largest_residual_relative_to_original_expert"],
            value=summary["largest_residual_relative_to_original"],
        ),
        "  largest R/M:           expert {expert} = {value:.6f}".format(
            expert=summary["largest_residual_relative_to_static_expert"],
            value=summary["largest_residual_relative_to_static"],
        ),
        f"  improved experts:      {summary['improved_experts']}/8",
        f"  degraded experts:      {summary['degraded_experts']}/8",
        "",
    ])


def format_moe_reconstruction_summary(metrics):
    lines = []
    for index, values in metrics["layers"].items():
        lines.extend([f"Layer {index} (tokens={values['tokens']})", f"Static   rel_L2={values['static_relative_l2']:.6f} cosine={values['static_cosine']:.6f}", f"Residual rel_L2={values['residual_relative_l2']:.6f} cosine={values['residual_cosine']:.6f}", ""])
        for subset_name, title in (
            ("all", "All"),
            ("same_group", "Same-group"),
            ("different_group", "Different-group"),
        ):
            decomposition = values["decomposition"][subset_name]
            lines.extend([
                f"Error decomposition [{title}] (tokens={decomposition['tokens']})",
                "  Static   self={static_self:.6e} cross={static_cross:.6e} total={static_total:.6e}".format(**decomposition),
                "  Residual self={residual_self:.6e} cross={residual_cross:.6e} total={residual_total:.6e}".format(**decomposition),
            ])
        magnitude = values["residual_magnitude"]
        lines.extend([
            "Residual magnitude:",
            f"  ||R|| / ||E|| = {magnitude['relative_to_original']:.6e}",
            f"  ||R|| / ||M|| = {magnitude['relative_to_static']:.6e}",
            "",
        ])
        _append_expert_diagnostics(lines, index, values)
    values = metrics["aggregate"]
    lines.extend(["Aggregate", f"Static   rel_L2={values['static_relative_l2']:.6f} cosine={values['static_cosine']:.6f}", f"Residual rel_L2={values['residual_relative_l2']:.6f} cosine={values['residual_cosine']:.6f}", "Improvement:", f"  relative_L2: {values['relative_l2_improvement_percent']:.2f}%"])
    for subset_name, title in (("all", "All"), ("same_group", "Same-group"), ("different_group", "Different-group")):
        decomposition = values["decomposition"][subset_name]
        lines.extend([
            f"Aggregate decomposition [{title}] (tokens={decomposition['tokens']})",
            "  Static   self={static_self:.6e} cross={static_cross:.6e} total={static_total:.6e}".format(**decomposition),
            "  Residual self={residual_self:.6e} cross={residual_cross:.6e} total={residual_total:.6e}".format(**decomposition),
        ])
    return "\n".join(lines)
