"""Sequential, retraining-free output/routing-drift-aware Mixtral merging."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from transformers.models.mixtral.modeling_mixtral import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from hcsmoe.merging.drift_aware_grouping import (
    Partition,
    canonical_partition,
    greedy_drift_aware_grouping,
    labels_from_partition,
)

EPSILON = 1e-12


@dataclass
class CalibrationBatch:
    """One intact calibration batch; sequence dimensions are never concatenated."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor


@dataclass
class LayerBatchContext:
    attention_residual: torch.Tensor
    moe_input: torch.Tensor
    router_logits: torch.Tensor
    selected_experts: torch.Tensor
    routing_weights: torch.Tensor
    teacher_hidden: torch.Tensor
    next_teacher_topk: Optional[torch.Tensor]


@dataclass
class ExpertBatchOutput:
    """One merged FFN call using an original expert ID's exact token rows."""

    topk_slots: torch.Tensor
    token_rows: torch.Tensor
    values: torch.Tensor


def _module_device(module: torch.nn.Module) -> torch.device:
    hook = getattr(module, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    for parameter in module.parameters():
        if not parameter.is_meta:
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _weight_tensor(linear: torch.nn.Module, device: torch.device) -> torch.Tensor:
    weight = linear.weight
    if weight.is_meta:
        hook = getattr(linear, "_hf_hook", None)
        weights_map = getattr(hook, "weights_map", None)
        if weights_map is None or "weight" not in weights_map:
            raise RuntimeError("offloaded linear weight is unavailable")
        weight = weights_map["weight"]
    return weight.detach().to(device)


def _attention_mask(model: torch.nn.Module, batch: CalibrationBatch, hidden: torch.Tensor) -> Optional[torch.Tensor]:
    mask = batch.attention_mask.to(hidden.device)
    shape = hidden.shape[:2]
    implementation = getattr(
        model.model,
        "_attn_implementation",
        getattr(model.config, "_attn_implementation", "eager"),
    )
    if implementation == "flash_attention_2":
        return mask if torch.any(mask == 0) else None
    if implementation == "sdpa":
        return _prepare_4d_causal_attention_mask_for_sdpa(
            mask, shape, hidden, 0, sliding_window=model.config.sliding_window
        )
    return _prepare_4d_causal_attention_mask(
        mask, shape, hidden, 0, sliding_window=model.config.sliding_window
    )


@torch.inference_mode()
def collect_calibration_batches(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
) -> tuple[list[CalibrationBatch], int]:
    """Embed calibration IDs while preserving every batch/sequence boundary."""
    embedding = model.get_input_embeddings()
    device = _module_device(embedding)
    batches: list[CalibrationBatch] = []
    token_count = 0
    for raw in dataloader:
        input_ids = raw["input_ids"].to(device)
        attention_mask = raw.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand(
            input_ids.shape[0], -1
        ).clone()
        hidden = embedding(input_ids).detach().cpu()
        batches.append(CalibrationBatch(hidden, attention_mask.detach().cpu(), position_ids))
        token_count += int(input_ids.numel())
    if not batches:
        raise RuntimeError("calibration dataloader yielded no batches")
    return batches, token_count


def _attention_prefix(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    batch: CalibrationBatch,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = _module_device(layer)
    hidden_states = hidden_states.to(device)
    normalized = layer.input_layernorm(hidden_states)
    attention_output = layer.self_attn(
        hidden_states=normalized,
        attention_mask=_attention_mask(model, batch, hidden_states),
        position_ids=batch.position_ids.to(device),
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    )[0]
    attention_residual = hidden_states + attention_output
    return attention_residual, layer.post_attention_layernorm(attention_residual)


def _routing_from_logits(
    moe: torch.nn.Module,
    logits: torch.Tensor,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    weights, selected = torch.topk(probabilities, int(moe.top_k), dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(output_dtype)
    return selected, weights


def _router_topk_after_attention(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    batch: CalibrationBatch,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, moe_input = _attention_prefix(model, layer, batch, hidden_states)
    logits = layer.block_sparse_moe.gate(moe_input.reshape(-1, moe_input.shape[-1]))
    topk = torch.topk(logits, int(layer.block_sparse_moe.top_k), dim=-1).indices
    return logits, topk


def normalized_output_cost(teacher: torch.Tensor, candidate: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    numerator = (candidate.float() - teacher.float()).square().sum(dim=-1)
    denominator = teacher.float().square().sum(dim=-1) + eps
    return (numerator / denominator).mean()


def topk_overlap_cost(teacher_topk: torch.Tensor, candidate_topk: torch.Tensor) -> torch.Tensor:
    if teacher_topk.shape != candidate_topk.shape or teacher_topk.ndim != 2:
        raise AssertionError("top-k tensors must have matching [tokens, K] shapes")
    preserved = (
        teacher_topk.unsqueeze(-1) == candidate_topk.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1).float()
    return (1.0 - preserved / teacher_topk.shape[-1]).mean()


def _weighted_group_weights(
    experts: torch.nn.ModuleList,
    group: tuple[int, ...],
    usage: torch.Tensor,
    device: torch.device,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    denominator = usage[list(group)].to(device).sum() + eps
    merged = []
    for name in ("w1", "w2", "w3"):
        weighted = torch.stack([
            _weight_tensor(getattr(experts[index], name), device) * usage[index].to(device)
            for index in group
        ]).sum(dim=0) / denominator
        merged.append(weighted)
    return merged[0], merged[1], merged[2]


def _expert_forward(hidden: torch.Tensor, weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    w1, w2, w3 = weights
    return F.linear(F.silu(F.linear(hidden, w1)) * F.linear(hidden, w3), w2)


@contextmanager
def _active_decoder_layers(
    model: torch.nn.Module,
    layer_indices: Iterable[int],
    execution_device: Optional[torch.device],
):
    """Temporarily stage complete decoder layers without splitting their modules."""
    indices = sorted(set(layer_indices))
    if execution_device is None:
        yield
        return
    moved: list[torch.nn.Module] = []
    try:
        for index in indices:
            layer = model.model.layers[index]
            if _module_device(layer) != execution_device:
                layer.to(execution_device)
                moved.append(layer)
        yield
    finally:
        for layer in moved:
            layer.to("cpu")
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()


class MixtralPartitionEvaluator:
    """Cache candidate-invariant layer quantities and score complete partitions."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_index: int,
        batches: list[CalibrationBatch],
        eps: float = EPSILON,
    ) -> None:
        self.model = model
        self.layer_index = layer_index
        self.layer = model.model.layers[layer_index]
        self.next_layer = model.model.layers[layer_index + 1] if layer_index + 1 < len(model.model.layers) else None
        self.batches = batches
        self.eps = eps
        self.num_experts = int(model.config.num_local_experts)
        self.contexts: list[LayerBatchContext] = []
        self._expert_output_cache: dict[
            tuple[tuple[int, ...], int, int], ExpertBatchOutput
        ] = {}
        self._computed_groups: set[tuple[int, ...]] = set()
        usage = torch.zeros(self.num_experts, dtype=torch.float32)
        with torch.inference_mode():
            for batch in batches:
                attention_residual, moe_input = _attention_prefix(
                    model, self.layer, batch, batch.hidden_states
                )
                teacher_moe, router_logits = self.layer.block_sparse_moe(moe_input)
                selected, routing_weights = _routing_from_logits(
                    self.layer.block_sparse_moe, router_logits, moe_input.dtype
                )
                usage += torch.bincount(selected.reshape(-1).cpu(), minlength=self.num_experts).float()
                teacher_hidden = attention_residual + teacher_moe
                next_topk = None
                if self.next_layer is not None:
                    _, next_topk_device = _router_topk_after_attention(
                        model, self.next_layer, batch, teacher_hidden
                    )
                    next_topk = next_topk_device.detach().cpu()
                self.contexts.append(LayerBatchContext(
                    attention_residual.detach().cpu(),
                    moe_input.detach().cpu(),
                    router_logits.detach().cpu(),
                    selected.detach().cpu(),
                    routing_weights.detach().cpu(),
                    teacher_hidden.detach().cpu(),
                    next_topk,
                ))
        self.usage = usage

    def _cache_group_expert_outputs(self, group: tuple[int, ...]) -> None:
        """Run the shared merged FFN separately for each original expert ID."""
        group = tuple(sorted(group))
        if group in self._computed_groups:
            return
        device = _module_device(self.layer.block_sparse_moe)
        weights = _weighted_group_weights(
            self.layer.block_sparse_moe.experts, group, self.usage, device, self.eps
        )
        with torch.inference_mode():
            for batch_index, context in enumerate(self.contexts):
                selected = context.selected_experts
                flat_input = context.moe_input.reshape(-1, context.moe_input.shape[-1])
                flat_input_device = flat_input.to(device)
                expert_mask = F.one_hot(
                    selected, num_classes=self.num_experts
                ).permute(2, 1, 0)
                for expert_index in group:
                    topk_slots, token_rows = torch.where(expert_mask[expert_index])
                    token_rows_device = token_rows.to(device)
                    current_state = flat_input_device[None, token_rows_device].reshape(
                        -1, flat_input.shape[-1]
                    )
                    values = _expert_forward(current_state, weights).detach().cpu().to(
                        flat_input.dtype
                    )
                    self._expert_output_cache[(group, expert_index, batch_index)] = (
                        ExpertBatchOutput(
                            topk_slots=topk_slots,
                            token_rows=token_rows,
                            values=values,
                        )
                    )
        self._computed_groups.add(group)

    def candidate_hidden(self, partition: Partition) -> list[torch.Tensor]:
        """Mirror HF Mixtral's expert-ID-order weighting and index_add accumulation."""
        partition = canonical_partition(partition)
        for group in partition:
            self._cache_group_expert_outputs(group)
        expert_to_group = {
            expert_index: group for group in partition for expert_index in group
        }
        device = _module_device(self.layer.block_sparse_moe)
        candidates: list[torch.Tensor] = []
        for batch_index, context in enumerate(self.contexts):
            routing_weights = context.routing_weights
            flat_input = context.moe_input.reshape(-1, context.moe_input.shape[-1])
            final_hidden = torch.zeros(
                flat_input.shape, dtype=flat_input.dtype, device=device
            )
            for expert_index in range(self.num_experts):
                group = expert_to_group[expert_index]
                cached = self._expert_output_cache[(group, expert_index, batch_index)]
                topk_slot = cached.topk_slots
                token_rows = cached.token_rows
                if token_rows.numel() == 0:
                    continue
                current_hidden = cached.values.to(device)
                current_hidden = current_hidden * routing_weights[
                    token_rows, topk_slot, None
                ].to(device)
                final_hidden.index_add_(
                    0,
                    token_rows.to(device),
                    current_hidden.to(flat_input.dtype),
                )
            candidate = context.attention_residual.to(device) + final_hidden.reshape(
                context.moe_input.shape
            )
            candidates.append(candidate.detach().cpu())
        return candidates

    def evaluate(self, partition: Partition) -> dict[str, Any]:
        started = time.perf_counter()
        candidates = self.candidate_hidden(partition)
        output_sum = torch.zeros((), dtype=torch.float64)
        route_sum = torch.zeros((), dtype=torch.float64)
        tokens = 0
        for batch, context, candidate in zip(self.batches, self.contexts, candidates):
            count = candidate.shape[0] * candidate.shape[1]
            output_sum += normalized_output_cost(context.teacher_hidden, candidate, self.eps).double() * count
            if self.next_layer is not None:
                _, candidate_topk = _router_topk_after_attention(
                    self.model, self.next_layer, batch, candidate
                )
                route_sum += topk_overlap_cost(context.next_teacher_topk, candidate_topk.cpu()).double() * count
            tokens += count
        return {
            "c_out": float(output_sum / tokens),
            "c_route": None if self.next_layer is None else float(route_sum / tokens),
            "elapsed_seconds": time.perf_counter() - started,
        }


@torch.inference_mode()
def commit_partition(
    layer: torch.nn.Module,
    partition: Partition,
    usage: torch.Tensor,
    eps: float = EPSILON,
) -> None:
    """Apply HC-SMoE aliasing from weights computed before any in-place mutation."""
    experts = layer.block_sparse_moe.experts
    device = _module_device(layer.block_sparse_moe)
    partition = canonical_partition(partition)
    # Every merged tensor is materialized while all source experts are pristine.
    # Consequently later groups can never observe an earlier representative copy_.
    merged_weights = [
        (group, _weighted_group_weights(experts, group, usage, device, eps))
        for group in partition
    ]
    for group, weights in merged_weights:
        representative = experts[group[0]]
        for name, weight in zip(("w1", "w2", "w3"), weights):
            getattr(representative, name).weight.copy_(weight)
        for expert_index in group[1:]:
            experts[expert_index] = representative


def _same_group_top2_count(context: LayerBatchContext, partition: Partition) -> int:
    labels = torch.tensor(
        labels_from_partition(partition, int(context.router_logits.shape[-1])),
        dtype=torch.long,
    )
    grouped = labels[context.selected_experts]
    return int((grouped[:, 0] == grouped[:, 1]).sum().item())


@torch.inference_mode()
def _committed_output_from_cached_context(
    layer: torch.nn.Module,
    context: LayerBatchContext,
) -> torch.Tensor:
    """Run the real committed MoE on cached prefix tensors without recomputing attention."""
    device = _module_device(layer.block_sparse_moe)
    moe_output, _ = layer.block_sparse_moe(context.moe_input.to(device))
    return (context.attention_residual.to(device) + moe_output).detach().cpu()


def _mismatch_statistics(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | int]:
    left_float = left.float()
    right_float = right.float()
    absolute = (left_float - right_float).abs()
    mismatched = ~torch.isclose(
        left_float, right_float, rtol=rtol, atol=atol, equal_nan=False
    )
    count = int(mismatched.sum().item())
    if count == 0:
        return {
            "mismatched_elements": 0,
            "max_absolute_difference": 0.0,
            "max_relative_difference": 0.0,
        }
    mismatched_absolute = absolute[mismatched]
    mismatched_relative = mismatched_absolute / right_float.abs()[mismatched].clamp_min(
        EPSILON
    )
    return {
        "mismatched_elements": count,
        "max_absolute_difference": float(mismatched_absolute.max().item()),
        "max_relative_difference": float(mismatched_relative.max().item()),
    }


def _assert_post_commit_consistency(
    fast_candidate: torch.Tensor,
    committed_cached: torch.Tensor,
    full_decoder: torch.Tensor,
    *,
    layer_index: int,
    batch_index: int,
    partition: Partition,
    same_group_top2_tokens: int,
    rtol: float = 2e-2,
    atol: float = 2e-2,
) -> None:
    comparisons = {
        "A_vs_B": _mismatch_statistics(
            fast_candidate, committed_cached, rtol=rtol, atol=atol
        ),
        "B_vs_C": _mismatch_statistics(
            committed_cached, full_decoder, rtol=rtol, atol=atol
        ),
        "A_vs_C": _mismatch_statistics(
            fast_candidate, full_decoder, rtol=rtol, atol=atol
        ),
    }
    if all(values["mismatched_elements"] == 0 for values in comparisons.values()):
        return
    formatted = "; ".join(
        f"{name}: mismatched_elements={values['mismatched_elements']}/{fast_candidate.numel()}, "
        f"max_absolute_difference={values['max_absolute_difference']:.10g}, "
        f"max_relative_difference={values['max_relative_difference']:.10g}"
        for name, values in comparisons.items()
    )
    raise AssertionError(
        "post-commit A/B/C consistency failure: "
        f"layer={layer_index}, batch={batch_index}, {formatted}; "
        f"dtype={fast_candidate.dtype}, partition={[list(group) for group in partition]}, "
        f"same_group_top2_tokens={same_group_top2_tokens}, rtol={rtol}, atol={atol}"
    )


@torch.inference_mode()
def forward_decoder_layer(
    model: torch.nn.Module,
    layer_index: int,
    batches: list[CalibrationBatch],
) -> list[CalibrationBatch]:
    layer = model.model.layers[layer_index]
    outputs = []
    for batch in batches:
        device = _module_device(layer)
        hidden = batch.hidden_states.to(device)
        result = layer(
            hidden,
            attention_mask=_attention_mask(model, batch, hidden),
            position_ids=batch.position_ids.to(device),
            use_cache=False,
            output_attentions=False,
            output_router_logits=False,
        )[0]
        outputs.append(CalibrationBatch(result.detach().cpu(), batch.attention_mask, batch.position_ids))
    return outputs


@torch.inference_mode()
def pristine_layer_inputs(
    model: torch.nn.Module,
    initial_batches: list[CalibrationBatch],
    execution_device: Optional[torch.device] = None,
) -> list[list[CalibrationBatch]]:
    states = initial_batches
    inputs = []
    for layer_index in range(len(model.model.layers)):
        inputs.append(states)
        with _active_decoder_layers(model, [layer_index], execution_device):
            states = forward_decoder_layer(model, layer_index, states)
    return inputs


@torch.inference_mode()
def run_sequential_drift_aware_merging(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    num_average_groups: int = 4,
    lambda_route: float = 0.5,
    start_layer: int = 0,
    sequential: bool = True,
    max_layers: Optional[int] = None,
    calibration_seed: int = 42,
    eps: float = EPSILON,
    execution_device: Optional[torch.device | str] = None,
) -> dict[str, Any]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if execution_device is not None:
        execution_device = torch.device(execution_device)
    embedding = model.get_input_embeddings()
    embedding_original_device = _module_device(embedding)
    if execution_device is not None and embedding_original_device != execution_device:
        embedding.to(execution_device)
    initial, token_count = collect_calibration_batches(model, dataloader)
    if execution_device is not None and embedding_original_device != execution_device:
        embedding.to("cpu")
    total_layers = len(model.model.layers)
    end_layer = total_layers if max_layers is None else min(total_layers, start_layer + max_layers)
    if not 0 <= start_layer < total_layers or end_layer <= start_layer:
        raise ValueError("invalid start_layer/max_layers range")
    pristine = pristine_layer_inputs(model, initial, execution_device) if not sequential else None
    states = initial
    for layer_index in range(start_layer):
        with _active_decoder_layers(model, [layer_index], execution_device):
            states = forward_decoder_layer(model, layer_index, states)

    group_state: dict[str, torch.Tensor] = {
        f"model.layers.{index}.block_sparse_moe": torch.arange(model.config.num_local_experts)
        for index in range(total_layers)
    }
    merge_trace: dict[str, Any] = {}
    layer_summaries: list[dict[str, Any]] = []
    for layer_index in range(start_layer, end_layer):
        layer_started = time.perf_counter()
        layer_inputs = states if sequential else pristine[layer_index]
        print(
            f"[Drift-aware] Layer {layer_index}: exact greedy "
            f"{model.config.num_local_experts}->{num_average_groups} grouping"
        )
        active_indices = [layer_index]
        if layer_index + 1 < total_layers:
            active_indices.append(layer_index + 1)
        with _active_decoder_layers(model, active_indices, execution_device):
            evaluator = MixtralPartitionEvaluator(model, layer_index, layer_inputs, eps)
            grouped = greedy_drift_aware_grouping(
                [(expert,) for expert in range(model.config.num_local_experts)],
                num_average_groups,
                evaluator.evaluate,
                lambda_route,
                eps,
                evaluator.usage.tolist(),
            )
            final_partition = grouped["groups"]
            manual_committed = evaluator.candidate_hidden(final_partition)
            commit_partition(model.model.layers[layer_index], final_partition, evaluator.usage, eps)
            committed_cached = [
                _committed_output_from_cached_context(
                    model.model.layers[layer_index], context
                )
                for context in evaluator.contexts
            ]
            actual_committed = forward_decoder_layer(model, layer_index, layer_inputs)
            for batch_index, (fast, cached, actual, context) in enumerate(
                zip(
                    manual_committed,
                    committed_cached,
                    actual_committed,
                    evaluator.contexts,
                )
            ):
                _assert_post_commit_consistency(
                    fast,
                    cached,
                    actual.hidden_states,
                    layer_index=layer_index,
                    batch_index=batch_index,
                    partition=final_partition,
                    same_group_top2_tokens=_same_group_top2_count(
                        context, final_partition
                    ),
                )
            usage_frequencies = evaluator.usage.tolist()
            del evaluator
        if sequential:
            states = actual_committed
        name = f"model.layers.{layer_index}.block_sparse_moe"
        group_state[name] = torch.tensor(
            labels_from_partition(final_partition, int(model.config.num_local_experts)), dtype=torch.long
        )
        for step in grouped["merge_trace"]:
            step["elapsed_seconds"] = sum(row["elapsed_seconds"] for row in step["candidates"])
        merge_trace[name] = grouped["merge_trace"]
        selected_merges = [step["selected_candidate"] for step in grouped["merge_trace"]]
        layer_summaries.append({
            "layer": layer_index,
            "input_token_count": token_count,
            "final_groups": [list(group) for group in final_partition],
            "selected_merges": selected_merges,
            "final_c_out": selected_merges[-1]["c_out"],
            "final_c_route": selected_merges[-1]["c_route"],
            "lambda_route": lambda_route,
            "sequential": sequential,
            "calibration_seed": calibration_seed,
            "usage_frequencies": usage_frequencies,
            "elapsed_seconds": time.perf_counter() - layer_started,
        })
        print(
            f"[Drift-aware] Layer {layer_index}: groups="
            f"{[list(group) for group in final_partition]}, "
            f"C_out={selected_merges[-1]['c_out']:.8f}, "
            f"C_route={selected_merges[-1]['c_route']}"
        )
    return {
        "group_state_dict": group_state,
        "merge_trace": merge_trace,
        "per_layer_summary": layer_summaries,
        "calibration_token_count": token_count,
        "processed_layers": list(range(start_layer, end_layer)),
    }


__all__ = [
    "CalibrationBatch",
    "MixtralPartitionEvaluator",
    "commit_partition",
    "collect_calibration_batches",
    "forward_decoder_layer",
    "normalized_output_cost",
    "pristine_layer_inputs",
    "run_sequential_drift_aware_merging",
    "topk_overlap_cost",
]
