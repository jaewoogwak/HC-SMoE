"""Sequential hidden-state and sparse-router drift diagnostics for Qwen MoE.

Each model performs an independent end-to-end forward. Only input or forced
token IDs are shared; hidden states and routing decisions are never injected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from tqdm import tqdm

from hcsmoe.merging.sequential_drift_mixtral import GlobalSamplePlan, make_global_sample_plan

EPSILON = 1e-12


def _forward_input_device(model: torch.nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()
    hook = getattr(embedding, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    for parameter in embedding.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    raise RuntimeError("CPU-offloaded Qwen forward requires a CUDA execution device")


def _sparse_moe_layers(model: torch.nn.Module) -> list[tuple[int, torch.nn.Module]]:
    sparse = []
    for layer_index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "experts") and hasattr(mlp, "shared_expert"):
            sparse.append((layer_index, mlp))
    if not sparse:
        raise AssertionError("Qwen model contains no sparse MoE layers")
    return sparse


def freeze_for_analysis(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def router_weight_snapshot(model: torch.nn.Module) -> dict[int, torch.Tensor]:
    return {
        layer_index: mlp.gate.weight.detach().cpu().clone()
        for layer_index, mlp in _sparse_moe_layers(model)
    }


def assert_router_weights_unchanged(reference: dict[int, torch.Tensor], model: torch.nn.Module) -> None:
    sparse = dict(_sparse_moe_layers(model))
    if set(reference) != set(sparse):
        raise AssertionError("original and merged Qwen models have different sparse-layer indices")
    expected_experts = int(model.config.num_experts)
    for layer_index, expected in reference.items():
        actual = sparse[layer_index].gate.weight.detach().cpu()
        if actual.shape[0] != expected_experts:
            raise AssertionError(
                f"layer {layer_index}: router width {actual.shape[0]} != original expert count {expected_experts}"
            )
        if not torch.equal(expected, actual):
            raise AssertionError(f"layer {layer_index}: merged checkpoint changed the original router weights")


@dataclass
class RoutingDriftTrace:
    hidden: torch.Tensor  # [sparse_layers, tokens, hidden]
    topk: torch.Tensor  # [sparse_layers, tokens, K]
    margin: torch.Tensor  # [sparse_layers, tokens]
    layer_indices: tuple[int, ...]
    top_k: int
    num_experts: int
    input_id_batches: list[torch.Tensor] = field(default_factory=list)
    token_ids: Optional[torch.Tensor] = None


class _FixedRoutingCapture:
    def __init__(
        self,
        plan: GlobalSamplePlan,
        layer_indices: tuple[int, ...],
        top_k: int,
        num_experts: int,
    ) -> None:
        if not 1 <= top_k < num_experts:
            raise ValueError(f"top-k boundary requires 1 <= K < experts, got K={top_k}, experts={num_experts}")
        self.plan = plan
        self.layer_indices = layer_indices
        self.top_k = top_k
        self.num_experts = num_experts
        count = len(layer_indices)
        self.hidden: list[Optional[torch.Tensor]] = [None] * count
        self.topk: list[Optional[torch.Tensor]] = [None] * count
        self.margin: list[Optional[torch.Tensor]] = [None] * count
        self.hidden_seen = [torch.zeros(plan.token_count, dtype=torch.bool) for _ in range(count)]
        self.router_seen = [torch.zeros(plan.token_count, dtype=torch.bool) for _ in range(count)]
        self.slots = torch.empty(0, dtype=torch.long)
        self.rows = torch.empty(0, dtype=torch.long)
        self.batch_tokens: Optional[int] = None

    def begin_batch(self, offset: int, batch_tokens: int) -> None:
        self.batch_tokens = batch_tokens
        self.slots, self.rows = self.plan.selection(offset, batch_tokens)

    def end_batch(self) -> None:
        self.batch_tokens = None

    def _sample(self, value: torch.Tensor, label: str) -> torch.Tensor:
        if self.batch_tokens is None:
            raise RuntimeError(f"{label}: hook ran outside an active batch")
        flattened = value.detach().reshape(-1, value.shape[-1])
        if len(flattened) != self.batch_tokens:
            raise AssertionError(f"{label}: expected {self.batch_tokens} tokens, got {len(flattened)}")
        return flattened.index_select(0, self.rows.to(flattened.device)).cpu()

    @staticmethod
    def _allocate(current: Optional[torch.Tensor], plan: GlobalSamplePlan, values: torch.Tensor) -> torch.Tensor:
        if current is None:
            return torch.empty((plan.token_count, *values.shape[1:]), dtype=values.dtype)
        return current

    def moe_input_hook(self, trace_index: int, layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            values = self._sample(inputs[0], f"layer {layer_index} MoE input")
            self.hidden[trace_index] = self._allocate(self.hidden[trace_index], self.plan, values)
            self.hidden[trace_index][self.slots] = values
            self.hidden_seen[trace_index][self.slots] = True

        return hook

    def router_hook(self, trace_index: int, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer_index}: expected Qwen MoE hidden output and router logits")
            logits = self._sample(output[1], f"layer {layer_index} router logits").float()
            if logits.shape[-1] != self.num_experts:
                raise AssertionError(
                    f"layer {layer_index}: router width {logits.shape[-1]} != {self.num_experts} sparse experts"
                )
            boundary = torch.topk(logits, self.top_k + 1, dim=-1)
            ids = boundary.indices[:, : self.top_k]
            margins = boundary.values[:, self.top_k - 1] - boundary.values[:, self.top_k]
            self.topk[trace_index] = self._allocate(self.topk[trace_index], self.plan, ids)
            self.margin[trace_index] = self._allocate(self.margin[trace_index], self.plan, margins)
            self.topk[trace_index][self.slots] = ids
            self.margin[trace_index][self.slots] = margins
            self.router_seen[trace_index][self.slots] = True

        return hook

    def register(self, sparse_layers: list[tuple[int, torch.nn.Module]]) -> list[Any]:
        handles = []
        for trace_index, (layer_index, mlp) in enumerate(sparse_layers):
            handles.append(mlp.register_forward_pre_hook(self.moe_input_hook(trace_index, layer_index)))
            handles.append(mlp.register_forward_hook(self.router_hook(trace_index, layer_index)))
        return handles

    def finalize(self, input_id_batches: list[torch.Tensor]) -> RoutingDriftTrace:
        for trace_index, layer_index in enumerate(self.layer_indices):
            if not self.hidden_seen[trace_index].all() or not self.router_seen[trace_index].all():
                raise AssertionError(f"layer {layer_index}: fixed-input trace is incomplete")
        return RoutingDriftTrace(
            hidden=torch.stack([value for value in self.hidden if value is not None]),
            topk=torch.stack([value for value in self.topk if value is not None]),
            margin=torch.stack([value for value in self.margin if value is not None]),
            layer_indices=self.layer_indices,
            top_k=self.top_k,
            num_experts=self.num_experts,
            input_id_batches=input_id_batches,
        )


@torch.inference_mode()
def collect_fixed_routing_trace(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    sample_plan: GlobalSamplePlan,
    expected_input_batches: Optional[list[torch.Tensor]] = None,
    description: str = "[Qwen routing drift] fixed input",
) -> RoutingDriftTrace:
    freeze_for_analysis(model)
    sparse_layers = _sparse_moe_layers(model)
    layer_indices = tuple(index for index, _ in sparse_layers)
    capture = _FixedRoutingCapture(
        sample_plan,
        layer_indices,
        int(model.config.num_experts_per_tok),
        int(model.config.num_experts),
    )
    handles = capture.register(sparse_layers)
    input_batches: list[torch.Tensor] = []
    offset = 0
    batch_count = 0
    try:
        for batch_index, batch in enumerate(tqdm(dataloader, desc=description)):
            input_ids = batch["input_ids"].detach().cpu()
            if expected_input_batches is None:
                input_batches.append(input_ids.clone())
            elif batch_index >= len(expected_input_batches) or not torch.equal(
                input_ids, expected_input_batches[batch_index]
            ):
                raise AssertionError("original and merged models must receive identical fixed input_ids")
            batch_tokens = input_ids.numel()
            capture.begin_batch(offset, batch_tokens)
            inputs = {
                key: value.to(_forward_input_device(model))
                for key, value in batch.items()
                if key != "labels"
            }
            model.model(**inputs, use_cache=False, return_dict=True)
            capture.end_batch()
            offset += batch_tokens
            batch_count += 1
    finally:
        for handle in handles:
            handle.remove()
    if expected_input_batches is not None and len(expected_input_batches) != batch_count:
        raise AssertionError("fixed-input dataloader yielded a different batch count")
    if offset != sample_plan.total_tokens:
        raise AssertionError(f"fixed-input stream changed: expected {sample_plan.total_tokens} tokens, saw {offset}")
    return capture.finalize(input_batches if expected_input_batches is None else expected_input_batches)


def _validate_trace_pair(original: RoutingDriftTrace, merged: RoutingDriftTrace) -> None:
    if original.layer_indices != merged.layer_indices:
        raise AssertionError("original and merged traces have different sparse layers")
    if original.top_k != merged.top_k or original.num_experts != merged.num_experts:
        raise AssertionError("original and merged traces have different routing configurations")
    if original.topk.shape != merged.topk.shape or original.topk.shape[-1] != original.top_k:
        raise AssertionError("routing traces must have matching [..., K] shapes")


def _route_comparison(
    original: RoutingDriftTrace, merged: RoutingDriftTrace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_trace_pair(original, merged)
    original_set = original.topk.sort(dim=-1).values
    merged_set = merged.topk.sort(dim=-1).values
    exact = (original_set == merged_set).all(dim=-1)
    preserved = (merged.topk.unsqueeze(-1) == original.topk.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    changed = original.top_k - preserved
    return exact, preserved, changed


def _optional_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    return float(values[mask].mean()) if mask.any() else None


def compare_fixed_routing_traces(
    original: RoutingDriftTrace,
    merged: RoutingDriftTrace,
    sample_positions: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    if len(original.input_id_batches) != len(merged.input_id_batches) or any(
        not torch.equal(left, right) for left, right in zip(original.input_id_batches, merged.input_id_batches)
    ):
        raise AssertionError("original and merged fixed input IDs differ")
    flattened_input_ids = torch.cat([batch.reshape(-1) for batch in original.input_id_batches])
    if sample_positions.numel() and int(sample_positions.max()) >= flattened_input_ids.numel():
        raise AssertionError("sample position falls outside the fixed input stream")
    exact, preserved, changed = _route_comparison(original, merged)
    overlap = preserved.float() / original.top_k
    reference = original.hidden.float()
    hidden_relative_l2 = torch.linalg.vector_norm(merged.hidden.float() - reference, dim=-1) / torch.linalg.vector_norm(
        reference, dim=-1
    ).clamp_min(EPSILON)
    layers: list[dict[str, Any]] = []
    for trace_index, layer_index in enumerate(original.layer_indices):
        shifted = ~exact[trace_index]
        histogram = {
            str(count): int((changed[trace_index] == count).sum())
            for count in range(original.top_k + 1)
        }
        fractions = {
            str(count): float((changed[trace_index] == count).float().mean())
            for count in range(original.top_k + 1)
        }
        layers.append({
            "layer": layer_index,
            "tokens": int(exact.shape[1]),
            "top_k": original.top_k,
            "num_sparse_experts": original.num_experts,
            "exact_topk_match_rate": float(exact[trace_index].float().mean()),
            "routing_shift_rate": float(shifted.float().mean()),
            "mean_topk_overlap": float(overlap[trace_index].mean()),
            "mean_changed_experts": float(changed[trace_index].float().mean()),
            "changed_expert_count_histogram": histogram,
            "changed_expert_count_fraction": fractions,
            "mean_hidden_relative_l2": float(hidden_relative_l2[trace_index].mean()),
            "median_hidden_relative_l2": float(hidden_relative_l2[trace_index].median()),
            "original_margin_shifted": _optional_mean(original.margin[trace_index], shifted),
            "original_margin_non_shifted": _optional_mean(original.margin[trace_index], ~shifted),
            "merged_margin_shifted": _optional_mean(merged.margin[trace_index], shifted),
            "merged_margin_non_shifted": _optional_mean(merged.margin[trace_index], ~shifted),
        })
    if not exact[0].all():
        mismatches = int((~exact[0]).sum())
        layer_index = original.layer_indices[0]
        raise AssertionError(
            f"first sparse MoE layer {layer_index} routing differs for {mismatches} sampled tokens; "
            "router/input invariant failed"
        )
    token_metrics = {
        "sample_positions": sample_positions.cpu(),
        "sampled_input_token_ids": flattened_input_ids.index_select(0, sample_positions.cpu()),
        "layer_indices": torch.tensor(original.layer_indices),
        "top_k": torch.tensor(original.top_k),
        "num_sparse_experts": torch.tensor(original.num_experts),
        "original_topk": original.topk.cpu(),
        "merged_topk": merged.topk.cpu(),
        "original_margin": original.margin.cpu(),
        "merged_margin": merged.margin.cpu(),
        "exact_topk_match": exact.cpu(),
        "preserved_expert_count": preserved.cpu(),
        "changed_expert_count": changed.cpu(),
        "topk_overlap": overlap.cpu(),
        "hidden_relative_l2": hidden_relative_l2.cpu(),
    }
    return layers, token_metrics


class _DecodeRoutingCapture:
    def __init__(self, layer_indices: tuple[int, ...], top_k: int, num_experts: int) -> None:
        self.layer_indices = layer_indices
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden: list[list[torch.Tensor]] = [[] for _ in layer_indices]
        self.topk: list[list[torch.Tensor]] = [[] for _ in layer_indices]
        self.margin: list[list[torch.Tensor]] = [[] for _ in layer_indices]

    def moe_input_hook(self, trace_index: int, layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if len(rows) != 1:
                raise AssertionError(f"layer {layer_index}: decode trace must contain only the new token")
            self.hidden[trace_index].append(rows[0].cpu())

        return hook

    def router_hook(self, trace_index: int, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer_index}: expected Qwen MoE hidden output and router logits")
            logits = output[1].detach().reshape(-1, output[1].shape[-1]).float()
            if logits.shape != (1, self.num_experts):
                raise AssertionError(
                    f"layer {layer_index}: decode router logits shape {tuple(logits.shape)} != (1, {self.num_experts})"
                )
            boundary = torch.topk(logits[0], self.top_k + 1)
            self.topk[trace_index].append(boundary.indices[: self.top_k].cpu())
            self.margin[trace_index].append(
                (boundary.values[self.top_k - 1] - boundary.values[self.top_k]).cpu()
            )

        return hook

    def register(self, sparse_layers: list[tuple[int, torch.nn.Module]]) -> list[Any]:
        handles = []
        for trace_index, (layer_index, mlp) in enumerate(sparse_layers):
            handles.append(mlp.register_forward_pre_hook(self.moe_input_hook(trace_index, layer_index)))
            handles.append(mlp.register_forward_hook(self.router_hook(trace_index, layer_index)))
        return handles

    def finalize(self, token_ids: list[int], decode_steps: int) -> RoutingDriftTrace:
        for trace_index, layer_index in enumerate(self.layer_indices):
            if len(self.hidden[trace_index]) != decode_steps or len(self.topk[trace_index]) != decode_steps:
                raise AssertionError(f"layer {layer_index}: incomplete incremental decode trace")
        return RoutingDriftTrace(
            hidden=torch.stack([torch.stack(values) for values in self.hidden]),
            topk=torch.stack([torch.stack(values) for values in self.topk]),
            margin=torch.stack([torch.stack(values) for values in self.margin]),
            layer_indices=self.layer_indices,
            top_k=self.top_k,
            num_experts=self.num_experts,
            token_ids=torch.tensor(token_ids, dtype=torch.long),
        )


@torch.inference_mode()
def autoregressive_routing_trace(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    decode_steps: int,
    forced_tokens: Optional[torch.Tensor] = None,
    description: str = "decode",
) -> RoutingDriftTrace:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must have shape [1, prompt_length]")
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    if forced_tokens is not None and forced_tokens.numel() != decode_steps:
        raise ValueError("forced token count must equal decode_steps")
    freeze_for_analysis(model)
    sparse_layers = _sparse_moe_layers(model)
    layer_indices = tuple(index for index, _ in sparse_layers)
    device = _forward_input_device(model)
    prompt = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt)
    output = model(input_ids=prompt, attention_mask=attention_mask, use_cache=True, return_dict=True)
    past_key_values = output.past_key_values
    next_logits = output.logits[:, -1]
    capture = _DecodeRoutingCapture(
        layer_indices,
        int(model.config.num_experts_per_tok),
        int(model.config.num_experts),
    )
    handles = capture.register(sparse_layers)
    generated: list[int] = []
    try:
        for step in tqdm(range(decode_steps), desc=f"[Qwen routing drift] {description}"):
            if forced_tokens is None:
                token = next_logits.argmax(dim=-1).reshape(1, 1)
            else:
                token = forced_tokens[step].reshape(1, 1).to(device)
            generated.append(int(token.item()))
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)), dim=1
            )
            output = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            next_logits = output.logits[:, -1]
    finally:
        for handle in handles:
            handle.remove()
    return capture.finalize(generated, decode_steps)


def compare_decode_traces(
    original: RoutingDriftTrace,
    merged: RoutingDriftTrace,
    require_identical_tokens: bool,
) -> dict[str, torch.Tensor]:
    if original.token_ids is None or merged.token_ids is None:
        raise AssertionError("decode traces must include generated token IDs")
    token_equal = original.token_ids == merged.token_ids
    if require_identical_tokens and not token_equal.all():
        raise AssertionError("forced decode must use identical continuation token IDs")
    exact, preserved, changed = _route_comparison(original, merged)
    overlap = preserved.float() / original.top_k
    hidden_relative_l2 = torch.linalg.vector_norm(
        merged.hidden.float() - original.hidden.float(), dim=-1
    ) / torch.linalg.vector_norm(original.hidden.float(), dim=-1).clamp_min(EPSILON)
    return {
        "original_token_ids": original.token_ids.cpu(),
        "merged_token_ids": merged.token_ids.cpu(),
        "token_equal": token_equal.cpu(),
        "layer_indices": torch.tensor(original.layer_indices),
        "top_k": torch.tensor(original.top_k),
        "num_sparse_experts": torch.tensor(original.num_experts),
        "original_topk": original.topk.cpu(),
        "merged_topk": merged.topk.cpu(),
        "exact_topk_match": exact.cpu(),
        "routing_shift": (~exact).cpu(),
        "preserved_expert_count": preserved.cpu(),
        "changed_expert_count": changed.cpu(),
        "topk_overlap": overlap.cpu(),
        "hidden_relative_l2": hidden_relative_l2.cpu(),
        "original_margin": original.margin.cpu(),
        "merged_margin": merged.margin.cpu(),
    }


def free_generation_summary(metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
    token_equal = metrics["token_equal"]
    token_mismatch = torch.nonzero(~token_equal, as_tuple=False).flatten()
    first_token = int(token_mismatch[0]) if token_mismatch.numel() else None
    steps = token_equal.numel()
    post_token_divergence = torch.zeros(steps, dtype=torch.bool)
    if first_token is not None:
        post_token_divergence[first_token:] = True
    any_routing_shift = metrics["routing_shift"].any(dim=0)
    clean_indices = torch.nonzero(any_routing_shift & ~post_token_divergence, as_tuple=False).flatten()
    first_routing = int(clean_indices[0]) if clean_indices.numel() else None
    metrics["post_token_divergence"] = post_token_divergence
    return {
        "first_routing_divergence_step": first_routing,
        "first_token_divergence_step": first_token,
        "routing_divergence_precedes_token_divergence": (
            first_routing is not None and (first_token is None or first_routing < first_token)
        ),
        "clean_prefix_decode_steps": first_token if first_token is not None else steps,
    }


def forced_decode_summary(metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
    shift = metrics["routing_shift"].float()
    overlap = metrics["topk_overlap"].float()
    changed = metrics["changed_expert_count"].float()
    hidden = metrics["hidden_relative_l2"].float()
    return {
        "mean_routing_shift_rate": float(shift.mean()),
        "mean_topk_overlap": float(overlap.mean()),
        "mean_changed_experts": float(changed.mean()),
        "mean_hidden_relative_l2": float(hidden.mean()),
        "routing_shift_rate_by_step": shift.mean(dim=0).tolist(),
        "routing_shift_rate_by_layer": shift.mean(dim=1).tolist(),
        "hidden_relative_l2_by_step": hidden.mean(dim=0).tolist(),
    }


def save_routing_drift_plots(
    layer_metrics: list[dict[str, Any]],
    forced_metrics: dict[str, torch.Tensor],
    output_dir: str | Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    layers = [row["layer"] for row in layer_metrics]
    paths: list[Path] = []

    def line_plot(x, values, xlabel: str, ylabel: str, filename: str) -> None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(x, values, marker="o")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if xlabel == "Qwen layer":
            axis.set_xticks(x)
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    line_plot(
        layers,
        [row["routing_shift_rate"] for row in layer_metrics],
        "Qwen layer",
        "Routing shift rate",
        "routing_shift_by_layer.png",
    )
    line_plot(
        layers,
        [row["mean_hidden_relative_l2"] for row in layer_metrics],
        "Qwen layer",
        "Mean hidden relative L2",
        "hidden_drift_by_layer.png",
    )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for key, label, linestyle in (
        ("original_margin_shifted", "Original, shifted", "-"),
        ("original_margin_non_shifted", "Original, non-shifted", "-"),
        ("merged_margin_shifted", "Merged, shifted", "--"),
        ("merged_margin_non_shifted", "Merged, non-shifted", "--"),
    ):
        axis.plot(
            layers,
            [math.nan if row[key] is None else row[key] for row in layer_metrics],
            label=label,
            linestyle=linestyle,
        )
    axis.set_xlabel("Qwen layer")
    axis.set_ylabel("Top-K / next-expert router margin")
    axis.legend()
    figure.tight_layout()
    path = output / "margin_shift_analysis.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    def heatmap(values: torch.Tensor, label: str, filename: str, vmin: float, vmax: float) -> None:
        figure, axis = plt.subplots(figsize=(10, 6))
        image = axis.imshow(values.float().numpy(), aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        axis.set_xlabel("Decode step")
        axis.set_ylabel("Qwen sparse layer index")
        axis.set_yticks(range(len(layers)), labels=layers)
        figure.colorbar(image, ax=axis, label=label)
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    heatmap(forced_metrics["routing_shift"], "Routing shift", "forced_routing_shift_heatmap.png", 0.0, 1.0)
    heatmap(forced_metrics["topk_overlap"], "Top-K overlap", "forced_topk_overlap_heatmap.png", 0.0, 1.0)
    decode_steps = list(range(forced_metrics["routing_shift"].shape[1]))
    line_plot(
        decode_steps,
        forced_metrics["routing_shift"].float().mean(dim=0).tolist(),
        "Decode step",
        "Mean routing shift rate across layers",
        "forced_shift_vs_decode_step.png",
    )
    line_plot(
        layers,
        forced_metrics["routing_shift"].float().mean(dim=1).tolist(),
        "Qwen layer",
        "Mean routing shift rate across decode steps",
        "forced_shift_vs_layer.png",
    )
    line_plot(
        decode_steps,
        forced_metrics["hidden_relative_l2"].float().mean(dim=0).tolist(),
        "Decode step",
        "Mean hidden relative L2 across layers",
        "forced_hidden_drift_vs_decode_step.png",
    )
    return paths


__all__ = [
    "GlobalSamplePlan",
    "RoutingDriftTrace",
    "assert_router_weights_unchanged",
    "autoregressive_routing_trace",
    "collect_fixed_routing_trace",
    "compare_decode_traces",
    "compare_fixed_routing_traces",
    "forced_decode_summary",
    "free_generation_summary",
    "freeze_for_analysis",
    "make_global_sample_plan",
    "router_weight_snapshot",
    "save_routing_drift_plots",
]
