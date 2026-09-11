"""Contiguous-PG19 routing-drift analysis for vanilla HC-SMoE Mixtral.

The original and merged models always execute independently.  Only token IDs
are shared.  Router decisions stay in the original expert-ID space.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from tqdm import tqdm

from hcsmoe.merging.sequential_drift_mixtral import EPSILON, _forward_input_device, freeze_for_analysis


@dataclass(frozen=True)
class PG19ModelAdapter:
    """Small architecture adapter; all PG19 collection remains model-independent."""

    name: str
    moe_attribute: str
    top_k_config: str
    expert_count_config: str
    requires_gate_attribute: bool = True

    def sparse_layers(self, model: torch.nn.Module) -> list[tuple[int, torch.nn.Module]]:
        layers: list[tuple[int, torch.nn.Module]] = []
        for index, layer in enumerate(model.model.layers):
            moe = getattr(layer, self.moe_attribute, None)
            # Qwen may have non-MoE MLPs; its shared expert is deliberately not
            # inspected or included.  Mixtral has an MoE block in every layer.
            if moe is not None and (not self.requires_gate_attribute or hasattr(moe, "gate")):
                layers.append((index, moe))
        if not layers:
            raise AssertionError(f"{self.name}: no sparse MoE layers found via layer.{self.moe_attribute}")
        return layers

    def routing_config(self, model: torch.nn.Module) -> tuple[int, int]:
        top_k = int(getattr(model.config, self.top_k_config))
        num_experts = int(getattr(model.config, self.expert_count_config))
        if not 1 <= top_k < num_experts:
            raise ValueError(f"{self.name}: invalid sparse routing K={top_k}, experts={num_experts}")
        return top_k, num_experts


MIXTRAL_ADAPTER = PG19ModelAdapter(
    "Mixtral", "block_sparse_moe", "num_experts_per_tok", "num_local_experts", requires_gate_attribute=False
)
QWEN_ADAPTER = PG19ModelAdapter("Qwen", "mlp", "num_experts_per_tok", "num_experts")


@dataclass
class RoutingDriftTrace:
    """CPU-resident generic sparse-router trace in original expert-ID space."""

    hidden: torch.Tensor
    topk: torch.Tensor
    margin: torch.Tensor
    layer_indices: tuple[int, ...]
    top_k: int
    num_experts: int
    token_ids: Optional[torch.Tensor] = None
    input_id_batches: list[torch.Tensor] = field(default_factory=list)

    @property
    def top2(self) -> torch.Tensor:
        """Compatibility alias for existing Mixtral PG19 consumers (K remains dynamic)."""
        return self.topk


def _route_comparison(original_topk: torch.Tensor, merged_topk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Set-based top-K equality and overlap; ordering of router top-k IDs is irrelevant."""
    if original_topk.shape != merged_topk.shape or original_topk.ndim < 1:
        raise AssertionError("routing traces must have matching [..., K] shapes")
    top_k = original_topk.shape[-1]
    if top_k <= 0:
        raise AssertionError("routing traces need a positive top-k dimension")
    original_set = original_topk.sort(dim=-1).values
    merged_set = merged_topk.sort(dim=-1).values
    exact = (original_set == merged_set).all(dim=-1)
    overlap = (
        (merged_topk.unsqueeze(-1) == original_topk.unsqueeze(-2)).any(dim=-1).sum(dim=-1).float() / top_k
    )
    return exact, overlap


def _trace_topk(trace: Any) -> torch.Tensor:
    """Read generic traces and the legacy Mixtral ``top2`` trace used by old tests."""
    return trace.topk if hasattr(trace, "topk") else trace.top2


def _same_routing_configuration(original: Any, merged: Any) -> bool:
    fields = ("layer_indices", "top_k", "num_experts")
    return all(
        not hasattr(original, field) or not hasattr(merged, field) or getattr(original, field) == getattr(merged, field)
        for field in fields
    )


@dataclass(frozen=True)
class PG19Document:
    """One deterministic contiguous PG19 evaluation segment."""

    document_id: str
    source_index: int
    title: str
    publication_date: Optional[int]
    url: str
    input_ids: torch.Tensor  # [prefill_tokens + decode_steps], CPU

    def metadata(self, prefill_tokens: int) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_index": self.source_index,
            "title": self.title,
            "publication_date": self.publication_date,
            "url": self.url,
            "segment_start": 0,
            "segment_tokens": int(self.input_ids.numel()),
            "prefill_tokens": prefill_tokens,
            "continuation_tokens": int(self.input_ids.numel()) - prefill_tokens,
            "input_token_ids": self.input_ids.tolist(),
        }


@dataclass
class DocumentRoutingTraces:
    """CPU-resident traces for one document and one model."""

    prefill: RoutingDriftTrace
    forced: RoutingDriftTrace
    free: RoutingDriftTrace


def select_pg19_documents_from_rows(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    num_documents: int,
    prefill_tokens: int,
    decode_steps: int,
    seed: int,
) -> list[PG19Document]:
    """Select documents deterministically and take only their first contiguous segment."""
    if num_documents <= 0 or prefill_tokens <= 0 or decode_steps <= 0:
        raise ValueError("document count, prefill length, and decode length must be positive")
    required_tokens = prefill_tokens + decode_steps
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected: list[PG19Document] = []
    for source_index in indices:
        row = rows[source_index]
        encoded = tokenizer(
            row["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=required_tokens,
            return_attention_mask=False,
        )["input_ids"]
        if len(encoded) < required_tokens:
            continue
        url = str(row.get("url") or "")
        title = str(row.get("short_book_title") or row.get("title") or "")
        document_id = url or f"pg19:{source_index}"
        selected.append(
            PG19Document(
                document_id=document_id,
                source_index=source_index,
                title=title,
                publication_date=row.get("publication_date"),
                url=url,
                input_ids=torch.tensor(encoded[:required_tokens], dtype=torch.long),
            )
        )
        if len(selected) == num_documents:
            return selected
    raise RuntimeError(
        f"PG19 split contains only {len(selected)} deterministically selected documents "
        f"with at least {required_tokens} tokenized tokens; requested {num_documents}"
    )


def load_pg19_documents(
    tokenizer: Any,
    num_documents: int,
    prefill_tokens: int,
    decode_steps: int,
    seed: int,
    dataset_name: str = "emozilla/pg19-test",
    dataset_split: str = "test",
) -> list[PG19Document]:
    """Load the Parquet PG19 mirror and select held-out test documents."""
    from datasets import load_dataset

    rows = load_dataset(dataset_name, split=dataset_split)
    return select_pg19_documents_from_rows(
        rows, tokenizer, num_documents, prefill_tokens, decode_steps, seed
    )


class _ContiguousRoutingCapture:
    """Capture every MoE input and router decision in one forward."""

    def __init__(self, layer_indices: tuple[int, ...], token_count: int, top_k: int, num_experts: int) -> None:
        if not 1 <= top_k < num_experts:
            raise ValueError(f"router boundary needs 1 <= K < experts, got K={top_k}, experts={num_experts}")
        self.layer_indices = layer_indices
        self.num_layers = len(layer_indices)
        self.token_count = token_count
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self.topk: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self.margin: list[Optional[torch.Tensor]] = [None] * self.num_layers

    def moe_input_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if rows.shape[0] != self.token_count:
                raise AssertionError(
                    f"layer {layer_index}: expected all {self.token_count} contiguous tokens, got {rows.shape[0]}"
                )
            # Keep one detached full prefill tensor on this layer's execution device.
            # It is moved to CPU once in finalize(), after the complete forward.
            self.hidden[layer_index] = rows.clone()

        return hook

    def router_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer_index}: expected MoE hidden output and router logits")
            logits = output[1].detach().reshape(-1, output[1].shape[-1]).float()
            if logits.shape != (self.token_count, self.num_experts):
                raise AssertionError(
                    f"layer {layer_index}: expected router logits [{self.token_count}, {self.num_experts}], "
                    f"got {list(logits.shape)}"
                )
            boundary = torch.topk(logits, self.top_k + 1, dim=-1)
            self.topk[layer_index] = boundary.indices[:, : self.top_k].clone()
            self.margin[layer_index] = (boundary.values[:, self.top_k - 1] - boundary.values[:, self.top_k]).clone()

        return hook

    def register(self, sparse_layers: list[tuple[int, torch.nn.Module]]) -> list[Any]:
        handles = []
        for trace_index, (layer_index, moe) in enumerate(sparse_layers):
            handles.append(moe.register_forward_pre_hook(self.moe_input_hook(trace_index)))
            handles.append(moe.register_forward_hook(self.router_hook(trace_index)))
        return handles

    def finalize(self, token_ids: torch.Tensor) -> RoutingDriftTrace:
        if any(value is None for value in self.hidden + self.topk + self.margin):
            raise AssertionError("contiguous prefill trace is incomplete")
        trace = RoutingDriftTrace(
            hidden=torch.stack([value.detach().cpu() for value in self.hidden if value is not None]),
            topk=torch.stack([value.detach().cpu() for value in self.topk if value is not None]),
            margin=torch.stack([value.detach().cpu() for value in self.margin if value is not None]),
            layer_indices=self.layer_indices,
            top_k=self.top_k,
            num_experts=self.num_experts,
            token_ids=token_ids.detach().cpu().reshape(-1).clone(),
        )
        # Release device-side buffers before forced/free decoding begins.
        self.hidden, self.topk, self.margin = [], [], []
        return trace


class _IncrementalRoutingCapture:
    """Capture routing information for each newly processed decode token."""

    def __init__(self, layer_indices: tuple[int, ...], top_k: int, num_experts: int) -> None:
        self.layer_indices = layer_indices
        self.num_layers = len(layer_indices)
        self.top_k = top_k
        self.num_experts = num_experts
        self.decode_steps: Optional[int] = None
        self.step: Optional[int] = None
        self.hidden: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self.topk: list[Optional[torch.Tensor]] = [None] * self.num_layers
        self.margin: list[Optional[torch.Tensor]] = [None] * self.num_layers

    def begin_step(self, step: int, decode_steps: int) -> None:
        if self.decode_steps is None:
            self.decode_steps = decode_steps
        elif self.decode_steps != decode_steps:
            raise AssertionError("decode length changed within one trace")
        self.step = step

    def _slot(self) -> int:
        if self.step is None or self.decode_steps is None:
            raise RuntimeError("incremental hook ran outside an active decode step")
        return self.step

    @staticmethod
    def _buffer(current: Optional[torch.Tensor], steps: int, value: torch.Tensor) -> torch.Tensor:
        if current is None:
            return torch.empty((steps, *value.shape), device=value.device, dtype=value.dtype)
        if current.device != value.device:
            raise AssertionError("a sparse layer changed execution device during one decode trace")
        return current

    def moe_input_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if rows.shape[0] != 1:
                raise AssertionError("incremental decode must process exactly one new token")
            slot = self._slot()
            self.hidden[layer_index] = self._buffer(self.hidden[layer_index], self.decode_steps, rows[0])
            self.hidden[layer_index][slot].copy_(rows[0])

        return hook

    def router_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer_index}: expected MoE hidden output and router logits")
            logits = output[1].detach().reshape(-1, output[1].shape[-1]).float()
            if logits.shape != (1, self.num_experts):
                raise AssertionError(f"layer {layer_index}: incremental router trace has shape {list(logits.shape)}")
            boundary = torch.topk(logits[0], self.top_k + 1)
            slot = self._slot()
            ids = boundary.indices[: self.top_k]
            margin = boundary.values[self.top_k - 1] - boundary.values[self.top_k]
            self.topk[layer_index] = self._buffer(self.topk[layer_index], self.decode_steps, ids)
            self.margin[layer_index] = self._buffer(self.margin[layer_index], self.decode_steps, margin)
            self.topk[layer_index][slot].copy_(ids)
            self.margin[layer_index][slot].copy_(margin)

        return hook

    def register(self, sparse_layers: list[tuple[int, torch.nn.Module]]) -> list[Any]:
        handles = []
        for trace_index, (layer_index, moe) in enumerate(sparse_layers):
            handles.append(moe.register_forward_pre_hook(self.moe_input_hook(trace_index)))
            handles.append(moe.register_forward_hook(self.router_hook(trace_index)))
        return handles

    def finalize(self, token_ids: torch.Tensor, decode_steps: int) -> RoutingDriftTrace:
        for layer_index in range(self.num_layers):
            if any(values is None for values in (self.hidden[layer_index], self.topk[layer_index], self.margin[layer_index])):
                raise AssertionError(f"layer {layer_index}: incomplete incremental decode trace")
        trace = RoutingDriftTrace(
            hidden=torch.stack([value.detach().cpu() for value in self.hidden if value is not None]),
            topk=torch.stack([value.detach().cpu() for value in self.topk if value is not None]),
            margin=torch.stack([value.detach().cpu() for value in self.margin if value is not None]),
            layer_indices=self.layer_indices,
            top_k=self.top_k,
            num_experts=self.num_experts,
            token_ids=token_ids.detach().cpu(),
        )
        self.hidden, self.topk, self.margin = [], [], []
        return trace


def _as_legacy_cache(past_key_values: Any) -> Any:
    """Use immutable tuple caches so forced/free branches share only the prefill state."""
    converter = getattr(past_key_values, "to_legacy_cache", None)
    return converter() if callable(converter) else past_key_values


@torch.inference_mode()
def _decode_from_prefill(
    model: torch.nn.Module,
    prefill_length: int,
    past_key_values: Any,
    next_logits: torch.Tensor,
    decode_steps: int,
    top_k: int,
    num_experts: int,
    forced_tokens: Optional[torch.Tensor],
    description: str,
    sparse_layers: list[tuple[int, torch.nn.Module]],
) -> RoutingDriftTrace:
    device = _forward_input_device(model)
    full_attention_mask = torch.ones((1, prefill_length + decode_steps), dtype=torch.long, device=device)
    cache = past_key_values
    logits = next_logits
    capture = _IncrementalRoutingCapture(tuple(index for index, _ in sparse_layers), top_k, num_experts)
    handles = capture.register(sparse_layers)
    token_ids = torch.empty((decode_steps,), dtype=torch.long, device=device)
    forced_device = None if forced_tokens is None else forced_tokens.to(device=device, dtype=torch.long)
    try:
        for step in tqdm(range(decode_steps), desc=f"[PG19 routing drift] {description}"):
            if forced_tokens is None:
                token = logits.argmax(dim=-1).reshape(1, 1)
            else:
                token = forced_device[step].reshape(1, 1)
            token_ids[step].copy_(token.reshape(()))
            capture.begin_step(step, decode_steps)
            output = model(
                input_ids=token,
                attention_mask=full_attention_mask[:, : prefill_length + step + 1],
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = output.past_key_values
            logits = output.logits[:, -1].detach()
    finally:
        for handle in handles:
            handle.remove()
    return capture.finalize(token_ids, decode_steps)


@torch.inference_mode()
def collect_document_routing_traces(
    model: torch.nn.Module,
    document: PG19Document,
    prefill_tokens: int,
    decode_steps: int,
    description: str,
    adapter: PG19ModelAdapter = MIXTRAL_ADAPTER,
) -> DocumentRoutingTraces:
    """Run one prefill and branch its immutable cache into forced and greedy decode."""
    if document.input_ids.numel() != prefill_tokens + decode_steps:
        raise AssertionError("document segment length changed")
    freeze_for_analysis(model)
    top_k, num_experts = adapter.routing_config(model)
    sparse_layers = adapter.sparse_layers(model)
    device = _forward_input_device(model)
    prefix = document.input_ids[:prefill_tokens].reshape(1, -1).to(device)
    continuation = document.input_ids[prefill_tokens:].cpu()
    attention_mask = torch.ones_like(prefix)
    layer_indices = tuple(index for index, _ in sparse_layers)
    capture = _ContiguousRoutingCapture(layer_indices, prefill_tokens, top_k, num_experts)
    handles = capture.register(sparse_layers)
    try:
        output = model(
            input_ids=prefix,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    prefill = capture.finalize(prefix)
    base_cache = _as_legacy_cache(output.past_key_values)
    # Clone the final row before releasing the full [1, prefill, vocab] logits.
    # Keeping a view would unnecessarily pin the much larger prefill allocation.
    base_logits = output.logits[:, -1].detach().clone()
    del output
    forced = _decode_from_prefill(
        model,
        prefill_tokens,
        base_cache,
        base_logits,
        decode_steps,
        top_k,
        num_experts,
        continuation,
        f"{description} forced",
        sparse_layers,
    )
    free = _decode_from_prefill(
        model,
        prefill_tokens,
        base_cache,
        base_logits,
        decode_steps,
        top_k,
        num_experts,
        None,
        f"{description} free",
        sparse_layers,
    )
    if not torch.equal(forced.token_ids, continuation):
        raise AssertionError("forced decode did not process the exact PG19 continuation")
    return DocumentRoutingTraces(prefill=prefill, forced=forced, free=free)


def _relative_l2(original: torch.Tensor, merged: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(merged.float() - original.float(), dim=-1) / torch.linalg.vector_norm(
        original.float(), dim=-1
    ).clamp_min(EPSILON)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    return float(values[mask].mean()) if mask.any() else None


def compare_prefill_document(
    original: RoutingDriftTrace,
    merged: RoutingDriftTrace,
    document_id: str,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Compare every contiguous prefill position and enforce the layer-0 invariant."""
    if original.token_ids is None or merged.token_ids is None or not torch.equal(original.token_ids, merged.token_ids):
        raise AssertionError("original and merged prefill token IDs differ")
    if not _same_routing_configuration(original, merged):
        raise AssertionError("original and merged prefill routing configurations differ")
    original_topk, merged_topk = _trace_topk(original), _trace_topk(merged)
    exact, overlap = _route_comparison(original_topk, merged_topk)
    hidden_l2 = _relative_l2(original.hidden, merged.hidden)
    if not exact[0].all():
        mismatches = int((~exact[0]).sum().item())
        raise AssertionError(
            f"document {document_id}: layer-0 routing differs at {mismatches} prefill positions; "
            "router/input invariant failed"
        )
    rows: list[dict[str, Any]] = []
    for layer_index in range(exact.shape[0]):
        shifted = ~exact[layer_index]
        rows.append({
            "layer": layer_index,
            "tokens": int(exact.shape[1]),
            "routing_shift_rate": float(shifted.float().mean()),
            "exact_topk_match_rate": float(exact[layer_index].float().mean()),
            "mean_topk_overlap": float(overlap[layer_index].mean()),
            "mean_hidden_relative_l2": float(hidden_l2[layer_index].mean()),
            "median_hidden_relative_l2": float(hidden_l2[layer_index].median()),
            "original_margin_shifted": _masked_mean(original.margin[layer_index], shifted),
            "original_margin_non_shifted": _masked_mean(original.margin[layer_index], ~shifted),
            "merged_margin_shifted": _masked_mean(merged.margin[layer_index], shifted),
            "merged_margin_non_shifted": _masked_mean(merged.margin[layer_index], ~shifted),
        })
    raw = {
        "input_token_ids": original.token_ids.cpu(),
        "original_topk": original_topk.cpu(),
        "merged_topk": merged_topk.cpu(),
        "exact_topk_match": exact.cpu(),
        "routing_shift": (~exact).cpu(),
        "topk_overlap": overlap.cpu(),
        "hidden_relative_l2": hidden_l2.cpu(),
        "original_margin": original.margin.cpu(),
        "merged_margin": merged.margin.cpu(),
    }
    return rows, raw


def compare_decode_document(
    original: RoutingDriftTrace,
    merged: RoutingDriftTrace,
    require_identical_tokens: bool,
) -> dict[str, torch.Tensor]:
    if original.token_ids is None or merged.token_ids is None:
        raise AssertionError("decode traces must include token IDs")
    token_equal = original.token_ids == merged.token_ids
    if require_identical_tokens and not token_equal.all():
        raise AssertionError("forced original/merged runs must receive identical PG19 continuation IDs")
    if not _same_routing_configuration(original, merged):
        raise AssertionError("original and merged decode routing configurations differ")
    original_topk, merged_topk = _trace_topk(original), _trace_topk(merged)
    exact, overlap = _route_comparison(original_topk, merged_topk)
    return {
        "original_token_ids": original.token_ids.cpu(),
        "merged_token_ids": merged.token_ids.cpu(),
        "token_equal": token_equal.cpu(),
        "original_topk": original_topk.cpu(),
        "merged_topk": merged_topk.cpu(),
        "exact_topk_match": exact.cpu(),
        "routing_shift": (~exact).cpu(),
        "topk_overlap": overlap.cpu(),
        "hidden_relative_l2": _relative_l2(original.hidden, merged.hidden).cpu(),
        "original_margin": original.margin.cpu(),
        "merged_margin": merged.margin.cpu(),
    }


def summarize_free_document(
    metrics: dict[str, torch.Tensor],
    prefill_routing_shift: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    token_equal = metrics["token_equal"]
    mismatch = torch.nonzero(~token_equal, as_tuple=False).flatten()
    first_token = int(mismatch[0]) if mismatch.numel() else None
    post = torch.zeros_like(token_equal, dtype=torch.bool)
    if first_token is not None:
        post[first_token:] = True
    any_shift = metrics["routing_shift"].any(dim=0)
    any_indices = torch.nonzero(any_shift, as_tuple=False).flatten()
    first_decode_routing = int(any_indices[0]) if any_indices.numel() else None
    clean = any_shift & ~post
    clean_indices = torch.nonzero(clean, as_tuple=False).flatten()
    first_routing = int(clean_indices[0]) if clean_indices.numel() else None
    metrics["post_token_divergence"] = post
    prefill_has = None if prefill_routing_shift is None else bool(prefill_routing_shift.any())
    last_prefill = None if prefill_routing_shift is None else bool(prefill_routing_shift[:, -1].any())
    return {
        "prefill_has_routing_divergence": prefill_has,
        "last_prefill_position_has_routing_divergence": last_prefill,
        "first_decode_routing_divergence_step": first_decode_routing,
        "first_routing_divergence_step": first_routing,
        "first_token_divergence_step": first_token,
        "routing_divergence_precedes_token_divergence": (
            first_routing is not None and (first_token is None or first_routing < first_token)
        ),
        "clean_prefix_decode_steps": first_token if first_token is not None else int(token_equal.numel()),
    }


def stack_document_metrics(document_ids: list[str], metrics: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    if len(document_ids) != len(metrics):
        raise AssertionError("document ID and metric counts differ")
    keys = metrics[0].keys()
    if any(item.keys() != metrics[0].keys() for item in metrics[1:]):
        raise AssertionError("per-document metric schemas differ")
    return {
        "document_ids": document_ids,
        **{key: torch.stack([item[key] for item in metrics]) for key in keys},
    }


def _std(values: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    return values.float().std(dim=dim, correction=0)


def _nan_stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = ~torch.isnan(values)
    count = valid.sum(dim=0).clamp_min(1)
    mean = torch.where(valid, values, 0.0).sum(dim=0) / count
    variance = torch.where(valid, (values - mean) ** 2, 0.0).sum(dim=0) / count
    mean = mean.masked_fill(~valid.any(dim=0), float("nan"))
    return mean, variance.sqrt().masked_fill(~valid.any(dim=0), float("nan"))


def _json_number(value: torch.Tensor | float) -> Optional[float]:
    number = float(value)
    return number if math.isfinite(number) else None


def aggregate_prefill_metrics(
    document_rows: list[list[dict[str, Any]]],
    raw: dict[str, Any],
) -> list[dict[str, Any]]:
    """Aggregate equally sized documents while retaining document-level standard deviations."""
    shift = raw["routing_shift"].float()  # [documents, layers, tokens]
    exact = raw["exact_topk_match"].float()
    overlap = raw["topk_overlap"].float()
    hidden = raw["hidden_relative_l2"].float()
    original_margin = raw["original_margin"].float()
    merged_margin = raw["merged_margin"].float()
    documents, layers, tokens = shift.shape
    shifted = shift.bool()

    def conditional_by_document(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=2)
        means = (values * mask).sum(dim=2) / counts.clamp_min(1)
        return means.masked_fill(counts == 0, float("nan"))

    conditional = {
        "original_margin_shifted": conditional_by_document(original_margin, shifted),
        "original_margin_non_shifted": conditional_by_document(original_margin, ~shifted),
        "merged_margin_shifted": conditional_by_document(merged_margin, shifted),
        "merged_margin_non_shifted": conditional_by_document(merged_margin, ~shifted),
    }
    conditional_stats = {key: _nan_stats(value) for key, value in conditional.items()}
    rows: list[dict[str, Any]] = []
    for layer in range(layers):
        pooled_hidden = hidden[:, layer].reshape(-1)
        row: dict[str, Any] = {
            "layer": layer,
            "num_documents": documents,
            "tokens_per_document": tokens,
            "total_tokens": documents * tokens,
            "routing_shift_rate": float(shift[:, layer].mean()),
            "routing_shift_rate_std_across_documents": float(_std(shift[:, layer].mean(dim=1), 0)),
            "exact_topk_match_rate": float(exact[:, layer].mean()),
            "exact_topk_match_rate_std_across_documents": float(_std(exact[:, layer].mean(dim=1), 0)),
            "mean_topk_overlap": float(overlap[:, layer].mean()),
            "mean_topk_overlap_std_across_documents": float(_std(overlap[:, layer].mean(dim=1), 0)),
            "mean_hidden_relative_l2": float(pooled_hidden.mean()),
            "mean_hidden_relative_l2_std_across_documents": float(_std(hidden[:, layer].mean(dim=1), 0)),
            "median_hidden_relative_l2": float(pooled_hidden.median()),
            "median_hidden_relative_l2_std_across_documents": float(
                _std(hidden[:, layer].median(dim=1).values, 0)
            ),
        }
        for key, (means, stds) in conditional_stats.items():
            row[key] = _json_number(means[layer])
            row[f"{key}_std_across_documents"] = _json_number(stds[layer])
        rows.append(row)
    if len(document_rows) != documents or any(len(rows_) != layers for rows_ in document_rows):
        raise AssertionError("prefill document metric shape differs from raw tensors")
    return rows


def aggregate_forced_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    shift = raw["routing_shift"].float()  # [documents, layers, steps]
    overlap = raw["topk_overlap"].float()
    hidden = raw["hidden_relative_l2"].float()
    by_step_doc = shift.mean(dim=1)
    by_layer_doc = shift.mean(dim=2)
    hidden_step_doc = hidden.mean(dim=1)
    return {
        "num_documents": int(shift.shape[0]),
        "num_layers": int(shift.shape[1]),
        "decode_steps": int(shift.shape[2]),
        "mean_routing_shift_rate": float(shift.mean()),
        "mean_topk_overlap": float(overlap.mean()),
        "mean_hidden_relative_l2": float(hidden.mean()),
        "routing_shift_heatmap_mean": shift.mean(dim=0).tolist(),
        "routing_shift_heatmap_std_across_documents": _std(shift, 0).tolist(),
        "topk_overlap_heatmap_mean": overlap.mean(dim=0).tolist(),
        "topk_overlap_heatmap_std_across_documents": _std(overlap, 0).tolist(),
        "routing_shift_by_step_mean": by_step_doc.mean(dim=0).tolist(),
        "routing_shift_by_step_std_across_documents": _std(by_step_doc, 0).tolist(),
        "routing_shift_by_layer_mean": by_layer_doc.mean(dim=0).tolist(),
        "routing_shift_by_layer_std_across_documents": _std(by_layer_doc, 0).tolist(),
        "hidden_relative_l2_by_step_mean": hidden_step_doc.mean(dim=0).tolist(),
        "hidden_relative_l2_by_step_std_across_documents": _std(hidden_step_doc, 0).tolist(),
    }


def aggregate_free_summaries(document_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    decode_routing_steps = [
        row["first_decode_routing_divergence_step"]
        for row in document_summaries
        if row["first_decode_routing_divergence_step"] is not None
    ]
    routing_steps = [
        row["first_routing_divergence_step"]
        for row in document_summaries
        if row["first_routing_divergence_step"] is not None
    ]
    token_steps = [
        row["first_token_divergence_step"]
        for row in document_summaries
        if row["first_token_divergence_step"] is not None
    ]

    def observed(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"observed_documents": 0, "mean": None, "median": None}
        return {
            "observed_documents": len(values),
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
        }

    return {
        "num_documents": len(document_summaries),
        "fraction_routing_divergence_precedes_token_divergence": sum(
            bool(row["routing_divergence_precedes_token_divergence"]) for row in document_summaries
        ) / len(document_summaries),
        "first_routing_divergence_step": observed(routing_steps),
        "first_decode_routing_divergence_step": observed(decode_routing_steps),
        "first_token_divergence_step": observed(token_steps),
        "documents_with_prefill_routing_divergence": sum(
            bool(row.get("prefill_has_routing_divergence")) for row in document_summaries
        ),
        "documents_with_last_prefill_position_routing_divergence": sum(
            bool(row.get("last_prefill_position_has_routing_divergence")) for row in document_summaries
        ),
        "documents": document_summaries,
    }


def save_pg19_plots(
    prefill_layers: list[dict[str, Any]],
    forced_summary: dict[str, Any],
    output_dir: str | Path,
    model_label: str = "Mixtral",
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    layers = [row["layer"] for row in prefill_layers]

    def curve_with_std(x, mean, std, xlabel, ylabel, filename, label=None):
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(x, mean, marker="o" if len(x) < 100 else None, label=label)
        axis.fill_between(x, torch.tensor(mean) - torch.tensor(std), torch.tensor(mean) + torch.tensor(std), alpha=0.2)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if label:
            axis.legend()
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    curve_with_std(
        layers,
        [row["routing_shift_rate"] for row in prefill_layers],
        [row["routing_shift_rate_std_across_documents"] for row in prefill_layers],
        f"{model_label} layer",
        "Routing shift rate",
        "prefill_routing_shift_by_layer.png",
    )
    curve_with_std(
        layers,
        [row["mean_hidden_relative_l2"] for row in prefill_layers],
        [row["mean_hidden_relative_l2_std_across_documents"] for row in prefill_layers],
        f"{model_label} layer",
        "Mean hidden-state relative L2",
        "prefill_hidden_drift_by_layer.png",
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    for key, label, style in (
        ("original_margin_shifted", "Original, shifted", "-"),
        ("original_margin_non_shifted", "Original, stable", "-"),
        ("merged_margin_shifted", "Merged, shifted", "--"),
        ("merged_margin_non_shifted", "Merged, stable", "--"),
    ):
        mean = [float("nan") if row[key] is None else row[key] for row in prefill_layers]
        std = [
            float("nan") if row[f"{key}_std_across_documents"] is None else row[f"{key}_std_across_documents"]
            for row in prefill_layers
        ]
        axis.plot(layers, mean, label=label, linestyle=style)
        axis.fill_between(layers, torch.tensor(mean) - torch.tensor(std), torch.tensor(mean) + torch.tensor(std), alpha=0.12)
    axis.set_xlabel(f"{model_label} layer")
    axis.set_ylabel("Top-K / top-(K+1) router boundary margin")
    axis.legend()
    figure.tight_layout()
    path = output / "router_margin_shifted_vs_stable.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    def heatmap(values, label, filename, vmin, vmax):
        figure, axis = plt.subplots(figsize=(11, 6))
        image = axis.imshow(values, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        axis.set_xlabel("Forced decode step")
        axis.set_ylabel(f"{model_label} layer")
        figure.colorbar(image, ax=axis, label=label)
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    heatmap(
        forced_summary["routing_shift_heatmap_mean"],
        "Mean routing shift rate across documents",
        "forced_routing_shift_heatmap.png",
        0.0,
        1.0,
    )
    heatmap(
        forced_summary["topk_overlap_heatmap_mean"],
        "Mean top-K overlap across documents",
        "forced_topk_overlap_heatmap.png",
        0.0,
        1.0,
    )
    steps = list(range(forced_summary["decode_steps"]))
    curve_with_std(
        steps,
        forced_summary["routing_shift_by_step_mean"],
        forced_summary["routing_shift_by_step_std_across_documents"],
        "Forced decode step",
        "Mean routing shift rate across layers",
        "forced_routing_shift_by_step.png",
    )
    curve_with_std(
        layers,
        forced_summary["routing_shift_by_layer_mean"],
        forced_summary["routing_shift_by_layer_std_across_documents"],
        f"{model_label} layer",
        "Mean routing shift rate across forced steps",
        "forced_routing_shift_by_layer.png",
    )
    curve_with_std(
        steps,
        forced_summary["hidden_relative_l2_by_step_mean"],
        forced_summary["hidden_relative_l2_by_step_std_across_documents"],
        "Forced decode step",
        "Mean hidden-state relative L2 across layers",
        "forced_hidden_drift_by_step.png",
    )
    return paths


__all__ = [
    "DocumentRoutingTraces",
    "MIXTRAL_ADAPTER",
    "PG19Document",
    "PG19ModelAdapter",
    "QWEN_ADAPTER",
    "RoutingDriftTrace",
    "aggregate_forced_metrics",
    "aggregate_free_summaries",
    "aggregate_prefill_metrics",
    "collect_document_routing_traces",
    "compare_decode_document",
    "compare_prefill_document",
    "load_pg19_documents",
    "save_pg19_plots",
    "select_pg19_documents_from_rows",
    "stack_document_metrics",
    "summarize_free_document",
]
