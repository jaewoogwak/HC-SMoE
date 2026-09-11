"""End-to-end sequential hidden-state and router drift diagnostics for Mixtral.

This module intentionally does not reuse frozen-routing reconstruction data.  Each
model performs its own decoder forward, propagates its own hidden states, and
selects its own top-2 experts.  Only the input token positions are shared.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from tqdm import tqdm

from hcsmoe.merging.residual_mixtral import _input_device

TOP_K = 2
EPSILON = 1e-12


def _forward_input_device(model: torch.nn.Module) -> torch.device:
    """Find the active input device, including Accelerate CPU-offloaded models."""
    device = _input_device(model)
    if device.type != "meta":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CPU-offloaded Mixtral forward requires a CUDA execution device")
    return torch.device("cuda:0")


@dataclass(frozen=True)
class GlobalSamplePlan:
    """One deterministic set of flattened C4 token positions for every run."""

    positions: torch.Tensor
    total_tokens: int
    seed: int

    @property
    def token_count(self) -> int:
        return int(self.positions.numel())

    def selection(self, batch_offset: int, batch_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sample slots, local flattened token rows) for one batch."""
        batch_end = batch_offset + batch_tokens
        start = int(torch.searchsorted(self.positions, batch_offset, right=False).item())
        end = int(torch.searchsorted(self.positions, batch_end, right=False).item())
        slots = torch.arange(start, end, dtype=torch.long)
        return slots, self.positions[start:end] - batch_offset


def make_global_sample_plan(total_tokens: int, sample_tokens: int, seed: int) -> GlobalSamplePlan:
    """Draw unique global positions once; sorting preserves a stable storage order."""
    if total_tokens <= 0:
        raise ValueError("calibration stream has no tokens")
    if sample_tokens <= 0:
        raise ValueError("--sample-tokens must be positive")
    selected = min(sample_tokens, total_tokens)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    positions = torch.randperm(total_tokens, generator=generator)[:selected].sort().values
    return GlobalSamplePlan(positions=positions, total_tokens=total_tokens, seed=seed)


@dataclass
class OriginalTrajectory:
    """CPU-resident teacher references for just the globally sampled tokens."""

    decoder_input: Optional[torch.Tensor]
    layer_hidden_states: list[Optional[torch.Tensor]]
    router_top2: list[Optional[torch.Tensor]]
    input_id_batches: list[torch.Tensor] = field(default_factory=list)


class _SampledHookCapture:
    """Common batch-position bookkeeping for decoder and MoE hooks."""

    def __init__(self, sample_plan: GlobalSamplePlan, num_layers: int) -> None:
        self.sample_plan = sample_plan
        self.num_layers = num_layers
        self._batch_tokens: Optional[int] = None
        self._slots = torch.empty(0, dtype=torch.long)
        self._local_rows = torch.empty(0, dtype=torch.long)
        self._seen = {
            "decoder_input": torch.zeros(sample_plan.token_count, dtype=torch.bool),
            "hidden": [torch.zeros(sample_plan.token_count, dtype=torch.bool) for _ in range(num_layers)],
            "router": [torch.zeros(sample_plan.token_count, dtype=torch.bool) for _ in range(num_layers)],
        }

    def begin_batch(self, batch_offset: int, batch_tokens: int) -> None:
        self._batch_tokens = batch_tokens
        self._slots, self._local_rows = self.sample_plan.selection(batch_offset, batch_tokens)

    def end_batch(self) -> None:
        self._batch_tokens = None

    def sample_rows(self, value: torch.Tensor, label: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Move only this batch's selected flattened rows to CPU."""
        if self._batch_tokens is None:
            raise RuntimeError(f"{label}: hook ran outside an active batch")
        flattened = value.detach().reshape(-1, *value.shape[2:]) if value.ndim >= 3 else value.detach().reshape(-1, *value.shape[1:])
        if flattened.shape[0] != self._batch_tokens:
            raise AssertionError(
                f"{label}: expected {self._batch_tokens} flattened tokens, got {flattened.shape[0]}"
            )
        if self._local_rows.numel() == 0:
            return self._slots, torch.empty((0, *flattened.shape[1:]), dtype=flattened.dtype)
        rows = self._local_rows.to(device=flattened.device)
        return self._slots, flattened.index_select(0, rows).cpu()

    def mark_seen(self, kind: str, layer: Optional[int], slots: torch.Tensor, label: str) -> None:
        seen = self._seen[kind] if layer is None else self._seen[kind][layer]
        if seen[slots].any():
            raise AssertionError(f"{label}: sampled token captured more than once")
        seen[slots] = True

    def assert_complete(self) -> None:
        missing_input = int((~self._seen["decoder_input"]).sum().item())
        if missing_input:
            raise AssertionError(f"layer-0 decoder input missed {missing_input} sampled tokens")
        for kind in ("hidden", "router"):
            for layer, seen in enumerate(self._seen[kind]):
                missing = int((~seen).sum().item())
                if missing:
                    raise AssertionError(f"layer {layer} {kind} missed {missing} sampled tokens")


class OriginalTrajectoryCapture(_SampledHookCapture):
    """Capture teacher block outputs and the teacher's actual router decisions."""

    def __init__(self, sample_plan: GlobalSamplePlan, num_layers: int) -> None:
        super().__init__(sample_plan, num_layers)
        self.trajectory = OriginalTrajectory(
            decoder_input=None,
            layer_hidden_states=[None] * num_layers,
            router_top2=[None] * num_layers,
        )

    def _store(self, collection: list[Optional[torch.Tensor]], layer: int, slots: torch.Tensor, values: torch.Tensor, label: str) -> None:
        if collection[layer] is None:
            collection[layer] = torch.empty(
                (self.sample_plan.token_count, *values.shape[1:]), dtype=values.dtype
            )
        collection[layer][slots] = values
        self.mark_seen("hidden" if collection is self.trajectory.layer_hidden_states else "router", layer, slots, label)

    def decoder_input_hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        slots, values = self.sample_rows(inputs[0], "layer 0 decoder input")
        if self.trajectory.decoder_input is None:
            self.trajectory.decoder_input = torch.empty(
                (self.sample_plan.token_count, *values.shape[1:]), dtype=values.dtype
            )
        self.trajectory.decoder_input[slots] = values
        self.mark_seen("decoder_input", None, slots, "layer 0 decoder input")

    def decoder_output_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            slots, values = self.sample_rows(hidden, f"layer {layer} decoder output")
            self._store(self.trajectory.layer_hidden_states, layer, slots, values, f"layer {layer} decoder output")
        return hook

    def router_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer}: expected MoE output and router logits")
            slots, logits = self.sample_rows(output[1], f"layer {layer} router logits")
            top2 = torch.topk(logits, TOP_K, dim=-1).indices
            self._store(self.trajectory.router_top2, layer, slots, top2, f"layer {layer} router")
        return hook

    def register(self, model: torch.nn.Module) -> list[Any]:
        handles: list[Any] = [model.model.layers[0].register_forward_pre_hook(self.decoder_input_hook)]
        for layer_index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(self.decoder_output_hook(layer_index)))
            handles.append(layer.block_sparse_moe.register_forward_hook(self.router_hook(layer_index)))
        return handles

    def finalize(self) -> OriginalTrajectory:
        self.assert_complete()
        if self.trajectory.decoder_input is None:
            raise AssertionError("teacher decoder input was not captured")
        if any(value is None for value in self.trajectory.layer_hidden_states + self.trajectory.router_top2):
            raise AssertionError("teacher trajectory capture is incomplete")
        return self.trajectory


class CandidateTrajectoryCapture(_SampledHookCapture):
    """Compare one sequential candidate forward against a captured teacher path."""

    def __init__(self, sample_plan: GlobalSamplePlan, reference: OriginalTrajectory, num_layers: int) -> None:
        super().__init__(sample_plan, num_layers)
        self.reference = reference
        self.hidden_totals = [defaultdict(float) for _ in range(num_layers)]
        self.router_totals = [defaultdict(int) for _ in range(num_layers)]

    def decoder_input_hook(self, _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        slots, values = self.sample_rows(inputs[0], "layer 0 decoder input")
        torch.testing.assert_close(values, self.reference.decoder_input[slots], rtol=0.0, atol=0.0)
        self.mark_seen("decoder_input", None, slots, "layer 0 decoder input")

    def decoder_output_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            slots, values = self.sample_rows(hidden, f"layer {layer} decoder output")
            reference = self.reference.layer_hidden_states[layer][slots].float()
            candidate = values.float()
            totals = self.hidden_totals[layer]
            totals["target_norm"] += reference.square().sum().item()
            totals["candidate_norm"] += candidate.square().sum().item()
            totals["error"] += (candidate - reference).square().sum().item()
            totals["dot"] += (candidate * reference).sum().item()
            totals["tokens"] += candidate.shape[0]
            self.mark_seen("hidden", layer, slots, f"layer {layer} decoder output")
        return hook

    def router_hook(self, layer: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer}: expected MoE output and router logits")
            slots, logits = self.sample_rows(output[1], f"layer {layer} router logits")
            candidate = torch.topk(logits, TOP_K, dim=-1).indices
            reference = self.reference.router_top2[layer][slots]
            candidate_set = candidate.sort(dim=-1).values
            reference_set = reference.sort(dim=-1).values
            preserved = (candidate[:, :, None] == reference[:, None, :]).any(dim=-1).sum(dim=-1)
            totals = self.router_totals[layer]
            totals["tokens"] += candidate.shape[0]
            totals["both_same"] += int((preserved == 2).sum().item())
            totals["one_changed"] += int((preserved == 1).sum().item())
            totals["both_changed"] += int((preserved == 0).sum().item())
            totals["exact"] += int((candidate_set == reference_set).all(dim=-1).sum().item())
            self.mark_seen("router", layer, slots, f"layer {layer} router")
        return hook

    def register(self, model: torch.nn.Module) -> list[Any]:
        handles: list[Any] = [model.model.layers[0].register_forward_pre_hook(self.decoder_input_hook)]
        for layer_index, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(self.decoder_output_hook(layer_index)))
            handles.append(layer.block_sparse_moe.register_forward_hook(self.router_hook(layer_index)))
        return handles

    def finalize(self) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        self.assert_complete()
        hidden, router = [], []
        for layer in range(self.num_layers):
            hidden_totals = self.hidden_totals[layer]
            router_totals = self.router_totals[layer]
            if hidden_totals["tokens"] != self.sample_plan.token_count:
                raise AssertionError(f"layer {layer}: hidden token count changed during forward")
            if router_totals["tokens"] != self.sample_plan.token_count:
                raise AssertionError(f"layer {layer}: router token count changed during forward")
            hidden.append(_hidden_metrics(hidden_totals))
            router.append(_router_metrics(router_totals))
        return hidden, router


def _hidden_metrics(totals: dict[str, float]) -> dict[str, float]:
    target_norm = max(totals["target_norm"], EPSILON)
    cosine_denominator = max(math.sqrt(totals["target_norm"] * totals["candidate_norm"]), EPSILON)
    return {
        "tokens": int(totals["tokens"]),
        "relative_l2": math.sqrt(totals["error"] / target_norm),
        "cosine": totals["dot"] / cosine_denominator,
    }


def _router_metrics(totals: dict[str, int]) -> dict[str, float | int]:
    tokens = totals["tokens"]
    if tokens <= 0:
        raise AssertionError("router metrics received no sampled tokens")
    both_same = totals["both_same"]
    one_changed = totals["one_changed"]
    both_changed = totals["both_changed"]
    if both_same + one_changed + both_changed != tokens:
        raise AssertionError("router overlap categories do not partition sampled tokens")
    return {
        "tokens": tokens,
        "exact_match_rate": totals["exact"] / tokens,
        "top2_overlap": (2 * both_same + one_changed) / (2 * tokens),
        "both_same_count": both_same,
        "one_changed_count": one_changed,
        "both_changed_count": both_changed,
        "both_same_fraction": both_same / tokens,
        "one_changed_fraction": one_changed / tokens,
        "both_changed_fraction": both_changed / tokens,
    }


@torch.no_grad()
def _run_forward(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    capture: _SampledHookCapture,
    input_id_batches: Optional[list[torch.Tensor]],
    description: str,
) -> Optional[list[torch.Tensor]]:
    """Run one model's real sequential forward while hooks inspect sampled rows."""
    handles = capture.register(model)
    captured_inputs: list[torch.Tensor] = []
    offset = 0
    try:
        model.eval()
        for batch_index, batch in enumerate(tqdm(dataloader, desc=description)):
            input_ids = batch["input_ids"].detach().cpu()
            if input_id_batches is None:
                captured_inputs.append(input_ids.clone())
            else:
                if batch_index >= len(input_id_batches) or not torch.equal(input_ids, input_id_batches[batch_index]):
                    raise AssertionError("all models must receive identical C4 input_ids in identical order")
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
    finally:
        for handle in handles:
            handle.remove()
    if offset != capture.sample_plan.total_tokens:
        raise AssertionError(
            f"calibration stream changed: expected {capture.sample_plan.total_tokens} tokens, saw {offset}"
        )
    return captured_inputs if input_id_batches is None else None


@torch.no_grad()
def collect_original_trajectory(model: torch.nn.Module, dataloader: Iterable[dict[str, torch.Tensor]], sample_plan: GlobalSamplePlan) -> OriginalTrajectory:
    """Run the uncompressed teacher and retain only sampled reference tensors."""
    capture = OriginalTrajectoryCapture(sample_plan, len(model.model.layers))
    input_ids = _run_forward(model, dataloader, capture, None, "[Sequential drift] original C4")
    trajectory = capture.finalize()
    trajectory.input_id_batches = input_ids or []
    return trajectory


@torch.no_grad()
def evaluate_candidate_trajectory(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    sample_plan: GlobalSamplePlan,
    reference: OriginalTrajectory,
    description: str,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Run Static or Static+Residual without ever injecting teacher state/routes."""
    capture = CandidateTrajectoryCapture(sample_plan, reference, len(model.model.layers))
    _run_forward(model, dataloader, capture, reference.input_id_batches, description)
    return capture.finalize()


def combine_metrics(
    static_hidden: list[dict[str, float]],
    static_router: list[dict[str, float]],
    residual_hidden: list[dict[str, float]],
    residual_router: list[dict[str, float]],
) -> dict[str, Any]:
    """Combine candidate results into the requested JSON-ready per-layer schema."""
    if not (len(static_hidden) == len(static_router) == len(residual_hidden) == len(residual_router)):
        raise AssertionError("all model trajectories must contain the same layer count")
    layers: dict[str, Any] = {}
    for index in range(len(static_hidden)):
        static_h, residual_h = static_hidden[index], residual_hidden[index]
        static_r, residual_r = static_router[index], residual_router[index]
        layers[str(index)] = {
            "hidden": {
                "tokens": static_h["tokens"],
                "static_relative_l2": static_h["relative_l2"],
                "residual_relative_l2": residual_h["relative_l2"],
                "static_cosine": static_h["cosine"],
                "residual_cosine": residual_h["cosine"],
            },
            "router": {
                "tokens": static_r["tokens"],
                "static_exact_match_rate": static_r["exact_match_rate"],
                "residual_exact_match_rate": residual_r["exact_match_rate"],
                "static_top2_overlap": static_r["top2_overlap"],
                "residual_top2_overlap": residual_r["top2_overlap"],
                **_prefixed_router_counts("static", static_r),
                **_prefixed_router_counts("residual", residual_r),
            },
        }
    return {"layers": layers}


def _prefixed_router_counts(prefix: str, values: dict[str, float]) -> dict[str, float | int]:
    return {
        f"{prefix}_both_same_count": values["both_same_count"],
        f"{prefix}_one_changed_count": values["one_changed_count"],
        f"{prefix}_both_changed_count": values["both_changed_count"],
        f"{prefix}_both_same_fraction": values["both_same_fraction"],
        f"{prefix}_one_changed_fraction": values["one_changed_fraction"],
        f"{prefix}_both_changed_fraction": values["both_changed_fraction"],
    }


def format_sequential_drift_summary(result: dict[str, Any]) -> str:
    """Render the compact console/text summary requested for this diagnostic."""
    lines = [
        "Layer | HiddenL2 Static | HiddenL2 Residual | RouteMatch Static | RouteMatch Residual | Overlap Static | Overlap Residual"
    ]
    layers = result["layers"]
    for layer_index in sorted(layers, key=int):
        hidden, router = layers[layer_index]["hidden"], layers[layer_index]["router"]
        lines.append(
            f"{int(layer_index):>5} | {hidden['static_relative_l2']:.6f} | {hidden['residual_relative_l2']:.6f} | "
            f"{router['static_exact_match_rate']:.6f} | {router['residual_exact_match_rate']:.6f} | "
            f"{router['static_top2_overlap']:.6f} | {router['residual_top2_overlap']:.6f}"
        )
    values = list(layers.values())
    final = values[-1]
    mean = lambda field, group: sum(value[group][field] for value in values) / len(values)
    lines.extend([
        "",
        "Final-layer hidden drift:",
        f"  Static   {final['hidden']['static_relative_l2']:.6f}",
        f"  Residual {final['hidden']['residual_relative_l2']:.6f}",
        "Mean hidden drift:",
        f"  Static   {mean('static_relative_l2', 'hidden'):.6f}",
        f"  Residual {mean('residual_relative_l2', 'hidden'):.6f}",
        "Mean router exact-match:",
        f"  Static   {mean('static_exact_match_rate', 'router'):.6f}",
        f"  Residual {mean('residual_exact_match_rate', 'router'):.6f}",
        "Mean router overlap:",
        f"  Static   {mean('static_top2_overlap', 'router'):.6f}",
        f"  Residual {mean('residual_top2_overlap', 'router'):.6f}",
    ])
    return "\n".join(lines)


def save_sequential_drift_plots(result: dict[str, Any], output_path: str | Path) -> list[Path]:
    """Save hidden drift, router match, and router overlap plots beside JSON."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    json_path = Path(output_path)
    output_dir = json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = result["layers"]
    indices = [int(index) for index in sorted(layers, key=int)]

    def plot(field: str, group: str, ylabel: str, filename: str) -> Path:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(indices, [layers[str(index)][group][f"static_{field}"] for index in indices], label="Static")
        axis.plot(indices, [layers[str(index)][group][f"residual_{field}"] for index in indices], label="Residual")
        axis.set_xlabel("Layer")
        axis.set_ylabel(ylabel)
        axis.set_xticks(indices)
        axis.legend()
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        return path

    return [
        plot("relative_l2", "hidden", "Relative L2 hidden-state drift vs Original", "sequential_hidden_drift.png"),
        plot("exact_match_rate", "router", "Top-2 exact match rate vs Original", "sequential_router_match.png"),
        plot("top2_overlap", "router", "Mean top-2 overlap vs Original", "sequential_router_overlap.png"),
    ]


# ---------------------------------------------------------------------------
# Vanilla HC-SMoE routing-drift analysis
# ---------------------------------------------------------------------------


@dataclass
class RoutingDriftTrace:
    """Compact CPU trace for sampled tokens or incremental decode steps."""

    hidden: torch.Tensor  # [layers, tokens, hidden]
    top2: torch.Tensor  # [layers, tokens, 2]
    margin: torch.Tensor  # [layers, tokens]
    input_id_batches: list[torch.Tensor] = field(default_factory=list)
    token_ids: Optional[torch.Tensor] = None


def freeze_for_analysis(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def router_weight_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    """Small CPU copy used to prove the merged checkpoint kept every router."""
    snapshots = []
    for layer_index, layer in enumerate(model.model.layers):
        gate = layer.block_sparse_moe.gate
        weight = gate.weight
        if weight.is_meta:
            hook = getattr(gate, "_hf_hook", None)
            weights_map = getattr(hook, "weights_map", None)
            if weights_map is None:
                raise RuntimeError(
                    f"layer {layer_index}: offloaded router weight has no Accelerate weights_map"
                )
            weight = weights_map["weight"]
        snapshots.append(weight.detach().cpu().clone())
    return snapshots


def assert_router_weights_unchanged(reference: list[torch.Tensor], model: torch.nn.Module) -> None:
    if len(reference) != len(model.model.layers):
        raise AssertionError("original and merged models have different layer counts")
    actual_weights = router_weight_snapshot(model)
    for layer_index, (expected, actual) in enumerate(zip(reference, actual_weights)):
        if not torch.equal(expected, actual):
            raise AssertionError(f"layer {layer_index}: merged checkpoint changed the original router weights")


class _FixedRoutingCapture:
    def __init__(self, plan: GlobalSamplePlan, num_layers: int) -> None:
        self.plan = plan
        self.num_layers = num_layers
        self.hidden: list[Optional[torch.Tensor]] = [None] * num_layers
        self.top2: list[Optional[torch.Tensor]] = [None] * num_layers
        self.margin: list[Optional[torch.Tensor]] = [None] * num_layers
        self.hidden_seen = [torch.zeros(plan.token_count, dtype=torch.bool) for _ in range(num_layers)]
        self.router_seen = [torch.zeros(plan.token_count, dtype=torch.bool) for _ in range(num_layers)]
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

    def moe_input_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            values = self._sample(inputs[0], f"layer {layer_index} MoE input")
            self.hidden[layer_index] = self._allocate(self.hidden[layer_index], self.plan, values)
            self.hidden[layer_index][self.slots] = values
            self.hidden_seen[layer_index][self.slots] = True
        return hook

    def router_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer_index}: expected MoE hidden output and router logits")
            logits = self._sample(output[1], f"layer {layer_index} router logits").float()
            top3 = torch.topk(logits, 3, dim=-1)
            ids = top3.indices[:, :TOP_K]
            margins = top3.values[:, 1] - top3.values[:, 2]
            self.top2[layer_index] = self._allocate(self.top2[layer_index], self.plan, ids)
            self.margin[layer_index] = self._allocate(self.margin[layer_index], self.plan, margins)
            self.top2[layer_index][self.slots] = ids
            self.margin[layer_index][self.slots] = margins
            self.router_seen[layer_index][self.slots] = True
        return hook

    def register(self, model: torch.nn.Module) -> list[Any]:
        handles = []
        for layer_index, layer in enumerate(model.model.layers):
            moe = layer.block_sparse_moe
            handles.append(moe.register_forward_pre_hook(self.moe_input_hook(layer_index)))
            handles.append(moe.register_forward_hook(self.router_hook(layer_index)))
        return handles

    def finalize(self, input_id_batches: list[torch.Tensor]) -> RoutingDriftTrace:
        for layer_index in range(self.num_layers):
            if not self.hidden_seen[layer_index].all() or not self.router_seen[layer_index].all():
                raise AssertionError(f"layer {layer_index}: fixed-input trace is incomplete")
        return RoutingDriftTrace(
            hidden=torch.stack([value for value in self.hidden if value is not None]),
            top2=torch.stack([value for value in self.top2 if value is not None]),
            margin=torch.stack([value for value in self.margin if value is not None]),
            input_id_batches=input_id_batches,
        )


@torch.inference_mode()
def collect_fixed_routing_trace(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    sample_plan: GlobalSamplePlan,
    expected_input_batches: Optional[list[torch.Tensor]] = None,
    description: str = "[Routing drift] fixed input",
) -> RoutingDriftTrace:
    """Run a real sequential model forward and retain only sampled token rows."""
    freeze_for_analysis(model)
    capture = _FixedRoutingCapture(sample_plan, len(model.model.layers))
    handles = capture.register(model)
    input_batches: list[torch.Tensor] = []
    offset = 0
    batch_count = 0
    try:
        for batch_index, batch in enumerate(tqdm(dataloader, desc=description)):
            input_ids = batch["input_ids"].detach().cpu()
            if expected_input_batches is None:
                input_batches.append(input_ids.clone())
            elif batch_index >= len(expected_input_batches) or not torch.equal(input_ids, expected_input_batches[batch_index]):
                raise AssertionError("original and merged models must receive identical fixed input_ids")
            batch_tokens = input_ids.numel()
            capture.begin_batch(offset, batch_tokens)
            inputs = {key: value.to(_forward_input_device(model)) for key, value in batch.items() if key != "labels"}
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


def _route_comparison(original_top2: torch.Tensor, merged_top2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if original_top2.shape != merged_top2.shape or original_top2.shape[-1] != TOP_K:
        raise AssertionError("routing traces must have matching [..., 2] shapes")
    original_set = original_top2.sort(dim=-1).values
    merged_set = merged_top2.sort(dim=-1).values
    exact = (original_set == merged_set).all(dim=-1)
    preserved = (merged_top2.unsqueeze(-1) == original_top2.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
    return exact, preserved.float() / TOP_K


def _optional_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    return float(values[mask].mean()) if mask.any() else None


def compare_fixed_routing_traces(
    original: RoutingDriftTrace,
    merged: RoutingDriftTrace,
    sample_positions: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Compute tokenwise and aggregate fixed-input metrics without group projection."""
    if len(original.input_id_batches) != len(merged.input_id_batches) or any(
        not torch.equal(left, right) for left, right in zip(original.input_id_batches, merged.input_id_batches)
    ):
        raise AssertionError("original and merged fixed input IDs differ")
    flattened_input_ids = torch.cat([batch.reshape(-1) for batch in original.input_id_batches])
    if sample_positions.numel() and int(sample_positions.max()) >= flattened_input_ids.numel():
        raise AssertionError("sample position falls outside the fixed input stream")
    exact, overlap = _route_comparison(original.top2, merged.top2)
    reference = original.hidden.float()
    candidate = merged.hidden.float()
    hidden_relative_l2 = torch.linalg.vector_norm(candidate - reference, dim=-1) / torch.linalg.vector_norm(
        reference, dim=-1
    ).clamp_min(EPSILON)
    layers: list[dict[str, Any]] = []
    for layer_index in range(original.top2.shape[0]):
        shifted = ~exact[layer_index]
        both_same = overlap[layer_index] == 1.0
        one_changed = overlap[layer_index] == 0.5
        both_changed = overlap[layer_index] == 0.0
        count = exact.shape[1]
        layers.append({
            "layer": layer_index,
            "tokens": count,
            "exact_top2_match_rate": float(exact[layer_index].float().mean()),
            "routing_shift_rate": float(shifted.float().mean()),
            "mean_top2_overlap": float(overlap[layer_index].mean()),
            "both_same_fraction": float(both_same.float().mean()),
            "one_changed_fraction": float(one_changed.float().mean()),
            "both_changed_fraction": float(both_changed.float().mean()),
            "mean_hidden_relative_l2": float(hidden_relative_l2[layer_index].mean()),
            "median_hidden_relative_l2": float(hidden_relative_l2[layer_index].median()),
            "original_margin_shifted": _optional_mean(original.margin[layer_index], shifted),
            "original_margin_non_shifted": _optional_mean(original.margin[layer_index], ~shifted),
            "merged_margin_shifted": _optional_mean(merged.margin[layer_index], shifted),
            "merged_margin_non_shifted": _optional_mean(merged.margin[layer_index], ~shifted),
        })
    if not exact[0].all():
        mismatches = int((~exact[0]).sum())
        raise AssertionError(f"first MoE layer routing differs for {mismatches} sampled tokens; router/input invariant failed")
    token_metrics = {
        "sample_positions": sample_positions.cpu(),
        "sampled_input_token_ids": flattened_input_ids.index_select(0, sample_positions.cpu()),
        "original_top2": original.top2.cpu(),
        "merged_top2": merged.top2.cpu(),
        "original_margin": original.margin.cpu(),
        "merged_margin": merged.margin.cpu(),
        "exact_top2_match": exact.cpu(),
        "top2_overlap": overlap.cpu(),
        "hidden_relative_l2": hidden_relative_l2.cpu(),
    }
    return layers, token_metrics


class _DecodeRoutingCapture:
    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.hidden: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.top2: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        self.margin: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

    def moe_input_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            rows = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if len(rows) != 1:
                raise AssertionError("decode trace must capture only the newly decoded token")
            self.hidden[layer_index].append(rows[0].cpu())
        return hook

    def router_hook(self, layer_index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(f"layer {layer_index}: expected MoE hidden output and router logits")
            logits = output[1].detach().reshape(-1, output[1].shape[-1]).float()
            if len(logits) != 1:
                raise AssertionError("decode router trace must contain one new token")
            top3 = torch.topk(logits[0], 3)
            self.top2[layer_index].append(top3.indices[:TOP_K].cpu())
            self.margin[layer_index].append((top3.values[1] - top3.values[2]).cpu())
        return hook

    def register(self, model: torch.nn.Module) -> list[Any]:
        handles = []
        for layer_index, layer in enumerate(model.model.layers):
            moe = layer.block_sparse_moe
            handles.append(moe.register_forward_pre_hook(self.moe_input_hook(layer_index)))
            handles.append(moe.register_forward_hook(self.router_hook(layer_index)))
        return handles

    def finalize(self, token_ids: list[int], decode_steps: int) -> RoutingDriftTrace:
        for layer_index in range(self.num_layers):
            if len(self.hidden[layer_index]) != decode_steps or len(self.top2[layer_index]) != decode_steps:
                raise AssertionError(f"layer {layer_index}: incomplete incremental decode trace")
        return RoutingDriftTrace(
            hidden=torch.stack([torch.stack(values) for values in self.hidden]),
            top2=torch.stack([torch.stack(values) for values in self.top2]),
            margin=torch.stack([torch.stack(values) for values in self.margin]),
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
    """Greedy or forced incremental decode with a model-owned KV cache."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("prompt_ids must have shape [1, prompt_length]")
    if decode_steps <= 0:
        raise ValueError("decode_steps must be positive")
    if forced_tokens is not None and forced_tokens.numel() != decode_steps:
        raise ValueError("forced token count must equal decode_steps")
    freeze_for_analysis(model)
    device = _forward_input_device(model)
    prompt = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt)
    output = model(input_ids=prompt, attention_mask=attention_mask, use_cache=True, return_dict=True)
    past_key_values = output.past_key_values
    next_logits = output.logits[:, -1]
    capture = _DecodeRoutingCapture(len(model.model.layers))
    handles = capture.register(model)
    generated: list[int] = []
    try:
        for step in tqdm(range(decode_steps), desc=f"[Routing drift] {description}"):
            if forced_tokens is None:
                token = next_logits.argmax(dim=-1).reshape(1, 1)
            else:
                token = forced_tokens[step].reshape(1, 1).to(device)
            generated.append(int(token.item()))
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)), dim=1)
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


def compare_decode_traces(original: RoutingDriftTrace, merged: RoutingDriftTrace, require_identical_tokens: bool) -> dict[str, torch.Tensor]:
    if original.token_ids is None or merged.token_ids is None:
        raise AssertionError("decode traces must include generated token IDs")
    token_equal = original.token_ids == merged.token_ids
    if require_identical_tokens and not token_equal.all():
        raise AssertionError("forced decode must use identical continuation token IDs")
    exact, overlap = _route_comparison(original.top2, merged.top2)
    hidden_relative_l2 = torch.linalg.vector_norm(merged.hidden.float() - original.hidden.float(), dim=-1) / torch.linalg.vector_norm(
        original.hidden.float(), dim=-1
    ).clamp_min(EPSILON)
    return {
        "original_token_ids": original.token_ids.cpu(),
        "merged_token_ids": merged.token_ids.cpu(),
        "token_equal": token_equal.cpu(),
        "original_top2": original.top2.cpu(),
        "merged_top2": merged.top2.cpu(),
        "routing_shift": (~exact).cpu(),
        "top2_overlap": overlap.cpu(),
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
    clean_routing_shift = any_routing_shift & ~post_token_divergence
    clean_indices = torch.nonzero(clean_routing_shift, as_tuple=False).flatten()
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
    overlap = metrics["top2_overlap"].float()
    hidden = metrics["hidden_relative_l2"].float()
    return {
        "mean_routing_shift_rate": float(shift.mean()),
        "mean_top2_overlap": float(overlap.mean()),
        "mean_hidden_relative_l2": float(hidden.mean()),
        "routing_shift_rate_by_step": shift.mean(dim=0).tolist(),
        "routing_shift_rate_by_layer": shift.mean(dim=1).tolist(),
    }


def save_routing_drift_plots(
    layer_metrics: list[dict[str, Any]],
    token_metrics: dict[str, torch.Tensor],
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

    def line_plot(values: list[float], ylabel: str, filename: str) -> None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(layers, values, marker="o")
        axis.set_xlabel("Mixtral layer")
        axis.set_ylabel(ylabel)
        axis.set_xticks(layers)
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    line_plot([row["routing_shift_rate"] for row in layer_metrics], "Routing shift rate", "routing_shift_by_layer.png")
    line_plot([row["mean_hidden_relative_l2"] for row in layer_metrics], "Mean hidden relative L2", "hidden_drift_by_layer.png")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(layers, [math.nan if row["original_margin_shifted"] is None else row["original_margin_shifted"] for row in layer_metrics], label="Original, shifted")
    axis.plot(layers, [math.nan if row["original_margin_non_shifted"] is None else row["original_margin_non_shifted"] for row in layer_metrics], label="Original, non-shifted")
    axis.plot(layers, [math.nan if row["merged_margin_shifted"] is None else row["merged_margin_shifted"] for row in layer_metrics], label="Merged, shifted", linestyle="--")
    axis.plot(layers, [math.nan if row["merged_margin_non_shifted"] is None else row["merged_margin_non_shifted"] for row in layer_metrics], label="Merged, non-shifted", linestyle="--")
    axis.set_xlabel("Mixtral layer")
    axis.set_ylabel("Top-2 / top-3 router margin")
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
        axis.set_ylabel("Mixtral layer")
        figure.colorbar(image, ax=axis, label=label)
        figure.tight_layout()
        path = output / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    heatmap(forced_metrics["routing_shift"], "Routing shift", "forced_routing_shift_heatmap.png", 0.0, 1.0)
    heatmap(forced_metrics["top2_overlap"], "Top-2 overlap", "forced_top2_overlap_heatmap.png", 0.0, 1.0)
    decode_steps = list(range(forced_metrics["routing_shift"].shape[1]))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(decode_steps, forced_metrics["routing_shift"].float().mean(dim=0).tolist(), marker="o")
    axis.set_xlabel("Decode step")
    axis.set_ylabel("Mean routing shift rate across layers")
    figure.tight_layout()
    path = output / "forced_shift_vs_decode_step.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    line_plot(forced_metrics["routing_shift"].float().mean(dim=1).tolist(), "Mean routing shift rate across decode steps", "forced_shift_vs_layer.png")
    return paths
