"""Calibration and training for Mixtral expert-specific residuals.

The collector intentionally runs before frequency merging.  It keeps only the
selected tokens (and their target deltas) on CPU, so it never needs a second
full Mixtral model or a full-model deepcopy.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from hcsmoe.models.mixtral.utils import (
    attach_residual_experts,
    group_members_from_labels,
    residual_state_dict,
)

FP32_EPS = 1e-7


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _expert_device(expert: torch.nn.Module) -> torch.device:
    return next(expert.parameters()).device


def _frequency_merged_output(moe, members: Sequence[int], usage: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply exactly the weights produced by merge=freq, without mutating moe."""
    expert0 = moe.experts[members[0]]
    device, dtype = _expert_device(expert0), expert0.w1.weight.dtype
    hidden_states = hidden_states.to(device=device, dtype=dtype)
    # Keep usage in its original (normally fp32) dtype.  This mirrors the
    # promotion and final bf16 copy performed by merge=freq exactly.
    weights = usage[members].to(device=device)
    denominator = weights.sum() + FP32_EPS

    def merged_weight(name: str) -> torch.Tensor:
        # Keep stack -> sum order identical to
        # _merge_mlp_experts_by_usage_frequency_weighting.
        stacked = torch.stack(
            [getattr(moe.experts[idx], name).weight * weights[pos] for pos, idx in enumerate(members)],
            dim=0,
        )
        return torch.sum(stacked, dim=0) / denominator

    w1, w2, w3 = (merged_weight(name).to(dtype=dtype) for name in ("w1", "w2", "w3"))
    return F.linear(F.silu(F.linear(hidden_states, w1)) * F.linear(hidden_states, w3), w2)


@torch.no_grad()
def collect_residual_calibration(
    model,
    dataloader,
    group_state: Mapping[str, torch.Tensor],
    usage_frequency: Mapping[str, torch.Tensor],
    residual_data_limit: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect selected C4 inputs, renormalized gates, and E_i-M_g deltas on CPU."""
    if residual_data_limit <= 0:
        raise ValueError("--residual_data_limit must be positive when residuals are enabled")
    groups = {name: group_members_from_labels(labels) for name, labels in group_state.items()}
    selected_members = {
        name: {expert for members in layer_groups.values() if len(members) > 1 for expert in members}
        for name, layer_groups in groups.items()
    }
    stored = defaultdict(lambda: {"hidden_states": [], "routing_weights": [], "target_delta": []})
    counts = defaultdict(int)
    captured_inputs: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            captured_inputs[name] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        name = f"model.layers.{layer_idx}.block_sparse_moe"
        if name in selected_members and selected_members[name]:
            handles.append(layer.block_sparse_moe.register_forward_pre_hook(make_hook(name)))

    model.eval()
    for batch in tqdm(dataloader, desc="[Residual] collecting C4 calibration"):
        captured_inputs.clear()
        batch = {key: value.to(_input_device(model)) for key, value in batch.items()}
        outputs = model(**batch, output_router_logits=True)
        for layer_idx, router_logits in enumerate(outputs.router_logits):
            name = f"model.layers.{layer_idx}.block_sparse_moe"
            if name not in captured_inputs or not selected_members.get(name):
                continue
            moe = model.model.layers[layer_idx].block_sparse_moe
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, moe.top_k, dim=-1)
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            inputs = captured_inputs[name]
            for expert_idx in selected_members[name]:
                remaining = residual_data_limit - counts[(name, expert_idx)]
                if remaining <= 0:
                    continue
                token_idx, route_idx = torch.where(selected_experts.detach().cpu() == expert_idx)
                if token_idx.numel() == 0:
                    continue
                token_idx, route_idx = token_idx[:remaining], route_idx[:remaining]
                selected_inputs = inputs[token_idx]
                group_label = int(group_state[name][expert_idx].item())
                members = groups[name][group_label]
                target_device = _expert_device(moe.experts[expert_idx])
                original = moe.experts[expert_idx](selected_inputs.to(target_device, dtype=moe.experts[expert_idx].w1.weight.dtype))
                static = _frequency_merged_output(moe, members, usage_frequency[name], selected_inputs)
                key = f"{layer_idx}.{expert_idx}"
                stored[key]["hidden_states"].append(selected_inputs.cpu())
                stored[key]["routing_weights"].append(routing_weights.detach().cpu()[token_idx, route_idx])
                stored[key]["target_delta"].append((original - static).detach().cpu())
                counts[(name, expert_idx)] += token_idx.numel()
        del outputs
        if all(counts[(name, expert)] >= residual_data_limit for name, experts in selected_members.items() for expert in experts):
            break
    for handle in handles:
        handle.remove()
    return {
        key: {field: torch.cat(values, dim=0) for field, values in fields.items()}
        for key, fields in stored.items()
    }


def _split_indices(num_samples: int, val_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if num_samples < 2:
        return torch.arange(num_samples), torch.empty(0, dtype=torch.long)
    val_count = min(max(1, int(round(num_samples * val_ratio))), num_samples - 1)
    permutation = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
    return permutation[val_count:], permutation[:val_count]


@torch.no_grad()
def _reconstruction_metrics(static_expert, residual, data: Mapping[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, float]:
    if indices.numel() == 0:
        return {"relative_l2": float("nan"), "cosine": float("nan"), "squared_error": 0.0, "target_squared_norm": 0.0, "dot": 0.0, "static_norm": 0.0, "target_norm": 0.0, "count": 0}
    device, dtype = _expert_device(static_expert), static_expert.w1.weight.dtype
    hidden_states = data["hidden_states"][indices].to(device=device, dtype=dtype)
    target_delta = data["target_delta"][indices].to(device=device, dtype=dtype)
    static = static_expert(hidden_states).float()
    target = static + target_delta.float()
    pred = static + residual(hidden_states).float()
    error = pred - target
    static_error = static - target
    return {
        "relative_l2": (torch.linalg.vector_norm(error) / torch.linalg.vector_norm(target).clamp_min(FP32_EPS)).item(),
        "cosine": F.cosine_similarity(pred.flatten(), target.flatten(), dim=0).item(),
        "static_relative_l2": (torch.linalg.vector_norm(static_error) / torch.linalg.vector_norm(target).clamp_min(FP32_EPS)).item(),
        "static_cosine": F.cosine_similarity(static.flatten(), target.flatten(), dim=0).item(),
        "squared_error": error.square().sum().item(),
        "static_squared_error": static_error.square().sum().item(),
        "target_squared_norm": target.square().sum().item(),
        "dot": (pred * target).sum().item(),
        "static_dot": (static * target).sum().item(),
        "pred_norm": pred.square().sum().item(),
        "static_norm": static.square().sum().item(),
        "target_norm": target.square().sum().item(),
        "count": int(indices.numel()),
    }


def train_residuals(
    model,
    group_state: Mapping[str, torch.Tensor],
    calibration: Mapping[str, Mapping[str, torch.Tensor]],
    residual_width: int,
    residual_epochs: int,
    residual_lr: float,
    residual_batch_size: int,
    residual_val_ratio: float,
    residual_patience: int,
    seed: int,
) -> Dict[str, object]:
    """Freeze the static model and optimize one CPU-buffered residual at a time."""
    attach_residual_experts(model, group_state, residual_width)
    model.requires_grad_(False)
    metrics: Dict[str, object] = {"experts": {}, "aggregate": {}}
    totals = defaultdict(float)
    for key, data in tqdm(calibration.items(), desc="[Residual] training experts"):
        layer_idx, expert_idx = (int(value) for value in key.split("."))
        moe = model.model.layers[layer_idx].block_sparse_moe
        residual = moe.residual_experts[str(expert_idx)]
        static_expert = moe.experts[expert_idx]
        device, dtype = _expert_device(static_expert), static_expert.w1.weight.dtype
        residual.to(device=device, dtype=dtype).requires_grad_(True)
        train_idx, val_idx = _split_indices(data["hidden_states"].shape[0], residual_val_ratio, seed + layer_idx * 1000 + expert_idx)
        optimizer = torch.optim.AdamW(residual.parameters(), lr=residual_lr)
        best_state, best_val, stale_steps = None, float("inf"), 0
        for _epoch in range(residual_epochs):
            residual.train()
            permutation = train_idx[torch.randperm(train_idx.numel(), generator=torch.Generator().manual_seed(seed + _epoch + layer_idx * 1000 + expert_idx))]
            for start in range(0, permutation.numel(), residual_batch_size):
                indices = permutation[start:start + residual_batch_size]
                h = data["hidden_states"][indices].to(device=device, dtype=dtype)
                target = data["target_delta"][indices].to(device=device, dtype=dtype)
                gate = data["routing_weights"][indices].to(device=device, dtype=dtype)
                optimizer.zero_grad(set_to_none=True)
                loss = (gate.square().unsqueeze(-1) * (residual(h) - target).float().square()).mean()
                loss.backward()
                optimizer.step()
            residual.eval()
            if val_idx.numel():
                with torch.no_grad():
                    h = data["hidden_states"][val_idx].to(device=device, dtype=dtype)
                    target = data["target_delta"][val_idx].to(device=device, dtype=dtype)
                    gate = data["routing_weights"][val_idx].to(device=device, dtype=dtype)
                    val_loss = (gate.square().unsqueeze(-1) * (residual(h) - target).float().square()).mean().item()
            else:
                val_loss = 0.0
            if val_loss < best_val:
                best_val, stale_steps = val_loss, 0
                best_state = {name: value.detach().cpu().clone() for name, value in residual.state_dict().items()}
            else:
                stale_steps += 1
                if stale_steps >= residual_patience:
                    break
        residual.load_state_dict(best_state, strict=True)
        residual.eval()
        heldout = val_idx if val_idx.numel() else train_idx
        result = _reconstruction_metrics(static_expert, residual, data, heldout)
        result.update({
            "layer": layer_idx,
            "expert": expert_idx,
            "group": int(group_state[f"model.layers.{layer_idx}.block_sparse_moe"][expert_idx]),
            "group_size": int(moe.residual_group_sizes[expert_idx]),
            "training_samples": int(train_idx.numel()),
            "validation_samples": int(val_idx.numel()),
            "best_validation_loss": best_val,
        })
        print("[Residual] layer={layer} group={group} expert={expert} size={group_size} samples={training_samples} static_rel_l2={static_relative_l2:.6f} static_cos={static_cosine:.6f} residual_rel_l2={relative_l2:.6f} residual_cos={cosine:.6f}".format(**result))
        metrics["experts"][key] = result
        for field in ("squared_error", "static_squared_error", "target_squared_norm", "dot", "static_dot", "pred_norm", "static_norm", "target_norm", "count"):
            totals[field] += result[field]
        residual.to("cpu")
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_norm = max(totals["target_squared_norm"], FP32_EPS)
    metrics["aggregate"] = {
        "static_hcsmoe_reconstruction_error": (totals["static_squared_error"] / target_norm) ** 0.5,
        "residual_reconstruction_error": (totals["squared_error"] / target_norm) ** 0.5,
        "static_hcsmoe_cosine": totals["static_dot"] / max((totals["static_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        "residual_cosine": totals["dot"] / max((totals["pred_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        "heldout_tokens": int(totals["count"]),
    }
    original_expert_params = 0
    static_expert_params = 0
    seen = set()
    residual_params = 0
    for layer in model.model.layers:
        moe = layer.block_sparse_moe
        per_expert = sum(parameter.numel() for parameter in moe.experts[0].parameters())
        original_expert_params += moe.num_experts * per_expert
        for expert in moe.experts:
            for parameter in expert.parameters():
                if id(parameter) not in seen:
                    static_expert_params += parameter.numel()
                    seen.add(id(parameter))
        for residual in getattr(moe, "residual_experts", {}).values():
            residual_params += sum(parameter.numel() for parameter in residual.parameters())
    metrics["aggregate"].update({
        "residual_parameter_count": residual_params,
        "residual_params_percent_of_original_experts": 100.0 * residual_params / original_expert_params,
        "logical_total_expert_parameter_ratio_after_compression": 100.0 * (static_expert_params + residual_params) / original_expert_params,
    })
    print("[Residual] aggregate=" + json.dumps(metrics["aggregate"], sort_keys=True))
    return metrics


def save_residual_artifacts(output_path: str, model, residual_width: int, metrics: Mapping[str, object], config: Mapping[str, object]) -> None:
    os.makedirs(output_path, exist_ok=True)
    torch.save(residual_state_dict(model, residual_width), os.path.join(output_path, "residuals.pth"))
    with open(os.path.join(output_path, "residual_config.json"), "w") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True)
    with open(os.path.join(output_path, "residual_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
