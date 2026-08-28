"""Calibration and FP32 training for Mixtral expert-specific residuals."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Mapping

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


@torch.no_grad()
def _expert_outputs_in_chunks(expert, hidden_states: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Run one frozen expert without accumulating selected tokens in VRAM."""
    if chunk_size <= 0:
        raise ValueError("--residual_batch_size must be positive")
    device, dtype = _expert_device(expert), expert.w1.weight.dtype
    outputs = []
    for start in range(0, hidden_states.shape[0], chunk_size):
        chunk = hidden_states[start:start + chunk_size].to(device=device, dtype=dtype)
        outputs.append(expert(chunk).detach().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def collect_residual_calibration(
    model,
    dataloader,
    group_state: Mapping[str, torch.Tensor],
    residual_data_limit: int,
    residual_batch_size: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Store only selected h, post-top-k g_i, and original E_i(h), all on CPU."""
    if residual_data_limit <= 0:
        raise ValueError("--residual_data_limit must be positive when residuals are enabled")
    groups = {name: group_members_from_labels(labels) for name, labels in group_state.items()}
    selected_members = {
        name: {expert for members in layer_groups.values() if len(members) > 1 for expert in members}
        for name, layer_groups in groups.items()
    }
    stored = defaultdict(lambda: {"hidden_states": [], "routing_weights": [], "original_outputs": []})
    counts = defaultdict(int)
    captured_inputs: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            captured_inputs[name] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        name = f"model.layers.{layer_idx}.block_sparse_moe"
        if selected_members.get(name):
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
            routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).detach().cpu()
            selected_experts = selected_experts.detach().cpu()
            inputs = captured_inputs[name]
            for expert_idx in selected_members[name]:
                remaining = residual_data_limit - counts[(name, expert_idx)]
                if remaining <= 0:
                    continue
                token_idx, route_idx = torch.where(selected_experts == expert_idx)
                if token_idx.numel() == 0:
                    continue
                token_idx, route_idx = token_idx[:remaining], route_idx[:remaining]
                selected_inputs = inputs[token_idx]
                key = f"{layer_idx}.{expert_idx}"
                stored[key]["hidden_states"].append(selected_inputs)
                stored[key]["routing_weights"].append(routing_weights[token_idx, route_idx])
                stored[key]["original_outputs"].append(
                    _expert_outputs_in_chunks(moe.experts[expert_idx], selected_inputs, residual_batch_size)
                )
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
def _reconstruction_metrics(static_expert, residual, data, indices, batch_size: int) -> Dict[str, float]:
    """Evaluate M_g and M_g + R_i against stored E_i(h), in bounded batches."""
    if indices.numel() == 0:
        return {"relative_l2": float("nan"), "cosine": float("nan"), "squared_error": 0.0, "target_squared_norm": 0.0, "dot": 0.0, "static_norm": 0.0, "target_norm": 0.0, "count": 0}
    device, static_dtype = _expert_device(static_expert), static_expert.w1.weight.dtype
    totals = defaultdict(float)
    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start:start + batch_size]
        static_input = data["hidden_states"][batch_indices].to(device=device, dtype=static_dtype)
        static = static_expert(static_input).float()
        original = data["original_outputs"][batch_indices].to(device=device, dtype=torch.float32)
        residual_input = data["hidden_states"][batch_indices].to(device=device, dtype=torch.float32)
        pred = static + residual(residual_input)
        error, static_error = pred - original, static - original
        totals["squared_error"] += error.square().sum().item()
        totals["static_squared_error"] += static_error.square().sum().item()
        totals["target_squared_norm"] += original.square().sum().item()
        totals["dot"] += (pred * original).sum().item()
        totals["static_dot"] += (static * original).sum().item()
        totals["pred_norm"] += pred.square().sum().item()
        totals["static_norm"] += static.square().sum().item()
        totals["target_norm"] += original.square().sum().item()
    target_norm = max(totals["target_squared_norm"], FP32_EPS)
    return {
        "relative_l2": (totals["squared_error"] / target_norm) ** 0.5,
        "cosine": totals["dot"] / max((totals["pred_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        "static_relative_l2": (totals["static_squared_error"] / target_norm) ** 0.5,
        "static_cosine": totals["static_dot"] / max((totals["static_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        **totals,
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
    """Freeze static M_g and train each R_i in FP32, one residual at a time."""
    if residual_epochs <= 0:
        raise ValueError("--residual_epochs must be positive")
    attach_residual_experts(model, group_state, residual_width)
    model.requires_grad_(False)
    metrics: Dict[str, object] = {"experts": {}, "aggregate": {}}
    totals = defaultdict(float)
    for key, data in tqdm(calibration.items(), desc="[Residual] training experts"):
        layer_idx, expert_idx = (int(value) for value in key.split("."))
        moe = model.model.layers[layer_idx].block_sparse_moe
        residual, static_expert = moe.residual_experts[str(expert_idx)], moe.experts[expert_idx]
        device, static_dtype = _expert_device(static_expert), static_expert.w1.weight.dtype
        residual.to(device=device, dtype=torch.float32).requires_grad_(True)
        train_idx, val_idx = _split_indices(data["hidden_states"].shape[0], residual_val_ratio, seed + layer_idx * 1000 + expert_idx)
        optimizer = torch.optim.AdamW(residual.parameters(), lr=residual_lr)
        best_state, best_val, stale_steps = None, float("inf"), 0
        for epoch in range(residual_epochs):
            residual.train()
            permutation = train_idx[torch.randperm(train_idx.numel(), generator=torch.Generator().manual_seed(seed + epoch + layer_idx * 1000 + expert_idx))]
            for start in range(0, permutation.numel(), residual_batch_size):
                indices = permutation[start:start + residual_batch_size]
                static_input = data["hidden_states"][indices].to(device=device, dtype=static_dtype)
                with torch.no_grad():
                    static = static_expert(static_input).float()
                original = data["original_outputs"][indices].to(device=device, dtype=torch.float32)
                residual_input = data["hidden_states"][indices].to(device=device, dtype=torch.float32)
                gate = data["routing_weights"][indices].to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                loss = (gate.square().unsqueeze(-1) * (residual(residual_input) - (original - static)).square()).mean()
                loss.backward()
                optimizer.step()
            residual.eval()
            if val_idx.numel():
                with torch.no_grad():
                    static_input = data["hidden_states"][val_idx].to(device=device, dtype=static_dtype)
                    static = static_expert(static_input).float()
                    original = data["original_outputs"][val_idx].to(device=device, dtype=torch.float32)
                    residual_input = data["hidden_states"][val_idx].to(device=device, dtype=torch.float32)
                    gate = data["routing_weights"][val_idx].to(device=device, dtype=torch.float32)
                    val_loss = (gate.square().unsqueeze(-1) * (residual(residual_input) - (original - static)).square()).mean().item()
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
        result = _reconstruction_metrics(static_expert, residual, data, heldout, residual_batch_size)
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
    original_expert_params = static_expert_params = residual_params = 0
    seen = set()
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
