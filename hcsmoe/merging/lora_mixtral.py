"""FP32 training for original-expert-specific weight-space Mixtral LoRA."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Mapping

import torch
from tqdm import tqdm

from hcsmoe.merging.residual_mixtral import FP32_EPS, _expert_device, _split_indices
from hcsmoe.models.mixtral.utils import (
    attach_lora_experts,
    lora_expert_output,
    lora_params_per_expert,
    lora_state_dict,
)


@torch.no_grad()
def _lora_reconstruction_metrics(static_expert, adapter, data, indices, batch_size: int) -> Dict[str, float]:
    """Evaluate static M_g and LoRA-adjusted M_g against stored original outputs."""
    if indices.numel() == 0:
        return {
            "static_relative_l2": float("nan"), "lora_relative_l2": float("nan"),
            "static_cosine": float("nan"), "lora_cosine": float("nan"),
            "weighted_static_relative_l2": float("nan"), "weighted_lora_relative_l2": float("nan"),
            "static_squared_error": 0.0, "lora_squared_error": 0.0, "target_squared_norm": 0.0,
            "weighted_static_squared_error": 0.0, "weighted_lora_squared_error": 0.0,
            "weighted_target_squared_norm": 0.0, "static_dot": 0.0, "lora_dot": 0.0,
            "static_norm": 0.0, "lora_norm": 0.0, "target_norm": 0.0, "count": 0,
        }
    device, static_dtype = _expert_device(static_expert), static_expert.w1.weight.dtype
    totals = defaultdict(float)
    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start:start + batch_size]
        static_input = data["hidden_states"][batch_indices].to(device=device, dtype=static_dtype)
        static = static_expert(static_input).float()
        lora_input = data["hidden_states"][batch_indices].to(device=device, dtype=torch.float32)
        lora = lora_expert_output(static_expert, adapter, lora_input).float()
        original = data["original_outputs"][batch_indices].to(device=device, dtype=torch.float32)
        gate = data["routing_weights"][batch_indices].to(device=device, dtype=torch.float32)
        weight = gate.square().unsqueeze(-1)
        static_error, lora_error = static - original, lora - original
        totals["static_squared_error"] += static_error.square().sum().item()
        totals["lora_squared_error"] += lora_error.square().sum().item()
        totals["target_squared_norm"] += original.square().sum().item()
        totals["weighted_static_squared_error"] += (weight * static_error.square()).sum().item()
        totals["weighted_lora_squared_error"] += (weight * lora_error.square()).sum().item()
        totals["weighted_target_squared_norm"] += (weight * original.square()).sum().item()
        totals["static_dot"] += (static * original).sum().item()
        totals["lora_dot"] += (lora * original).sum().item()
        totals["static_norm"] += static.square().sum().item()
        totals["lora_norm"] += lora.square().sum().item()
        totals["target_norm"] += original.square().sum().item()
    target_norm = max(totals["target_squared_norm"], FP32_EPS)
    weighted_target_norm = max(totals["weighted_target_squared_norm"], FP32_EPS)
    return {
        "static_relative_l2": (totals["static_squared_error"] / target_norm) ** 0.5,
        "lora_relative_l2": (totals["lora_squared_error"] / target_norm) ** 0.5,
        "weighted_static_relative_l2": (totals["weighted_static_squared_error"] / weighted_target_norm) ** 0.5,
        "weighted_lora_relative_l2": (totals["weighted_lora_squared_error"] / weighted_target_norm) ** 0.5,
        "static_cosine": totals["static_dot"] / max((totals["static_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        "lora_cosine": totals["lora_dot"] / max((totals["lora_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        **totals,
        "count": int(indices.numel()),
    }


@torch.no_grad()
def _lora_validation_loss(static_expert, adapter, data, indices, batch_size: int) -> float:
    if indices.numel() == 0:
        return 0.0
    device = _expert_device(static_expert)
    total, element_count = 0.0, 0
    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start:start + batch_size]
        hidden_states = data["hidden_states"][batch_indices].to(device=device, dtype=torch.float32)
        original = data["original_outputs"][batch_indices].to(device=device, dtype=torch.float32)
        gate = data["routing_weights"][batch_indices].to(device=device, dtype=torch.float32)
        error = lora_expert_output(static_expert, adapter, hidden_states) - original
        weighted_error = gate.square().unsqueeze(-1) * error.square()
        total += weighted_error.sum().item()
        element_count += weighted_error.numel()
    return total / element_count


def save_lora_loss_curves(output_path: str, metrics: Mapping[str, object]) -> None:
    """Save log-scale epoch train/validation loss curves for LoRA experts."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    curve_dir = os.path.join(output_path, "lora_loss_curves")
    os.makedirs(curve_dir, exist_ok=True)

    def plot_value(value: float) -> float:
        return max(float(value), FP32_EPS)

    experts = metrics["experts"]
    for key, result in experts.items():
        epochs = result["epochs"]
        figure, axis = plt.subplots(figsize=(7, 4.5))
        trained = [item for item in epochs if item["train_loss"] is not None]
        if trained:
            axis.plot(
                [item["epoch"] for item in trained],
                [plot_value(item["train_loss"]) for item in trained],
                marker="o", label="Train loss",
            )
        axis.plot(
            [item["epoch"] for item in epochs],
            [plot_value(item["validation_loss"]) for item in epochs],
            marker="o", label="Validation loss",
        )
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Gate-weighted MSE loss")
        axis.set_title(f"LoRA Training Loss — Layer {result['layer']} Expert {result['expert']}")
        axis.legend()
        figure.tight_layout()
        path = os.path.join(curve_dir, f"layer_{result['layer']}_expert_{result['expert']}.png")
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"[LoRA] Saved loss curve: {path}")

    figure, axis = plt.subplots(figsize=(10, 5.5))
    for key, result in experts.items():
        epochs = result["epochs"]
        axis.plot(
            [item["epoch"] for item in epochs],
            [plot_value(item["validation_loss"]) for item in epochs],
            marker="o", label=f"L{result['layer']}-E{result['expert']}",
        )
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Gate-weighted validation MSE loss")
    axis.set_title("LoRA Validation Loss — All Adapted Experts")
    if experts:
        axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    path = os.path.join(curve_dir, "all_lora_experts.png")
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"[LoRA] Saved loss curve: {path}")


def train_lora_experts(
    model,
    group_state: Mapping[str, torch.Tensor],
    calibration: Mapping[str, Mapping[str, torch.Tensor]],
    lora_rank: int,
    lora_alpha: float,
    lora_epochs: int,
    lora_lr: float,
    lora_batch_size: int,
    lora_val_ratio: float,
    lora_patience: int,
    seed: int,
) -> Dict[str, object]:
    """Freeze the static model and train one FP32 original-expert LoRA at a time."""
    if lora_rank <= 0:
        raise ValueError("--lora_rank must be positive when LoRA is enabled")
    if lora_epochs <= 0:
        raise ValueError("--lora_epochs must be positive")
    attach_lora_experts(model, group_state, lora_rank, lora_alpha)
    model.requires_grad_(False)
    print(f"[LoRA] rank={lora_rank} alpha={lora_alpha:g} scale={lora_alpha / lora_rank:g}")
    metrics: Dict[str, object] = {"experts": {}, "aggregate": {}}
    totals = defaultdict(float)
    for key, data in tqdm(calibration.items(), desc="[LoRA] training experts"):
        layer_idx, expert_idx = (int(value) for value in key.split("."))
        moe = model.model.layers[layer_idx].block_sparse_moe
        adapter, static_expert = moe.lora_experts[str(expert_idx)], moe.experts[expert_idx]
        device = _expert_device(static_expert)
        adapter.to(device=device, dtype=torch.float32).requires_grad_(True)
        train_idx, val_idx = _split_indices(data["hidden_states"].shape[0], lora_val_ratio, seed + layer_idx * 1000 + expert_idx)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lora_lr)
        epoch0_loss = _lora_validation_loss(static_expert, adapter, data, val_idx, lora_batch_size)
        epoch_history = [{"epoch": 0, "train_loss": None, "validation_loss": epoch0_loss}]
        best_state, best_val, best_epoch, stale_steps = None, float("inf"), None, 0
        for epoch in range(lora_epochs):
            adapter.train()
            permutation = train_idx[torch.randperm(train_idx.numel(), generator=torch.Generator().manual_seed(seed + epoch + layer_idx * 1000 + expert_idx))]
            train_loss_sum, train_sample_count = 0.0, 0
            for start in range(0, permutation.numel(), lora_batch_size):
                indices = permutation[start:start + lora_batch_size]
                hidden_states = data["hidden_states"][indices].to(device=device, dtype=torch.float32)
                original = data["original_outputs"][indices].to(device=device, dtype=torch.float32)
                gate = data["routing_weights"][indices].to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                prediction = lora_expert_output(static_expert, adapter, hidden_states)
                loss = (gate.square().unsqueeze(-1) * (prediction - original).square()).mean()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.detach().item() * indices.numel()
                train_sample_count += indices.numel()
            adapter.eval()
            val_loss = _lora_validation_loss(static_expert, adapter, data, val_idx, lora_batch_size)
            improved = val_loss < best_val
            if improved:
                best_val, best_epoch, stale_steps = val_loss, epoch + 1, 0
                best_state = {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}
            epoch_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss_sum / train_sample_count if train_sample_count else None,
                "validation_loss": val_loss,
            })
            print(
                f"[LoRA] layer={layer_idx} expert={expert_idx} epoch={epoch + 1} "
                f"train_loss={epoch_history[-1]['train_loss']:.6g} val_loss={val_loss:.6g}"
            )
            if not improved:
                stale_steps += 1
                if stale_steps >= lora_patience:
                    break
        adapter.load_state_dict(best_state, strict=True)
        adapter.eval()
        heldout = val_idx if val_idx.numel() else train_idx
        result = _lora_reconstruction_metrics(static_expert, adapter, data, heldout, lora_batch_size)
        result.update({
            "layer": layer_idx,
            "expert": expert_idx,
            "group": int(group_state[f"model.layers.{layer_idx}.block_sparse_moe"][expert_idx]),
            "group_size": int(moe.lora_group_sizes[expert_idx]),
            "training_samples": int(train_idx.numel()),
            "validation_samples": int(val_idx.numel()),
            "epochs": epoch_history,
            "epoch0_validation_loss": epoch0_loss,
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
        })
        print(
            "[LoRA] layer={layer} group={group} expert={expert} size={group_size} samples={training_samples} "
            "static_rel_l2={static_relative_l2:.6f} lora_rel_l2={lora_relative_l2:.6f} "
            "static_cos={static_cosine:.6f} lora_cos={lora_cosine:.6f}".format(**result)
        )
        metrics["experts"][key] = result
        for field in (
            "static_squared_error", "lora_squared_error", "target_squared_norm",
            "weighted_static_squared_error", "weighted_lora_squared_error", "weighted_target_squared_norm",
            "static_dot", "lora_dot", "static_norm", "lora_norm", "target_norm", "count",
        ):
            totals[field] += result[field]
        adapter.to("cpu")
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    target_norm = max(totals["target_squared_norm"], FP32_EPS)
    metrics["aggregate"] = {
        "static_hcsmoe_reconstruction_error": (totals["static_squared_error"] / target_norm) ** 0.5,
        "lora_reconstruction_error": (totals["lora_squared_error"] / target_norm) ** 0.5,
        "static_hcsmoe_cosine": totals["static_dot"] / max((totals["static_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        "lora_cosine": totals["lora_dot"] / max((totals["lora_norm"] * totals["target_norm"]) ** 0.5, FP32_EPS),
        "heldout_tokens": int(totals["count"]),
    }
    original_expert_params = static_expert_params = lora_params = adapted_experts = 0
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
        for adapter in getattr(moe, "lora_experts", {}).values():
            lora_params += sum(parameter.numel() for parameter in adapter.parameters())
            adapted_experts += 1
    if adapted_experts:
        expected = lora_params_per_expert(model.config.hidden_size, model.config.intermediate_size, lora_rank)
        if lora_params // adapted_experts != expected:
            raise AssertionError("Unexpected LoRA parameter count per adapted expert")
    metrics["aggregate"].update({
        "lora_params": lora_params,
        "lora_params_per_adapted_expert": lora_params // adapted_experts if adapted_experts else 0,
        "lora_params_percent_of_original_experts": 100.0 * lora_params / original_expert_params,
        "logical_total_expert_parameter_ratio_after_compression": 100.0 * (static_expert_params + lora_params) / original_expert_params,
    })
    print("[LoRA] params/adapted expert:\n  " + f"{metrics['aggregate']['lora_params_per_adapted_expert']:,}")
    print("[LoRA] aggregate=" + json.dumps(metrics["aggregate"], sort_keys=True))
    return metrics


def save_lora_artifacts(output_path: str, model, lora_rank: int, lora_alpha: float, metrics: Mapping[str, object], config: Mapping[str, object]) -> None:
    """Save LoRA-only weights and reconstruction metrics beside the static checkpoint."""
    os.makedirs(output_path, exist_ok=True)
    torch.save(lora_state_dict(model, lora_rank, lora_alpha), os.path.join(output_path, "lora.pth"))
    with open(os.path.join(output_path, "lora_config.json"), "w") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True)
    with open(os.path.join(output_path, "lora_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
