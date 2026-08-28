import inspect
import math
import types
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers.activations import ACT2FN


class TinySwiGLUResidual(nn.Module):
    """A small, bias-free SwiGLU residual for one original Mixtral expert."""

    def __init__(self, hidden_size: int, residual_width: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_size, residual_width, bias=False)
        self.w2 = nn.Linear(residual_width, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, residual_width, bias=False)
        # A newly attached residual is exactly a no-op.
        nn.init.zeros_(self.w2.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


def residual_aware_moe_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mixtral MoE forward with residuals keyed by original expert indices."""
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and getattr(self, "jitter_noise", 0.0) > 0:
        hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
            1.0 - self.jitter_noise, 1.0 + self.jitter_noise
        )
    flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
    router_logits = self.gate(flat_hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(flat_hidden_states.dtype)
    final_hidden_states = torch.zeros_like(flat_hidden_states)
    expert_mask = F.one_hot(selected_experts, num_classes=self.gate_num_experts).permute(2, 1, 0)

    for expert_idx in range(self.num_experts):
        route_idx, token_idx = torch.where(expert_mask[expert_idx])
        if token_idx.numel() == 0:
            continue
        current_state = flat_hidden_states[token_idx]
        # Frequency merging aliases every member of a group to its static M_g.
        current_hidden_states = self.experts[expert_idx](current_state)
        residual = self.residual_experts[str(expert_idx)] if str(expert_idx) in self.residual_experts else None
        if residual is not None:
            if next(residual.parameters()).device != current_state.device:
                residual.to(device=current_state.device, dtype=current_state.dtype)
            current_hidden_states = current_hidden_states + residual(current_state)
        current_hidden_states = current_hidden_states * routing_weights[token_idx, route_idx, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(flat_hidden_states.dtype))
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


def group_members_from_labels(group_labels: torch.Tensor) -> Dict[int, List[int]]:
    """Return all members for every label; labels need not be contiguous."""
    members: Dict[int, List[int]] = {}
    for expert_idx, label in enumerate(group_labels.detach().cpu().tolist()):
        members.setdefault(int(label), []).append(expert_idx)
    return members


def bind_shared_experts_from_group_state(model, group_state: Mapping[str, torch.Tensor]) -> None:
    """Rebind static merged experts before loading a frequency-merged checkpoint."""
    for layer_idx, layer in enumerate(model.model.layers):
        name = f"model.layers.{layer_idx}.block_sparse_moe"
        if name not in group_state:
            continue
        for members in group_members_from_labels(group_state[name]).values():
            representative = members[0]
            for expert_idx in members[1:]:
                layer.block_sparse_moe.experts[expert_idx] = layer.block_sparse_moe.experts[representative]


def attach_residual_experts(model, group_state: Mapping[str, torch.Tensor], residual_width: int) -> None:
    """Attach CPU-resident residuals and install the residual-aware forward."""
    if residual_width <= 0:
        return
    for layer_idx, layer in enumerate(model.model.layers):
        moe = layer.block_sparse_moe
        name = f"model.layers.{layer_idx}.block_sparse_moe"
        if name not in group_state:
            continue
        groups = group_members_from_labels(group_state[name])
        residuals = nn.ModuleDict()
        for members in groups.values():
            if len(members) > 1:
                for expert_idx in members:
                    residuals[str(expert_idx)] = TinySwiGLUResidual(moe.hidden_dim, residual_width)
        moe.residual_experts = residuals
        moe.residual_group_sizes = {
            expert_idx: len(members) for members in groups.values() for expert_idx in members
        }
        moe.forward = types.MethodType(residual_aware_moe_forward, moe)


def residual_state_dict(model, residual_width: int) -> Dict[str, object]:
    """Serialize only residual weights, never duplicate the static checkpoint."""
    state: Dict[str, torch.Tensor] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        residuals = getattr(layer.block_sparse_moe, "residual_experts", None)
        if residuals is None:
            continue
        for expert_idx, residual in residuals.items():
            for name, tensor in residual.state_dict().items():
                state[f"{layer_idx}.{expert_idx}.{name}"] = tensor.detach().cpu().clone()
    return {"residual_width": int(residual_width), "state_dict": state}


def load_residual_state_dict(model, payload: Mapping[str, object], group_state: Mapping[str, torch.Tensor]) -> int:
    """Attach residuals and restore them from residuals.pth."""
    residual_width = int(payload["residual_width"])
    attach_residual_experts(model, group_state, residual_width)
    state = payload["state_dict"]
    for layer_idx, layer in enumerate(model.model.layers):
        residuals = getattr(layer.block_sparse_moe, "residual_experts", None)
        if residuals is None:
            continue
        for expert_idx, residual in residuals.items():
            prefix = f"{layer_idx}.{expert_idx}."
            local_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if not local_state:
                raise KeyError(f"Missing residual state for layer {layer_idx}, expert {expert_idx}")
            residual.load_state_dict(local_state, strict=True)
    return residual_width


def merged_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
    )

    # One hot encode the selected experts to create an expert mask
    # this will be used to easily index which expert is going to be sollicitated
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Loop over all available experts in the model and perform the computation on each expert
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[self.expert_dict[expert_idx]]
        idx, top_x = torch.where(expert_mask[expert_idx])

        # Index the correct hidden states and compute the expert hidden state for
        # the current expert. We need to make sure to multiply the output hidden
        # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

        # However `index_add_` only support torch tensors for indexing so we'll use
        # the `top_x` tensor here.
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits

class MoEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.expert_to_group = {} # expert_idx: group_label
        self.group_to_expert = {} # group label: [expert idx]
        self.unmerge_matrix = {} # group label: unmerge matrix for w2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.model.training and self.model.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.model.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.model.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.model.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.model.num_experts):
            expert_layer = self.model.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            group_label = self.expert_to_group[expert_idx]
            if len(self.group_to_expert[group_label]) == 1:
                group_idx = 0
            else:
                group_idx = torch.where(self.group_to_expert[group_label] == expert_idx)[0].item()

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            if self.unmerge_matrix[group_label] is not None:
                current_hidden_states = torch.matmul(expert_layer(current_state), self.unmerge_matrix[group_label][:, group_idx * self.model.hidden_dim:(group_idx+1) * self.model.hidden_dim]) * routing_weights[top_x, idx, None]
            else:
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class SharedLinearLayers(nn.Module):
    def __init__(self, config, shared_w1, shared_w2, shared_w3):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        
        self.w1_layers = shared_w1
        self.w2_layers = shared_w2
        self.w3_layers = shared_w3

class ModifiedMixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config, shared_layers, w1_id, w2_id, w3_id):
        super().__init__()
        self.shared_layers = shared_layers
        self.act_fn = ACT2FN[config.hidden_act]
        self.w1_id = w1_id
        self.w2_id = w2_id
        self.w3_id = w3_id

    def forward(self, hidden_states):
        w1 = self.shared_layers.w1_layers[self.w1_id]
        w2 = self.shared_layers.w2_layers[self.w2_id]
        w3 = self.shared_layers.w3_layers[self.w3_id]
        
        current_hidden_states = self.act_fn(w1(hidden_states)) * w3(hidden_states)
        current_hidden_states = w2(current_hidden_states)
        return current_hidden_states
