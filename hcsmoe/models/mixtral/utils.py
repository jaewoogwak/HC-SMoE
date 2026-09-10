"""Mixtral shared-expert utilities used by the static HC-SMoE checkpoint path."""

from typing import Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


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


def validate_shared_expert_topology(model, group_state: Mapping[str, torch.Tensor]) -> Dict[str, int]:
    """Assert that every saved group is represented by one Python module."""
    counts: Dict[str, int] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        name = f"model.layers.{layer_idx}.block_sparse_moe"
        if name not in group_state:
            raise KeyError(f"Missing group mapping for evaluation layer: {name}")
        groups = group_members_from_labels(group_state[name])
        moe = layer.block_sparse_moe
        if len({id(expert) for expert in moe.experts}) != len(groups):
            raise AssertionError(f"{name}: shared expert count does not match group count")
        for members in groups.values():
            if len({id(moe.experts[index]) for index in members}) != 1:
                raise AssertionError(f"{name}: experts {members} do not share one module")
        counts[name] = len(groups)
    return counts


def expand_shared_expert_state_dict(state_dict: Dict[str, torch.Tensor], group_state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fill alias keys omitted by offload-aware state-dict serialization."""
    for layer_name, labels in group_state.items():
        for members in group_members_from_labels(labels).values():
            source_prefix = next((f"{layer_name}.experts.{idx}." for idx in members
                                  if any(key.startswith(f"{layer_name}.experts.{idx}.") for key in state_dict)), None)
            if source_prefix is None:
                raise KeyError(f"No saved static expert weights for {layer_name} group {members}")
            source_keys = [key for key in state_dict if key.startswith(source_prefix)]
            for index in members:
                target_prefix = f"{layer_name}.experts.{index}."
                for source_key in source_keys:
                    state_dict.setdefault(target_prefix + source_key[len(source_prefix):], state_dict[source_key])
    return state_dict


def merged_moe_forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
    final = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
    mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    for expert_idx in range(self.num_experts):
        idx, token_idx = torch.where(mask[expert_idx])
        expert = self.experts[self.expert_dict[expert_idx]]
        output = expert(hidden_states[None, token_idx].reshape(-1, hidden_dim)) * routing_weights[token_idx, idx, None]
        final.index_add_(0, token_idx, output.to(hidden_states.dtype))
    return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


class MoEWrapper(nn.Module):
    """Legacy non-frequency merge wrapper retained for the original grouping module."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.expert_to_group, self.group_to_expert, self.unmerge_matrix = {}, {}, {}

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.model.gate(hidden_states)
        weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        weights, selected = torch.topk(weights, self.model.top_k, dim=-1)
        weights = (weights / weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
        final = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)
        mask = F.one_hot(selected, num_classes=self.model.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.model.num_experts):
            route_idx, token_idx = torch.where(mask[expert_idx])
            group = self.expert_to_group[expert_idx]
            member_idx = 0 if len(self.group_to_expert[group]) == 1 else torch.where(self.group_to_expert[group] == expert_idx)[0].item()
            output = self.model.experts[expert_idx](hidden_states[None, token_idx].reshape(-1, hidden_dim))
            if self.unmerge_matrix[group] is not None:
                output = output @ self.unmerge_matrix[group][:, member_idx * hidden_dim:(member_idx + 1) * hidden_dim]
            final.index_add_(0, token_idx, (output * weights[token_idx, route_idx, None]).to(hidden_states.dtype))
        return final.reshape(batch_size, sequence_length, hidden_dim), router_logits


class SharedLinearLayers(nn.Module):
    def __init__(self, config, shared_w1, shared_w2, shared_w3):
        super().__init__()
        self.hidden_dim, self.ffn_dim = config.hidden_size, config.intermediate_size
        self.w1_layers, self.w2_layers, self.w3_layers = shared_w1, shared_w2, shared_w3


class ModifiedMixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config, shared_layers, w1_id, w2_id, w3_id):
        super().__init__()
        self.shared_layers, self.act_fn = shared_layers, ACT2FN[config.hidden_act]
        self.w1_id, self.w2_id, self.w3_id = w1_id, w2_id, w3_id

    def forward(self, hidden_states):
        w1, w2, w3 = self.shared_layers.w1_layers[self.w1_id], self.shared_layers.w2_layers[self.w2_id], self.shared_layers.w3_layers[self.w3_id]
        return w2(self.act_fn(w1(hidden_states)) * w3(hidden_states))
