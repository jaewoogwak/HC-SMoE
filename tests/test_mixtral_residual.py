"""CPU smoke tests for residual-aware Mixtral MoE routing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hcsmoe.models.mixtral.utils import (
    attach_residual_experts,
    bind_shared_experts_from_group_state,
    load_residual_state_dict,
    residual_state_dict,
)


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(4, 7, bias=False)
        self.w2 = nn.Linear(7, 4, bias=False)
        self.w3 = nn.Linear(4, 7, bias=False)

    def forward(self, hidden_states):
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 4
        self.num_experts = self.gate_num_experts = 6
        self.top_k = 2
        self.gate = nn.Linear(4, 6, bias=False)
        self.experts = nn.ModuleList([TinyExpert() for _ in range(6)])

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
        output = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.gate_num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            route_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel():
                output.index_add_(0, token_idx, self.experts[expert_idx](hidden_states[token_idx]) * routing_weights[token_idx, route_idx, None])
        return output.reshape(batch, sequence, hidden), router_logits


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.block_sparse_moe = TinyMoE()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])


def test_residual_moe_handles_imbalanced_groups_and_reloads():
    torch.manual_seed(13)
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 1, 2, 2])}
    model = TinyModel().eval()
    moe = model.model.layers[0].block_sparse_moe
    # Simulate static frequency merging: group sizes are 3, 1, and 2.
    for members in ((0, 1, 2), (4, 5)):
        for expert_idx in members[1:]:
            moe.experts[expert_idx] = moe.experts[members[0]]

    hidden_states = torch.randn(5, 3, 4)
    static_output, static_logits = moe(hidden_states)
    attach_residual_experts(model, group_state, residual_width=0)
    width_zero_output, width_zero_logits = moe(hidden_states)
    assert torch.equal(static_output, width_zero_output)
    assert torch.equal(static_logits, width_zero_logits)

    attach_residual_experts(model, group_state, residual_width=3)
    initial_output, residual_logits = moe(hidden_states)
    assert torch.equal(static_output, initial_output)
    assert torch.equal(static_logits, residual_logits)
    assert set(moe.residual_experts) == {"0", "1", "2", "4", "5"}

    with torch.no_grad():
        moe.residual_experts["2"].w2.weight.fill_(0.2)
    expected_output, _ = moe(hidden_states)
    payload = residual_state_dict(model, residual_width=3)

    reloaded = TinyModel().eval()
    bind_shared_experts_from_group_state(reloaded, group_state)
    reloaded.load_state_dict(model.state_dict(), strict=False)
    load_residual_state_dict(reloaded, payload, group_state)
    actual_output, actual_logits = reloaded.model.layers[0].block_sparse_moe(hidden_states)
    assert torch.equal(expected_output, actual_output)
    assert torch.equal(static_logits, actual_logits)
