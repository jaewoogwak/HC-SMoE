"""CPU smoke tests matching the HF MixtralSparseMoeBlock residual interface."""

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
    """Deliberately has no gate_num_experts, like HF MixtralSparseMoeBlock."""

    def __init__(self):
        super().__init__()
        self.hidden_dim = 4
        self.num_experts = 8
        self.top_k = 2
        self.gate = nn.Linear(4, self.num_experts, bias=False)
        self.experts = nn.ModuleList([TinyExpert() for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        batch, sequence, hidden = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = (routing_weights / routing_weights.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)
        output = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
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


def _make_static_model():
    model = TinyModel().eval()
    # Imbalanced groups: size 4, singleton, and size 3.
    for members in ((0, 1, 2, 3), (5, 6, 7)):
        for expert_idx in members[1:]:
            model.model.layers[0].block_sparse_moe.experts[expert_idx] = (
                model.model.layers[0].block_sparse_moe.experts[members[0]]
            )
    return model


def test_residual_moe_hf_interface_imbalanced_groups_and_reload():
    torch.manual_seed(13)
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = _make_static_model()
    moe = model.model.layers[0].block_sparse_moe
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
    assert set(moe.residual_experts) == {"0", "1", "2", "3", "5", "6", "7"}

    # A route to singleton expert 4 must remain exactly static even after the
    # residuals for all non-singleton experts become non-zero.
    original_gate = moe.gate.weight.detach().clone()
    original_top_k = moe.top_k
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[4].fill_(1.0)
        for residual in moe.residual_experts.values():
            residual.w2.weight.fill_(0.2)
    moe.top_k = 1
    singleton_input = torch.ones(1, 1, 4)
    singleton_output, singleton_logits = moe(singleton_input)
    assert torch.equal(singleton_output, moe.experts[4](singleton_input))
    assert torch.equal(singleton_logits, moe.gate(singleton_input.reshape(-1, 4)))
    moe.top_k = original_top_k
    with torch.no_grad():
        moe.gate.weight.copy_(original_gate)

    expected_output, expected_logits = moe(hidden_states)
    payload = residual_state_dict(model, residual_width=3)
    reloaded = _make_static_model()
    bind_shared_experts_from_group_state(reloaded, group_state)
    reloaded.load_state_dict(model.state_dict(), strict=False)
    load_residual_state_dict(reloaded, payload, group_state)
    actual_output, actual_logits = reloaded.model.layers[0].block_sparse_moe(hidden_states)
    assert torch.equal(expected_output, actual_output)
    assert torch.equal(expected_logits, actual_logits)


def test_cpu_bfloat16_input_casts_cpu_float32_residual():
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = _make_static_model().bfloat16().eval()
    attach_residual_experts(model, group_state, residual_width=3)
    moe = model.model.layers[0].block_sparse_moe
    assert next(moe.residual_experts["0"].parameters()).device.type == "cpu"
    assert next(moe.residual_experts["0"].parameters()).dtype == torch.float32

    output, _ = moe(torch.randn(2, 3, 4, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert next(moe.residual_experts["0"].parameters()).dtype == torch.bfloat16
