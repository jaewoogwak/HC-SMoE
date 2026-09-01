"""CPU smoke tests matching the HF MixtralSparseMoeBlock residual interface."""

import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from hcsmoe.merging.residual_mixtral import (
    _expert_device,
    _parse_diagnostic_experts,
    save_residual_loss_curves,
    train_residuals,
)
from hcsmoe.merging.lora_mixtral import save_lora_loss_curves, train_lora_experts
from hcsmoe.models.mixtral.utils import (
    ExpertLoRA,
    attach_lora_experts,
    attach_residual_experts,
    bind_shared_experts_from_group_state,
    expand_shared_expert_state_dict,
    load_lora_state_dict,
    load_residual_state_dict,
    lora_expert_output,
    lora_params_per_expert,
    lora_state_dict,
    residual_state_dict,
    validate_shared_expert_topology,
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
        self.config = SimpleNamespace(hidden_size=4, intermediate_size=7)


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

    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0].fill_(1.0)
    moe.top_k = 1
    output, _ = moe(torch.ones(2, 3, 4, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert next(moe.residual_experts["0"].parameters()).dtype == torch.bfloat16


def test_lora_zero_init_weight_space_forward_and_singleton():
    torch.manual_seed(23)
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = _make_static_model().eval()
    moe = model.model.layers[0].block_sparse_moe
    hidden_states = torch.randn(3, 2, 4)
    static_output, static_logits = moe(hidden_states)

    attach_lora_experts(model, group_state, lora_rank=3, lora_alpha=3)
    lora_output, lora_logits = moe(hidden_states)
    assert torch.equal(static_output, lora_output)
    assert torch.equal(static_logits, lora_logits)
    assert set(moe.lora_experts) == {"0", "1", "2", "3", "5", "6", "7"}

    bf16_model = _make_static_model().bfloat16().eval()
    bf16_moe = bf16_model.model.layers[0].block_sparse_moe
    with torch.no_grad():
        bf16_moe.gate.weight.zero_()
        bf16_moe.gate.weight[0].fill_(1.0)
    bf16_moe.top_k = 1
    bf16_inputs = torch.ones(2, 2, 4, dtype=torch.bfloat16)
    bf16_static, _ = bf16_moe(bf16_inputs)
    attach_lora_experts(bf16_model, group_state, lora_rank=3, lora_alpha=3)
    bf16_lora, _ = bf16_moe(bf16_inputs)
    assert torch.equal(bf16_static, bf16_lora)
    assert next(bf16_moe.lora_experts["0"].parameters()).dtype == torch.bfloat16

    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[4].fill_(1.0)
        for adapter in moe.lora_experts.values():
            adapter.w2.B.weight.fill_(0.2)
    moe.top_k = 1
    singleton_input = torch.ones(1, 1, 4)
    singleton_output, _ = moe(singleton_input)
    assert torch.equal(singleton_output, moe.experts[4](singleton_input))


def test_lora_matches_explicit_materialized_weights_and_expert_identity():
    torch.manual_seed(29)
    expert = TinyExpert().float()
    adapter = ExpertLoRA(hidden_size=4, intermediate_size=7, rank=3, alpha=3).float()
    with torch.no_grad():
        for projection in (adapter.w1, adapter.w2, adapter.w3):
            projection.A.weight.copy_(torch.randn_like(projection.A.weight))
            projection.B.weight.copy_(torch.randn_like(projection.B.weight))
    inputs = torch.randn(5, 4)
    actual = lora_expert_output(expert, adapter, inputs)
    w1 = expert.w1.weight + adapter.w1.B.weight @ adapter.w1.A.weight
    w2 = expert.w2.weight + adapter.w2.B.weight @ adapter.w2.A.weight
    w3 = expert.w3.weight + adapter.w3.B.weight @ adapter.w3.A.weight
    expected = F.linear(F.silu(F.linear(inputs, w1)) * F.linear(inputs, w3), w2)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)

    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = _make_static_model().eval()
    moe = model.model.layers[0].block_sparse_moe
    attach_lora_experts(model, group_state, lora_rank=3, lora_alpha=3)
    with torch.no_grad():
        moe.lora_experts["0"].w2.B.weight.fill_(0.1)
        moe.lora_experts["1"].w2.B.weight.fill_(-0.2)
    assert id(moe.experts[0]) == id(moe.experts[1])
    assert not torch.equal(
        lora_expert_output(moe.experts[0], moe.lora_experts["0"], inputs),
        lora_expert_output(moe.experts[1], moe.lora_experts["1"], inputs),
    )


def test_lora_save_reload_and_parameter_count():
    torch.manual_seed(31)
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = _make_static_model().eval()
    attach_lora_experts(model, group_state, lora_rank=3, lora_alpha=3)
    with torch.no_grad():
        model.model.layers[0].block_sparse_moe.lora_experts["0"].w1.B.weight.fill_(0.25)
    hidden_states = torch.randn(2, 2, 4)
    expected, _ = model.model.layers[0].block_sparse_moe(hidden_states)
    payload = lora_state_dict(model, lora_rank=3, lora_alpha=3)
    reloaded = _make_static_model().eval()
    bind_shared_experts_from_group_state(reloaded, group_state)
    reloaded.load_state_dict(model.state_dict(), strict=False)
    load_lora_state_dict(reloaded, payload, group_state)
    actual, _ = reloaded.model.layers[0].block_sparse_moe(hidden_states)
    assert torch.equal(expected, actual)
    assert lora_params_per_expert(4096, 14336, 56) == 3_096_576


def test_lora_training_uses_output_reconstruction_and_reports_budget():
    torch.manual_seed(37)
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = _make_static_model().eval()
    inputs = torch.randn(10, 4)
    static = model.model.layers[0].block_sparse_moe.experts[0](inputs).detach()
    calibration = {
        "0.0": {
            "hidden_states": inputs,
            "routing_weights": torch.full((10,), 0.5),
            "original_outputs": static + 0.1,
        }
    }
    metrics = train_lora_experts(
        model=model,
        group_state=group_state,
        calibration=calibration,
        lora_rank=2,
        lora_alpha=2,
        lora_epochs=2,
        lora_lr=1e-3,
        lora_batch_size=3,
        lora_val_ratio=0.2,
        lora_patience=2,
        seed=0,
    )
    result = metrics["experts"]["0.0"]
    assert result["epoch0_validation_loss"] > 0.0
    assert result["best_epoch"] in (1, 2)
    assert "weighted_lora_relative_l2" in result
    assert [item["epoch"] for item in result["epochs"]] == [0, 1, 2]
    assert metrics["aggregate"]["lora_params_per_adapted_expert"] == 3 * (4 + 7) * 2
    with tempfile.TemporaryDirectory() as output_path:
        save_lora_loss_curves(output_path, metrics)
        curve_dir = os.path.join(output_path, "lora_loss_curves")
        assert os.path.isfile(os.path.join(curve_dir, "layer_0_expert_0.png"))
        assert os.path.isfile(os.path.join(curve_dir, "all_lora_experts.png"))


def test_offloaded_meta_expert_uses_accelerate_execution_device():
    expert = TinyExpert().to(device="meta", dtype=torch.bfloat16)
    expert._hf_hook = SimpleNamespace(execution_device=torch.device("cpu"))

    assert _expert_device(expert) == torch.device("cpu")


def test_residual_training_diagnostics_include_zero_residual_epoch():
    torch.manual_seed(7)
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = _make_static_model().eval()
    inputs = torch.randn(10, 4)
    static = model.model.layers[0].block_sparse_moe.experts[0](inputs).detach()
    calibration = {
        "0.0": {
            "hidden_states": inputs,
            "routing_weights": torch.full((10,), 0.5),
            "original_outputs": static + 0.1,
        }
    }
    diagnostics = {}

    metrics = train_residuals(
        model=model,
        group_state=group_state,
        calibration=calibration,
        residual_width=3,
        residual_epochs=2,
        residual_lr=1e-3,
        residual_batch_size=3,
        residual_val_ratio=0.2,
        residual_patience=2,
        seed=0,
        residual_diagnostic_experts="0.0",
        diagnostics=diagnostics,
    )

    assert "diagnostics" not in metrics
    entry = diagnostics["0.0"]
    assert [item["epoch"] for item in entry["epochs"]] == [0, 1, 2]
    assert entry["epochs"][0]["train_loss"] is None
    assert entry["epochs"][0]["end_global_step"] == -1
    assert entry["epochs"][0]["residual_relative_to_original"] == 0.0
    assert entry["epochs"][0]["token_residual_ratio"]["max"] == 0.0
    assert [item["global_step"] for item in entry["steps"]] == list(range(len(entry["steps"])))
    assert all(item["grad_norm"] >= 0.0 for item in entry["steps"])
    assert entry["epochs"][-1]["end_global_step"] == entry["steps"][-1]["global_step"]
    assert entry["best_trained_epoch"] in (1, 2)
    assert "did_training_beat_static_baseline" in entry


def test_residual_diagnostics_do_not_change_training_result():
    torch.manual_seed(17)
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    source = _make_static_model().eval()
    model_state = source.state_dict()
    inputs = torch.randn(10, 4)
    static = source.model.layers[0].block_sparse_moe.experts[0](inputs).detach()
    calibration = {
        "0.0": {
            "hidden_states": inputs,
            "routing_weights": torch.full((10,), 0.5),
            "original_outputs": static + 0.1,
        }
    }
    kwargs = dict(
        group_state=group_state,
        calibration=calibration,
        residual_width=3,
        residual_epochs=2,
        residual_lr=1e-3,
        residual_batch_size=3,
        residual_val_ratio=0.2,
        residual_patience=2,
        seed=0,
    )
    baseline, instrumented = _make_static_model().eval(), _make_static_model().eval()
    baseline.load_state_dict(model_state)
    instrumented.load_state_dict(model_state)

    torch.manual_seed(19)
    baseline_metrics = train_residuals(model=baseline, **kwargs)
    diagnostics = {}
    torch.manual_seed(19)
    instrumented_metrics = train_residuals(
        model=instrumented,
        residual_diagnostic_experts="0.0",
        diagnostics=diagnostics,
        **kwargs,
    )

    assert baseline_metrics == instrumented_metrics
    for name, tensor in residual_state_dict(baseline, 3)["state_dict"].items():
        assert torch.equal(tensor, residual_state_dict(instrumented, 3)["state_dict"][name])


def test_residual_loss_curve_output_and_fire_tuple_selector():
    assert _parse_diagnostic_experts((3.4, 4.6)) == {(3, 4), (4, 6)}
    diagnostics = {
        "3.4": {
            "layer": 3,
            "expert": 4,
            "steps": [{"global_step": 0, "train_loss": 1e-3}],
            "epochs": [
                {"epoch": 0, "end_global_step": -1, "val_loss": 1e-3},
                {"epoch": 1, "end_global_step": 0, "val_loss": 2e-3},
            ],
        }
    }
    with tempfile.TemporaryDirectory() as output_path:
        save_residual_loss_curves(output_path, diagnostics)
        curve_dir = os.path.join(output_path, "residual_loss_curves")
        assert os.path.isfile(os.path.join(curve_dir, "layer_3_expert_4.png"))
        assert os.path.isfile(os.path.join(curve_dir, "all_diagnostic_experts.png"))


def test_group_binding_restores_unique_expert_topology():
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    model = TinyModel()
    bind_shared_experts_from_group_state(model, group_state)

    assert validate_shared_expert_topology(model, group_state) == {
        "model.layers.0.block_sparse_moe": 3
    }


def test_meta_checkpoint_reload_preserves_shared_expert_topology():
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    source = _make_static_model()
    state_dict = {name: value.detach().clone() for name, value in source.state_dict().items()}
    reloaded = TinyModel().to("meta")
    bind_shared_experts_from_group_state(reloaded, group_state)
    reloaded.load_state_dict(state_dict, strict=True, assign=True)

    assert not any(parameter.is_meta for parameter in reloaded.parameters())
    assert validate_shared_expert_topology(reloaded, group_state) == {
        "model.layers.0.block_sparse_moe": 3
    }


def test_expand_shared_expert_state_dict_restores_omitted_aliases():
    group_state = {"model.layers.0.block_sparse_moe": torch.tensor([0, 0, 0, 0, 1, 2, 2, 2])}
    source = _make_static_model()
    state_dict = {name: value.detach().clone() for name, value in source.state_dict().items()}
    for expert_idx in (1, 2, 3, 6, 7):
        prefix = f"model.layers.0.block_sparse_moe.experts.{expert_idx}."
        for name in [name for name in state_dict if name.startswith(prefix)]:
            del state_dict[name]

    expanded = expand_shared_expert_state_dict(state_dict, group_state)
    reloaded = TinyModel().to("meta")
    bind_shared_experts_from_group_state(reloaded, group_state)
    reloaded.load_state_dict(expanded, strict=True, assign=True)
    assert validate_shared_expert_topology(reloaded, group_state) == {
        "model.layers.0.block_sparse_moe": 3
    }
