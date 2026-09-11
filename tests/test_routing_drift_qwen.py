from types import SimpleNamespace
import tempfile
import unittest

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from hcsmoe.merging.qwen_checkpoint import (
    bind_shared_experts_from_group_state,
    load_compressed_model_for_analysis,
    validate_shared_expert_topology,
)
from hcsmoe.merging.sequential_drift_qwen import (
    RoutingDriftTrace,
    autoregressive_routing_trace,
    collect_fixed_routing_trace,
    compare_decode_traces,
    compare_fixed_routing_traces,
    free_generation_summary,
    make_global_sample_plan,
)


def _trace(topk, hidden=None, tokens=(1, 2, 3)):
    topk = torch.tensor(topk, dtype=torch.long)
    layers, steps = topk.shape[:2]
    if hidden is None:
        hidden = torch.ones(layers, steps, 6)
    return RoutingDriftTrace(
        hidden=hidden,
        topk=topk,
        margin=torch.ones(layers, steps),
        layer_indices=tuple(range(layers)),
        top_k=4,
        num_experts=6,
        input_id_batches=[torch.tensor([[7, 8]])],
        token_ids=torch.tensor(tokens[:steps]),
    )


class TinyQwenMoE(nn.Module):
    def __init__(self, delta=None):
        super().__init__()
        self.gate = nn.Linear(6, 6, bias=False)
        self.experts = nn.ModuleList([nn.Linear(6, 6, bias=False) for _ in range(6)])
        self.shared_expert = nn.Linear(6, 6, bias=False)
        self.shared_expert_gate = nn.Linear(6, 1, bias=False)
        self.register_buffer("delta", torch.zeros(6) if delta is None else torch.tensor(delta, dtype=torch.float32))
        self.grad_flags = []
        with torch.no_grad():
            self.gate.weight.copy_(torch.eye(6))

    def forward(self, hidden):
        self.grad_flags.append(torch.is_grad_enabled())
        logits = self.gate(hidden.reshape(-1, 6))
        shared_effect = 0.01 * self.shared_expert(hidden)
        return hidden + self.delta + shared_effect, logits


class TinyLayer(nn.Module):
    def __init__(self, delta=None):
        super().__init__()
        self.mlp = TinyQwenMoE(delta)


class TinyBackbone(nn.Module):
    def __init__(self, first_delta=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 6)
        self.layers = nn.ModuleList([TinyLayer(first_delta), TinyLayer()])

    def forward(self, input_ids, **_kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden, _ = layer.mlp(hidden)
        return hidden


class TinyQwenCausalLM(nn.Module):
    def __init__(self, first_delta=None):
        super().__init__()
        self.model = TinyBackbone(first_delta)
        self.config = SimpleNamespace(num_experts_per_tok=4, num_experts=6)
        self.cache_identity = object()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, past_key_values=None, **_kwargs):
        if past_key_values is not None:
            assert past_key_values[0] is self.cache_identity
            prior_length = past_key_values[1]
        else:
            prior_length = 0
        hidden = self.model(input_ids)
        logits = torch.zeros((*input_ids.shape, 16))
        next_ids = (input_ids + 1) % 16
        logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
        return SimpleNamespace(
            logits=logits,
            past_key_values=(self.cache_identity, prior_length + input_ids.shape[1]),
        )


class QwenRoutingDriftUnitTests(unittest.TestCase):
    def test_dynamic_topk_metrics_and_changed_count_histogram(self):
        original = _trace([
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3]],
        ])
        merged = _trace([
            [[3, 2, 1, 0], [2, 3, 1, 0]],
            [[0, 1, 2, 4], [0, 1, 4, 5]],
        ])
        layers, tokens = compare_fixed_routing_traces(original, merged, torch.tensor([0, 1]))
        self.assertEqual(layers[0]["routing_shift_rate"], 0.0)
        self.assertEqual(layers[1]["mean_changed_experts"], 1.5)
        self.assertEqual(layers[1]["changed_expert_count_histogram"], {"0": 0, "1": 1, "2": 1, "3": 0, "4": 0})
        self.assertEqual(tokens["original_topk"].shape[-1], 4)
        self.assertEqual(int(tokens["num_sparse_experts"]), 6)

    def test_layer_zero_mismatch_fails_loudly(self):
        original = _trace([[[0, 1, 2, 3]]], tokens=(1,))
        merged = _trace([[[0, 1, 2, 4]]], tokens=(1,))
        with self.assertRaisesRegex(AssertionError, "first sparse MoE layer"):
            compare_fixed_routing_traces(original, merged, torch.tensor([0]))

    def test_candidate_propagates_its_own_hidden_and_shared_path_is_not_a_route(self):
        torch.manual_seed(5)
        original_model = TinyQwenCausalLM()
        merged_model = TinyQwenCausalLM(first_delta=[0, 0, 0, 0, 5, 0])
        merged_model.load_state_dict(original_model.state_dict(), strict=False)
        merged_model.model.layers[0].mlp.delta.copy_(torch.tensor([0, 0, 0, 0, 5, 0]))
        loader = [{"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[1, 2, 3, 4]])}]
        plan = make_global_sample_plan(4, 4, 1)
        original = collect_fixed_routing_trace(original_model, loader, plan)
        merged = collect_fixed_routing_trace(
            merged_model, loader, plan, expected_input_batches=original.input_id_batches
        )
        self.assertTrue(torch.equal(original.hidden[0], merged.hidden[0]))
        self.assertFalse(torch.equal(original.hidden[1], merged.hidden[1]))
        self.assertEqual(original.topk.shape[-1], original_model.config.num_experts_per_tok)
        self.assertEqual(original.num_experts, 6)
        self.assertTrue(all(not flag for layer in original_model.model.layers for flag in layer.mlp.grad_flags))
        compare_fixed_routing_traces(original, merged, plan.positions)

    def test_forced_decode_uses_identical_tokens_and_separate_caches(self):
        torch.manual_seed(7)
        original_model = TinyQwenCausalLM()
        merged_model = TinyQwenCausalLM()
        merged_model.load_state_dict(original_model.state_dict())
        prompt = torch.tensor([[1, 2]])
        forced = torch.tensor([3, 4, 5])
        original = autoregressive_routing_trace(original_model, prompt, 3, forced, "tiny original")
        merged = autoregressive_routing_trace(merged_model, prompt, 3, forced, "tiny merged")
        metrics = compare_decode_traces(original, merged, require_identical_tokens=True)
        self.assertTrue(torch.equal(metrics["original_token_ids"], forced))
        self.assertTrue(torch.equal(metrics["merged_token_ids"], forced))
        self.assertFalse(metrics["routing_shift"].any())

    def test_free_generation_marks_post_token_divergence(self):
        original = _trace([
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
        ], tokens=(5, 6, 7))
        merged = _trace([
            [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 4, 5]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 4, 5]],
        ], tokens=(5, 6, 9))
        metrics = compare_decode_traces(original, merged, require_identical_tokens=False)
        summary = free_generation_summary(metrics)
        self.assertEqual(summary["first_routing_divergence_step"], 1)
        self.assertEqual(summary["first_token_divergence_step"], 2)
        self.assertTrue(summary["routing_divergence_precedes_token_divergence"])
        self.assertEqual(metrics["post_token_divergence"].tolist(), [False, False, True])

    def test_checkpoint_aliases_sparse_experts_but_not_shared_expert(self):
        model = TinyQwenCausalLM()
        group_state = {
            "model.layers.0.mlp": torch.tensor([0, 0, 1, 1, 2, 2]),
            "model.layers.1.mlp": torch.tensor([0, 0, 1, 1, 2, 2]),
        }
        shared_ids = [id(layer.mlp.shared_expert) for layer in model.model.layers]
        bind_shared_experts_from_group_state(model, group_state)
        counts = validate_shared_expert_topology(model, group_state, expected_num_groups=3)
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(shared_ids, [id(layer.mlp.shared_expert) for layer in model.model.layers])
        for layer, shared_id in zip(model.model.layers, shared_ids):
            self.assertIs(layer.mlp.experts[0], layer.mlp.experts[1])
            self.assertIsNot(layer.mlp.experts[0], layer.mlp.shared_expert)
            self.assertEqual(id(layer.mlp.shared_expert), shared_id)

    def test_assign_load_preserves_checkpoint_expert_aliases(self):
        source = TinyQwenCausalLM()
        group_state = {
            "model.layers.0.mlp": torch.tensor([0, 0, 1, 1, 2, 2]),
            "model.layers.1.mlp": torch.tensor([0, 0, 1, 1, 2, 2]),
        }
        bind_shared_experts_from_group_state(source, group_state)
        state_dict = source.state_dict()
        with init_empty_weights():
            restored = TinyQwenCausalLM()
        bind_shared_experts_from_group_state(restored, group_state)
        restored.load_state_dict(state_dict, strict=True, assign=True)
        counts = validate_shared_expert_topology(restored, group_state, expected_num_groups=3)
        self.assertEqual(set(counts.values()), {3})

    def test_actual_transformers_qwen_sparse_block_contract(self):
        config = Qwen2MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            num_experts=6,
            num_experts_per_tok=4,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            decoder_sparse_step=1,
        )
        model = Qwen2MoeForCausalLM(config)
        loader = [{"input_ids": torch.tensor([[1, 2, 3, 4]])}]
        plan = make_global_sample_plan(4, 4, 11)
        trace = collect_fixed_routing_trace(model, loader, plan)
        self.assertEqual(trace.topk.shape, (2, 4, 4))
        self.assertEqual(trace.margin.shape, (2, 4))
        self.assertEqual(trace.num_experts, 6)

    @unittest.skipUnless(torch.cuda.is_available(), "loader requires the analysis CUDA environment")
    def test_actual_qwen_checkpoint_loader_restores_alias_topology(self):
        config = Qwen2MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            num_experts=6,
            num_experts_per_tok=4,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            decoder_sparse_step=1,
        )
        source = Qwen2MoeForCausalLM(config).to(dtype=torch.bfloat16)
        group_state = {
            "model.layers.0.mlp": torch.tensor([0, 0, 1, 1, 2, 2]),
            "model.layers.1.mlp": torch.tensor([0, 0, 1, 1, 2, 2]),
        }
        bind_shared_experts_from_group_state(source, group_state)
        source_routers = [layer.mlp.gate.weight.detach().clone() for layer in source.model.layers]
        with tempfile.TemporaryDirectory() as tmp:
            config.save_pretrained(tmp)
            model_path = f"{tmp}/model.pth"
            group_path = f"{tmp}/group_state_dict.pt"
            torch.save(source.state_dict(), model_path)
            torch.save(group_state, group_path)
            restored, restored_groups = load_compressed_model_for_analysis(
                tmp, model_path, group_path, expected_num_groups=3
            )
        validate_shared_expert_topology(restored, restored_groups, expected_num_groups=3)
        for layer, expected_router in zip(restored.model.layers, source_routers):
            self.assertTrue(torch.equal(layer.mlp.gate.weight, expected_router))


if __name__ == "__main__":
    unittest.main()
