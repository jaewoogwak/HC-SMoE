from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from hcsmoe.merging.sequential_drift_mixtral import (
    RoutingDriftTrace,
    autoregressive_routing_trace,
    collect_fixed_routing_trace,
    compare_decode_traces,
    compare_fixed_routing_traces,
    free_generation_summary,
    make_global_sample_plan,
)


def _trace(top2, hidden=None, tokens=(1, 2, 3)):
    top2 = torch.tensor(top2, dtype=torch.long)
    layers, steps = top2.shape[:2]
    if hidden is None:
        hidden = torch.ones(layers, steps, 2)
    return RoutingDriftTrace(
        hidden=hidden,
        top2=top2,
        margin=torch.ones(layers, steps),
        input_id_batches=[torch.tensor([[7, 8]])],
        token_ids=torch.tensor(tokens[:steps]),
    )


def test_fixed_metrics_and_layer_zero_router_invariant():
    original = _trace([[[0, 1], [2, 3]], [[0, 1], [2, 3]]], hidden=torch.ones(2, 2, 2))
    merged_hidden = torch.ones(2, 2, 2)
    merged_hidden[1] += 1.0
    merged = _trace([[[1, 0], [3, 2]], [[0, 2], [4, 5]]], hidden=merged_hidden)
    layers, tokens = compare_fixed_routing_traces(original, merged, torch.tensor([0, 1]))
    assert layers[0]["routing_shift_rate"] == 0.0
    assert layers[1]["routing_shift_rate"] == 1.0
    assert layers[1]["mean_top2_overlap"] == 0.25
    assert tokens["sampled_input_token_ids"].tolist() == [7, 8]
    assert tokens["hidden_relative_l2"][1].gt(0).all()


def test_layer_zero_routing_mismatch_fails_loudly():
    original = _trace([[[0, 1]]], tokens=(1,))
    merged = _trace([[[0, 2]]], tokens=(1,))
    try:
        compare_fixed_routing_traces(original, merged, torch.tensor([0]))
    except AssertionError as error:
        assert "first MoE layer routing differs" in str(error)
    else:
        raise AssertionError("layer-0 mismatch should fail")


def test_free_generation_marks_post_token_divergence():
    original = _trace([[[0, 1], [0, 1], [0, 1]], [[2, 3], [2, 3], [2, 3]]], tokens=(5, 6, 7))
    merged = _trace([[[0, 1], [0, 2], [4, 5]], [[2, 3], [2, 3], [0, 1]]], tokens=(5, 6, 9))
    metrics = compare_decode_traces(original, merged, require_identical_tokens=False)
    summary = free_generation_summary(metrics)
    assert summary["first_routing_divergence_step"] == 1
    assert summary["first_token_divergence_step"] == 2
    assert summary["routing_divergence_precedes_token_divergence"] is True
    assert metrics["post_token_divergence"].tolist() == [False, False, True]


class TinyMoE(nn.Module):
    def __init__(self, delta=None):
        super().__init__()
        self.register_buffer("delta", torch.zeros(4) if delta is None else torch.tensor(delta, dtype=torch.float32))

    def forward(self, hidden):
        logits = hidden.reshape(-1, 4)
        return hidden + self.delta, logits


class TinyLayer(nn.Module):
    def __init__(self, delta=None):
        super().__init__()
        self.block_sparse_moe = TinyMoE(delta)


class TinyBackbone(nn.Module):
    def __init__(self, first_delta=None):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([TinyLayer(first_delta), TinyLayer()])

    def forward(self, input_ids, **_kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden, _ = layer.block_sparse_moe(hidden)
        return hidden


class TinyCausalLM(nn.Module):
    def __init__(self, first_delta=None):
        super().__init__()
        self.model = TinyBackbone(first_delta)
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
            hidden_states=hidden,
        )


def test_fixed_forward_uses_candidate_propagated_hidden_states_and_identical_ids():
    torch.manual_seed(5)
    original_model = TinyCausalLM()
    merged_model = TinyCausalLM(first_delta=[0.0, 0.0, 5.0, 0.0])
    merged_model.model.embed_tokens.load_state_dict(original_model.model.embed_tokens.state_dict())
    loader = [{"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[1, 2, 3, 4]])}]
    plan = make_global_sample_plan(total_tokens=4, sample_tokens=4, seed=1)
    original = collect_fixed_routing_trace(original_model, loader, plan)
    merged = collect_fixed_routing_trace(merged_model, loader, plan, expected_input_batches=original.input_id_batches)
    assert torch.equal(original.hidden[0], merged.hidden[0])
    assert not torch.equal(original.hidden[1], merged.hidden[1])
    compare_fixed_routing_traces(original, merged, plan.positions)


def test_forced_decode_uses_identical_tokens_and_model_owned_caches():
    torch.manual_seed(7)
    original_model = TinyCausalLM()
    merged_model = TinyCausalLM()
    merged_model.load_state_dict(original_model.state_dict())
    prompt = torch.tensor([[1, 2]])
    forced = torch.tensor([3, 4, 5])
    original = autoregressive_routing_trace(original_model, prompt, 3, forced, "tiny original forced")
    merged = autoregressive_routing_trace(merged_model, prompt, 3, forced, "tiny merged forced")
    metrics = compare_decode_traces(original, merged, require_identical_tokens=True)
    assert torch.equal(metrics["original_token_ids"], forced)
    assert torch.equal(metrics["merged_token_ids"], forced)
    assert not metrics["routing_shift"].any()


class RoutingDriftUnitTests(unittest.TestCase):
    """Expose the dependency-free checks to the standard-library test runner."""

    test_fixed_metrics_and_layer_zero_router_invariant = staticmethod(
        test_fixed_metrics_and_layer_zero_router_invariant
    )
    test_layer_zero_routing_mismatch_fails_loudly = staticmethod(
        test_layer_zero_routing_mismatch_fails_loudly
    )
    test_free_generation_marks_post_token_divergence = staticmethod(
        test_free_generation_marks_post_token_divergence
    )
    test_fixed_forward_uses_candidate_propagated_hidden_states_and_identical_ids = staticmethod(
        test_fixed_forward_uses_candidate_propagated_hidden_states_and_identical_ids
    )
    test_forced_decode_uses_identical_tokens_and_model_owned_caches = staticmethod(
        test_forced_decode_uses_identical_tokens_and_model_owned_caches
    )


if __name__ == "__main__":
    unittest.main()
