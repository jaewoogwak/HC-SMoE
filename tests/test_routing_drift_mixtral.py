import inspect
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from accelerate.utils import compute_module_sizes
from transformers import MixtralConfig, MixtralForCausalLM

from hcsmoe.merging.sequential_drift_mixtral import (
    RoutingDriftTrace,
    assert_router_weights_unchanged,
    autoregressive_routing_trace,
    collect_fixed_routing_trace,
    compare_decode_traces,
    compare_fixed_routing_traces,
    free_generation_summary,
    make_global_sample_plan,
    router_weight_snapshot,
)
from hcsmoe.merging.pg19_drift_mixtral import (
    _ContiguousRoutingCapture,
    _IncrementalRoutingCapture,
    _decode_from_prefill,
    aggregate_forced_metrics,
    aggregate_free_summaries,
    collect_document_routing_traces,
    compare_decode_document,
    compare_prefill_document,
    save_pg19_plots,
    select_pg19_documents_from_rows,
    stack_document_metrics,
    summarize_free_document,
)
from hcsmoe.merging.device_placement import (
    GIB,
    PlacementSettings,
    place_model_for_analysis,
    resolve_placement_settings,
)
from hcsmoe.models.mixtral.utils import (
    bind_shared_experts_from_group_state,
    validate_shared_expert_topology,
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
        self.config = SimpleNamespace(num_experts_per_tok=2, num_local_experts=4)

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


class TinyTokenizer:
    def __call__(self, text, max_length, **_kwargs):
        return {"input_ids": [ord(character) % 16 for character in text[:max_length]]}


def test_pg19_document_selection_is_seeded_contiguous_and_length_filtered():
    rows = [
        {"text": "a" * 3, "url": "short"},
        {"text": "b" * 10, "url": "book-b", "short_book_title": "B"},
        {"text": "c" * 10, "url": "book-c", "short_book_title": "C"},
        {"text": "d" * 10, "url": "book-d", "short_book_title": "D"},
    ]
    first = select_pg19_documents_from_rows(rows, TinyTokenizer(), 2, 4, 2, seed=17)
    second = select_pg19_documents_from_rows(rows, TinyTokenizer(), 2, 4, 2, seed=17)
    assert [document.source_index for document in first] == [document.source_index for document in second]
    assert all(document.source_index != 0 for document in first)
    assert all(document.input_ids.numel() == 6 for document in first)
    for document in first:
        expected = [ord(character) % 16 for character in rows[document.source_index]["text"][:6]]
        assert document.input_ids.tolist() == expected


def test_pg19_prefill_forced_and_free_use_full_contiguous_inputs():
    torch.manual_seed(23)
    original_model = TinyCausalLM()
    merged_model = TinyCausalLM(first_delta=[0.0, 0.0, 5.0, 0.0])
    merged_model.model.embed_tokens.load_state_dict(original_model.model.embed_tokens.state_dict())
    document = select_pg19_documents_from_rows(
        [{"text": "abcdefgh", "url": "tiny-book"}], TinyTokenizer(), 1, 5, 3, seed=1
    )[0]
    original = collect_document_routing_traces(original_model, document, 5, 3, "tiny original")
    merged = collect_document_routing_traces(merged_model, document, 5, 3, "tiny merged")
    assert original.prefill.hidden.shape[:2] == (2, 5)
    rows, raw = compare_prefill_document(original.prefill, merged.prefill, document.document_id)
    assert rows[0]["routing_shift_rate"] == 0.0
    assert raw["input_token_ids"].tolist() == document.input_ids[:5].tolist()
    forced = compare_decode_document(original.forced, merged.forced, require_identical_tokens=True)
    assert forced["original_token_ids"].tolist() == document.input_ids[5:].tolist()
    assert forced["merged_token_ids"].tolist() == document.input_ids[5:].tolist()


def test_pg19_optimized_hooks_only_bulk_transfer_after_collection():
    """Guard against reintroducing decode-step CUDA-to-CPU synchronization."""
    incremental_source = inspect.getsource(_IncrementalRoutingCapture.moe_input_hook)
    incremental_source += inspect.getsource(_IncrementalRoutingCapture.router_hook)
    contiguous_source = inspect.getsource(_ContiguousRoutingCapture.moe_input_hook)
    contiguous_source += inspect.getsource(_ContiguousRoutingCapture.router_hook)
    decode_source = inspect.getsource(_decode_from_prefill)
    assert ".cpu(" not in incremental_source
    assert ".cpu(" not in contiguous_source
    assert ".item(" not in decode_source
    assert "torch.cat(" not in decode_source


def test_pg19_document_aggregation_retains_document_axis_and_std():
    base = {
        "original_token_ids": torch.tensor([1, 2]),
        "merged_token_ids": torch.tensor([1, 2]),
        "token_equal": torch.tensor([True, True]),
        "original_topk": torch.zeros(2, 2, 2, dtype=torch.long),
        "merged_topk": torch.zeros(2, 2, 2, dtype=torch.long),
        "exact_topk_match": torch.ones(2, 2, dtype=torch.bool),
        "routing_shift": torch.zeros(2, 2, dtype=torch.bool),
        "topk_overlap": torch.ones(2, 2),
        "hidden_relative_l2": torch.zeros(2, 2),
        "original_margin": torch.ones(2, 2),
        "merged_margin": torch.ones(2, 2),
    }
    changed = {key: value.clone() for key, value in base.items()}
    changed["routing_shift"].fill_(True)
    changed["exact_topk_match"].fill_(False)
    raw = stack_document_metrics(["a", "b"], [base, changed])
    summary = aggregate_forced_metrics(raw)
    assert raw["routing_shift"].shape == (2, 2, 2)
    assert summary["routing_shift_by_step_mean"] == [0.5, 0.5]
    assert summary["routing_shift_by_step_std_across_documents"] == [0.5, 0.5]


def test_pg19_free_summary_uses_only_shared_token_prefix():
    original = _trace([[[0, 1], [0, 1], [0, 1]], [[2, 3], [2, 3], [2, 3]]], tokens=(5, 6, 7))
    merged = _trace([[[0, 1], [0, 2], [4, 5]], [[2, 3], [2, 3], [0, 1]]], tokens=(5, 6, 9))
    metrics = compare_decode_document(original, merged, require_identical_tokens=False)
    document = summarize_free_document(metrics)
    aggregate = aggregate_free_summaries([document])
    assert document["first_routing_divergence_step"] == 1
    assert document["first_token_divergence_step"] == 2
    assert metrics["post_token_divergence"].tolist() == [False, False, True]
    assert aggregate["fraction_routing_divergence_precedes_token_divergence"] == 1.0


def test_pg19_required_plots_are_created():
    layers = []
    for layer in range(2):
        row = {
            "layer": layer,
            "routing_shift_rate": 0.1 * layer,
            "routing_shift_rate_std_across_documents": 0.01,
            "mean_hidden_relative_l2": 0.2 * layer,
            "mean_hidden_relative_l2_std_across_documents": 0.02,
        }
        for key in (
            "original_margin_shifted",
            "original_margin_non_shifted",
            "merged_margin_shifted",
            "merged_margin_non_shifted",
        ):
            row[key] = None if layer == 0 and key.endswith("shifted") and not key.endswith("non_shifted") else 0.3
            row[f"{key}_std_across_documents"] = None if row[key] is None else 0.02
        layers.append(row)
    forced = {
        "decode_steps": 3,
        "routing_shift_heatmap_mean": [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3]],
        "topk_overlap_heatmap_mean": [[1.0, 0.9, 0.8], [0.9, 0.8, 0.7]],
        "routing_shift_by_step_mean": [0.05, 0.15, 0.25],
        "routing_shift_by_step_std_across_documents": [0.01] * 3,
        "routing_shift_by_layer_mean": [0.1, 0.2],
        "routing_shift_by_layer_std_across_documents": [0.01] * 2,
        "hidden_relative_l2_by_step_mean": [0.1, 0.2, 0.3],
        "hidden_relative_l2_by_step_std_across_documents": [0.01] * 3,
    }
    with tempfile.TemporaryDirectory() as directory:
        paths = save_pg19_plots(layers, forced, directory)
        assert len(paths) == 8
        assert all(Path(path).is_file() for path in paths)


def test_auto_placement_budget_detects_a100_40gb_and_80gb():
    with patch("torch.cuda.is_available", return_value=True):
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(name="NVIDIA A100-SXM4-80GB", total_memory=80 * GIB),
        ):
            eighty = resolve_placement_settings("auto", None, "1500GiB")
        with patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(name="NVIDIA A100-SXM4-40GB", total_memory=40 * GIB),
        ):
            forty = resolve_placement_settings("auto", None, "1500GiB")
    assert eighty.gpu_budget_bytes == 70 * GIB
    assert eighty.gpu_reserve_bytes == 10 * GIB
    assert forty.gpu_budget_bytes == 32 * GIB
    assert forty.gpu_reserve_bytes == 8 * GIB


@unittest.skipUnless(torch.cuda.is_available(), "placement equivalence needs CUDA")
def test_auto_and_cpu_offload_placement_preserve_tiny_outputs_and_aliases():
    torch.manual_seed(31)
    auto_model = TinyCausalLM()
    cpu_offload_model = TinyCausalLM()
    cpu_offload_model.load_state_dict(auto_model.state_dict())
    shared = nn.Linear(4, 4, bias=False)
    auto_model.model.layers[0].aliases = nn.ModuleList([shared, shared])
    shared_copy = nn.Linear(4, 4, bias=False)
    shared_copy.load_state_dict(shared.state_dict())
    cpu_offload_model.model.layers[0].aliases = nn.ModuleList([shared_copy, shared_copy])
    auto_settings = resolve_placement_settings("auto", "1GiB", "16GiB")
    offload_settings = PlacementSettings(
        mode="cpu-offload",
        gpu_name=auto_settings.gpu_name,
        gpu_total_bytes=auto_settings.gpu_total_bytes,
        gpu_budget_bytes=None,
        gpu_reserve_bytes=None,
        cpu_budget_bytes=auto_settings.cpu_budget_bytes,
    )
    auto_model = place_model_for_analysis(auto_model, auto_settings)
    cpu_offload_model = place_model_for_analysis(cpu_offload_model, offload_settings)
    input_ids = torch.tensor([[1, 2, 3]], device="cuda:0")
    with torch.inference_mode():
        auto_output = auto_model.model(input_ids)
        offload_output = cpu_offload_model.model(input_ids)
    torch.testing.assert_close(auto_output, offload_output, rtol=1e-5, atol=1e-6)
    assert auto_model.model.layers[0].aliases[0] is auto_model.model.layers[0].aliases[1]
    assert cpu_offload_model.model.layers[0].aliases[0] is cpu_offload_model.model.layers[0].aliases[1]


@unittest.skipUnless(torch.cuda.is_available(), "hybrid Mixtral placement needs CUDA")
def test_actual_mixtral_hybrid_map_keeps_whole_layers_and_shared_experts():
    config = MixtralConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
    )
    model = MixtralForCausalLM(config).to(dtype=torch.bfloat16)
    group_state = {
        f"model.layers.{layer}.block_sparse_moe": torch.tensor([0, 0, 1, 1])
        for layer in range(config.num_hidden_layers)
    }
    bind_shared_experts_from_group_state(model, group_state)
    router_snapshot = router_weight_snapshot(model)
    total_bytes = int(compute_module_sizes(model, dtype=torch.bfloat16)[""])
    settings = PlacementSettings(
        mode="auto",
        gpu_name=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_budget_bytes=int(total_bytes * 0.72),
        gpu_reserve_bytes=None,
        cpu_budget_bytes=16 * GIB,
    )
    model = place_model_for_analysis(model, settings)
    assert_router_weights_unchanged(router_snapshot, model)
    targets = {"cuda" if isinstance(target, int) or str(target).startswith("cuda") else str(target) for target in model.hf_device_map.values()}
    assert "cuda" in targets and "cpu" in targets
    assert validate_shared_expert_topology(model, group_state) == {
        key: 2 for key in group_state
    }
    with torch.inference_mode():
        output = model(torch.tensor([[1, 2, 3]], device="cuda:0"), use_cache=True)
    assert output.logits.shape == (1, 3, config.vocab_size)
    assert output.logits.device.type == "cuda"


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
    test_pg19_document_selection_is_seeded_contiguous_and_length_filtered = staticmethod(
        test_pg19_document_selection_is_seeded_contiguous_and_length_filtered
    )
    test_pg19_prefill_forced_and_free_use_full_contiguous_inputs = staticmethod(
        test_pg19_prefill_forced_and_free_use_full_contiguous_inputs
    )
    test_pg19_optimized_hooks_only_bulk_transfer_after_collection = staticmethod(
        test_pg19_optimized_hooks_only_bulk_transfer_after_collection
    )
    test_pg19_document_aggregation_retains_document_axis_and_std = staticmethod(
        test_pg19_document_aggregation_retains_document_axis_and_std
    )
    test_pg19_free_summary_uses_only_shared_token_prefix = staticmethod(
        test_pg19_free_summary_uses_only_shared_token_prefix
    )
    test_pg19_required_plots_are_created = staticmethod(test_pg19_required_plots_are_created)
    test_auto_placement_budget_detects_a100_40gb_and_80gb = staticmethod(
        test_auto_placement_budget_detects_a100_40gb_and_80gb
    )
    test_auto_and_cpu_offload_placement_preserve_tiny_outputs_and_aliases = staticmethod(
        test_auto_and_cpu_offload_placement_preserve_tiny_outputs_and_aliases
    )
    test_actual_mixtral_hybrid_map_keeps_whole_layers_and_shared_experts = staticmethod(
        test_actual_mixtral_hybrid_map_keeps_whole_layers_and_shared_experts
    )


if __name__ == "__main__":
    unittest.main()
