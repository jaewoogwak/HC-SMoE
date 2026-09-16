import copy
import unittest
from unittest.mock import patch

import torch
from transformers import MixtralConfig, MixtralForCausalLM

import hcsmoe.merging.sequential_drift_aware_mixtral as sequential_module
from hcsmoe.merging.sequential_drift_aware_mixtral import (
    CalibrationBatch,
    MixtralPartitionEvaluator,
    _assert_post_commit_consistency,
    _attention_prefix,
    _router_topk_after_attention,
    collect_calibration_batches,
    commit_partition,
    forward_decoder_layer,
    normalized_output_cost,
    run_sequential_drift_aware_merging,
    topk_overlap_cost,
)
from hcsmoe.merging.sequential_drift_mixtral import (
    assert_router_weights_unchanged,
    router_weight_snapshot,
)


def tiny_mixtral(num_layers=2):
    torch.manual_seed(17)
    config = MixtralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    return MixtralForCausalLM(config).eval()


def raw_batch(values):
    input_ids = torch.tensor(values, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def test_output_and_top2_cost_definitions():
    teacher = torch.tensor([[[3.0, 4.0], [1.0, 0.0]]])
    candidate = torch.tensor([[[0.0, 0.0], [2.0, 0.0]]])
    assert abs(normalized_output_cost(teacher, candidate).item() - 1.0) < 1e-7

    reference = torch.tensor([[0, 1], [2, 3], [4, 5]])
    assert topk_overlap_cost(reference, reference).item() == 0.0
    one_changed = torch.tensor([[0, 2], [2, 4], [4, 0]])
    assert topk_overlap_cost(reference, one_changed).item() == 0.5
    both_changed = torch.tensor([[2, 3], [4, 5], [0, 1]])
    assert topk_overlap_cost(reference, both_changed).item() == 1.0


def test_calibration_preserves_sequence_boundaries_and_positions():
    model = tiny_mixtral()
    batches, token_count = collect_calibration_batches(
        model,
        [raw_batch([[1, 2, 3], [4, 5, 6]]), raw_batch([[7, 8, 9]])],
    )
    assert token_count == 9
    assert [tuple(batch.hidden_states.shape[:2]) for batch in batches] == [(2, 3), (1, 3)]
    assert batches[0].position_ids.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert batches[1].position_ids.tolist() == [[0, 1, 2]]


def test_weighted_group_is_rebuilt_from_originals_and_router_is_unchanged():
    model = tiny_mixtral(num_layers=1)
    layer = model.model.layers[0]
    originals = {}
    for expert_index, expert in enumerate(layer.block_sparse_moe.experts):
        for name in ("w1", "w2", "w3"):
            weight = getattr(expert, name).weight
            weight.data.fill_(float(expert_index + 1))
            originals[(expert_index, name)] = weight.detach().clone()
    routers = router_weight_snapshot(model)
    usage = torch.tensor([1.0, 2.0, 3.0, 4.0])
    commit_partition(layer, ((0, 1), (2, 3)), usage)

    for name in ("w1", "w2", "w3"):
        first_expected = sum(originals[(index, name)] * usage[index] for index in (0, 1)) / 3.0
        second_expected = sum(originals[(index, name)] * usage[index] for index in (2, 3)) / 7.0
        torch.testing.assert_close(
            getattr(layer.block_sparse_moe.experts[0], name).weight, first_expected
        )
        torch.testing.assert_close(
            getattr(layer.block_sparse_moe.experts[2], name).weight, second_expected
        )
    assert layer.block_sparse_moe.experts[0] is layer.block_sparse_moe.experts[1]
    assert layer.block_sparse_moe.experts[2] is layer.block_sparse_moe.experts[3]
    assert_router_weights_unchanged(routers, model)


def test_manual_candidate_matches_real_aliased_layer_and_next_router():
    teacher_model = tiny_mixtral()
    batches, _ = collect_calibration_batches(teacher_model, [raw_batch([[1, 2, 3, 4]])])
    evaluator = MixtralPartitionEvaluator(teacher_model, 0, batches)
    routed_pair = tuple(sorted(evaluator.contexts[0].selected_experts[0].tolist()))
    routed_pair_partition = tuple(
        [routed_pair]
        + [(expert,) for expert in range(4) if expert not in routed_pair]
    )
    partitions = [
        routed_pair_partition,
        ((0, 2), (1,), (3,)),
        ((0, 1), (2, 3)),
        ((0, 1, 3), (2,)),
    ]
    for partition in partitions:
        actual_model = copy.deepcopy(teacher_model)
        manual_hidden = evaluator.candidate_hidden(partition)
        commit_partition(actual_model.model.layers[0], partition, evaluator.usage)
        actual_hidden = forward_decoder_layer(actual_model, 0, batches)
        torch.testing.assert_close(
            manual_hidden[0], actual_hidden[0].hidden_states, rtol=1e-5, atol=1e-5
        )

        manual_logits, manual_top2 = _router_topk_after_attention(
            teacher_model, teacher_model.model.layers[1], batches[0], manual_hidden[0]
        )
        actual_logits, actual_top2 = _router_topk_after_attention(
            actual_model, actual_model.model.layers[1], batches[0], actual_hidden[0].hidden_states
        )
        torch.testing.assert_close(manual_logits, actual_logits, rtol=1e-5, atol=1e-5)
        assert torch.equal(manual_top2, actual_top2)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 required")
def test_bf16_same_group_top2_matches_hf_expert_id_accumulation():
    teacher_model = tiny_mixtral().to(device="cuda:0", dtype=torch.bfloat16)
    batches, _ = collect_calibration_batches(teacher_model, [raw_batch([[1, 2, 3, 4, 5, 6]])])
    evaluator = MixtralPartitionEvaluator(teacher_model, 0, batches)
    selected = evaluator.contexts[0].selected_experts
    routed_pair = tuple(sorted(selected[0].tolist()))
    partition = tuple(
        [routed_pair]
        + [(expert,) for expert in range(4) if expert not in routed_pair]
    )
    assert set(selected[0].tolist()).issubset(set(routed_pair))

    actual_model = copy.deepcopy(teacher_model)
    manual_hidden = evaluator.candidate_hidden(partition)[0]
    commit_partition(actual_model.model.layers[0], partition, evaluator.usage)

    attention_residual, moe_input = _attention_prefix(
        actual_model, actual_model.model.layers[0], batches[0], batches[0].hidden_states
    )
    actual_moe, _ = actual_model.model.layers[0].block_sparse_moe(moe_input)
    actual_from_moe = (attention_residual + actual_moe).detach().cpu()
    actual_decoder = forward_decoder_layer(actual_model, 0, batches)[0].hidden_states

    torch.testing.assert_close(manual_hidden, actual_from_moe, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual_from_moe, actual_decoder, rtol=0.0, atol=0.0)

    manual_logits, manual_top2 = _router_topk_after_attention(
        teacher_model, teacher_model.model.layers[1], batches[0], manual_hidden
    )
    actual_logits, actual_top2 = _router_topk_after_attention(
        actual_model, actual_model.model.layers[1], batches[0], actual_decoder
    )
    torch.testing.assert_close(
        manual_logits.cpu(), actual_logits.cpu(), rtol=5e-3, atol=5e-3
    )
    assert torch.equal(manual_top2.cpu(), actual_top2.cpu())


def test_post_commit_mismatch_reports_actionable_diagnostics():
    try:
        _assert_post_commit_consistency(
            torch.zeros(1, 2, 3),
            torch.ones(1, 2, 3),
            layer_index=19,
            batch_index=2,
            partition=((0, 1), (2, 3)),
            same_group_top2_tokens=7,
        )
    except AssertionError as error:
        message = str(error)
        for expected in (
            "layer=19",
            "batch=2",
            "mismatched_elements=6/6",
            "max_absolute_difference=1",
            "max_relative_difference=1",
            "dtype=torch.float32",
            "partition=[[0, 1], [2, 3]]",
            "same_group_top2_tokens=7",
        ):
            assert expected in message
    else:
        raise AssertionError("a deliberate post-commit mismatch must fail")


def test_sequential_next_layer_input_is_committed_output_and_final_route_is_null():
    model = tiny_mixtral()
    pristine = copy.deepcopy(model)
    loader = [raw_batch([[1, 2, 3, 4]])]
    captured_inputs = []
    original_evaluator = sequential_module.MixtralPartitionEvaluator

    class RecordingEvaluator(original_evaluator):
        def __init__(self, model, layer_index, batches, eps=1e-12):
            captured_inputs.append([batch.hidden_states.clone() for batch in batches])
            super().__init__(model, layer_index, batches, eps)

    routers = router_weight_snapshot(model)
    with patch.object(sequential_module, "MixtralPartitionEvaluator", RecordingEvaluator):
        artifacts = run_sequential_drift_aware_merging(
            model,
            loader,
            num_average_groups=2,
            lambda_route=0.5,
            sequential=True,
            calibration_seed=42,
        )
    assert_router_weights_unchanged(routers, model)
    assert len(captured_inputs) == 2

    initial, _ = collect_calibration_batches(model, loader)
    committed = forward_decoder_layer(model, 0, initial)
    pristine_output = forward_decoder_layer(pristine, 0, initial)
    torch.testing.assert_close(captured_inputs[1][0], committed[0].hidden_states)
    assert not torch.equal(captured_inputs[1][0], pristine_output[0].hidden_states)

    final_summary = artifacts["per_layer_summary"][-1]
    assert final_summary["final_c_route"] is None
    for candidate in artifacts["merge_trace"]["model.layers.1.block_sparse_moe"][-1]["candidates"]:
        assert candidate["c_route"] is None
        assert candidate["normalized_c_route"] is None


class SequentialDriftAwareMixtralTest(unittest.TestCase):
    test_output_and_top2_cost_definitions = staticmethod(test_output_and_top2_cost_definitions)
    test_calibration_preserves_sequence_boundaries_and_positions = staticmethod(
        test_calibration_preserves_sequence_boundaries_and_positions
    )
    test_weighted_group_is_rebuilt_from_originals_and_router_is_unchanged = staticmethod(
        test_weighted_group_is_rebuilt_from_originals_and_router_is_unchanged
    )
    test_manual_candidate_matches_real_aliased_layer_and_next_router = staticmethod(
        test_manual_candidate_matches_real_aliased_layer_and_next_router
    )
    test_bf16_same_group_top2_matches_hf_expert_id_accumulation = staticmethod(
        test_bf16_same_group_top2_matches_hf_expert_id_accumulation
    )
    test_post_commit_mismatch_reports_actionable_diagnostics = staticmethod(
        test_post_commit_mismatch_reports_actionable_diagnostics
    )
    test_sequential_next_layer_input_is_committed_output_and_final_route_is_null = staticmethod(
        test_sequential_next_layer_input_is_committed_output_and_final_route_is_null
    )


if __name__ == "__main__":
    unittest.main()
