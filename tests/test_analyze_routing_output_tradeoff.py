import json
import math

import torch

from hcsmoe.analyze_routing_output_tradeoff import (
    discover_alpha_runs,
    local_moe_metrics,
    pair_categories,
    pair_output_statistics,
    relative_l2,
    run_moe_on_input,
    routing_overlap,
    validate_mappings,
)
from hcsmoe.merging.routing_aware_grouping import routing_metrics


def test_pair_categories_ignore_arbitrary_label_ids():
    left = pair_categories(torch.tensor([4, 4, 9, 9]), torch.tensor([0, 1, 1, 0]), torch.tensor([7, 7, 8, 8]))
    right = pair_categories(torch.tensor([1, 1, 6, 6]), torch.tensor([9, 3, 3, 9]), torch.tensor([0, 0, 5, 5]))
    assert [(row["expert_i"], row["expert_j"], row["category"]) for row in left] == [(row["expert_i"], row["expert_j"], row["category"]) for row in right]


def test_routing_overlap_toy_values():
    topk = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 1]])
    union, intersection, coactivation, jaccard = routing_overlap(topk, 0, 1)
    assert (union, intersection) == (4, 2)
    assert coactivation == 0.5
    assert jaccard == 0.5


def test_active_union_and_corouted_relative_l2_toy_values():
    class Signed(torch.nn.Module):
        def __init__(self, sign):
            super().__init__()
            self.sign = sign

        def forward(self, x):
            return self.sign * x

    rows = pair_output_statistics(
        [Signed(1.0), Signed(-1.0), Signed(1.0)], torch.ones(3, 1),
        torch.tensor([[0, 1], [0, 2], [1, 2]]),
        [{"expert_i": 0, "expert_j": 1}], torch.device("cpu"), chunk_size=2, min_corouted_tokens=1,
    )
    assert rows[0]["num_union_tokens"] == 3
    assert rows[0]["num_corouted_tokens"] == 1
    assert rows[0]["active_union_rel_l2"] == 2.0
    assert rows[0]["corouted_rel_l2"] == 2.0


def test_pair_batching_preserves_pair_metrics():
    class Scale(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return self.scale * x

    kwargs = dict(
        experts=[Scale(1.0), Scale(-1.0), Scale(0.5)],
        x_cpu=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        topk_cpu=torch.tensor([[0, 1], [0, 2], [1, 2], [0, 1]]),
        pair_rows=[{"expert_i": 0, "expert_j": 1}, {"expert_i": 0, "expert_j": 2}],
        device=torch.device("cpu"), chunk_size=2, min_corouted_tokens=1,
    )
    scalar_batches = pair_output_statistics(**kwargs, pair_batch_size=1)
    vector_batches = pair_output_statistics(**kwargs, pair_batch_size=2)
    for scalar, vector in zip(scalar_batches, vector_batches):
        for key in ("hc_mean_l2", "coactivation_rate", "routing_jaccard", "active_union_rel_l2", "corouted_rel_l2"):
            assert math.isclose(scalar[key], vector[key], abs_tol=1e-6)


def test_heldout_u_rates_sum_to_one():
    metrics = routing_metrics(torch.tensor([0, 0, 1, 2]), torch.tensor([[0, 1], [0, 2], [1, 2]]))
    assert math.isclose(metrics["u1_rate"] + metrics["u2_rate"], 1.0, abs_tol=1e-6)


def test_discovery_and_mapping_validation(tmp_path):
    for alpha, dirname in ((0.0, "a000"), (0.5, "a0500"), (1.0, "a100")):
        run = tmp_path / dirname
        run.mkdir()
        (run / "group_mapping_metadata.json").write_text(json.dumps({"grouping_method": "routing_aware", "alpha": alpha}))
        torch.save({"layer.0": torch.tensor([10, 10, 4, 4])}, run / "group_state_dict.pt")
    mappings = validate_mappings(discover_alpha_runs(tmp_path), expected_layers=1, expected_experts=4)
    assert all(mapping["layer.0"].unique().numel() == 2 for mapping in mappings.values())


def test_no_nonfinite_except_explicit_low_coroute_case():
    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    row = pair_output_statistics(
        [Identity(), Identity(), Identity()], torch.ones(3, 1),
        torch.tensor([[0, 1], [0, 2], [1, 2]]), [{"expert_i": 0, "expert_j": 1}],
        torch.device("cpu"), chunk_size=2, min_corouted_tokens=2,
    )[0]
    assert all(math.isfinite(float(row[key])) for key in ("hc_mean_l2", "coactivation_rate", "routing_jaccard", "active_union_rel_l2"))
    assert math.isnan(row["corouted_rel_l2"])


def test_run_moe_on_input_restores_three_dimensions_and_tuple_output():
    class ThreeDimensionalMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_shapes = []

        def forward(self, x):
            assert x.ndim == 3
            self.input_shapes.append(tuple(x.shape))
            return x + 1.0, torch.zeros(x.shape[:-1])

    moe = ThreeDimensionalMoE()
    x = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    actual = run_moe_on_input(moe, x, is_qwen=False, chunk_size=2)
    assert moe.input_shapes == [(1, 2, 2), (1, 2, 2), (1, 1, 2)]
    assert torch.equal(actual, x + 1.0)


def test_local_comparison_uses_the_same_x_tensor():
    seen = []

    class Recorder(torch.nn.Module):
        def forward(self, x):
            seen.append(x.data_ptr())
            return x

    x = torch.randn(3, 2)
    metrics = local_moe_metrics(Recorder(), Recorder(), x, is_qwen=False, chunk_size=16)
    assert seen == [seen[0], seen[0]]
    assert metrics["local_rel_l2"] == 0.0
