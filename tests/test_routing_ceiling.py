import torch

from hcsmoe.merging.pairwise_scores import (
    enumerate_canonical_partitions,
    exact_routing_partition,
    grouping_metrics,
    same_group_routing_count,
)


def test_canonical_set_partitions_match_stirling_s_8_4():
    partitions = enumerate_canonical_partitions(8, 4)
    assert len(partitions) == 1701
    assert len(set(partitions)) == 1701
    for labels in partitions:
        assert len(labels) == 8
        assert labels[0] == 0
        assert len(set(labels)) == 4
        assert set(labels) == set(range(4))
        for expert in range(1, len(labels)):
            assert labels[expert] <= max(labels[:expert]) + 1


def test_exact_routing_partition_finds_hand_computed_four_expert_optimum():
    # Upper-triangular routing mass sums to 20 tokens. Pairing (0, 1) and
    # (2, 3) retains 7 + 6 = 13 same-group top-2 routes, the unique optimum.
    routing_count = torch.tensor(
        [[0, 7, 2, 2], [7, 0, 1, 2], [2, 1, 0, 6], [2, 2, 6, 0]],
        dtype=torch.int64,
    )
    result = exact_routing_partition(routing_count, num_groups=2)
    assert torch.equal(result["labels"], torch.tensor([0, 0, 1, 1]))
    assert result["same_group_count"] == 13
    assert result["num_ties"] == 1


def test_routing_count_j_route_and_expected_unique_group_formula():
    routing_count = torch.tensor(
        [[0, 7, 2, 2], [7, 0, 1, 2], [2, 1, 0, 6], [2, 2, 6, 0]],
        dtype=torch.int64,
    )
    labels = torch.tensor([0, 0, 1, 1])
    num_tokens = 20
    j_route = same_group_routing_count(labels, routing_count) / num_tokens
    assert j_route == 13 / 20

    metrics = grouping_metrics(
        labels,
        output_distance=torch.zeros((4, 4)),
        routing_rate=routing_count.float() / num_tokens,
    )
    assert abs(metrics["same_group_routing_rate"] - j_route) < 1e-7
    assert abs(metrics["expected_unique_groups_per_token"] - (2 - j_route)) < 1e-7
