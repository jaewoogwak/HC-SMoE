import torch

from hcsmoe.merging.clustering import hierarchical_clustering, pairwise_distances
from hcsmoe.merging.routing_aware_grouping import exact_routing_delta, routing_aware_grouping, routing_metrics


def _same_partition(left, right):
    return bool(torch.all((left[:, None] == left[None, :]) == (right[:, None] == right[None, :])))


def test_exact_delta_is_unique_group_reduction():
    topk = torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3], [0, 3]])
    labels_before = torch.arange(4)
    labels_after = torch.tensor([0, 0, 1, 2])
    expected = routing_metrics(labels_before, topk)["mean_unique_groups"] - routing_metrics(labels_after, topk)["mean_unique_groups"]
    assert abs(exact_routing_delta(topk, (0,), (1,)) - expected) < 1e-6


def test_alpha_one_always_selects_maximum_exact_gain():
    distance = torch.tensor([[0., 1., 2., 3.], [1., 0., 2., 3.], [2., 2., 0., 1.], [3., 3., 1., 0.]])
    topk = torch.tensor([[0, 1], [0, 1], [0, 2], [1, 2], [2, 3]])
    result = routing_aware_grouping(distance, topk, num_groups=2, alpha=1.0)
    for step in result["merge_trace"]:
        assert abs(step["routing_delta"] - step["max_routing_delta"]) < 1e-6


def test_alpha_zero_matches_average_linkage_hcsmoe_partition():
    features = torch.tensor([[0., 0.], [0.2, 0.], [3., 0.], [3.3, 0.], [8., 0.]])
    distance = pairwise_distances(features, method="average")
    distance.fill_diagonal_(0.)
    topk = torch.tensor([[0, 1], [2, 3], [0, 4]])
    actual = routing_aware_grouping(distance, topk, num_groups=3, alpha=0.0)["labels"]
    expected, _ = hierarchical_clustering(features, n_clusters=3, method="average")
    assert _same_partition(actual, expected)


def test_deterministic_lexicographic_tie_breaking():
    distance = torch.zeros(4, 4)
    topk = torch.tensor([[0, 1], [2, 3]])
    first = routing_aware_grouping(distance, topk, num_groups=3, alpha=0.5)["merge_trace"][0]
    second = routing_aware_grouping(distance, topk, num_groups=3, alpha=0.5)["merge_trace"][0]
    assert (first["cluster_a"], first["cluster_b"]) == ([0], [1])
    assert first == second


def test_final_group_count():
    distance = torch.cdist(torch.arange(8, dtype=torch.float32)[:, None], torch.arange(8, dtype=torch.float32)[:, None])
    topk = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    result = routing_aware_grouping(distance, topk, num_groups=4, alpha=1.0)
    assert len(result["groups"]) == 4
    assert all(group for group in result["groups"])
