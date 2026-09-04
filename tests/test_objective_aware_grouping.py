import torch

from hcsmoe.merging.clustering import hierarchical_clustering_from_pairwise_distance
from hcsmoe.merging.objective_aware_grouping import (
    cluster_routing_delta,
    objective_aware_grouping,
)
from hcsmoe.merging.pairwise_scores import (
    accumulate_corouting,
    build_hybrid_score_matrices,
    compute_topk_grouping_metrics,
    partitions_equal,
    select_topk_experts,
)


def test_generic_topk_unique_group_histogram_and_mean():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
    topk_experts = torch.tensor([
        [0, 4, 8, 10],
        [0, 1, 4, 8],
        [0, 1, 4, 5],
        [0, 1, 2, 4],
        [0, 1, 2, 3],
    ])
    metrics = compute_topk_grouping_metrics(labels, topk_experts)
    assert metrics["unique_group_count"] == {"1": 1, "2": 2, "3": 1, "4": 1}
    assert metrics["unique_group_rate"] == {"1": 0.2, "2": 0.4, "3": 0.2, "4": 0.2}
    assert abs(metrics["mean_unique_groups"] - 2.4) < 1e-6


def test_singleton_delta_matches_pairwise_routing_rate():
    logits = torch.tensor([
        [9.0, 8.0, 7.0, 6.0, 0.0],
        [9.0, 8.0, 1.0, 7.0, 6.0],
        [1.0, 9.0, 8.0, 7.0, 6.0],
    ])
    topk_experts = select_topk_experts(logits, top_k=4)
    routing_count = torch.zeros((5, 5), dtype=torch.int64)
    usage_count = torch.zeros(5, dtype=torch.int64)
    accumulate_corouting(routing_count, usage_count, logits, top_k=4, selected_experts=topk_experts)
    assert cluster_routing_delta(topk_experts, (0,), (1,)) == routing_count[0, 1].item() / len(logits)


def test_cluster_delta_equals_direct_unique_group_reduction():
    topk_experts = torch.tensor([[0, 1, 2, 3], [0, 2, 4, 5], [1, 3, 4, 5]])
    labels_before = torch.tensor([0, 1, 2, 3, 4, 5])
    labels_after = torch.tensor([0, 0, 2, 3, 4, 5])
    before = compute_topk_grouping_metrics(labels_before, topk_experts)["mean_unique_groups"]
    after = compute_topk_grouping_metrics(labels_after, topk_experts)["mean_unique_groups"]
    assert abs(cluster_routing_delta(topk_experts, (0,), (1,)) - (before - after)) < 1e-6


def test_alpha_zero_selects_maximum_exact_routing_gain_each_step():
    topk_experts = torch.tensor([
        [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4],
        [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5],
    ])
    distance = torch.ones((6, 6)) - torch.eye(6)
    result = objective_aware_grouping(distance, topk_experts, num_groups=3, alpha=0.0)
    assert len(result["merge_trace"]) == 3
    for merge in result["merge_trace"]:
        assert merge["delta_route"] == merge["max_delta_route"]


def test_alpha_one_matches_average_linkage_output_partition():
    fingerprints = torch.tensor([[0.0], [0.1], [3.0], [3.1], [7.0], [7.1]])
    distance = torch.cdist(fingerprints, fingerprints)
    topk_experts = torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]])
    objective = objective_aware_grouping(distance, topk_experts, num_groups=3, alpha=1.0)
    hybrid = build_hybrid_score_matrices(distance, torch.zeros_like(distance), alpha=1.0)
    legacy, _ = hierarchical_clustering_from_pairwise_distance(
        hybrid["hybrid_distance"], 3, method="average", features_for_centers=fingerprints
    )
    assert partitions_equal(objective["labels"], legacy)


def test_top4_routing_contract_and_metric_bounds():
    logits = torch.tensor([
        [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        [4.0, 9.0, 8.0, 7.0, 6.0, 5.0],
    ])
    selected = select_topk_experts(logits, top_k=4)
    routing_count = torch.zeros((6, 6), dtype=torch.int64)
    usage_count = torch.zeros(6, dtype=torch.int64)
    accumulate_corouting(routing_count, usage_count, logits, top_k=4, selected_experts=selected)
    assert torch.triu(routing_count, diagonal=1).sum().item() == 6 * logits.shape[0]
    assert usage_count.sum().item() == 4 * logits.shape[0]
    metrics = compute_topk_grouping_metrics(torch.arange(6), selected)
    assert sum(metrics["unique_group_rate"].values()) == 1.0
    assert 1 <= metrics["mean_unique_groups"] <= 4
