import os

import torch

from hcsmoe.merging.clustering import (
    group_experts_by_clustering,
    hierarchical_clustering_from_pairwise_distance,
)
from hcsmoe.merging.pairwise_scores import (
    build_hybrid_score_matrices,
    build_output_score_matrices,
    canonical_groups,
    changed_expert_count,
    grouping_metrics,
    load_or_compute_pairwise_scores,
    partitions_equal,
)


def _metadata(seed=0):
    return {
        "model_name": "tiny-mixtral",
        "dataset": "c4",
        "num_blocks": 1,
        "block_size": 4,
        "num_calibration_tokens": 4,
        "num_experts": 4,
        "top_k": 2,
        "chunk_size": 2,
        "seed": seed,
        "output_definition": "mean expert output over common MoE inputs",
        "routing_definition": "token-level top-k co-activation count",
    }


def _layers():
    fingerprints = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
    scores = build_output_score_matrices(fingerprints)
    routing_count = torch.tensor([
        [0, 2, 1, 0], [2, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0],
    ], dtype=torch.int64)
    scores.update({
        "routing_count": routing_count,
        "routing_rate": routing_count.float() / 4,
        "usage_count": torch.tensor([3, 3, 1, 1], dtype=torch.int64),
        "num_tokens": 4,
    })
    return {"model.layers.0.block_sparse_moe": scores}


def test_hybrid_score_normalization_endpoints_and_constant_matrix():
    distance = torch.tensor([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])
    routing = torch.tensor([[0.0, 0.1, 0.3], [0.1, 0.0, 0.2], [0.3, 0.2, 0.0]])
    result = build_hybrid_score_matrices(distance, routing, alpha=0.25)
    assert result["output_score"][0, 1] == 1
    assert result["output_score"][0, 2] == 0
    assert result["routing_score"][0, 2] == 1
    assert result["routing_score"][0, 1] == 0
    assert torch.allclose(result["hybrid_distance"], result["hybrid_distance"].T)
    assert torch.equal(result["hybrid_distance"].diag(), torch.zeros(3))
    assert torch.isfinite(result["hybrid_distance"]).all()
    output_endpoint = build_hybrid_score_matrices(distance, routing, alpha=1.0)
    expected_output_distance = 1.0 - output_endpoint["output_score"]
    expected_output_distance.fill_diagonal_(0)
    assert torch.allclose(output_endpoint["hybrid_distance"], expected_output_distance)
    routing_endpoint = build_hybrid_score_matrices(distance, routing, alpha=0.0)
    expected_routing_distance = 1.0 - routing_endpoint["routing_score"]
    expected_routing_distance.fill_diagonal_(0)
    assert torch.allclose(routing_endpoint["hybrid_distance"], expected_routing_distance)
    constant = torch.ones((3, 3)) - torch.eye(3)
    neutral = build_hybrid_score_matrices(constant, constant, alpha=0.5)
    mask = ~torch.eye(3, dtype=torch.bool)
    assert torch.all(neutral["output_score"][mask] == 0.5)
    assert torch.all(neutral["routing_score"][mask] == 0.5)


def test_hybrid_alpha_validation_and_precomputed_alpha_one_partition():
    distance = _layers()["model.layers.0.block_sparse_moe"]["output_distance"]
    routing = _layers()["model.layers.0.block_sparse_moe"]["routing_rate"]
    for alpha in (-0.01, 1.01):
        try:
            build_hybrid_score_matrices(distance, routing, alpha)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid alpha should raise ValueError")

    fingerprints = _layers()["model.layers.0.block_sparse_moe"]["output_fingerprint"]
    _, legacy = group_experts_by_clustering(
        model="mixtral", num_groups=2, cluster="hierarchical", linkage="average",
        hierarchical_stopping_metric="silhouette", num_experts=4, experts=fingerprints,
    )
    hybrid = build_hybrid_score_matrices(distance, routing, alpha=1.0)
    precomputed, _ = hierarchical_clustering_from_pairwise_distance(
        hybrid["hybrid_distance"], 2, method="average", features_for_centers=fingerprints
    )
    assert partitions_equal(legacy, precomputed)


def test_label_invariant_metrics_and_mixtral_top2_formula():
    labels = [0, 0, 1, 1]
    permuted = [7, 7, 3, 3]
    scores = _layers()["model.layers.0.block_sparse_moe"]
    assert canonical_groups(labels) == [[0, 1], [2, 3]]
    assert partitions_equal(labels, permuted)
    assert changed_expert_count(labels, permuted) == 0
    assert grouping_metrics(labels, scores["output_distance"], scores["routing_rate"]) == grouping_metrics(
        permuted, scores["output_distance"], scores["routing_rate"]
    )
    metrics = grouping_metrics(labels, scores["output_distance"], scores["routing_rate"])
    assert metrics["same_group_routing_rate"] == 0.5
    assert metrics["expected_unique_groups_per_token"] == 1.5


class _FakeGrouper:
    sparse_layer_indices = [0]
    num_experts = 4
    d_model = 2

    def __init__(self):
        self.calls = 0

    def compute_pairwise_score_matrices(self, model, dataloader, chunk_size):
        self.calls += 1
        return _layers()


def test_pairwise_cache_hit_miss_invalid_and_forced_recompute(tmp_path):
    cache_path = os.path.join(tmp_path, "scores.pt")
    grouper = _FakeGrouper()
    metadata = _metadata()
    first = load_or_compute_pairwise_scores(cache_path, None, grouper, None, metadata, chunk_size=2)
    assert grouper.calls == 1 and os.path.exists(cache_path)
    second = load_or_compute_pairwise_scores(cache_path, None, grouper, None, metadata, chunk_size=999)
    assert grouper.calls == 1
    assert torch.equal(first["layers"]["model.layers.0.block_sparse_moe"]["routing_count"],
                       second["layers"]["model.layers.0.block_sparse_moe"]["routing_count"])
    try:
        load_or_compute_pairwise_scores(cache_path, None, grouper, None, _metadata(seed=1), chunk_size=2)
    except ValueError as error:
        assert "cached seed" in str(error)
    else:
        raise AssertionError("invalid cache metadata should not be overwritten")
    load_or_compute_pairwise_scores(cache_path, None, grouper, None, metadata, chunk_size=2, recompute=True)
    assert grouper.calls == 2
