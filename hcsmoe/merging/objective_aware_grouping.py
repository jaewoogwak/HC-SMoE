"""Greedy grouping that optimizes token-level top-k unique-group reduction."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from hcsmoe.merging.pairwise_scores import canonical_groups, compute_topk_grouping_metrics


def _normalized_affinity(values: torch.Tensor, smaller_is_better: bool) -> torch.Tensor:
    minimum, maximum = values.min(), values.max()
    if torch.isclose(minimum, maximum):
        return torch.full_like(values, 0.5, dtype=torch.float32)
    if smaller_is_better:
        return (maximum - values) / (maximum - minimum)
    return (values - minimum) / (maximum - minimum)


def _labels_from_clusters(clusters: Sequence[tuple[int, ...]], num_experts: int) -> torch.Tensor:
    labels = torch.empty(num_experts, dtype=torch.long)
    for label, members in enumerate(sorted(clusters)):
        labels[list(members)] = label
    return labels


def _initial_routing_counts(topk_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
    counts = torch.zeros((num_experts, num_experts), dtype=torch.int64)
    top_k = topk_experts.shape[1]
    for left in range(top_k):
        for right in range(left + 1, top_k):
            flat = topk_experts[:, left] * num_experts + topk_experts[:, right]
            pair_counts = torch.bincount(flat, minlength=num_experts * num_experts)
            pair_counts = pair_counts.reshape(num_experts, num_experts).to(torch.int64)
            counts += pair_counts + pair_counts.T
    counts.fill_diagonal_(0)
    return counts


def cluster_routing_delta(
    topk_experts: torch.Tensor,
    cluster_a: Sequence[int],
    cluster_b: Sequence[int],
) -> float:
    """Return the exact one-step E[U] reduction from merging two clusters."""
    topk_experts = topk_experts.detach().to(device="cpu", dtype=torch.long)
    if topk_experts.ndim != 2 or topk_experts.shape[0] == 0:
        raise ValueError("topk_experts must be a non-empty matrix")
    if not cluster_a or not cluster_b or set(cluster_a).intersection(cluster_b):
        raise ValueError("clusters must be non-empty and disjoint")
    members_a = torch.tensor(sorted(cluster_a), dtype=torch.long)
    members_b = torch.tensor(sorted(cluster_b), dtype=torch.long)
    touches_a = torch.isin(topk_experts, members_a).any(dim=1)
    touches_b = torch.isin(topk_experts, members_b).any(dim=1)
    return float(torch.count_nonzero(touches_a & touches_b)) / topk_experts.shape[0]


def _validate_inputs(output_distance: torch.Tensor, topk_experts: torch.Tensor, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    output_distance = output_distance.detach().to(device="cpu", dtype=torch.float32)
    topk_experts = topk_experts.detach().to(device="cpu", dtype=torch.long)
    if output_distance.ndim != 2 or output_distance.shape[0] != output_distance.shape[1]:
        raise ValueError("output_distance must be square")
    if not torch.isfinite(output_distance).all() or not torch.allclose(output_distance, output_distance.T):
        raise ValueError("output_distance must be finite and symmetric")
    num_experts = output_distance.shape[0]
    if not 1 <= num_groups <= num_experts:
        raise ValueError("num_groups must be in [1, num_experts]")
    if topk_experts.ndim != 2 or topk_experts.shape[0] == 0 or topk_experts.shape[1] == 0:
        raise ValueError("topk_experts must be a non-empty [num_tokens, top_k] tensor")
    if torch.any(topk_experts < 0) or torch.any(topk_experts >= num_experts):
        raise ValueError("topk_experts contains an out-of-range expert ID")
    sorted_experts = torch.sort(topk_experts, dim=-1).values
    if torch.any(sorted_experts[:, 1:] == sorted_experts[:, :-1]):
        raise ValueError("topk_experts must be distinct within each token")
    return output_distance, topk_experts


def objective_aware_grouping(
    output_distance: torch.Tensor,
    topk_experts: torch.Tensor,
    num_groups: int,
    alpha: float,
) -> Dict[str, object]:
    """Greedily merge clusters with output affinity and exact unique-group gain."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    output_distance, topk_experts = _validate_inputs(output_distance, topk_experts, num_groups)
    num_experts = output_distance.shape[0]
    num_tokens = topk_experts.shape[0]

    clusters = [(expert,) for expert in range(num_experts)]
    expert_ids = torch.arange(num_experts, dtype=torch.long)
    touches = (topk_experts.unsqueeze(-1) == expert_ids).any(dim=1).T.contiguous()
    output_cost = output_distance.clone()
    delta_counts = _initial_routing_counts(topk_experts, num_experts)
    merge_trace = []

    while len(clusters) > num_groups:
        order = sorted(range(len(clusters)), key=lambda index: clusters[index])
        clusters = [clusters[index] for index in order]
        index_tensor = torch.tensor(order, dtype=torch.long)
        touches = touches[index_tensor]
        output_cost = output_cost[index_tensor][:, index_tensor]
        delta_counts = delta_counts[index_tensor][:, index_tensor]

        left, right = torch.triu_indices(len(clusters), len(clusters), offset=1)
        candidate_output_cost = output_cost[left, right]
        candidate_delta_counts = delta_counts[left, right]
        output_affinity = _normalized_affinity(candidate_output_cost, smaller_is_better=True)
        routing_affinity = _normalized_affinity(candidate_delta_counts.float() / num_tokens, smaller_is_better=False)
        scores = alpha * output_affinity + (1.0 - alpha) * routing_affinity
        best_score = scores.max()
        tied = torch.nonzero(scores == best_score, as_tuple=False).flatten().tolist()
        selected = min(
            tied,
            key=lambda index: (clusters[int(left[index])], clusters[int(right[index])]),
        )
        i, j = int(left[selected]), int(right[selected])
        cluster_a, cluster_b = clusters[i], clusters[j]
        labels_before = _labels_from_clusters(clusters, num_experts)
        mean_before = compute_topk_grouping_metrics(labels_before, topk_experts)["mean_unique_groups"]
        delta_route = float(candidate_delta_counts[selected]) / num_tokens

        surviving = [index for index in range(len(clusters)) if index not in (i, j)]
        merged_cluster = tuple(sorted(cluster_a + cluster_b))
        merged_touch = touches[i] | touches[j]
        merged_size = len(merged_cluster)
        new_clusters = [clusters[index] for index in surviving] + [merged_cluster]
        new_touches = torch.cat((touches[surviving], merged_touch.unsqueeze(0)), dim=0)
        surviving_tensor = torch.tensor(surviving, dtype=torch.long)
        old_output_cost = output_cost[surviving_tensor][:, surviving_tensor]
        old_delta_counts = delta_counts[surviving_tensor][:, surviving_tensor]
        new_size = len(surviving) + 1
        new_output_cost = torch.zeros((new_size, new_size), dtype=torch.float32)
        new_delta_counts = torch.zeros((new_size, new_size), dtype=torch.int64)
        new_output_cost[:-1, :-1] = old_output_cost
        new_delta_counts[:-1, :-1] = old_delta_counts
        output_to_merged = (
            len(cluster_a) * output_cost[i, surviving_tensor]
            + len(cluster_b) * output_cost[j, surviving_tensor]
        ) / merged_size
        delta_to_merged = torch.count_nonzero(touches[surviving_tensor] & merged_touch, dim=1).to(torch.int64)
        new_output_cost[:-1, -1] = output_to_merged
        new_output_cost[-1, :-1] = output_to_merged
        new_delta_counts[:-1, -1] = delta_to_merged
        new_delta_counts[-1, :-1] = delta_to_merged

        labels_after = _labels_from_clusters(new_clusters, num_experts)
        mean_after = compute_topk_grouping_metrics(labels_after, topk_experts)["mean_unique_groups"]
        if abs((mean_before - mean_after) - delta_route) > 1e-6:
            raise AssertionError("selected merge delta disagrees with token-level unique-group reduction")
        merge_trace.append(
            {
                "step": len(merge_trace),
                "num_groups_before": len(clusters),
                "cluster_A": list(cluster_a),
                "cluster_B": list(cluster_b),
                "delta_route": delta_route,
                "max_delta_route": float(candidate_delta_counts.max()) / num_tokens,
                "D_out": float(candidate_output_cost[selected]),
                "S_route": float(routing_affinity[selected]),
                "S_out": float(output_affinity[selected]),
                "hybrid_merge_score": float(scores[selected]),
                "mean_unique_groups_before": mean_before,
                "mean_unique_groups_after": mean_after,
            }
        )
        clusters, touches = new_clusters, new_touches
        output_cost, delta_counts = new_output_cost, new_delta_counts

    labels = _labels_from_clusters(clusters, num_experts)
    return {
        "labels": labels,
        "groups": canonical_groups(labels),
        "metrics": compute_topk_grouping_metrics(labels, topk_experts, output_distance),
        "merge_trace": merge_trace,
    }
