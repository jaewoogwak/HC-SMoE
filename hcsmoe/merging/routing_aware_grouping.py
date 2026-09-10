"""Greedy grouping over current Mixtral expert clusters.

The routing objective is intentionally evaluated on *current* clusters.  It
is not a static expert-pair co-activation matrix followed by linkage.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch


def _normalise(values: torch.Tensor, *, smaller_is_better: bool) -> torch.Tensor:
    low, high = values.min(), values.max()
    if torch.isclose(low, high):
        return torch.full_like(values, 0.5, dtype=torch.float32)
    return (high - values if smaller_is_better else values - low) / (high - low)


def _labels(clusters: Sequence[tuple[int, ...]], num_experts: int) -> torch.Tensor:
    result = torch.empty(num_experts, dtype=torch.long)
    for group_id, cluster in enumerate(sorted(clusters)):
        result[list(cluster)] = group_id
    return result


def routing_metrics(labels: torch.Tensor, topk_experts: torch.Tensor) -> Dict[str, float]:
    """Token-level locality metrics for a routing assignment."""
    grouped = labels[topk_experts]
    # ``torch.unique(..., dim=1)`` cannot give per-row counts; top-k is tiny.
    counts = torch.tensor([torch.unique(row).numel() for row in grouped], dtype=torch.long)
    metrics = {
        "mean_unique_groups": float(counts.float().mean()),
        "u1_rate": float((counts == 1).float().mean()),
        "num_tokens": int(counts.numel()),
    }
    for value in range(2, topk_experts.shape[1] + 1):
        metrics[f"u{value}_rate"] = float((counts == value).float().mean())
    return metrics


def exact_routing_delta(
    topk_experts: torch.Tensor,
    cluster_a: Iterable[int],
    cluster_b: Iterable[int],
) -> float:
    """Exact reduction in mean unique groups after merging two groups."""
    a = torch.tensor(sorted(cluster_a), dtype=torch.long)
    b = torch.tensor(sorted(cluster_b), dtype=torch.long)
    if not len(a) or not len(b) or set(a.tolist()) & set(b.tolist()):
        raise ValueError("clusters must be non-empty and disjoint")
    touches_a = torch.isin(topk_experts, a).any(dim=1)
    touches_b = torch.isin(topk_experts, b).any(dim=1)
    return float((touches_a & touches_b).float().mean())


def _validate(output_distance: torch.Tensor, topk_experts: torch.Tensor, num_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
    distance = output_distance.detach().to(device="cpu", dtype=torch.float32)
    routing = topk_experts.detach().to(device="cpu", dtype=torch.long)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("output_distance must be square")
    if not torch.isfinite(distance).all() or not torch.allclose(distance, distance.T):
        raise ValueError("output_distance must be finite and symmetric")
    if not 1 <= num_groups <= distance.shape[0]:
        raise ValueError("num_groups must be between one and the expert count")
    if routing.ndim != 2 or not routing.shape[0] or not routing.shape[1]:
        raise ValueError("topk_experts must be non-empty [num_tokens, top_k]")
    if routing.min() < 0 or routing.max() >= distance.shape[0]:
        raise ValueError("topk_experts contains an invalid expert id")
    if routing.shape[1] > 1 and torch.any(torch.sort(routing, dim=1).values[:, 1:] == torch.sort(routing, dim=1).values[:, :-1]):
        raise ValueError("each token must select distinct experts")
    return distance, routing


def routing_aware_grouping(
    output_distance: torch.Tensor,
    topk_experts: torch.Tensor,
    num_groups: int,
    alpha: float = 1.0,
) -> Dict[str, object]:
    """Merge current groups using exact routing gain and average-linkage output affinity.

    ``alpha=1`` is routing-only, ``alpha=0`` is original average-linkage
    output HC, and intermediate values use per-step min/max normalisation.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    output_cost, topk_experts = _validate(output_distance, topk_experts, num_groups)
    num_experts, num_tokens = output_cost.shape[0], topk_experts.shape[0]
    clusters = [(expert,) for expert in range(num_experts)]
    expert_ids = torch.arange(num_experts)
    touches = (topk_experts.unsqueeze(-1) == expert_ids).any(dim=1).T.contiguous()
    trace = []

    while len(clusters) > num_groups:
        order = sorted(range(len(clusters)), key=lambda index: clusters[index])
        clusters = [clusters[index] for index in order]
        index = torch.tensor(order)
        touches, output_cost = touches[index], output_cost[index][:, index]
        left, right = torch.triu_indices(len(clusters), len(clusters), offset=1)
        deltas = (touches[left] & touches[right]).sum(dim=1).float() / num_tokens
        costs = output_cost[left, right]
        routing_score = _normalise(deltas, smaller_is_better=False)
        output_score = _normalise(costs, smaller_is_better=True)
        scores = alpha * routing_score + (1.0 - alpha) * output_score
        best = scores.max()
        tied = torch.nonzero(scores == best, as_tuple=False).flatten().tolist()
        selected = min(tied, key=lambda k: (clusters[int(left[k])], clusters[int(right[k])]))
        i, j = int(left[selected]), int(right[selected])
        a, b = clusters[i], clusters[j]
        before_labels = _labels(clusters, num_experts)
        before = routing_metrics(before_labels, topk_experts)["mean_unique_groups"]
        delta = float(deltas[selected])

        keep = [k for k in range(len(clusters)) if k not in (i, j)]
        keep_index = torch.tensor(keep)
        merged = tuple(sorted(a + b))
        merged_touch = touches[i] | touches[j]
        new_clusters = [clusters[k] for k in keep] + [merged]
        new_touches = torch.cat((touches[keep_index], merged_touch.unsqueeze(0)))
        new_cost = torch.zeros((len(new_clusters), len(new_clusters)), dtype=torch.float32)
        if keep:
            new_cost[:-1, :-1] = output_cost[keep_index][:, keep_index]
            to_merged = (len(a) * output_cost[i, keep_index] + len(b) * output_cost[j, keep_index]) / len(merged)
            new_cost[:-1, -1] = to_merged
            new_cost[-1, :-1] = to_merged

        after_labels = _labels(new_clusters, num_experts)
        after = routing_metrics(after_labels, topk_experts)["mean_unique_groups"]
        if abs((before - after) - delta) > 1e-6:
            raise AssertionError("routing delta does not equal the unique-group reduction")
        trace.append({
            "step": len(trace), "num_groups_before": len(clusters),
            "cluster_a": list(a), "cluster_b": list(b),
            "routing_delta": delta, "max_routing_delta": float(deltas.max()),
            "routing_score": float(routing_score[selected]),
            "output_score": float(output_score[selected]),
            "merge_score": float(scores[selected]),
            "mean_unique_groups_before": before,
            "mean_unique_groups_after": after,
        })
        clusters, touches, output_cost = new_clusters, new_touches, new_cost

    labels = _labels(clusters, num_experts)
    return {"labels": labels, "groups": [list(group) for group in sorted(clusters)],
            "metrics": routing_metrics(labels, topk_experts), "merge_trace": trace}
