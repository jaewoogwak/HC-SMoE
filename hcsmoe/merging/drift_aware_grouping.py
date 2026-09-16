"""Generic greedy grouping for output- and downstream-routing-drift costs."""
from __future__ import annotations

import itertools
from typing import Any, Callable, Iterable, Optional


Group = tuple[int, ...]
Partition = tuple[Group, ...]


def canonical_partition(groups: Iterable[Iterable[int]]) -> Partition:
    partition = tuple(sorted(tuple(sorted(group)) for group in groups))
    flattened = [expert for group in partition for expert in group]
    if any(not group for group in partition) or len(flattened) != len(set(flattened)):
        raise ValueError("groups must be non-empty and disjoint")
    return partition


def merge_partition(partition: Partition, left: Group, right: Group) -> Partition:
    if left == right or left not in partition or right not in partition:
        raise ValueError("candidate groups must be distinct members of the partition")
    return canonical_partition(
        [group for group in partition if group not in (left, right)] + [tuple(sorted(left + right))]
    )


def minmax_normalize(values: list[float], eps: float = 1e-12) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= eps:
        return [0.0] * len(values)
    return [(value - low) / (high - low + eps) for value in values]


def labels_from_partition(partition: Partition, num_experts: int) -> list[int]:
    labels = [-1] * num_experts
    for label, group in enumerate(canonical_partition(partition)):
        for expert in group:
            if not 0 <= expert < num_experts:
                raise ValueError(f"expert id {expert} is outside [0, {num_experts})")
            labels[expert] = label
    if any(label < 0 for label in labels):
        raise ValueError("partition does not cover every expert")
    return labels


def greedy_drift_aware_grouping(
    initial_groups: Iterable[Iterable[int]],
    target_num_groups: int,
    evaluate_partition: Callable[[Partition], dict[str, Any]],
    lambda_route: float,
    eps: float = 1e-12,
    usage_frequencies: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Greedily merge the minimum normalized drift-cost partition."""
    if not 0.0 <= lambda_route <= 1.0:
        raise ValueError("lambda_route must be in [0, 1]")
    partition = canonical_partition(initial_groups)
    if not 1 <= target_num_groups <= len(partition):
        raise ValueError("target_num_groups must be between one and the initial group count")
    trace: list[dict[str, Any]] = []

    while len(partition) > target_num_groups:
        started_groups = partition
        candidate_rows: list[dict[str, Any]] = []
        for left, right in itertools.combinations(started_groups, 2):
            candidate = merge_partition(started_groups, left, right)
            measured = evaluate_partition(candidate)
            c_out = float(measured["c_out"])
            c_route = measured.get("c_route")
            candidate_rows.append({
                "group_a": list(left),
                "group_b": list(right),
                "resulting_group": list(sorted(left + right)),
                "partition": [list(group) for group in candidate],
                "c_out": c_out,
                "c_route": None if c_route is None else float(c_route),
                **{key: value for key, value in measured.items() if key not in {"c_out", "c_route"}},
                "_partition": candidate,
                "_tie_key": (left, right),
            })

        out_norm = minmax_normalize([row["c_out"] for row in candidate_rows], eps)
        route_available = all(row["c_route"] is not None for row in candidate_rows)
        route_norm = (
            minmax_normalize([float(row["c_route"]) for row in candidate_rows], eps)
            if route_available else [None] * len(candidate_rows)
        )
        for row, normalized_out, normalized_route in zip(candidate_rows, out_norm, route_norm):
            row["normalized_c_out"] = normalized_out
            row["normalized_c_route"] = normalized_route
            row["c_total"] = (
                normalized_out
                if normalized_route is None
                else (1.0 - lambda_route) * normalized_out + lambda_route * normalized_route
            )
        selected = min(candidate_rows, key=lambda row: (row["c_total"], row["_tie_key"]))
        partition = selected["_partition"]
        for row in candidate_rows:
            row["selected"] = row is selected
            row.pop("_partition")
            row.pop("_tie_key")
        trace.append({
            "step": len(trace),
            "current_groups": [list(group) for group in started_groups],
            "usage_frequencies": usage_frequencies,
            "candidates": candidate_rows,
            "selected_candidate": {
                "group_a": selected["group_a"],
                "group_b": selected["group_b"],
                "resulting_group": selected["resulting_group"],
                "c_out": selected["c_out"],
                "c_route": selected["c_route"],
                "normalized_c_out": selected["normalized_c_out"],
                "normalized_c_route": selected["normalized_c_route"],
                "c_total": selected["c_total"],
            },
        })
    return {"groups": partition, "merge_trace": trace}


__all__ = [
    "Group",
    "Partition",
    "canonical_partition",
    "greedy_drift_aware_grouping",
    "labels_from_partition",
    "merge_partition",
    "minmax_normalize",
]
