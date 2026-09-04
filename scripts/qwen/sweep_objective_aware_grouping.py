#!/usr/bin/env python3
"""Offline Qwen top-4 unique-group-aware and legacy pairwise-HC sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hcsmoe.merging.clustering import hierarchical_clustering_from_pairwise_distance
from hcsmoe.merging.objective_aware_grouping import objective_aware_grouping
from hcsmoe.merging.pairwise_scores import (
    build_hybrid_score_matrices,
    canonical_groups,
    compute_topk_grouping_metrics,
    partitions_equal,
    same_group_routing_count,
    validate_pairwise_scores,
)


METHOD_OBJECTIVE = "objective_aware"
METHOD_LEGACY = "legacy_pairwise_hc"
CSV_FIELDS = (
    "method", "alpha", "layer", "groups", "group_size_pattern", "num_groups",
    "unique_group_count_1", "unique_group_count_2", "unique_group_count_3", "unique_group_count_4",
    "unique_group_rate_1", "unique_group_rate_2", "unique_group_rate_3", "unique_group_rate_4",
    "mean_unique_groups", "mean_intragroup_output_distance", "initial_pairwise_same_group_routing_mass",
)


def parse_alphas(value: str) -> list[float]:
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas or any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("--alphas must be non-empty values in [0, 1]")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("--alphas must not contain duplicates")
    return alphas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_path", type=Path, required=True)
    parser.add_argument("--alphas", type=parse_alphas, required=True)
    parser.add_argument("--num_groups", type=int, default=30)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("layers"), dict):
        raise ValueError("Invalid pairwise score artifact")
    validate_pairwise_scores(payload["layers"])
    return payload


def labels_for_static_hc(scores: dict[str, Any], alpha: float, num_groups: int) -> torch.Tensor:
    hybrid = build_hybrid_score_matrices(scores["output_distance"], scores["routing_rate"], alpha)
    labels, _ = hierarchical_clustering_from_pairwise_distance(
        hybrid["hybrid_distance"],
        num_groups,
        method="average",
        features_for_centers=scores["output_fingerprint"],
    )
    return labels.detach().to(device="cpu", dtype=torch.long)


def result_record(
    method: str,
    alpha: float,
    layer: int,
    labels: torch.Tensor,
    scores: dict[str, Any],
) -> dict[str, Any]:
    metrics = compute_topk_grouping_metrics(labels, scores["topk_experts"], scores["output_distance"])
    num_tokens = int(scores["num_tokens"])
    return {
        "method": method,
        "alpha": alpha,
        "layer": layer,
        "groups": canonical_groups(labels),
        "group_size_pattern": sorted((len(group) for group in canonical_groups(labels)), reverse=True),
        "num_groups": len(set(int(label) for label in labels)),
        **{f"unique_group_count_{groups}": metrics["unique_group_count"][str(groups)] for groups in range(1, 5)},
        **{f"unique_group_rate_{groups}": metrics["unique_group_rate"][str(groups)] for groups in range(1, 5)},
        "mean_unique_groups": metrics["mean_unique_groups"],
        "mean_intragroup_output_distance": metrics["mean_intragroup_output_distance"],
        "initial_pairwise_same_group_routing_mass": same_group_routing_count(labels, scores["routing_count"]) / num_tokens,
    }


def summarize(records: list[dict[str, Any]], reference: dict[str, Any] | None) -> dict[str, Any]:
    summary = {
        "mean_unique_group_rate_1": mean(record["unique_group_rate_1"] for record in records),
        "mean_unique_group_rate_2": mean(record["unique_group_rate_2"] for record in records),
        "mean_unique_group_rate_3": mean(record["unique_group_rate_3"] for record in records),
        "mean_unique_group_rate_4": mean(record["unique_group_rate_4"] for record in records),
        "mean_unique_groups": mean(record["mean_unique_groups"] for record in records),
        "mean_intragroup_output_distance": mean(record["mean_intragroup_output_distance"] for record in records),
    }
    if reference is not None:
        summary.update({
            "delta_mean_unique_groups_vs_alpha1": summary["mean_unique_groups"] - reference["mean_unique_groups"],
            "delta_rate1_vs_alpha1": summary["mean_unique_group_rate_1"] - reference["mean_unique_group_rate_1"],
            "delta_rate2_vs_alpha1": summary["mean_unique_group_rate_2"] - reference["mean_unique_group_rate_2"],
            "delta_rate3_vs_alpha1": summary["mean_unique_group_rate_3"] - reference["mean_unique_group_rate_3"],
            "delta_rate4_vs_alpha1": summary["mean_unique_group_rate_4"] - reference["mean_unique_group_rate_4"],
            "output_distance_change_vs_alpha1": (
                summary["mean_intragroup_output_distance"] - reference["mean_intragroup_output_distance"]
            ),
        })
    else:
        summary.update({
            "delta_mean_unique_groups_vs_alpha1": 0.0,
            "delta_rate1_vs_alpha1": 0.0,
            "delta_rate2_vs_alpha1": 0.0,
            "delta_rate3_vs_alpha1": 0.0,
            "delta_rate4_vs_alpha1": 0.0,
            "output_distance_change_vs_alpha1": 0.0,
        })
    return summary


def write_plots(output_dir: Path, alpha_summaries: list[dict[str, Any]]) -> None:
    objective = [row for row in alpha_summaries if row["method"] == METHOD_OBJECTIVE]
    objective.sort(key=lambda row: row["alpha"], reverse=True)
    alphas = [row["alpha"] for row in objective]
    x = list(range(len(alphas)))
    labels = [f"{alpha:g}" for alpha in alphas]
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759"]

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bottom = [0.0] * len(objective)
    for groups, color in zip(range(1, 5), colors):
        values = [row[f"mean_unique_group_rate_{groups}"] for row in objective]
        axis.bar(x, values, bottom=bottom, color=color, label=f"U={groups}")
        bottom = [left + right for left, right in zip(bottom, values)]
    axis.set(xlabel="Objective-aware alpha", ylabel="Mean token-layer proportion", title="Qwen top-4 unique-group distribution")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.legend(title="Unique groups")
    figure.savefig(output_dir / "qwen_unique_group_distribution_vs_alpha.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(alphas, [row["mean_unique_groups"] for row in objective], marker="o", color="#222222")
    axis.set(xlabel="Objective-aware alpha", ylabel="Mean unique groups per top-4 token", title="Qwen routing locality improves as mean U decreases")
    axis.invert_xaxis()
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "qwen_mean_unique_groups_vs_alpha.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for row, alpha in zip(objective, alphas):
        axis.scatter(row["mean_intragroup_output_distance"], row["mean_unique_groups"], s=70, color="#4E79A7")
        axis.annotate(f"{alpha:g}", (row["mean_intragroup_output_distance"], row["mean_unique_groups"]), xytext=(5, 5), textcoords="offset points")
    axis.set(
        xlabel="Mean intragroup output distance (smaller is better)",
        ylabel="Mean unique groups per top-4 token (smaller is better)",
        title="Qwen output compatibility vs. routing locality",
    )
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "qwen_output_cost_vs_routing_locality.png", dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    score_path = args.score_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not score_path.is_file():
        raise FileNotFoundError(f"Pairwise score artifact not found: {score_path}")
    if 1.0 not in args.alphas:
        raise ValueError("--alphas must include 1.0 for the HC-output comparison baseline")
    payload = load_payload(score_path)
    metadata = payload.get("metadata", {})
    if metadata.get("top_k") != 4:
        raise ValueError(f"Qwen objective-aware sweep requires top_k=4, found {metadata.get('top_k')!r}")
    layer_names = sorted(payload["layers"], key=lambda name: int(name.split(".")[2]))
    if len(layer_names) != 24:
        raise AssertionError(f"Expected Qwen's 24 MoE layers, found {len(layer_names)}")
    for layer_name in layer_names:
        scores = payload["layers"][layer_name]
        if scores["output_distance"].shape != (60, 60) or "topk_experts" not in scores:
            raise ValueError(f"{layer_name} must contain 60-expert output scores and topk_experts")
        if scores["topk_experts"].shape != (scores["num_tokens"], 4):
            raise ValueError(f"{layer_name} topk_experts must have shape [num_tokens, 4]")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    merge_traces: list[dict[str, Any]] = []
    alpha_summaries: list[dict[str, Any]] = []
    records_by_method_alpha: dict[tuple[str, float], list[dict[str, Any]]] = {}

    for alpha in args.alphas:
        objective_state, legacy_state = {}, {}
        alpha_records = {METHOD_OBJECTIVE: [], METHOD_LEGACY: []}
        for layer_name in layer_names:
            scores = payload["layers"][layer_name]
            layer = int(layer_name.split(".")[2])
            legacy_labels = labels_for_static_hc(scores, alpha, args.num_groups)
            objective = objective_aware_grouping(
                scores["output_distance"], scores["topk_experts"], args.num_groups, alpha,
            )
            objective_labels = objective["labels"]
            if alpha == 1.0 and not partitions_equal(objective_labels, legacy_labels):
                raise AssertionError(f"alpha=1 objective-aware partition differs from HC-output at {layer_name}")
            if alpha == 0.0:
                for merge in objective["merge_trace"]:
                    if abs(merge["delta_route"] - merge["max_delta_route"]) > 1e-6:
                        raise AssertionError(f"alpha=0 selected a non-maximal routing-gain merge at {layer_name}")
            objective_record = result_record(METHOD_OBJECTIVE, alpha, layer, objective_labels, scores)
            legacy_record = result_record(METHOD_LEGACY, alpha, layer, legacy_labels, scores)
            all_records.extend((objective_record, legacy_record))
            alpha_records[METHOD_OBJECTIVE].append(objective_record)
            alpha_records[METHOD_LEGACY].append(legacy_record)
            objective_state[layer_name] = objective_labels
            legacy_state[layer_name] = legacy_labels
            merge_traces.append({"method": METHOD_OBJECTIVE, "alpha": alpha, "layer": layer, "merges": objective["merge_trace"]})
            print(
                f"alpha={alpha:.2f} layer={layer:02d} "
                f"objective E[U]={objective_record['mean_unique_groups']:.6f} "
                f"legacy E[U]={legacy_record['mean_unique_groups']:.6f}"
            )

        for method, records in alpha_records.items():
            records_by_method_alpha[(method, alpha)] = records
        alpha_name = f"alpha_{alpha}"
        alpha_dir = output_dir / alpha_name
        alpha_dir.mkdir(exist_ok=True)
        torch.save(objective_state, alpha_dir / "group_state_dict.pt")
        legacy_dir = output_dir / METHOD_LEGACY / alpha_name
        legacy_dir.mkdir(parents=True, exist_ok=True)
        torch.save(legacy_state, legacy_dir / "group_state_dict.pt")

    for alpha in args.alphas:
        for method in (METHOD_OBJECTIVE, METHOD_LEGACY):
            baseline = summarize(records_by_method_alpha[(method, 1.0)], reference=None)
            summary = summarize(records_by_method_alpha[(method, alpha)], reference=baseline)
            summary.update({"method": method, "alpha": alpha})
            alpha_summaries.append(summary)

    with (output_dir / "per_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in all_records:
            row = {key: record[key] for key in CSV_FIELDS}
            row["groups"] = json.dumps(row["groups"], separators=(",", ":"))
            row["group_size_pattern"] = json.dumps(row["group_size_pattern"], separators=(",", ":"))
            writer.writerow(row)
    summary_fields = tuple(alpha_summaries[0].keys())
    with (output_dir / "per_alpha_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(alpha_summaries)
    with (output_dir / "merge_trace.json").open("w", encoding="utf-8") as handle:
        json.dump(merge_traces, handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metadata": metadata,
                "score_path": str(score_path),
                "num_groups": args.num_groups,
                "methods": [METHOD_OBJECTIVE, METHOD_LEGACY],
                "per_alpha_summary": alpha_summaries,
            },
            handle,
            indent=2,
        )
    write_plots(output_dir, alpha_summaries)
    print(f"Saved objective-aware Qwen sweep: {output_dir}")


if __name__ == "__main__":
    main()
