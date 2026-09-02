#!/usr/bin/env python3
"""Offline Mixtral output+routing hybrid grouping sweep from a score cache."""

import argparse
import json
import os
from statistics import mean

import torch

from hcsmoe.merging.clustering import (
    group_experts_by_clustering,
    hierarchical_clustering_from_pairwise_distance,
)
from hcsmoe.merging.pairwise_scores import (
    build_hybrid_score_matrices,
    canonical_groups,
    changed_expert_count,
    grouping_metrics,
    partitions_equal,
    validate_pairwise_scores,
)


def _parse_alphas(value):
    alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not alphas:
        raise argparse.ArgumentTypeError("--alphas must contain at least one value")
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise argparse.ArgumentTypeError("all alpha values must be in [0, 1]")
    return alphas


def _result_for_layer(scores, alpha, num_groups):
    hybrid = build_hybrid_score_matrices(scores["output_distance"], scores["routing_rate"], alpha)
    labels, dominant_experts = hierarchical_clustering_from_pairwise_distance(
        hybrid["hybrid_distance"], num_groups, method="average",
        features_for_centers=scores["output_fingerprint"],
    )
    metrics = grouping_metrics(labels, scores["output_distance"], scores["routing_rate"])
    return {
        "labels": labels.tolist(),
        "groups": canonical_groups(labels),
        "dominant_experts": dominant_experts,
        "normalization": hybrid["normalization"],
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_path", required=True)
    parser.add_argument("--alphas", type=_parse_alphas, required=True)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--verify_alpha_one", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not os.path.exists(args.score_path):
        raise FileNotFoundError(
            f"Pairwise score cache not found: {args.score_path}\n"
            "Run scripts/mixtral/run.sh with --hybrid_grouping=True or use --score_only=True first."
        )
    payload = torch.load(args.score_path, map_location="cpu")
    if payload.get("version") != 1 or not isinstance(payload.get("layers"), dict):
        raise ValueError("Invalid pairwise score artifact")
    validate_pairwise_scores(payload["layers"])

    layer_names = sorted(payload["layers"], key=lambda name: int(name.split(".")[2]))
    alpha_one = {
        layer_name: _result_for_layer(payload["layers"][layer_name], 1.0, args.num_groups)
        for layer_name in layer_names
    }
    if args.verify_alpha_one:
        for layer_name in layer_names:
            scores = payload["layers"][layer_name]
            _, legacy_labels = group_experts_by_clustering(
                model="mixtral", num_groups=args.num_groups, cluster="hierarchical",
                linkage="average", hierarchical_stopping_metric="silhouette",
                num_experts=scores["output_fingerprint"].shape[0],
                experts=scores["output_fingerprint"],
            )
            if not partitions_equal(legacy_labels, alpha_one[layer_name]["labels"]):
                raise AssertionError(
                    f"alpha=1 partition mismatch for {layer_name}: "
                    f"legacy={canonical_groups(legacy_labels)}, "
                    f"precomputed={alpha_one[layer_name]['groups']}"
                )
        print("[Hybrid sweep] Alpha=1 score-artifact regression passed for all layers")

    results = {"metadata": payload["metadata"], "num_groups": args.num_groups, "alphas": {}}
    for alpha in args.alphas:
        layer_results = {}
        for layer_name in layer_names:
            result = _result_for_layer(payload["layers"][layer_name], alpha, args.num_groups)
            baseline = alpha_one[layer_name]
            result["same_partition_as_alpha_1"] = partitions_equal(result["labels"], baseline["labels"])
            result["changed_expert_count_vs_alpha_1"] = changed_expert_count(baseline["labels"], result["labels"])
            layer_results[layer_name] = result
            print(
                f"{layer_name} alpha={alpha:.3f} groups={result['groups']} "
                f"changed={result['changed_expert_count_vs_alpha_1']} "
                f"same_group_routing_rate={result['same_group_routing_rate']:.6f} "
                f"expected_unique_groups_per_token={result['expected_unique_groups_per_token']:.6f} "
                f"mean_intragroup_output_distance={result['mean_intragroup_output_distance']}"
            )
        valid_distances = [
            result["mean_intragroup_output_distance"] for result in layer_results.values()
            if result["mean_intragroup_output_distance"] is not None
        ]
        summary = {
            "mean_same_group_routing_rate": mean(result["same_group_routing_rate"] for result in layer_results.values()),
            "mean_expected_unique_groups_per_token": mean(
                result["expected_unique_groups_per_token"] for result in layer_results.values()
            ),
            "mean_intragroup_output_distance": mean(valid_distances) if valid_distances else None,
        }
        print(f"[Hybrid sweep] alpha={alpha:.3f} summary={summary}")
        results["alphas"][str(alpha)] = {"layers": layer_results, "summary": summary}

    directory = os.path.dirname(args.output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"[Hybrid sweep] Saved: {args.output_path}")


if __name__ == "__main__":
    main()
