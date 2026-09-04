#!/usr/bin/env python3
"""Compute an exact Mixtral top-2 routing-locality ceiling from a C4 cache."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from statistics import mean
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hcsmoe.merging.clustering import hierarchical_clustering_from_pairwise_distance
from hcsmoe.merging.pairwise_scores import (
    build_hybrid_score_matrices,
    canonical_groups,
    enumerate_canonical_partitions,
    exact_routing_partition,
    grouping_metrics,
    partitions_equal,
    same_group_routing_count,
    validate_pairwise_scores,
)


CSV_FIELDS = (
    "layer",
    "hc_output_J_route",
    "hc_routing_J_route",
    "oracle_J_route",
    "algorithm_gap",
    "oracle_gain_vs_hc_output",
    "hc_output_expected_unique_groups",
    "hc_routing_expected_unique_groups",
    "oracle_expected_unique_groups",
    "hc_output_mean_intragroup_output_distance",
    "hc_routing_mean_intragroup_output_distance",
    "oracle_mean_intragroup_output_distance",
    "oracle_same_group_count",
    "oracle_num_ties",
    "hc_output_groups",
    "hc_routing_groups",
    "oracle_groups",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score_path", type=Path, required=True)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Invalid pairwise score artifact")
    if not isinstance(payload.get("layers"), dict):
        raise ValueError("Pairwise score artifact has no layer mapping")
    validate_pairwise_scores(payload["layers"])
    return payload


def _assert_routing_contract(scores: dict[str, Any], tolerance: float) -> None:
    routing_count = scores["routing_count"].detach().to(device="cpu", dtype=torch.int64)
    routing_rate = scores["routing_rate"].detach().to(device="cpu", dtype=torch.float32)
    num_tokens = int(scores["num_tokens"])
    if int(torch.triu(routing_count, diagonal=1).sum()) != num_tokens:
        raise AssertionError("Mixtral top-2 routing count does not sum to num_tokens")
    expected_rate = routing_count.float() / num_tokens
    if not torch.allclose(routing_rate, expected_rate, atol=tolerance, rtol=tolerance):
        raise AssertionError("routing_rate does not match routing_count / num_tokens")


def _group_result(
    labels: torch.Tensor,
    scores: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    routing_count = scores["routing_count"]
    num_tokens = int(scores["num_tokens"])
    same_count = same_group_routing_count(labels, routing_count)
    j_route = same_count / num_tokens
    metrics = grouping_metrics(labels, scores["output_distance"], scores["routing_rate"])
    if abs(float(metrics["same_group_routing_rate"]) - j_route) > tolerance:
        raise AssertionError("grouping_metrics same_group_routing_rate disagrees with routing counts")
    expected_unique_groups = 2.0 - j_route
    if abs(float(metrics["expected_unique_groups_per_token"]) - expected_unique_groups) > tolerance:
        raise AssertionError("expected unique-group metric disagrees with top-2 formula")
    return {
        "labels": labels.detach().to(device="cpu", dtype=torch.long),
        "groups": canonical_groups(labels),
        "J_route": j_route,
        "expected_unique_groups": expected_unique_groups,
        "mean_intragroup_output_distance": metrics["mean_intragroup_output_distance"],
        "same_group_count": same_count,
    }


def _hc_result(scores: dict[str, Any], alpha: float, num_groups: int, tolerance: float) -> dict[str, Any]:
    hybrid = build_hybrid_score_matrices(scores["output_distance"], scores["routing_rate"], alpha=alpha)
    labels, _ = hierarchical_clustering_from_pairwise_distance(
        hybrid["hybrid_distance"],
        num_groups,
        method="average",
        features_for_centers=scores["output_fingerprint"],
    )
    return _group_result(labels, scores, tolerance)


def _json_groups(groups: list[list[int]]) -> str:
    return json.dumps(groups, separators=(",", ":"))


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return mean(float(record[key]) for record in records)


def main() -> None:
    args = parse_args()
    score_path = args.score_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not score_path.is_file():
        raise FileNotFoundError(f"Pairwise score cache not found: {score_path}")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")

    payload = _load_payload(score_path)
    layer_names = sorted(payload["layers"], key=lambda name: int(name.split(".")[2]))
    if len(layer_names) != 32:
        raise AssertionError(f"Expected 32 Mixtral MoE layers, found {len(layer_names)}")
    partitions = enumerate_canonical_partitions(num_experts=8, num_groups=args.num_groups)
    if args.num_groups == 4:
        assert len(partitions) == 1701, f"Expected S(8,4)=1701, got {len(partitions)}"

    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir = output_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    oracle_state: dict[str, torch.Tensor] = {}

    for layer_name in layer_names:
        scores = payload["layers"][layer_name]
        _assert_routing_contract(scores, args.tolerance)
        hc_output = _hc_result(scores, alpha=1.0, num_groups=args.num_groups, tolerance=args.tolerance)
        hc_routing = _hc_result(scores, alpha=0.0, num_groups=args.num_groups, tolerance=args.tolerance)
        oracle_choice = exact_routing_partition(
            scores["routing_count"], args.num_groups, partitions=partitions
        )
        oracle = _group_result(oracle_choice["labels"], scores, args.tolerance)
        if oracle["same_group_count"] != oracle_choice["same_group_count"]:
            raise AssertionError("Oracle routing count disagrees with exact partition helper")
        if oracle["J_route"] + args.tolerance < hc_routing["J_route"]:
            raise AssertionError(f"Oracle is worse than HC-routing for {layer_name}")
        if oracle["J_route"] + args.tolerance < hc_output["J_route"]:
            raise AssertionError(f"Oracle is worse than HC-output for {layer_name}")

        layer = int(layer_name.split(".")[2])
        record = {
            "layer": layer,
            "hc_output_J_route": hc_output["J_route"],
            "hc_routing_J_route": hc_routing["J_route"],
            "oracle_J_route": oracle["J_route"],
            "algorithm_gap": oracle["J_route"] - hc_routing["J_route"],
            "oracle_gain_vs_hc_output": oracle["J_route"] - hc_output["J_route"],
            "hc_output_expected_unique_groups": hc_output["expected_unique_groups"],
            "hc_routing_expected_unique_groups": hc_routing["expected_unique_groups"],
            "oracle_expected_unique_groups": oracle["expected_unique_groups"],
            "hc_output_mean_intragroup_output_distance": hc_output["mean_intragroup_output_distance"],
            "hc_routing_mean_intragroup_output_distance": hc_routing["mean_intragroup_output_distance"],
            "oracle_mean_intragroup_output_distance": oracle["mean_intragroup_output_distance"],
            "oracle_same_group_count": oracle["same_group_count"],
            "oracle_num_ties": int(oracle_choice["num_ties"]),
            "hc_output_groups": hc_output["groups"],
            "hc_routing_groups": hc_routing["groups"],
            "oracle_groups": oracle["groups"],
            "hc_routing_matches_oracle_partition": partitions_equal(hc_routing["labels"], oracle["labels"]),
        }
        records.append(record)
        oracle_state[layer_name] = oracle["labels"]
        print(f"Layer {layer:02d}")
        print(f"  HC-output : J={hc_output['J_route']:.6f}, E[U]={hc_output['expected_unique_groups']:.6f}, groups={hc_output['groups']}")
        print(f"  HC-routing: J={hc_routing['J_route']:.6f}, E[U]={hc_routing['expected_unique_groups']:.6f}, groups={hc_routing['groups']}")
        print(f"  Oracle    : J={oracle['J_route']:.6f}, E[U]={oracle['expected_unique_groups']:.6f}, groups={oracle['groups']}")
        print(f"  Gap       : oracle - HC-routing = {record['algorithm_gap']:.6f}")

    summary = {
        "mean_hc_output_J_route": _mean(records, "hc_output_J_route"),
        "mean_hc_routing_J_route": _mean(records, "hc_routing_J_route"),
        "mean_oracle_J_route": _mean(records, "oracle_J_route"),
        "mean_algorithm_gap": _mean(records, "algorithm_gap"),
        "mean_oracle_gain_vs_hc_output": _mean(records, "oracle_gain_vs_hc_output"),
        "mean_hc_output_expected_unique_groups": _mean(records, "hc_output_expected_unique_groups"),
        "mean_hc_routing_expected_unique_groups": _mean(records, "hc_routing_expected_unique_groups"),
        "mean_oracle_expected_unique_groups": _mean(records, "oracle_expected_unique_groups"),
        "oracle_matches_hc_routing_partition_layers": sum(
            record["hc_routing_matches_oracle_partition"] for record in records
        ),
        "hc_routing_better_than_hc_output_layers": sum(
            record["hc_routing_J_route"] > record["hc_output_J_route"] + args.tolerance
            for record in records
        ),
        "hc_routing_worse_than_hc_output_layers": sum(
            record["hc_routing_J_route"] + args.tolerance < record["hc_output_J_route"]
            for record in records
        ),
    }
    report = {
        "metadata": payload["metadata"],
        "score_path": str(score_path),
        "num_groups": args.num_groups,
        "num_partitions": len(partitions),
        "layers": records,
        "summary": summary,
    }

    with (output_dir / "per_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in CSV_FIELDS}
            for key in ("hc_output_groups", "hc_routing_groups", "oracle_groups"):
                row[key] = _json_groups(row[key])
            writer.writerow(row)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    torch.save(oracle_state, oracle_dir / "group_state_dict.pt")

    print(f"[Routing ceiling] mean summary={summary}")
    print(f"[Routing ceiling] Saved: {output_dir}")


if __name__ == "__main__":
    main()
