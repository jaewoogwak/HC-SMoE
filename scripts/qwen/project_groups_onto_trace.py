#!/usr/bin/env python3
"""Annotate a Qwen MoE decode trace with C4-calibrated HC-SMoE groups.

The input is a base-Qwen top-4 routing trace.  It is preserved verbatim and
augmented with the layer-local HC-SMoE group for each selected expert.  This
does not run the merged checkpoint: it measures how often the base model's
greedy decode routing can be served from one HC-SMoE group.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


NUM_QWEN_EXPERTS = 60
QWEN_TOP_K = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Base-Qwen top-4 decode trace CSV.")
    parser.add_argument(
        "--group-state",
        type=Path,
        required=True,
        help="Qwen HC-SMoE group_state_dict.pt produced by C4-calibrated grouping.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Annotated CSV to create.")
    return parser.parse_args()


def load_group_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with the HC-SMoE pinned PyTorch build.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}, found {type(value).__name__}.")

    groups: dict[str, torch.Tensor] = {}
    for key, labels in value.items():
        if not isinstance(key, str) or not isinstance(labels, torch.Tensor):
            raise TypeError("group_state_dict.pt must map layer names to tensor labels.")
        labels = labels.detach().to(device="cpu", dtype=torch.long).flatten()
        if labels.numel() != NUM_QWEN_EXPERTS:
            raise ValueError(f"{key} has {labels.numel()} labels; expected Qwen's {NUM_QWEN_EXPERTS} experts.")
        groups[key] = labels
    if not groups:
        raise ValueError("group_state_dict.pt contains no layer mappings.")
    return groups


def layer_labels(groups: dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    key = f"model.layers.{layer}.mlp"
    try:
        return groups[key]
    except KeyError as error:
        raise KeyError(f"No Qwen group mapping for layer {layer}; expected key {key!r}.") from error


def format_group(layer: int, group_id: int) -> str:
    # Cluster label IDs only have meaning inside a layer.
    return f"L{layer}:G{group_id}"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def unique_group_summary(counts: dict[str, int], rows: int) -> dict[str, Any]:
    histogram = {str(groups): int(counts.get(str(groups), 0)) for groups in range(1, QWEN_TOP_K + 1)}
    rates = {str(groups): histogram[str(groups)] / rows if rows else 0.0 for groups in range(1, QWEN_TOP_K + 1)}
    return {
        "unique_group_count": histogram,
        "unique_group_rate": rates,
        "mean_unique_groups": sum(groups * rates[str(groups)] for groups in range(1, QWEN_TOP_K + 1)),
    }


def main() -> None:
    args = parse_args()
    trace_path = args.trace.expanduser().resolve()
    group_state_path = args.group_state.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"Trace CSV not found: {trace_path}")
    if not group_state_path.is_file():
        raise FileNotFoundError(f"Group mapping not found: {group_state_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    groups = load_group_state(group_state_path)

    top_expert_fields = [f"top{rank}_expert" for rank in range(1, QWEN_TOP_K + 1)]
    top_group_id_fields = [f"top{rank}_group_id" for rank in range(1, QWEN_TOP_K + 1)]
    top_group_fields = [f"top{rank}_group" for rank in range(1, QWEN_TOP_K + 1)]
    output_fields = [
        *top_group_id_fields,
        *top_group_fields,
        "unique_groups",
        "num_unique_groups",
        "group_local_routing",
    ]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_rows = 0
    group_local_rows = 0
    per_layer: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "group_local_rows": 0,
            "unique_group_count": {str(groups): 0 for groups in range(1, QWEN_TOP_K + 1)},
        }
    )
    global_unique_group_count = {str(groups): 0 for groups in range(1, QWEN_TOP_K + 1)}

    with trace_path.open(newline="", encoding="utf-8") as source, temporary_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        required_fields = {"layer", *top_expert_fields}
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(f"Trace must contain {sorted(required_fields)}; found {reader.fieldnames}.")
        collisions = set(reader.fieldnames).intersection(output_fields)
        if collisions:
            raise ValueError(f"Trace already has group annotation fields: {sorted(collisions)}")
        writer = csv.DictWriter(destination, fieldnames=[*reader.fieldnames, *output_fields])
        writer.writeheader()

        for row_number, row in enumerate(reader, start=2):
            try:
                layer = int(row["layer"])
                experts = [int(row[field]) for field in top_expert_fields]
            except ValueError as error:
                raise ValueError(f"Invalid layer/expert ID at CSV row {row_number}.") from error
            if any(expert < 0 or expert >= NUM_QWEN_EXPERTS for expert in experts):
                raise ValueError(
                    f"Expert IDs must be in [0, {NUM_QWEN_EXPERTS - 1}] at CSV row {row_number}."
                )

            labels = layer_labels(groups, layer)
            group_ids = [int(labels[expert]) for expert in experts]
            selected_groups = [format_group(layer, group_id) for group_id in group_ids]
            unique_groups = list(dict.fromkeys(selected_groups))
            is_group_local = len(set(group_ids)) == 1
            row.update(
                {
                    **dict(zip(top_group_id_fields, group_ids)),
                    **dict(zip(top_group_fields, selected_groups)),
                    "unique_groups": "|".join(unique_groups),
                    "num_unique_groups": len(unique_groups),
                    "group_local_routing": str(is_group_local).lower(),
                }
            )
            writer.writerow(row)
            total_rows += 1
            per_layer[layer]["rows"] += 1
            unique_count = len(unique_groups)
            per_layer[layer]["unique_group_count"][str(unique_count)] += 1
            global_unique_group_count[str(unique_count)] += 1
            if is_group_local:
                group_local_rows += 1
                per_layer[layer]["group_local_rows"] += 1

        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary_path, output_path)

    per_layer_summary = {
        str(layer): {
            **counts,
            "group_local_routing_rate": counts["group_local_rows"] / counts["rows"],
            **unique_group_summary(counts["unique_group_count"], counts["rows"]),
        }
        for layer, counts in sorted(per_layer.items())
    }
    summary = {
        "source_trace": str(trace_path),
        "group_state": str(group_state_path),
        "annotated_trace": str(output_path),
        "num_layers_in_mapping": len(groups),
        "experts_per_layer": NUM_QWEN_EXPERTS,
        "experts_per_token": QWEN_TOP_K,
        "rows": total_rows,
        "group_local_routing_rows": group_local_rows,
        "group_local_routing_rate": group_local_rows / total_rows if total_rows else 0.0,
        **unique_group_summary(global_unique_group_count, total_rows),
        "per_layer": per_layer_summary,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    atomic_json(summary_path, summary)
    print(f"Wrote annotated trace: {output_path}")
    print(f"Wrote group-local routing summary: {summary_path}")


if __name__ == "__main__":
    main()
