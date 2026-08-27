#!/usr/bin/env python3
"""Annotate a base-Mixtral decode trace with HC-SMoE group-local routing.

The input trace is not modified.  Every row retains its original top-2 expert
IDs and receives the HC-SMoE group for each expert plus whether both selected
experts are local to the same group.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="Existing base-Mixtral decode trace CSV.")
    parser.add_argument(
        "--group-state",
        type=Path,
        required=True,
        help="HC-SMoE group_state_dict.pt produced during C4-calibrated merging.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Annotated CSV to create.")
    return parser.parse_args()


def load_group_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch compatibility for the HC-SMoE pinned environment.
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}, found {type(value).__name__}.")
    groups: dict[str, torch.Tensor] = {}
    for key, labels in value.items():
        if not isinstance(key, str) or not isinstance(labels, torch.Tensor):
            raise TypeError("group_state_dict.pt must map layer names to tensor labels.")
        labels = labels.detach().to(device="cpu", dtype=torch.long).flatten()
        if labels.numel() != 8:
            raise ValueError(f"{key} has {labels.numel()} labels; expected Mixtral's 8 experts.")
        groups[key] = labels
    return groups


def layer_labels(groups: dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    key = f"model.layers.{layer}.block_sparse_moe"
    try:
        return groups[key]
    except KeyError as error:
        raise KeyError(f"No group mapping for layer {layer}; expected key {key!r}.") from error


def format_group(layer: int, group_id: int) -> str:
    # Group labels are layer-local. Including the layer prevents accidental
    # comparison of arbitrary clustering-label IDs across different layers.
    return f"L{layer}:G{group_id}"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_fields = (
        "top1_group_id",
        "top2_group_id",
        "top1_group",
        "top2_group",
        "unique_groups",
        "num_unique_groups",
        "group_local_routing",
    )
    total_rows = 0
    group_local_rows = 0
    per_layer: dict[int, dict[str, int]] = defaultdict(lambda: {"rows": 0, "group_local_rows": 0})

    with trace_path.open(newline="", encoding="utf-8") as source, temporary_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source)
        required_fields = {"layer", "top1_expert", "top2_expert"}
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
                top1_expert = int(row["top1_expert"])
                top2_expert = int(row["top2_expert"])
            except ValueError as error:
                raise ValueError(f"Invalid layer/expert ID at CSV row {row_number}.") from error
            if not 0 <= top1_expert < 8 or not 0 <= top2_expert < 8:
                raise ValueError(f"Expert IDs must be in [0, 7] at CSV row {row_number}.")

            labels = layer_labels(groups, layer)
            top1_group_id = int(labels[top1_expert])
            top2_group_id = int(labels[top2_expert])
            top1_group = format_group(layer, top1_group_id)
            top2_group = format_group(layer, top2_group_id)
            unique_groups = [top1_group]
            if top2_group != top1_group:
                unique_groups.append(top2_group)
            is_group_local = top1_group_id == top2_group_id

            row.update(
                {
                    "top1_group_id": top1_group_id,
                    "top2_group_id": top2_group_id,
                    "top1_group": top1_group,
                    "top2_group": top2_group,
                    "unique_groups": "|".join(unique_groups),
                    "num_unique_groups": len(unique_groups),
                    "group_local_routing": str(is_group_local).lower(),
                }
            )
            writer.writerow(row)
            total_rows += 1
            per_layer[layer]["rows"] += 1
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
        }
        for layer, counts in sorted(per_layer.items())
    }
    summary = {
        "source_trace": str(trace_path),
        "group_state": str(group_state_path),
        "annotated_trace": str(output_path),
        "rows": total_rows,
        "group_local_routing_rows": group_local_rows,
        "group_local_routing_rate": group_local_rows / total_rows if total_rows else 0.0,
        "per_layer": per_layer_summary,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    atomic_json(summary_path, summary)
    print(f"Wrote annotated trace: {output_path}")
    print(f"Wrote group-local routing summary: {summary_path}")


if __name__ == "__main__":
    main()
