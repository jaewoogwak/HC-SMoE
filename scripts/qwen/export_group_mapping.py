#!/usr/bin/env python3
"""Export a Qwen HC-SMoE group_state_dict as an Excel-friendly TSV grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        state = torch.load(args.group_state, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.group_state, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError("Expected group_state_dict.pt to contain a dictionary.")

    layers: dict[int, torch.Tensor] = {}
    for key, labels in state.items():
        if not isinstance(key, str) or not key.startswith("model.layers.") or not key.endswith(".mlp"):
            continue
        layer = int(key.removeprefix("model.layers.").removesuffix(".mlp"))
        layers[layer] = labels.detach().to(dtype=torch.long, device="cpu").flatten()
    if not layers:
        raise ValueError("No Qwen layer mappings found.")
    num_groups = max(int(labels.max()) for labels in layers.values()) + 1

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["layer", *[f"G{group}" for group in range(num_groups)]])
        for layer, labels in sorted(layers.items()):
            groups = [[] for _ in range(num_groups)]
            for expert, group in enumerate(labels.tolist()):
                groups[group].append(str(expert))
            writer.writerow([f"L{layer:02d}", *[",".join(experts) for experts in groups]])
    print(f"Wrote Excel-friendly mapping grid: {output}")


if __name__ == "__main__":
    main()
