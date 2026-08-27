#!/usr/bin/env python3
"""Create tab-separated Qwen group rows intended for copy/paste into Excel."""

from __future__ import annotations

import argparse
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

    layers = {}
    for key, labels in state.items():
        if isinstance(key, str) and key.startswith("model.layers.") and key.endswith(".mlp"):
            layer = int(key.removeprefix("model.layers.").removesuffix(".mlp"))
            layers[layer] = labels.detach().to(device="cpu", dtype=torch.long).flatten().tolist()
    num_groups = max(max(labels) for labels in layers.values()) + 1

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        for layer, labels in sorted(layers.items()):
            groups = [[] for _ in range(num_groups)]
            for expert, group in enumerate(labels):
                groups[group].append(expert)
            cells = [f"L{layer:02d}", *[f"G{group}:{members}" for group, members in enumerate(groups)]]
            handle.write("\t".join(cells) + "\n")
    print(f"Wrote copy/paste text: {output}")


if __name__ == "__main__":
    main()
