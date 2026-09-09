#!/usr/bin/env python3
"""Plot objective-aware per-layer P(U=1) against the HC-SMoE baseline."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir", type=Path, required=True,
        help="Directory containing per_layer.csv from an objective-aware sweep.",
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--baseline_alpha", type=float, default=1.0)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--annotation_threshold_pp", type=float, default=7.0)
    parser.add_argument("--max_annotations", type=int, default=8)
    return parser.parse_args()


def same_alpha(left: float, right: float) -> bool:
    return math.isclose(left, right, abs_tol=1e-9, rel_tol=0.0)


def num_tokens(row: dict[str, str]) -> int:
    columns = [key for key in row if key.startswith("unique_group_count_")]
    if not columns:
        raise ValueError("per_layer.csv has no unique_group_count columns")
    return sum(int(row[column]) for column in columns)


def load_comparison(
    csv_path: Path, alpha: float, baseline_alpha: float,
) -> tuple[list[int], list[float], list[float], float, float]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"method", "alpha", "layer", "unique_group_rate_1", "unique_group_count_1"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{csv_path} is missing required columns: {sorted(required)}")
    baseline_by_layer = {
        int(row["layer"]): row for row in rows
        if row["method"] == "legacy_pairwise_hc" and same_alpha(float(row["alpha"]), baseline_alpha)
    }
    objective_by_layer = {
        int(row["layer"]): row for row in rows
        if row["method"] == "objective_aware" and same_alpha(float(row["alpha"]), alpha)
    }
    layers = sorted(baseline_by_layer)
    if not layers or layers != sorted(objective_by_layer) or layers != list(range(len(layers))):
        raise ValueError("Baseline and objective rows must cover identical contiguous layer IDs")
    baseline = [float(baseline_by_layer[layer]["unique_group_rate_1"]) * 100 for layer in layers]
    objective = [float(objective_by_layer[layer]["unique_group_rate_1"]) * 100 for layer in layers]
    base_total = sum(num_tokens(baseline_by_layer[layer]) for layer in layers)
    objective_total = sum(num_tokens(objective_by_layer[layer]) for layer in layers)
    base_overall = sum(int(baseline_by_layer[layer]["unique_group_count_1"]) for layer in layers) / base_total * 100
    objective_overall = sum(int(objective_by_layer[layer]["unique_group_count_1"]) for layer in layers) / objective_total * 100
    return layers, baseline, objective, base_overall, objective_overall


def infer_model_name(num_layers: int) -> str:
    return "Mixtral" if num_layers == 32 else "Qwen2-MoE" if num_layers == 24 else f"MoE ({num_layers} layers)"


def plot(
    layers: list[int],
    baseline: list[float],
    objective: list[float],
    base_overall: float,
    objective_overall: float,
    alpha: float,
    model_name: str,
    threshold: float,
    max_annotations: int,
    output_path: Path,
) -> None:
    delta = [candidate - reference for reference, candidate in zip(baseline, objective)]
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.12},
    )
    figure.patch.set_facecolor("white")
    for axis in (top, bottom):
        axis.set_facecolor("#FAFAFA")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.75, alpha=0.7)
        for spine in axis.spines.values():
            spine.set_color("#888888")
            spine.set_linewidth(1.0)

    top.plot(layers, baseline, color="#4C78A8", marker="o", markersize=3.5, linewidth=2.0,
             label="HC-SMoE legacy (alpha=1.0)")
    top.plot(layers, objective, color="#F15A5A", marker="o", markersize=3.5, linewidth=2.0,
             label=f"HC-SMoE + objective-aware (alpha={alpha:g})")
    top.set_ylabel("Rate of U=1 (%)")
    top.set_ylim(0, max(70, math.ceil((max(baseline + objective) + 5) / 10) * 10))
    top.legend(loc="upper left", frameon=False, ncol=2)
    top.text(
        0.985, 0.045,
        f"Overall: {base_overall:.2f}% -> {objective_overall:.2f}% ({objective_overall - base_overall:+.2f} pp)",
        transform=top.transAxes, ha="right", va="bottom", fontsize=10, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F2F2F2", "edgecolor": "#DDDDDD"},
    )

    bars = bottom.bar(layers, delta, color=["#D95F5F" if value > 0 else "#4C9F70" for value in delta], width=0.72)
    bottom.axhline(0, color="#666666", linewidth=1.1, linestyle="--")
    bottom.set_ylabel("Delta (pp)")
    bottom.set_xlabel(f"{model_name} layer")
    bottom.set_xticks(layers)
    bottom.set_xlim(min(layers) - 1, max(layers) + 1)
    bound = max(10, math.ceil((max(abs(value) for value in delta) + 4) / 5) * 5)
    bottom.set_ylim(-bound, bound)
    bottom.text(0.015, 0.92, f"Layer mean: {sum(delta) / len(delta):+.2f} pp",
                transform=bottom.transAxes, ha="left", va="top", color="#555555")
    candidates = sorted(
        (index for index, value in enumerate(delta) if abs(value) >= threshold),
        key=lambda index: abs(delta[index]), reverse=True,
    )
    selected = []
    for index in candidates:
        if all(abs(index - previous) >= 2 for previous in selected):
            selected.append(index)
        if len(selected) == max_annotations:
            break
    for index, (layer, value, _bar) in enumerate(zip(layers, delta, bars)):
        if index in selected:
            bottom.text(
                layer, value + (1.0 if value >= 0 else -1.2), f"L{layer}\n{value:+.2f} pp",
                ha="center", va="bottom" if value >= 0 else "top", fontsize=8, fontweight="bold",
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_annotations < 0:
        raise ValueError("--max_annotations must be non-negative")
    results_dir = args.results_dir.expanduser().resolve()
    layers, baseline, objective, base_overall, objective_overall = load_comparison(
        results_dir / "per_layer.csv", args.alpha, args.baseline_alpha,
    )
    model_name = args.model_name or infer_model_name(len(layers))
    default_name = f"{model_name.lower().replace('-', '_').replace(' ', '_')}_objective_aware_alpha_{args.alpha:g}_vs_legacy.png"
    output_path = (args.output_path or results_dir / default_name).expanduser().resolve()
    plot(layers, baseline, objective, base_overall, objective_overall, args.alpha, model_name,
         args.annotation_threshold_pp, args.max_annotations, output_path)
    print(f"Saved: {output_path}")
    print(f"Overall P(U=1): {base_overall:.4f}% -> {objective_overall:.4f}% ({objective_overall - base_overall:+.4f} pp)")


if __name__ == "__main__":
    main()
