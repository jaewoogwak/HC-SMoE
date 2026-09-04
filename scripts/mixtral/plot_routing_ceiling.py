#!/usr/bin/env python3
"""Plot HC-SMoE routing locality against the exact routing ceiling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "layer",
    "hc_output_J_route",
    "hc_routing_J_route",
    "oracle_J_route",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/routing_ceiling/mixtral_c4"),
        help="Directory containing per_layer.csv from analyze_routing_ceiling.py.",
    )
    parser.add_argument(
        "--output_stem",
        default="routing_locality_ceiling",
        help="Filename stem for the PNG and PDF written under --results_dir.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required columns: {sorted(REQUIRED_COLUMNS)}")
        rows = [
            {
                "layer": int(row["layer"]),
                "hc_output": float(row["hc_output_J_route"]),
                "hc_routing": float(row["hc_routing_J_route"]),
                "oracle": float(row["oracle_J_route"]),
            }
            for row in reader
        ]
    if len(rows) != 32 or [row["layer"] for row in sorted(rows, key=lambda row: row["layer"])] != list(range(32)):
        raise ValueError("Expected one row for each Mixtral layer 0 through 31")
    return sorted(rows, key=lambda row: row["layer"])


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    rows = load_rows(results_dir / "per_layer.csv")

    layers = [row["layer"] for row in rows]
    hc_output = [row["hc_output"] for row in rows]
    hc_routing = [row["hc_routing"] for row in rows]
    oracle = [row["oracle"] for row in rows]
    best_hc = [max(output, routing) for output, routing in zip(hc_output, hc_routing)]
    remaining_gap = [ceiling - baseline for ceiling, baseline in zip(oracle, best_hc)]

    figure, axis = plt.subplots(figsize=(12, 6.4), constrained_layout=True)
    axis.fill_between(
        layers,
        best_hc,
        oracle,
        color="#59A14F",
        alpha=0.18,
        label="Remaining gap above best HC baseline",
        zorder=1,
    )
    axis.plot(
        layers,
        hc_output,
        color="#4E79A7",
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        label="HC-SMoE output",
        zorder=3,
    )
    axis.plot(
        layers,
        hc_routing,
        color="#F28E2B",
        linewidth=2.0,
        marker="s",
        markersize=4.2,
        label="HC-SMoE routing",
        zorder=4,
    )
    axis.plot(
        layers,
        oracle,
        color="#222222",
        linewidth=2.6,
        marker="D",
        markersize=4.0,
        label="Exact routing ceiling",
        zorder=5,
    )

    axis.set(
        title="Mixtral-8x7B C4 routing locality: HC-SMoE vs. exact 4-group ceiling",
        xlabel="MoE layer",
        ylabel=r"Same-group routing locality, $J_{route}$",
        xlim=(-0.5, 31.5),
    )
    axis.set_xticks(range(0, 32, 4))
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.legend(loc="upper left", frameon=True, framealpha=0.94)
    annotation = (
        f"Mean J: output {mean(hc_output):.3f} | routing {mean(hc_routing):.3f} | ceiling {mean(oracle):.3f}\n"
        f"Mean remaining gap: ceiling - best HC = {mean(remaining_gap):.3f}"
    )
    axis.text(
        0.02,
        0.02,
        annotation,
        transform=axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.94, "boxstyle": "round,pad=0.45"},
    )

    png_path = results_dir / f"{args.output_stem}.png"
    pdf_path = results_dir / f"{args.output_stem}.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
