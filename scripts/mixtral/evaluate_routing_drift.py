#!/usr/bin/env python3
"""Measure vanilla HC-SMoE routing drift on contiguous held-out PG19 books."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, MixtralForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcsmoe.merging.mixtral_checkpoint import load_compressed_model_for_evaluation
from hcsmoe.merging.device_placement import (
    PLACEMENT_CHOICES,
    PlacementSettings,
    place_model_for_analysis,
    pretrained_cpu_load_kwargs,
    print_model_placement,
    print_placement_environment,
    resolve_placement_settings,
)
from hcsmoe.merging.pg19_drift_mixtral import (
    DocumentRoutingTraces,
    aggregate_forced_metrics,
    aggregate_free_summaries,
    aggregate_prefill_metrics,
    collect_document_routing_traces,
    compare_decode_document,
    compare_prefill_document,
    load_pg19_documents,
    save_pg19_plots,
    stack_document_metrics,
    summarize_free_document,
)
from hcsmoe.merging.sequential_drift_mixtral import (
    assert_router_weights_unchanged,
    freeze_for_analysis,
    router_weight_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--model-path", required=True, help="Saved vanilla HC-SMoE model.pth")
    parser.add_argument("--group-state-path", required=True, help="Saved vanilla HC-SMoE group_state_dict.pt")
    parser.add_argument(
        "--dataset-name",
        default="emozilla/pg19-test",
        help="Test-only Parquet mirror of the PG19 held-out split",
    )
    parser.add_argument("--dataset-split", default="test", help="Held-out PG19 split")
    parser.add_argument("--num-documents", type=int, default=8)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=512)
    parser.add_argument("--placement", choices=PLACEMENT_CHOICES, default="auto")
    parser.add_argument(
        "--gpu-memory",
        default=None,
        help="Optional auto-placement CUDA budget, e.g. 70GiB; default is detected GPU memory minus a safety reserve",
    )
    parser.add_argument("--cpu-memory", default="1500GiB", help="CPU budget used by auto placement")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/routing_drift/mixtral_hcsmoe_8to4_pg19")
    return parser.parse_args()


def load_original_for_analysis(
    model_name: str,
    placement: PlacementSettings,
) -> tuple[MixtralForCausalLM, list[torch.Tensor]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Mixtral routing-drift analysis requires CUDA")
    model = MixtralForCausalLM.from_pretrained(
        model_name,
        **pretrained_cpu_load_kwargs(),
    )
    routers = router_weight_snapshot(model)
    model = place_model_for_analysis(model, placement)
    assert_router_weights_unchanged(routers, model)
    print_model_placement(model, "original")
    return freeze_for_analysis(model), routers


def load_merged_for_analysis(
    model_name: str,
    model_path: str,
    group_state_path: str,
    original_routers: list[torch.Tensor],
    placement: PlacementSettings,
) -> MixtralForCausalLM:
    model, _ = load_compressed_model_for_evaluation(
        model_name,
        model_path,
        group_state_path,
        False,
        None,
        load_to_cpu=True,
    )
    assert_router_weights_unchanged(original_routers, model)
    print("[PG19 routing drift] Vanilla HC-SMoE router weights are bitwise identical")
    model = place_model_for_analysis(model, placement)
    assert_router_weights_unchanged(original_routers, model)
    print_model_placement(model, "vanilla HC-SMoE")
    return freeze_for_analysis(model)


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    for name in ("num_documents", "prefill_tokens", "decode_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not Path(args.model_path).is_file() or not Path(args.group_state_path).is_file():
        raise FileNotFoundError("--model-path and --group-state-path must point to saved vanilla HC-SMoE artifacts")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.manual_seed(args.seed)
    placement = resolve_placement_settings(args.placement, args.gpu_memory, args.cpu_memory)
    print_placement_environment(placement)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    documents = load_pg19_documents(
        tokenizer,
        args.num_documents,
        args.prefill_tokens,
        args.decode_steps,
        args.seed,
        args.dataset_name,
        args.dataset_split,
    )
    document_ids = [document.document_id for document in documents]
    config = {
        "model_name": args.model_name,
        "model_path": str(Path(args.model_path).resolve()),
        "group_state_path": str(Path(args.group_state_path).resolve()),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "num_documents": args.num_documents,
        "prefill_tokens": args.prefill_tokens,
        "decode_steps": args.decode_steps,
        "seed": args.seed,
        "comparison": "Original Mixtral vs vanilla HC-SMoE 8-to-4",
        "router_comparison": "unchanged original expert IDs; no projection to merged groups",
        "prefill": "all positions in one contiguous segment per PG19 document",
        "forced_decode": "ground-truth PG19 continuation with separate model-owned KV caches",
        "free_decode": "independent greedy decoding; post-token-divergence routes are flagged",
    }
    _json_dump(output_dir / "config.json", config)
    _json_dump(
        output_dir / "selected_documents.json",
        {
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "selection": "seeded document permutation; first qualifying contiguous segment starts at token 0",
            "documents": [document.metadata(args.prefill_tokens) for document in documents],
        },
    )

    original_model, original_routers = load_original_for_analysis(args.model_name, placement)
    original_traces: list[Optional[DocumentRoutingTraces]] = []
    for index, document in enumerate(tqdm(documents, desc="[PG19 routing drift] original documents")):
        original_traces.append(
            collect_document_routing_traces(
                original_model,
                document,
                args.prefill_tokens,
                args.decode_steps,
                f"original document {index}",
            )
        )
    del original_model
    _clear_cuda()

    merged_model = load_merged_for_analysis(
        args.model_name,
        args.model_path,
        args.group_state_path,
        original_routers,
        placement,
    )
    prefill_document_rows: list[list[dict[str, Any]]] = []
    prefill_raw_documents: list[dict[str, torch.Tensor]] = []
    forced_documents: list[dict[str, torch.Tensor]] = []
    free_documents: list[dict[str, torch.Tensor]] = []
    free_document_summaries: list[dict[str, Any]] = []
    for index, document in enumerate(tqdm(documents, desc="[PG19 routing drift] merged documents")):
        original = original_traces[index]
        if original is None:
            raise AssertionError("original document trace was released too early")
        merged = collect_document_routing_traces(
            merged_model,
            document,
            args.prefill_tokens,
            args.decode_steps,
            f"merged document {index}",
        )
        rows, prefill_raw = compare_prefill_document(original.prefill, merged.prefill, document.document_id)
        forced = compare_decode_document(original.forced, merged.forced, require_identical_tokens=True)
        free = compare_decode_document(original.free, merged.free, require_identical_tokens=False)
        free_summary = summarize_free_document(free, prefill_raw["routing_shift"])
        free_summary["document_id"] = document.document_id
        free_summary["source_index"] = document.source_index
        prefill_document_rows.append(rows)
        prefill_raw_documents.append(prefill_raw)
        forced_documents.append(forced)
        free_documents.append(free)
        free_document_summaries.append(free_summary)
        original_traces[index] = None
        del original, merged
        gc.collect()
    del merged_model, original_traces
    _clear_cuda()

    prefill_raw = stack_document_metrics(document_ids, prefill_raw_documents)
    forced_raw = stack_document_metrics(document_ids, forced_documents)
    free_raw = stack_document_metrics(document_ids, free_documents)
    prefill_layer_metrics = aggregate_prefill_metrics(prefill_document_rows, prefill_raw)
    prefill_document_metrics = [
        {
            "document_id": document.document_id,
            "source_index": document.source_index,
            "layers": rows,
        }
        for document, rows in zip(documents, prefill_document_rows)
    ]
    forced_summary = aggregate_forced_metrics(forced_raw)
    forced_summary["document_ids"] = document_ids
    free_summary = aggregate_free_summaries(free_document_summaries)

    _json_dump(output_dir / "prefill_layer_metrics.json", prefill_layer_metrics)
    _json_dump(output_dir / "prefill_document_metrics.json", prefill_document_metrics)
    torch.save(prefill_raw, output_dir / "prefill_token_metrics.pt")
    torch.save(forced_raw, output_dir / "forced_decode_metrics.pt")
    _json_dump(output_dir / "forced_decode_summary.json", forced_summary)
    torch.save(free_raw, output_dir / "free_generation_metrics.pt")
    _json_dump(output_dir / "free_generation_summary.json", free_summary)
    plot_paths = save_pg19_plots(prefill_layer_metrics, forced_summary, output_dir)

    print(f"[PG19 routing drift] Layer-0 exact routing match: {prefill_layer_metrics[0]['exact_topk_match_rate']:.6f}")
    print(
        "[PG19 routing drift] Routing-before-token fraction: "
        f"{free_summary['fraction_routing_divergence_precedes_token_divergence']:.6f}"
    )
    print(f"[PG19 routing drift] Output directory: {output_dir}")
    for path in plot_paths:
        print(f"[PG19 routing drift] Plot: {path}")


if __name__ == "__main__":
    main()
