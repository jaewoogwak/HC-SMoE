#!/usr/bin/env python3
"""Held-out, layer-local diagnosis for routing/output trade-offs.

This entry point deliberately consumes completed HC-SMoE runs.  It does not
run grouping, change a checkpoint, or invoke lm-eval.  The only model forward
passes are on a held-out C4 trace and compare original/merged sparse-MoE blocks
on exactly the original block input.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Support direct ``python hcsmoe/analyze_routing_output_tradeoff.py`` use.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hcsmoe.evaluation import get_calib_dataloder
from hcsmoe.merging.routing_aware_grouping import routing_metrics


EPS = 1e-8
TASKS = ("winogrande", "arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu", "openbookqa", "rte")
ALPHAS = (0.0, 0.5, 1.0)


def alpha_key(value: float) -> str:
    return f"{float(value):.1f}"


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def discover_alpha_runs(results_dir: Path) -> Dict[float, Dict[str, Any]]:
    """Discover completed routing-aware runs by metadata, never by dirname."""
    found: Dict[float, Dict[str, Any]] = {}
    for metadata_path in results_dir.rglob("group_mapping_metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("grouping_method") != "routing_aware" or metadata.get("alpha") is None:
            continue
        alpha = float(metadata["alpha"])
        if alpha not in ALPHAS:
            continue
        run_dir = metadata_path.parent
        mapping_path = run_dir / metadata.get("group_mapping_file", "group_state_dict.pt")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing saved grouping for alpha={alpha}: {mapping_path}")
        if alpha in found:
            raise ValueError(f"Multiple routing-aware runs advertise alpha={alpha}: {found[alpha]['dir']} and {run_dir}")
        found[alpha] = {"dir": run_dir, "metadata": metadata, "mapping_path": mapping_path,
                        "model_path": run_dir / "model.pth", "lm_eval_path": run_dir / "lm_eval.txt"}
    missing = [alpha_key(alpha) for alpha in ALPHAS if alpha not in found]
    if missing:
        raise FileNotFoundError(f"Missing required routing-aware alpha runs: {', '.join(missing)} under {results_dir}")
    return found


def validate_mappings(runs: Mapping[float, Mapping[str, Any]], expected_layers: int, expected_experts: int) -> Dict[float, Dict[str, torch.Tensor]]:
    """Load mappings and enforce the common layer/expert/target-group topology."""
    mappings = {alpha: _torch_load(run["mapping_path"]) for alpha, run in runs.items()}
    expected_names = None
    group_counts = set()
    for alpha, mapping in mappings.items():
        if len(mapping) != expected_layers:
            raise ValueError(f"alpha={alpha}: expected {expected_layers} sparse layers, found {len(mapping)}")
        names = set(mapping)
        expected_names = names if expected_names is None else expected_names
        if names != expected_names:
            raise ValueError(f"alpha={alpha}: sparse layer names differ from the other alpha runs")
        for name, labels in mapping.items():
            labels = labels.detach().cpu().long()
            if labels.shape != (expected_experts,) or labels.min().item() < 0:
                raise ValueError(f"alpha={alpha}, {name}: invalid expert mapping shape/range")
            mappings[alpha][name] = labels
            group_counts.add(int(labels.unique().numel()))
    if len(group_counts) != 1:
        raise ValueError(f"alpha runs must have the same target group count, found {sorted(group_counts)}")
    target_groups = next(iter(group_counts))
    for alpha, run in runs.items():
        declared = run["metadata"].get("num_average_groups")
        if declared is not None and int(declared) != target_groups:
            raise ValueError(f"alpha={alpha}: metadata declares {declared} groups but mapping has {target_groups}")
    return mappings


def pair_categories(labels0: torch.Tensor, labels05: torch.Tensor, labels1: torch.Tensor) -> List[Dict[str, Any]]:
    """Return category rows; partition labels themselves need not be canonical."""
    rows: List[Dict[str, Any]] = []
    num_experts = labels0.numel()
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            same0 = bool(labels0[i] == labels0[j])
            same05 = bool(labels05[i] == labels05[j])
            same1 = bool(labels1[i] == labels1[j])
            categories: List[Tuple[str, str]] = []
            if same1 and not same05:
                categories.append(("05_vs_1", "routing_only"))
            if same05 and not same1:
                categories.append(("05_vs_1", "hybrid_only_vs_routing"))
            if same05 and same1:
                categories.append(("05_vs_1", "shared_05_1"))
            if same0 and not same05:
                categories.append(("0_vs_05", "hc_only"))
            if same05 and not same0:
                categories.append(("0_vs_05", "hybrid_only_vs_hc"))
            if same0 and same05:
                categories.append(("0_vs_05", "shared_0_05"))
            for comparison, category in categories:
                rows.append({"expert_i": i, "expert_j": j, "comparison": comparison, "category": category,
                             "same_group_a0": same0, "same_group_a05": same05, "same_group_a1": same1})
    return rows


def routing_overlap(topk: torch.Tensor, expert_i: int, expert_j: int) -> Tuple[int, int, float, float]:
    active_i = (topk == expert_i).any(dim=1)
    active_j = (topk == expert_j).any(dim=1)
    intersection = int((active_i & active_j).sum())
    union = int((active_i | active_j).sum())
    return union, intersection, intersection / len(topk), (intersection / union if union else 0.0)


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.linalg.vector_norm(left.float() - right.float(), dim=-1) / (
        torch.linalg.vector_norm(left.float(), dim=-1) + torch.linalg.vector_norm(right.float(), dim=-1) + EPS
    )


def pair_output_statistics(
    experts: Sequence[torch.nn.Module], x_cpu: torch.Tensor, topk_cpu: torch.Tensor,
    pair_rows: Sequence[Mapping[str, Any]], device: torch.device, chunk_size: int, min_corouted_tokens: int,
) -> List[Dict[str, Any]]:
    """Stream expert output statistics without retaining all-token expert outputs."""
    unique_pairs = sorted({(int(row["expert_i"]), int(row["expert_j"])) for row in pair_rows})
    num_experts, hidden = len(experts), x_cpu.shape[-1]
    means = torch.zeros((num_experts, hidden), dtype=torch.float32)
    stats = {pair: {"union_sum": 0.0, "corouted_sum": 0.0, "union_count": 0, "corouted_count": 0}
             for pair in unique_pairs}
    for start in range(0, len(x_cpu), chunk_size):
        x = x_cpu[start:start + chunk_size].to(device, non_blocking=True)
        # This is bounded [experts, chunk, hidden], not a whole-trace tensor.
        outputs = torch.stack([expert(x).float() for expert in experts], dim=0)
        means += outputs.sum(dim=1).cpu()
        topk = topk_cpu[start:start + len(x)]
        for i, j in unique_pairs:
            active_i = (topk == i).any(dim=1).to(device)
            active_j = (topk == j).any(dim=1).to(device)
            union = active_i | active_j
            corouted = active_i & active_j
            rel = relative_l2(outputs[i], outputs[j])
            if union.any():
                stats[(i, j)]["union_sum"] += float(rel[union].sum())
                stats[(i, j)]["union_count"] += int(union.sum())
            if corouted.any():
                stats[(i, j)]["corouted_sum"] += float(rel[corouted].sum())
                stats[(i, j)]["corouted_count"] += int(corouted.sum())
        del x, outputs
    means /= len(x_cpu)
    result: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, j in unique_pairs:
        union, corouted, coactivation, jaccard = routing_overlap(topk_cpu, i, j)
        values = stats[(i, j)]
        if union != values["union_count"] or corouted != values["corouted_count"]:
            raise AssertionError("routing mask counts changed while computing pair statistics")
        corouted_rel = (values["corouted_sum"] / corouted if corouted >= min_corouted_tokens else math.nan)
        result[(i, j)] = {
            "hc_mean_l2": float(torch.linalg.vector_norm(means[i] - means[j])),
            "hc_mean_rel_l2": float(relative_l2(means[i][None], means[j][None])[0]),
            "coactivation_rate": coactivation, "routing_jaccard": jaccard,
            "active_union_rel_l2": values["union_sum"] / union if union else math.nan,
            "corouted_rel_l2": corouted_rel, "num_union_tokens": union, "num_corouted_tokens": corouted,
        }
    return [{**dict(row), **result[(int(row["expert_i"]), int(row["expert_j"]))]} for row in pair_rows]


def moe_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def execution_device(module: torch.nn.Module, is_qwen: bool) -> torch.device:
    if is_qwen:
        from hcsmoe.merging.grouping_qwen import module_execution_device
        return module_execution_device(module)
    hook = getattr(module, "_hf_hook", None)
    if getattr(hook, "execution_device", None) is not None:
        return torch.device(hook.execution_device)
    parameter = next((parameter for parameter in module.parameters() if parameter.device.type != "meta"), None)
    return parameter.device if parameter is not None else torch.device("cpu")


@torch.no_grad()
def run_moe_on_input(moe: torch.nn.Module, x_cpu: torch.Tensor, *, is_qwen: bool, chunk_size: int) -> torch.Tensor:
    device = execution_device(moe, is_qwen)
    chunks = []
    for start in range(0, len(x_cpu), chunk_size):
        x = x_cpu[start:start + chunk_size].to(device, non_blocking=True)
        chunks.append(moe_output(moe(x)).float().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def local_moe_metrics(original_moe: torch.nn.Module, merged_moe: torch.nn.Module, x_cpu: torch.Tensor, *, is_qwen: bool, chunk_size: int) -> Dict[str, float]:
    """Testing helper: both blocks consume the identical tensor object ``x_cpu``."""
    original = run_moe_on_input(original_moe, x_cpu, is_qwen=is_qwen, chunk_size=chunk_size)
    merged = run_moe_on_input(merged_moe, x_cpu, is_qwen=is_qwen, chunk_size=chunk_size)
    return local_metrics_from_outputs(original, merged)


def local_metrics_from_outputs(original: torch.Tensor, merged: torch.Tensor) -> Dict[str, float]:
    if original.shape != merged.shape:
        raise ValueError(f"MoE output shape mismatch: {tuple(original.shape)} vs {tuple(merged.shape)}")
    per_token = torch.linalg.vector_norm(original - merged, dim=-1) / (torch.linalg.vector_norm(original, dim=-1) + EPS)
    cosine = F.cosine_similarity(original.float(), merged.float(), dim=-1, eps=EPS)
    if not torch.isfinite(per_token).all() or not torch.isfinite(cosine).all():
        raise FloatingPointError("non-finite local MoE comparison")
    return {"local_rel_l2": float(per_token.mean()), "local_rel_l2_p95": float(torch.quantile(per_token, 0.95)),
            "local_cosine": float(cosine.mean())}


def capture_heldout_trace(model: Any, dataloader: Iterable[Mapping[str, torch.Tensor]], layer_names: Sequence[str], *, is_qwen: bool, top_k: int) -> Dict[str, Dict[str, torch.Tensor]]:
    """Capture original MoE inputs, IDs, logits-derived selected probabilities on CPU."""
    inputs = {name: [] for name in layer_names}
    selected = {name: [] for name in layer_names}
    probabilities = {name: [] for name in layer_names}
    logits_by_layer = {name: [] for name in layer_names}
    handles = []
    layers = model.model.layers

    def hook(name: str):
        def capture(_, module_inputs, __):
            inputs[name].append(module_inputs[0].detach().reshape(-1, module_inputs[0].shape[-1]).cpu())
        return capture

    for index, name in enumerate(layer_names):
        moe = layers[index].mlp if is_qwen else layers[index].block_sparse_moe
        handles.append(moe.register_forward_hook(hook(name)))
    try:
        for batch in tqdm(dataloader, desc="[analysis] Collecting held-out routing and MoE inputs"):
            batch = {key: value.cuda() for key, value in batch.items() if key != "labels"}
            outputs = model(**batch, output_router_logits=True, use_cache=False)
            for index, name in enumerate(layer_names):
                logits = outputs.router_logits[index].reshape(-1, outputs.router_logits[index].shape[-1]).float()
                full_probs = F.softmax(logits, dim=-1)
                probs, ids = torch.topk(full_probs, top_k, dim=-1)
                moe = layers[index].mlp if is_qwen else layers[index].block_sparse_moe
                # Mixtral always normalizes selected weights; Qwen does so only
                # when norm_topk_prob is enabled in the saved base configuration.
                if not is_qwen or getattr(moe, "norm_topk_prob", False):
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                selected[name].append(ids.cpu())
                probabilities[name].append(probs.cpu())
                logits_by_layer[name].append(logits.cpu())
            del outputs
    finally:
        for handle in handles:
            handle.remove()
    trace = {}
    for name in layer_names:
        trace[name] = {"inputs": torch.cat(inputs[name]), "topk": torch.cat(selected[name]),
                       "router_logits": torch.cat(logits_by_layer[name]), "routing_probs": torch.cat(probabilities[name])}
        if len(trace[name]["inputs"]) != len(trace[name]["topk"]):
            raise AssertionError(f"{name}: input and router trace have different token counts")
    return trace


def parse_raw_accuracy(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text()
    scores = {}
    for task in TASKS:
        match = re.search(rf"^\|\s*{re.escape(task)}\s*\|.*?\|\s*acc\s*\|.*?\|\s*([0-9]+(?:\.[0-9]+)?)\s*\|", text, re.MULTILINE)
        if match:
            scores[task] = float(match.group(1))
    return scores


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def representative_pairs(pair_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    def finite(row: Mapping[str, Any], key: str) -> float:
        value = float(row.get(key, math.nan))
        return value if math.isfinite(value) else -math.inf
    candidates = {
        "mean_output_false_friends": [row for row in pair_rows if row["category"] == "hc_only"],
        "hybrid_compatible_pairs": [row for row in pair_rows if row["category"] == "hybrid_only_vs_hc"],
        "routing_only_harmful_pairs": [row for row in pair_rows if row["category"] == "routing_only"],
    }
    return {
        "mean_output_false_friends": sorted(candidates["mean_output_false_friends"], key=lambda r: (r["hc_mean_l2"], r["coactivation_rate"], -finite(r, "active_union_rel_l2")))[:3],
        "hybrid_compatible_pairs": sorted(candidates["hybrid_compatible_pairs"], key=lambda r: (-r["coactivation_rate"], finite(r, "active_union_rel_l2"), r["hc_mean_l2"]))[:3],
        "routing_only_harmful_pairs": sorted(candidates["routing_only_harmful_pairs"], key=lambda r: (-r["coactivation_rate"], -finite(r, "corouted_rel_l2")))[:3],
    }


def write_summary_markdown(path: Path, layer_rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]], accuracies: Mapping[float, Mapping[str, float]], representatives: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    lines = ["# Routing/output trade-off (held-out C4)", "", "## A. Final trade-off", "", "| alpha | raw-acc avg | held-out Mean U | U=1 | local relL2 | cosine |", "|---:|---:|---:|---:|---:|---:|"]
    for alpha in ALPHAS:
        rows = [row for row in layer_rows if row["alpha"] == alpha]
        accuracy = accuracies.get(alpha, {})
        avg_acc = sum(accuracy.values()) / len(accuracy) if len(accuracy) == len(TASKS) else math.nan
        lines.append("| %.1f | %s | %.4f | %.4f | %.4f | %.4f |" % (
            alpha, f"{avg_acc:.4f}" if math.isfinite(avg_acc) else "missing",
            _mean(rows, "mean_unique_groups") or math.nan, _mean(rows, "u1_rate") or math.nan,
            _mean(rows, "local_rel_l2") or math.nan, _mean(rows, "local_cosine") or math.nan))
    metric_names = ("hc_mean_l2", "coactivation_rate", "routing_jaccard", "active_union_rel_l2", "corouted_rel_l2")
    for title, categories in (("B. alpha=0.5 vs alpha=1 pair mechanism", ("routing_only", "hybrid_only_vs_routing", "shared_05_1")),
                              ("C. alpha=0 vs alpha=0.5 pair mechanism", ("hc_only", "hybrid_only_vs_hc", "shared_0_05"))):
        lines.extend(["", f"## {title}", "", "| category | pairs | hc mean L2 | coactivation | Jaccard | active-union relL2 | corouted relL2 |", "|---|---:|---:|---:|---:|---:|---:|"])
        for category in categories:
            rows = [row for row in pair_rows if row["category"] == category]
            values = [_mean(rows, metric) for metric in metric_names]
            lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (category, len(rows), *["—" if value is None else f"{value:.4f}" for value in values]))
    lines.extend(["", "## Representative pairs", ""])
    for title, rows in representatives.items():
        lines.append(f"### {title.replace('_', ' ')}")
        for row in rows:
            lines.append("- L%s E%s/E%s: hc=%.4f, coact=%.4f, union-relL2=%.4f, corouted-relL2=%s" % (
                row["layer"], row["expert_i"], row["expert_j"], row["hc_mean_l2"], row["coactivation_rate"], row["active_union_rel_l2"],
                "—" if not math.isfinite(float(row["corouted_rel_l2"])) else f"{row['corouted_rel_l2']:.4f}"))
        if not rows:
            lines.append("- No pair in this category.")
        lines.append("")
    path.write_text("\n".join(lines))


def model_spec(model: str, requested_model_name: str | None) -> Dict[str, Any]:
    if model == "mixtral":
        return {"model_name": requested_model_name or "mistralai/Mixtral-8x7B-v0.1", "results_dir": Path("results/mixtral_8to4"),
                "experts": 8, "is_qwen": False, "layer_attr": "block_sparse_moe"}
    return {"model_name": requested_model_name or "Qwen/Qwen1.5-MoE-A2.7B-Chat", "results_dir": Path("results/qwen_60to30"),
            "experts": 60, "is_qwen": True, "layer_attr": "mlp"}


def load_original(spec: Mapping[str, Any]) -> Any:
    from transformers import MixtralForCausalLM, Qwen2MoeForCausalLM
    cls = Qwen2MoeForCausalLM if spec["is_qwen"] else MixtralForCausalLM
    return cls.from_pretrained(spec["model_name"], torch_dtype=torch.bfloat16, device_map="auto").eval()


def load_merged(spec: Mapping[str, Any], run: Mapping[str, Any]) -> Any:
    model_path = run["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved merged checkpoint is required: {model_path}")
    if not spec["is_qwen"]:
        from hcsmoe.merging.mixtral_checkpoint import load_compressed_model_for_evaluation
        model, _ = load_compressed_model_for_evaluation(spec["model_name"], str(model_path), str(run["mapping_path"]))
        return model.eval()
    from transformers import Qwen2MoeForCausalLM
    model = Qwen2MoeForCausalLM.from_pretrained(spec["model_name"], torch_dtype=torch.bfloat16, device_map="auto").eval()
    result = model.load_state_dict(_torch_load(model_path), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Unexpected Qwen checkpoint mismatch: {result}")
    return model


def analyze(args: argparse.Namespace) -> None:
    spec = model_spec(args.model, args.model_name)
    results_dir = Path(args.results_dir) if args.results_dir else spec["results_dir"]
    output_dir = Path(args.output_dir) if args.output_dir else Path("results/analysis") / args.model / "routing_output_tradeoff"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_alpha_runs(results_dir)
    metadata_name = runs[0.0]["metadata"].get("model_name")
    if args.model_name is None and metadata_name:
        spec["model_name"] = metadata_name

    original = load_original(spec)
    num_layers = original.config.num_hidden_layers
    num_experts = original.config.num_experts if spec["is_qwen"] else original.config.num_local_experts
    if num_experts != spec["experts"]:
        raise ValueError(f"{args.model}: expected {spec['experts']} experts, model has {num_experts}")
    mappings = validate_mappings(runs, num_layers, num_experts)
    layer_names = [f"model.layers.{index}.{'mlp' if spec['is_qwen'] else 'block_sparse_moe'}" for index in range(num_layers)]
    top_k = original.config.num_experts_per_tok
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["model_name"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    loader = get_calib_dataloder("c4", tokenizer, args.block_size, args.analysis_blocks, args.batch_size, args.num_workers, args.analysis_seed)
    trace = capture_heldout_trace(original, loader, layer_names, is_qwen=spec["is_qwen"], top_k=top_k)

    pair_rows: List[Dict[str, Any]] = []
    original_outputs: Dict[str, Path] = {}
    cache_dir = Path(tempfile.mkdtemp(prefix="routing-output-cache-", dir=output_dir))
    try:
        for index, name in enumerate(tqdm(layer_names, desc="[analysis] Pair-level held-out output metrics")):
            labels0, labels05, labels1 = mappings[0.0][name], mappings[0.5][name], mappings[1.0][name]
            categories = pair_categories(labels0, labels05, labels1)
            moe = original.model.layers[index].mlp if spec["is_qwen"] else original.model.layers[index].block_sparse_moe
            values = pair_output_statistics(moe.experts, trace[name]["inputs"], trace[name]["topk"], categories,
                                            execution_device(moe, spec["is_qwen"]), args.chunk_size, args.min_corouted_tokens)
            for row in values:
                for metric in ("hc_mean_l2", "hc_mean_rel_l2", "coactivation_rate", "routing_jaccard", "active_union_rel_l2"):
                    if not math.isfinite(float(row[metric])):
                        raise FloatingPointError(f"{name} E{row['expert_i']}/E{row['expert_j']}: non-finite {metric}")
                if row["num_corouted_tokens"] >= args.min_corouted_tokens and not math.isfinite(float(row["corouted_rel_l2"])):
                    raise FloatingPointError(f"{name} E{row['expert_i']}/E{row['expert_j']}: non-finite corouted relL2")
            pair_rows.extend({"model": args.model, "layer": index, **row} for row in values)
            original_path = cache_dir / f"original_layer_{index}.pt"
            torch.save(run_moe_on_input(moe, trace[name]["inputs"], is_qwen=spec["is_qwen"], chunk_size=args.chunk_size), original_path)
            original_outputs[name] = original_path
        del original
        gc.collect()
        torch.cuda.empty_cache()

        layer_rows: List[Dict[str, Any]] = []
        for alpha in ALPHAS:
            merged = load_merged(spec, runs[alpha])
            for index, name in enumerate(tqdm(layer_names, desc=f"[analysis] Local distortion alpha={alpha:g}")):
                moe = merged.model.layers[index].mlp if spec["is_qwen"] else merged.model.layers[index].block_sparse_moe
                merged_output = run_moe_on_input(moe, trace[name]["inputs"], is_qwen=spec["is_qwen"], chunk_size=args.chunk_size)
                metrics = local_metrics_from_outputs(_torch_load(original_outputs[name]), merged_output)
                locality = routing_metrics(mappings[alpha][name], trace[name]["topk"])
                row = {"model": args.model, "layer": index, "alpha": alpha, **locality, **metrics}
                if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"model", "layer", "alpha", "num_tokens"}):
                    raise FloatingPointError(f"Non-finite held-out layer metric: {row}")
                layer_rows.append(row)
            del merged
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    pair_fields = ("model", "layer", "expert_i", "expert_j", "comparison", "category", "same_group_a0", "same_group_a05", "same_group_a1",
                   "hc_mean_l2", "hc_mean_rel_l2", "coactivation_rate", "routing_jaccard", "active_union_rel_l2", "corouted_rel_l2", "num_union_tokens", "num_corouted_tokens")
    layer_fields = ("model", "layer", "alpha", "num_tokens", "mean_unique_groups", *[f"u{index}_rate" for index in range(1, top_k + 1)],
                    "local_rel_l2", "local_rel_l2_p95", "local_cosine")
    write_csv(output_dir / "pair_metrics.csv", pair_rows, pair_fields)
    write_csv(output_dir / "layer_metrics.csv", layer_rows, layer_fields)
    accuracies = {alpha: parse_raw_accuracy(runs[alpha]["lm_eval_path"]) for alpha in ALPHAS}
    category_summary = {category: {metric: _mean([row for row in pair_rows if row["category"] == category], metric)
                                   for metric in ("hc_mean_l2", "coactivation_rate", "routing_jaccard", "active_union_rel_l2", "corouted_rel_l2")}
                        for category in sorted({row["category"] for row in pair_rows})}
    summary = {"model": args.model, "model_name": spec["model_name"], "heldout": {"dataset": "c4", "analysis_seed": args.analysis_seed,
               "analysis_blocks": args.analysis_blocks, "block_size": args.block_size, "top_k": top_k},
               "runs": {alpha_key(alpha): {"path": str(runs[alpha]["dir"]), "raw_accuracy": accuracies[alpha],
                           "raw_accuracy_average": _mean([{"score": score} for score in accuracies[alpha].values()], "score")}
                        for alpha in ALPHAS},
               "layer_average": {alpha_key(alpha): {key: _mean([row for row in layer_rows if row["alpha"] == alpha], key)
                                                      for key in ("mean_unique_groups", "u1_rate", "local_rel_l2", "local_rel_l2_p95", "local_cosine")}
                                 for alpha in ALPHAS},
               "pair_category_average": category_summary}
    representatives = representative_pairs(pair_rows)
    summary["representative_pairs"] = representatives
    (output_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    write_summary_markdown(output_dir / "summary.md", layer_rows, pair_rows, accuracies, representatives)
    print(f"[analysis] wrote {output_dir / 'pair_metrics.csv'}")
    print(f"[analysis] wrote {output_dir / 'layer_metrics.csv'}")
    print(f"[analysis] wrote {output_dir / 'summary.json'}")
    print(f"[analysis] wrote {output_dir / 'summary.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mixtral", "qwen"), required=True)
    parser.add_argument("--results_dir", help="Completed routing-aware result root; alpha runs are discovered from metadata.")
    parser.add_argument("--output_dir", help="Default: results/analysis/<model>/routing_output_tradeoff")
    parser.add_argument("--model_name", help="Override model name from run metadata.")
    parser.add_argument("--analysis_seed", type=int, default=123)
    parser.add_argument("--analysis_blocks", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--min_corouted_tokens", type=int, default=32)
    return parser


if __name__ == "__main__":
    analyze(build_parser().parse_args())
