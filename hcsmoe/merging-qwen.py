"""Merge Qwen MoE experts with HC-SMoE or shared routing-aware grouping."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fire import Fire
from transformers import AutoTokenizer, Qwen2MoeForCausalLM

from hcsmoe.evaluation import evaluate_fewshot, get_calib_dataloder
from hcsmoe.merging.grouping_qwen import ExpertsGrouperForQwen2MoE, merge_by_groups_with_usage_weighted
from hcsmoe.merging.routing_aware_grouping import routing_aware_grouping


DEFAULT_TASKS = "winogrande,arc_challenge,arc_easy,boolq,hellaswag,mmlu,openbookqa,rte"


class Args:
    def __init__(self, task, n_sentences, train_batch_size, eval_batch_size, result_path, num_fewshot):
        self.task = task
        self.n_sentences = n_sentences
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.result_path = result_path
        self.num_fewshot = num_fewshot


def get_dataloader(args, tokenizer, calib_seed: int):
    return get_calib_dataloder(
        dataset="c4", tokenizer=tokenizer, max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, batch_size=args.train_batch_size,
        num_workers=4, seed=calib_seed,
    )


def evaluate(args, model, tokenizer):
    if not args.result_path:
        return
    result_dir = os.path.dirname(args.result_path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    tasks = args.task.split(",") if isinstance(args.task, str) else list(args.task)
    for task in tasks:
        evaluate_fewshot(
            model, tokenizer=tokenizer, task=task.strip(),
            num_fewshot=args.num_fewshot, eval_batch_size=args.eval_batch_size,
            output_path=args.result_path, log=True,
        )


def _save_routing_diagnostics(output_path, routing_results, alpha: float, calib_seed: int):
    with open(os.path.join(output_path, "routing_merge_trace.json"), "w") as handle:
        json.dump({name: result["merge_trace"] for name, result in routing_results.items()}, handle, indent=2)
    with open(os.path.join(output_path, "routing_locality_metrics.json"), "w") as handle:
        json.dump({"alpha": alpha, "calib_seed": calib_seed,
                   "layers": {name: result["metrics"] for name, result in routing_results.items()}}, handle, indent=2)


def run_hcsmoe(
        task: str = DEFAULT_TASKS,
        num_average_groups: int = 30,
        model_name: str = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        grouping_method: str = "hcsmoe",
        alpha: float = 1.0,
        calib_seed: int = 42,
        n_sentences: int = 32,
        train_batch_size: int = 2,
        eval_batch_size: int = 16,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        num_fewshot: int = 0,
        start_layer: int = 0,
        similarity_base: str = "expert-output",
        cluster: str = "hierarchical",
        linkage: str = "average",
        merge: str = "freq",
):
    """Run a 60-expert to configurable-group Qwen comparison."""
    if grouping_method not in {"hcsmoe", "routing_aware"}:
        raise ValueError("grouping_method must be 'hcsmoe' or 'routing_aware'")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if similarity_base != "expert-output" or cluster != "hierarchical" or linkage != "average":
        raise ValueError("this comparison requires expert-output hierarchical average-linkage grouping")
    if merge != "freq":
        raise ValueError("this comparison uses only the original merge=freq implementation")
    if not output_path:
        raise ValueError("--output_path is required for a merged model")
    args = Args(task, n_sentences, train_batch_size, eval_batch_size, result_path, num_fewshot)
    torch.manual_seed(calib_seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2MoeForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    if not 1 <= num_average_groups <= model.config.num_experts:
        raise ValueError(f"num_average_groups must be in [1, {model.config.num_experts}]")

    dataloader = get_dataloader(args, tokenizer, calib_seed)
    grouper = ExpertsGrouperForQwen2MoE(
        config=model.config, similarity_base=similarity_base, start_layer=start_layer,
        cluster=cluster, linkage=linkage,
    )
    print(f"[HC-SMoE] grouping_method={grouping_method}, alpha={alpha:g}, calib_seed={calib_seed}")
    started = time.time()
    routing_results = None
    if grouping_method == "hcsmoe":
        # Original Qwen expert-output HC-SMoE grouping path.
        grouper.compute_all_usages(model, dataloader)
        grouper.cluster_experts(model, dataloader, num_average_groups)
    else:
        collected = grouper.collect_routing_aware_data(model, dataloader)
        routing_results = {
            name: routing_aware_grouping(values["output_distance"], values["topk_experts"], num_average_groups, alpha)
            for name, values in collected.items()
        }
        grouper.set_group_state_dict({name: result["labels"] for name, result in routing_results.items()})
    group_state = grouper.group_state_dict()
    print(f"[HC-SMoE] Grouping completed in {time.time() - started:.1f}s")

    # Both policies deliberately share the unchanged frequency-weighted merge.
    model = merge_by_groups_with_usage_weighted(
        model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
    )
    os.makedirs(output_path, exist_ok=True)
    grouper.save_group_state_dict(output_path)
    metadata = {
        "model_name": model_name, "grouping_method": grouping_method,
        "alpha": alpha if grouping_method == "routing_aware" else None,
        "calibration_dataset": "c4", "calibration_blocks": n_sentences,
        "calibration_block_size": 2048, "calib_seed": calib_seed,
        "similarity_base": similarity_base, "cluster": cluster, "linkage": linkage,
        "num_average_groups": num_average_groups, "merge": merge,
        "top_k": model.config.num_experts_per_tok,
        "group_mapping_file": "group_state_dict.pt",
    }
    with open(os.path.join(output_path, "group_mapping_metadata.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    if routing_results is not None:
        _save_routing_diagnostics(output_path, routing_results, alpha, calib_seed)
    torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))
    torch.cuda.empty_cache()
    evaluate(args, model, tokenizer)


if __name__ == "__main__":
    Fire(run_hcsmoe)
