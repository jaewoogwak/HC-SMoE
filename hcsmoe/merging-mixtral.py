# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18

# modified
# @Author: wazenmai
# @Time: 2024/8/13

import os
import sys
import json
import time
import torch
import pickle
import logging
import itertools
from fire import Fire
from tqdm import tqdm
from typing import Optional
from accelerate import init_empty_weights
from accelerate.utils.modeling import get_state_dict_offloaded_model
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer

from hcsmoe.evaluation import (
    evaluate_fewshot,
    evaluate_generation,
    get_calib_dataloder,
    validate_generation_tasks,
)
from hcsmoe.merging.grouping_mixtral import ExpertsGrouperForMixtral
from hcsmoe.merging.grouping_mixtral import merge_by_groups_with_usage_weighted, merge_by_groups_within_and_across_models
from hcsmoe.merging.pairwise_scores import (
    canonical_groups,
    load_or_compute_pairwise_scores,
    partitions_equal,
    save_pairwise_scores,
)
from hcsmoe.merging.residual_mixtral import (
    collect_residual_calibration,
    save_residual_loss_curves,
    save_residual_artifacts,
    train_residuals,
    _parse_diagnostic_experts,
)
from hcsmoe.merging.lora_mixtral import save_lora_artifacts, save_lora_loss_curves, train_lora_experts
from hcsmoe.models.mixtral.utils import (
    expand_shared_expert_state_dict,
)
from hcsmoe.merging.mixtral_checkpoint import load_compressed_model_for_evaluation

logger = logging.getLogger(__name__)

from typing import Optional


###############################
#  Helper class and functions #
###############################
class Args:
    def __init__(
        self,
        task,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        dominant: Optional[str] = "knowledge",
        similarity_base: Optional[str] = "router-logits",
        merge: Optional[str] = "zipit",
        mode: Optional[str] = "normal",
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        random_start_center: Optional[bool] = False,
        weight= None,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        overlap_metric: Optional[str] = "cosine",
        dynamic_group: Optional[bool] = False,
    ):
        self.task = task
        self.num_average_groups = num_average_groups
        self.model_name = model_name
        self.dominant = dominant
        self.similarity_base = similarity_base
        self.merge = merge
        self.mode = mode
        self.n_sentences = n_sentences
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.partition = partition
        self.start_layer = start_layer
        self.output_path = output_path
        self.result_path = result_path
        self.model_path = model_path
        self.group_limit = group_limit
        self.data_limit = data_limit
        self.num_fewshot = num_fewshot
        self.random_start_center = random_start_center
        self.weight = weight
        self.cluster = cluster
        self.linkage = linkage
        self.hierarchical_stopping_metric = hierarchical_stopping_metric
        self.overlap_metric = overlap_metric
        self.dynamic_group = dynamic_group

def get_dataloader(args, tokenizer):
    return get_calib_dataloder(
        dataset="c4",
        tokenizer=tokenizer,
        max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, # 32, 128
        batch_size=args.train_batch_size,
        num_workers=4,
    )

def get_grouper(args, config):
    return ExpertsGrouperForMixtral(
        config=config,
        similarity_base=args.similarity_base,
        start_layer=args.start_layer,
        group_limit=args.group_limit,
        data_limit=args.data_limit,
        random_start_center=args.random_start_center,
        cluster=args.cluster,
        linkage=args.linkage,
        hierarchical_stopping_metric=args.hierarchical_stopping_metric,
        overlap_metric=args.overlap_metric,
        dynamic_group=args.dynamic_group,
    )

def evaluation(args, model, tokenizer):
    result_dir = args.result_path.split("/")[:-1]
    result_dir = "/".join(result_dir)

    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if isinstance(args.task, str):
        tasks = args.task.split(",")
    else:
        tasks = list(args.task)

    for t in tasks:
        evaluate_fewshot(
            model,
            tokenizer=tokenizer,
            task=t,
            num_fewshot=args.num_fewshot,
            eval_batch_size=args.eval_batch_size,
            output_path=args.result_path,
            log=True,
        )


def run_requested_evaluation(
        args,
        model,
        tokenizer,
        eval_generation: bool,
        eval_coding: bool,
        eval_math: bool,
        eval_limit: Optional[int],
):
    """Keep existing MC evaluation unchanged; generation is an explicit suite."""
    if eval_generation or eval_coding or eval_math:
        generation_output_path = args.output_path or os.path.dirname(args.result_path) or "."
        evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            eval_coding=eval_generation or eval_coding,
            eval_math=eval_generation or eval_math,
            eval_batch_size=args.eval_batch_size,
            output_path=generation_output_path,
            eval_limit=eval_limit,
        )
        return
    evaluation(args, model, tokenizer)


def print_usage_frequency(usage_dict):
    for k in usage_dict:
        for num in usage_dict[k]:
            print(round(num.item(), 4), end=',')
        print()


###############################
###      Main function      ###
###############################
def run_hcsmoe(
        task: str,
        num_average_groups: int,
        model_name: Optional[str] = "mistralai/Mixtral-8x7B-v0.1",
        dominant: Optional[str] = "knowledge", # random, frequency, no
        similarity_base: Optional[str] = "router-logits", # router-logits, weight, expert-output
        merge: Optional[str] = "zipit", # no, freq, zipit, kl-weight, fix-dom-same
        mode: Optional[str] = "normal", # normal, activation-with-router-logits, input-weight, all
        n_sentences: Optional[int] = 32,
        train_batch_size: Optional[int] = 4,
        eval_batch_size: Optional[int] = 32,
        partition: Optional[int] = 1,
        start_layer: Optional[int] = 0,
        output_path: Optional[str] = None,
        result_path: Optional[str] = None,
        model_path: Optional[str] = None,
        group_limit: Optional[int] = 4,
        data_limit: Optional[int] = 50000,
        num_fewshot: Optional[int] = 0,
        random_start_center: Optional[bool] = False,
        cluster: Optional[str] = "kmeans",
        linkage: Optional[str] = "ward",
        hierarchical_stopping_metric: Optional[str] = "silhouette",
        ingredient: Optional[str] = "act", # act, weight, act+weight
        overlap_metric: Optional[str] = "cosine", # kl-divergence, wasserstein, cosine,
        dynamic_group: Optional[bool] = False,
        eval_only: Optional[bool] = False,
        residual_width: Optional[int] = 0,
        residual_data_limit: Optional[int] = 4096,
        residual_epochs: Optional[int] = 3,
        residual_lr: Optional[float] = 1e-3,
        residual_batch_size: Optional[int] = 64,
        residual_val_ratio: Optional[float] = 0.1,
        residual_patience: Optional[int] = 2,
        residual_diagnostic_experts: Optional[str] = "",
        residual_eval_only: Optional[bool] = False,
        residual_path: Optional[str] = None,
        lora_rank: Optional[int] = 0,
        lora_alpha: Optional[float] = 56,
        lora_data_limit: Optional[int] = 4096,
        lora_epochs: Optional[int] = 3,
        lora_lr: Optional[float] = 1e-4,
        lora_batch_size: Optional[int] = 64,
        lora_val_ratio: Optional[float] = 0.1,
        lora_patience: Optional[int] = 2,
        lora_eval_only: Optional[bool] = False,
        lora_path: Optional[str] = None,
        group_state_path: Optional[str] = None,
        seed: Optional[int] = 0,
        eval_generation: Optional[bool] = False,
        eval_coding: Optional[bool] = False,
        eval_math: Optional[bool] = False,
        eval_limit: Optional[int] = None,
        score_only: bool = False,
        score_output_path: Optional[str] = None,
        score_chunk_size: int = 256,
        hybrid_grouping: bool = False,
        hybrid_alpha: float = 1.0,
        pairwise_score_path: Optional[str] = None,
        recompute_pairwise_scores: bool = False,
        verify_alpha_one: bool = True,
):
    print(f"Merge model {model_name} with {num_average_groups} group, {dominant} dominant + {similarity_base} grouping + {merge} {mode} merge with ingredient {ingredient}, evaluate on {task}")
    print(f"Cluster: {cluster}, linkage: {linkage}, hierarchical_stopping_metric: {hierarchical_stopping_metric}, overlap_metric: {overlap_metric}, dynamic_group: {dynamic_group}")

    if score_only and hybrid_grouping:
        raise ValueError("--score_only=True and --hybrid_grouping=True cannot be used together.")
    if hybrid_grouping and not pairwise_score_path:
        raise ValueError("--hybrid_grouping=True requires --pairwise_score_path.")
    if hybrid_grouping and not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("--hybrid_alpha must be in [0, 1].")
    if hybrid_grouping and (cluster != "hierarchical" or linkage != "average"):
        raise ValueError("Hybrid grouping supports only --cluster=hierarchical and --linkage=average.")


    ### 1. Initialization
    args = Args(
        task=task,
        num_average_groups=num_average_groups,
        model_name=model_name,
        dominant=dominant,
        similarity_base=similarity_base,
        merge=merge,
        mode=mode,
        n_sentences=n_sentences,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        partition=partition,
        start_layer=start_layer,
        output_path=output_path,
        result_path=result_path,
        model_path=model_path,
        group_limit=group_limit,
        data_limit=data_limit,
        num_fewshot=num_fewshot,
        random_start_center=random_start_center,
        cluster=cluster,
        linkage=linkage,
        hierarchical_stopping_metric=hierarchical_stopping_metric,
        overlap_metric=overlap_metric,
        dynamic_group=dynamic_group,
    )

    if eval_generation or eval_coding or eval_math:
        validate_generation_tasks(eval_generation or eval_coding, eval_generation or eval_math)
    
    torch.manual_seed(seed)
    eval_ppl = (task == "minipile")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if eval_only:
        if residual_eval_only and lora_eval_only:
            raise ValueError("--residual_eval_only and --lora_eval_only are mutually exclusive")
        if not model_path:
            raise ValueError("--eval_only=True requires --model_path.")
        checkpoint_dir = os.path.dirname(model_path) or "."
        group_state_path = group_state_path or os.path.join(checkpoint_dir, "group_state_dict.pt")
        residual_path = residual_path or os.path.join(checkpoint_dir, "residuals.pth")
        lora_path = lora_path or os.path.join(checkpoint_dir, "lora.pth")
        model, _ = load_compressed_model_for_evaluation(
            model_name, model_path, group_state_path, residual_eval_only, residual_path,
            lora_eval_only=lora_eval_only, lora_path=lora_path,
        )
        print(f"[HC-SMoE] Evaluating saved model from {model_path}")
        model.eval()
        run_requested_evaluation(
            args, model, tokenizer, eval_generation, eval_coding, eval_math, eval_limit
        )
        return
    model = MixtralForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    if model_path:
        state_dict = torch.load(model_path, map_location="cpu")
        load_result = model.load_state_dict(state_dict)
        print(
            "[HC-SMoE] Checkpoint load keys: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
        for expert_idx in (0, 1):
            checkpoint_key = (
                f"model.layers.0.block_sparse_moe.experts.{expert_idx}.w1.weight"
            )
            loaded_weight = (
                model.model.layers[0].block_sparse_moe.experts[expert_idx].w1.weight
            )
            if loaded_weight.is_meta:
                raise RuntimeError(
                    f"Checkpoint smoke check cannot read meta parameter: {checkpoint_key}"
                )
            matches_checkpoint = torch.equal(
                loaded_weight.detach().to("cpu"), state_dict[checkpoint_key]
            )
            print(
                f"[HC-SMoE] Checkpoint smoke check {checkpoint_key}: "
                f"{matches_checkpoint}"
            )
            if not matches_checkpoint:
                raise RuntimeError(f"Checkpoint smoke check failed: {checkpoint_key}")
        del state_dict
    model.eval()
    
    # Generation C4 calibration dataset 
    dataloader_for_merging = get_dataloader(args, tokenizer) # 2048 token block * 32 (bs 2)

    grouper = get_grouper(args, model.config)

    if score_only:
        score_path = score_output_path or (os.path.join(output_path, "pairwise_scores.pt") if output_path else None)
        if not score_path:
            raise ValueError("--score_only=True requires --score_output_path or --output_path.")
        layers = grouper.compute_pairwise_score_matrices(model, dataloader_for_merging, chunk_size=score_chunk_size)
        save_pairwise_scores(score_path, {
            "model_name": model_name,
            "dataset": "c4",
            "num_blocks": n_sentences,
            "block_size": 2048,
            "num_calibration_tokens": n_sentences * 2048,
            "num_experts": grouper.num_experts,
            "top_k": grouper.topk,
            "chunk_size": score_chunk_size,
            "seed": seed,
            "output_definition": "mean expert output over common MoE inputs",
            "routing_definition": "token-level top-k co-activation count",
        }, layers)
        print(f"[HC-SMoE] Saved pairwise scores: {score_path}")
        return

    pairwise_scores = None
    if hybrid_grouping:
        cache_metadata = {
            "model_name": model_name,
            "dataset": "c4",
            "num_blocks": n_sentences,
            "block_size": 2048,
            "num_calibration_tokens": n_sentences * 2048,
            "num_experts": grouper.num_experts,
            "top_k": grouper.topk,
            "chunk_size": score_chunk_size,
            "seed": seed,
            "output_definition": "mean expert output over common MoE inputs",
            "routing_definition": "token-level top-k co-activation count",
        }
        pairwise_scores = load_or_compute_pairwise_scores(
            pairwise_score_path, model, grouper, dataloader_for_merging,
            cache_metadata, score_chunk_size, recompute=recompute_pairwise_scores,
        )

    
    print("[HC-SMoE] Number of parameters before merging:", model.num_parameters())
    print(f"[HC-SMoE] Merging into average {num_average_groups} groups...")
    group_st = time.time()
    if merge == "freq" or dominant == "frequency" or mode == "freq":
        grouper.compute_all_usages(model, dataloader_for_merging)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    if dynamic_group:
        grouper.compute_all_usages(model, dataloader_for_merging, mode=hierarchical_stopping_metric)
        print_usage_frequency(grouper._usage_frequency_state_dict)
    

    ### 2. Get dominant experts
    dom_experts = None
    if hybrid_grouping:
        if dominant != "no" or similarity_base != "expert-output":
            raise ValueError(
                "Hybrid grouping requires --dominant=no and --similarity_base=expert-output "
                "to preserve the Mixtral output-grouping baseline."
            )
        if hybrid_alpha == 1.0:
            # The actual alpha=1 grouping must stay on the original HC-SMoE
            # path.  The precomputed result is verification only.
            dom_experts = grouper.cluster_experts(
                model=model, dataloader=dataloader_for_merging, num_groups=num_average_groups
            )
            baseline_labels = grouper.group_state_dict()
            precomputed_labels, _, precomputed_details = grouper.hybrid_grouping_results(
                pairwise_scores, num_average_groups, hybrid_alpha
            )
            mismatches = []
            for layer_name, baseline in baseline_labels.items():
                if not partitions_equal(baseline, precomputed_labels[layer_name]):
                    details = precomputed_details[layer_name]
                    mismatches.append(
                        f"{layer_name}\n"
                        f"baseline labels/groups: {baseline.tolist()} / {canonical_groups(baseline)}\n"
                        f"precomputed alpha=1 labels/groups: {precomputed_labels[layer_name].tolist()} / {canonical_groups(precomputed_labels[layer_name])}\n"
                        f"raw output min/max: {details['normalization']['output_min']} / {details['normalization']['output_max']}\n"
                        f"normalized output distance: {(1.0 - details['output_score']).tolist()}"
                    )
            if mismatches:
                message = "[Hybrid] alpha=1 partition verification failed:\n" + "\n".join(mismatches)
                print(message)
                if verify_alpha_one:
                    raise RuntimeError(message)
            else:
                print("[Hybrid] alpha=1 partition verification passed for all layers")
        else:
            dom_experts = grouper.group_experts_by_hybrid_scores(
                pairwise_scores, num_average_groups, hybrid_alpha
            )
    elif merge == "fsm" or merge == "no":
        pass
    elif dominant == "random":
        grouper.group_experts_randomly(num_groups=args.num_average_groups)
        dom_experts = None
    elif dominant == "frequency":
        grouper.compute_all_similarities(model, dataloader_for_merging)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "routing-score":
        grouper.compute_all_usages(model, dataloader_for_merging, mode="routing-score")
        print_usage_frequency(grouper._usage_frequency_state_dict)
        dom_experts = grouper.group_experts_globally_from_dominant_experts(
            num_average_groups=num_average_groups, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    elif dominant == "no":
        dom_experts = grouper.cluster_experts(model=model, dataloader=dataloader_for_merging, num_groups=num_average_groups)
    else:
        raise ValueError(
            f"Accepted dominant are `random`, `frequency`, `no`, but the input is `{dominant}`")

    if residual_width > 0 and lora_rank > 0:
        raise ValueError("TinySwiGLU residual and LoRA experiments are mutually exclusive")
    lora_enabled = lora_rank > 0
    if lora_enabled and merge != "freq":
        raise ValueError("LoRA weight correction currently requires --merge=freq to match the static baseline exactly.")
    residual_calibration = None
    group_state = grouper.group_state_dict()
    if residual_width > 0 or lora_enabled:
        if merge != "freq":
            raise ValueError("Expert-specific residual PoC currently requires --merge=freq to match the static baseline exactly.")
        calibration_limit = residual_data_limit if residual_width > 0 else lora_data_limit
        calibration_batch_size = residual_batch_size if residual_width > 0 else lora_batch_size
        calibration_label = "Residual" if residual_width > 0 else "LoRA"
        print(f"[{calibration_label}] Collecting up to {calibration_limit} selected C4 tokens per non-singleton expert")
        residual_calibration = collect_residual_calibration(
            model=model,
            dataloader=dataloader_for_merging,
            group_state=group_state,
            residual_data_limit=calibration_limit,
            residual_batch_size=calibration_batch_size,
        )

    
    ### 3. Merge experts
    if merge == "no":
        pass
    elif merge == "freq":
        model = merge_by_groups_with_usage_weighted(
            model, grouper=grouper, merging_layers=list(range(start_layer, model.config.num_hidden_layers))
        )
    else:
        model = merge_by_groups_within_and_across_models(
            mixtral_model=model,
            grouper=grouper,
            dataloader=dataloader_for_merging,
            merge=merge,
            mode=mode,
            partition=partition,
            dominant_alone=False,
            core_experts=dom_experts,
            ingredient=ingredient,
        )
    
    print(f"[HC-SMoE] Merging time: {time.time() - group_st:2f} seconds")
    
    
    if merge != "no":
        ### 4. Print grouping results
        print(f"[HC-SMoE] ========= Grouping results ========= ")
        for name, state in grouper.group_state_dict().items():
            if dom_experts is None:
                print(f"Group {name}: {state.tolist()}")
            else:
                print(f"Group {name}: {state.tolist()} (DOMs are {dom_experts[name]})")

        # Preserve the layer-local expert -> group mapping used to create this
        # merged checkpoint.  It is needed to project a base-Mixtral routing
        # trace into HC-SMoE groups without rerunning the C4 calibration.
        if not output_path:
            raise ValueError("--output_path is required when saving a merged HC-SMoE model")
        os.makedirs(output_path, exist_ok=True)
        grouper.save_group_state_dict(output_path)
        group_metadata = {
            "model_name": model_name,
            "similarity_base": similarity_base,
            "cluster": cluster,
            "linkage": linkage,
            "hierarchical_stopping_metric": hierarchical_stopping_metric,
            "num_average_groups": num_average_groups,
            "start_layer": start_layer,
            "merge": merge,
            "group_mapping_file": "group_state_dict.pt",
        }
        with open(os.path.join(output_path, "group_mapping_metadata.json"), "w") as handle:
            json.dump(group_metadata, handle, indent=2)
        print(f"[HC-SMoE] Saved group mapping: {os.path.join(output_path, 'group_state_dict.pt')}")
        del grouper

    if merge == "unmerge":
        print(f"[HC-SMoE] ======= Grouping of unmerge ======= ")
        for layer_idx in range(start_layer, model.config.num_hidden_layers):
            print(f"--- Layer {layer_idx} ---")
            print(f"expert_to_group: {model.model.layers[layer_idx].block_sparse_moe.expert_to_group}")
            print(f"group_to_expert: {model.model.layers[layer_idx].block_sparse_moe.group_to_expert}")
            print(f"unmerge_matrix: {model.model.layers[layer_idx].block_sparse_moe.unmerge_matrix}")
    
    ### 5. Save the model
    print("[HC-SMoE] Number of parameters after merging:", model.num_parameters())
    if num_average_groups < model.config.num_experts_per_tok:
        model.config.num_experts_per_tok = num_average_groups
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # device_map="auto" may offload inactive modules to meta. Materialize their
    # CPU values before serialization so eval-only reload is lossless.
    static_state_dict = get_state_dict_offloaded_model(model)
    torch.save(expand_shared_expert_state_dict(static_state_dict, group_state), output_path+"/model.pth")
    torch.cuda.empty_cache()

    if residual_width > 0:
        residual_diagnostics = {} if _parse_diagnostic_experts(residual_diagnostic_experts) else None
        residual_metrics = train_residuals(
            model=model,
            group_state=group_state,
            calibration=residual_calibration,
            residual_width=residual_width,
            residual_epochs=residual_epochs,
            residual_lr=residual_lr,
            residual_batch_size=residual_batch_size,
            residual_val_ratio=residual_val_ratio,
            residual_patience=residual_patience,
            seed=seed,
            residual_diagnostic_experts=residual_diagnostic_experts,
            diagnostics=residual_diagnostics,
        )
        residual_config = {
            "residual_width": residual_width,
            "residual_data_limit": residual_data_limit,
            "residual_epochs": residual_epochs,
            "residual_lr": residual_lr,
            "residual_batch_size": residual_batch_size,
            "residual_val_ratio": residual_val_ratio,
            "residual_patience": residual_patience,
            "seed": seed,
            "merge": merge,
            "model_name": model_name,
        }
        save_residual_artifacts(output_path, model, residual_width, residual_metrics, residual_config)
        if residual_diagnostics is not None:
            diagnostic_path = os.path.join(output_path, "residual_training_diagnostics.json")
            with open(diagnostic_path, "w") as handle:
                json.dump(residual_diagnostics, handle, indent=2, sort_keys=True)
            print(f"[ResidualDiag] Saved training diagnostics in {diagnostic_path}")
            save_residual_loss_curves(output_path, residual_diagnostics)
        print(f"[Residual] Saved residual artifacts in {output_path}")

    if lora_enabled:
        lora_metrics = train_lora_experts(
            model=model,
            group_state=group_state,
            calibration=residual_calibration,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_epochs=lora_epochs,
            lora_lr=lora_lr,
            lora_batch_size=lora_batch_size,
            lora_val_ratio=lora_val_ratio,
            lora_patience=lora_patience,
            seed=seed,
        )
        lora_config = {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_data_limit": lora_data_limit,
            "lora_epochs": lora_epochs,
            "lora_lr": lora_lr,
            "lora_batch_size": lora_batch_size,
            "lora_val_ratio": lora_val_ratio,
            "lora_patience": lora_patience,
            "seed": seed,
            "merge": merge,
            "model_name": model_name,
        }
        save_lora_artifacts(output_path, model, lora_rank, lora_alpha, lora_metrics, lora_config)
        save_lora_loss_curves(output_path, lora_metrics)
        print(f"[LoRA] Saved LoRA artifacts in {output_path}")

    ### 6. Evaluation
    run_requested_evaluation(
        args, model, tokenizer, eval_generation, eval_coding, eval_math, eval_limit
    )

if __name__ == "__main__":
    Fire(run_hcsmoe)
