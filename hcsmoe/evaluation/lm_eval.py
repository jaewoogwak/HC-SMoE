# -*- modified by wazenmai -*-
# @Time: 2024/07/03

# -*- coding: utf-8 -*-
# @Author: pingzhili
# @Time: 2024/2/18

import inspect
import os
import json
import numpy as np
from pathlib import Path
from typing import Any, Mapping, Optional

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
# from lm_eval.tasks import initialize_tasks
from lm_eval.utils import make_table

TASK_TO_NUM_FEWSHOT = {
    "arc_challenge": 25,
    "hellaswag": 10,
    "truthfulqa": 0,
    "mmlu": 5,
    "winogrande": 5,
    "gsm8k": 5
}

CODING_GENERATION_TASKS = ("humaneval", "humaneval_plus", "mbpp", "mbpp_plus")
MATH_GENERATION_TASKS = ("gsm8k", "hendrycks_math500")


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def evaluate_fewshot(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task: str,
        num_fewshot: int,
        eval_batch_size: Optional[int] = 4,
        log: Optional[bool] = True,
        output_path: Optional[str] = None,
):
    # initialize_tasks(verbosity="WARNING")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        device_map="auto"
    )
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task,
        num_fewshot=num_fewshot,
        batch_size=eval_batch_size,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
    )

    if log:
        print(make_table(results))
        
        if "groups" in results:
            print(make_table(results, "groups"))
    
    if output_path:
        f = open(output_path, "a")
        print(make_table(results), file=f)
        if "groups" in results:
            print(make_table(results, "groups"), file=f)
        f.close() 
        


    return results


def _generation_tasks(eval_coding: bool, eval_math: bool) -> list[str]:
    tasks = []
    if eval_coding:
        tasks.extend(CODING_GENERATION_TASKS)
    if eval_math:
        tasks.extend(MATH_GENERATION_TASKS)
    if not tasks:
        raise ValueError("Generation evaluation requires --eval_generation, --eval_coding, or --eval_math.")
    return tasks


def _validate_generation_tasks(tasks: list[str]) -> None:
    available = set(TaskManager().all_tasks)
    missing = [task for task in tasks if task not in available]
    if missing:
        raise RuntimeError(
            "Generation suite requires a recent lm-evaluation-harness with these registered tasks: "
            f"{', '.join(missing)}. Upgrade lm-eval before running this suite."
        )


def validate_generation_tasks(eval_coding: bool, eval_math: bool) -> list[str]:
    """Fail fast when the installed harness lacks a requested generation task."""
    tasks = _generation_tasks(eval_coding, eval_math)
    _validate_generation_tasks(tasks)
    return tasks


def _primary_metric(task: str, task_results: Mapping[str, Any]) -> dict[str, Any]:
    """Find a task's primary pass@1/exact-match metric without positional access."""
    numeric = {
        key: float(value)
        for key, value in task_results.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and "stderr" not in key.lower()
    }
    if not numeric:
        raise RuntimeError(f"No numeric metrics returned for generation task {task!r}: {dict(task_results)}")
    if task in CODING_GENERATION_TASKS:
        preferred = ("pass@1", "pass_at_1", "pass_at_k")
    else:
        preferred = ("exact_match", "exact-match", "exact match", "acc")
    for needle in preferred:
        for key, value in numeric.items():
            if needle in key.lower():
                return {"metric": key, "score": value}
    if len(numeric) == 1:
        key, value = next(iter(numeric.items()))
        return {"metric": key, "score": value}
    raise RuntimeError(
        f"Could not identify a primary metric for {task!r}. Available numeric metrics: {sorted(numeric)}"
    )


def _generation_summary(raw_results: Mapping[str, Any], tasks: list[str]) -> dict[str, Any]:
    task_results = raw_results.get("results", {})
    scores = {}
    for task in tasks:
        if task not in task_results:
            raise RuntimeError(f"lm-eval did not return results for requested task {task!r}.")
        scores[task] = _primary_metric(task, task_results[task])

    summary: dict[str, Any] = {"coding": {}, "math": {}}
    coding_scores = []
    for task in CODING_GENERATION_TASKS:
        if task in scores:
            summary["coding"][task] = scores[task]
            coding_scores.append(scores[task]["score"])
    if coding_scores:
        summary["coding"]["evalplus_avg"] = sum(coding_scores) / len(coding_scores)

    math_scores = []
    for task in MATH_GENERATION_TASKS:
        if task in scores:
            summary["math"][task] = scores[task]
            math_scores.append(scores[task]["score"])
    if math_scores:
        summary["math"]["math_avg"] = sum(math_scores) / len(math_scores)
    return summary


def evaluate_generation(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        eval_coding: bool,
        eval_math: bool,
        eval_batch_size: Optional[int] = 4,
        output_path: Optional[str] = None,
        eval_limit: Optional[int] = None,
):
    """Run latest lm-eval generation tasks without changing MC few-shot behavior."""
    tasks = validate_generation_tasks(eval_coding, eval_math)
    evaluation_kwargs = {
        "model": HFLM(pretrained=model, tokenizer=tokenizer, batch_size=eval_batch_size, device_map="auto"),
        "tasks": tasks,
        # None preserves each task YAML's num_fewshot setting (not the MC CLI default).
        "num_fewshot": None,
        "batch_size": eval_batch_size,
        "limit": eval_limit,
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
    }
    if eval_coding:
        if "confirm_run_unsafe_code" not in inspect.signature(evaluator.simple_evaluate).parameters:
            raise RuntimeError(
                "Installed lm-evaluation-harness does not support confirm_run_unsafe_code. "
                "Upgrade to the latest lm-evaluation-harness before running code generation tasks."
            )
        # Retain compatibility with harness code paths that also inspect this environment variable.
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        evaluation_kwargs["confirm_run_unsafe_code"] = True
    raw_results = evaluator.simple_evaluate(**evaluation_kwargs)
    summary = _generation_summary(raw_results, tasks)
    payload = {"summary": summary, "raw_lm_eval_results": raw_results}
    print("[Generation] Summary:")
    print(json.dumps(summary, indent=2, default=_handle_non_serializable))
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        destination = os.path.join(output_path, "generation_results.json")
        with open(destination, "w") as handle:
            json.dump(payload, handle, indent=2, default=_handle_non_serializable)
        print(f"[Generation] Saved results: {destination}")
    return payload
