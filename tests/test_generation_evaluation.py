import json
from types import SimpleNamespace

import pytest

from hcsmoe.evaluation import lm_eval


def _fake_results():
    return {
        "results": {
            "humaneval": {"pass@1,create_test": 0.10},
            "humaneval_plus": {"pass@1,create_test": 0.20},
            "mbpp": {"pass@1": 0.30},
            "mbpp_plus": {"pass@1": 0.40},
            "gsm8k": {"exact_match,strict-match": 0.50},
            "hendrycks_math500": {"exact_match,flexible-extract": 0.60},
        }
    }


def test_generation_summary_uses_primary_metrics_and_averages():
    tasks = list(lm_eval.CODING_GENERATION_TASKS + lm_eval.MATH_GENERATION_TASKS)
    summary = lm_eval._generation_summary(_fake_results(), tasks)

    assert summary["coding"]["humaneval"]["score"] == 0.10
    assert summary["coding"]["evalplus_avg"] == pytest.approx(0.25)
    assert summary["math"]["gsm8k"]["score"] == 0.50
    assert summary["math"]["math_avg"] == pytest.approx(0.55)


def test_generation_evaluation_smoke_uses_task_defaults_and_unsafe_confirmation(monkeypatch, tmp_path):
    task_names = list(lm_eval.CODING_GENERATION_TASKS + lm_eval.MATH_GENERATION_TASKS)
    monkeypatch.setattr(
        lm_eval,
        "TaskManager",
        lambda: SimpleNamespace(all_tasks=task_names),
    )
    monkeypatch.setattr(lm_eval, "HFLM", lambda **kwargs: "fake-lm")
    captured = {}

    def fake_simple_evaluate(*, confirm_run_unsafe_code=None, **kwargs):
        captured.update(kwargs)
        captured["confirm_run_unsafe_code"] = confirm_run_unsafe_code
        return _fake_results()

    monkeypatch.setattr(lm_eval.evaluator, "simple_evaluate", fake_simple_evaluate)

    payload = lm_eval.evaluate_generation(
        model=object(),
        tokenizer=object(),
        eval_coding=True,
        eval_math=True,
        eval_limit=5,
        output_path=str(tmp_path),
    )

    assert captured["tasks"] == task_names
    assert captured["num_fewshot"] is None
    assert captured["limit"] == 5
    assert captured["confirm_run_unsafe_code"] is True
    with open(tmp_path / "generation_results.json") as handle:
        saved = json.load(handle)
    assert saved["summary"] == payload["summary"]


def test_generation_registry_error_lists_missing_tasks(monkeypatch):
    monkeypatch.setattr(lm_eval, "TaskManager", lambda: SimpleNamespace(all_tasks=[]))
    with pytest.raises(RuntimeError, match="humaneval"):
        lm_eval._validate_generation_tasks(["humaneval"])
