import json
from types import SimpleNamespace

import torch

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
    hflm_kwargs = {}

    def fake_hflm(**kwargs):
        hflm_kwargs.update(kwargs)
        return "fake-lm"

    monkeypatch.setattr(lm_eval, "HFLM", fake_hflm)
    captured = {}

    def fake_simple_evaluate(*, confirm_run_unsafe_code=None, **kwargs):
        captured.update(kwargs)
        captured["confirm_run_unsafe_code"] = confirm_run_unsafe_code
        return _fake_results()

    monkeypatch.setattr(lm_eval.evaluator, "simple_evaluate", fake_simple_evaluate)

    model = SimpleNamespace()
    model.generate = lambda input_ids, **_: input_ids
    payload = lm_eval.evaluate_generation(
        model=model,
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
    assert "device_map" not in hflm_kwargs
    assert "generation_runtime" in payload
    with open(tmp_path / "generation_results.json") as handle:
        saved = json.load(handle)
    assert saved["summary"] == payload["summary"]


def test_generation_registry_error_lists_missing_tasks(monkeypatch):
    monkeypatch.setattr(lm_eval, "TaskManager", lambda: SimpleNamespace(all_tasks=[]))
    with pytest.raises(RuntimeError, match="humaneval"):
        lm_eval._validate_generation_tasks(["humaneval"])


def test_humaneval_one_sample_generation_smoke_tracks_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(lm_eval, "CODING_GENERATION_TASKS", ("humaneval",))
    monkeypatch.setattr(lm_eval, "TaskManager", lambda: SimpleNamespace(all_tasks=["humaneval"]))
    monkeypatch.setattr(
        lm_eval,
        "HFLM",
        lambda **kwargs: SimpleNamespace(model=kwargs["pretrained"]),
    )

    def fake_simple_evaluate(*, confirm_run_unsafe_code=None, **kwargs):
        assert kwargs["tasks"] == ["humaneval"]
        assert kwargs["limit"] == 1
        kwargs["model"].model.generate(torch.ones(1, 3, dtype=torch.long))
        return {"results": {"humaneval": {"pass@1": 0.0}}}

    monkeypatch.setattr(lm_eval.evaluator, "simple_evaluate", fake_simple_evaluate)
    model = SimpleNamespace()
    model.generate = lambda input_ids, **_: torch.cat((input_ids, input_ids[:, :1]), dim=-1)
    payload = lm_eval.evaluate_generation(model, object(), True, False, eval_limit=1, output_path=str(tmp_path))
    assert payload["generation_runtime"]["generated_tokens"] == 1
    assert payload["generation_runtime"]["tokens_per_second"] > 0
