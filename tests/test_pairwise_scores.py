import torch
from torch import nn

from hcsmoe.merging.pairwise_scores import (
    accumulate_corouting,
    build_output_score_matrices,
    compute_output_fingerprint,
    validate_pairwise_scores,
)


def _assert_routing_contract(routing_count, usage_count):
    assert torch.equal(routing_count, routing_count.T)
    assert torch.equal(routing_count.diag(), torch.zeros_like(routing_count.diag()))
    assert torch.all(routing_count <= torch.minimum(usage_count[:, None], usage_count[None, :]))


def test_corouting_counts_for_mixtral_top2_and_qwen_top4():
    mixtral_logits = torch.tensor([
        [9.0, 8.0, 1.0, 0.0, -1.0],
        [1.0, 9.0, 8.0, 0.0, -1.0],
        [8.0, 1.0, 0.0, 9.0, -1.0],
    ])
    mixtral_count = torch.zeros((5, 5), dtype=torch.int64)
    mixtral_usage = torch.zeros(5, dtype=torch.int64)
    accumulate_corouting(mixtral_count, mixtral_usage, mixtral_logits, top_k=2)
    _assert_routing_contract(mixtral_count, mixtral_usage)
    assert torch.triu(mixtral_count, diagonal=1).sum().item() == mixtral_logits.shape[0]

    qwen_logits = torch.tensor([
        [9.0, 8.0, 7.0, 6.0, 0.0, -1.0],
        [1.0, 9.0, 8.0, 7.0, 6.0, 0.0],
    ])
    qwen_count = torch.zeros((6, 6), dtype=torch.int64)
    qwen_usage = torch.zeros(6, dtype=torch.int64)
    accumulate_corouting(qwen_count, qwen_usage, qwen_logits, top_k=4)
    _assert_routing_contract(qwen_count, qwen_usage)
    assert torch.triu(qwen_count, diagonal=1).sum().item() == qwen_logits.shape[0] * 6


def test_chunked_output_fingerprint_and_score_matrices():
    torch.manual_seed(0)
    inputs = torch.randn(11, 3)
    experts = [nn.Linear(3, 3, bias=False), nn.Linear(3, 3, bias=False)]
    fingerprints = torch.stack([
        compute_output_fingerprint(expert, inputs, chunk_size=4) for expert in experts
    ])
    direct = torch.stack([expert(inputs).float().mean(dim=0) for expert in experts])
    assert fingerprints.shape == (2, 3)
    assert torch.allclose(fingerprints, direct, atol=1e-6)

    scores = build_output_score_matrices(fingerprints)
    assert scores["output_distance"].shape == (2, 2)
    assert scores["output_cosine"].shape == (2, 2)
    assert torch.allclose(scores["output_distance"], scores["output_distance"].T)
    assert torch.allclose(scores["output_cosine"], scores["output_cosine"].T)
    assert torch.allclose(scores["output_distance"].diag(), torch.zeros(2))
    assert torch.allclose(scores["output_cosine"].diag(), torch.ones(2))
    assert torch.isfinite(scores["output_distance"]).all()
    assert torch.isfinite(scores["output_cosine"]).all()

    routing_count = torch.zeros((2, 2), dtype=torch.int64)
    usage_count = torch.zeros(2, dtype=torch.int64)
    accumulate_corouting(routing_count, usage_count, torch.tensor([[1.0, 0.0]]), top_k=2)
    scores.update({
        "routing_count": routing_count,
        "routing_rate": routing_count.float(),
        "usage_count": usage_count,
        "num_tokens": 1,
    })
    validate_pairwise_scores({"dummy.layer": scores})
