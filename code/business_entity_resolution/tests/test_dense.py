import numpy as np
import pytest
import torch

from er.retrieval.dense import mean_pooling, search_dense


def test_mean_pooling():
    # 2 sentences: one len 3, one len 2 (padded with 0)
    token_embeds = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[2.0, 2.0], [4.0, 4.0], [0.0, 0.0]],
    ])
    attention_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0],
    ])
    pooled = mean_pooling(token_embeds, attention_mask)
    assert pooled.shape == (2, 2)
    # Check L2 normalization
    norms = np.linalg.norm(pooled, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)


def test_search_dense():
    # 2 queries, 3 pool items
    pool_embeds = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7071, 0.7071],
    ], dtype=np.float32)
    query_embeds = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    pool_ids = ["P1", "P2", "P3"]
    query_ids = ["Q1", "Q2"]

    cands, scores = search_dense(query_embeds, pool_embeds, pool_ids, query_ids, top_k=2)
    assert cands["Q1"][0] == "P1"
    assert cands["Q2"][0] == "P2"
    assert len(cands["Q1"]) == 2
