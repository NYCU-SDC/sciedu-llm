import math

from rag.fusion import rrf_merge


def test_single_ranking_is_preserved():
    fused = rrf_merge([[10, 20, 30]], k=60)
    assert [doc_id for doc_id, _ in fused] == [10, 20, 30]


def test_doc_in_both_rankings_outranks_doc_in_one():
    fused = dict(rrf_merge([[1, 2, 3], [3, 4, 5]], k=60))
    # 3 appears at rank 2 in list A and rank 0 in list B; should beat anything appearing only once.
    assert fused[3] > fused[1]
    assert fused[3] > fused[5]


def test_score_formula_matches_rrf_definition():
    fused = dict(rrf_merge([[1, 2], [2, 1]], k=60))
    expected_for_each = (1 / (60 + 1)) + (1 / (60 + 2))
    assert math.isclose(fused[1], expected_for_each)
    assert math.isclose(fused[2], expected_for_each)


def test_smaller_k_amplifies_top_ranks():
    fused_small_k = dict(rrf_merge([[1, 2], [2, 1]], k=1))
    fused_large_k = dict(rrf_merge([[1, 2], [2, 1]], k=100))

    # Score gap between rank-0 contribution and rank-1 contribution shrinks as k grows.
    gap_small = (1 / (1 + 1)) - (1 / (1 + 2))
    gap_large = (1 / (100 + 1)) - (1 / (100 + 2))
    assert gap_small > gap_large
    # Both ties hold at equal totals.
    assert math.isclose(fused_small_k[1], fused_small_k[2])
    assert math.isclose(fused_large_k[1], fused_large_k[2])


def test_empty_rankings_returns_empty():
    assert rrf_merge([], k=60) == []
    assert rrf_merge([[], []], k=60) == []
