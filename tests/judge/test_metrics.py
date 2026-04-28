import math

from judge.metrics import f1_at_k, mrr, precision_at_k, recall_at_k


def test_recall_at_k_counts_unique_relevant_hits():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=4) == 2 / 3


def test_recall_at_k_truncates_to_k():
    assert recall_at_k([1, 2, 3, 4], {3, 4}, k=2) == 0.0
    assert recall_at_k([1, 2, 3, 4], {3, 4}, k=4) == 1.0


def test_recall_at_k_empty_relevant_returns_zero():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_precision_at_k_uses_k_as_denominator():
    assert precision_at_k([1, 2, 3, 4], {2, 4}, k=4) == 0.5
    assert precision_at_k([1, 2, 3, 4], {2, 4}, k=2) == 0.5


def test_precision_at_k_zero_for_nonpositive_k():
    assert precision_at_k([1, 2], {1}, k=0) == 0.0


def test_f1_at_k_combines_precision_and_recall():
    # retrieved top-2 = {1,2}; relevant = {1,3} → p=0.5, r=0.5 → f1=0.5
    assert math.isclose(f1_at_k([1, 2, 3, 4], {1, 3}, k=2), 0.5)


def test_f1_at_k_zero_when_no_overlap():
    assert f1_at_k([1, 2], {3, 4}, k=2) == 0.0


def test_mrr_returns_reciprocal_of_first_hit_rank():
    assert mrr([5, 6, 7, 8], {7, 8}) == 1 / 3


def test_mrr_zero_when_no_relevant_retrieved():
    assert mrr([1, 2, 3], {99}) == 0.0


def test_mrr_zero_for_empty_relevant():
    assert mrr([1, 2, 3], set()) == 0.0
