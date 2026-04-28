from collections.abc import Iterable


def recall_at_k(retrieved: list[int], relevant: Iterable[int], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: list[int], relevant: Iterable[int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    top_k = retrieved[:k]
    hits = sum(1 for chunk_id in top_k if chunk_id in relevant_set)
    return hits / k


def f1_at_k(retrieved: list[int], relevant: Iterable[int], k: int) -> float:
    relevant_set = set(relevant)
    p = precision_at_k(retrieved, relevant_set, k)
    r = recall_at_k(retrieved, relevant_set, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(retrieved: list[int], relevant: Iterable[int]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / rank
    return 0.0
