from collections import defaultdict


def rrf_merge(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Each input ranking is an ordered list of doc ids (best first). Returns the
    fused ranking as `(doc_id, score)` pairs sorted by score descending.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
