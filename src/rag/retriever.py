import faiss
import jieba
import numpy as np
from langchain_community.retrievers import BM25Retriever


def jieba_tokenize(text: str) -> list[str]:
    return [token for token in jieba.cut_for_search(text) if token.strip()]


class BM25Index:
    """LangChain BM25 retriever wrapped to return chunk-id rankings."""

    def __init__(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("BM25Index requires at least one text.")
        self._retriever = BM25Retriever.from_texts(
            texts,
            metadatas=[{"chunk_id": i} for i in range(len(texts))],
            preprocess_func=jieba_tokenize,
        )

    def search(self, query: str, k: int) -> list[int]:
        self._retriever.k = max(k, 1)
        docs = self._retriever.invoke(query)
        return [doc.metadata["chunk_id"] for doc in docs]


class DenseIndex:
    """In-memory FAISS cosine-similarity index over chunk embeddings."""

    def __init__(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("DenseIndex requires a non-empty 2-D embedding matrix.")
        prepared = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(prepared)
        self._index = faiss.IndexFlatIP(prepared.shape[1])
        self._index.add(prepared)  # type: ignore[call-arg]
        self._size = prepared.shape[0]

    def search(self, query_embedding: np.ndarray, k: int) -> list[int]:
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(query)
        _, indices = self._index.search(query, min(max(k, 1), self._size))  # type: ignore[call-arg]
        return [int(i) for i in indices[0] if i >= 0]
