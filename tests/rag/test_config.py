import pytest

from rag.config import RAGConfig


def test_retrieval_knobs_default_values(monkeypatch):
    for name in (
        "RAG_BM25_TOP_N",
        "RAG_DENSE_TOP_N",
        "RAG_RRF_K",
        "RAG_RERANK_POOL_SIZE",
        "RAG_FINAL_K",
    ):
        monkeypatch.delenv(name, raising=False)

    config = RAGConfig(_env_file=None)

    assert config.bm25_top_n == 50
    assert config.dense_top_n == 50
    assert config.rrf_k == 60
    assert config.rerank_pool_size == 30
    assert config.final_k == 5


def test_retrieval_knobs_read_from_env(monkeypatch):
    monkeypatch.setenv("RAG_BM25_TOP_N", "80")
    monkeypatch.setenv("RAG_DENSE_TOP_N", "70")
    monkeypatch.setenv("RAG_RRF_K", "40")
    monkeypatch.setenv("RAG_RERANK_POOL_SIZE", "25")
    monkeypatch.setenv("RAG_FINAL_K", "8")

    config = RAGConfig(_env_file=None)

    assert config.bm25_top_n == 80
    assert config.dense_top_n == 70
    assert config.rrf_k == 40
    assert config.rerank_pool_size == 25
    assert config.final_k == 8


def test_retrieval_knobs_validate_positive(monkeypatch):
    monkeypatch.setenv("RAG_FINAL_K", "0")

    with pytest.raises(ValueError):
        RAGConfig(_env_file=None)
