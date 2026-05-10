import asyncio
import logging
import os
from typing import Any

import numpy as np
from langfuse import Langfuse
from openai import AsyncOpenAI

from rag.chunker import CorpusChunker
from rag.fusion import rrf_merge
from rag.reranker import Reranker
from rag.retriever import BM25Index, DenseIndex
from rag.retry import with_openai_retry

logger = logging.getLogger(__name__)

EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))
DEFAULT_MAX_CONCURRENCY = int(os.getenv("RAG_MAX_CONCURRENCY", "64"))


class RAGPipeline:
    """Hybrid BM25 + dense + reranker pipeline backed by Langfuse-managed assets."""

    def __init__(
        self,
        openai: AsyncOpenAI,
        langfuse: Langfuse,
        *,
        embedding_model: str = "bge-m3",
        rerank_model: str = "BGE-Reranker-V2-M3",
        generator_prompt_name: str = "rag-generator-instruction",
    ) -> None:
        self._openai = openai
        self._langfuse = langfuse
        self._embedding_model = embedding_model
        self._rerank_model = rerank_model
        self._generator_prompt_name = generator_prompt_name
        self._reranker = Reranker(
            base_url=str(openai.base_url),
            api_key=openai.api_key,
            model=rerank_model,
        )
        self._chunker: CorpusChunker | None = None
        self._dense: DenseIndex | None = None
        self._bm25: BM25Index | None = None

    async def build(
        self,
        corpus_dataset_names: list[str],
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        max_concurrency: int | None = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        """Aggregate corpus datasets, chunk, and build BM25 + dense indexes."""
        chunker = CorpusChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for name in corpus_dataset_names:
            dataset = self._langfuse.get_dataset(name)
            for item in dataset.items:
                metadata = item.metadata or {}
                payload = item.input or {}
                chapter = metadata.get("chapter")
                content = payload.get("content")
                if not chapter or not content:
                    logger.warning(
                        "Skipping dataset item %s in '%s' — missing chapter or content",
                        getattr(item, "id", "?"),
                        name,
                    )
                    continue
                chunker.add_chapter(chapter, content)

        if not chunker.chunks:
            raise ValueError(
                "No chunks produced — corpus datasets were empty or malformed."
            )

        logger.info(
            "Embedding %d chunks across %d chapter(s)",
            len(chunker.chunks),
            len(chunker.chapters),
        )
        semaphore = (
            asyncio.BoundedSemaphore(max_concurrency)
            if max_concurrency is not None
            else None
        )
        embeddings = await self._embed_chunks(
            [chunk.text for chunk in chunker.chunks],
            semaphore=semaphore,
        )

        self._chunker = chunker
        self._dense = DenseIndex(embeddings)
        self._bm25 = BM25Index([chunk.text for chunk in chunker.chunks])

    def resolve_chunks(self, chapter: str, start: int, end: int) -> list[int]:
        chunker = self._require_built()
        return chunker.resolve_chunks(chapter, start, end)

    async def generate(
        self,
        *,
        query: str,
        model: str,
        bm25_top_n: int = 50,
        dense_top_n: int = 50,
        rrf_k: int = 60,
        rerank_pool_size: int = 30,
        final_k: int = 5,
    ) -> dict[str, Any]:
        """Run hybrid retrieval, rerank, and generate an answer for the query."""
        chunker = self._require_built()
        assert self._dense is not None and self._bm25 is not None

        with self._langfuse.start_as_current_observation(
            name="rag-retrieve", as_type="retriever", input={"query": query}
        ) as retrieve_span:
            query_embedding = await self._embed_query(query)
            bm25_ranking = self._bm25.search(query, k=bm25_top_n)
            dense_ranking = self._dense.search(query_embedding, k=dense_top_n)
            fused = rrf_merge([bm25_ranking, dense_ranking], k=rrf_k)
            pool_ids = [chunk_id for chunk_id, _ in fused[:rerank_pool_size]]
            pool_texts = [chunker.chunks[cid].text for cid in pool_ids]

            reranked = await self._reranker.rerank(
                query=query,
                documents=pool_texts,
                top_n=min(final_k, len(pool_texts)),
            )
            final_chunk_ids = [pool_ids[idx] for idx, _ in reranked]

            retrieve_span.update(
                output={
                    "bm25_top": bm25_ranking[:5],
                    "dense_top": dense_ranking[:5],
                    "reference_chunks": final_chunk_ids,
                }
            )

        prompt = self._langfuse.get_prompt(self._generator_prompt_name)
        context = "\n\n".join(chunker.chunks[cid].text for cid in final_chunk_ids)
        compiled = prompt.compile(context=context, query=query)

        with self._langfuse.start_as_current_observation(
            name="rag-generate",
            as_type="generation",
            model=model,
            input={"context": context, "query": query},
        ) as generation_span:
            self._langfuse.update_current_generation(prompt=prompt)
            response = await self._chat_complete(
                model=model, system=compiled, query=query
            )
            output_text = response.choices[0].message.content or ""
            usage = response.usage
            generation_span.update(
                output=output_text,
                usage_details={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                }
                if usage is not None
                else None,
            )

        return {
            "output_text": output_text,
            "reference_chunks": final_chunk_ids,
        }

    async def _embed_query(self, text: str) -> np.ndarray:
        response = await self._embed_call(text)
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    async def _embed_chunks(
        self, texts: list[str], semaphore: asyncio.Semaphore | None = None
    ) -> np.ndarray:
        batches = [
            texts[offset : offset + EMBEDDING_BATCH_SIZE]
            for offset in range(0, len(texts), EMBEDDING_BATCH_SIZE)
        ]
        results = await asyncio.gather(
            *(self._embed_batch(batch, semaphore=semaphore) for batch in batches)
        )
        vectors = [
            embedding for batch_vectors in results for embedding in batch_vectors
        ]
        return np.asarray(vectors, dtype=np.float32)

    async def _embed_batch(
        self, batch: list[str], semaphore: asyncio.Semaphore | None = None
    ) -> list[list[float]]:
        response = await self._embed_call(batch, semaphore=semaphore)
        return [item.embedding for item in response.data]

    @with_openai_retry()
    async def _embed_call(self, payload, semaphore: asyncio.Semaphore | None = None):
        if semaphore:
            async with semaphore:
                return await self._openai.embeddings.create(
                    model=self._embedding_model, input=payload
                )
        return await self._openai.embeddings.create(
            model=self._embedding_model, input=payload
        )

    @with_openai_retry()
    async def _chat_complete(self, *, model: str, system: str, query: str):
        return await self._openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
        )

    def _require_built(self) -> CorpusChunker:
        if self._chunker is None or self._dense is None or self._bm25 is None:
            raise RuntimeError(
                "RAGPipeline.build(...) must be called before generate/resolve."
            )
        return self._chunker
