import asyncio
import logging
import time
from typing import Any

import numpy as np
from langfuse import Langfuse
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from rag.chunker import CorpusChunker
from rag.config import RAGConfig, get_rag_config
from rag.fusion import rrf_merge
from rag.reranker import Reranker
from rag.retriever import BM25Index, DenseIndex
from rag.retry import with_openai_retry

logger = logging.getLogger(__name__)

# Sentinel for `build(..., max_concurrency=...)` so callers can still pass `None`
# to mean "unlimited" while omitting the kwarg pulls the configured default.
_UNSET: Any = object()

#: How often the embedding pass reports how far along it is. A real corpus is
#: hundreds of batches, so one line each would bury everything else in the log;
#: a line every few seconds is enough to tell a slow build from a stuck one.
_PROGRESS_INTERVAL_SECONDS = 10.0


def _format_seconds(value: float) -> str:
    total = max(0, round(value))
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m {total % 60:02d}s"


class _EmbedProgress:
    """Throttled progress logging for the embedding pass of a build.

    Embedding is one long `asyncio.gather` over batches, so how many of them have
    come back is the only progress there is to report — there is no byte count,
    no server-side percentage. The remaining time is a straight extrapolation
    from the rate so far, and is logged as such rather than as a promise.
    """

    def __init__(self, *, batches: int, chunks: int) -> None:
        self._batches = batches
        self._chunks = chunks
        self._done = 0
        self._embedded = 0
        self._started = time.monotonic()
        self._last_logged = self._started

    def batch_done(self, size: int) -> None:
        self._done += 1
        self._embedded += size
        now = time.monotonic()
        final = self._done >= self._batches
        if not final and now - self._last_logged < _PROGRESS_INTERVAL_SECONDS:
            return
        self._last_logged = now
        elapsed = now - self._started
        share = self._done / self._batches if self._batches else 1.0
        left = (elapsed / share - elapsed) if share > 0 else 0.0
        logger.info(
            "Embedding %d/%d batches (%d/%d chunks, %.0f%%) — %s elapsed%s",
            self._done,
            self._batches,
            self._embedded,
            self._chunks,
            share * 100,
            _format_seconds(elapsed),
            "" if final else f", ~{_format_seconds(left)} left at this rate",
        )


class RAGPipeline:
    """Hybrid BM25 + dense + reranker pipeline backed by Langfuse-managed assets."""

    #: Config fields that are baked into the indexes at :meth:`build` time, as
    #: opposed to the retrieval knobs, which are read fresh on every query.
    #: Changing one of these means nothing until the next build — and if that
    #: build never lands, the value describes an index that does not exist, which
    #: is why `app.rag_builds` puts these back when a build is abandoned.
    BUILD_TIME_FIELDS = (
        "embedding_model",
        "embedding_batch_size",
        "chunk_size",
        "chunk_overlap",
    )

    def __init__(
        self,
        openai: AsyncOpenAI,
        langfuse: Langfuse,
        *,
        embedding_model: str | None = None,
        rerank_model: str | None = None,
        generator_system_prompt_name: str | None = None,
        generator_user_prompt_name: str | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self._config = config or get_rag_config()
        self._openai = openai
        self._langfuse = langfuse
        self._embedding_model = embedding_model or self._config.embedding_model
        self._rerank_model = rerank_model or self._config.rerank_model
        self._generator_system_prompt_name = (
            generator_system_prompt_name or self._config.generator_system_prompt_name
        )
        self._generator_user_prompt_name = (
            generator_user_prompt_name or self._config.generator_user_prompt_name
        )
        self._embedding_batch_size = self._config.embedding_batch_size
        self._reranker = Reranker(
            base_url=str(openai.base_url),
            api_key=openai.api_key,
            model=self._rerank_model,
        )
        self._chunker: CorpusChunker | None = None
        self._dense: DenseIndex | None = None
        self._bm25: BM25Index | None = None
        # Corpus datasets the current indexes were built from; retained so the
        # admin API can rebuild without re-plumbing the dataset names.
        self._corpus_dataset_names: list[str] = []

    async def build(
        self,
        corpus_dataset_names: list[str],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_concurrency: int | None = _UNSET,
    ) -> None:
        """Aggregate corpus datasets, chunk, and build BM25 + dense indexes."""
        if chunk_size is None:
            chunk_size = self._config.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self._config.chunk_overlap
        if max_concurrency is _UNSET:
            max_concurrency = self._config.max_concurrency

        started = time.monotonic()
        logger.info(
            "Index build starting | datasets=%s chunk_size=%d chunk_overlap=%d "
            "embedding_model=%s batch_size=%d max_concurrency=%s",
            ", ".join(corpus_dataset_names) or "(none)",
            chunk_size,
            chunk_overlap,
            self._embedding_model,
            self._embedding_batch_size,
            max_concurrency if max_concurrency is not None else "unlimited",
        )

        chunker = CorpusChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        total_datasets = len(corpus_dataset_names)
        for position, name in enumerate(corpus_dataset_names, start=1):
            # The Langfuse SDK call is blocking; off-loop so a build (minutes of
            # them, for a real corpus) neither stalls the server nor sits at an
            # uncancellable point while an operator is asking it to stop.
            dataset = await asyncio.to_thread(self._langfuse.get_dataset, name)
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

            logger.info(
                "Collected dataset %d/%d '%s' — %d chunk(s) so far, %s elapsed",
                position,
                total_datasets,
                name,
                len(chunker.chunks),
                _format_seconds(time.monotonic() - started),
            )

        if not chunker.chunks:
            raise ValueError(
                "No chunks produced — corpus datasets were empty or malformed."
            )

        logger.info(
            "Embedding %d chunks across %d chapter(s)",
            len(chunker.chunks),
            len(chunker.chapters),
        )

        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer or None")

        semaphore = (
            asyncio.BoundedSemaphore(max_concurrency)
            if max_concurrency is not None
            else None
        )
        embeddings = await self._embed_chunks(
            [chunk.text for chunk in chunker.chunks],
            semaphore=semaphore,
        )

        logger.info("Indexing %d embedded chunk(s)", len(chunker.chunks))
        self._chunker = chunker
        self._dense = DenseIndex(embeddings)
        self._bm25 = BM25Index([chunk.text for chunk in chunker.chunks])
        self._corpus_dataset_names = list(corpus_dataset_names)
        logger.info(
            "Index build finished | %d chunk(s), %d chapter(s) from %d dataset(s) in %s",
            len(chunker.chunks),
            len(chunker.chapters),
            total_datasets,
            _format_seconds(time.monotonic() - started),
        )

    @property
    def is_built(self) -> bool:
        return (
            self._chunker is not None
            and self._dense is not None
            and self._bm25 is not None
        )

    @property
    def corpus_dataset_names(self) -> list[str]:
        """The corpus the current indexes were built from.

        Re-indexing the same corpus is `build(pipeline.corpus_dataset_names)`.
        There is deliberately no `rebuild()` shortcut: builds against the live
        pipeline go through `app.rag_builds.RagBuildManager`, which is what makes
        them cancellable and stops two from running at once, and a second
        entrypoint would quietly bypass it.
        """
        return list(self._corpus_dataset_names)

    def config_snapshot(self) -> dict[str, Any]:
        """Return the currently effective config values (env defaults + overrides)."""
        config = self._config
        return {
            "embedding_model": self._embedding_model,
            "rerank_model": self._rerank_model,
            "embedding_batch_size": self._embedding_batch_size,
            "max_concurrency": config.max_concurrency,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "generator_system_prompt_name": self._generator_system_prompt_name,
            "generator_user_prompt_name": self._generator_user_prompt_name,
            "bm25_top_n": config.bm25_top_n,
            "dense_top_n": config.dense_top_n,
            "rrf_k": config.rrf_k,
            "rerank_pool_size": config.rerank_pool_size,
            "final_k": config.final_k,
        }

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply runtime config overrides in place.

        Sets each field on the underlying ``RAGConfig`` (re-validated via
        ``validate_assignment``) and re-syncs the derived state that ``__init__``
        caches (the ``Reranker``, embedding/model/prompt attributes). Build-time
        changes (chunk sizes, embedding model/batch) take effect on the next
        :meth:`build`.
        """
        for key, value in overrides.items():
            setattr(self._config, key, value)

        if "embedding_model" in overrides:
            self._embedding_model = overrides["embedding_model"]
        if "embedding_batch_size" in overrides:
            self._embedding_batch_size = overrides["embedding_batch_size"]
        if "generator_system_prompt_name" in overrides:
            self._generator_system_prompt_name = overrides[
                "generator_system_prompt_name"
            ]
        if "generator_user_prompt_name" in overrides:
            self._generator_user_prompt_name = overrides["generator_user_prompt_name"]
        if "rerank_model" in overrides:
            self._rerank_model = overrides["rerank_model"]
            self._reranker = Reranker(
                base_url=str(self._openai.base_url),
                api_key=self._openai.api_key,
                model=self._rerank_model,
            )

    def resolve_chunks(self, chapter: str, start: int, end: int) -> list[int]:
        chunker = self._require_built()
        return chunker.resolve_chunks(chapter, start, end)

    async def retrieve(
        self,
        *,
        query: str,
        bm25_top_n: int | None = None,
        dense_top_n: int | None = None,
        rrf_k: int | None = None,
        rerank_pool_size: int | None = None,
        final_k: int | None = None,
    ) -> dict[str, Any]:
        """Run hybrid retrieval + rerank and return the top context for the query.

        The knob arguments default to ``None`` and are resolved from the (possibly
        runtime-overridden) ``RAGConfig`` when omitted. Returns a dict with the
        joined ``context`` string and the ordered ``reference_chunks`` (chunk ids).
        Emits a ``rag-retrieve`` retriever span.
        """
        if bm25_top_n is None:
            bm25_top_n = self._config.bm25_top_n
        if dense_top_n is None:
            dense_top_n = self._config.dense_top_n
        if rrf_k is None:
            rrf_k = self._config.rrf_k
        if rerank_pool_size is None:
            rerank_pool_size = self._config.rerank_pool_size
        if final_k is None:
            final_k = self._config.final_k

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
            final_chunks = [chunker.chunks[cid] for cid in final_chunk_ids]

            retrieve_span.update(
                output=[
                    {
                        "id": chunk.id,
                        "chapter": chunk.chapter,
                        "start": chunk.start,
                        "end": chunk.end,
                        "text": chunk.text,
                    }
                    for chunk in final_chunks
                ],
                metadata={
                    "bm25_top": bm25_ranking[:5],
                    "dense_top": dense_ranking[:5],
                },
            )

        context = "\n\n".join(chunker.chunks[cid].text for cid in final_chunk_ids)
        return {"context": context, "reference_chunks": final_chunk_ids}

    def compile_generator_prompt(
        self, *, context: str, query: str
    ) -> tuple[ChatCompletionMessageParam, ChatCompletionMessageParam, Any]:
        """Compile the split generator prompts into separate system + user messages.

        The system instructions come from ``generator_system_prompt_name`` (no
        variables); ``generator_user_prompt_name`` injects the retrieved
        ``context`` and the ``query``.

        Returns ``(system_message, user_message, prompt)`` — the two messages are
        ready to drop into a chat request, and ``prompt`` is the user prompt
        object (the one carrying the dynamic variables) so callers can link it to
        their generation via ``update_current_generation(prompt=...)``.
        """
        system_prompt = self._langfuse.get_prompt(self._generator_system_prompt_name)
        user_prompt = self._langfuse.get_prompt(self._generator_user_prompt_name)
        system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": system_prompt.compile(),
        }
        user_message: ChatCompletionMessageParam = {
            "role": "user",
            "content": user_prompt.compile(context=context, query=query),
        }
        return system_message, user_message, user_prompt

    async def generate(
        self,
        *,
        query: str,
        model: str,
        bm25_top_n: int | None = None,
        dense_top_n: int | None = None,
        rrf_k: int | None = None,
        rerank_pool_size: int | None = None,
        final_k: int | None = None,
    ) -> dict[str, Any]:
        """Run hybrid retrieval, rerank, and generate an answer for the query.

        The knob arguments default to ``None`` and are resolved from the (possibly
        runtime-overridden) ``RAGConfig`` inside :meth:`retrieve` when omitted.
        """
        retrieval = await self.retrieve(
            query=query,
            bm25_top_n=bm25_top_n,
            dense_top_n=dense_top_n,
            rrf_k=rrf_k,
            rerank_pool_size=rerank_pool_size,
            final_k=final_k,
        )
        context = retrieval["context"]
        final_chunk_ids = retrieval["reference_chunks"]
        system_message, user_message, prompt = self.compile_generator_prompt(
            context=context, query=query
        )

        with self._langfuse.start_as_current_observation(
            name="rag-generate",
            as_type="generation",
            model=model,
            input={"context": context, "query": query},
        ) as generation_span:
            self._langfuse.update_current_generation(prompt=prompt)
            response = await self._chat_complete(
                model=model, messages=[system_message, user_message]
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
        batch_size = self._embedding_batch_size
        batches = [
            texts[offset : offset + batch_size]
            for offset in range(0, len(texts), batch_size)
        ]
        progress = _EmbedProgress(batches=len(batches), chunks=len(texts))
        results = await asyncio.gather(
            *(
                self._embed_batch(batch, semaphore=semaphore, progress=progress)
                for batch in batches
            )
        )
        vectors = [
            embedding for batch_vectors in results for embedding in batch_vectors
        ]
        return np.asarray(vectors, dtype=np.float32)

    async def _embed_batch(
        self,
        batch: list[str],
        semaphore: asyncio.Semaphore | None = None,
        progress: "_EmbedProgress | None" = None,
    ) -> list[list[float]]:
        response = await self._embed_call(batch, semaphore=semaphore)
        if progress is not None:
            progress.batch_done(len(batch))
        return [item.embedding for item in response.data]

    @with_openai_retry()
    async def _embed_call(self, payload, semaphore: asyncio.Semaphore | None = None):
        if semaphore is not None:
            async with semaphore:
                return await self._openai.embeddings.create(
                    model=self._embedding_model, input=payload
                )
        return await self._openai.embeddings.create(
            model=self._embedding_model, input=payload
        )

    @with_openai_retry()
    async def _chat_complete(
        self, *, model: str, messages: list[ChatCompletionMessageParam]
    ):
        return await self._openai.chat.completions.create(
            model=model,
            messages=messages,
        )

    def _require_built(self) -> CorpusChunker:
        if self._chunker is None or self._dense is None or self._bm25 is None:
            raise RuntimeError(
                "RAGPipeline.build(...) must be called before generate/resolve."
            )
        return self._chunker
