"""Seeded local preview: real preset/cache/index logic, synthetic model services.

Run `.venv/bin/python scripts/dev_preview.py` and point Vite's
VITE_DEV_API_TARGET at http://127.0.0.1:8000. Edits persist in memory until this
process exits. No external service or real Langfuse project is contacted.
"""

import asyncio
import copy
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fake_stack import (
    FakeLangfuse,
    FakePrompt,
    FakeStream,
    app,
    chunk,
    text_chunks,
    tool_turn,
)
from fastapi import HTTPException
from langfuse.api.commons.errors.not_found_error import NotFoundError

from app.dependencies import (
    Settings,
    get_langfuse_client,
    get_openai_client,
    get_rag_cache,
    get_settings,
)
from app.presets import PresetRegistry
from app.rag_cache import RagBuildCache
from rag import RAGPipeline
from rag.config import RAGConfig

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {
    "corpus/demo/biology": (
        "biology_photosynthesis",
        "光合作用分為光反應與卡爾文循環。光反應發生於葉綠體的類囊體膜，"
        "利用光能分解水、釋放氧氣，並產生 ATP 與 NADPH。"
        "卡爾文循環在葉綠體基質利用 ATP 與 NADPH 固定二氧化碳。",
    ),
    "corpus/demo/chemistry": (
        "chemistry_acids",
        "酸鹼中和反應會消耗氫離子與氫氧根離子，產生水。"
        "以鹽酸與氫氧化鈉為例，生成物為氯化鈉與水。"
        "在攝氏 25 度，純水的 pH 約為 7；酸性溶液的 pH 小於 7。",
    ),
}


def item(name, document, **kwargs):
    return SimpleNamespace(
        id=name, input=copy.deepcopy(document), metadata=None, status="ACTIVE", **kwargs
    )


class SeededLangfuse(FakeLangfuse):
    def __init__(self):
        super().__init__()
        self.datasets = {"config/presets": {}}
        for path in sorted((ROOT / "data/presets").glob("*.json")):
            document = json.loads(path.read_text())
            if document["tools"]["rag"]["enable_tool"]:
                document["tools"]["rag"]["corpus_datasets"] = ["corpus/demo/biology"]
            self.datasets["config/presets"][document["name"]] = item(
                document["name"], document
            )
        chemistry = copy.deepcopy(self.datasets["config/presets"]["default-chat"].input)
        chemistry.update(
            name="chemistry", description="化學教材範例；與生物教材使用獨立索引。"
        )
        chemistry["tools"]["rag"]["corpus_datasets"] = ["corpus/demo/chemistry"]
        self.datasets["config/presets"]["chemistry"] = item("chemistry", chemistry)
        for name, (chapter, content) in CORPORA.items():
            entry = item(chapter, {"content": content})
            entry.metadata = {"chapter": chapter}
            self.datasets[name] = {chapter: entry}
        self.datasets["questions/demo/biology"] = {
            "photosynthesis": item("photosynthesis", {"question": "光反應在哪裡發生？"})
        }
        self.api.datasets.list = lambda **_: SimpleNamespace(
            data=[SimpleNamespace(name=name) for name in sorted(self.datasets)],
            meta=SimpleNamespace(total_pages=1),
        )
        self.api.dataset_items = SimpleNamespace(delete=self.delete_item)

    def get_dataset(self, name):
        if name not in self.datasets:
            raise NotFoundError(body=f"Unknown local dataset: {name}")
        return SimpleNamespace(items=copy.deepcopy(list(self.datasets[name].values())))

    def create_dataset(self, *, name, **_kwargs):
        self.datasets.setdefault(name, {})

    def create_dataset_item(
        self, *, dataset_name, id, input, status="ACTIVE", **_kwargs
    ):
        entry = item(id, input)
        entry.status = status
        self.datasets.setdefault(dataset_name, {})[id] = entry

    def delete_item(self, *, id):
        for dataset in self.datasets.values():
            dataset.pop(id, None)

    def get_prompt(self, name, type=None):
        if name == "rag-generator-user":
            return SimpleNamespace(
                compile=lambda **values: (
                    f"教材：{values.get('context', '')}\n問題：{values.get('query', '')}"
                )
            )
        return FakePrompt()


class LocalModels:
    def list(self):
        async def models():
            for name, mode in [
                ("gpt-oss-120b", "chat"),
                ("bge-m3", "embedding"),
                ("BGE-Reranker-V2-M3", "rerank"),
            ]:
                yield SimpleNamespace(id=name, model_mode=mode)

        return models()


class PreviewCompletions:
    async def create(self, **kwargs):
        messages = kwargs["messages"]
        tools = {tool["function"]["name"] for tool in kwargs.get("tools") or []}
        results = [message for message in messages if message.get("role") == "tool"]
        if "rag_search" in tools and not results:
            query = next(
                (
                    message.get("content", "")
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "教材",
            )
            return FakeStream(
                tool_turn(
                    "preview-search",
                    "rag_search",
                    json.dumps({"query": query}, ensure_ascii=False),
                )
            )
        content = (
            str(results[-1].get("content", ""))
            if results
            else "你可以切換預設值並詢問光合作用或酸鹼中和，以檢查教材的選擇。"
        )
        return FakeStream(
            text_chunks("本機示範回覆（使用模擬模型）：\n\n" + content)
            + [chunk(finish_reason="stop")]
        )


class LocalEmbeddings:
    async def create(self, *, model, input):
        texts = input if isinstance(input, list) else [input]
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for character in text:
                bucket = hashlib.sha256(character.encode()).digest()[0] % len(vector)
                vector[bucket] += 1.0
            vectors.append(SimpleNamespace(embedding=vector))
        return SimpleNamespace(data=vectors)


class LocalReranker:
    async def rerank(self, *, query, documents, top_n):
        ranked = [
            (index, float(len(set(query) & set(document))))
            for index, document in enumerate(documents)
        ]
        return sorted(ranked, key=lambda pair: pair[1], reverse=True)[:top_n]


class PreviewPipeline(RAGPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reranker = LocalReranker()

    def apply_overrides(self, overrides):
        super().apply_overrides(overrides)
        self._reranker = LocalReranker()

    async def build(self, names, **kwargs):
        # Leave enough time to inspect/cancel a build from a second preset.
        await asyncio.sleep(4)
        await super().build(names, **kwargs)


class PreviewEvaluations:
    def list(self):
        return []

    def get(self, run_id):
        return None

    def start(self, **kwargs):
        raise HTTPException(
            503,
            "This local preview supports preset/RAG verification; evaluations need real model services.",
        )


def build_preview_app():
    settings = Settings(
        _env_file=None,
        openai_api_key="local-preview",
        openai_default_model="gpt-oss-120b",
        allowed_models="gpt-oss-120b",
    )
    store = SeededLangfuse()
    model = SimpleNamespace(
        base_url="http://127.0.0.1:8000/local-model",
        api_key="local-preview",
        models=LocalModels(),
        embeddings=LocalEmbeddings(),
        chat=SimpleNamespace(completions=PreviewCompletions()),
    )
    cache = RagBuildCache(
        lambda config: PreviewPipeline(model, store, config=config),
        config=RAGConfig(_env_file=None),
    )
    registry = PresetRegistry(langfuse=store, settings=settings)
    app.dependency_overrides.update(
        {
            get_settings: lambda: settings,
            get_langfuse_client: lambda: store,
            get_openai_client: lambda: model,
            get_rag_cache: lambda: cache,
        }
    )
    app.state.preset_registry = registry
    app.state.rag_cache = cache
    app.state.eval_runner = PreviewEvaluations()

    @asynccontextmanager
    async def lifespan(_app):
        await registry.refresh()
        cache.warm(registry.snapshot().values())
        managers = {cache.manager(p) for p in registry.snapshot().values()}
        await asyncio.gather(*(manager.wait() for manager in managers if manager))
        try:
            yield
        finally:
            await cache.aclose()

    app.router.lifespan_context = lifespan
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_preview_app(), host="127.0.0.1", port=8000)
