"""Process-wide index cache; corpus ownership belongs to preset documents.

The key is the canonical corpus set plus settings baked into the indexes.
Retrieval settings remain global and are applied to every cached pipeline.
Completed and in-flight builds are shared, but different keys never mutate one
another's pipeline. Cache contents are intentionally lost on restart.
"""

import asyncio
from collections.abc import Callable, Iterable
from typing import Any

from app.presets import Preset
from app.rag_builds import BuildStatus, RagBuildManager
from rag import RAGPipeline
from rag.config import RAGConfig


class RagBuildCache:
    def __init__(
        self,
        factory: Callable[[RAGConfig], RAGPipeline],
        *,
        config: RAGConfig | None = None,
    ) -> None:
        self._factory = factory
        self._config = (config or RAGConfig()).model_copy(deep=True)
        self._entries: dict[tuple[object, ...], RagBuildManager] = {}

    def config_snapshot(self) -> dict[str, Any]:
        return self._config.model_dump()

    def manager(self, preset: Preset) -> RagBuildManager | None:
        names = preset.tools.rag.corpus_datasets
        if not preset.tools.rag.enable_tool or not names:
            return None
        names = sorted(set(names))
        key = (
            tuple(names),
            *(getattr(self._config, field) for field in RAGPipeline.BUILD_TIME_FIELDS),
        )
        if key not in self._entries:
            pipeline = self._factory(self._config.model_copy(deep=True))
            self._entries[key] = RagBuildManager(pipeline)
        return self._entries[key]

    def ensure(self, preset: Preset) -> RagBuildManager | None:
        """Start a missing build once. Failed/cancelled jobs need an explicit retry."""
        manager = self.manager(preset)
        if (
            manager is not None
            and not manager.pipeline.is_built
            and manager.state.status is BuildStatus.IDLE
        ):
            manager.start(sorted(set(preset.tools.rag.corpus_datasets)))
        return manager

    def pipeline(self, preset: Preset) -> RAGPipeline | None:
        """Return this preset's built index, scheduling a missing entry first."""
        manager = self.ensure(preset)
        if manager is None or not manager.pipeline.is_built:
            return None
        return manager.pipeline

    def warm(self, presets: Iterable[Preset]) -> bool:
        started = False
        for preset in presets:
            manager = self.manager(preset)
            if (
                manager is not None
                and not manager.pipeline.is_built
                and manager.state.status is BuildStatus.IDLE
            ):
                self.ensure(preset)
                started = True
        return started

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        # Validate the complete candidate before changing any serving state.
        candidate = RAGConfig.model_validate(self.config_snapshot() | overrides)
        self._config = candidate
        retrieval = {
            key: value
            for key, value in candidate.model_dump().items()
            if key not in RAGPipeline.BUILD_TIME_FIELDS
        }
        for manager in self._entries.values():
            manager.pipeline.apply_overrides(retrieval)

    def shutdown(self) -> None:
        for manager in self._entries.values():
            manager.shutdown()

    async def aclose(self) -> None:
        self.shutdown()
        await asyncio.gather(*(manager.wait() for manager in self._entries.values()))
