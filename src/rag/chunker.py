from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Order matters — try larger structural breaks first, fall back to characters.
DEFAULT_SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    " ",
    "",
]


@dataclass(frozen=True)
class Chunk:
    id: int
    chapter: str
    start: int
    end: int
    text: str


class CorpusChunker:
    """Splits chapter texts and tracks (chapter, char_range) → chunk id mappings."""

    def __init__(
        self,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators if separators is not None else DEFAULT_SEPARATORS,
            add_start_index=True,
        )
        self.chunks: list[Chunk] = []
        self._by_chapter: dict[str, list[Chunk]] = {}

    @property
    def chapters(self) -> list[str]:
        return list(self._by_chapter.keys())

    def add_chapter(self, chapter: str, content: str) -> list[Chunk]:
        docs = self._splitter.create_documents([content])
        added: list[Chunk] = []
        for doc in docs:
            start = doc.metadata["start_index"]
            chunk = Chunk(
                id=len(self.chunks),
                chapter=chapter,
                start=start,
                end=start + len(doc.page_content),
                text=doc.page_content,
            )
            self.chunks.append(chunk)
            self._by_chapter.setdefault(chapter, []).append(chunk)
            added.append(chunk)
        return added

    def resolve_chunks(self, chapter: str, start: int, end: int) -> list[int]:
        """Return chunk ids in `chapter` whose range overlaps with `[start, end)`."""
        if start > end:
            start, end = end, start
        return [
            chunk.id
            for chunk in self._by_chapter.get(chapter, [])
            if chunk.start < end and chunk.end > start
        ]
