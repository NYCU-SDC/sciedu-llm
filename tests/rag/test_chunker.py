from rag.chunker import CorpusChunker


def test_chunks_have_sequential_ids_and_start_indices_match_source():
    content = "ABCDEFGHIJ" * 100  # 1,000 chars, easily splittable
    chunker = CorpusChunker(chunk_size=200, chunk_overlap=50)

    chunks = chunker.add_chapter("ch1", content)

    assert [c.id for c in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert content[chunk.start : chunk.end] == chunk.text


def test_resolve_chunks_returns_overlapping_chunks_only():
    content = "ABCDEFGHIJ" * 100
    chunker = CorpusChunker(chunk_size=200, chunk_overlap=50)
    chunker.add_chapter("ch1", content)

    overlapping = chunker.resolve_chunks("ch1", 100, 250)
    assert overlapping, "expected at least one overlapping chunk"

    for chunk_id in overlapping:
        chunk = chunker.chunks[chunk_id]
        assert chunk.chapter == "ch1"
        assert chunk.start < 250 and chunk.end > 100


def test_resolve_chunks_inverts_swapped_bounds():
    content = "ABCDEFGHIJ" * 100
    chunker = CorpusChunker(chunk_size=200, chunk_overlap=50)
    chunker.add_chapter("ch1", content)

    forward = chunker.resolve_chunks("ch1", 100, 250)
    backward = chunker.resolve_chunks("ch1", 250, 100)
    assert forward == backward


def test_resolve_chunks_empty_for_unknown_chapter():
    chunker = CorpusChunker(chunk_size=200, chunk_overlap=50)
    chunker.add_chapter("ch1", "ABCDEFGHIJ" * 100)

    assert chunker.resolve_chunks("ch_does_not_exist", 0, 999) == []


def test_resolve_chunks_empty_for_non_overlapping_range():
    chunker = CorpusChunker(chunk_size=200, chunk_overlap=50)
    chunker.add_chapter("ch1", "ABCDEFGHIJ" * 50)  # 500 chars total

    # Range starts past the end of the content; no chunk should match.
    assert chunker.resolve_chunks("ch1", 10_000, 10_500) == []


def test_resolve_chunks_isolates_per_chapter():
    chunker = CorpusChunker(chunk_size=200, chunk_overlap=50)
    chunker.add_chapter("ch1", "ABCDEFGHIJ" * 100)
    chunker.add_chapter("ch2", "0123456789" * 100)

    ch1_chunks = chunker.resolve_chunks("ch1", 0, 99_999)
    ch2_chunks = chunker.resolve_chunks("ch2", 0, 99_999)

    assert set(ch1_chunks).isdisjoint(set(ch2_chunks))
    for cid in ch1_chunks:
        assert chunker.chunks[cid].chapter == "ch1"
    for cid in ch2_chunks:
        assert chunker.chunks[cid].chapter == "ch2"
