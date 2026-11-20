"""
Tests for the store rules: unique chunk ids when adding, replacing a file,
and the check that stops mixing embedders (MemoryStore, no model needed).
"""

import pytest
from store_contract import StoreContract

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import DenseRetriever
from rag_titan.vector_stores import BaseStore, MemoryStore


class TestMemoryStore(StoreContract):
    def make_store(self, tmp_path):
        return MemoryStore()


@pytest.fixture
def files(tmp_path):
    """Two small policy files: 8 words each, so 2 chunks each with 4-word chunks."""
    leave = tmp_path / "leave.md"
    leave.write_text("You get twenty paid leave days every year.", encoding="utf-8")
    remote = tmp_path / "remote.md"
    remote.write_text("Remote work is allowed two days each week.", encoding="utf-8")
    return leave, remote


def pipeline(store, dimensions=64):
    return IndexingPipeline(WordChunker(4, 0), HashEmbedder(dimensions), store)


def ids(store):
    return sorted(int(r["id"].split("_")[1]) for r in store.search(
        HashEmbedder(64).embed("days"), top_k=100))


def test_fresh_index_starts_at_chunk_0_and_saves_its_info(files):
    store = MemoryStore()
    pipeline(store).run([files[0]])

    assert ids(store) == [0, 1]
    assert store.get_info() == {"embed_model": "hash", "dimensions": 64,
                                "chunker": "WordChunker", "next_chunk": 2}


def test_adding_a_file_continues_the_ids(files):
    store = MemoryStore()
    pipeline(store).run([files[0]])
    pipeline(store).run([files[1]], reset=False)

    assert ids(store) == [0, 1, 2, 3]
    assert store.get_info()["next_chunk"] == 4


def test_indexing_a_file_again_replaces_it(files):
    store = MemoryStore()
    pipeline(store).run(list(files))
    files[0].write_text("Leave is now thirty days.", encoding="utf-8")

    pipeline(store).run([files[0]], reset=False)

    texts = [r["text"] for r in store.search(HashEmbedder(64).embed("leave"), top_k=100)]
    assert store.count() == 4                     # 2 remote + 2 new leave chunks
    assert not any("twenty" in text for text in texts)
    assert ids(store) == [2, 3, 4, 5]             # old ids are never given out again


def test_reset_starts_over(files):
    store = MemoryStore()
    pipeline(store).run(list(files))
    pipeline(store).run([files[1]])

    assert ids(store) == [0, 1] and store.get_info()["next_chunk"] == 2


def test_adding_with_another_embedder_is_refused_and_changes_nothing(files):
    store = MemoryStore()
    pipeline(store, dimensions=64).run([files[0]])

    with pytest.raises(ValueError, match=r"built with hash \(64 dims\).*hash \(32 dims\)"):
        pipeline(store, dimensions=32).run([files[1]], reset=False)

    assert store.count() == 2 and store.get_info()["dimensions"] == 64


def test_searching_with_another_embedder_is_refused(files):
    store = MemoryStore()
    pipeline(store, dimensions=64).run([files[0]])

    retriever = DenseRetriever(HashEmbedder(32), store)
    with pytest.raises(ValueError, match="reset=True"):
        retriever.retrieve("leave days")
    assert DenseRetriever(HashEmbedder(64), store).retrieve("leave days")


class OldStyleStore(BaseStore):
    """A custom store written before 0.10.0: only the four original methods."""

    def __init__(self):
        self.inner = MemoryStore()

    def add(self, chunks, vectors):
        self.inner.add(chunks, vectors)

    def search(self, query_vector, top_k=5):
        return self.inner.search(query_vector, top_k)

    def clear(self):
        self.inner.clear()

    def count(self):
        return self.inner.count()


def test_a_store_without_the_new_methods_still_works(files, caplog):
    store = OldStyleStore()
    pipeline(store).run([files[0]])
    pipeline(store).run([files[1]], reset=False)

    assert ids(store) == [0, 1, 2, 3]             # numbering falls back to count()
    assert store.get_info() is None
    assert DenseRetriever(HashEmbedder(64), store).retrieve("remote work")

    with pytest.raises(NotImplementedError, match="reset=True"):
        store.delete_source("leave.md")
    pipeline(store).run([files[1]], reset=False)
    assert "may be stored twice" in caplog.text
