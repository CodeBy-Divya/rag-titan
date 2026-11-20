"""Tests for SQLiteStore: the shared store checks plus what a file adds."""

import numpy as np
import pytest
from store_contract import QUERY, VECTORS, StoreContract, chunk

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import DenseRetriever
from rag_titan.vector_stores import SQLiteStore


class TestSQLiteStore(StoreContract):
    persistent = True

    def make_store(self, tmp_path):
        return SQLiteStore(tmp_path / "index.db")

    def reopen(self, store, tmp_path):
        store.close()
        return SQLiteStore(tmp_path / "index.db")


def test_index_today_ask_tomorrow(tmp_path):
    policy = tmp_path / "leave.md"
    policy.write_text("You get twenty paid leave days. Remote work needs a laptop.",
                      encoding="utf-8")
    first = SQLiteStore(tmp_path / "db" / "rag.db")           # the folder is created
    IndexingPipeline(WordChunker(5, 0), HashEmbedder(64), first).run([policy])
    first.close()

    later = SQLiteStore(tmp_path / "db" / "rag.db")           # a new Python session
    results = DenseRetriever(HashEmbedder(64), later).retrieve("paid leave days", top_k=1)

    assert later.get_info()["embed_model"] == "hash"
    assert results[0]["source"] == "leave.md" and "leave" in results[0]["text"]


def test_two_stores_on_one_file_see_each_others_writes(tmp_path):
    reader, writer = SQLiteStore(tmp_path / "x.db"), SQLiteStore(tmp_path / "x.db")
    writer.add([chunk(0)], VECTORS[:1])
    assert [r["id"] for r in reader.search(QUERY)] == ["chunk_0"]

    writer.add([chunk(1)], VECTORS[1:2])
    writer.delete_source("hr.md")
    writer.add([chunk(2, "leave.md")], VECTORS[2:])
    assert [r["id"] for r in reader.search(QUERY)] == ["chunk_2"]


def test_same_id_is_replaced_and_zero_vectors_score_zero(tmp_path):
    store = SQLiteStore(tmp_path / "x.db")
    store.add([chunk(0)], VECTORS[:1])
    store.add([{**chunk(0), "text": "new"}], [np.zeros(4, dtype=np.float32)])

    result = store.search(QUERY)
    assert store.count() == 1 and result[0]["text"] == "new" and result[0]["score"] == 0.0


def test_clear_errors(tmp_path):
    with pytest.raises(ValueError, match="is a folder"):
        SQLiteStore(tmp_path)

    broken = tmp_path / "notes.db"
    broken.write_text("this is not a database " * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="not a rag-titan SQLite index"):
        SQLiteStore(broken)

    store = SQLiteStore(tmp_path / "x.db")
    store.add([chunk(0)], VECTORS[:1])
    with pytest.raises(ValueError, match=r"same length, got \[3, 4\]"):
        store.add([chunk(1)], [np.ones(3, dtype=np.float32)])
    assert store.count() == 1 and "x.db" in repr(store)
