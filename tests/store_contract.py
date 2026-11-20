"""
Checks that every vector store must pass, written once.

Use it for a new store by inheriting and saying how to make (and reopen) one:

    class TestMyStore(StoreContract):
        def make_store(self, tmp_path):
            return MyStore(tmp_path / "index")

        def reopen(self, store, tmp_path):      # persistent stores only
            return MyStore(tmp_path / "index")
"""

import numpy as np
import pytest


def chunk(number, source="hr.md", **extra):
    """A chunk like the chunkers make, with optional location keys."""
    return {"id": f"chunk_{number}", "source": source, "text": f"text {number}",
            "start_offset": 10 * number, "end_offset": 10 * number + 9, **extra}


# three directions in 4D: the query [1, 0, 0, 0] is closest to chunk_0, then chunk_1
VECTORS = [np.array(v, dtype=np.float32) for v in
           ([1, 0, 0, 0], [0.6, 0.8, 0, 0], [0, 0, 1, 0])]
QUERY = np.array([1, 0, 0, 0], dtype=np.float32)


class StoreContract:
    """Inherit this in a Test... class and define make_store (and reopen)."""

    persistent = False

    def make_store(self, tmp_path):
        raise NotImplementedError

    def reopen(self, store, tmp_path):
        raise NotImplementedError

    @pytest.fixture
    def store(self, tmp_path):
        store = self.make_store(tmp_path)
        store.add([chunk(0), chunk(1, "leave.md"), chunk(2, "leave.md")], VECTORS)
        return store

    def test_search_returns_the_best_first_with_cosine_scores(self, store):
        results = store.search(QUERY, top_k=2)

        assert [r["id"] for r in results] == ["chunk_0", "chunk_1"]
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
        assert results[1]["score"] == pytest.approx(0.6, abs=1e-5)
        assert "vector" not in results[0]

    def test_top_k_larger_than_the_store(self, store):
        assert len(store.search(QUERY, top_k=10)) == 3

    def test_chunk_keys_come_back_unchanged(self, tmp_path):
        store = self.make_store(tmp_path)
        rich = chunk(0, page_numbers=[3, 4], section_path=["Leave", "Paid leave"])
        bare = chunk(1, start_offset=None, end_offset=None)
        store.add([rich, bare], VECTORS[:2])

        by_id = {r["id"]: r for r in store.search(QUERY, top_k=2)}

        assert {k: v for k, v in by_id["chunk_0"].items() if k != "score"} == rich
        assert {k: v for k, v in by_id["chunk_1"].items() if k != "score"} == bare

    def test_delete_source_count_and_clear(self, store):
        assert store.count() == 3
        assert store.delete_source("leave.md") == 2
        assert store.delete_source("missing.md") == 0
        assert [r["id"] for r in store.search(QUERY, top_k=5)] == ["chunk_0"]

        store.set_info({"embed_model": "m", "dimensions": 4, "next_chunk": 3})
        store.clear()
        assert store.count() == 0 and store.search(QUERY) == [] and store.get_info() is None

    def test_info_is_saved(self, store):
        info = {"embed_model": "nomic-embed-text", "dimensions": 4,
                "chunker": "WordChunker", "next_chunk": 3}
        store.set_info(info)
        assert store.get_info() == info

    def test_add_checks_lengths(self, tmp_path):
        with pytest.raises(ValueError, match="same length"):
            self.make_store(tmp_path).add([chunk(0)], VECTORS)

    def test_data_survives_reopening(self, store, tmp_path):
        if not self.persistent:
            pytest.skip("this store keeps everything in memory")
        store.set_info({"embed_model": "m", "dimensions": 4, "next_chunk": 3})

        again = self.reopen(store, tmp_path)

        assert again.count() == 3 and again.get_info()["next_chunk"] == 3
        assert again.search(QUERY, top_k=1)[0]["id"] == "chunk_0"
