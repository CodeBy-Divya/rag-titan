"""Tests for retrieval/fusion.py: RRF, fused 0-1 scores and the multi-query search."""

import pytest

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.rag_types import Trace
from rag_titan.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    fuse_ranked,
    rrf_fuse,
    search_and_fuse,
)
from rag_titan.vector_stores import MemoryStore


def result(chunk_id, score=0.5, source="a.md", text=None):
    """A search result as a retriever returns it."""
    return {"id": chunk_id, "source": source, "text": text or f"text of {chunk_id}",
            "score": score, "retriever": "dense", "score_trace": {"dense": score}}


@pytest.fixture
def store(tmp_path):
    (tmp_path / "hr.md").write_text("You get 20 paid leave days per year. Sick leave needs "
                                    "a note from the doctor.", encoding="utf-8")
    (tmp_path / "it.md").write_text("Remote work needs a laptop. Error E-4022 means the "
                                    "printer is offline.", encoding="utf-8")
    store = MemoryStore()
    IndexingPipeline(WordChunker(6, 0), HashEmbedder(64), store).run(
        [tmp_path / "hr.md", tmp_path / "it.md"])
    return store


def test_rrf_by_hand_over_three_lists():
    lists = {"q1": [result("a"), result("b")], "q2": [result("b"), result("c")],
             "q3": [result("b")]}
    fused = rrf_fuse(lists, k=60)

    assert [chunk["id"] for chunk, _ in fused] == ["b", "a", "c"]
    trace = fused[0][1]                  # b: second, first, first
    assert trace["rrf"] == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
    assert (trace["q1_rank"], trace["q2_rank"], trace["q3_rank"]) == (2, 1, 1)
    assert "score" not in fused[0][0] and "retriever" not in fused[0][0]


def test_equal_rrf_goes_to_more_relevance_then_first_seen():
    lists = {"q1": [result("a", 0.9), result("b", 0.2)], "q2": [result("b", 0.2), result("a", 0.1)]}
    assert [chunk["id"] for chunk, _ in rrf_fuse(lists)] == ["a", "b"]      # 1.0 vs 0.4

    same = {"q1": [result("x", 0.5), result("y", 0.5)], "q2": [result("y", 0.5), result("x", 0.5)]}
    assert [chunk["id"] for chunk, _ in rrf_fuse(same)] == ["x", "y"]       # seen first


def test_first_in_every_list_scores_one_and_scores_stay_in_0_1():
    lists = {"q1": [result("a"), result("b")], "q2": [result("a")], "q3": [result("a")]}
    fused = fuse_ranked(lists, top_k=5)

    assert fused[0]["id"] == "a" and fused[0]["score"] == pytest.approx(1.0)
    assert all(0 < chunk["score"] <= 1 for chunk in fused)
    assert len(fuse_ranked(lists, top_k=1)) == 1


def test_near_copies_are_dropped_after_fusing():
    same_text = "Every employee gets twenty paid leave days each year in the company"
    lists = {"q1": [result("a", text=same_text), result("c")],
             "q2": [result("b", text=same_text), result("c")]}

    kept = [chunk["id"] for chunk in fuse_ranked(lists, top_k=5, dedupe_overlap=0.5)]
    assert kept.count("a") + kept.count("b") == 1
    assert len(fuse_ranked(lists, top_k=5, dedupe_overlap=None)) == 3


def test_search_and_fuse_records_every_search_then_the_fuse(store):
    trace = Trace()
    retriever = DenseRetriever(HashEmbedder(64), store)
    fused = search_and_fuse(retriever, ["leave days", "doctor note", "laptop"], 3, trace)

    assert [step["step"] for step in trace.steps] == ["retrieve"] * 3 + ["fuse_queries"]
    assert [step["input"]["query"] for step in trace.steps[:3]] == ["leave days", "doctor note",
                                                                     "laptop"]
    assert len(fused) <= 3 and all(0 < chunk["score"] <= 1 for chunk in fused)
    assert any(key.endswith("_rank") for key in fused[0]["score_trace"])


def test_search_and_fuse_with_hybrid_shows_hybrid_steps_per_query(store):
    trace = Trace()
    hybrid = HybridRetriever(DenseRetriever(HashEmbedder(64), store, dedupe_overlap=None),
                             BM25Retriever(store, dedupe_overlap=None))
    search_and_fuse(hybrid, ["E-4022", "printer offline"], 2, trace)

    one_search = ["vector_search", "keyword_search", "fuse"]
    assert [step["step"] for step in trace.steps] == one_search * 2 + ["fuse_queries"]


def test_one_query_is_a_plain_search(store):
    trace = Trace()
    found = search_and_fuse(DenseRetriever(HashEmbedder(64), store), ["leave", "  "], 2, trace)
    assert [step["step"] for step in trace.steps] == ["retrieve"] and len(found) == 2


@pytest.mark.parametrize("queries, top_k, message", [
    ([], 3, "at least one query"),
    (["", None], 3, "at least one query"),
    (["leave"], 0, "top_k"),
    (["leave"], True, "top_k"),
])
def test_bad_input_gives_a_clear_error(store, queries, top_k, message):
    with pytest.raises(ValueError, match=message):
        search_and_fuse(DenseRetriever(HashEmbedder(64), store), queries, top_k, Trace())


def test_hybrid_still_uses_the_same_fusion(store):
    hybrid = HybridRetriever(DenseRetriever(HashEmbedder(64), store, dedupe_overlap=None),
                             BM25Retriever(store, dedupe_overlap=None))
    dense = hybrid.dense.retrieve("printer error", 20)
    keyword = hybrid.keyword.retrieve("printer error", 20)

    assert hybrid.fuse({"dense": dense, "bm25": keyword}) == rrf_fuse(
        {"dense": dense, "bm25": keyword}, 60)
    assert [m["id"] for m in hybrid.retrieve("printer error", 3)] == [
        m["id"] for m in fuse_ranked({"dense": dense, "bm25": keyword}, 3)]
