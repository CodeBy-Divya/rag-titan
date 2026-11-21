"""Tests for HybridRetriever (RRF), MMRReranker and the 0-1 relevance scores."""

import numpy as np
import pytest

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_titan.retrieval.rerankers import KeywordReranker, MMRReranker
from rag_titan.vector_stores import MemoryStore

POLICIES = {
    "billing.md": "Invoice INV-2041 was paid late. Invoices are paid within 30 days.",
    "invoices.md": "All invoices need a purchase order. Late invoices are flagged.",
    "leave.md": "You get twenty paid leave days every year.",
    "remote.md": "Remote work is allowed two days a week with a laptop.",
}


@pytest.fixture
def store(tmp_path):
    store = MemoryStore()
    paths = []
    for name, text in POLICIES.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        paths.append(tmp_path / name)
    IndexingPipeline(WordChunker(50, 0), HashEmbedder(256), store).run(paths)
    return store


def hybrid(store, **options):
    return HybridRetriever(DenseRetriever(HashEmbedder(256), store), BM25Retriever(store),
                           **options)


def ranked(*ids):
    return [{"id": i, "source": "a.md", "text": i, "score": 0.5, "score_trace": {}} for i in ids]


def test_rrf_by_hand():
    fused = HybridRetriever(None, None, k=60).fuse({"dense": ranked("a", "b", "c"),
                                                    "bm25": ranked("c", "d")})

    assert [chunk["id"] for chunk, _ in fused] == ["c", "a", "b", "d"]   # b, d tie: b seen first
    trace = fused[0][1]
    assert trace["rrf"] == pytest.approx(1 / 63 + 1 / 61)
    assert (trace["dense_rank"], trace["bm25_rank"]) == (3, 1)
    assert "bm25_rank" not in fused[1][1]                                 # a: vector search only


def test_hybrid_finds_the_exact_id_and_explains_the_score(store):
    results = hybrid(store).retrieve("what happened to INV-2041", top_k=3)

    first = results[0]
    assert first["source"] == "billing.md" and first["retriever"] == "hybrid"
    assert {"dense", "dense_rank", "bm25", "bm25_rank", "rrf"} <= set(first["score_trace"])
    assert first["score"] == pytest.approx(first["score_trace"]["rrf"] / (2 / 61))
    assert all(0 < r["score"] <= 1 for r in results) and len(results) == 3


def test_first_in_both_lists_scores_one(store):
    first = hybrid(store).retrieve("twenty paid leave days", top_k=1)[0]
    assert (first["score_trace"]["dense_rank"], first["score_trace"]["bm25_rank"]) == (1, 1)
    assert first["score"] == pytest.approx(1.0)


def test_every_retriever_scores_from_0_to_1(store):
    for retriever in (DenseRetriever(HashEmbedder(256), store), BM25Retriever(store),
                      hybrid(store)):
        scores = [r["score"] for r in retriever.retrieve("paid invoices late days", top_k=4)]
        assert scores and all(-1 <= s <= 1 for s in scores), retriever.name
        reranked = KeywordReranker().rerank("paid invoices", retriever.retrieve("paid", 4), 2)
        assert all(r["rerank_score"] < 2 for r in reranked)    # the bonus stays a nudge


def test_hybrid_settings_and_empty_store(store):
    with pytest.raises(ValueError, match="k must"):
        hybrid(store, k=0)
    with pytest.raises(ValueError, match="top_k"):
        hybrid(store, pool_size=0)
    empty = MemoryStore()
    assert hybrid(empty).retrieve("leave") == [] and not hybrid(empty).has_documents()
    assert len(hybrid(store, pool_size=2).retrieve("paid days", top_k=5)) <= 4


class CountingEmbedder(HashEmbedder):
    """HashEmbedder that counts how many texts it embedded."""

    def __init__(self):
        super().__init__(64)
        self.texts = 0

    def embed_many(self, texts):
        self.texts += len(texts)
        return super().embed_many(texts)


def matches():
    return [{"id": "a", "text": "paid leave days every year", "score": 0.90},
            {"id": "a2", "text": "paid leave days every year", "score": 0.85},
            {"id": "b", "text": "remote work needs a laptop", "score": 0.60}]


def test_mmr_with_lambda_1_keeps_the_relevance_order():
    picked = MMRReranker(HashEmbedder(64), lambda_=1).rerank("leave", matches(), top_n=3)
    assert [m["id"] for m in picked] == ["a", "a2", "b"]
    assert picked[0]["rerank_score"] == pytest.approx(0.90)


def test_mmr_swaps_a_near_copy_for_something_new():
    picked = MMRReranker(HashEmbedder(64), lambda_=0.5).rerank("leave", matches(), top_n=2)

    assert [m["id"] for m in picked] == ["a", "b"]
    assert picked[1]["rerank_score"] == pytest.approx(0.5 * 0.60, abs=0.02)   # b shares no word


def test_mmr_embeds_each_text_once_and_checks_lambda():
    embedder = CountingEmbedder()
    reranker = MMRReranker(embedder)
    reranker.rerank("q", matches(), 2)
    reranker.rerank("q", matches(), 2)

    assert embedder.texts == 2                    # two different texts, embedded once
    assert reranker.rerank("q", [], 3) == []
    assert np.isclose(reranker._similarities(["a b", "a b"])[0, 1], 1.0)
    for bad in (-0.1, 1.5, "0.7"):
        with pytest.raises(ValueError, match="lambda_"):
            MMRReranker(embedder, lambda_=bad)


def chunk_at(chunk_id, start, score):
    return {"id": chunk_id, "source": "expenses.md", "text": chunk_id, "start_offset": start,
            "end_offset": start + 100, "score": score, "score_trace": {}}


def test_near_copies_are_removed_after_fusing_not_before():
    """Each search ranks a different copy of the same passage first (overlapping
    word chunks). The passage must still win, once, over a chunk both rank lower."""
    copy_a, copy_b = chunk_at("chunk_1", 0, 0.9), chunk_at("chunk_2", 50, 0.9)  # 50% overlap
    middling = chunk_at("chunk_8", 500, 0.5)
    retriever = HybridRetriever(None, None, dedupe_overlap=0.5)

    picked = retriever.fuse_top(dense=[copy_b, copy_a, middling],
                                keyword=[copy_a, copy_b, middling], top_k=2)

    assert [c["id"] for c in picked] == ["chunk_2", "chunk_8"]     # one copy, then the rest
    assert picked[0]["score_trace"]["dense_rank"] == 1
    assert picked[0]["score_trace"]["bm25_rank"] == 2               # both searches voted for it


def test_equal_rrf_scores_go_to_the_stronger_evidence():
    """#1 + #2 and #2 + #1 tie on RRF; the chunk with more relevance in total wins."""
    weak_keyword = {"id": "vpn", "source": "a.md", "text": "vpn", "score": 0.40, "score_trace": {}}
    strong_keyword = {"id": "account", "source": "b.md", "text": "account", "score": 1.0,
                      "score_trace": {}}
    dense = [{**weak_keyword, "score": 0.67}, {**strong_keyword, "score": 0.65}]

    fused = HybridRetriever(None, None).fuse({"dense": dense, "bm25": [strong_keyword,
                                                                       weak_keyword]})

    assert [chunk["id"] for chunk, _ in fused] == ["account", "vpn"]
    assert fused[0][1]["rrf"] == pytest.approx(fused[1][1]["rrf"])


def test_mmr_scores_are_plain_floats_for_json():
    import json
    picked = MMRReranker(HashEmbedder(64), lambda_=0.5).rerank("leave", matches(), top_n=3)
    assert all(type(m["rerank_score"]) is float for m in picked)
    json.dumps([m["rerank_score"] for m in picked])
