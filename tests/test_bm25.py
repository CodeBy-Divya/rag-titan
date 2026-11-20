"""Tests for BM25Index and BM25Retriever (no models needed)."""

import math

import pytest

from rag_titan.chunking import WordChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import BM25Index, BM25Retriever, tokenize
from rag_titan.vector_stores import BaseStore, MemoryStore, SQLiteStore

TEXTS = ["leave days leave", "remote work days", "salary paid monthly"]


def test_tokenize_keeps_letters_and_digits_in_any_language():
    assert tokenize("Invoice INV-2041, paid!") == ["invoice", "inv", "2041", "paid"]
    assert tokenize("Urlaubstage für Mitarbeiter") == ["urlaubstage", "für", "mitarbeiter"]


def test_score_matches_the_formula_by_hand():
    index = BM25Index(TEXTS, k1=1.5, b=0.75)

    # "leave": in 1 of 3 texts, twice in text 0; every text has 3 words, so |d| = avgdl
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf * 2 * (1.5 + 1) / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 3))
    [(number, score)] = index.search("leave")

    assert number == 0 and score == pytest.approx(expected)
    assert expected == pytest.approx(1.40119, abs=1e-5)


def test_common_words_count_less_and_ties_keep_text_order():
    index = BM25Index(TEXTS)
    assert index.idf("days") < index.idf("leave")                 # in 2 texts vs 1
    assert [n for n, _ in index.search("days")] == [0, 1]          # equal scores
    assert index.search("holiday") == [] and index.search("") == []


def test_longer_texts_need_more_repeats():
    index = BM25Index(["leave", "leave and many other words about the office"], b=0.75)
    assert [n for n, _ in index.search("leave")] == [0, 1]
    flat = BM25Index(["leave", "leave and many other words about the office"], b=0)
    assert flat.search("leave")[0][1] == pytest.approx(flat.search("leave")[1][1])


@pytest.mark.parametrize("k1, b", [(-1, 0.75), (1.5, 1.5), ("1", 0.5)])
def test_bad_parameters(k1, b):
    with pytest.raises(ValueError, match="k1|b must"):
        BM25Index(TEXTS, k1=k1, b=b)
    with pytest.raises(ValueError):
        BM25Retriever(MemoryStore(), k1=k1, b=b)


def index_files(store, tmp_path, texts, reset=True):
    """Write one file per text and index them with 50-word chunks."""
    paths = []
    for name, text in texts.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    IndexingPipeline(WordChunker(50, 0), HashEmbedder(64), store).run(paths, reset=reset)


POLICIES = {
    "billing.md": "Invoice INV-2041 was paid late. Invoices are paid within 30 days.",
    "invoices.md": "All invoices need a purchase order. Late invoices are flagged.",
    "leave.md": "You get twenty paid leave days every year.",
}


def test_an_exact_id_ranks_first_with_the_evidence_format(tmp_path):
    store = MemoryStore()
    index_files(store, tmp_path, POLICIES)

    results = BM25Retriever(store).retrieve("status of INV-2041", top_k=2)

    assert results[0]["source"] == "billing.md"
    assert results[0]["retriever"] == "bm25"
    assert results[0]["score_trace"] == {"bm25": results[0]["score"]}
    assert len(results) == 1                      # no other chunk has "inv" or "2041"


@pytest.mark.parametrize("make_store", [lambda p: MemoryStore(), lambda p: SQLiteStore(p / "x.db")])
def test_the_index_follows_every_store_change(make_store, tmp_path):
    store = make_store(tmp_path)
    retriever = BM25Retriever(store)
    index_files(store, tmp_path, {"leave.md": POLICIES["leave.md"]})
    assert retriever.retrieve("leave days")[0]["source"] == "leave.md"

    # start over with a different file of the same size: same count, new revision
    index_files(store, tmp_path, {"remote.md": "You get twenty remote work days every year."})
    assert [r["source"] for r in retriever.retrieve("twenty days")] == ["remote.md"]

    index_files(store, tmp_path, {"remote.md": "Remote work needs a laptop."}, reset=False)
    assert retriever.retrieve("twenty days") == []           # the replaced text is gone
    assert retriever.retrieve("laptop")[0]["source"] == "remote.md"

    store.clear()
    assert not retriever.has_documents() and retriever.retrieve("laptop") == []


def test_top_k_and_near_copies(tmp_path):
    store = MemoryStore()
    index_files(store, tmp_path, POLICIES)
    retriever = BM25Retriever(store)

    assert len(retriever.retrieve("paid invoices", top_k=1)) == 1
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("paid", top_k=0)

    overlapping = [{"id": f"chunk_{n}", "source": "a.md", "text": "paid leave days",
                    "start_offset": n, "end_offset": n + 15} for n in range(2)]
    store = MemoryStore()
    store.add(overlapping, HashEmbedder(64).embed_many(["x", "y"]))
    assert len(BM25Retriever(store).retrieve("paid")) == 1
    assert len(BM25Retriever(store, dedupe_overlap=None).retrieve("paid")) == 2


def test_a_store_that_cannot_list_chunks_says_so():
    class TinyStore(BaseStore):
        def count(self):
            return 1

    with pytest.raises(NotImplementedError, match="RETRIEVER=dense"):
        BM25Retriever(TinyStore()).retrieve("leave")
