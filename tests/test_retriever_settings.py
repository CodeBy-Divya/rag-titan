"""
Tests for RETRIEVER / RERANKER settings, RAGTitan.count(), and comparing
retrievers with `rag-titan evaluate` (hash embedder, fake LLM, no Ollama).
"""

import json

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan import RAGTitan, cli, get_settings
from rag_titan.embeddings import HashEmbedder
from rag_titan.evaluation import compare_setups, evaluate_retriever, load_questions
from rag_titan.factory import build_reranker, build_retriever
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from rag_titan.retrieval.rerankers import KeywordReranker, MMRReranker
from rag_titan.vector_stores import MemoryStore


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """Two policy files and a questions file, the hash embedder and a fake LLM."""
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    for name in ("RETRIEVER", "RERANKER", "VECTOR_STORE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "billing.md").write_text("Invoice INV-2041 was paid late.", encoding="utf-8")
    (tmp_path / "leave.md").write_text("You get twenty paid leave days.", encoding="utf-8")
    questions = [{"id": "q1", "question": "What about INV-2041?",
                  "required_evidence_ids": ["billing.md"]},
                 {"id": "q2", "question": "How many leave days?",
                  "required_evidence_ids": ["leave.md"]}]
    (tmp_path / "q.json").write_text(json.dumps(questions), encoding="utf-8")
    return tmp_path


def test_settings_pick_the_retriever_and_reranker():
    embedder, store = HashEmbedder(), MemoryStore()
    expected = {"dense": DenseRetriever, "bm25": BM25Retriever, " Hybrid ": HybridRetriever}
    for name, kind in expected.items():
        assert type(build_retriever(get_settings({"retriever": name}), embedder, store)) is kind

    assert get_settings()["retriever"] == "dense"                       # unchanged default
    assert isinstance(build_reranker(get_settings({"reranker": "mmr"}), embedder=embedder),
                      MMRReranker)
    assert isinstance(build_reranker(get_settings()), KeywordReranker)  # old call still works


@pytest.mark.parametrize("overrides, message", [
    ({"retriever": "sparse"}, "Unknown retriever: 'sparse'. Use one of: dense, bm25, hybrid"),
    ({"reranker": "cohere"},
     "Unknown reranker: 'cohere'. Use one of: keyword, none, mmr, llm, cross_encoder"),
])
def test_typos_list_the_choices(overrides, message):
    settings = get_settings(overrides)
    with pytest.raises(ValueError, match=message):
        build_retriever(settings, HashEmbedder(), MemoryStore())
        build_reranker(settings, embedder=HashEmbedder())
    with pytest.raises(ValueError, match="needs the embedder"):
        build_reranker(get_settings({"reranker": "mmr"}))


def test_rag_titan_count_and_hybrid_answers(folder):
    rag = RAGTitan(settings={"retriever": "hybrid", "reranker": "mmr"})
    assert rag.count() == 0
    rag.index(["billing.md", "leave.md"])

    result = rag.run("What about INV-2041?")

    assert rag.count() == 2 and result["answer"] == "the answer"
    assert result["sources"][0]["source"] == "billing.md"
    assert result["sources"][0]["retriever"] == "hybrid"


def test_setups_combine_rag_types_chunkers_and_retrievers():
    args = cli.build_parser().parse_args(
        ["evaluate", "--files", "a.md", "--questions", "q.json", "--rag-type", "simple",
         "--chunker", "word", "--retriever", "dense", "hybrid"])

    setups = cli.build_setups(args)

    assert [s["name"] for s in setups] == ["simple + word + dense", "simple + word + hybrid"]
    assert setups[1]["settings"] == {"chunker": "word", "retriever": "hybrid"}


def test_retrievers_share_one_index_when_compared(folder, monkeypatch):
    runs = []
    real_run = IndexingPipeline.run
    monkeypatch.setattr(IndexingPipeline, "run",
                        lambda self, files, reset=True: runs.append(1) or real_run(self, files))
    setups = [{"name": name, "rag_type": "simple", "settings": {"retriever": name}}
              for name in ("dense", "bm25", "hybrid")]

    results = compare_setups(["billing.md", "leave.md"], "q.json", setups)

    assert len(runs) == 1                                   # embedded once, searched 3 ways
    assert [r["name"] for r in results] == ["dense", "bm25", "hybrid"]


def test_cli_compares_retrievers_without_an_llm(folder, capsys):
    cli.main(["evaluate", "--files", "billing.md", "leave.md", "--questions", "q.json",
              "--retriever", "dense", "bm25", "hybrid", "--retrieval-only", "--top-k", "1"])

    out = capsys.readouterr().out
    assert all(name in out for name in ("dense", "bm25", "hybrid"))


def test_recall_is_the_same_on_every_run(folder):
    """The handbook's "reproducible recall" gate: same index, same questions, same numbers."""
    def scores():
        rag = RAGTitan(settings={"retriever": "hybrid"})
        rag.index(["billing.md", "leave.md"])
        result = evaluate_retriever(rag.parts["retriever"], load_questions("q.json"), 1)
        return [(row["id"], row["recall"], row["reciprocal_rank"]) for row in result["rows"]]

    assert scores() == scores()


def test_ask_with_a_retriever_option(folder, capsys):
    cli.main(["ask", "--files", "billing.md", "leave.md", "--question", "INV-2041?",
              "--retriever", "bm25", "--trace"])
    # only billing.md shares a word with the question; dense search would return both files
    assert "chunks=1 chunk" in capsys.readouterr().out
