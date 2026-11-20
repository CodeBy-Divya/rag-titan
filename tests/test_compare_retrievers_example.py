"""Tests for examples/compare_retrievers.py and its copy on the Retrieval page."""

import runpy
from pathlib import Path

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan.chunking import ParagraphChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.evaluation import evaluate_retriever, load_questions
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import BM25Retriever
from rag_titan.vector_stores import MemoryStore

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "examples" / "compare_retrievers.py"
PAGE = ROOT / "docs" / "retrieval.md"

pytestmark = pytest.mark.skipif(not EXAMPLE.is_file(), reason="examples are only in the repo")


def test_the_page_shows_the_script_exactly():
    assert f"```python\n{EXAMPLE.read_text(encoding='utf-8')}```" in PAGE.read_text(
        encoding="utf-8")


def test_the_example_runs(monkeypatch, capsys):
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    for name in ("RETRIEVER", "VECTOR_STORE"):
        monkeypatch.delenv(name, raising=False)

    runpy.run_path(str(EXAMPLE))

    out = capsys.readouterr().out
    assert out.startswith("Retrieval report: 6 questions, top_k=1")
    assert all(f"  {name}" in out for name in ("dense", "bm25", "hybrid"))
    assert "bm25    missed: meaning_1, meaning_2, meaning_3" in out   # no shared words
    assert "Answer: the answer" in out


def test_bm25_finds_every_error_code_and_no_reworded_question():
    """The page's claim about bm25 does not depend on any model: check it exactly."""
    store = MemoryStore()
    IndexingPipeline(ParagraphChunker(300, 0), HashEmbedder(), store).run(
        sorted((ROOT / "examples" / "retrieval_docs").glob("*.md")))
    result = evaluate_retriever(BM25Retriever(store),
                                load_questions(ROOT / "examples" / "retrieval_questions.json"), 1)

    recall = {row["id"]: row["recall"] for row in result["rows"]}
    assert recall == {"code_1": 1.0, "code_2": 1.0, "code_3": 1.0,
                      "meaning_1": 0.0, "meaning_2": 0.0, "meaning_3": 0.0}
