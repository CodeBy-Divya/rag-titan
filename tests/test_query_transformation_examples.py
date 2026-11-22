"""
Tests for examples/hyde_rag.py and examples/rag_fusion_rag.py and their copies on
the RAG types page. The model is faked; the page's keyword-search claims do not
depend on a model, so they are checked exactly.
"""

import runpy
from pathlib import Path

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan.chunking import ParagraphChunker
from rag_titan.embeddings import HashEmbedder
from rag_titan.pipelines import IndexingPipeline
from rag_titan.retrieval import BM25Retriever
from rag_titan.vector_stores import MemoryStore

ROOT = Path(__file__).parent.parent
EXAMPLES = ["hyde_rag.py", "rag_fusion_rag.py"]
PAGE = ROOT / "docs" / "rag-types.md"
DOCS = ROOT / "examples" / "retrieval_docs"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="examples are only in the repo")


@pytest.fixture
def fakes(monkeypatch):
    """A fake LLM and the hash embedder, so the examples run without Ollama."""
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    for name in ("RETRIEVER", "RERANKER", "VECTOR_STORE", "TOP_K", "TOP_N"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_page_shows_the_script_exactly(name):
    script = (ROOT / "examples" / name).read_text(encoding="utf-8")
    assert f"```python\n{script}```" in PAGE.read_text(encoding="utf-8")


def test_hyde_example_runs(fakes, capsys):
    runpy.run_path(str(ROOT / "examples" / "hyde_rag.py"))
    out = capsys.readouterr().out
    assert "Made-up answer (only used to search):" in out and "Search 2 finds:" in out
    assert out.strip().endswith("Answer: the answer")


def test_rag_fusion_example_runs(fakes, capsys):
    runpy.run_path(str(ROOT / "examples" / "rag_fusion_rag.py"))
    out = capsys.readouterr().out
    assert "query 1: What gear does a newcomer receive?" in out
    assert out.strip().endswith("Answer: the answer")


def keyword_search():
    store = MemoryStore()
    IndexingPipeline(ParagraphChunker(300, 0), HashEmbedder(), store).run(sorted(DOCS.glob("*.md")))
    return BM25Retriever(store)


def test_keyword_search_claims_on_the_page():
    """The page: the question and two rewordings find nothing; "new player equipment" and
    the made-up answer's words find new_starters.md first."""
    search = keyword_search()

    def files(query):
        return [chunk["source"] for chunk in search.retrieve(query, 5)]

    for query in ["What gear does a newcomer receive?", "newcomer starting gear",
                  "beginner motorcycle kit"]:
        assert files(query) == []
    assert files("new player equipment")[:2] == ["new_starters.md", "account_errors.md"]
    made_up = "New employees are provided with a standard starter kit upon hire."
    assert files(made_up)[0] == "new_starters.md"
