"""
Tests for the three RAG type examples (examples/simple_rag.py, adaptive_rag.py,
parent_child_rag.py) and their copies on the RAG types page. The model answers are
faked; the page's claims that do not depend on a model are checked exactly.
"""

import runpy
from pathlib import Path

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan.chunking import ParentChildChunker, WordChunker
from rag_titan.loaders import load_document

ROOT = Path(__file__).parent.parent
EXAMPLES = ["simple_rag.py", "adaptive_rag.py", "parent_child_rag.py"]
PAGE = ROOT / "docs" / "rag-types.md"
OFFBOARDING = ROOT / "examples" / "parent_child_docs" / "offboarding.md"

# the six steps of "What do I have to do before my last day?"
STEPS = ["four weeks", "exit form", "Hand over", "personal files", "return slot", "leave days"]

pytestmark = pytest.mark.skipif(not OFFBOARDING.is_file(), reason="examples are only in the repo")


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


def test_simple_example_runs(fakes, capsys):
    runpy.run_path(str(ROOT / "examples" / "simple_rag.py"))
    out = capsys.readouterr().out
    assert "top_k=5" in out and "top_n=3" in out and out.strip().endswith("Answer: the answer")


def test_adaptive_example_runs(fakes, capsys):
    runpy.run_path(str(ROOT / "examples" / "adaptive_rag.py"))
    out = capsys.readouterr().out
    assert out.count("Question: ") == 2 and "1. route" in out and "Answer: the answer" in out


def test_parent_child_example_runs(fakes, capsys):
    runpy.run_path(str(ROOT / "examples" / "parent_child_rag.py"))
    out = capsys.readouterr().out
    assert "simple: the model reads" in out and "parent_child: the model reads" in out
    assert "offboarding.md#parent_" in out or "it_handbook.md#parent_" in out


def test_the_section_holds_all_six_steps_and_two_pieces_cannot():
    """The page's claim: two 25-word pieces cannot hold the six steps, the section does."""
    document = load_document(OFFBOARDING)
    simple_pieces = WordChunker(25, 0).chunk([document])
    children = ParentChildChunker(WordChunker(25, 0)).chunk([document])

    def covered(texts):
        joined = " ".join(" ".join(text.split()) for text in texts)
        return sum(step in joined for step in STEPS)

    best_two = max(covered([a["text"], b["text"]]) for a in simple_pieces
                   for b in simple_pieces if a is not b)
    assert best_two < len(STEPS)
    section = [c for c in children if "Before your last day" in c["parent_text"]]
    assert covered([section[0]["parent_text"]]) == len(STEPS)
    assert len(section[0]["parent_text"]) == 525           # the size the page shows
