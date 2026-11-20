"""
Tests for the saved-index examples (examples/persistent_rag*.py) and their
copies on the Vector stores page (hash embedder and a fake LLM, no Ollama).
"""

import runpy
from pathlib import Path

import pytest
from conftest import PromptLLM

import rag_titan.embeddings
import rag_titan.factory as factory
import rag_titan.llm
from rag_titan.embeddings import HashEmbedder

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
PAGE = ROOT / "docs" / "vector-stores.md"

# examples/ and docs/ are only in the repository, not in the source package
pytestmark = pytest.mark.skipif(not (EXAMPLES / "persistent_rag.py").is_file(),
                                reason="examples are only in the repo")


@pytest.mark.parametrize("name", ["persistent_rag.py", "persistent_rag_parts.py"])
def test_the_page_shows_the_script_exactly(name):
    script = (EXAMPLES / name).read_text(encoding="utf-8")
    assert f"```python\n{script}```" in PAGE.read_text(encoding="utf-8")


@pytest.fixture
def fake_models(tmp_path, monkeypatch):
    """Run in tmp_path with the hash embedder and a fake LLM instead of Ollama."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setattr(rag_titan.llm, "OllamaLLM", lambda model: PromptLLM())
    monkeypatch.setattr(rag_titan.embeddings, "OllamaEmbedder", lambda model: HashEmbedder())
    return tmp_path


def test_first_run_indexes_second_run_reuses(fake_models, capsys):
    runpy.run_path(str(EXAMPLES / "persistent_rag.py"))
    first = capsys.readouterr().out
    runpy.run_path(str(EXAMPLES / "persistent_rag.py"))
    second = capsys.readouterr().out

    assert first.startswith("Indexing 2 files...")
    assert second.startswith("Using the saved index: 3 chunks, no indexing needed")
    assert "The index holds 3 chunks" in second          # remote_work.md replaced, not doubled
    assert second.count("A: the answer") == 2 and "(chunk_3)" in second
    assert (fake_models / "hr_index.db").is_file()


def test_parts_example_reuses_its_index(fake_models, capsys):
    runpy.run_path(str(EXAMPLES / "persistent_rag_parts.py"))
    runpy.run_path(str(EXAMPLES / "persistent_rag_parts.py"))

    out = capsys.readouterr().out
    assert "Indexed 3 chunks" in out and "Using the saved index: 3 chunks" in out
    assert out.strip().endswith("the answer")
