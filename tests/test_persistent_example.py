"""Tests for examples/persistent_rag.py and its copy on the Vector stores page."""

import importlib.util
from pathlib import Path

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory

ROOT = Path(__file__).parent.parent
EXAMPLE = ROOT / "examples" / "persistent_rag.py"
PAGE = ROOT / "docs" / "vector-stores.md"

# examples/ and docs/ are only in the repository, not in the source package
pytestmark = pytest.mark.skipif(not EXAMPLE.is_file(), reason="examples are only in the repo")


def load_example():
    spec = importlib.util.spec_from_file_location("persistent_rag", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_page_shows_the_script_exactly():
    assert f"```python\n{EXAMPLE.read_text(encoding='utf-8')}```" in PAGE.read_text(
        encoding="utf-8")


def test_index_ask_add_info_as_separate_sessions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    monkeypatch.chdir(tmp_path)
    example = load_example()

    example.main(["index"])
    example.main(["ask", "How many paid leave days do I get?"])
    example.main(["add", str(ROOT / "examples" / "sample_docs" / "remote_work.md")])
    example.main(["info"])

    out = capsys.readouterr().out
    assert "Indexed 3 chunks from 3 files into hr_index.db" in out
    assert "Answer: the answer" in out and "from leave_policy.md" in out
    assert "the index now holds 3 chunks" in out and "'next_chunk': 4" in out
    assert (tmp_path / "hr_index.db").is_file()

    with pytest.raises(SystemExit, match="index once, ask later"):
        example.main(["ask"])
