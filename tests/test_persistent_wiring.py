"""
Tests for VECTOR_STORE=sqlite through settings and RAGTitan, and that evaluation
never touches a saved index (hash embedder and a fake LLM, no Ollama).
"""

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan import RAGTitan, get_settings
from rag_titan.evaluation import compare_setups
from rag_titan.factory import build_store
from rag_titan.vector_stores import MemoryStore, SQLiteStore

GOLDEN = [{"id": "q1", "question": "How many leave days?",
           "expected_evidence": ["twenty paid leave days"]}]


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Two policy files, a fake LLM for every RAGTitan, and tmp_path as the folder."""
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    monkeypatch.delenv("VECTOR_STORE_PATH", raising=False)
    (tmp_path / "leave.md").write_text("You get twenty paid leave days.", encoding="utf-8")
    (tmp_path / "remote.md").write_text("Remote work needs a laptop.", encoding="utf-8")
    return ["leave.md", "remote.md"]


def sqlite_rag(path="index.db"):
    return RAGTitan(settings={"vector_store": "sqlite", "vector_store_path": path,
                              "embed_provider": "hash"})


def test_settings_build_the_store(tmp_path, monkeypatch):
    monkeypatch.delenv("VECTOR_STORE_PATH", raising=False)
    assert isinstance(build_store(get_settings()), MemoryStore)
    assert get_settings()["vector_store_path"] == "rag_titan.db"

    monkeypatch.setenv("VECTOR_STORE", " SQLite ")
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "saved" / "my.db"))
    store = build_store(get_settings())
    assert isinstance(store, SQLiteStore) and store.path == tmp_path / "saved" / "my.db"

    with pytest.raises(ValueError, match="Unknown vector store: 'chromadb'. Use one of: "
                                         "memory, sqlite"):
        build_store(get_settings({"vector_store": "chromadb"}))


def test_index_once_then_ask_in_a_new_session(files):
    assert sqlite_rag().index(files) == 2

    later = sqlite_rag()                      # like a new Python session: no index() call
    result = later.run("How many paid leave days?")

    assert result["answer"] == "the answer"
    assert result["sources"][0]["source"] == "leave.md"


def test_an_empty_index_asks_to_index_first(files):
    assert sqlite_rag("empty.db").ask("How many leave days?") == \
        "Please index your documents first."


def test_evaluate_stages_leaves_the_saved_index_alone(files):
    rag = sqlite_rag()
    rag.index(["remote.md"])
    before = (rag.pipeline.store.count(), rag.pipeline.store.get_info())

    report = rag.evaluate_stages(files, GOLDEN, show=False)

    again = sqlite_rag()
    assert report["scorecard"]
    assert (again.pipeline.store.count(), again.pipeline.store.get_info()) == before
    assert {r["source"] for r in again.parts["retriever"].retrieve("leave", top_k=5)} == \
        {"remote.md"}


def test_compare_setups_never_writes_a_saved_index(files, tmp_path):
    setups = [{"name": "saved", "rag_type": "simple",
               "settings": {"vector_store": "sqlite", "vector_store_path": "compare.db",
                            "embed_provider": "hash"}}]

    questions = [{"id": "q1", "question": "How many leave days?",
                  "required_evidence_ids": ["leave.md"]}]
    results = compare_setups(files, questions, setups)

    assert results[0]["name"] == "saved" and results[0]["recall"] == 1.0
    assert not (tmp_path / "compare.db").exists()
