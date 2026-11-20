"""Tests for `rag-titan index` and asking a saved index (hash embedder, fake LLM)."""

import json

import pytest
from conftest import PromptLLM

import rag_titan.factory as factory
from rag_titan import cli
from rag_titan.embeddings import HashEmbedder
from rag_titan.vector_stores import SQLiteStore


@pytest.fixture(autouse=True)
def folder(tmp_path, monkeypatch):
    """Work in tmp_path with two policy files, the hash embedder and a fake LLM."""
    monkeypatch.setattr(factory, "build_llm", lambda settings, **kwargs: PromptLLM())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EMBED_PROVIDER", "hash")
    for name in ("VECTOR_STORE", "VECTOR_STORE_PATH"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "leave.md").write_text("You get twenty paid leave days.", encoding="utf-8")
    (tmp_path / "remote.md").write_text("Remote work needs a laptop.", encoding="utf-8")
    return tmp_path


def test_index_once_then_ask_without_files(capsys):
    cli.main(["index", "--files", "leave.md", "--vector-store", "sqlite"])
    cli.main(["ask", "--question", "How many leave days?", "--vector-store", "sqlite"])

    out = capsys.readouterr().out
    assert "Indexed 1 chunks from 1 file(s) into rag_titan.db; it now holds 1 chunks." in out
    assert "Using 1 chunks saved in rag_titan.db." in out
    assert out.strip().endswith("the answer")


def test_add_keeps_what_was_indexed(folder, capsys, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "sqlite")           # from .env instead of options
    cli.main(["index", "--files", "leave.md", "--store-path", "saved/my.db"])
    cli.main(["index", "--files", "remote.md", "leave.md", "--store-path", "saved/my.db",
              "--add"])

    store = SQLiteStore(folder / "saved" / "my.db")
    assert "it now holds 2 chunks" in capsys.readouterr().out
    assert store.count() == 2 and store.get_info()["next_chunk"] == 3


def test_chat_uses_the_saved_index(capsys, monkeypatch):
    cli.main(["index", "--files", "leave.md", "--vector-store", "sqlite"])
    answers = iter(["How many leave days?", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    cli.main(["chat", "--vector-store", "sqlite", "--rag-type", "adaptive"])

    assert "RAG: the answer" in capsys.readouterr().out


@pytest.mark.parametrize("argv, message", [
    (["index", "--files", "leave.md"], "VECTOR_STORE=memory keeps nothing"),
    (["ask", "--question", "leave?"], "rag-titan ask: nothing is saved with VECTOR_STORE=memory"),
    (["ask", "--question", "leave?", "--vector-store", "sqlite"],
     "nothing is indexed in rag_titan.db yet. Run `rag-titan index --files ...` first."),
    (["chat", "--vector-store", "sqlite", "--store-path", "none.db"],
     "rag-titan chat: nothing is indexed in none.db"),
])
def test_clear_messages(argv, message):
    with pytest.raises(SystemExit, match=message.replace("?", r"\?").replace(".", r"\.")):
        cli.main(argv)


def test_ask_with_files_works_as_before(capsys):
    cli.main(["ask", "--files", "leave.md", "remote.md", "--question", "leave?"])
    assert "Indexed 2 chunks from 2 file(s)." in capsys.readouterr().out


def test_retrieval_evaluation_leaves_the_saved_index_alone(folder, monkeypatch):
    cli.main(["index", "--files", "remote.md", "--vector-store", "sqlite"])
    questions = folder / "q.json"
    questions.write_text(json.dumps([{"id": "q1", "question": "leave days",
                                      "required_evidence_ids": ["leave.md"]}]))
    monkeypatch.setenv("VECTOR_STORE", "sqlite")

    cli.main(["evaluate", "--files", "leave.md", "remote.md", "--questions", str(questions),
              "--retrieval-only"])

    store = SQLiteStore(folder / "rag_titan.db")
    assert store.count() == 1
    assert store.search(HashEmbedder().embed("remote work"), top_k=1)[0]["source"] == "remote.md"
