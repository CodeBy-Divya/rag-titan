"""
The shared store checks for ChromaStore and QdrantStore, in local folder mode.
Skipped when chromadb / qdrant-client are not installed.
"""

import builtins
from pathlib import Path

import pytest
from store_contract import StoreContract

from rag_titan import get_settings
from rag_titan.factory import build_store
from rag_titan.vector_stores import ChromaStore, QdrantStore


class TestChromaStore(StoreContract):
    persistent = True

    @pytest.fixture(autouse=True)
    def needs_chroma(self):
        pytest.importorskip("chromadb")

    def make_store(self, tmp_path):
        return ChromaStore(path=tmp_path / "chroma", collection="test")

    def reopen(self, store, tmp_path):
        return ChromaStore(path=tmp_path / "chroma", collection="test")


class TestQdrantStore(StoreContract):
    persistent = True

    @pytest.fixture(autouse=True)
    def needs_qdrant(self):
        pytest.importorskip("qdrant_client")

    def make_store(self, tmp_path):
        return QdrantStore(path=tmp_path / "qdrant", collection="test")

    def reopen(self, store, tmp_path):
        store.close()                          # a local folder is used by one client at a time
        return QdrantStore(path=tmp_path / "qdrant", collection="test")


def test_settings_build_both_stores(tmp_path):
    pytest.importorskip("chromadb")
    pytest.importorskip("qdrant_client")
    chroma = build_store(get_settings({"vector_store": "chroma", "collection": "hr_docs",
                                       "vector_store_path": str(tmp_path / "c")}))
    qdrant = build_store(get_settings({"vector_store": "qdrant",
                                       "vector_store_path": str(tmp_path / "q")}))

    assert isinstance(chroma, ChromaStore) and repr(chroma).endswith("collection='hr_docs')")
    assert isinstance(qdrant, QdrantStore) and qdrant.name == "rag_titan"
    assert chroma.persistent and qdrant.persistent


def test_collections_keep_indexes_apart(tmp_path):
    pytest.importorskip("qdrant_client")
    from store_contract import VECTORS, chunk
    first = QdrantStore(path=tmp_path / "q", collection="hr")
    first.add([chunk(0)], VECTORS[:1])
    first.close()

    other = QdrantStore(path=tmp_path / "q", collection="legal")
    assert other.count() == 0 and other.get_info() is None


@pytest.mark.parametrize("store_class, module, extra", [
    (ChromaStore, "chromadb", "chroma"), (QdrantStore, "qdrant_client", "qdrant")])
def test_missing_package_says_what_to_install(store_class, module, extra, monkeypatch, tmp_path):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] == module:
            raise ImportError(f"No module named {module!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=rf'pip install "rag-titan\[{extra}\]"'):
        store_class(path=tmp_path / "x")


def test_bad_names_and_urls(tmp_path):
    pytest.importorskip("chromadb")
    for bad in [" ", "hr", "my docs", "-hr-"]:
        with pytest.raises(ValueError, match="Chroma collection names need 3"):
            ChromaStore(path=tmp_path / "c", collection=bad)
    with pytest.raises(ValueError, match="http://localhost:8000"):
        ChromaStore(url="localhost:8000")


@pytest.mark.parametrize("name", ["chroma", "qdrant"])
def test_index_then_search_from_a_new_store(name, tmp_path):
    pytest.importorskip("chromadb" if name == "chroma" else "qdrant_client")
    from rag_titan.chunking import WordChunker
    from rag_titan.embeddings import HashEmbedder
    from rag_titan.pipelines import IndexingPipeline
    from rag_titan.retrieval import DenseRetriever

    policy = tmp_path / "leave.md"
    policy.write_text("You get twenty paid leave days. Remote work needs a laptop.",
                      encoding="utf-8")
    settings = get_settings({"vector_store": name, "vector_store_path": str(tmp_path / name)})
    IndexingPipeline(WordChunker(5, 0), HashEmbedder(64), build_store(settings)).run([policy])

    again = build_store(settings)        # same process, same folder: works for both
    results = DenseRetriever(HashEmbedder(64), again).retrieve("paid leave days", top_k=1)

    assert again.count() == 3 and again.get_info()["next_chunk"] == 3   # 11 words, 5 per chunk
    assert "leave" in results[0]["text"]
    with pytest.raises(ValueError, match="built with hash"):
        DenseRetriever(HashEmbedder(32), again).retrieve("leave")


def test_a_program_using_local_qdrant_exits_quietly(tmp_path):
    pytest.importorskip("qdrant_client")
    import subprocess
    import sys

    script = ("from rag_titan.vector_stores import QdrantStore\n"
              "from store_contract import VECTORS, chunk\n"
              f"QdrantStore(path={str(tmp_path / 'q')!r}).add([chunk(0)], VECTORS[:1])\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=str(Path(__file__).parent), timeout=60)

    assert done.returncode == 0 and done.stderr == ""
