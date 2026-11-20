"""
Pick the vector store named in the settings.

Adding a store: create its class in this folder, add its name to
VECTOR_STORES and one `if name == ...` line in build_store().
"""

from rag_titan.vector_stores.chroma_store import ChromaStore
from rag_titan.vector_stores.memory_store import MemoryStore
from rag_titan.vector_stores.qdrant_store import QdrantStore
from rag_titan.vector_stores.sqlite_store import SQLiteStore

# every vector store build_store() knows, used in error messages and the docs
VECTOR_STORES = ["memory", "sqlite", "chroma", "qdrant"]


def build_store(settings):
    """
    Create the vector store named in settings["vector_store"].

    "memory" forgets everything when Python exits. The others keep the index:
    "sqlite" in the file settings["vector_store_path"]; "chroma" and "qdrant"
    in that folder, or on the server at settings["vector_store_url"], in the
    collection settings["collection"] (they need their extra installed).

    Args:
        settings: the dict from get_settings().
    """
    name = str(settings["vector_store"]).strip().lower()

    if name == "memory":
        return MemoryStore()

    if name == "sqlite":
        return SQLiteStore(settings["vector_store_path"])

    if name in ("chroma", "qdrant"):
        store_class = ChromaStore if name == "chroma" else QdrantStore
        return store_class(path=settings["vector_store_path"], url=settings["vector_store_url"],
                           collection=settings["collection"])

    raise ValueError(f"Unknown vector store: {settings['vector_store']!r}. "
                     f"Use one of: {', '.join(VECTOR_STORES)}")
