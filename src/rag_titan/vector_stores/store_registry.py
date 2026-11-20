"""
Pick the vector store named in the settings.

Adding a store: create its class in this folder, add its name to
VECTOR_STORES and one `if name == ...` line in build_store().
"""

from rag_titan.vector_stores.memory_store import MemoryStore
from rag_titan.vector_stores.sqlite_store import SQLiteStore

# every vector store build_store() knows, used in error messages and the docs
VECTOR_STORES = ["memory", "sqlite"]


def build_store(settings):
    """
    Create the vector store named in settings["vector_store"].

    "memory" forgets everything when Python exits; "sqlite" saves the index in
    the file settings["vector_store_path"], so it can be used again later.

    Args:
        settings: the dict from get_settings().
    """
    name = str(settings["vector_store"]).strip().lower()

    if name == "memory":
        return MemoryStore()

    if name == "sqlite":
        return SQLiteStore(settings["vector_store_path"])

    raise ValueError(f"Unknown vector store: {settings['vector_store']!r}. "
                     f"Use one of: {', '.join(VECTOR_STORES)}")
