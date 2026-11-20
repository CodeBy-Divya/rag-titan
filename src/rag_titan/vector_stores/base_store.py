"""
Base class for every vector store (in-memory, SQLite, Chroma, Qdrant, ...).

To add one, inherit from BaseStore and implement add(), search(), clear() and
count(). Persistent stores should also implement delete_source(), get_info()
and set_info(), so files can be replaced and mixed-up indexes are caught.
Run tests/store_contract.py against your store to check it.
"""


class BaseStore:
    """Saves chunks with their vectors and finds the most similar ones."""

    # True for stores that keep the index after Python exits; evaluation then
    # works on a temporary copy so it never replaces your saved index
    persistent = False

    def add(self, chunks, vectors):
        """Save chunks together with their vectors (same order)."""
        raise NotImplementedError

    def search(self, query_vector, top_k=5):
        """
        Return the top_k most similar chunks, best first.
        Each result is the chunk dict plus a "score" key (cosine similarity,
        higher is more similar).
        """
        raise NotImplementedError

    def clear(self):
        """Remove everything from the store, including the saved info."""
        raise NotImplementedError

    def count(self):
        """Return how many chunks are stored."""
        raise NotImplementedError

    def delete_source(self, source):
        """
        Remove every chunk that came from one file, so it can be indexed again.

        Args:
            source: the file name saved in each chunk's "source" key.

        Returns:
            How many chunks were removed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot replace a file's chunks: index with "
            "reset=True, or implement delete_source()"
        )

    def all_chunks(self):
        """
        Return every stored chunk (without its vector), for keyword search.

        Returns:
            A list of chunk dicts, in any order.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot list its chunks, which BM25 and hybrid "
            "search need: implement all_chunks(), or use RETRIEVER=dense"
        )

    def get_info(self):
        """
        Return how the index was built, or None when unknown.

        Example: {"embed_model": "nomic-embed-text", "dimensions": 768,
        "chunker": "WordChunker", "next_chunk": 58, "revision": "3f2a9c01b7de"}
        """
        return None

    def set_info(self, info):
        """
        Save how the index was built (see get_info). Stores that cannot save
        it simply ignore it; the mismatch checks are then skipped.

        Args:
            info: a dict with plain values (text and numbers).
        """
