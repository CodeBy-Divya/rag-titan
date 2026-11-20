"""
Vector store backed by Chroma (https://www.trychroma.com).

Install it first: pip install "rag-titan[chroma]"

A local folder needs no server; a URL talks to a Chroma server
(`chroma run --path ./chroma_data`).

Example:
    from rag_titan.vector_stores import ChromaStore

    store = ChromaStore(path="rag_titan_chroma")              # local folder
    store = ChromaStore(url="http://localhost:8000")          # Chroma server
"""

import json
import re
from urllib.parse import urlsplit

from rag_titan.vector_stores.base_store import BaseStore

INSTALL_HINT = 'ChromaStore needs the chromadb package: pip install "rag-titan[chroma]"'

# Chroma's rule for collection names, checked here to give a clearer message
NAME_RULE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,505}[a-zA-Z0-9]")


class ChromaStore(BaseStore):
    """
    Keeps chunks in a Chroma collection that survives restarts.

    Args:
        path: local folder for the data (used when url is None).
        url: address of a Chroma server, for example "http://localhost:8000".
        collection: collection name; use another name for another index.
    """

    persistent = True

    def __init__(self, path="rag_titan_chroma", url=None, collection="rag_titan"):
        try:
            import chromadb
        except ImportError:
            raise ImportError(INSTALL_HINT) from None

        if not NAME_RULE.fullmatch(str(collection)):
            raise ValueError(f"Chroma collection names need 3 to 506 letters, digits, '.', "
                             f"'_' or '-' (starting and ending with a letter or digit), "
                             f"got {collection!r}")
        self.name, self.where = str(collection), url or str(path)
        self.client = _connect(chromadb, path, url)
        self.collection = self._open(self.name)
        self.info_collection = self._open(self.name + "_info")

    def __repr__(self):
        return f"ChromaStore({self.where!r}, collection={self.name!r})"

    def add(self, chunks, vectors):
        """Save each chunk with its vector (a chunk with the same id is replaced)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        # Chroma metadata only takes flat values, so the whole chunk is saved as JSON
        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            embeddings=[[float(x) for x in vector] for vector in vectors],
            metadatas=[{"source": chunk["source"], "chunk": json.dumps(chunk)}
                       for chunk in chunks],
        )

    def search(self, query_vector, top_k=5):
        """Return the top_k most similar chunks, best first, with cosine scores."""
        total = self.count()
        if total == 0:
            return []
        found = self.collection.query(query_embeddings=[[float(x) for x in query_vector]],
                                      n_results=min(top_k, total),
                                      include=["metadatas", "distances"])
        # Chroma gives cosine *distance* (0 = same direction); score = 1 - distance
        return [{**json.loads(meta["chunk"]), "score": 1.0 - float(distance)}
                for meta, distance in zip(found["metadatas"][0], found["distances"][0])]

    def clear(self):
        """Remove every stored chunk and the saved info."""
        for name in (self.name, self.name + "_info"):
            self.client.delete_collection(name)
        self.collection = self._open(self.name)
        self.info_collection = self._open(self.name + "_info")

    def count(self):
        """Return how many chunks are stored."""
        return self.collection.count()

    def delete_source(self, source):
        """Remove every chunk of one file and return how many were removed."""
        ids = self.collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def get_info(self):
        """Return how the index was built, or None for an empty store."""
        found = self.info_collection.get(ids=["index"], include=["metadatas"])
        return json.loads(found["metadatas"][0]["info"]) if found["ids"] else None

    def set_info(self, info):
        """Remember how the index was built."""
        self.info_collection.upsert(ids=["index"], embeddings=[[1.0]],
                                    metadatas=[{"info": json.dumps(info)}])

    def _open(self, name):
        """Get or create a collection that compares vectors by cosine."""
        return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"},
                                                    embedding_function=None)


def _connect(chromadb, path, url):
    """Return a client for a server (url) or a local folder (path)."""
    if url is None:
        return chromadb.PersistentClient(path=str(path))

    address = urlsplit(url)
    if address.scheme not in ("http", "https") or not address.hostname:
        raise ValueError(f"VECTOR_STORE_URL must look like http://localhost:8000, got {url!r}")
    secure = address.scheme == "https"
    return chromadb.HttpClient(host=address.hostname, ssl=secure,
                               port=address.port or (443 if secure else 8000))
