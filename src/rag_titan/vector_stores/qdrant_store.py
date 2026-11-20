"""
Vector store backed by Qdrant (https://qdrant.tech).

Install it first: pip install "rag-titan[qdrant]"

A local folder needs no server; a URL talks to a Qdrant server or Qdrant
Cloud (the key is read from the QDRANT_API_KEY environment variable).

Example:
    from rag_titan.vector_stores import QdrantStore

    store = QdrantStore(path="rag_titan_qdrant")                # local folder
    store = QdrantStore(url="http://localhost:6333")            # Qdrant server
"""

import os
import uuid
from pathlib import Path

from rag_titan.vector_stores.base_store import BaseStore

INSTALL_HINT = 'QdrantStore needs the qdrant-client package: pip install "rag-titan[qdrant]"'

# Qdrant point ids must be numbers or UUIDs: the chunk id is turned into a fixed UUID
ID_SPACE = uuid.UUID("5f1d7a52-3c4b-4f43-9a8e-2f6f7c0d9b11")

# a local folder can be opened by one Qdrant client at a time, so every
# QdrantStore on the same folder in this Python process shares one client
# (re-running `RAGTitan(...)` in a notebook would fail otherwise)
LOCAL_CLIENTS = {}


class QdrantStore(BaseStore):
    """
    Keeps chunks in a Qdrant collection that survives restarts.

    Args:
        path: local folder for the data (used when url is None).
        url: address of a Qdrant server or cloud cluster.
        collection: collection name; use another name for another index.
        api_key: Qdrant Cloud key. None reads QDRANT_API_KEY.
    """

    persistent = True

    def __init__(self, path="rag_titan_qdrant", url=None, collection="rag_titan", api_key=None):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError:
            raise ImportError(INSTALL_HINT) from None

        if not collection or not str(collection).strip():
            raise ValueError("collection must be a non-empty name")
        self.models, self.name, self.where = models, str(collection), url or str(path)
        if url is None:
            self.folder = str(Path(path).resolve())
            if self.folder not in LOCAL_CLIENTS:
                LOCAL_CLIENTS[self.folder] = QdrantClient(path=self.folder)
            self.client = LOCAL_CLIENTS[self.folder]
        else:
            self.folder = None
            self.client = QdrantClient(url=url, api_key=api_key or os.getenv("QDRANT_API_KEY"))

    def __repr__(self):
        return f"QdrantStore({self.where!r}, collection={self.name!r})"

    def add(self, chunks, vectors):
        """Save each chunk with its vector (a chunk with the same id is replaced)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        # the collection is made on the first add, when the vector length is known
        self._ensure(self.name, len(vectors[0]))
        points = [self.models.PointStruct(id=_point_id(chunk["id"]),
                                          vector=[float(x) for x in vector],
                                          payload={"source": chunk["source"], "chunk": chunk})
                  for chunk, vector in zip(chunks, vectors)]
        self.client.upsert(self.name, points=points, wait=True)

    def search(self, query_vector, top_k=5):
        """Return the top_k most similar chunks, best first, with cosine scores."""
        if self.count() == 0:
            return []
        found = self.client.query_points(self.name, query=[float(x) for x in query_vector],
                                         limit=top_k, with_payload=True).points
        return [{**point.payload["chunk"], "score": float(point.score)} for point in found]

    def clear(self):
        """Remove every stored chunk and the saved info."""
        for name in (self.name, self.name + "_info"):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)

    def count(self):
        """Return how many chunks are stored."""
        if not self.client.collection_exists(self.name):
            return 0
        return self.client.count(self.name, exact=True).count

    def delete_source(self, source):
        """Remove every chunk of one file and return how many were removed."""
        if not self.client.collection_exists(self.name):
            return 0
        models = self.models
        match = models.Filter(must=[models.FieldCondition(
            key="source", match=models.MatchValue(value=source))])
        removed = self.client.count(self.name, count_filter=match, exact=True).count
        if removed:
            self.client.delete(self.name, points_selector=models.FilterSelector(filter=match),
                               wait=True)
        return removed

    def get_info(self):
        """Return how the index was built, or None for an empty store."""
        name = self.name + "_info"
        if not self.client.collection_exists(name):
            return None
        found = self.client.retrieve(name, ids=[_point_id("index")], with_payload=True)
        return dict(found[0].payload) if found else None

    def set_info(self, info):
        """Remember how the index was built (in a small companion collection)."""
        name = self.name + "_info"
        self._ensure(name, 1)
        point = self.models.PointStruct(id=_point_id("index"), vector=[1.0], payload=dict(info))
        self.client.upsert(name, points=[point], wait=True)

    def close(self):
        """Release the local folder, so another program can open it."""
        if self.folder:
            LOCAL_CLIENTS.pop(self.folder, None)
        self.client.close()

    def _ensure(self, name, dimensions):
        """Create a cosine collection with vectors of this length if it is missing."""
        if not self.client.collection_exists(name):
            self.client.create_collection(name, vectors_config=self.models.VectorParams(
                size=dimensions, distance=self.models.Distance.COSINE))


def _point_id(chunk_id):
    """The same chunk id always gives the same UUID."""
    return str(uuid.uuid5(ID_SPACE, str(chunk_id)))
