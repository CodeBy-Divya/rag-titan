"""
Keyword retrieval with BM25: finds chunks that share the question's words.

Strong on exact words (ids, names, codes, rare terms), blind to synonyms. It
needs no embedding model. The keyword index is built from the chunks in the
vector store on the first search and built again whenever the store changes,
so it always matches what is indexed (also for a saved store after a restart).
"""

from rag_titan.retrieval.base_retriever import BaseRetriever
from rag_titan.retrieval.bm25 import BM25Index
from rag_titan.retrieval.dedupe import remove_duplicates
from rag_titan.utils.logger import get_logger

logger = get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """
    Finds chunks by keyword overlap, scored with BM25 (see retrieval/bm25.py).

    Args:
        store: a BaseStore with all_chunks(), the one the files were indexed into.
        k1: word-repeat saturation, usually 1.2 to 2.0.
        b: length normalisation, from 0 (none) to 1 (full).
        dedupe_overlap: drop a result that overlaps a better one from the same
            file by at least this share (like DenseRetriever); None keeps all.
    """

    name = "bm25"
    label = "Keyword search"

    def __init__(self, store, k1=1.5, b=0.75, dedupe_overlap=0.5):
        if dedupe_overlap is not None and not 0 < dedupe_overlap <= 1:
            raise ValueError("dedupe_overlap must be more than 0 and at most 1, or None")
        BM25Index([], k1, b)                # checks k1 and b now, not at the first search
        self.store, self.k1, self.b = store, k1, b
        self.dedupe_overlap = dedupe_overlap
        self._cache = None                  # (store version, chunks, BM25Index)

    def has_documents(self):
        """Return True when the store holds at least one chunk."""
        return self.store.count() > 0

    def retrieve(self, query, top_k=5):
        """
        Return the top_k chunks with the highest BM25 score, best first.

        Chunks without any word of the question are not returned, so there can
        be fewer than top_k results. "score" is the BM25 score divided by the
        best one for this question (the best chunk gets 1.0); the raw BM25
        score is in "score_trace" (see BaseRetriever).
        """
        self._check_top_k(top_k)
        if not self.has_documents():
            logger.warning("Vector store is empty, nothing to retrieve")
            return []

        chunks, index = self._index()
        wanted = top_k if self.dedupe_overlap is None else 2 * top_k
        found = index.search(query, wanted)
        best = found[0][1] if found else 1.0
        matches = [{**chunks[number], "score": score / best, "bm25": score}
                   for number, score in found]
        if self.dedupe_overlap is not None:
            matches = remove_duplicates(matches, self.dedupe_overlap)
        traces = [{"bm25": match.pop("bm25")} for match in matches[:top_k]]
        return self._with_trace(matches[:top_k], traces)

    def _index(self):
        """Return (chunks, BM25Index), building them again when the store changed."""
        version = self._store_version()
        if self._cache is None or self._cache[0] != version:
            chunks = sorted(self.store.all_chunks(), key=_chunk_order)
            index = BM25Index([chunk["text"] for chunk in chunks], self.k1, self.b)
            logger.info("Built the BM25 index over %d chunks", len(chunks))
            self._cache = (version, chunks, index)
        return self._cache[1], self._cache[2]

    def _store_version(self):
        """
        A value that changes whenever the store changes.

        Every indexing run saves a new "revision" in the store's info; stores
        without info fall back to their chunk count.
        """
        info = self.store.get_info() or {}
        return info.get("revision"), self.store.count()


def _chunk_order(chunk):
    """Sort "chunk_2" before "chunk_10" (stores may list chunks in any order)."""
    number = str(chunk.get("id", "")).rpartition("_")[2]
    return (0, int(number), "") if number.isdigit() else (1, 0, str(chunk.get("id")))
