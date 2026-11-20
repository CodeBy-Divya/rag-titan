"""
Hybrid retrieval: keyword (BM25) and vector search together, merged with
Reciprocal Rank Fusion (RAG handbook 2026, sections 3.03 and 5.3).

    RRF(d) = sum over the result lists r:  1 / (k + rank_r(d))

Only ranks are used, never the raw scores, so a BM25 score of 7 and a cosine
of 0.7 never have to be compared. A chunk found by both searches beats one
found by only one; k (60 in the original paper) keeps one top rank from
deciding everything.
"""

from rag_titan.retrieval.base_retriever import BaseRetriever


class HybridRetriever(BaseRetriever):
    """
    Runs a vector and a keyword retriever and fuses their rankings.

    Args:
        dense: a DenseRetriever (meaning).
        keyword: a BM25Retriever (exact words), on the same store.
        k: the RRF constant; larger makes the lists' top ranks matter less.
        pool_size: how many results each search returns before fusing;
            None uses max(20, 4 * top_k).
    """

    name = "hybrid"

    def __init__(self, dense, keyword, k=60, pool_size=None):
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a whole number of at least 1, got {k!r}")
        if pool_size is not None:
            self._check_top_k(pool_size)
        self.dense, self.keyword = dense, keyword
        self.k, self.pool_size = k, pool_size

    def has_documents(self):
        """Return True when there is at least one chunk to search."""
        return self.dense.has_documents()

    def retrieve(self, query, top_k=5):
        """
        Return the top_k chunks with the highest fused score, best first.

        "score" is the RRF score divided by the best possible one (first in
        both lists), so it runs from 0 to 1. "score_trace" shows each part:
        {"dense": 0.71, "dense_rank": 2, "bm25": 6.3, "bm25_rank": 1, "rrf": 0.0325}
        (a search that did not find the chunk has no entry).
        """
        self._check_top_k(top_k)
        if not self.has_documents():
            return []

        pool = self.pool_size or max(20, 4 * top_k)
        lists = {"dense": self.dense.retrieve(query, pool),
                 "bm25": self.keyword.retrieve(query, pool)}
        fused = self.fuse(lists)[:top_k]

        best_possible = len(lists) / (self.k + 1)
        matches = [{**chunk, "score": trace["rrf"] / best_possible} for chunk, trace in fused]
        return self._with_trace(matches, [trace for _, trace in fused])

    def fuse(self, lists):
        """
        Merge ranked lists with RRF.

        Args:
            lists: {"dense": [chunks, best first], "bm25": [...]}.

        Returns:
            A list of (chunk, score_trace) pairs, best fused score first; equal
            scores keep the order in which chunks were first seen.
        """
        chunks, traces, first_seen = {}, {}, {}
        for name, results in lists.items():
            for rank, result in enumerate(results, start=1):
                key = result["id"]
                first_seen.setdefault(key, len(first_seen))
                chunks.setdefault(key, _plain_chunk(result))
                trace = traces.setdefault(key, {"rrf": 0.0})
                trace[name] = result["score_trace"].get(name, result["score"])
                trace[f"{name}_rank"] = rank
                trace["rrf"] += 1 / (self.k + rank)

        order = sorted(chunks, key=lambda key: (-traces[key]["rrf"], first_seen[key]))
        return [(chunks[key], traces[key]) for key in order]


def _plain_chunk(result):
    """The chunk without the scores one retriever added."""
    return {key: value for key, value in result.items()
            if key not in ("score", "retriever", "score_trace")}
