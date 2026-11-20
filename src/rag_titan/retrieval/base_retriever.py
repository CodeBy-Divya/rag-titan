"""
Base class for every retriever (dense, BM25, hybrid, graph, ...).

To add one, inherit from BaseRetriever, give it a short `name`, and implement
retrieve() and has_documents(). Pass the results through _with_trace() so every
retriever returns the same evidence format.
"""


class BaseRetriever:
    """
    Finds the chunks that best match a question.

    Every result is a chunk dict ("id", "source", "text", offsets, ...) plus:
        "score": relevance from 0 to 1, higher is better, so rerankers work the
            same with every retriever (cosine for dense; BM25 and hybrid scale
            their own scores, see score_trace for the raw ones).
        "retriever": the `name` of the retriever that found it, e.g. "dense".
        "score_trace": how the score was made, e.g. {"dense": 0.83}.
            A hybrid retriever would add one entry per part, e.g.
            {"bm25": 7.1, "dense": 0.83, "rrf": 0.032}.

    "id" and "source" are the chunk id and source id of the evidence.
    """

    name = "base"
    label = "Retrieve"          # the search node's name in the flow diagram

    def retrieve(self, query, top_k=5):
        """Return the top_k best chunks, best first."""
        raise NotImplementedError

    def has_documents(self):
        """Return True when there is at least one chunk to search."""
        raise NotImplementedError

    def _check_top_k(self, top_k):
        """Raise a clear error for a top_k that cannot return anything."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be a whole number of at least 1, got {top_k!r}")

    def _with_trace(self, matches, traces=None):
        """
        Return copies of the matches with "retriever" and "score_trace" added.

        Args:
            matches: chunk dicts with a "score".
            traces: optional score_trace dict per match; default {name: score}.
        """
        traced = []

        for number, match in enumerate(matches):
            new_match = dict(match)
            new_match["retriever"] = self.name
            new_match["score_trace"] = traces[number] if traces else {self.name: match["score"]}
            traced.append(new_match)

        return traced
